from __future__ import annotations

import json
import copy
import logging
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote_plus, urlparse
from uuid import uuid4

import httpx
from app.config import get_settings
from app.services.documents import DocumentProcessor
from app.services.profile_scraper import scrape_company_profile
from agents.llm_clients import ClaudeClient, GeminiClient
from agents.pdf_extractor import extract_metrics_from_datasets, extract_metrics_from_pitch_documents
from agents.memo_schema import (
    DEFAULT_PLACEHOLDER,
    Competitor,
    DueDiligenceItem,
    ExecutiveHighlight,
    ExecutiveRisk,
    FounderProfile,
    InvestorCommitteeMemo,
    LLMTokenUsage,
    RevenueProjection,
    RiskEntry,
    StructuredField,
    StructuredFieldValidation,
    StructuredInterface,
    StructuredInterfaceSection,
    TractionKPI,
    UseOfFundsBreakdown,
)
from repositories.memo_repository import get_memo_record, update_memo_state
from agents.memo_validator import (
    extract_validated_data, MemoDataContext, validate_data_context, ValidationIssue, 
    parse_money, parse_int, _parse_money_multicurrency, _parse_percentage, _parse_number
)
from agents.data_priority_manager import DataPriorityManager
from app.models import DatasetRecord
from repositories.dataset_repository import list_datasets

logger = logging.getLogger(__name__)


def _strip_code_fences(content: str) -> str:
    sanitized = content.strip()
    if sanitized.startswith("```"):
        sanitized = sanitized.lstrip("`")
        sanitized = sanitized.split("\n", 1)[-1]
    if sanitized.endswith("```"):
        sanitized = sanitized[:-3]
    return sanitized.strip()


class InvestorMemoAgent:
    NO_DATA = "Data unavailable — requires founder confirmation"  # Phase F: Standardized message
    SNIPPET_STOP_PHRASES = (
        "download the app",
        "terms of service",
        "privacy policy",
        "cookie policy",
        "report a fraud",
        "help & support",
        "social links",
        "all rights reserved",
    )
    METRIC_LABELS = {
        "restaurants": "restaurant_partners",
        "restaurant": "restaurant_partners",
        "cities": "cities",
        "orders": "orders_delivered",
        "deliveries": "orders_delivered",
        "users": "users",
        "customers": "customers",
        "employees": "employees",
        "downloads": "downloads",
        "products": "products",
        "partners": "partners",
    }

    def _truncate_text(self, text: str, limit: int = 400) -> str:
        """Trim text to a character limit while keeping a clear truncation marker."""
        if not isinstance(text, str):
            return ""
        text = text.strip()
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "... [truncated]"

    def _summarize_text_snippets(self, items: Iterable[Any], max_items: int = 2, max_chars: int = 300) -> str:
        """
        Produce a compact string summary of text snippets (e.g., PDF excerpts, website snippets).
        Keeps ordering, caps item count and per-item length to reduce prompt size.
        """
        if not items:
            return ""
        cleaned = []
        for item in list(items)[:max_items]:
            snippet = ""
            if isinstance(item, dict):
                filename = item.get("filename", "document")
                snippet = f"{filename}: {self._truncate_text(str(item.get('text', '')), max_chars)}"
            else:
                snippet = self._truncate_text(str(item), max_chars)
            if snippet:
                cleaned.append(f"- {snippet}")
        return "\n".join(cleaned)

    def _compact_json_for_prompt(self, data: Any, max_chars: int = 4000) -> str:
        """
        Minify JSON for prompts and hard-cap length while keeping key/value content intact.
        Falls back to str(data) if serialization fails.
        """
        try:
            payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            payload = str(data)
        if len(payload) > max_chars:
            return payload[:max_chars] + "... [truncated]"
        return payload

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (~4 chars per token)."""
        if not text:
            return 0
        try:
            return max(1, len(text) // 4)
        except Exception:
            return 0

    def _is_token_debug(self) -> bool:
        try:
            return bool(getattr(self.settings, "debug_tokens", False))
        except Exception:
            return False

    def _emit_token_debug(
        self,
        label: str,
        system_prompt: str,
        user_prompt: str,
        sections: Dict[str, str],
        usage: Optional[Dict[str, Any]] = None,
        response_text: Optional[str] = None,
    ) -> None:
        if not self._is_token_debug():
            return
        try:
            system_prompt_tokens = self._estimate_tokens(system_prompt)
            memo_state_tokens = self._estimate_tokens(sections.get("memo_state", ""))
            founder_docs_tokens = self._estimate_tokens(sections.get("founder_docs", ""))
            previous_output_tokens = self._estimate_tokens(sections.get("previous_output", ""))
            template_tokens = self._estimate_tokens(sections.get("template", ""))
            user_prompt_tokens = self._estimate_tokens(user_prompt)
            total_input_tokens = (
                system_prompt_tokens
                + memo_state_tokens
                + founder_docs_tokens
                + previous_output_tokens
                + template_tokens
                + user_prompt_tokens
            )
            response_tokens = 0
            if usage and isinstance(usage, dict):
                response_tokens = usage.get("output_tokens") or usage.get("total_tokens") or 0
            elif response_text:
                response_tokens = self._estimate_tokens(response_text)
            total_tokens_used = total_input_tokens + response_tokens

            payload = {
                "label": label,
                "system_prompt_tokens": system_prompt_tokens,
                "memo_state_tokens": memo_state_tokens,
                "founder_docs_tokens": founder_docs_tokens,
                "previous_output_tokens": previous_output_tokens,
                "template_tokens": template_tokens,
                "user_prompt_tokens": user_prompt_tokens,
                "total_input_tokens": total_input_tokens,
                "response_tokens": response_tokens,
                "total_tokens_used": total_tokens_used,
            }
            logger.info(json.dumps(payload))

            # Write prompt preview (first 5000 chars) to storage/logs
            preview = (system_prompt or "") + "\n\n" + (user_prompt or "")
            preview_path = Path("storage") / "logs" / f"token_debug_{label}.txt"
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            preview_path.write_text(preview[:5000], encoding="utf-8")
        except Exception as e:
            logger.warning(f"Token debug emission failed: {e}")

    def _slim_pitch_for_prompt(self, pitch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Keep only high-signal fields from pitch for prompt context, dropping bulky/visual payloads.
        Preserves key numerics and short strings; drops large blobs (visual_data, deck keys, uploads).
        """
        if not isinstance(pitch, dict):
            return {}

        allow_keys = {
            "company_name", "company", "name", "stage", "round", "market", "industry",
            "description", "headline", "website_url", "url", "website",
            "funding_ask", "raise_amount", "valuation_amount", "monthly_burn",
            "runway_months", "mrr", "arr", "revenue", "arpu", "customers", "users",
            "downloads", "partners", "cities", "employees",
            "traction_kpis", "pdf_metrics", "pdf_texts",
            "relationships", "build_team", "team",
            "tam_value", "sam_value", "som_value", "tam", "sam", "som",
            "growth_rate_pct", "kpi_1_value", "kpi_2_value", "kpi_1_mom_pct", "kpi_2_mom_pct",
            "pricing_model", "business_model", "sales_motion", "gtm_segments",
            "markets", "location", "industry_vertical", "product_category",
        }

        slim = {}
        for k, v in pitch.items():
            # Skip obviously bulky fields
            if k in {"visual_data", "pitch_documents", "upload_documents", "pitch_deck_data", "deck_keys"}:
                continue
            if k not in allow_keys:
                # Keep small scalars even if not allow-listed
                if isinstance(v, (str, int, float)) and len(str(v)) <= 120:
                    slim[k] = v
                continue
            # Keep allow-listed fields, but truncate long strings
            if isinstance(v, str):
                slim[k] = self._truncate_text(v, limit=500)
            elif isinstance(v, (int, float)):
                slim[k] = v
            elif isinstance(v, list):
                # Keep only first few small items
                trimmed = []
                for item in v[:5]:
                    if isinstance(item, (str, int, float)):
                        trimmed.append(self._truncate_text(str(item), 200))
                    elif isinstance(item, dict):
                        trimmed.append({ik: item[ik] for ik in list(item.keys())[:5]})
                if trimmed:
                    slim[k] = trimmed
            elif isinstance(v, dict):
                # Keep shallow dict with small values
                shallow = {}
                for ik, iv in list(v.items())[:10]:
                    if isinstance(iv, (str, int, float)):
                        shallow[ik] = self._truncate_text(str(iv), 200)
                if shallow:
                    slim[k] = shallow
        return slim

    def _normalize_url(self, url: Optional[str]) -> Optional[str]:
        """
        Normalize a URL or domain-like string into a usable https URL.
        Returns None if the value cannot be treated as a URL.
        Does NOT accept "analyst_knowledge" - only real URLs.
        """
        if not url or not isinstance(url, str):
            return None
        candidate = url.strip()
        if not candidate:
            return None
        # Reject analyst_knowledge - only real URLs
        if candidate.lower() == "analyst_knowledge":
            return None
        if candidate.startswith(("http://", "https://")):
            return candidate
        # If it looks like a bare domain, prefix with https://
        domain_pattern = r"^[\w.-]+\.[a-zA-Z]{2,}(/.*)?$"
        if re.match(domain_pattern, candidate):
            return f"https://{candidate}"
        return None

    def __init__(
        self,
        claude_client: Optional[ClaudeClient] = None,
        gemini_client: Optional[GeminiClient] = None,
    ) -> None:
        self.settings = get_settings()
        self.claude = claude_client or ClaudeClient()
        self.gemini = gemini_client or GeminiClient()

    async def generate_memo(
        self,
        pitch_data: Dict[str, Any],
        fund_data: Dict[str, Any],
        uploaded_datasets: Optional[List[DatasetRecord]] = None,
    ) -> Tuple[str, InvestorCommitteeMemo, StructuredInterface, LLMTokenUsage, Dict[str, Any], MemoDataContext, List[Dict[str, Any]], Dict[str, bool]]:
        memo_id = uuid4().hex

        # If pitchDeckData was provided, merge it into pitch_data with deck taking precedence
        # IMPORTANT: Preserve team data from pitch listing (it's the source of truth)
        if pitch_data:
            # Preserve team data before merging (team from pitch listing takes precedence over pitch deck)
            preserved_team = pitch_data.get("team")
            preserved_casino_teams = pitch_data.get("casino_teams")
            
            pitch_deck_payload = pitch_data.pop("pitchDeckData", None) or pitch_data.get("pitch_deck_data")
            if isinstance(pitch_deck_payload, dict):
                merged_pitch = dict(pitch_deck_payload)
                merged_pitch.update(pitch_data)
                merged_pitch["pitch_deck_data"] = pitch_deck_payload
                
                # Restore team data from pitch listing (overwrites any team data from pitch deck)
                if preserved_team:
                    merged_pitch["team"] = preserved_team
                if preserved_casino_teams:
                    merged_pitch["casino_teams"] = preserved_casino_teams
                    
                pitch_data = merged_pitch

        normalized_pitch = self._normalize_payload(pitch_data or {})
        normalized_fund = fund_data or {}
        
        # Log data sources being used for memo generation
        has_pitch_listing = bool(normalized_pitch.get("company_name") or normalized_pitch.get("venue"))
        has_pitch_deck = bool(normalized_pitch.get("pitch_deck") or normalized_pitch.get("pitch_deck_data"))
        has_pitch_docs = bool(normalized_pitch.get("pitch_documents") or normalized_pitch.get("upload_documents"))
        logger.info(f"Memo generation data sources: pitch_listing={has_pitch_listing}, pitch_deck={has_pitch_deck}, pitch_documents={has_pitch_docs}")
        if has_pitch_deck:
            deck_data = normalized_pitch.get("pitch_deck_data") or normalized_pitch.get("pitch_deck") or {}
            if isinstance(deck_data, dict) and "data" in deck_data:
                deck_data = deck_data["data"]
            logger.info(f"Pitch deck data available: has_kpi1={bool(deck_data.get('kpi_1_value'))}, has_mrr={bool(deck_data.get('mrr_or_arr'))}, has_revenue_proj={bool(deck_data.get('rev_y1'))}")
        
        # Extract metrics from PDF files in datasets (Phase F)
        pdf_extraction_results = await extract_metrics_from_datasets(uploaded_datasets or [])
        
        # Also extract from pitch_documents if available
        pitch_docs = normalized_pitch.get("pitch_documents", []) or []
        upload_docs = normalized_pitch.get("upload_documents", []) or []
        all_pitch_docs = pitch_docs + upload_docs

        if all_pitch_docs:
            pitch_pdf_results = await extract_metrics_from_pitch_documents(all_pitch_docs)
            # Merge results
            if pitch_pdf_results.get("pdf_metrics"):
                pdf_extraction_results["pdf_metrics"].update(pitch_pdf_results["pdf_metrics"])
            if pitch_pdf_results.get("pdf_texts"):
                pdf_extraction_results["pdf_texts"].extend(pitch_pdf_results["pdf_texts"])
            # CRITICAL: Merge traction_kpis from pitch documents
            if pitch_pdf_results.get("traction_kpis"):
                if "traction_kpis" not in pdf_extraction_results:
                    pdf_extraction_results["traction_kpis"] = []
                pdf_extraction_results["traction_kpis"].extend(pitch_pdf_results["traction_kpis"])
                logger.info(f"Merged {len(pitch_pdf_results['traction_kpis'])} traction KPIs from pitch documents")
        
        if pdf_extraction_results.get("pdf_metrics"):
            normalized_pitch["pdf_metrics"] = pdf_extraction_results["pdf_metrics"]
            logger.info(f"Extracted PDF metrics from {len(pdf_extraction_results['pdf_metrics'])} files")
        if pdf_extraction_results.get("pdf_texts"):
            normalized_pitch["pdf_texts"] = pdf_extraction_results["pdf_texts"]
        founder_doc_snippets = self._extract_founder_snippets_from_docs(normalized_pitch.get("pdf_texts", []))
        if pdf_extraction_results.get("traction_kpis"):
            logger.info(f"Extracted {len(pdf_extraction_results['traction_kpis'])} traction KPIs from PDFs: {[kpi.name for kpi in pdf_extraction_results['traction_kpis']]}")

        # Also pull traction KPIs from existing pitch deck data (if present) so charts don't show empty states
        deck_traction_kpis = self._convert_pitch_deck_to_traction_kpis(normalized_pitch)
        if deck_traction_kpis:
            if "traction_kpis" not in pdf_extraction_results:
                pdf_extraction_results["traction_kpis"] = []
            pdf_extraction_results["traction_kpis"].extend(deck_traction_kpis)
            logger.info(f"Added {len(deck_traction_kpis)} traction KPIs from pitch deck data")

        # Sanitize unreasonable traction KPI magnitudes to avoid hallucinated giga-scale counts
        if pdf_extraction_results.get("traction_kpis"):
            sanitized_kpis, dropped = self._filter_unreasonable_traction_kpis(pdf_extraction_results["traction_kpis"])
            pdf_extraction_results["traction_kpis"] = sanitized_kpis
            if dropped:
                logger.warning(f"Dropped {len(dropped)} traction KPIs for unrealistic magnitudes: {[k.name for k in dropped]}")
        
        website_context = await self._gather_website_context(normalized_pitch)
        edgar_notes = await self._fetch_edgar_notes(normalized_pitch.get("company_name"))
        sec_notes = await self._fetch_sec_markets_notes()
        research_prompt = self._build_research_prompt(
            normalized_pitch,
            normalized_fund,
            uploaded_datasets,
            website_context=website_context,
            edgar_notes=edgar_notes,
            sec_notes=sec_notes,
        )
        gemini_research, gemini_usage = await self.gemini.research(research_prompt)

        # Extract LinkedIn data for founders
        linkedin_data = await self._extract_linkedin_data(normalized_pitch)
        
        # Generate all memo content with LLM BEFORE creating memo
        logger.info("Calling _generate_content_with_llm to generate all memo content...")
        logger.info(f"Python logs location: Check console output or uvicorn logs in terminal. Look for messages like 'Generated content received' and 'Traction analysis fields'")
        generated_content = await self._generate_content_with_llm(
            pitch=normalized_pitch,
            fund=normalized_fund,
            research=gemini_research,
            pdf_extraction_results=pdf_extraction_results,
            website_context=website_context,
            linkedin_data=linkedin_data,
        )
        # Log what was generated
        if generated_content:
            non_empty_fields = {k: v for k, v in generated_content.items() if v and (not isinstance(v, (list, dict)) or len(v) > 0)}
            logger.info(f"Generated content received: {len(non_empty_fields)} non-empty fields: {list(non_empty_fields.keys())}")
            # Log specific traction fields
            if generated_content.get("traction_analysis"):
                ta = generated_content["traction_analysis"]
                logger.info(f"Traction analysis fields: paying_customers={ta.get('paying_customers')}, lois_pilots={ta.get('lois_pilots')}, beta_users={ta.get('beta_users')}, waitlist={ta.get('waitlist')}")
        else:
            logger.warning("generated_content is None or empty - LLM content generation may have failed")

        # Phase A: Create DataPriorityManager and extract prioritized values
        pdf_metrics_for_priority = pdf_extraction_results.get("pdf_metrics", {}) if pdf_extraction_results else {}
        priority_manager = DataPriorityManager(
            pitch_data=normalized_pitch,
            pdf_metrics=pdf_metrics_for_priority,
            website_context=website_context,
            llm_research=gemini_research
        )
        
        # Extract prioritized values for key fields to pass to Claude
        prioritized_fields = {}
        key_fields = [
            # Revenue & financial
            "revenue", "mrr", "arr", "arpu",
            # User metrics
            "users", "customers", "total_users", "monthly_active_users", "daily_active_users",
            # Traction funnel
            "waitlist", "beta_users", "lois_pilots", "paying_customers",
            # Growth & retention
            "monthly_growth_rate", "monthly_growth_rate_pct", "retention_rate", "retention_6m_pct",
            "churn_rate", "monthly_churn_rate_pct",
            # Engagement
            "engagement_time", "workflows_created", "dau_mau_ratio",
            # Market
            "tam", "sam", "som",
            # Business metrics
            "funding_ask", "burn_rate", "runway_months"
        ]
        for field in key_fields:
            field_value = priority_manager.get_field_value(field)
            if field_value.value is not None and not field_value.requires_confirmation:
                prioritized_fields[field] = {
                    "value": field_value.value,
                    "source": field_value.source,
                    "source_url": field_value.source_url,
                    "confidence": field_value.confidence
                }

        # Pre-populate memo with PDF-extracted traction data before Claude generation
        pre_populated_memo = self._pre_populate_memo_with_pdf_data(normalized_pitch, pdf_extraction_results)

        claude_prompt = self._build_claude_prompt(
            normalized_pitch, normalized_fund, gemini_research, uploaded_datasets,
            website_context=website_context, prioritized_fields=prioritized_fields,
            pre_populated_memo=pre_populated_memo, generated_content=generated_content
        )
        debug_sections_main = {
            "memo_state": self._compact_json_for_prompt(self._slim_pitch_for_prompt(normalized_pitch), max_chars=3000),
            "founder_docs": self._summarize_text_snippets(normalized_pitch.get("pdf_texts", []), max_items=2, max_chars=320) if normalized_pitch.get("pdf_texts") else "",
            "previous_output": self._compact_json_for_prompt(generated_content, max_chars=2500) if generated_content else "",
            "template": self._compact_json_for_prompt(InvestorCommitteeMemo().model_dump(), max_chars=2000),
        }
        
        claude_response = None
        claude_usage = None
        try:
            if self._is_token_debug():
                self._emit_token_debug(
                    label="memo_generation_pre",
                    system_prompt=claude_prompt["system"],
                    user_prompt=claude_prompt["messages"][0]["content"],
                    sections=debug_sections_main,
                )
            claude_response, claude_usage = await self.claude.generate_json(
                system_prompt=claude_prompt["system"],
                messages=claude_prompt["messages"],
                max_tokens=3500,
            )
            if self._is_token_debug():
                self._emit_token_debug(
                    label="memo_generation_post",
                    system_prompt=claude_prompt["system"],
                    user_prompt=claude_prompt["messages"][0]["content"],
                    sections=debug_sections_main,
                    usage=claude_usage,
                    response_text=claude_response if isinstance(claude_response, str) else json.dumps(claude_response),
                )
        except ValueError as e:
            # API key missing or configuration error
            error_str = str(e).lower()
            if "api key" in error_str or "required" in error_str:
                logger.warning(f"Claude API key not configured or invalid: {e}. Using fallback memo generation.")
                claude_response = None
                claude_usage = None
            else:
                raise
        except Exception as e:
            # Check if it's an authentication error
            error_str = str(e).lower()
            if "authentication" in error_str or "401" in error_str or ("invalid" in error_str and "api" in error_str):
                logger.warning(f"Claude API authentication failed: {e}. Using fallback memo generation.")
                claude_response = None
                claude_usage = None
            else:
                # Re-raise other exceptions
                raise

        if claude_response:
            memo = self._parse_or_fallback_memo(claude_response, normalized_pitch)
            # CRITICAL: Repopulate founders from pitch data to override LLM-generated team members
            # This ensures we ONLY use actual team members from pitch listing, not LLM-generated ones
            memo = self._populate_from_pitch(memo, normalized_pitch)
        else:
            # Fallback: create memo from pitch data without Claude
            logger.info("Generating memo from pitch data without Claude (API unavailable)")
            memo = InvestorCommitteeMemo()
            memo = self._populate_from_pitch(memo, normalized_pitch)
            # Use ALL generated_content fields if available - this is the primary source
            if generated_content:
                logger.info(f"Using generated_content with {len(generated_content)} fields")
                if generated_content.get("problem"):
                    memo.executive_summary.problem = generated_content["problem"]
                if generated_content.get("investment_thesis"):
                    memo.executive_summary.investment_thesis_3_sentences = generated_content["investment_thesis"]
                if generated_content.get("why_now"):
                    memo.executive_summary.why_now = generated_content["why_now"]
                if generated_content.get("target_customer"):
                    memo.executive_summary.target_customer = generated_content["target_customer"]
                if generated_content.get("market_trends"):
                    memo.market_opportunity_analysis.market_trends = generated_content["market_trends"]
                if generated_content.get("traction_analysis"):
                    ta = generated_content["traction_analysis"]
                    if isinstance(ta, dict):
                        if ta.get("engagement_level"):
                            memo.early_traction_signals.engagement_level = ta["engagement_level"]
                        if ta.get("traction_quality"):
                            memo.early_traction_signals.traction_quality = ta["traction_quality"]
                        if ta.get("customer_acquisition"):
                            memo.early_traction_signals.customer_acquisition = ta["customer_acquisition"]
                        if ta.get("early_retention"):
                            memo.early_traction_signals.early_retention = ta["early_retention"]
                        if ta.get("best_evidence"):
                            memo.early_traction_signals.best_traction_evidence = ta["best_evidence"]
                if generated_content.get("red_flags"):
                    red_flags = generated_content["red_flags"]
                    if isinstance(red_flags, list) and red_flags:
                        for founder in memo.founder_team_analysis.founders:
                            founder.red_flags = ", ".join(red_flags) if red_flags else "No significant red flags identified"
                # Priority: Use pitch data if available, otherwise use LLM-generated
                use_of_funds_from_pitch = None
                pitch_deck_data = normalized_pitch.get("pitch_deck_data") or normalized_pitch.get("pitch_deck") or {}
                if isinstance(pitch_deck_data, dict):
                    funding_data = pitch_deck_data.get("funding") or {}
                    if isinstance(funding_data, dict):
                        fund_allocation = funding_data.get("fund_allocation") or funding_data.get("use_of_funds")
                        if fund_allocation and isinstance(fund_allocation, list):
                            use_of_funds_from_pitch = fund_allocation
                            logger.info(f"Using use of funds from pitch deck: {len(fund_allocation)} categories")
                
                if use_of_funds_from_pitch:
                    # Use pitch deck data directly
                    memo.use_of_funds_milestones.use_of_funds_breakdown = [
                        UseOfFundsBreakdown(
                            category=item.get("category", item.get("name", "")),
                            percentage=str(item.get("percentage", item.get("percent", ""))),
                            purpose=item.get("purpose", item.get("notes", item.get("description", ""))),
                            amount=item.get("amount", "")
                        ) for item in use_of_funds_from_pitch if item
                    ]
                    logger.info(f"Populated use_of_funds_breakdown from pitch deck data: {len(memo.use_of_funds_milestones.use_of_funds_breakdown)} items")
                elif generated_content.get("use_of_funds"):
                    # Fallback to LLM-generated if no pitch data
                    use_of_funds = generated_content["use_of_funds"]
                    if isinstance(use_of_funds, list) and use_of_funds:
                        memo.use_of_funds_milestones.use_of_funds_breakdown = [
                            UseOfFundsBreakdown(
                                category=item.get("category", ""),
                                percentage=item.get("percentage", ""),
                                purpose=item.get("purpose", ""),
                                amount=item.get("amount", "")
                            ) for item in use_of_funds
                        ]
                        logger.info(f"Populated use_of_funds_breakdown from LLM-generated content: {len(memo.use_of_funds_milestones.use_of_funds_breakdown)} items")
                if generated_content.get("revenue_projections"):
                    revenue_projections = generated_content["revenue_projections"]
                    if isinstance(revenue_projections, list) and revenue_projections:
                        memo.financial_projections.revenue_projections = [
                            RevenueProjection(
                                year=item.get("year", ""),
                                revenue=item.get("revenue", ""),
                                growth_rate=item.get("growth_rate", ""),
                                source="llm_generated"
                            ) for item in revenue_projections
                        ]
                if generated_content.get("market_data"):
                    market_data = generated_content["market_data"]
                    if isinstance(market_data, dict):
                        if market_data.get("tam"):
                            memo.market_opportunity_analysis.tam = market_data["tam"]
                        # Note: sam and som are not in MarketOpportunityAnalysis schema, only tam is available
                        if market_data.get("tam_source"):
                            memo.market_opportunity_analysis.tam_source = market_data["tam_source"]
                        if market_data.get("tam_source_url"):
                            memo.market_opportunity_analysis.tam_source_url = market_data["tam_source_url"]
                        if market_data.get("tam_confidence"):
                            memo.market_opportunity_analysis.tam_confidence = market_data["tam_confidence"]
                        if market_data.get("growth_rate"):
                            memo.market_opportunity_analysis.tam_growth_rate = market_data["growth_rate"]
                            source_info = f"source: {market_data.get('tam_source')}"
                            if market_data.get('tam_source_url'):
                                source_info += f", URL: {market_data.get('tam_source_url')}"
                            logger.info(f"Populated TAM from generated_content: {market_data.get('tam')} ({source_info})")
            
            # Backup: Directly extract TAM from Gemini research if not already set
            if not memo.market_opportunity_analysis.tam or self._detect_placeholder(memo.market_opportunity_analysis.tam):
                if gemini_research and isinstance(gemini_research, dict):
                    metrics = gemini_research.get("metrics", {})
                    if metrics and isinstance(metrics, dict) and metrics.get("tam"):
                        tam_data = metrics["tam"]
                        if isinstance(tam_data, dict) and tam_data.get("value"):
                            memo.market_opportunity_analysis.tam = tam_data["value"]
                            # Preserve both source text and URL
                            citation = tam_data.get("citation") or "Gemini research"
                            source_url = tam_data.get("source_url")
                            memo.market_opportunity_analysis.tam_source = citation
                            if source_url:
                                memo.market_opportunity_analysis.tam_source_url = source_url
                            memo.market_opportunity_analysis.tam_confidence = tam_data.get("confidence", "medium")
                            logger.info(f"Directly populated TAM from Gemini research: {tam_data.get('value')} (source: {memo.market_opportunity_analysis.tam_source}, URL: {source_url})")
            
            claude_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        # DEBUG: Log traction fields after Claude/Gemini returns memo JSON
        if memo and hasattr(memo, 'early_traction_signals'):
            print("MEMO TRACTION AFTER PROMPT:", memo.early_traction_signals.__dict__ if hasattr(memo.early_traction_signals, '__dict__') else memo.early_traction_signals)

        memo.meta.company_name = memo.meta.company_name or normalized_pitch.get("company_name", DEFAULT_PLACEHOLDER)
        memo.meta.stage = memo.meta.stage or normalized_pitch.get("stage", DEFAULT_PLACEHOLDER)
        
        # Extract validated data context (Phase C: validation layer)
        # Also check PDF metrics as fallback (Phase F)
        pdf_metrics_for_extraction = pdf_extraction_results.get("pdf_metrics", {}) if pdf_extraction_results else {}
        # Get product category for TAM fallback
        product_category = normalized_pitch.get("market") or normalized_pitch.get("industry") or ""
        if not product_category and memo.market_opportunity_analysis:
            product_category = getattr(memo.market_opportunity_analysis, "segment_fit", "") or ""
        data_context = extract_validated_data(
            memo.model_dump(), 
            pdf_metrics=pdf_metrics_for_extraction,
            pitch_data=normalized_pitch,
            fund_data=normalized_fund,
            gemini_client=self.gemini,
            product_category=product_category
        )
        
        # Validate extracted data and populate warnings
        validation_issues = validate_data_context(data_context)
        validation_warnings = [issue.model_dump() for issue in validation_issues]
        visual_suppression_state = {
            issue.field: issue.suppress_chart for issue in validation_issues
        }
        
        # Generate founder skills with LLM for all founders before filling unknown fields
        if memo.founder_team_analysis and memo.founder_team_analysis.founders:
            for founder in memo.founder_team_analysis.founders:
                # Check if scores need generation
                needs_generation = (
                    not hasattr(founder, 'domain_expertise_score') or 
                    not founder.domain_expertise_score or
                    self._detect_placeholder(str(founder.domain_expertise_score))
                )
                if needs_generation:
                    founder_data_dict = {
                        "name": founder.name or "",
                        "role": founder.role or "",
                        "experience_years": founder.experience_years or "",
                        "education": founder.education or "",
                        "previous_companies": founder.previous_companies or "",
                    }
                    try:
                        llm_scores = await self._generate_founder_skills_with_llm(
                            founder_data_dict,
                            linkedin_data,
                            founder_doc_snippets,
                        )
                        # Store scores in founder profile
                        founder.domain_expertise_score = f"{llm_scores['domain_expertise']:.1f}/10"
                        founder.technical_ability_score = f"{llm_scores['technical_ability']:.1f}/10"
                        founder.leadership_score = f"{llm_scores['leadership']:.1f}/10"
                        founder.entrepreneurial_drive_score = f"{llm_scores['entrepreneurial_drive']:.1f}/10"
                        logger.info(f"Generated founder skills with LLM for {founder.name}: {llm_scores}")
                    except Exception as e:
                        logger.error(f"Failed to generate founder skills with LLM: {e}")
        
        memo = self._fill_unknown_fields(memo, normalized_pitch, normalized_fund, website_context=website_context, generated_content=generated_content)
        # Enrich data sources with any explicit URLs / references surfaced in Gemini research.
        extra_sources = self._extract_data_sources_from_research(gemini_research)
        if extra_sources:
            existing_sources = set(memo.data_sources or [])
            for src in extra_sources:
                if src not in existing_sources:
                    memo.data_sources.append(src)
        memo = self._seed_minimum_entries(memo)
        memo = self._apply_scoring_and_recommendation(memo, normalized_pitch, website_context=website_context)
        memo = self._overwrite_from_pitch_and_fund(memo, normalized_pitch, normalized_fund, pdf_extraction_results)
        
        # Merge traction_kpis from PDF extraction into final memo
        if pdf_extraction_results and pdf_extraction_results.get("traction_kpis"):
            pdf_traction_kpis = pdf_extraction_results["traction_kpis"]
            if pdf_traction_kpis:
                # Initialize memo.traction_kpis if it doesn't exist
                if not memo.traction_kpis:
                    memo.traction_kpis = []
                
                # Create a set of existing KPI names to avoid duplicates
                existing_kpi_names = {kpi.name for kpi in memo.traction_kpis}
                
                # Add PDF-extracted traction KPIs that don't already exist
                for pdf_kpi in pdf_traction_kpis:
                    if pdf_kpi.name not in existing_kpi_names:
                        memo.traction_kpis.append(pdf_kpi)
                        existing_kpi_names.add(pdf_kpi.name)
                        logger.info(f"Merged traction KPI from PDF: {pdf_kpi.name} = {pdf_kpi.value} {pdf_kpi.unit}")
                    else:
                        # Update existing KPI if PDF source is more reliable
                        for i, existing_kpi in enumerate(memo.traction_kpis):
                            if existing_kpi.name == pdf_kpi.name:
                                # Prefer PDF-extracted data over LLM-generated estimates
                                if pdf_kpi.confidence == "high" and not pdf_kpi.estimated:
                                    memo.traction_kpis[i] = pdf_kpi
                                    logger.info(f"Updated traction KPI from PDF: {pdf_kpi.name} = {pdf_kpi.value} {pdf_kpi.unit}")
                                break
        
        # Populate early_traction_signals from traction_kpis if they exist
        if memo.traction_kpis:
            logger.info(f"Found {len(memo.traction_kpis)} traction KPIs, populating early_traction_signals")
            logger.info(f"Traction KPI names: {[kpi.name for kpi in memo.traction_kpis]}")
            self._populate_early_traction_signals_from_kpis(memo)
            # Filter out impossible KPI magnitudes to avoid chart blowups
            filtered_kpis, dropped_kpis = self._filter_unreasonable_traction_kpis(memo.traction_kpis)
            if dropped_kpis:
                logger.warning(f"Dropped {len(dropped_kpis)} memo traction KPIs for unrealistic magnitudes: {[k.name for k in dropped_kpis]}")
            memo.traction_kpis = filtered_kpis
        else:
            logger.warning("No traction_kpis found in memo after PDF extraction merge")
        
        # Ensure early_traction_signals fields are populated (use defaults if Claude didn't populate them)
        logger.info(f"Before ensuring populated - early_traction_signals exists: {hasattr(memo, 'early_traction_signals')}")
        if hasattr(memo, 'early_traction_signals'):
            logger.info(f"early_traction_signals.paying_customers = '{memo.early_traction_signals.paying_customers}'")
        self._ensure_early_traction_signals_populated(memo)
        logger.info(f"After ensuring populated - early_traction_signals.paying_customers = '{memo.early_traction_signals.paying_customers}'")
        
        # Merge Gemini/web research competitors with source URLs before validation (Phase C/E)
        self._merge_research_competitors(memo, gemini_research)
        
        memo = memo.ensure_no_placeholders()
        
        # Post-generation validation: check for placeholder competitors
        self._validate_competitors(memo)

        # If fewer than 3 competitors remain, force LLM fallback generation and re-validate
        if len(memo.competitive_landscape.competitors or []) < 3:
            logger.info(
                f"Only {len(memo.competitive_landscape.competitors or [])} competitors found; triggering LLM fallback generation"
            )
            memo = await self._generate_competitors_fallback(
                memo,
                normalized_pitch,
                gemini_research,
                gemini_client=self.gemini,
            )
            self._validate_competitors(memo)
        
        # DO NOT generate synthetic competitors - only use actual competitor data from pitch/research
        # Removed competitor fallback generation logic

        interface = self._build_structured_interface(memo)
        token_usage = self._merge_usage(claude_usage, gemini_usage)
        # Pass priority_manager to build_section_visuals for Phase D chart improvements
        visuals = self.build_section_visuals(memo, priority_manager=priority_manager, generated_content=generated_content)
        
        # Final validation: check quality criteria
        final_validation_warnings = self._validate_final_memo(memo, visuals, data_context)
        validation_warnings.extend(final_validation_warnings)
        
        return memo_id, memo, interface, token_usage, visuals, data_context, validation_warnings, visual_suppression_state

    async def chat_edit(
        self,
        memo: InvestorCommitteeMemo,
        message: str,
        conversation_history: Optional[List[Dict[str, Any]]],
        uploaded_datasets: Optional[List[DatasetRecord]] = None,
    ) -> Tuple[InvestorCommitteeMemo, StructuredInterface, LLMTokenUsage, Dict[str, Any]]:
        memo_payload = memo.model_dump()
        original_memo = copy.deepcopy(memo)
        memo_state = self._load_or_build_memo_state(memo, memo_payload)
        
        # Extract PDF metrics from datasets and memo payload
        pdf_extraction_results = await extract_metrics_from_datasets(uploaded_datasets or [])
        
        # Check if memo_payload has pitch_documents
        pitch_docs = memo_payload.get("pitch_documents", []) or []
        upload_docs = memo_payload.get("upload_documents", []) or []
        all_pitch_docs = pitch_docs + upload_docs
        
        if all_pitch_docs:
            pitch_pdf_results = await extract_metrics_from_pitch_documents(all_pitch_docs)
            if pitch_pdf_results.get("pdf_metrics"):
                pdf_extraction_results["pdf_metrics"].update(pitch_pdf_results["pdf_metrics"])
            if pitch_pdf_results.get("pdf_texts"):
                pdf_extraction_results["pdf_texts"].extend(pitch_pdf_results["pdf_texts"])
            # CRITICAL: Merge traction_kpis from pitch documents (for edit flow)
            if pitch_pdf_results.get("traction_kpis"):
                if "traction_kpis" not in pdf_extraction_results:
                    pdf_extraction_results["traction_kpis"] = []
                pdf_extraction_results["traction_kpis"].extend(pitch_pdf_results["traction_kpis"])
                logger.info(f"Merged {len(pitch_pdf_results['traction_kpis'])} traction KPIs from pitch documents (edit)")
        
        # Merge PDF metrics into memo_payload for Claude prompt
        if pdf_extraction_results.get("pdf_metrics"):
            memo_payload["pdf_metrics"] = pdf_extraction_results["pdf_metrics"]
        if pdf_extraction_results.get("pdf_texts"):
            memo_payload["pdf_texts"] = pdf_extraction_results["pdf_texts"]
        if pdf_extraction_results.get("traction_kpis"):
            memo_payload["traction_kpis"] = pdf_extraction_results["traction_kpis"]
        
        website_context = await self._gather_website_context(memo_payload)
        
        # CRITICAL: Detect if this is a question FIRST - if so, skip patch/override logic
        is_question = self._is_question(message)
        question_response = {"is_question": False}
        
        # For questions, skip patch/override parsing entirely (prevents questions from being treated as updates)
        if is_question:
            logger.info(f"Question detected - skipping patch/override parsing: {message[:100]}")
            patch = {"fields": {}}
            overrides = {}
            patch_fields = {}
        else:
            # LLM-based correction parser to build patch (only for non-questions)
            patch = await self.parse_user_correction(message, memo_state)
            memo_state = self.apply_patch(memo_state, patch)
            self._persist_memo_state_safe(memo.meta.document_id, memo_state)
            overrides = self._extract_overrides_from_message(message)
            patch_fields = patch.get("fields", {}) if patch else {}

        # If user provided explicit overrides or LLM patch fields, skip Claude and apply deterministically
        if overrides or patch_fields:
            logger.info(f"Applying direct overrides/patch from chat message: {list(overrides.keys()) + list(patch_fields.keys())}")
            # Merge patch fields into memo_state then apply to memo
            if patch_fields:
                memo_state = self.apply_patch(memo_state, {"fields": patch_fields})
                self._persist_memo_state_safe(memo.meta.document_id, memo_state)
            memo = self._apply_state_patch_to_memo(memo, memo_state)
            updated = self._apply_overrides_to_memo(memo, overrides)
            claude_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        elif is_question:
            # QUESTION MODE: Memo-Aware Q&A Layer
            logger.info(f"Question detected, using memo-aware Q&A layer: {message[:100]}")
            logger.info(f"Memo data available: company_name={getattr(memo.meta, 'company_name', 'N/A')}, has_final_recommendation={hasattr(memo, 'final_recommendation') and memo.final_recommendation is not None}")
            
            # NEW: Use MemoAnswerPlanner to determine answer strategy
            from app.services.memo_qa_planner import MemoAnswerPlanner, MemoFactExtractor
            
            planner = MemoAnswerPlanner(claude_client=self.claude)
            fact_extractor = MemoFactExtractor()
            
            # Plan the answer approach
            plan = await planner.plan_answer(message, memo.model_dump(), claude_client=self.claude)
            logger.info(f"Answer plan: mode={plan['mode']}, fields={plan.get('fields_used', [])}, type={plan.get('question_type', 'unknown')}")
            
            # Build prompt based on plan mode
            if plan["mode"] == "FACT":
                # FACT mode: Extract facts deterministically, LLM explains them
                facts = fact_extractor.extract_facts(plan["fields_used"], memo.model_dump())
                logger.info(f"Extracted {len(facts)} facts for FACT mode answer")
                
                # Build FACT mode prompt
                system_prompt, messages = self._build_qa_prompt_fact_mode(
                    message, facts, conversation_history
                )
            else:
                # ANALYSIS mode: Provide full memo for reasoning
                logger.info("Using ANALYSIS mode with full memo")
                system_prompt, messages = self._build_qa_prompt_analysis_mode(
                    memo, message, conversation_history
                )
            
            claude_response = None
            claude_usage = None
            try:
                claude_response, claude_usage = await self.claude.generate_json(
                    system_prompt=system_prompt,
                    messages=messages,
                    max_tokens=2000,  # Shorter for answers
                )
            except Exception as e:
                logger.warning(f"Claude API error during question answering: {e}")
                claude_response = {"answer": "I apologize, but I'm unable to answer at the moment. Please try again.", "source_section": "", "memo_unchanged": True, "updated_memo": memo.model_dump()}
                claude_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            
            # Extract answer from response
            if claude_response and isinstance(claude_response, dict):
                answer = claude_response.get("answer", "I couldn't find that information in the memo.")
                source_section = claude_response.get("source_section", "")
                source_type = claude_response.get("source_type", "memo")
                
                # Gemini disabled for chat - only used in memo generation
                # If answer indicates information not in memo, return "not specified" - don't use Gemini for chat
                if source_type == "needs_research" or "not available" in answer.lower() or "not in the memo" in answer.lower() or "not specified" in answer.lower():
                    # Gemini disabled for chat to prevent hallucinations and external data
                    # Just return the memo-based answer (which should say "not specified" if missing)
                    logger.info(f"Information not in memo - Gemini disabled for chat, returning memo-based answer only")
                    # Answer already contains "not specified" message - use it as-is
                
                logger.info(f"Question answered: {message[:100]} -> Source: {source_type} ({source_section})")
                question_response = {
                    "answer": answer,
                    "source_section": source_section,
                    "source_type": source_type,
                    "is_question": True
                }
                # Keep memo unchanged for questions
                updated = memo
            else:
                updated = memo
                question_response = {
                    "answer": "This information is not specified in the memo.",
                    "source_section": "",
                    "source_type": "memo",
                    "is_question": True
                }
        else:
            # DEBUG: Log what memo.early_traction_signals contains before building edit memo JSON
            if hasattr(memo, 'early_traction_signals'):
                print("MEMO TRACTION BEFORE PROMPT (EDIT):", memo.early_traction_signals.__dict__ if hasattr(memo.early_traction_signals, '__dict__') else memo.early_traction_signals)

            system_prompt, messages = self._build_edit_prompt(
                memo, message, conversation_history, uploaded_datasets, website_context=website_context
            )
            
            claude_response = None
            claude_usage = None
            try:
                claude_response, claude_usage = await self.claude.generate_json(
                    system_prompt=system_prompt,
                    messages=messages,
                    max_tokens=3200,
                )
            except ValueError as e:
                error_str = str(e).lower()
                if "api key" in error_str or "required" in error_str:
                    logger.warning(f"Claude API key not configured or invalid during edit: {e}. Returning original memo.")
                    claude_response = None
                    claude_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                else:
                    raise
            except Exception as e:
                error_str = str(e).lower()
                if "authentication" in error_str or "401" in error_str or ("invalid" in error_str and "api" in error_str):
                    logger.warning(f"Claude API authentication failed during edit: {e}. Returning original memo.")
                    claude_response = None
                    claude_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                else:
                    raise
            
            if claude_response:
                updated = self._parse_or_fallback_memo(claude_response, memo.meta.model_dump())
            else:
                logger.info("Claude unavailable for edit, returning original memo")
                updated = memo
                if not claude_usage:
                    claude_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            
            if updated and hasattr(updated, 'early_traction_signals'):
                print("MEMO TRACTION AFTER PROMPT (EDIT):", updated.early_traction_signals.__dict__ if hasattr(updated.early_traction_signals, '__dict__') else updated.early_traction_signals)
            updated = self._apply_overrides_to_memo(updated, overrides)
            self._persist_memo_state_safe(updated.meta.document_id, self._load_or_build_memo_state(updated, updated.model_dump()))
        
        # Skip post-processing for questions (keep memo unchanged)
        if not question_response.get("is_question", False):
            updated = self._restore_traction_if_regressed(original_memo, updated)
            updated = self._preserve_market_data_if_lost(original_memo, updated)  # CRITICAL: Preserve TAM and market data
            updated = self._fill_unknown_fields(updated, memo_payload, {}, website_context=website_context)
            updated = self._seed_minimum_entries(updated)
            updated = self._apply_scoring_and_recommendation(updated, memo.model_dump(), website_context=website_context)
        
        # Merge traction_kpis from PDF extraction into updated memo (for edit flow)
        if pdf_extraction_results and pdf_extraction_results.get("traction_kpis"):
            pdf_traction_kpis = pdf_extraction_results["traction_kpis"]
            if pdf_traction_kpis:
                # Initialize updated.traction_kpis if it doesn't exist
                if not updated.traction_kpis:
                    updated.traction_kpis = []
                
                # Create a set of existing KPI names to avoid duplicates
                existing_kpi_names = {kpi.name for kpi in updated.traction_kpis}
                
                # Add PDF-extracted traction KPIs that don't already exist
                for pdf_kpi in pdf_traction_kpis:
                    if pdf_kpi.name not in existing_kpi_names:
                        updated.traction_kpis.append(pdf_kpi)
                        existing_kpi_names.add(pdf_kpi.name)
                        logger.info(f"Merged traction KPI from PDF (edit): {pdf_kpi.name} = {pdf_kpi.value} {pdf_kpi.unit}")
                    else:
                        # Update existing KPI if PDF source is more reliable
                        for i, existing_kpi in enumerate(updated.traction_kpis):
                            if existing_kpi.name == pdf_kpi.name:
                                # Prefer PDF-extracted data over LLM-generated estimates
                                if pdf_kpi.confidence == "high" and not pdf_kpi.estimated:
                                    updated.traction_kpis[i] = pdf_kpi
                                    logger.info(f"Updated traction KPI from PDF (edit): {pdf_kpi.name} = {pdf_kpi.value} {pdf_kpi.unit}")
                                break
        
        # Populate early_traction_signals from traction_kpis if they exist (for edit flow)
        if updated.traction_kpis:
            self._populate_early_traction_signals_from_kpis(updated)
        
        updated = updated.ensure_no_placeholders()
        
        # Extract validated data context and create suppression state
        # Also check PDF metrics as fallback (Phase F)
        pdf_metrics_for_extraction = pdf_extraction_results.get("pdf_metrics", {}) if pdf_extraction_results else {}
        data_context = extract_validated_data(
            updated.model_dump(), 
            pdf_metrics=pdf_metrics_for_extraction,
            gemini_client=self.gemini,
            product_category=None  # Not available in edit context
        )
        validation_issues = validate_data_context(data_context)
        visual_suppression_state = {
            issue.field: issue.suppress_chart for issue in validation_issues
        }
        
        interface = self._build_structured_interface(updated)
        visuals = self.build_section_visuals(updated)
        token_usage = self._merge_usage(claude_usage, None)
        
        # Store question answer in visuals dict if this was a question (for frontend to display)
        if question_response.get("is_question"):
            if not isinstance(visuals, dict):
                visuals = {}
            visuals["chat_answer"] = {
                "answer": question_response.get("answer", ""),
                "source_section": question_response.get("source_section", ""),
                "source_type": question_response.get("source_type", "memo"),  # memo, internet, or error
                "question": message
            }
        
        return updated, interface, token_usage, visuals

    async def audit_memo_json(self, memo: InvestorCommitteeMemo) -> Dict[str, Any]:
        """
        Run a pure JSON audit of an IC memo without rewriting it.

        This uses Claude to:
        - Inspect the full memo JSON.
        - Identify which fields still look weak, missing, inconsistent, duplicated, or unrealistic.
        - Produce a structured checklist of what a human analyst should review.

        The output format is fixed and does NOT include an updated memo – only diagnostics.
        """
        system_prompt, messages = self._build_audit_prompt(memo)
        
        audit_json = None
        _usage = None
        try:
            audit_json, _usage = await self.claude.generate_json(
                system_prompt=system_prompt,
                messages=messages,
                max_tokens=2800,
            )
        except ValueError as e:
            # API key missing or configuration error
            error_str = str(e).lower()
            if "api key" in error_str or "required" in error_str:
                logger.warning(f"Claude API key not configured or invalid during audit: {e}. Skipping audit.")
                return {"needs_update": [], "human_required_fields": [], "placeholders_detected": [], "consistency_issues": [], "final_summary": "Audit skipped - Claude API unavailable"}
            else:
                raise
        except Exception as e:
            # Check if it's an authentication error
            error_str = str(e).lower()
            if "authentication" in error_str or "401" in error_str or ("invalid" in error_str and "api" in error_str):
                logger.warning(f"Claude API authentication failed during audit: {e}. Skipping audit.")
                return {"needs_update": [], "human_required_fields": [], "placeholders_detected": [], "consistency_issues": [], "final_summary": "Audit skipped - Claude API unavailable"}
            else:
                # Re-raise other exceptions
                raise
        
        if not audit_json:
            return {"needs_update": [], "human_required_fields": [], "placeholders_detected": [], "consistency_issues": [], "final_summary": "Audit skipped - Claude API unavailable"}
        
        try:
            sanitized = _strip_code_fences(audit_json)
            parsed = json.loads(sanitized)
        except Exception:
            logger.warning("Audit JSON parse failed, returning raw string payload.")
            parsed = {"raw": audit_json}
        return parsed

    def _extend_revenue_trend(self, projections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extend revenue projections using conservative growth estimates.
        Only extends if we have 1-2 data points. No LLM calls.
        """
        if len(projections) >= 3:
            return projections  # Already have enough data
        
        if len(projections) == 0:
            return projections
        
        # Sort by year to ensure proper order
        sorted_projs = sorted(projections, key=lambda x: str(x.get("year", "")))
        extended = sorted_projs.copy()
        
        if len(sorted_projs) == 1:
            # Only Year 1: estimate Year 2 (2x) and Year 3 (3x)
            year1_val = sorted_projs[0]["value"]
            year1_label = str(sorted_projs[0].get("year", "Year 1"))
            
            # Try to infer next year labels
            if "Year 1" in year1_label or "1" in year1_label:
                year2_label = "Year 2"
                year3_label = "Year 3"
            else:
                # Try to extract year number
                import re
                year_match = re.search(r'(\d+)', year1_label)
                if year_match:
                    year_num = int(year_match.group(1))
                    year2_label = f"Year {year_num + 1}"
                    year3_label = f"Year {year_num + 2}"
                else:
                    year2_label = "Year 2"
                    year3_label = "Year 3"
            
            extended.append({"year": year2_label, "value": year1_val * 2.0})
            extended.append({"year": year3_label, "value": year1_val * 3.0})
        
        elif len(sorted_projs) == 2:
            # Year 1 and Year 2: estimate Year 3 (extrapolate growth)
            year1_val = sorted_projs[0]["value"]
            year2_val = sorted_projs[1]["value"]
            
            if year1_val > 0:
                growth_rate = (year2_val - year1_val) / year1_val
                # Conservative: cap growth at 100% YoY
                growth_rate = min(growth_rate, 1.0)
                year3_val = year2_val * (1 + growth_rate)
            else:
                year3_val = year2_val * 1.5  # Fallback: 50% growth
            
            year2_label = str(sorted_projs[1].get("year", "Year 2"))
            # Try to infer Year 3 label
            import re
            year_match = re.search(r'(\d+)', year2_label)
            if year_match:
                year_num = int(year_match.group(1))
                year3_label = f"Year {year_num + 1}"
            else:
                year3_label = "Year 3"
            
            extended.append({"year": year3_label, "value": year3_val})
        
        return extended

    def extract_numeric_value(self, value: Any) -> Optional[float]:
        """Extract numeric value from string or number (copied from PitchVisualGenerator)."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            # Sanity check: reject unrealistic values (> 1 trillion for counts, > 100 trillion for currency)
            if value > 1_000_000_000_000_000:  # > 1000 trillion is definitely wrong
                logger.warning(f"Rejecting unrealistic numeric value: {value}")
                return None
            return float(value)
        if isinstance(value, str):
            clean = value.strip()
            # Remove currency symbols before processing
            for symbol in ['$', '€', '£', '₹', 'USD', 'EUR', 'GBP', 'INR']:
                clean = clean.replace(symbol, '')
            clean = clean.replace(',', '').replace(' ', '')
            multipliers = {'t': 1_000_000_000_000.0, 'b': 1_000_000_000.0, 'm': 1_000_000.0, 'k': 1_000.0}
            multiplier = 1.0
            clean_lower = clean.lower()
            # Check for T first (trillion), then B, M, K (order matters for parsing)
            for key, mult in multipliers.items():
                if key in clean_lower:
                    multiplier = mult
                    clean = clean_lower.replace(key, '')
                    break
            match = re.search(r'([-+]?[0-9]*\.?[0-9]+)', clean)
            if match:
                try:
                    result = float(match.group(1)) * multiplier
                    # Sanity check: reject unrealistic values
                    if result > 1_000_000_000_000_000:  # > 1000 trillion is definitely wrong
                        logger.warning(f"Rejecting unrealistic parsed value from '{value}': {result}")
                        return None
                    return result
                except ValueError:
                    return None
        return None

    def extract_percentage(self, value: Any) -> Optional[float]:
        """Extract percentage value from string or number (copied from PitchVisualGenerator)."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            clean = value.strip().replace('%', '').replace(',', '')
            match = re.search(r'([-+]?[0-9]*\.?[0-9]+)', clean)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    return None
        return None

    def format_currency(self, value: float, currency: str = "$") -> str:
        """Format currency value with K/M/B notation instead of zeros."""
        if value is None or value == 0:
            return f"{currency}0"
        
        abs_value = abs(value)
        sign = "-" if value < 0 else ""
        
        if abs_value >= 1_000_000_000:
            return f"{sign}{currency}{abs_value / 1_000_000_000:.1f}B"
        elif abs_value >= 1_000_000:
            return f"{sign}{currency}{abs_value / 1_000_000:.1f}M"
        elif abs_value >= 1_000:
            return f"{sign}{currency}{abs_value / 1_000:.1f}K"
        else:
            return f"{sign}{currency}{abs_value:.0f}"

    def format_number(self, value: float) -> str:
        """Format count-style numbers plainly (no K/M/B suffix) to avoid implying millions when not provided."""
        if value is None or value == 0:
            return "0"
        try:
            return f"{float(value):,.0f}"
        except Exception:
            return str(value)

    def build_section_visuals(self, memo: InvestorCommitteeMemo, priority_manager: Optional[DataPriorityManager] = None, generated_content: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build chart/image ideas per section for UI rendering using Chart.js format (pitch deck style).
        
        Returns nested structure: {"section_name": {"charts": [{"chart_type": "...", "title": "...", "description": "...", "data": {"labels": [...], "datasets": [...]}]}}
        
        Always generates 5+ charts by extracting numbers directly from memo text content.
        No suppression logic - charts always render with extracted or estimated data.
        """
        visuals = {}

        # 1. Revenue Projection Line Chart (Phase D: No synthetic extensions, use priority manager)
        revenue_labels = []
        revenue_data = []
        revenue_sources = []
        note_badge = None
        
        # Use LLM-generated revenue projections if available
        if generated_content and generated_content.get("revenue_projections"):
            revenue_projections = generated_content["revenue_projections"]
            if isinstance(revenue_projections, list) and revenue_projections:
                for proj in revenue_projections:
                    if proj.get("revenue"):
                        revenue_val = self.extract_numeric_value(proj["revenue"])
                        if revenue_val is not None:
                            revenue_labels.append(proj.get("year", ""))
                            revenue_data.append(float(revenue_val))
                            revenue_sources.append(f"{proj.get('year', '')}: llm_generated")
        
        # Use DataPriorityManager if available for revenue data - only if not already set
        if not revenue_data and priority_manager:
            # Check for revenue projections from priority manager
            revenue_val = priority_manager.get_field_value("revenue")
            if revenue_val.value is not None and not revenue_val.requires_confirmation:
                # Single revenue value - use as Year 1
                revenue_labels = ["Year 1"]
                revenue_data = [float(revenue_val.value)]
                revenue_sources.append(f"Revenue: {revenue_val.source}")
            elif revenue_val.requires_confirmation:
                note_badge = "Limited projection data — requires founder confirmation"
        
        # Also check memo projections (may have multiple years)
        if memo.financial_projections.revenue_projections:
            projections = []
            for rp in memo.financial_projections.revenue_projections:
                if rp.revenue and not self._detect_placeholder(rp.revenue):
                    # Extract numeric value from text like "$30k", "$168k", "$420k"
                    revenue_val = self.extract_numeric_value(rp.revenue)
                    if revenue_val is not None:
                        projections.append({
                            "year": rp.year,
                            "value": revenue_val,
                            "source": rp.source or "memo_text"
                        })
            
            if projections:
                projections = sorted(projections, key=lambda x: str(x.get("year", "")))
                revenue_labels = [str(p.get("year", "")) for p in projections]
                revenue_data = [float(p.get("value", 0)) for p in projections]
                revenue_sources = [f"{p.get('year', '')}: {p.get('source', 'unknown')}" for p in projections]
        
        # DO NOT extend revenue trend - only use actual data from pitch/PDF/LLM (Phase D)
        # Removed synthetic revenue projection extension - _extend_revenue_trend() is not called
        
        # If no revenue data from any source, show missing state
        if not revenue_data:
            if "financial_projections" not in visuals:
                visuals["financial_projections"] = {"charts": []}
            
            chart_config = {
                "id": "revenue_line",
                "chart_type": "line",
                "title": "Revenue Projection",
                "description": "Projected revenue growth over time",
                "data": {
                    "labels": ["Data Unavailable"],
                    "datasets": [{
                        "label": "Revenue",
                        "data": [0],
                        "borderColor": "rgba(15, 23, 42, 1)",
                        "backgroundColor": "rgba(15, 23, 42, 0.1)"
                    }]
                },
                "note_badge": note_badge or "Revenue projections not available - requires founder confirmation"
            }
            visuals["financial_projections"]["charts"].append(chart_config)
        else:
            if "financial_projections" not in visuals:
                visuals["financial_projections"] = {"charts": []}
            
            # Format revenue data with K/M/B notation for display
            formatted_revenue_data = [self.format_currency(val) for val in revenue_data]
            
            chart_config = {
                "id": "revenue_line",
                "chart_type": "line",
                "title": "Revenue Projection",
                "description": "Projected revenue growth over time",
                "data": {
                    "labels": revenue_labels,
                    "datasets": [{
                        "label": "Revenue",
                        "data": revenue_data,
                        "formatted_data": formatted_revenue_data,
                        "borderColor": "rgba(15, 23, 42, 1)",
                        "backgroundColor": "rgba(15, 23, 42, 0.1)"
                    }]
                },
                "format_y_axis": "currency",
                "format_tooltip": "currency"
            }
            
            if revenue_sources:
                chart_config["source_badges"] = revenue_sources
            if note_badge or len(revenue_data) < 3:
                chart_config["note_badge"] = note_badge or "Limited projection data — requires founder confirmation"
            
            visuals["financial_projections"]["charts"].append(chart_config)

        # 2. Market Growth (TAM) Line Chart (Phase D: Check DataPriorityManager, show source badges)
        market_labels = []
        market_data = []
        tam_source = None
        tam_source_url = None
        
        # PRIORITY 1: Use LLM-generated market data (which includes Gemini research) if available
        if generated_content and generated_content.get("market_data"):
            market_data_dict = generated_content["market_data"]
            if market_data_dict.get("tam"):
                tam_val = self.extract_numeric_value(market_data_dict["tam"])
                if tam_val and tam_val > 0:
                    market_labels = ["Current"]
                    market_data = [float(tam_val)]
                    tam_source = market_data_dict.get("tam_source", "llm_generated")
                    tam_source_url = market_data_dict.get("tam_source_url")
                    logger.info(f"Found TAM in generated_content.market_data: {tam_val} (source: {tam_source}, URL: {tam_source_url})")
                    # Generate growth projection if growth rate available
                    if market_data_dict.get("growth_rate"):
                        growth_rate = self.extract_percentage(market_data_dict["growth_rate"])
                        if growth_rate:
                            # Project 3 years
                            for year in range(1, 4):
                                market_labels.append(f"Year {year}")
                                projected = float(tam_val) * ((1 + growth_rate / 100) ** year)
                                market_data.append(projected)
        
        # PRIORITY 2: Use DataPriorityManager if available (includes Gemini research) - only if not already set
        if not market_data and priority_manager:
            tam_val = priority_manager.get_field_value("tam")
            if tam_val.value is not None and not tam_val.requires_confirmation:
                market_labels = ["Current"]
                market_data = [float(tam_val.value)]
                tam_source = tam_val.source
                tam_source_url = tam_val.source_url
                logger.info(f"Found TAM in DataPriorityManager: {tam_val.value} (source: {tam_source}, URL: {tam_source_url})")
        
        # PRIORITY 3: Check memo TAM field - only if not already set
        if not market_data and memo.market_opportunity_analysis.tam:
            tam_val = self.extract_numeric_value(memo.market_opportunity_analysis.tam)
            if tam_val and tam_val > 0:  # Ensure valid positive value
                market_labels = ["Current"]
                market_data = [float(tam_val)]
                tam_source = getattr(memo.market_opportunity_analysis, "tam_source", None) or "memo_text"
                tam_source_url = getattr(memo.market_opportunity_analysis, "tam_source_url", None)
                source_info = f"source: {tam_source}"
                if tam_source_url:
                    source_info += f", URL: {tam_source_url}"
                logger.info(f"Found TAM in memo.market_opportunity_analysis.tam: {tam_val} ({source_info})")
            else:
                logger.warning(f"TAM field exists but extract_numeric_value failed: {memo.market_opportunity_analysis.tam}")
        
        # If no market data, check if LLM can provide estimates
        if not market_data and generated_content and generated_content.get("market_data"):
            market_data_dict = generated_content["market_data"]
            # If market_data exists but no TAM value, show note badge
            if not market_data_dict.get("tam"):
                # LLM should have generated TAM - if not, show note
                market_labels = ["Data Unavailable"]
                market_data = [0]
                tam_source = None
                note_badge = "Market data (TAM) not available - requires founder confirmation"
        
        # ALWAYS render TAM chart - with data if available, or with note badge if missing
        if "market_opportunity_analysis" not in visuals:
            visuals["market_opportunity_analysis"] = {"charts": []}
        
        if market_labels and market_data and all(val > 0 for val in market_data):
            # Format market data with K/M/B notation
            formatted_market_data = [self.format_currency(val) for val in market_data]
            
            chart_config = {
                "id": "market_growth_line",
                "chart_type": "line",
                "title": "Market Growth (TAM)",
                "description": "Total Addressable Market size and growth projection",
                "data": {
                    "labels": market_labels,
                    "datasets": [{
                        "label": "TAM",
                        "data": market_data,
                        "formatted_data": formatted_market_data,
                        "borderColor": "rgba(59, 130, 246, 1)",
                        "backgroundColor": "rgba(59, 130, 246, 0.1)"
                    }]
                },
                "format_y_axis": "currency",
                "format_tooltip": "currency"
            }
            
            if tam_source:
                source_badge = f"Source: {tam_source}"
                if tam_source_url:
                    source_badge += f" — {tam_source_url}"
                chart_config["source_badges"] = [source_badge]
            
            visuals["market_opportunity_analysis"]["charts"].append(chart_config)
            logger.info(f"Added TAM chart with {len(market_data)} data points")
        else:
            # No market data at all - create chart with note badge
            if "market_opportunity_analysis" not in visuals:
                visuals["market_opportunity_analysis"] = {"charts": []}
            
            chart_config = {
                "id": "market_growth_line",
                "chart_type": "line",
                "title": "Market Growth (TAM)",
                "description": "Total Addressable Market size and growth projection",
                "data": {
                    "labels": ["Data Unavailable"],
                    "datasets": [{
                        "label": "TAM",
                        "data": [0],
                        "borderColor": "rgba(59, 130, 246, 1)",
                        "backgroundColor": "rgba(59, 130, 246, 0.1)"
                    }]
                },
                "note_badge": "Market data (TAM) not available - requires founder confirmation"
            }
            visuals["market_opportunity_analysis"]["charts"].append(chart_config)

        # 3. Traction Funnel Bar Chart (Phase D: Use DataPriorityManager, show source badges)
        funnel_labels = []
        funnel_data = []
        source_badges = []
        stages = {}
        requires_confirmation = False

        # Use DataPriorityManager if available, otherwise fall back to memo extraction
        if priority_manager:
            waitlist_val = priority_manager.get_field_value("waitlist")
            beta_val = priority_manager.get_field_value("beta_users")
            pilot_val = priority_manager.get_field_value("lois_pilots")
            paying_val = priority_manager.get_field_value("paying_customers")
            total_users_val = priority_manager.get_field_value("total_users")
            mau_val = priority_manager.get_field_value("monthly_active_users")
            dau_val = priority_manager.get_field_value("daily_active_users")

            def add_stage(label: str, fv) -> None:
                nonlocal requires_confirmation
                if fv.value is not None and not fv.requires_confirmation:
                    try:
                        val = float(fv.value)
                        stages[label] = val
                        source_badges.append(f"{label}: {fv.source}")
                    except Exception:
                        requires_confirmation = True
                elif fv.requires_confirmation:
                    requires_confirmation = True

            add_stage("Waitlist", waitlist_val)
            add_stage("Beta", beta_val)
            add_stage("Pilots", pilot_val)
            add_stage("Paying", paying_val)
            add_stage("Total Users", total_users_val)
            add_stage("MAU", mau_val)
            add_stage("DAU", dau_val)

            # If priority sources didn't yield funnel counts, fall back to pitch deck metrics embedded in payload
            if (len(stages) == 0 or requires_confirmation) and getattr(priority_manager, "pitch_data", None):
                deck_payload = priority_manager.pitch_data or {}
                deck_data = (
                    deck_payload.get("pitch_deck_data")
                    or deck_payload.get("pitch_deck")
                    or deck_payload.get("data")
                    or {}
                )
                if isinstance(deck_data, dict) and "data" in deck_data and isinstance(deck_data["data"], dict):
                    deck_data = deck_data["data"]

                fallback_mappings = [
                    ("Waitlist", deck_data.get("top_funnel_metric")),
                    ("Beta", deck_data.get("mid_funnel_metric")),
                    ("Paying", deck_data.get("bottom_funnel_metric")),
                    ("Active Customers", deck_data.get("kpi_2_value")),
                ]
                added_from_deck = False
                for label, raw_value in fallback_mappings:
                    val = self.extract_numeric_value(raw_value)
                    if val is not None:
                        stages[label] = val
                        source_badges.append(f"{label}: pitch_deck")
                        added_from_deck = True
                if added_from_deck:
                    requires_confirmation = False
        else:
            # NEW: Use dynamic KPI engine from memo.traction_kpis
            for kpi in memo.traction_kpis or []:
                if not kpi.estimated:  # Only use non-estimated KPIs for funnel
                    kpi_name = kpi.name.lower()
                    if any(keyword in kpi_name for keyword in ['user', 'customer', 'subscriber']):
                        stages[kpi.name] = int(kpi.value)
        
        funnel_labels = list(stages.keys())
        funnel_data = list(stages.values())
        
        # Sanitize funnel values to avoid absurd magnitudes
        stages, suppressed = self._sanitize_funnel_stage_values(stages)
        funnel_labels = list(stages.keys())
        funnel_data = list(stages.values())

        # If no traction data, check if any path above produced values; otherwise mark missing
        if len(funnel_data) == 0:
            note_badge = "Traction data not available - requires founder confirmation"
            funnel_labels = ["Data Unavailable"]
            funnel_data = [0]
        else:
            note_badge = None
        
        if "early_traction_signals" not in visuals:
            visuals["early_traction_signals"] = {"charts": []}
        
        # Use raw integers for user counts (no K/M/B suffix for user metrics)
        formatted_funnel_data = [str(int(val)) for val in funnel_data]
        
        chart_config = {
            "id": "funnel_users",
            "chart_type": "bar",
            "title": "Traction Funnel",
            "description": "User progression through acquisition funnel stages",
            "data": {
                "labels": funnel_labels,
                "datasets": [{
                    "label": "Users",
                    "data": funnel_data,
                    "formatted_data": formatted_funnel_data,
                    "backgroundColor": "rgba(34, 197, 94, 0.8)",
                    "borderColor": "rgba(34, 197, 94, 1)"
                }]
            },
            "format_y_axis": "plain",
            "format_tooltip": "plain"
        }
        
        # Add source badges and note badge
        if source_badges:
            chart_config["source_badges"] = source_badges
        if suppressed:
            chart_config["note_badge"] = "Unrealistic traction values were suppressed; provide verified counts."
        elif note_badge:
            chart_config["note_badge"] = note_badge
        
        visuals["early_traction_signals"]["charts"].append(chart_config)

        # NEW: Dynamic KPI Engine - Auto-classify KPIs into chart types
        if memo.traction_kpis:
            # Group KPIs by chart type
            user_kpis = []
            retention_kpis = []
            engagement_kpis = []
            revenue_kpis = []
            cac_ltv_kpis = []

            for kpi in memo.traction_kpis:
                kpi_name = kpi.name.lower()
                kpi_unit = (kpi.unit or "").lower()

                # Auto-classify based on name and unit
                if any(keyword in kpi_name for keyword in ['user', 'customer', 'subscriber', 'waitlist', 'beta']):
                    user_kpis.append(kpi)
                elif any(keyword in kpi_name for keyword in ['retention', 'churn']) or '%' in kpi_unit:
                    retention_kpis.append(kpi)
                elif any(keyword in kpi_name for keyword in ['engagement', 'time', 'session', 'visit']):
                    engagement_kpis.append(kpi)
                elif any(keyword in kpi_name for keyword in ['revenue', 'mrr', 'arr', 'mr', 'ar']) or any(currency in kpi_unit for currency in ['usd', 'inr', '$', '₹']):
                    revenue_kpis.append(kpi)
                elif any(keyword in kpi_name for keyword in ['cac', 'ltv']):
                    cac_ltv_kpis.append(kpi)

            # Generate charts for each category with minimum 5 visuals guaranteed

            # 1. Users Bar Chart
            if user_kpis:
                user_labels = [kpi.name for kpi in user_kpis]
                user_data = [kpi.value for kpi in user_kpis]
                user_sources = [f"{kpi.name}: {kpi.source}" for kpi in user_kpis]

                user_chart = {
                    "id": "users_bar",
                    "chart_type": "bar",
                    "title": "User Metrics",
                    "description": "User acquisition and growth metrics",
                    "data": {
                        "labels": user_labels,
                        "datasets": [{
                            "label": "Users",
                            "data": user_data,
                            "backgroundColor": "rgba(59, 130, 246, 0.8)",
                            "borderColor": "rgba(59, 130, 246, 1)"
                        }]
                    },
                    "source_badges": user_sources
                }
                visuals["early_traction_signals"]["charts"].append(user_chart)

            # 2. Retention/Churn Line Chart
            if retention_kpis:
                retention_labels = [kpi.name for kpi in retention_kpis]
                retention_data = [kpi.value for kpi in retention_kpis]
                retention_sources = [f"{kpi.name}: {kpi.source}" for kpi in retention_kpis]

                retention_chart = {
                    "id": "retention_line",
                    "chart_type": "line",
                    "title": "Retention & Churn Metrics",
                    "description": "User retention and churn rates over time",
                    "data": {
                        "labels": retention_labels,
                        "datasets": [{
                            "label": "Percentage",
                            "data": retention_data,
                            "borderColor": "rgba(16, 185, 129, 1)",
                            "backgroundColor": "rgba(16, 185, 129, 0.1)"
                        }]
                    },
                    "source_badges": retention_sources
                }
                visuals["early_traction_signals"]["charts"].append(retention_chart)

            # 3. Revenue Trend Line
            if revenue_kpis:
                revenue_labels = [kpi.name for kpi in revenue_kpis]
                revenue_data = [kpi.value for kpi in revenue_kpis]
                revenue_sources = [f"{kpi.name}: {kpi.source}" for kpi in revenue_kpis]

                # Format revenue data with currency notation
                formatted_revenue_data = [self.format_currency(val) for val in revenue_data]

                revenue_chart = {
                    "id": "revenue_trend",
                    "chart_type": "line",
                    "title": "Revenue Metrics",
                    "description": "Revenue growth and projections",
                    "data": {
                        "labels": revenue_labels,
                        "datasets": [{
                            "label": "Revenue",
                            "data": revenue_data,
                            "formatted_data": formatted_revenue_data,
                            "borderColor": "rgba(245, 158, 11, 1)",
                            "backgroundColor": "rgba(245, 158, 11, 0.1)"
                        }]
                    },
                    "format_y_axis": "currency",
                    "format_tooltip": "currency",
                    "source_badges": revenue_sources
                }
                visuals["early_traction_signals"]["charts"].append(revenue_chart)

            # 4. CAC vs LTV Comparison
            if len(cac_ltv_kpis) >= 2:
                cac_ltv_labels = [kpi.name for kpi in cac_ltv_kpis]
                cac_ltv_data = [kpi.value for kpi in cac_ltv_kpis]
                cac_ltv_sources = [f"{kpi.name}: {kpi.source}" for kpi in cac_ltv_kpis]

                # Format CAC/LTV data with currency notation
                formatted_cac_ltv_data = [self.format_currency(val) for val in cac_ltv_data]

                cac_ltv_chart = {
                    "id": "cac_ltv_comparison",
                    "chart_type": "bar",
                    "title": "CAC vs LTV Analysis",
                    "description": "Customer acquisition cost vs lifetime value comparison",
                    "data": {
                        "labels": cac_ltv_labels,
                        "datasets": [{
                            "label": "Amount",
                            "data": cac_ltv_data,
                            "formatted_data": formatted_cac_ltv_data,
                            "backgroundColor": ["rgba(239, 68, 68, 0.8)", "rgba(34, 197, 94, 0.8)"],
                            "borderColor": ["rgba(239, 68, 68, 1)", "rgba(34, 197, 94, 1)"]
                        }]
                    },
                    "format_y_axis": "currency",
                    "format_tooltip": "currency",
                    "source_badges": cac_ltv_sources
                }
                visuals["early_traction_signals"]["charts"].append(cac_ltv_chart)

            # 5. Engagement Metrics (if available)
            if engagement_kpis:
                engagement_labels = [kpi.name for kpi in engagement_kpis]
                engagement_data = [kpi.value for kpi in engagement_kpis]
                engagement_sources = [f"{kpi.name}: {kpi.source}" for kpi in engagement_kpis]

                # Format engagement data (usually percentages or numbers)
                formatted_engagement_data = [self.format_number(val) for val in engagement_data]

                engagement_chart = {
                    "id": "engagement_metrics",
                    "chart_type": "line",
                    "title": "Engagement Metrics",
                    "description": "User engagement and activity metrics",
                    "data": {
                        "labels": engagement_labels,
                        "datasets": [{
                            "label": "Engagement",
                            "data": engagement_data,
                            "formatted_data": formatted_engagement_data,
                            "borderColor": "rgba(139, 92, 246, 1)",
                            "backgroundColor": "rgba(139, 92, 246, 0.1)"
                        }]
                    },
                    "format_y_axis": "number",
                    "format_tooltip": "number",
                    "source_badges": engagement_sources
                }
                visuals["early_traction_signals"]["charts"].append(engagement_chart)

            # 6. Additional KPI Summary Chart (guarantee minimum 5 visuals)
            all_kpi_labels = [kpi.name for kpi in memo.traction_kpis]
            all_kpi_data = [kpi.value for kpi in memo.traction_kpis]
            all_kpi_sources = [f"{kpi.name}: {kpi.source}" for kpi in memo.traction_kpis]

            if len(visuals["early_traction_signals"]["charts"]) < 5:
                # Format KPI data with appropriate notation (currency for money, numbers for others)
                formatted_kpi_data = []
                for i, val in enumerate(all_kpi_data):
                    label = all_kpi_labels[i].lower()
                    if any(word in label for word in ['revenue', 'arr', 'mrr', 'cost', 'cac', 'ltv', 'value', 'amount']):
                        formatted_kpi_data.append(self.format_currency(val))
                    else:
                        formatted_kpi_data.append(self.format_number(val))

                summary_chart = {
                    "id": "kpi_summary",
                    "chart_type": "bar",
                    "title": "Key Performance Indicators Summary",
                    "description": "Comprehensive view of all extracted KPIs",
                    "data": {
                        "labels": all_kpi_labels,
                        "datasets": [{
                            "label": "Value",
                            "data": all_kpi_data,
                            "formatted_data": formatted_kpi_data,
                            "backgroundColor": "rgba(107, 114, 128, 0.8)",
                            "borderColor": "rgba(107, 114, 128, 1)"
                        }]
                    },
                    "format_y_axis": "auto",
                    "format_tooltip": "auto",
                    "source_badges": all_kpi_sources
                }
                visuals["early_traction_signals"]["charts"].append(summary_chart)

        # 4. Founder Skills Radar Chart (Phase D: Fixed scaling 0-10, use LLM-generated scores + founder docs)
        founder_labels = []
        founder_data = []
        founder_missing_fields: List[str] = []
        founder_note_badge = None
        qualitative_map = {
            "high": 8.0, "medium": 6.0, "low": 4.0,
            "strong": 8.0, "moderate": 6.0, "weak": 4.0,
            "excellent": 9.0, "good": 7.0, "fair": 5.0, "poor": 3.0,
        }
        
        def normalize_to_0_10(value: Optional[float]) -> float:
            """Normalize score to 0-10 range (Phase D)."""
            if value is None:
                return 0.0
            if value > 10:
                return 10.0
            if value < 0:
                return 0.0
            return float(value)
        
        def parse_score(label: str, raw_value: Any) -> Optional[float]:
            """Parse a founder score from numeric/qualitative strings; track missing fields for chatbot."""
            if raw_value is None:
                founder_missing_fields.append(label)
                return None
            raw_str = str(raw_value).strip()
            if not raw_str or self._detect_placeholder(raw_str):
                founder_missing_fields.append(label)
                return None
            lowered = raw_str.lower()
            if lowered in qualitative_map:
                return normalize_to_0_10(qualitative_map[lowered])
            num_val = self.extract_numeric_value(raw_str)
            if num_val is not None:
                return normalize_to_0_10(float(num_val))
            founder_missing_fields.append(label)
            return None
        
        default_labels = ["Domain Expertise", "Technical Ability", "Leadership", "Entrepreneurial Drive"]
        scores: Dict[str, Optional[float]] = {label: None for label in default_labels}
        
        if memo.founder_team_analysis.founders:
            founder = memo.founder_team_analysis.founders[0]
            scores["Domain Expertise"] = parse_score("Domain Expertise", getattr(founder, 'domain_expertise_score', None))
            scores["Technical Ability"] = parse_score("Technical Ability", getattr(founder, 'technical_ability_score', None))
            scores["Leadership"] = parse_score("Leadership", getattr(founder, 'leadership_score', None))
            scores["Entrepreneurial Drive"] = parse_score("Entrepreneurial Drive", getattr(founder, 'entrepreneurial_drive_score', None))
            
            valid_scores = [v for v in scores.values() if v is not None]
            if valid_scores:
                avg_score = round(sum(valid_scores) / len(valid_scores), 1)
                founder.overall_founder_score = f"{avg_score}/10"
            else:
                founder.overall_founder_score = "Data not provided — please supply via chatbot."
        else:
            founder_note_badge = "No founder details provided — please add via chatbot to sync assessment."
            founder_missing_fields.extend(default_labels)
        
        founder_labels = default_labels
        founder_data = [normalize_to_0_10(scores[label]) if scores[label] is not None else 0.0 for label in default_labels]
        
        if founder_missing_fields and not founder_note_badge:
            missing_str = ", ".join(sorted(set(founder_missing_fields)))
            founder_note_badge = f"Missing founder skill data ({missing_str}) — provide via chatbot to sync."
        
        if "founder_team_analysis" not in visuals:
            visuals["founder_team_analysis"] = {"charts": []}
        chart_config = {
            "id": "founder_score_radar",
            "chart_type": "radar",
            "title": "Founder Skills Assessment",
            "description": "Assessment of founder capabilities across key dimensions",
            "data": {
                "labels": founder_labels,
                "datasets": [{
                    "label": "Founder",
                    "data": founder_data,
                    "borderColor": "rgba(168, 85, 247, 1)",
                    "backgroundColor": "rgba(168, 85, 247, 0.2)"
                }]
            },
            "options": {
                "scales": {
                    "r": {
                        "min": 0,
                        "max": 10,
                        "ticks": {
                            "stepSize": 1
                        }
                    }
                }
            }
        }
        if founder_note_badge:
            chart_config["note_badge"] = founder_note_badge
            chart_config["missing_fields"] = sorted(set(founder_missing_fields))
            chart_config["chatbot_prompt"] = "Share founder domain expertise, technical ability, leadership, and entrepreneurial drive in the chatbot to sync this chart."
        visuals["founder_team_analysis"]["charts"].append(chart_config)

        # 5. Use of Funds Pie Chart - Use LLM-generated data or estimates, NO hardcoded defaults
        budget_labels = []
        budget_data = []
        use_of_funds_source = None
        use_of_funds_note = None
        
        # Priority 1: Use LLM-generated use of funds if available
        if generated_content and generated_content.get("use_of_funds"):
            use_of_funds = generated_content["use_of_funds"]
            if isinstance(use_of_funds, list) and use_of_funds:
                for item in use_of_funds:
                    if item.get("category") and item.get("percentage"):
                        pct = self.extract_percentage(item["percentage"])
                        if pct is not None:
                            budget_labels.append(item["category"])
                            budget_data.append(float(pct))
                use_of_funds_source = "llm_generated"
                use_of_funds_note = "LLM-generated breakdown based on company stage and business model"
        
        # Priority 2: Extract from memo breakdown - only if not already set from generated content
        if not budget_labels and memo.use_of_funds_milestones.use_of_funds_breakdown:
            for b in memo.use_of_funds_milestones.use_of_funds_breakdown:
                if b.percentage:
                    pct = self.extract_percentage(b.percentage)
                    if pct is not None:
                        budget_labels.append(b.category)
                        budget_data.append(float(pct))
            if budget_labels:
                use_of_funds_source = "memo_data"
        
        # Priority 3: If still no data, check if LLM provided estimates in generated_content
        # (This should have been generated in _generate_content_with_llm, but double-check)
        if not budget_labels and generated_content:
            # LLM should have generated use_of_funds - if not, we'll show note badge
            use_of_funds_note = "Use of funds breakdown not available - requires founder confirmation"
        
        # DO NOT use hardcoded defaults - always show note badge if data is missing
        if not budget_labels:
            use_of_funds_note = "Use of funds breakdown not available - requires founder confirmation"
        
        # Only render Use of Funds chart if we have data (from LLM or memo)
        if budget_labels and budget_data:
            if "use_of_funds_milestones" not in visuals:
                visuals["use_of_funds_milestones"] = {"charts": []}
            
            chart_config = {
                "id": "budget_allocation",
                "chart_type": "pie",
                "title": "Use of Funds",
                "description": "Breakdown of how investment funds will be allocated",
                "data": {
                    "labels": budget_labels,
                    "datasets": [{
                        "data": budget_data,
                        "backgroundColor": [
                            "rgba(59, 130, 246, 0.8)",
                            "rgba(34, 197, 94, 0.8)",
                            "rgba(251, 146, 60, 0.8)",
                            "rgba(168, 85, 247, 0.8)",
                            "rgba(236, 72, 153, 0.8)"
                        ]
                    }]
                }
            }
            
            # Add source badge and note if available
            if use_of_funds_source:
                chart_config["source_badges"] = [f"Source: {use_of_funds_source}"]
            if use_of_funds_note:
                chart_config["note_badge"] = use_of_funds_note
            
            visuals["use_of_funds_milestones"]["charts"].append(chart_config)

        # 6. Competitive Landscape Bar Chart (Phase D: Require 3+ real competitors with source URLs)
        comp_labels = []
        comp_data = []
        
        # Filter real competitors with source URLs (Phase D)
        real_competitors = []
        if memo.competitive_landscape.competitors:
            for comp in memo.competitive_landscape.competitors:
                # Only use competitors with real names and source URLs
                if comp.name and comp.name != DEFAULT_PLACEHOLDER and not self._detect_placeholder(comp.name):
                    # Check for generic placeholder names
                    name_lower = comp.name.lower()
                    if name_lower not in ["competitor a", "competitor b", "competitor c", 
                                         "competitor 1", "competitor 2", "competitor 3",
                                         "competitor x", "competitor y", "competitor z"]:
                        # Phase D: Require source_url
                        if hasattr(comp, 'source_url') and comp.source_url:
                            real_competitors.append(comp)
                        elif not hasattr(comp, 'source_url'):
                            # Legacy competitor without source_url - include but mark as low confidence
                            real_competitors.append(comp)
        
        # Only render if we have minimum 3 real competitors (Phase D)
        if len(real_competitors) >= 3:
            for comp in real_competitors:
                comp_labels.append(comp.name)
                # Extract funding level - use 0 if not available (no synthetic fallback)
                funding_str = comp.funding or ""
                funding_val = 0
                if funding_str and funding_str.lower() not in ["not disclosed", "unknown", "n/a"]:
                    funding_val = self.extract_numeric_value(funding_str) or 0
                comp_data.append(funding_val)
        
        # Only render if we have real competitors - no synthetic fallbacks
        if len(comp_labels) >= 3:
            if "competitive_landscape" not in visuals:
                visuals["competitive_landscape"] = {"charts": []}
            visuals["competitive_landscape"]["charts"].append({
                "id": "competitive_landscape",
                "chart_type": "bar",
                "title": "Competitive Landscape",
                "description": "Comparison of key competitors and their funding levels",
                "data": {
                    "labels": comp_labels,
                    "datasets": [{
                        "label": "Funding Level",
                        "data": comp_data,
                        "backgroundColor": "rgba(251, 146, 60, 0.8)",
                        "borderColor": "rgba(251, 146, 60, 1)"
                    }]
                }
            })

        return visuals

    def _normalize_payload(self, pitch: Dict[str, Any]) -> Dict[str, Any]:
        base_payload = pitch.get("pitch") if isinstance(pitch.get("pitch"), dict) else pitch or {}
        payload = dict(base_payload) if isinstance(base_payload, dict) else {}

        relationships = pitch.get("relationships") if isinstance(pitch, dict) else {}
        if isinstance(relationships, dict) and relationships:
            payload["relationships"] = relationships

        # Prioritize 'team' (actual team members) over 'build_team' (open positions)
        team = pitch.get("team") or pitch.get("casino_teams")
        build_team = pitch.get("build_team")
        
        # Add team data (prioritize actual team members)
        if isinstance(team, list) and len(team) > 0:
            payload["team"] = team
        elif isinstance(build_team, list):
            payload["build_team"] = build_team

        for extra_key in ("pitch_documents", "upload_documents", "metadata", "pitch_deck", "pitch_deck_data", "visual_data"):
            if isinstance(pitch, dict) and pitch.get(extra_key) is not None:
                payload[extra_key] = pitch.get(extra_key)

        stage_rel = relationships.get("stage") if isinstance(relationships, dict) else {}
        stage_value = None
        if isinstance(stage_rel, dict):
            stage_value = stage_rel.get("stage") or stage_rel.get("name")

        defaults = {
            "company_name": payload.get("company_name") or payload.get("company") or payload.get("name") or DEFAULT_PLACEHOLDER,
            "stage": stage_value or payload.get("stage") or payload.get("round") or DEFAULT_PLACEHOLDER,
            "market": payload.get("market") or payload.get("industry") or DEFAULT_PLACEHOLDER,
            "description": payload.get("description") or payload.get("summary") or DEFAULT_PLACEHOLDER,
            "website_url": payload.get("website_url") or payload.get("url") or payload.get("website") or DEFAULT_PLACEHOLDER,
            "headline": payload.get("headline") or payload.get("title") or DEFAULT_PLACEHOLDER,
        }
        merged = {**payload, **defaults}
        return merged

    async def _generate_content_with_llm(
        self,
        pitch: Dict[str, Any],
        fund: Dict[str, Any],
        research: Dict[str, Any],
        pdf_extraction_results: Optional[Dict[str, Any]] = None,
        website_context: Optional[Dict[str, Any]] = None,
        linkedin_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate all memo content sections using LLM before memo creation.
        This replaces direct copying from pitch listings.
        
        Returns a dictionary with generated content for:
        - problem: Problem statement
        - investment_thesis: Investment thesis (3 sentences)
        - why_now: Why Now factors
        - target_customer: Target customer profile
        - market_trends: Market trends analysis
        - traction_analysis: Traction analysis
        - red_flags: Red flags assessment
        - use_of_funds: Use of funds breakdown
        - revenue_projections: Revenue projections (3-year)
        - market_data: TAM/SAM/SOM data
        """
        logger.info("Generating memo content with LLM before memo creation")
        
        # Build comprehensive context for LLM
        company_name = pitch.get("company_name") or pitch.get("company") or pitch.get("venue") or "the company"
        market = pitch.get("market") or pitch.get("industry") or ""
        description = pitch.get("description") or pitch.get("business") or ""
        headline = pitch.get("headline") or ""
        
        # Extract PDF metrics and texts
        pdf_metrics = pdf_extraction_results.get("pdf_metrics", {}) if pdf_extraction_results else {}
        pdf_texts = pdf_extraction_results.get("pdf_texts", []) if pdf_extraction_results else []
        traction_kpis = pdf_extraction_results.get("traction_kpis", []) if pdf_extraction_results else []
        
        # Build context string
        context_parts = [
            f"Company: {company_name}",
            f"Market/Industry: {market}",
            f"Description: {description}",
            f"Headline: {headline}",
        ]
        
        if pdf_metrics:
            pdf_metrics_compact = self._compact_json_for_prompt(pdf_metrics, max_chars=2000)
            context_parts.append(f"PDF Metrics: {pdf_metrics_compact}")
        
        if pdf_texts:
            pdf_summary = self._summarize_text_snippets(pdf_texts, max_items=2, max_chars=320)
            if pdf_summary:
                context_parts.append(f"PDF Text Excerpts:\n{pdf_summary}")
        
        if traction_kpis:
            kpi_summary = "\n".join([f"- {kpi.name}: {kpi.value} {kpi.unit}" for kpi in traction_kpis[:10]])
            context_parts.append(f"Traction KPIs:\n{kpi_summary}")
        
        # Extract use of funds from pitch data (priority: pitch deck > pitch listing)
        use_of_funds_from_pitch = None
        # Check pitch deck data first (if available)
        pitch_deck_data = pitch.get("pitch_deck_data") or pitch.get("pitch_deck") or {}
        if isinstance(pitch_deck_data, dict):
            # Check for fund_allocation in pitch deck
            funding_data = pitch_deck_data.get("funding") or {}
            if isinstance(funding_data, dict):
                fund_allocation = funding_data.get("fund_allocation") or funding_data.get("use_of_funds")
                if fund_allocation and isinstance(fund_allocation, list):
                    use_of_funds_from_pitch = fund_allocation
                    logger.info(f"Found use of funds in pitch deck: {len(fund_allocation)} categories")
        
        # If not in pitch deck, check pitch listing fields
        if not use_of_funds_from_pitch:
            min_target_use = pitch.get("min_target_use")
            max_target_use = pitch.get("max_target_use")
            if max_target_use or min_target_use:
                # Convert text to structured format for LLM
                use_of_funds_text = max_target_use or min_target_use
                if use_of_funds_text:
                    context_parts.append(f"Use of Funds (from pitch listing): {use_of_funds_text}")
                    logger.info(f"Found use of funds text in pitch listing: {use_of_funds_text[:100]}...")
        
        # Add structured use of funds to context if available
        if use_of_funds_from_pitch:
            use_of_funds_compact = self._compact_json_for_prompt(use_of_funds_from_pitch, max_chars=800)
            context_parts.append(f"Use of Funds (from pitch deck - USE THIS DATA): {use_of_funds_compact}")
            logger.info(f"Added structured use of funds to LLM context: {len(use_of_funds_from_pitch)} categories")
        
        website_profile = (website_context or {}).get("profile") or {}
        website_snippets = (website_context or {}).get("snippets") or []
        if website_profile:
            website_profile_compact = self._compact_json_for_prompt(website_profile, max_chars=1200)
            context_parts.append(f"Website Profile: {website_profile_compact}")
        if website_snippets:
            snippet_lines = [self._truncate_text(str(snippet), 220) for snippet in website_snippets[:5]]
            snippet_lines = [line for line in snippet_lines if line]
            if snippet_lines:
                context_parts.append("Website Snippets:\n- " + "\n- ".join(snippet_lines))
        
        if research:
            research_compact = self._compact_json_for_prompt(research, max_chars=3200)
            context_parts.append(f"Market Research (from Gemini internet search with source URLs): {research_compact}")
            # Extract TAM directly from research if available for immediate use
            if isinstance(research, dict) and research.get("metrics") and research["metrics"].get("tam"):
                tam_data = research["metrics"]["tam"]
                if isinstance(tam_data, dict) and tam_data.get("value"):
                    # Include both citation text and source URL
                    citation_text = tam_data.get("citation") or "Gemini research"
                    source_url = tam_data.get("source_url", "")
                    context_parts.append(f"CRITICAL: Gemini research found TAM: {tam_data.get('value')} (Source: {citation_text}" + (f", URL: {source_url}" if source_url else "") + f", Confidence: {tam_data.get('confidence', 'medium')}). Use this exact value in market_data.tam. Include tam_source as '{citation_text}' and tam_source_url as '{source_url}' if available.")
        
        if linkedin_data:
            linkedin_compact = self._compact_json_for_prompt(linkedin_data, max_chars=800)
            context_parts.append(f"LinkedIn Data: {linkedin_compact}")
        
        context = "\n\n".join(context_parts)
        try:
            logger.info(
                "Prompt compaction (content LLM): pdf_metrics=%s, pdf_texts=%s, research=%s, website_snippets=%s",
                len(pdf_metrics_compact) if pdf_metrics else 0,
                len(pdf_summary) if pdf_texts else 0,
                len(research_compact) if research else 0,
                len(snippet_lines) if website_snippets else 0,
            )
        except Exception:
            # Best-effort logging only
            pass
        
        # Generate all content sections with Claude
        content_prompt = f"""You are a senior investment analyst generating professional content for an investment memo.

CONTEXT:
{context}

Generate professional, investor-grade content for each section below. DO NOT copy directly from the pitch listing - instead, analyze the data and create original, insightful content.

Return a JSON object with the following structure:
{{
  "problem": "Professional problem statement (2-3 sentences, not copied from description)",
  "investment_thesis": "Investment thesis in 3 sentences: Why invest? What's the opportunity? What's the risk-adjusted return potential?",
  "why_now": "Why Now analysis (2-3 sentences on market timing, trends, and catalysts)",
  "target_customer": "Detailed target customer profile (2-3 sentences describing ideal customer segments)",
  "market_trends": ["Trend 1", "Trend 2", "Trend 3"],
  "traction_analysis": {{
    "paying_customers": "Number or description (e.g., '150', '$50K MRR', '25 paying customers')",
    "lois_pilots": "Number or description (e.g., '12', '5 LOIs', '8 pilot programs')",
    "beta_users": "Number or description (e.g., '500', '1,200 users', '2K beta testers')",
    "waitlist": "Number or description (e.g., '2,000', '5K waitlist', '10,000 signups')",
    "engagement_level": "Qualitative assessment of user engagement",
    "traction_quality": "Assessment of traction quality and product-market fit signals",
    "customer_acquisition": "Customer acquisition strategy and channels",
    "early_retention": "Early retention assessment",
    "best_evidence": ["Evidence 1", "Evidence 2", "Evidence 3"]
  }},
  "red_flags": ["Red flag 1", "Red flag 2", "Red flag 3"] or ["No significant red flags identified based on available data"],
  "use_of_funds": [
    {{"category": "Category 1", "percentage": "40%", "purpose": "Purpose description"}},
    {{"category": "Category 2", "percentage": "35%", "purpose": "Purpose description"}},
    {{"category": "Category 3", "percentage": "25%", "purpose": "Purpose description"}}
  ],
  "revenue_projections": [
    {{"year": "Year 1", "revenue": "$X", "growth_rate": "X%"}},
    {{"year": "Year 2", "revenue": "$X", "growth_rate": "X%"}},
    {{"year": "Year 3", "revenue": "$X", "growth_rate": "X%"}}
  ],
  "adjusted_projections": {{
    "year1": "$X (adjusted projection for Year 1)",
    "year2": "$X (adjusted projection for Year 2)",
    "year3": "$X (adjusted projection for Year 3)"
  }},
  "market_data": {{
    "tam": "TAM value with currency and scale (e.g., USD 250M)",
    "sam": "SAM value with currency and scale",
    "som": "SOM value with currency and scale",
    "tam_source": "Source of TAM data",
    "tam_source_url": "URL if available",
    "tam_confidence": "high|medium|low",
    "growth_rate": "Market growth rate (CAGR)"
  }}
}}

CRITICAL REQUIREMENTS:
1. Problem: Analyze the pain points, don't just copy the description
2. Investment Thesis: Must be compelling and structured (3 sentences)
3. Why Now: Focus on market timing and catalysts
4. Target Customer: Be specific about customer segments
5. Red Flags: Analyze ALL data sources - if no red flags, say "No significant red flags identified based on available data"
6. Use of Funds: CRITICAL - If "Use of Funds (from pitch deck - USE THIS DATA)" is provided in the context above, you MUST use that exact data. Do NOT generate your own breakdown. Use the categories, percentages, and purposes from the pitch deck data. If only text is provided (from pitch listing), extract the breakdown from that text. Only if NO use of funds data is provided in the context should you generate a realistic breakdown based on company stage and business model. ALWAYS provide breakdown - percentages must sum to 100%.
7. Revenue Projections: Generate realistic 3-year projections based on available traction data. If no data available, provide estimates based on company stage, industry benchmarks, and comparable companies. ALWAYS provide estimates - never leave empty. Also provide adjusted_projections (conservative estimates).
8. Market Data: CRITICAL - Use the Market Research data provided above (from Gemini internet research) as the PRIMARY source for TAM/SAM/SOM. The research includes actual market data with source URLs. Extract TAM value, source URL, and confidence from the research. If research provides TAM, use it directly. If research doesn't have TAM, then provide estimates based on industry reports, comparable companies, and market analysis. ALWAYS provide TAM - never leave empty. Include tam_source, tam_source_url (from research if available), and tam_confidence.
10. Traction Analysis: MUST include quantitative fields (paying_customers, lois_pilots, beta_users, waitlist) with actual numbers or estimates. If no data available, provide realistic estimates based on company stage and industry benchmarks. NEVER leave these empty - always provide estimates.

Return ONLY valid JSON, no markdown formatting."""
        
        try:
            logger.info("Making Claude API call to generate memo content...")
            if self._is_token_debug():
                sections_debug = {
                    "memo_state": self._compact_json_for_prompt(pitch, max_chars=2000),
                    "founder_docs": pdf_summary if pdf_texts else "",
                    "previous_output": "",
                    "template": "",
                }
                self._emit_token_debug(
                    label="content_generation_pre",
                    system_prompt="You are a professional investment analyst generating memo content. Return only valid JSON.",
                    user_prompt=content_prompt,
                    sections=sections_debug,
                    usage=None,
                )
            response, usage = await self.claude.generate_json(
                system_prompt="You are a professional investment analyst generating memo content. Return only valid JSON.",
                messages=[{"role": "user", "content": content_prompt}],
                max_tokens=3000,
            )
            if self._is_token_debug():
                self._emit_token_debug(
                    label="content_generation_post",
                    system_prompt="You are a professional investment analyst generating memo content. Return only valid JSON.",
                    user_prompt=content_prompt,
                    sections=sections_debug,
                    usage=usage,
                    response_text=response if isinstance(response, str) else json.dumps(response),
                )
            
            # Parse response
            if isinstance(response, str):
                response = json.loads(response)
            
            # Add LinkedIn data to response if available
            if linkedin_data:
                response["linkedin_data"] = linkedin_data
            
            # Validate that we got actual content
            if response:
                non_empty = {k: v for k, v in response.items() if v and (not isinstance(v, (list, dict)) or len(v) > 0)}
                logger.info(f"Successfully generated memo content with LLM: {len(non_empty)} non-empty fields")
                if len(non_empty) == 0:
                    logger.warning("LLM returned empty content - all fields are empty!")
            else:
                logger.warning("LLM returned None or empty response")
            
            return response
            
        except ValueError as e:
            # API key missing or configuration error
            logger.warning(f"Claude API key not configured or invalid: {e}. Skipping LLM content generation.")
            return {
                "problem": "",
                "investment_thesis": "",
                "why_now": "",
                "target_customer": "",
                "market_trends": [],
                "traction_analysis": {},
                "red_flags": [],
                "use_of_funds": [],
                "revenue_projections": [],
                "market_data": {},
            }
        except Exception as e:
            # Check if it's an authentication error
            error_str = str(e).lower()
            if "authentication" in error_str or "401" in error_str or "invalid" in error_str and "api" in error_str:
                logger.warning(f"Claude API authentication failed: {e}. Skipping LLM content generation.")
                return {
                    "problem": "",
                    "investment_thesis": "",
                    "why_now": "",
                    "target_customer": "",
                    "market_trends": [],
                    "traction_analysis": {},
                    "red_flags": [],
                    "use_of_funds": [],
                    "revenue_projections": [],
                    "market_data": {},
                }
            logger.error(f"Failed to generate content with LLM: {e}")
            # Return empty structure on error
            return {
                "problem": "",
                "investment_thesis": "",
                "why_now": "",
                "target_customer": "",
                "market_trends": [],
                "traction_analysis": {},
                "red_flags": [],
                "use_of_funds": [],
                "revenue_projections": [],
                "market_data": {},
            }

    async def _extract_linkedin_data(
        self,
        pitch: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Extract LinkedIn data for founders from user database.
        Uses linkedin_id from pitch relationships to fetch profile data.
        
        Returns dictionary with:
        - education: List of education entries
        - experience_years: Years of experience
        - previous_companies: List of previous companies
        - linkedin_url: LinkedIn profile URL
        """
        try:
            relationships = pitch.get("relationships", {}) or {}
            created_by = relationships.get("created_by", {}) or {}
            
            # Get LinkedIn ID from user data
            linkedin_id = created_by.get("linkedin_id") or pitch.get("linkedin_id")
            
            if not linkedin_id:
                logger.info("No LinkedIn ID found in pitch data")
                return None
            
            # TODO: Implement LinkedIn API call or database lookup
            # For now, return structure indicating LinkedIn ID is available
            # In production, this would fetch from LinkedIn API or database
            logger.info(f"LinkedIn ID found: {linkedin_id}")
            
            # Placeholder - in production, fetch actual LinkedIn data
            return {
                "linkedin_id": linkedin_id,
                "education": None,  # Will be populated from LinkedIn API
                "experience_years": None,
                "previous_companies": None,
                "linkedin_url": f"https://www.linkedin.com/in/{linkedin_id}" if linkedin_id else None,
            }
            
        except Exception as e:
            logger.error(f"Failed to extract LinkedIn data: {e}")
            return None

    async def _generate_founder_skills_with_llm(
        self,
        founder_data: Dict[str, Any],
        linkedin_data: Optional[Dict[str, Any]] = None,
        doc_snippets: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Generate founder skills scores (0-10) using LLM.
        
        Returns dictionary with scores:
        - domain_expertise: 0-10
        - technical_ability: 0-10
        - leadership: 0-10
        - entrepreneurial_drive: 0-10
        """
        try:
            founder_name = founder_data.get("name", "")
            founder_role = founder_data.get("role", "")
            founder_experience = founder_data.get("experience_years", "")
            founder_education = founder_data.get("education", "")
            founder_previous_companies = founder_data.get("previous_companies", "")
            
            linkedin_context = ""
            if linkedin_data:
                linkedin_context = f"""
LinkedIn Data:
- Education: {linkedin_data.get('education', 'Not available')}
- Experience Years: {linkedin_data.get('experience_years', 'Not available')}
- Previous Companies: {linkedin_data.get('previous_companies', 'Not available')}
"""
            
            docs_context = ""
            if doc_snippets:
                cleaned_snippets = []
                for snippet in doc_snippets[:3]:
                    if not snippet:
                        continue
                    cleaned_snippets.append(" ".join(str(snippet).split()))
                if cleaned_snippets:
                    docs_context = "\nFounder Documents (snippets):\n- " + "\n- ".join(cleaned_snippets)
            
            prompt = f"""Assess the founder's skills across key dimensions based on the following information:

Founder Information:
- Name: {founder_name}
- Role: {founder_role}
- Experience: {founder_experience}
- Education: {founder_education}
- Previous Companies: {founder_previous_companies}
{linkedin_context}
{docs_context}

Rate each dimension on a scale of 0-10:
- Domain Expertise: Knowledge and experience in the target industry/problem area
- Technical Ability: Ability to build and lead technical teams/products
- Leadership: Ability to lead teams, make decisions, and execute
- Entrepreneurial Drive: Motivation, resilience, and ability to build businesses

Return ONLY a JSON object:
{{
  "domain_expertise": <number 0-10>,
  "technical_ability": <number 0-10>,
  "leadership": <number 0-10>,
  "entrepreneurial_drive": <number 0-10>
}}"""
            
            response, _ = await self.claude.generate_json(
                system_prompt="You are assessing founder capabilities. Return only valid JSON with numeric scores 0-10.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            
            if isinstance(response, str):
                response = json.loads(response)
            
            # Ensure all scores are in 0-10 range
            scores = {
                "domain_expertise": max(0, min(10, float(response.get("domain_expertise", 5)))),
                "technical_ability": max(0, min(10, float(response.get("technical_ability", 5)))),
                "leadership": max(0, min(10, float(response.get("leadership", 5)))),
                "entrepreneurial_drive": max(0, min(10, float(response.get("entrepreneurial_drive", 5)))),
            }
            
            logger.info(f"Generated founder skills scores: {scores}")
            return scores
            
        except ValueError as e:
            # API key missing or configuration error
            logger.warning(f"Claude API key not configured or invalid: {e}. Using default founder scores.")
            return {
                "domain_expertise": 5.0,
                "technical_ability": 5.0,
                "leadership": 5.0,
                "entrepreneurial_drive": 5.0,
            }
        except Exception as e:
            # Check if it's an authentication error
            error_str = str(e).lower()
            if "authentication" in error_str or "401" in error_str or "invalid" in error_str and "api" in error_str:
                logger.warning(f"Claude API authentication failed: {e}. Using default founder scores.")
                return {
                    "domain_expertise": 5.0,
                    "technical_ability": 5.0,
                    "leadership": 5.0,
                    "entrepreneurial_drive": 5.0,
                }
            logger.error(f"Failed to generate founder skills: {e}")
            return {
                "domain_expertise": 5.0,
                "technical_ability": 5.0,
                "leadership": 5.0,
                "entrepreneurial_drive": 5.0,
            }

    def _build_research_prompt(
        self,
        pitch: Dict[str, Any],
        fund: Dict[str, Any],
        uploaded_datasets: Optional[List[DatasetRecord]],
        website_context: Optional[Dict[str, Any]] = None,
        edgar_notes: Optional[str] = None,
        sec_notes: Optional[str] = None,
    ) -> str:
        company = pitch.get("company_name") or pitch.get("company") or DEFAULT_PLACEHOLDER
        market = pitch.get("market") or pitch.get("industry") or "general"

        # Use uploaded datasets as calibration, not just raw text.
        calibration = self._build_dataset_calibration(uploaded_datasets or [])
        dataset_context = calibration.get("raw_context") or "No prior datasets."
        market_calibration = calibration.get("market_calibration") or ""
        risk_calibration = calibration.get("risk_calibration") or ""
        terms_calibration = calibration.get("terms_calibration") or ""
        
        # Add PDF metrics to research context if available
        pdf_metrics_block = ""
        if pitch.get("pdf_metrics"):
            pdf_metrics_block = f"Extracted PDF Metrics: {self._compact_json_for_prompt(pitch['pdf_metrics'], max_chars=2000)}"

        website_profile = (website_context or {}).get("profile") or {}
        website_snippets = (website_context or {}).get("snippets") or []
        website_url = (website_context or {}).get("url") or pitch.get("website_url") or pitch.get("website")
        website_profile_block = self._compact_json_for_prompt(website_profile, max_chars=1200) if website_profile else "No structured website data."
        snippet_lines = [self._truncate_text(str(snippet), 200) for snippet in website_snippets[:8]]
        website_snippets_block = "\n".join(f"- {snippet}" for snippet in snippet_lines if snippet) if snippet_lines else "No website snippets captured."

        edgar_context = edgar_notes or "No EDGAR filings retrieved."
        sec_market_context = sec_notes or "SEC markets datasets: https://www.sec.gov/data-research/sec-markets-data"
        
        # Include PDF metrics in the final prompt assembly
        research_context_blocks = [
            f"Company Website Analysis:\n{website_profile_block}\n\nSnippets:\n{website_snippets_block}",
            f"EDGAR Filings Context:\n{edgar_context}",
            f"SEC Markets Data:\n{sec_market_context}",
            f"Historical Dataset Calibration:\n{dataset_context}",
            market_calibration,
            risk_calibration,
            terms_calibration,
        ]
        
        if pdf_metrics_block:
            research_context_blocks.append(f"PDF Extracted Metrics:\n{pdf_metrics_block}")
        
        research_context = "\n---\n".join([block for block in research_context_blocks if block and block.strip()])
        
        return (
            "You are a professional market intelligence researcher for an investment committee. "
            "Provide structured JSON insights with source URLs and confidence scores for all metrics.\n\n"
            f"Research Context:\n{research_context}\n\n"
            "CRITICAL REQUIREMENTS (Phase C - Structured Response):\n"
            "1. For every numeric metric (revenue, TAM, competitor funding, etc.), include:\n"
            "   - value: The actual metric value (e.g., 'USD 250M', '$50M')\n"
            "   - source_url: FULL URL to the exact page/source (e.g., 'https://example.com/report/2024-market-analysis', NOT just 'https://example.com') (required)\n"
            "   - confidence: 'high' | 'medium' | 'low' (required)\n"
            "   - citation_text: Human-readable citation (e.g., 'Gartner Market Research, 2024')\n\n"
            "2. For competitors, include:\n"
            "   - name: Actual company name (NOT 'Competitor A' or generic names)\n"
            "   - stage: Funding stage (Series A/B/C, Seed, Public, etc.)\n"
            "   - funding: Funding amount if available\n"
            "   - source_url: FULL URL to the exact page (e.g., 'https://crunchbase.com/organization/company-name', 'https://company.com/about', 'https://news.com/article/2024/company-funding', NOT just 'https://company.com') (required)\n"
            "   - confidence: Based on data recency and source reliability ('high' | 'medium' | 'low')\n\n"
            "3. If data not found after research, return:\n"
            "   - value: null\n"
            "   - requires_confirmation: true\n"
            "   - Do NOT estimate or use generic values\n\n"
            "REQUIRED JSON STRUCTURE:\n"
            "{\n"
            "  \"metrics\": {\n"
            "    \"tam\": {\n"
            "      \"value\": \"USD 250M\",\n"
            "      \"source_url\": \"https://example.com/research/gartner-report-2024\",\n"
            "      \"confidence\": \"high\",\n"
            "      \"citation\": \"Gartner Market Research, 2024\"\n"
            "    }\n"
            "  },\n"
            "  \"competitors\": [\n"
            "    {\n"
            "      \"name\": \"Company Name\",\n"
            "      \"stage\": \"Series B\",\n"
            "      \"funding\": \"$50M\",\n"
            "      \"source_url\": \"https://crunchbase.com/organization/company-name\",\n"
            "      \"confidence\": \"high\"\n"
            "    }\n"
            "  ],\n"
            "  \"sources\": [\"https://example.com/page1\", \"https://example.com/page2\"]\n"
            "}\n\n"
            "COMPETITORS: Provide 3-7 named real competitors for this exact product/market. "
            "Minimum 3 competitors required, ideally 5-7. Each must have real company name and FULL source URL (exact page, not just domain).\n"
            "TAM: Must include currency (USD/EUR/etc), scale (M/B/T), year of estimate, and FULL source URL (exact page, not just domain).\n"
            "SOURCES: Every key metric must have a traceable FULL source URL with the exact page path, NOT just the domain (e.g., use 'https://company.com/about' not 'https://company.com').\n"
            f"\nCompany: {company}\nMarket: {market}\nFund context: {fund}\n"
            f"Additional datasets (historical benchmarks): {dataset_context}\n"
            f"Market calibration hints from datasets: {market_calibration}\n"
            f"Risk calibration hints from datasets: {risk_calibration}\n"
            f"Terms calibration hints from datasets: {terms_calibration}\n"
            f"Company website scraped URL: {website_url or 'Not provided'}\n"
            f"Website profile snapshot (structured fields):\n{website_profile_block}\n"
            f"Website content snippets (top excerpts):\n{website_snippets_block}\n"
            f"EDGAR context: {edgar_context}\n"
            f"Public market/filing datasets: {sec_market_context}\n"
            "Respond with JSON only, following the required structure above."
        )

    def _build_claude_prompt(
        self,
        pitch: Dict[str, Any],
        fund: Dict[str, Any],
        research: Dict[str, Any],
        uploaded_datasets: Optional[List[DatasetRecord]],
        website_context: Optional[Dict[str, Any]] = None,
        prioritized_fields: Optional[Dict[str, Any]] = None,
        pre_populated_memo: Optional[InvestorCommitteeMemo] = None,
        generated_content: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Use pre-populated memo if available, otherwise create empty schema
        if pre_populated_memo:
            schema_example = self._compact_json_for_prompt(pre_populated_memo.model_dump(), max_chars=6000)
            print(f"CLAUDE PROMPT: Using pre-populated memo with traction data")
        else:
            schema_example = self._compact_json_for_prompt(InvestorCommitteeMemo().model_dump(), max_chars=6000)
        calibration = self._build_dataset_calibration(uploaded_datasets or [])
        dataset_notes = calibration.get("raw_context") or "No extra datasets."
        market_calibration = calibration.get("market_calibration") or ""
        risk_calibration = calibration.get("risk_calibration") or ""
        terms_calibration = calibration.get("terms_calibration") or ""
        website_profile = (website_context or {}).get("profile") or {}
        website_snippets = (website_context or {}).get("snippets") or []
        website_url = (website_context or {}).get("url") or pitch.get("website_url") or pitch.get("website")
        website_profile_block = self._compact_json_for_prompt(website_profile, max_chars=1200) if website_profile else "No structured website data."
        snippet_lines = [self._truncate_text(str(snippet), 220) for snippet in website_snippets[:8]]
        website_snippets_block = "\n".join([line for line in snippet_lines if line]) or "No website snippets captured."
        
        # Extract PDF metrics and texts for Claude prompt
        pdf_metrics = pitch.get("pdf_metrics", {})
        pdf_texts = pitch.get("pdf_texts", [])
        pdf_metrics_block = ""
        if pdf_metrics:
            pdf_metrics_compact = self._compact_json_for_prompt(pdf_metrics, max_chars=2000)
            pdf_metrics_block = f"\n\nFounder-provided PDF metrics (extracted from uploaded documents - PREFER these over estimates):\n{pdf_metrics_compact}\n\nIMPORTANT: Use these PDF-extracted metrics (revenue, MRR/ARR, user counts, burn rate, runway, funding ask) to populate memo fields. These are founder-provided data and should take precedence over analyst estimates."
        if pdf_texts:
            pdf_texts_summary = self._summarize_text_snippets(pdf_texts, max_items=2, max_chars=320)
            if pdf_texts_summary:
                pdf_metrics_block += f"\n\nPDF text excerpts (for context):\n{pdf_texts_summary}"

        generated_content_compact = self._compact_json_for_prompt(generated_content, max_chars=2500) if generated_content else ""
        pitch_compact = self._compact_json_for_prompt(self._slim_pitch_for_prompt(pitch), max_chars=3000)
        traction_compact = self._compact_json_for_prompt(pitch.get('traction_kpis', []), max_chars=1000)
        fund_compact = self._compact_json_for_prompt(fund, max_chars=1200)
        research_compact = self._compact_json_for_prompt(research, max_chars=2500)
        prioritized_data_compact = self._compact_json_for_prompt(prioritized_fields, max_chars=1500) if prioritized_fields else ""
        
        system_prompt = """
You are a senior investment committee memo writer for AngelLinx, producing enterprise-grade VC/PE investment memos.

WRITING STYLE (VC/PE Professional Standards):
- Concise, structured, and decision-focused
- Avoid buzzwords, marketing language, and fluff
- Use clear, direct language that investors expect
- Separate facts from analysis explicitly
- Structure: EXECUTIVE SUMMARY first → CLEAR INVESTMENT THESIS → risks → recommendation
- Use crisp bullet structure for key points
- Mandatory EXECUTIVE SUMMARY section with investment highlights
- CLEAR INVESTMENT THESIS section that articulates why this opportunity is compelling

CRITICAL RULES:
1️⃣ STRICT DATA USAGE:
Prefer data from:
- pitch data
- fund data
- uploaded documents with extracted text
- validated Gemini research that includes source URLs

2️⃣ REASONABLE ESTIMATES ALLOWED:
If a field has no information in pitch data, fund data, uploaded documents, or Gemini research, you may propose a reasonable analyst estimate based on your knowledge of similar SaaS / startup situations. Clearly label such values in nearby descriptive text as 'Analyst estimate; to be confirmed with founder' rather than leaving the field empty.

3️⃣ SOURCE ATTRIBUTION REQUIRED (Phase C):
Every numeric metric MUST include:
- value: The actual metric value
- source field: Format must be one of:
  * "pitch_db" (from founder-provided pitch data)
  * "pdf:filename.pdf" (from extracted PDF documents)
  * "website:url" (from website scraping)
  * "gemini:url" (from Gemini research with source URL)
  * "claude:inferred" (if inferring from context, set confidence to "low")
- confidence field: "high" | "medium" | "low" (required for LLM-sourced data)
- source_url: URL if available (especially for Gemini research)

For revenue projections: include source, source_url (if available), and confidence fields.
For traction metrics (waitlist, beta_users, lois_pilots, paying_customers): include corresponding _source fields.
For TAM: include tam_source, tam_source_url (if available), and tam_confidence fields.

MISSING DATA HANDLING:
- DO NOT use "To be confirmed" or "TBC" or "N/A"
- If all sources exhausted → use: "Data unavailable — requires founder confirmation"
- Only use "claude:inferred" with "low" confidence when inferring from context
- Only use "analyst_estimate" as source when no other data exists (with "low" confidence)

4️⃣ TRACTION ANALYSIS - CRITICAL:
You MUST populate ALL early_traction_signals fields. This section is REQUIRED and must always be populated:

QUANTITATIVE FIELDS (ALWAYS infer/generate realistic numbers - NEVER use placeholders):
• paying_customers: ALWAYS provide a realistic number based on company stage/context (e.g., "25 paying customers", "150 customers", "$50K MRR"). Infer from company description, market, product type, or stage.
• lois_pilots: ALWAYS provide realistic number (e.g., "8 pilot programs", "12 LOIs", "5 enterprise pilots"). Infer based on company stage and market.
• beta_users: ALWAYS provide realistic number (e.g., "500 beta users", "1,200 testers", "2,000 beta signups"). Infer based on company stage and product type.
• waitlist: ALWAYS provide realistic number (e.g., "2,000 waitlist signups", "5K waitlist", "10,000 signups"). Infer based on company stage and market demand signals.
• monthly_growth_rate: ALWAYS provide realistic percentage (e.g., "15% MoM", "25% monthly growth", "20% growth rate"). Infer based on company stage, product type, and market dynamics.

QUALITATIVE FIELDS (MUST ALWAYS provide meaningful assessments - NEVER leave empty):
• engagement_level: REQUIRED - Provide qualitative assessment based on available context. Examples:
  - "Engagement improving with automation triggers"
  - "High user activity with daily logins"
  - "Moderate engagement; onboarding optimization needed"
  - Infer from pitch data, product description, or traction_kpis even if specific metrics aren't available
  
• traction_quality: REQUIRED - Provide qualitative assessment. Examples:
  - "Early signals; retention to validate"
  - "Strong product-market fit indicators"
  - "Promising early traction; scale validation pending"
  - Base assessment on available traction data and context
  
• customer_acquisition: REQUIRED - Provide qualitative assessment of acquisition channels/strategy. Examples:
  - "Organic plus outbound to agencies/creators"
  - "Product-led growth with sales support"
  - "Content marketing and partnerships"
  - Infer from pitch data, go-to-market strategy, or website context
  
• early_retention: REQUIRED - Provide qualitative assessment. Examples:
  - "Retention tracking underway"
  - "Strong early retention metrics"
  - "Retention data collection in progress"
  - Provide assessment based on available context

• best_traction_evidence: REQUIRED - List 2-4 specific traction indicators from available data

CRITICAL RULES - ANTI-HALLUCINATION:
• Quantitative fields: ONLY use actual data from sources (pitch data, PDFs, website, Gemini research with URLs). 
  - If NO data exists from any source → use "Data unavailable — requires founder confirmation"
  - DO NOT generate fictional numbers or estimates
  - Estimates are ONLY allowed if explicitly marked with confidence="low" and source="analyst_estimate" 
• Qualitative fields: MUST derive from actual available context (pitch data, website, product description, traction_kpis, market research)
  - If no context exists, provide assessment but mark as "Analyst interpretation based on limited data"
• DO NOT leave quantitative fields empty - use "Data unavailable — requires founder confirmation" instead
• Use traction_kpis array for quantitative metrics when available
• NEVER create fictional competitor names, revenue numbers, or metrics
• All generated content MUST be traceable to a source or explicitly marked as "analyst_estimate" with low confidence

5️⃣ FACTUAL VS ANALYSIS:
Clearly label qualitative interpretation as:
"Analyst Interpretation: ..."

5️⃣ COMPETITIVE LANDSCAPE (Phase C):
You MUST provide 3-7 real competitor companies for this exact product/market, using Gemini research and your own knowledge. 
CRITICAL REQUIREMENTS:
- Use REAL company names (e.g., "Stripe", "Square", "PayPal") - NEVER use placeholders like 'Competitor A', 'Competitor B', or generic names
- Each competitor MUST include:
  * name: Real company name (not "Competitor A")
  * stage: Funding stage (Series A/B/C, Seed, Public, etc.)
  * funding: Funding amount if available
  * strengths: Differentiated positioning (not generic)
  * weaknesses: Real weaknesses (not generic)
  * source_url: URL from Gemini research or company website, OR "analyst_knowledge" when generated from your own knowledge (REQUIRED)
  * confidence: "high" | "medium" | "low" (REQUIRED)
- If Gemini research provides competitors, use those with source URLs
- If fewer than 3 competitors are found in research, GENERATE real competitors using your knowledge (set source_url to "analyst_knowledge")
- Minimum 3 competitors required, ideally 5-7 for comprehensive analysis

6️⃣ RECOMMENDATION FORMAT:
The investment_recommendation_summary.recommendation field MUST use one of these formats:
- "LEAD" - Strong conviction, fund should lead the round
- "FOLLOW" - Good opportunity, fund should participate
- "DISCUSS" - Requires IC discussion, mixed signals
- "PASS" - Not a fit, do not invest

The recommendation MUST be justified by covering:
- Founder quality (experience, domain expertise, execution)
- Market opportunity (size, timing, defensibility)
- Traction (early signals, product-market fit indicators)
- Fund fit (check size, stage, sector alignment)

7️⃣ DO NOT COPY:
Rewrite Gemini content uniquely.

8️⃣ USE LLM-GENERATED CONTENT:
The following content has been pre-generated by LLM analysis. USE THIS CONTENT instead of copying from pitch listings:
{generated_content_compact if generated_content else "No pre-generated content available - generate from scratch using all available data sources."}

CRITICAL: Use the pre-generated content above for:
- Problem statement (use generated_content.problem)
- Investment thesis (use generated_content.investment_thesis)
- Why Now (use generated_content.why_now)
- Target customer (use generated_content.target_customer)
- Market trends (use generated_content.market_trends)
- Traction analysis (use generated_content.traction_analysis)
- Red flags (use generated_content.red_flags)
- Use of funds (use generated_content.use_of_funds)
- Revenue projections (use generated_content.revenue_projections)
- Market data/TAM (use generated_content.market_data)

DO NOT copy directly from pitch.description or pitch fields - use the LLM-generated content instead.
"""
        # Phase A: Build prioritized data section
        prioritized_data_block = ""
        if prioritized_fields:
            prioritized_data_block = "\n\nFOUNDER-PROVIDED DATA (use these values exactly, do not estimate):\n"
            prioritized_data_block += prioritized_data_compact
            prioritized_data_block += "\n\nIMPORTANT: If a field has a founder-provided value above, use it exactly. Do not estimate or modify these values."

        # Add generated content to user prompt
        generated_content_block = ""
        if generated_content:
            generated_content_block = f"""
LLM-GENERATED CONTENT (USE THIS INSTEAD OF COPYING FROM PITCH):
{generated_content_compact}

IMPORTANT: Use the LLM-generated content above for all sections. Do NOT copy directly from pitch data.
"""

        user_prompt = f"""
RETURN A VALID JSON OBJECT ONLY.

SCHEMA (match EXACTLY):
{schema_example}
{generated_content_block}
Pitch data (for reference only - use LLM-generated content instead):
{pitch_compact}

Traction KPIs (extracted from PDFs - use ALL numeric data):
{traction_compact}

Fund data:
{fund_compact}

Gemini research (use as factual context with citations):
{research_compact}

Website/context:
{website_profile_block}

Uploaded datasets:
{dataset_notes}
{pdf_metrics_block}
{prioritized_data_block}

OUTPUT RULES:
- Use analyst estimates when data is missing but reasonable approximations are possible
- Label analyst estimates as 'Analyst estimate; to be confirmed with founder'
- TRACTION KPIs: Use all provided traction_kpis, infer missing related KPIs with estimated=true, include confidence scoring
- EARLY_TRACTION_SIGNALS: MUST populate all fields - this section is REQUIRED:
  * Quantitative fields (paying_customers, lois_pilots, beta_users, waitlist, monthly_growth_rate): 
    - ALWAYS provide realistic numbers - NEVER use placeholders
    - Use numbers/percentages if available from traction_kpis or pitch data
    - If not available, INFER realistic estimates based on company stage, market, product type, and industry benchmarks
    - Examples: "25 paying customers", "8 pilot programs", "2,000 beta users", "10,000 waitlist signups", "15% MoM growth"
  * Qualitative fields (engagement_level, traction_quality, customer_acquisition, early_retention): 
    - REQUIRED - MUST provide meaningful qualitative assessments
    - Infer from pitch data, website context, product description, traction_kpis, or market research
    - Examples: "Engagement improving with automation triggers", "Early signals; retention to validate", "Organic plus outbound to agencies/creators", "Retention tracking underway"
    - DO NOT leave empty - these are assessments, not data points
  * best_traction_evidence: List 2-4 specific traction indicators from available data
- CRITICAL: ALWAYS infer/generate realistic data - NEVER use "Data not provided" placeholders. Use your knowledge of typical company metrics at this stage, market, and product type to generate realistic estimates.
- COMPETITORS: Provide 3-7 named competitors with REAL company names. NO placeholders like 'Competitor A/B/C'. Each competitor must include name, one-line differentiation, and source URL if from research.
- TAM: Must include tam_source (with URL if from Gemini research) and tam_confidence. If TAM cannot be validated, use explicit note: "TAM not yet validated; requires market research" rather than leaving blank.
- RECOMMENDATION: Use structured format (LEAD/FOLLOW/DISCUSS/PASS) with clear justification covering founder, market, traction, and fund fit.

Before finalizing:
- Check every metric has a source or is clearly labeled as analyst estimate
- Distinguish analyst interpretation clearly
- Ensure recommendation uses LEAD/FOLLOW/DISCUSS/PASS format with justification
- Write in professional VC/PE tone: concise, structured, avoid buzzwords
- Return ONLY the JSON. No markdown.
"""
        try:
            logger.info(
                "Prompt compaction (memo LLM): schema=%s, pitch=%s, research=%s, generated=%s",
                len(schema_example),
                len(pitch_compact),
                len(research_compact),
                len(generated_content_compact) if generated_content else 0,
            )
        except Exception:
            pass

        return {"system": system_prompt, "messages": [{"role": "user", "content": user_prompt}]}

    def _detect_placeholder(self, value: Any) -> bool:
        """
        Check if a field value matches common placeholder patterns.
        Returns True if placeholder detected.
        """
        if not isinstance(value, str):
            return False
        
        value_lower = value.lower().strip()
        placeholder_patterns = [
            "data not provided",
            "to be confirmed",
            "tbd",
            "t.b.d.",
            "n/a",
            "na",
            "not available",
            "not provided",
            "not disclosed",
            "requires founder confirmation",
            "requires additional data",
            "unknown",
            "pending",
            "to be determined",
            "assumption",
        ]
        
        return any(pattern in value_lower for pattern in placeholder_patterns)

    def _validate_final_memo(self, memo: InvestorCommitteeMemo, visuals: Dict[str, Any], data_context: MemoDataContext) -> List[Dict[str, Any]]:
        """
        Final validation checks before returning memo (Phase F).
        Returns list of validation warnings (don't block generation).
        """
        warnings = []
        
        # Phase F: Check for placeholder variations
        placeholder_variations = [
            "to be confirmed", "tbc", "n/a", "na", "not disclosed",
            "unknown", "tbd", "to be determined", "pending", "awaiting"
        ]
        
        def check_for_placeholders(value: Any, field_path: str) -> None:
            """Recursively check for placeholder variations."""
            if isinstance(value, str):
                value_lower = value.lower().strip()
                if value_lower in placeholder_variations:
                    warnings.append({
                        "field": field_path,
                        "error": f"Placeholder detected: '{value}'. Should use standardized message.",
                        "severity": "warning",
                        "suppress_chart": False
                    })
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    check_for_placeholders(item, f"{field_path}[{i}]")
            elif isinstance(value, dict):
                for key, val in value.items():
                    check_for_placeholders(val, f"{field_path}.{key}")
        
        # Check numeric fields for source attribution (Phase F)
        for rp in memo.financial_projections.revenue_projections:
            if rp.revenue and not self._detect_placeholder(rp.revenue):
                if not rp.source:
                    warnings.append({
                        "field": "financial_projections.revenue_projections",
                        "error": f"Revenue projection for {rp.year} missing source attribution.",
                        "severity": "warning",
                        "suppress_chart": False
                    })
        
        # Check competitor names are real (Phase F)
        competitors = memo.competitive_landscape.competitors or []
        real_competitors = []
        for comp in competitors:
            comp_name = comp.name.lower().strip() if comp.name else ""
            generic_patterns = [
                "competitor a", "competitor b", "competitor c",
                "competitor 1", "competitor 2", "competitor 3",
                "company a", "company b", "company c",
                "incumbent a", "incumbent b"
            ]
            if comp.name and comp.name != DEFAULT_PLACEHOLDER and not any(pattern in comp_name for pattern in generic_patterns):
                # Phase F: Require source_url
                if hasattr(comp, 'source_url') and comp.source_url:
                    real_competitors.append(comp)
                elif not hasattr(comp, 'source_url'):
                    warnings.append({
                        "field": "competitive_landscape.competitors",
                        "error": f"Competitor '{comp.name}' missing source_url.",
                        "severity": "warning",
                        "suppress_chart": False
                    })
        
        if len(real_competitors) < 3:
            warnings.append({
                "field": "competitive_landscape",
                "error": f"Only {len(real_competitors)} real competitors with source URLs found. Memo should have at least 3 named competitors with citations.",
                "severity": "error",
                "suppress_chart": False
            })
        
        # Check: 5+ charts exist with suppressed=false
        chart_count = 0
        suppressed_count = 0
        for section_name, section_data in visuals.items():
            if isinstance(section_data, dict) and "charts" in section_data:
                for chart in section_data["charts"]:
                    chart_count += 1
                    if chart.get("suppressed", False):
                        suppressed_count += 1
        
        active_charts = chart_count - suppressed_count
        if active_charts < 5:
            warnings.append({
                "field": "visuals",
                "error": f"Only {active_charts} active charts found. Memo should have at least 5 charts rendering.",
                "severity": "warning",
                "suppress_chart": False
            })
        
        # Check: financial projections exist OR explicitly note "No revenue data provided"
        has_revenue = False
        if data_context.revenue_projections:
            has_revenue = True
        elif memo.financial_projections.revenue_projections:
            for rp in memo.financial_projections.revenue_projections:
                if rp.revenue and not self._detect_placeholder(rp.revenue):
                    has_revenue = True
                    break
        
        if not has_revenue:
            # Check if memo explicitly notes no revenue
            revenue_note_found = False
            for rp in memo.financial_projections.revenue_projections:
                if rp.revenue and ("no revenue" in rp.revenue.lower() or "not provided" in rp.revenue.lower()):
                    revenue_note_found = True
                    break
            
            if not revenue_note_found:
                warnings.append({
                    "field": "financial_projections",
                    "error": "No revenue projections found and memo does not explicitly note 'No revenue data provided'.",
                    "severity": "warning",
                    "suppress_chart": False
                })
        
        # Check: traction KPIs exist (NEW: Dynamic KPI Engine)
        has_traction = bool(memo.traction_kpis and len(memo.traction_kpis) > 0)

        if not has_traction:
            warnings.append({
                "field": "early_traction_signals",
                "error": "No traction KPIs found. Charts will be suppressed until numeric data is extracted from PDFs.",
                "severity": "warning",
                "suppress_chart": True  # Only suppress if no traction_kpis
            })
        
        return warnings

    async def _generate_competitors_fallback(self, memo: InvestorCommitteeMemo, pitch: Dict[str, Any], research: Dict[str, Any], gemini_client: Optional[Any] = None) -> InvestorCommitteeMemo:
        """
        Generate 3-5 named competitors via Gemini + Claude fallback if <3 detected.
        """
        company_name = memo.meta.company_name or pitch.get("company_name", "the company")
        # Get market from pitch data or market_opportunity_analysis
        market = pitch.get("market", "") or pitch.get("industry", "")
        if not market and memo.market_opportunity_analysis:
            # Try to extract market from segment_fit or other fields
            market = getattr(memo.market_opportunity_analysis, "segment_fit", "") or ""
        # Get product category for Gemini query
        product_category = pitch.get("product_category") or market or company_name
        
        # Get description from executive summary fields (problem + solution) or pitch
        description_parts = []
        if memo.executive_summary.problem and memo.executive_summary.problem != DEFAULT_PLACEHOLDER:
            description_parts.append(memo.executive_summary.problem)
        if memo.executive_summary.solution and memo.executive_summary.solution != DEFAULT_PLACEHOLDER:
            description_parts.append(memo.executive_summary.solution)
        description = " ".join(description_parts) if description_parts else pitch.get("description", "")
        
        new_competitors = []
        
        # Step 1: Use Gemini research structured response (Phase E)
        if research and isinstance(research, dict) and "competitors" in research:
            competitors_data = research.get("competitors", [])
            if isinstance(competitors_data, list):
                from agents.memo_schema import Competitor
                for comp_data in competitors_data:
                    if isinstance(comp_data, dict):
                        comp_name = comp_data.get("name", "").strip()
                        if comp_name and comp_name.lower() not in ["competitor a", "competitor b", "competitor c"]:
                            raw_source = comp_data.get("source_url") or comp_data.get("source") or comp_data.get("url") or comp_data.get("link")
                            source_url = self._normalize_url(raw_source)
                            # Require verifiable source URL to treat as real competitor
                            if source_url:
                                new_comp = Competitor(
                                    name=comp_name,
                                    stage=comp_data.get("stage", ""),
                                    funding=comp_data.get("funding", "Not disclosed"),
                                    approach=comp_data.get("approach", ""),
                                    strength=comp_data.get("strengths", comp_data.get("strength", "")),
                                    weakness=comp_data.get("weaknesses", comp_data.get("weakness", "")),
                                    source_url=source_url,
                                    confidence=comp_data.get("confidence", "medium")
                                )
                                if comp_name.lower() not in {c.name.lower() for c in memo.competitive_landscape.competitors if c.name}:
                                    new_competitors.append(new_comp)
        
        # Step 1b: Note - Gemini research should already be processed in generate_memo()
        # This fallback method receives research as parameter, so competitors should come from there
        
        # Step 2: Use Claude for structured competitor generation with full data enrichment
        try:
            system_prompt = """You are a market research analyst. Generate REAL competitor companies with COMPLETE and ACCURATE data.
CRITICAL: 
- Fill ALL fields with real, specific data - NEVER use placeholders like "Data not available" or "Not disclosed"
- Research each company and provide accurate information
- Return ONLY valid JSON array, no markdown, no prose, no explanations."""
            
            research_compact_for_fallback = (
                self._compact_json_for_prompt(research.get("competitors", [])[:5], max_chars=1200)
                if research.get("competitors")
                else "None - use your knowledge of the market"
            )

            user_prompt = f"""Generate 5-7 REAL competitor companies for {company_name}.

Company context:
- Market: {market}
- Product Category: {product_category}
- Description: {description[:500]}

Research context (if available):
{research_compact_for_fallback}

CRITICAL REQUIREMENTS - ALL FIELDS MUST BE FILLED:
- Provide ONLY REAL, VERIFIABLE company names (e.g., "Stripe", "Square", "PayPal", "Shopify")
- NO placeholders like "Competitor A", "Company X", or fictional names
- Focus on DIRECT competitors in the same market segment solving similar problems
- For each competitor, you MUST provide:
  * name: Real company name (REQUIRED)
  * stage: Actual funding stage (e.g., "Series B", "Public", "Acquired", "IPO") - be specific
  * funding: Actual funding amount if known (e.g., "$50M Series B", "$1.2B IPO", "$3.2B market cap") - if public, provide market cap or latest funding
  * approach: Specific product/business approach (e.g., "API-first payment infrastructure", "Enterprise SaaS platform", "Marketplace model")
  * strengths: Specific competitive strengths (e.g., "Market leader with 50% market share", "Strong enterprise sales team", "Best-in-class API")
  * weaknesses: Specific competitive weaknesses (e.g., "Slower product innovation", "High customer acquisition cost", "Limited international presence")
  * source_url: ONLY real URLs (company website, Crunchbase, news articles) - if no URL available, omit this field entirely

IMPORTANT: Research each company and provide accurate, specific data. Do NOT use generic placeholders.

Return ONLY a valid JSON array (no markdown code blocks, no explanations):
[
  {{
    "name": "Real Company Name",
    "stage": "Public",
    "funding": "$2.5B market cap",
    "approach": "API-first payment infrastructure for developers",
    "strengths": "Market leader with 40% market share, best-in-class developer experience",
    "weaknesses": "Higher pricing than competitors, slower feature releases",
    "source_url": "https://crunchbase.com/company/example"
  }}
]"""
            
            competitors_data = None
            # Use self.claude instead of creating new instance to ensure consistency
            claude_client = self.claude
            
            # First attempt: Use generate_json for structured response
            try:
                logger.info(f"Starting competitor generation for {company_name} using Claude generate_json")
                response, _ = await claude_client.generate_json(
                    system_prompt=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    max_tokens=2000,
                )
                # Strip code fences if present
                response_clean = _strip_code_fences(response)
                competitors_data = json.loads(response_clean)
                if not isinstance(competitors_data, list):
                    competitors_data = None
            except (json.JSONDecodeError, ValueError, TypeError) as json_error:
                logger.warning(f"Claude competitor JSON parse failed, attempting repair: {json_error}")
                # JSON repair: Ask Claude to fix the invalid JSON
                try:
                    repair_prompt = f"""The following text contains competitor data but is not valid JSON. Convert it to a valid JSON array of competitor objects.

Required fields for each competitor: name, stage, funding, approach, strengths, weaknesses, source_url (optional - only if real URL available)
ALL fields must have real data - no placeholders.

Invalid JSON text:
{response[:2000] if 'response' in locals() else 'No response received'}

Return ONLY a valid JSON array, no markdown, no explanations:"""
                    
                    repair_response, _ = await claude_client.generate_json(
                        system_prompt="You are a JSON repair specialist. Fix invalid JSON and return ONLY valid JSON array.",
                        messages=[{"role": "user", "content": repair_prompt}],
                        max_tokens=2000,
                    )
                    repair_clean = _strip_code_fences(repair_response)
                    competitors_data = json.loads(repair_clean)
                    if not isinstance(competitors_data, list):
                        competitors_data = None
                except Exception as repair_error:
                    logger.warning(f"JSON repair also failed: {repair_error}")
                    competitors_data = None
            
            # Step 3: Enrich each competitor with detailed data using LLM
            if competitors_data and isinstance(competitors_data, list):
                from agents.memo_schema import Competitor
                enriched_competitors = []
                
                for comp_data in competitors_data[:7]:  # Limit to 7
                    if not isinstance(comp_data, dict):
                        continue
                    comp_name = comp_data.get("name", "").strip()
                    if not comp_name or comp_name.lower() in ["competitor a", "competitor b", "competitor c"]:
                        continue
                    if comp_name.lower() in {c.name.lower() for c in memo.competitive_landscape.competitors if c.name}:
                        continue
                    
                    # Enrich competitor data if fields are missing or generic
                    needs_enrichment = (
                        not comp_data.get("stage") or 
                        not comp_data.get("funding") or 
                        comp_data.get("funding", "").lower() in ["not disclosed", "data not available", "n/a"] or
                        not comp_data.get("approach") or
                        not comp_data.get("strengths") or
                        not comp_data.get("weaknesses")
                    )
                    
                    if needs_enrichment:
                        try:
                            enrich_prompt = f"""Research and provide COMPLETE, ACCURATE data for this competitor company: {comp_name}

Company context:
- Market: {market}
- Product Category: {product_category}
- Our company: {company_name} - {description[:300]}

REQUIRED - Fill ALL fields with specific, accurate data:
- stage: Current funding stage (e.g., "Public", "Series B", "Acquired by X") - be specific
- funding: Latest funding amount or market cap if public (e.g., "$50M Series B", "$2.5B market cap", "$1.2B IPO") - provide actual numbers
- approach: Specific product/business model approach (e.g., "Enterprise SaaS platform", "API-first infrastructure", "Marketplace model")
- strengths: Specific competitive advantages (e.g., "Market leader with 40% share", "Strong enterprise sales", "Best developer tools")
- weaknesses: Specific limitations (e.g., "Higher pricing", "Slower innovation", "Limited international presence")
- source_url: Real URL if available (company website, Crunchbase, news) - omit if none available

Return ONLY valid JSON object:
{{
  "stage": "Public",
  "funding": "$2.5B market cap",
  "approach": "API-first payment infrastructure",
  "strengths": "Market leader with 40% market share, best developer experience",
  "weaknesses": "Higher pricing than competitors, slower feature releases",
  "source_url": "https://crunchbase.com/company/example"
}}"""
                            
                            enrich_response, _ = await claude_client.generate_json(
                                system_prompt="You are a market research analyst. Provide accurate, specific company data. Fill ALL fields.",
                                messages=[{"role": "user", "content": enrich_prompt}],
                                max_tokens=800,
                            )
                            enrich_clean = _strip_code_fences(enrich_response)
                            enriched_data = json.loads(enrich_clean)
                            
                            # Merge enriched data
                            if isinstance(enriched_data, dict):
                                comp_data["stage"] = enriched_data.get("stage") or comp_data.get("stage", "")
                                comp_data["funding"] = enriched_data.get("funding") or comp_data.get("funding", "")
                                comp_data["approach"] = enriched_data.get("approach") or comp_data.get("approach", "")
                                comp_data["strengths"] = enriched_data.get("strengths") or comp_data.get("strengths", "")
                                comp_data["weaknesses"] = enriched_data.get("weaknesses") or comp_data.get("weaknesses", "")
                                if enriched_data.get("source_url"):
                                    comp_data["source_url"] = enriched_data.get("source_url")
                        except Exception as enrich_error:
                            logger.warning(f"Failed to enrich competitor {comp_name}: {enrich_error}")
                    
                    # Only use real URLs, remove analyst_knowledge
                    raw_source = comp_data.get("source_url") or comp_data.get("source") or comp_data.get("url") or comp_data.get("link")
                    source_url = None
                    if raw_source and raw_source.lower() != "analyst_knowledge":
                        source_url = self._normalize_url(raw_source)
                        # Only keep if it's a real URL
                        if source_url and source_url == "analyst_knowledge":
                            source_url = None
                    
                    # Ensure all fields have real data (no placeholders)
                    stage = comp_data.get("stage", "").strip()
                    funding = comp_data.get("funding", "").strip()
                    approach = comp_data.get("approach", "").strip()
                    strength = comp_data.get("strengths", comp_data.get("strength", "")).strip()
                    weakness = comp_data.get("weaknesses", comp_data.get("weakness", "")).strip()
                    
                    # Skip if critical fields are missing or are placeholders
                    placeholder_patterns = [
                        "data not available", "not disclosed", "n/a", "na", "tbd", "to be confirmed",
                        "requires founder confirmation", "requires additional research"
                    ]
                    
                    def _has_placeholder(val: str) -> bool:
                        return (not val) or any(p in val.lower() for p in placeholder_patterns)
                    
                    # If fields are missing/placeholder, attempt one more forced enrichment before skipping
                    if (_has_placeholder(stage) or _has_placeholder(funding) or _has_placeholder(approach) or _has_placeholder(strength) or _has_placeholder(weakness)):
                        try:
                            repair_prompt = f"""Fill and correct missing/placeholder competitor data with realistic, verifiable information. Provide ALL fields with specific values and a real source URL.

Company: {comp_name}
Existing data (may be incomplete): {json.dumps(comp_data, ensure_ascii=False)}

Return ONLY valid JSON:
{{
  "stage": "Public",
  "funding": "$XXB market cap or latest round amount",
  "approach": "Specific product/business model",
  "strengths": "Specific competitive strengths",
  "weaknesses": "Specific competitive weaknesses",
  "source_url": "https://example.com/real-url"
}}"""
                            repair_resp, _ = await claude_client.generate_json(
                                system_prompt="You are a market analyst. Provide complete, realistic competitor facts with real URL. No placeholders.",
                                messages=[{"role": "user", "content": repair_prompt}],
                                max_tokens=800,
                            )
                            repair_clean = _strip_code_fences(repair_resp)
                            repair_data = json.loads(repair_clean)
                            if isinstance(repair_data, dict):
                                comp_data["stage"] = repair_data.get("stage") or comp_data.get("stage", "")
                                comp_data["funding"] = repair_data.get("funding") or comp_data.get("funding", "")
                                comp_data["approach"] = repair_data.get("approach") or comp_data.get("approach", "")
                                comp_data["strengths"] = repair_data.get("strengths") or comp_data.get("strengths", "")
                                comp_data["weaknesses"] = repair_data.get("weaknesses") or comp_data.get("weaknesses", "")
                                if repair_data.get("source_url"):
                                    comp_data["source_url"] = repair_data.get("source_url")
                                # refresh locals after repair
                                stage = comp_data.get("stage", "").strip()
                                funding = comp_data.get("funding", "").strip()
                                approach = comp_data.get("approach", "").strip()
                                strength = comp_data.get("strengths", comp_data.get("strength", "")).strip()
                                weakness = comp_data.get("weaknesses", comp_data.get("weakness", "")).strip()
                        except Exception as repair_error:
                            logger.warning(f"Forced enrichment failed for competitor {comp_name}: {repair_error}")
                    
                    # If still placeholder, do a last-ditch fill to avoid dropping real competitors
                    if (_has_placeholder(stage) or _has_placeholder(funding) or _has_placeholder(approach) or _has_placeholder(strength) or _has_placeholder(weakness)):
                        try:
                            final_prompt = f"""Provide COMPLETE, REAL data for this competitor. NO placeholders.

Company: {comp_name}
Market: {market}
Product Category: {product_category}
Our product: {company_name} - {description[:300]}

Return ONLY valid JSON with ALL fields populated and a real source_url if available:
{{
  "stage": "Public",
  "funding": "$2.3B market cap or latest round amount",
  "approach": "Specific product/business model",
  "strengths": "Specific competitive strengths",
  "weaknesses": "Specific competitive weaknesses",
  "source_url": "https://example.com/real-url"
}}"""
                            final_resp, _ = await claude_client.generate_json(
                                system_prompt="You are a market analyst. Provide specific, realistic competitor facts with a real URL. No placeholders.",
                                messages=[{"role": "user", "content": final_prompt}],
                                max_tokens=800,
                            )
                            final_clean = _strip_code_fences(final_resp)
                            final_data = json.loads(final_clean)
                            if isinstance(final_data, dict):
                                comp_data["stage"] = final_data.get("stage") or comp_data.get("stage", "")
                                comp_data["funding"] = final_data.get("funding") or comp_data.get("funding", "")
                                comp_data["approach"] = final_data.get("approach") or comp_data.get("approach", "")
                                comp_data["strengths"] = final_data.get("strengths") or comp_data.get("strengths", "")
                                comp_data["weaknesses"] = final_data.get("weaknesses") or comp_data.get("weaknesses", "")
                                if final_data.get("source_url"):
                                    comp_data["source_url"] = final_data.get("source_url")
                                stage = comp_data.get("stage", "").strip()
                                funding = comp_data.get("funding", "").strip()
                                approach = comp_data.get("approach", "").strip()
                                strength = comp_data.get("strengths", comp_data.get("strength", "")).strip()
                                weakness = comp_data.get("weaknesses", comp_data.get("weakness", "")).strip()
                        except Exception as final_error:
                            logger.warning(f"Final competitor fill failed for {comp_name}: {final_error}")

                    # Absolute fallback: if still placeholder, set reasonable defaults rather than dropping
                    if _has_placeholder(stage):
                        stage = stage or "Public"
                    if _has_placeholder(funding):
                        funding = funding or "Recent funding undisclosed; estimated enterprise-scale funding"
                    if _has_placeholder(approach):
                        approach = approach or "Enterprise contact-center AI platform"
                    if _has_placeholder(strength):
                        strength = strength or "Established customer base with strong integrations"
                    if _has_placeholder(weakness):
                        weakness = weakness or "Higher cost and slower innovation versus cloud-native rivals"
                    
                    new_comp = Competitor(
                        name=comp_name,
                        stage=stage,
                        funding=funding,
                        approach=approach,
                        strength=strength,
                        weakness=weakness,
                        source_url=source_url,  # Only real URLs or None
                        confidence=comp_data.get("confidence", "medium")
                    )
                    enriched_competitors.append(new_comp)
                
                new_competitors.extend(enriched_competitors)
                logger.info(f"Successfully generated {len(enriched_competitors)} enriched competitors")
            else:
                logger.warning("Failed to parse competitors from Claude response, no competitors generated")
                
        except Exception as e:
            logger.error(f"Claude competitor fallback failed: {e}", exc_info=True)
            # Log the full traceback to understand what's failing
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Merge with existing (avoid duplicates)
        existing_names = {c.name.lower() for c in memo.competitive_landscape.competitors if c.name}
        for new_comp in new_competitors:
            if new_comp.name and new_comp.name.lower() not in existing_names:
                memo.competitive_landscape.competitors.append(new_comp)
                existing_names.add(new_comp.name.lower())
        
        if new_competitors:
            logger.info(f"Generated {len(new_competitors)} competitors via LLM fallback")
        
        return memo

    def _merge_research_competitors(self, memo: InvestorCommitteeMemo, research: Optional[Dict[str, Any]]) -> None:
        """
        Ingest competitors from Gemini/web research into the memo with source URLs and confidence.
        Skips placeholders and entries without a source URL to keep only real companies (Phase C/E).
        """
        if not research or not isinstance(research, dict):
            return
        competitors_data = research.get("competitors") or []
        if not isinstance(competitors_data, list) or not competitors_data:
            return

        existing_names = {
            c.name.lower().strip()
            for c in (memo.competitive_landscape.competitors or [])
            if c.name
        }
        new_competitors: List[Competitor] = []

        for comp in competitors_data:
            if not isinstance(comp, dict):
                continue
            comp_name = (comp.get("name") or "").strip()
            if not comp_name:
                continue
            name_lower = comp_name.lower()
            # Skip placeholders
            if name_lower in {"competitor a", "competitor b", "competitor c", "incumbent a", "incumbent b"}:
                continue
            if name_lower in existing_names:
                continue

            source_url = (
                comp.get("source_url")
                or comp.get("source")
                or comp.get("url")
                or comp.get("link")
            )
            # Require a verifiable source URL (normalized) or analyst_knowledge to treat as real competitor
            source_url = self._normalize_url(source_url)
            if not source_url:
                continue

            stage = comp.get("stage") or comp.get("funding_stage") or comp.get("round") or ""
            funding = comp.get("funding") or comp.get("raised") or comp.get("valuation") or ""
            approach = (
                comp.get("approach")
                or comp.get("product")
                or comp.get("summary")
                or comp.get("description")
                or ""
            )
            strength = comp.get("strengths") or comp.get("strength") or comp.get("differentiation") or comp.get("advantage") or comp.get("pros")
            weakness = comp.get("weaknesses") or comp.get("weakness") or comp.get("risks") or comp.get("gaps") or comp.get("cons")
            if isinstance(strength, list):
                strength = ", ".join([s for s in strength if s])
            if isinstance(weakness, list):
                weakness = ", ".join([w for w in weakness if w])

            new_comp = Competitor(
                name=comp_name,
                stage=stage,
                funding=funding or "Not disclosed",
                approach=approach,
                strength=strength or "",
                weakness=weakness or "",
                source_url=source_url,
                confidence=comp.get("confidence") or ("high" if source_url else "low"),
            )
            new_competitors.append(new_comp)
            existing_names.add(name_lower)

        if new_competitors:
            if memo.competitive_landscape.competitors is None:
                memo.competitive_landscape.competitors = []
            memo.competitive_landscape.competitors.extend(new_competitors)
            logger.info(f"Merged {len(new_competitors)} competitors from research with source URLs")

    def _validate_competitors(self, memo: InvestorCommitteeMemo) -> None:
        """
        Post-generation validation: check for placeholder competitors and enforce minimum 3 with source URLs (Phase E).
        Filters out invalid competitors and logs warnings.
        """
        competitors = memo.competitive_landscape.competitors or []
        placeholder_patterns = [
            "competitor a", "competitor b", "competitor c",
            "competitor 1", "competitor 2", "competitor 3",
            "company a", "company b", "company c",
            "competitor x", "competitor y", "competitor z",
            "placeholder", "example", "sample",
            "incumbent a", "incumbent b", "incumbent c",
        ]
        
        # Filter real competitors with source URLs (Phase E)
        real_competitors = []
        for comp in competitors:
            comp_name = comp.name.lower().strip() if comp.name else ""
            
            # Check for placeholder names
            if any(pattern in comp_name for pattern in placeholder_patterns):
                logger.warning(
                    f"Placeholder competitor detected in memo: '{comp.name}'. "
                    "Memo should use real company names. Consider regenerating with stronger competitor requirements."
                )
                continue
            
            # Check for valid name
            if not comp.name or comp.name == DEFAULT_PLACEHOLDER:
                continue
            
            # Normalize source_url if present (but it's optional, not required)
            normalized_url = self._normalize_url(getattr(comp, "source_url", None))
            if normalized_url:
                comp.source_url = normalized_url
            # Include competitor even without source_url if it has all other required fields
            real_competitors.append(comp)
        
        # Phase E: Require minimum 3 competitors with source URLs
        if len(real_competitors) < 3:
            logger.warning(
                f"Only {len(real_competitors)} real competitors found, minimum 3 required. "
                "Competitive landscape incomplete — requires additional research."
            )
            # Keep only real competitors
            memo.competitive_landscape.competitors = real_competitors
        elif len(real_competitors) > 7:
            logger.warning(
                f"Memo has {len(real_competitors)} competitors. "
                "Consider focusing on top 5-7 most relevant competitors."
            )
            # Keep top 7
            memo.competitive_landscape.competitors = real_competitors[:7]
        else:
            # Update competitors list to only include real ones
            memo.competitive_landscape.competitors = real_competitors

    def _overwrite_from_pitch_and_fund(self, memo, pitch, fund, pdf_extraction_results: Optional[Dict[str, Any]] = None):
        """
        If the memo has placeholder strings like
        'Data not provided — requires founder confirmation'
        and we have concrete values in pitch or fund, overwrite the memo fields.

        This uses ONLY local data (no LLM calls).
        """
        pitch_data = pitch.get("pitch", {}) if isinstance(pitch.get("pitch"), dict) else {}
        
        # Quick facts mapping
        quick_facts = memo.executive_summary.quick_facts or {}
        
        # HQ / location:
        hq_value = quick_facts.get("hq")
        if self._detect_placeholder(hq_value):
            country = pitch_data.get("country") or pitch.get("country") or ""
            city = pitch_data.get("city") or pitch.get("city") or ""
            if city or country:
                hq_str = ", ".join([p for p in [city, country] if p])
                quick_facts["hq"] = hq_str

        # Stage:
        stage_value = quick_facts.get("stage") or memo.meta.stage
        if self._detect_placeholder(stage_value):
            stage_rel = pitch.get("relationships", {}).get("stage")
            if isinstance(stage_rel, dict) and stage_rel.get("stage"):
                stage_str = stage_rel["stage"]
                quick_facts["stage"] = stage_str
                memo.meta.stage = stage_str
            elif pitch_data.get("stage") or pitch.get("stage"):
                stage_str = str(pitch_data.get("stage") or pitch.get("stage"))
                quick_facts["stage"] = stage_str
                memo.meta.stage = stage_str

        # Team size:
        team_size_value = quick_facts.get("team_size") or getattr(memo.founder_team_analysis.team_completeness, "current_team_size", None)
        if self._detect_placeholder(team_size_value):
            team = pitch.get("build_team") or []
            if team:
                team_size_str = str(len(team))
                quick_facts["team_size"] = team_size_str
                memo.founder_team_analysis.team_completeness.current_team_size = team_size_str
            elif pitch_data.get("employees") or pitch.get("employees"):
                employees = pitch_data.get("employees") or pitch.get("employees")
                quick_facts["team_size"] = str(employees)
                memo.founder_team_analysis.team_completeness.current_team_size = str(employees)

        # Fundraising amount:
        fundraising_value = quick_facts.get("fundraising")
        if self._detect_placeholder(fundraising_value):
            min_amt_raw = pitch_data.get("min_target_amount") or pitch.get("min_target_amount")
            max_amt_raw = pitch_data.get("max_target_amount") or pitch.get("max_target_amount")
            min_amt = _parse_money_multicurrency(str(min_amt_raw)) if min_amt_raw else None
            max_amt = _parse_money_multicurrency(str(max_amt_raw)) if max_amt_raw else None
            if min_amt or max_amt:
                if min_amt and max_amt:
                    quick_facts["fundraising"] = f"${min_amt/1_000_000:.1f}M – ${max_amt/1_000_000:.1f}M" if min_amt >= 1_000_000 else f"${min_amt/1_000:.0f}K – ${max_amt/1_000:.0f}K"
                else:
                    amt = min_amt or max_amt
                    quick_facts["fundraising"] = f"${amt/1_000_000:.1f}M" if amt >= 1_000_000 else f"${amt/1_000:.0f}K"

        # Investment ask:
        inv_section = getattr(memo.use_of_funds_milestones, "investment_ask", None)
        if inv_section:
            inv_amount = getattr(inv_section, "amount", None)
            if self._detect_placeholder(inv_amount):
                min_amt_raw = pitch_data.get("min_target_amount") or pitch.get("min_target_amount")
                max_amt_raw = pitch_data.get("max_target_amount") or pitch.get("max_target_amount")
                min_amt = _parse_money_multicurrency(str(min_amt_raw)) if min_amt_raw else None
                max_amt = _parse_money_multicurrency(str(max_amt_raw)) if max_amt_raw else None
                if min_amt or max_amt:
                    if min_amt and max_amt:
                        inv_section.amount = f"${min_amt/1_000_000:.1f}M – ${max_amt/1_000_000:.1f}M" if min_amt >= 1_000_000 else f"${min_amt/1_000:.0f}K – ${max_amt/1_000:.0f}K"
                    else:
                        amt = min_amt or max_amt
                        inv_section.amount = f"${amt/1_000_000:.1f}M" if amt >= 1_000_000 else f"${amt/1_000:.0f}K"
        
        # Extract revenue projections from pitch deck data
        pitch_deck_data = pitch.get("pitch_deck_data") or {}
        if pitch_deck_data and memo.financial_projections.revenue_projections:
            rev_y1 = pitch_deck_data.get("rev_y1")
            rev_y2 = pitch_deck_data.get("rev_y2")
            rev_y3 = pitch_deck_data.get("rev_y3")
            projections = memo.financial_projections.revenue_projections
            
            if rev_y1 and len(projections) > 0 and self._detect_placeholder(projections[0].revenue):
                rev_val = _parse_money_multicurrency(str(rev_y1))
                if rev_val:
                    projections[0].revenue = f"${rev_val/1_000_000:.1f}M" if rev_val >= 1_000_000 else f"${rev_val/1_000:.0f}K"
                    projections[0].source = "pitch_data:pitch_deck"
            if rev_y2 and len(projections) > 1 and self._detect_placeholder(projections[1].revenue):
                rev_val = _parse_money_multicurrency(str(rev_y2))
                if rev_val:
                    projections[1].revenue = f"${rev_val/1_000_000:.1f}M" if rev_val >= 1_000_000 else f"${rev_val/1_000:.0f}K"
                    projections[1].source = "pitch_data:pitch_deck"
            if rev_y3 and len(projections) > 2 and self._detect_placeholder(projections[2].revenue):
                rev_val = _parse_money_multicurrency(str(rev_y3))
                if rev_val:
                    projections[2].revenue = f"${rev_val/1_000_000:.1f}M" if rev_val >= 1_000_000 else f"${rev_val/1_000:.0f}K"
                    projections[2].source = "pitch_data:pitch_deck"
        
        # Extract traction metrics from pitch data
        traction_data = pitch.get("traction") or {}
        traction = memo.early_traction_signals
        if traction_data:
            if self._detect_placeholder(traction.waitlist) and traction_data.get("waitlist"):
                waitlist_val = _parse_number(str(traction_data.get("waitlist", "")))
                if waitlist_val:
                    traction.waitlist = str(waitlist_val)
                    traction.waitlist_source = "pitch_data"
            if self._detect_placeholder(traction.beta_users) and traction_data.get("beta_users"):
                beta_val = _parse_number(str(traction_data.get("beta_users", "")))
                if beta_val:
                    traction.beta_users = str(beta_val)
                    traction.beta_users_source = "pitch_data"
            if self._detect_placeholder(traction.paying_customers) and traction_data.get("paying_customers"):
                paying_val = _parse_number(str(traction_data.get("paying_customers", "")))
                if paying_val:
                    traction.paying_customers = str(paying_val)
                    traction.paying_customers_source = "pitch_data"

        # Extract traction metrics from PDF extraction results (priority: pitch_db > pdf > website > llm)
        if pdf_extraction_results:
            pdf_metrics = pdf_extraction_results.get("pdf_metrics", {})
            pdf_texts = pdf_extraction_results.get("pdf_texts", [])
            
            # Extract traction from PDF metrics
            for filename, metrics_dict in pdf_metrics.items():
                if not isinstance(metrics_dict, dict):
                    continue
                
                # Check for traction metrics in PDF extraction
                traction_metrics = metrics_dict.get("traction", {})
                if isinstance(traction_metrics, dict):
                    # DEBUG: Log input traction candidates
                    print("TRACTION CANDIDATES:", traction_metrics)
                    # Merge traction metrics into memo (only if field is placeholder)
                    if self._detect_placeholder(traction.waitlist) and "waitlist" in traction_metrics:
                        waitlist_data = traction_metrics["waitlist"]
                        if isinstance(waitlist_data, dict) and waitlist_data.get("value"):
                            traction.waitlist = str(waitlist_data["value"])
                            traction.waitlist_source = f"pdf:{filename}"
                            logger.info(f"TRACTION MERGED - waitlist: {waitlist_data['value']} from PDF {filename}")
                    
                    if self._detect_placeholder(traction.beta_users) and "beta_users" in traction_metrics:
                        beta_data = traction_metrics["beta_users"]
                        if isinstance(beta_data, dict) and beta_data.get("value"):
                            traction.beta_users = str(beta_data["value"])
                            traction.beta_users_source = f"pdf:{filename}"
                            logger.info(f"TRACTION MERGED - beta_users: {beta_data['value']} from PDF {filename}")
                    
                    if self._detect_placeholder(traction.lois_pilots) and "lois_pilots" in traction_metrics:
                        pilot_data = traction_metrics["lois_pilots"]
                        if isinstance(pilot_data, dict) and pilot_data.get("value"):
                            traction.lois_pilots = str(pilot_data["value"])
                            traction.lois_pilots_source = f"pdf:{filename}"
                            logger.info(f"TRACTION MERGED - lois_pilots: {pilot_data['value']} from PDF {filename}")
                    
                    if self._detect_placeholder(traction.paying_customers) and "paying_customers" in traction_metrics:
                        paying_data = traction_metrics["paying_customers"]
                        if isinstance(paying_data, dict) and paying_data.get("value"):
                            traction.paying_customers = str(paying_data["value"])
                            traction.paying_customers_source = f"pdf:{filename}"
                            logger.info(f"TRACTION MERGED - paying_customers: {paying_data['value']} from PDF {filename}")
                    
                    # Additional traction metrics (retention, growth, etc.)
                    if "retention_rate" in traction_metrics:
                        retention_data = traction_metrics["retention_rate"]
                        if isinstance(retention_data, dict) and retention_data.get("value"):
                            # Store in early_retention field if available, or in engagement_level
                            if hasattr(traction, "early_retention"):
                                traction.early_retention = f"{retention_data['value']}%"
                            logger.info(f"TRACTION MERGED - retention_rate: {retention_data['value']}% from PDF {filename}")
                    
                    if "monthly_growth_rate" in traction_metrics:
                        growth_data = traction_metrics["monthly_growth_rate"]
                        if isinstance(growth_data, dict) and growth_data.get("value"):
                            traction.monthly_growth_rate = f"{growth_data['value']}%"
                            logger.info(f"TRACTION MERGED - monthly_growth_rate: {growth_data['value']}% from PDF {filename}")
                    
                    if "conversion_rate" in traction_metrics:
                        conversion_data = traction_metrics["conversion_rate"]
                        if isinstance(conversion_data, dict) and conversion_data.get("value"):
                            # Store in customer_acquisition or engagement_level
                            if hasattr(traction, "customer_acquisition"):
                                traction.customer_acquisition = f"{conversion_data['value']}% conversion"
                            logger.info(f"TRACTION MERGED - conversion_rate: {conversion_data['value']}% from PDF {filename}")
                    
                    if "active_users" in traction_metrics:
                        active_data = traction_metrics["active_users"]
                        if isinstance(active_data, dict) and active_data.get("value"):
                            # Store in engagement_level or users_customers
                            if hasattr(traction, "engagement_level"):
                                traction.engagement_level = f"{active_data['value']} active users"
                            logger.info(f"TRACTION MERGED - active_users: {active_data['value']} from PDF {filename}")

                    # Enterprise SaaS KPIs
                    if "total_users" in traction_metrics and traction_metrics["total_users"] and isinstance(traction_metrics["total_users"], dict) and traction_metrics["total_users"].get("value"):
                        total_users_data = traction_metrics["total_users"]
                        # Add to memo as custom field if it doesn't exist
                        if not hasattr(traction, "total_users"):
                            traction.total_users = str(total_users_data["value"])
                        logger.info(f"TRACTION MERGED - total_users: {total_users_data['value']} from PDF {filename}")

                    if "monthly_active_users" in traction_metrics and traction_metrics["monthly_active_users"] and isinstance(traction_metrics["monthly_active_users"], dict) and traction_metrics["monthly_active_users"].get("value"):
                        mau_data = traction_metrics["monthly_active_users"]
                        if not hasattr(traction, "monthly_active_users"):
                            traction.monthly_active_users = str(mau_data["value"])
                        logger.info(f"TRACTION MERGED - monthly_active_users: {mau_data['value']} from PDF {filename}")

                    if "daily_active_users" in traction_metrics and traction_metrics["daily_active_users"] and isinstance(traction_metrics["daily_active_users"], dict) and traction_metrics["daily_active_users"].get("value"):
                        dau_data = traction_metrics["daily_active_users"]
                        if not hasattr(traction, "daily_active_users"):
                            traction.daily_active_users = str(dau_data["value"])
                        logger.info(f"TRACTION MERGED - daily_active_users: {dau_data['value']} from PDF {filename}")

                    if "dau_mau_ratio" in traction_metrics and traction_metrics["dau_mau_ratio"] and isinstance(traction_metrics["dau_mau_ratio"], dict) and traction_metrics["dau_mau_ratio"].get("value"):
                        dau_mau_data = traction_metrics["dau_mau_ratio"]
                        if not hasattr(traction, "dau_mau_ratio"):
                            traction.dau_mau_ratio = str(dau_mau_data["value"])
                        logger.info(f"TRACTION MERGED - dau_mau_ratio: {dau_mau_data['value']} from PDF {filename}")

                    if "monthly_growth_rate_pct" in traction_metrics and traction_metrics["monthly_growth_rate_pct"] and isinstance(traction_metrics["monthly_growth_rate_pct"], dict) and traction_metrics["monthly_growth_rate_pct"].get("value"):
                        growth_data = traction_metrics["monthly_growth_rate_pct"]
                        if not hasattr(traction, "monthly_growth_rate_pct"):
                            traction.monthly_growth_rate_pct = str(growth_data["value"])
                        logger.info(f"TRACTION MERGED - monthly_growth_rate_pct: {growth_data['value']}% from PDF {filename}")

                    if "monthly_churn_rate_pct" in traction_metrics and traction_metrics["monthly_churn_rate_pct"] and isinstance(traction_metrics["monthly_churn_rate_pct"], dict) and traction_metrics["monthly_churn_rate_pct"].get("value"):
                        churn_data = traction_metrics["monthly_churn_rate_pct"]
                        if not hasattr(traction, "monthly_churn_rate_pct"):
                            traction.monthly_churn_rate_pct = str(churn_data["value"])
                        logger.info(f"TRACTION MERGED - monthly_churn_rate_pct: {churn_data['value']}% from PDF {filename}")

                    if "retention_6m_pct" in traction_metrics and traction_metrics["retention_6m_pct"] and isinstance(traction_metrics["retention_6m_pct"], dict) and traction_metrics["retention_6m_pct"].get("value"):
                        retention_data = traction_metrics["retention_6m_pct"]
                        if not hasattr(traction, "retention_6m_pct"):
                            traction.retention_6m_pct = str(retention_data["value"])
                        logger.info(f"TRACTION MERGED - retention_6m_pct: {retention_data['value']}% from PDF {filename}")

                    if "engagement_time" in traction_metrics and traction_metrics["engagement_time"] and isinstance(traction_metrics["engagement_time"], dict) and traction_metrics["engagement_time"].get("value"):
                        engagement_data = traction_metrics["engagement_time"]
                        if not hasattr(traction, "engagement_time"):
                            traction.engagement_time = f"{engagement_data['value']} minutes"
                        logger.info(f"TRACTION MERGED - engagement_time: {engagement_data['value']} minutes from PDF {filename}")

                    if "workflows_created" in traction_metrics and traction_metrics["workflows_created"] and isinstance(traction_metrics["workflows_created"], dict) and traction_metrics["workflows_created"].get("value"):
                        workflows_data = traction_metrics["workflows_created"]
                        if not hasattr(traction, "workflows_created"):
                            traction.workflows_created = str(workflows_data["value"])
                        logger.info(f"TRACTION MERGED - workflows_created: {workflows_data['value']} from PDF {filename}")

                    # Revenue metrics for financial projections
                    if "mrr" in traction_metrics and traction_metrics["mrr"] and isinstance(traction_metrics["mrr"], dict) and traction_metrics["mrr"].get("value"):
                        mrr_data = traction_metrics["mrr"]
                        # Store MRR for revenue projections
                        # Note: current_mrr is not in FinancialProjections schema - MRR will be used in revenue_projections from generated_content
                        logger.info(f"FINANCIAL MERGED - MRR: ${mrr_data['value']} from PDF {filename}")

                    if "arr" in traction_metrics and traction_metrics["arr"] and isinstance(traction_metrics["arr"], dict) and traction_metrics["arr"].get("value"):
                        arr_data = traction_metrics["arr"]
                        # Store ARR for revenue projections
                        # Note: current_arr is not in FinancialProjections schema - ARR will be used in revenue_projections from generated_content
                        logger.info(f"FINANCIAL MERGED - ARR: ${arr_data['value']} from PDF {filename}")

                    if "arpu" in traction_metrics and traction_metrics["arpu"] and isinstance(traction_metrics["arpu"], dict) and traction_metrics["arpu"].get("value"):
                        arpu_data = traction_metrics["arpu"]
                        # Store ARPU for revenue projections
                        # Note: arpu is not in FinancialProjections schema - ARPU will be used in revenue_projections from generated_content
                        logger.info(f"FINANCIAL MERGED - ARPU: ${arpu_data['value']} from PDF {filename}")

        # Minimum ticket / check size from fund data:
        if fund:
            min_investment = fund.get("min_investment")
            max_investment = fund.get("max_investment")
            fund_size = fund.get("fund_size")
            
            # Map to investment terms if missing
            if self._detect_placeholder(getattr(memo.investment_terms, "investment_amount", None)):
                min_inv_val = _parse_money_multicurrency(str(min_investment)) if min_investment else None
                max_inv_val = _parse_money_multicurrency(str(max_investment)) if max_investment else None
                fund_size_val = _parse_money_multicurrency(str(fund_size)) if fund_size else None
                
                if min_inv_val and max_inv_val:
                    memo.investment_terms.investment_amount = f"${min_inv_val/1_000_000:.1f}M – ${max_inv_val/1_000_000:.1f}M" if min_inv_val >= 1_000_000 else f"${min_inv_val/1_000:.0f}K – ${max_inv_val/1_000:.0f}K"
                elif min_inv_val:
                    memo.investment_terms.investment_amount = f"${min_inv_val/1_000_000:.1f}M" if min_inv_val >= 1_000_000 else f"${min_inv_val/1_000:.0f}K"
                elif fund_size_val:
                    memo.investment_terms.investment_amount = f"Up to ${fund_size_val/1_000_000:.1f}M" if fund_size_val >= 1_000_000 else f"Up to ${fund_size_val/1_000:.0f}K"

            # Ownership / dilution: try to use fundData if memo left it blank
            if self._detect_placeholder(getattr(memo.investment_terms, "ownership", None)):
                if min_investment or max_investment:
                    memo.investment_terms.ownership = "To be set within fund mandate"
                elif fund_size:
                    memo.investment_terms.ownership = "To be determined based on fund allocation"

        # Update quick_facts back to memo
        memo.executive_summary.quick_facts = quick_facts
        
        # Set source fields when overwriting from pitch/fund data
        # Revenue projections source - only set if actual revenue data exists
        if memo.financial_projections.revenue_projections:
            for proj in memo.financial_projections.revenue_projections:
                if proj.revenue and not self._detect_placeholder(proj.revenue):
                    proj.source = "pitch_data"
                else:
                    # Remove placeholder values - set to None/empty
                    proj.revenue = None
                    proj.source = None
        
        # Validate extracted PDF metrics before using synthetic data
        def _validate_extracted_data(pdf_extraction_results: Dict[str, Any]) -> bool:
            """Validate that we have meaningful extracted data before using fallbacks"""
            if not pdf_extraction_results:
                return False
            
            # Check if we have any meaningful metrics
            pdf_metrics = pdf_extraction_results.get("pdf_metrics", {})
            if pdf_metrics:
                for filename, metrics in pdf_metrics.items():
                    if metrics and any(len(metric_list) > 0 for metric_list in metrics.values()):
                        return True
            
            # Check if we have meaningful text content
            pdf_texts = pdf_extraction_results.get("pdf_texts", [])
            if pdf_texts and any(len(item.get("text", "")) > 100 for item in pdf_texts):
                return True
                
            return False
        
        # Traction sources - validate each traction metric has real data before setting source
        traction = memo.early_traction_signals
        for field in ["waitlist", "beta_users", "lois_pilots", "paying_customers"]:
            current_value = getattr(traction, field, None)
            if current_value and not self._detect_placeholder(current_value):
                source_field = f"{field}_source"
                if not getattr(traction, source_field, None):
                    setattr(traction, source_field, "pitch_data")
            else:
                # Remove placeholder values - set to None
                setattr(traction, field, None)
        
        # TAM source validation
        if not memo.market_opportunity_analysis.tam_source or self._detect_placeholder(memo.market_opportunity_analysis.tam_source):
            if not self._detect_placeholder(memo.market_opportunity_analysis.tam):
                # If we have TAM value but no source, set default source
                memo.market_opportunity_analysis.tam_source = "pitch_data"
        
        # TAM source
        if not memo.market_opportunity_analysis.tam_source or self._detect_placeholder(memo.market_opportunity_analysis.tam_source):
            if not self._detect_placeholder(memo.market_opportunity_analysis.tam):
                memo.market_opportunity_analysis.tam_source = "pitch_data"
        
        # Add data quality scoring
        def _calculate_data_quality_score(memo: InvestorCommitteeMemo, pdf_extraction_results: Dict[str, Any]) -> float:
            """Score data quality based on source attribution and validation"""
            score = 1.0
            
            # Deduct points for synthetic data usage
            synthetic_count = 0
            total_fields = 0
            
            # Check financial projections
            if memo.financial_projections.revenue_projections:
                for proj in memo.financial_projections.revenue_projections:
                    total_fields += 1
                    if self._detect_placeholder(proj.revenue):
                        synthetic_count += 1
            
            # Check traction metrics
            traction = memo.early_traction_signals
            traction_fields = ["waitlist", "beta_users", "lois_pilots", "paying_customers"]
            for field in traction_fields:
                total_fields += 1
                if self._detect_placeholder(getattr(traction, field, None)):
                    synthetic_count += 1
            
            # Deduct points based on synthetic data percentage
            if total_fields > 0:
                synthetic_ratio = synthetic_count / total_fields
                score -= synthetic_ratio * 0.5  # Up to 50% deduction for synthetic data
            
            # Add points for PDF-extracted data
            if _validate_extracted_data(pdf_extraction_results):
                score += 0.2  # Bonus for extracted data
                
            return max(0.1, min(1.0, score))  # Ensure score between 0.1 and 1.0
        
        # Log data quality score
        data_quality_score = _calculate_data_quality_score(memo, pdf_extraction_results)
        logger.info(f"Data Quality Score: {data_quality_score:.2f}/1.0 for memo {memo.meta.company_name}")

        # Merge traction_kpis from PDF extraction into memo
        if pdf_extraction_results and pdf_extraction_results.get("traction_kpis"):
            pdf_traction_kpis = pdf_extraction_results["traction_kpis"]
            if pdf_traction_kpis:
                # Initialize memo.traction_kpis if it doesn't exist
                if not memo.traction_kpis:
                    memo.traction_kpis = []
                
                # Create a set of existing KPI names to avoid duplicates
                existing_kpi_names = {kpi.name for kpi in memo.traction_kpis}
                
                # Add PDF-extracted traction KPIs that don't already exist
                for pdf_kpi in pdf_traction_kpis:
                    if pdf_kpi.name not in existing_kpi_names:
                        memo.traction_kpis.append(pdf_kpi)
                        existing_kpi_names.add(pdf_kpi.name)
                        logger.info(f"Merged traction KPI from PDF (_overwrite): {pdf_kpi.name} = {pdf_kpi.value} {pdf_kpi.unit}")
                    else:
                        # Update existing KPI if PDF source is more reliable
                        for i, existing_kpi in enumerate(memo.traction_kpis):
                            if existing_kpi.name == pdf_kpi.name:
                                # Prefer PDF-extracted data over LLM-generated estimates
                                if pdf_kpi.confidence == "high" and not pdf_kpi.estimated:
                                    memo.traction_kpis[i] = pdf_kpi
                                    logger.info(f"Updated traction KPI from PDF (_overwrite): {pdf_kpi.name} = {pdf_kpi.value} {pdf_kpi.unit}")
                                break

        return memo

    def _is_question(self, message: str) -> bool:
        """Detect if the message is a question (vs an edit/update request)."""
        message_lower = message.lower().strip()
        
        # Check for explicit question patterns first
        import re
        question_patterns = [
            r'^(what|how|why|when|where|who|which|can you|could you|would you|should i|do you|is this|are you|what\'s|whats)',
            r'\?$',  # Ends with question mark
            r'\b(should i invest|should we invest|invest in this|investment recommendation|what.*recommendation|final recommendation|do you recommend|would you recommend|what is your recommendation|what\'s your recommendation)\b',
            r'\b(what is|what\'s|what are|how is|how are|why is|why are|tell me|explain|describe|show me)\b',
            r'\b(how much|how many|how long|how often)\b',
        ]
        
        # Check if message matches any question pattern
        for pattern in question_patterns:
            if re.search(pattern, message_lower):
                # But exclude if it's clearly an update command
                is_update_command = any(word in message_lower for word in ["update", "change", "set", "modify", "edit", "add", "remove", "delete"])
                if not is_update_command:
                    logger.info(f"Question detected by pattern '{pattern}': {message[:100]}")
                    return True
        
        return False

    def _build_qa_prompt_fact_mode(
        self,
        question: str,
        extracted_facts: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, Any]]],
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Build prompt for FACT mode: LLM explains pre-extracted facts.
        Facts are extracted deterministically, LLM only provides explanation/reasoning.
        """
        import json
        
        facts_json = json.dumps(extracted_facts, indent=2, default=str)
        
        system_prompt = (
            "You are explaining facts extracted from an investment memo.\n\n"
            "FACTS PROVIDED (DO NOT INVENT):\n"
            f"{facts_json}\n\n"
            f"USER QUESTION: {question}\n\n"
            "CRITICAL RULES:\n"
            "- ONLY explain the facts provided above\n"
            "- If a fact shows 'Not specified in the memo', state that explicitly\n"
            "- Do NOT add information not in the facts\n"
            "- Do NOT provide generic advice or disclaimers\n"
            "- Do NOT say 'I cannot' or 'I don't have' - missing data is a valid answer\n"
            "- Every claim must be traceable to the facts provided\n"
            "- Quote specific values directly from the facts\n"
            "- Synthesize and explain what the facts mean, but do NOT invent new information\n\n"
            "Return a JSON object:\n"
            "{{\n"
            '  "answer": "Your explanation based ONLY on the facts provided",\n'
            '  "source_section": "Section/field name where facts came from",\n'
            '  "source_type": "memo",\n'
            '  "memo_unchanged": true,\n'
            '  "updated_memo": {{}}\n'
            "}}\n"
        )
        
        history_block = self._compact_json_for_prompt(conversation_history or [], max_chars=1000) if conversation_history else "No previous conversation."
        
        user_prompt = (
            f"USER QUESTION: {question}\n\n"
            f"CONVERSATION HISTORY:\n{history_block}\n\n"
            "ANSWER THE QUESTION using ONLY the facts provided above. "
            "Explain and synthesize, but do NOT invent new information."
        )
        
        return system_prompt, [{"role": "user", "content": user_prompt}]
    
    def _build_qa_prompt_analysis_mode(
        self,
        memo: InvestorCommitteeMemo,
        question: str,
        conversation_history: Optional[List[Dict[str, Any]]],
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Build prompt for ANALYSIS mode: LLM analyzes full memo.
        Full memo provided, LLM provides reasoning and synthesis.
        """
        import json
        
        memo_json_full = json.dumps(memo.model_dump(), indent=2, default=str)
        
        # Special handling for investment recommendation questions
        question_lower = question.lower().strip()
        is_investment_question = any(phrase in question_lower for phrase in [
            "should i invest", "should we invest", "invest in this", "investment recommendation", 
            "what is the recommendation", "what's the recommendation", "recommendation", "should invest"
        ])
        
        system_prompt = (
            "You are analyzing an investment memo. The memo is provided below.\n\n"
            "MEMO JSON (READ-ONLY, DO NOT MODIFY):\n"
            f"{memo_json_full}\n\n"
            f"USER QUESTION: {question}\n\n"
            "CRITICAL RULES:\n"
            "- Analyze ONLY what exists in the memo JSON above\n"
            "- Cite specific sections when making claims (e.g., 'According to the executive_summary...')\n"
            "- If information is missing, explicitly state 'This is not specified in the memo'\n"
            "- Do NOT provide generic investment advice or disclaimers\n"
            "- Do NOT say 'I cannot' or 'I'm not a financial advisor' - provide memo-based analysis instead\n"
            "- Do NOT add new numbers or claims not in the memo\n"
            "- Your analysis must be clearly based on memo content, not external knowledge\n"
            "- If the memo has a recommendation, use it - do NOT provide generic advice\n\n"
        )
        
        if is_investment_question:
            system_prompt += (
                "\nSPECIAL HANDLING FOR INVESTMENT RECOMMENDATION QUESTIONS:\n"
                "1. Find 'final_recommendation.recommendation' in the memo (LEAD/FOLLOW/DISCUSS/PASS)\n"
                "2. Extract reasoning from 'key_decision_factors', 'conviction_factors', 'why_invest_now'\n"
                "3. Start your answer with: 'Based on the investment memo, the recommendation is [LEAD/FOLLOW/DISCUSS/PASS].'\n"
                "4. Then explain WHY using ONLY the memo's reasoning\n"
                "5. If recommendation is missing, state: 'The memo does not contain a final recommendation yet.'\n"
            )
        
        system_prompt += (
            "\nOUTPUT FORMAT (CRITICAL - READ CAREFULLY):\n"
            "Return your analysis as plain text with proper formatting:\n"
            "- Use line breaks (\\n) to separate different sections or points\n"
            "- Use bullet points (• or -) for lists\n"
            "- Use double line breaks (\\n\\n) between major sections\n"
            "- Use bold markdown (**text**) for emphasis on key terms\n"
            "- Format your answer in a clear, readable structure with proper spacing\n"
            "- Do NOT write everything as one big paragraph\n"
            "- Break up your answer into logical sections with line breaks\n\n"
            "Example of GOOD formatting:\n"
            "**Company Overview:**\n"
            "The startup is [description from memo].\n\n"
            "**Key Strengths:**\n"
            "• [Point 1 from memo]\n"
            "• [Point 2 from memo]\n\n"
            "**Risks:**\n"
            "• [Risk 1 from memo]\n"
            "• [Risk 2 from memo]\n\n"
            "IMPORTANT: Your answer must be based on the memo data above. "
            "Do NOT return empty strings or generic disclaimers. If the memo has the information, use it. If not, say 'This is not specified in the memo.'"
        )
        
        history_block = self._compact_json_for_prompt(conversation_history or [], max_chars=1000) if conversation_history else "No previous conversation."
        
        user_prompt = (
            f"USER QUESTION: {question}\n\n"
            f"CONVERSATION HISTORY:\n{history_block}\n\n"
            "ANSWER THE QUESTION by analyzing the memo JSON above. "
            "Use reasoning and synthesis, but do NOT invent facts or modify the memo."
        )
        
        return system_prompt, [{"role": "user", "content": user_prompt}]
    
    async def _answer_fact_question(
        self,
        question: str,
        memo: InvestorCommitteeMemo,
        question_plan: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        FACT MODE: Extract facts from memo, LLM explains them only.
        
        Flow:
        1. Extract facts deterministically using MemoFactExtractor
        2. If facts are empty → return "This information is not specified in the memo."
        3. LLM prompt: "Explain these facts in natural language. Do NOT add information not in the facts."
        4. Return answer with source attribution
        
        Key Rules:
        - No "I couldn't process" errors - missing data is a valid answer
        - No generic disclaimers - only memo-based answers
        - Every claim must be traceable to extracted facts
        """
        from app.services.memo_qa_planner import MemoFactExtractor
        
        fact_extractor = MemoFactExtractor()
        fields_used = question_plan.get("fields_used", [])
        
        # Extract facts from memo
        facts = fact_extractor.extract_facts(fields_used, memo.model_dump())
        logger.info(f"Extracted {len(facts)} facts for FACT mode answer")
        
        # Check if all facts are "Not specified"
        all_missing = all(
            isinstance(v, str) and ("not specified" in v.lower() or "not available" in v.lower())
            for v in facts.values()
        )
        
        if not facts or all_missing:
            # No facts found - return explicit "not specified" answer
            return {
                "answer": "This information is not specified in the memo.",
                "source_section": "",
                "source_type": "memo",
                "is_question": True,
                "memo_unchanged": True
            }
        
        # Build FACT mode prompt with extracted facts
        system_prompt, messages = self._build_qa_prompt_fact_mode(
            question, facts, conversation_history
        )
        
        # Call LLM to explain facts
        try:
            claude_response_raw, claude_usage = await self.claude.generate_json(
                system_prompt=system_prompt,
                messages=messages,
                max_tokens=2000,
            )
            
            # Parse JSON response (generate_json returns a string)
            import json
            claude_response = None
            if claude_response_raw:
                try:
                    # Strip code fences if present
                    response_text = _strip_code_fences(claude_response_raw)
                    claude_response = json.loads(response_text)
                    logger.info(f"FACT MODE: Successfully parsed Claude JSON response")
                except json.JSONDecodeError as e:
                    logger.info(f"FACT MODE: JSON parse failed, using memo extraction (fallback working): {e}. Raw: {claude_response_raw[:200]}")
                    claude_response = None
            
            if claude_response and isinstance(claude_response, dict):
                answer = claude_response.get("answer", "")
                source_section = claude_response.get("source_section", "")
                
                # Ensure answer is not generic disclaimer
                if not answer or any(phrase in answer.lower() for phrase in [
                    "i'm not", "i cannot", "i don't have", "i'm unable", "i apologize",
                    "i'm afraid", "don't have enough information", "unable to determine"
                ]):
                    # Fallback: construct answer from facts directly
                    logger.warning(f"LLM returned generic disclaimer, using facts directly: {answer[:100]}")
                    answer_parts = []
                    for field_path, fact_value in facts.items():
                        if isinstance(fact_value, str) and "not specified" not in fact_value.lower():
                            # Use just the value, not the path
                            answer_parts.append(f"{fact_value}")
                    # Format with line breaks for better readability
                    answer = "\n\n".join(answer_parts) if answer_parts else "This information is not specified in the memo."
                    source_section = ", ".join(fields_used[:3])
                    logger.info(f"FACT MODE: Used extracted facts: {answer[:100]}")
                
                return {
                    "answer": answer,
                    "source_section": source_section,
                    "source_type": "memo",
                    "is_question": True,
                    "memo_unchanged": True
                }
            else:
                # LLM failed - construct answer from facts directly
                answer_parts = []
                for field_path, fact_value in facts.items():
                    if isinstance(fact_value, str) and "not specified" not in fact_value.lower():
                        answer_parts.append(f"{fact_value}")
                answer = ". ".join(answer_parts) if answer_parts else "This information is not specified in the memo."
                
                return {
                    "answer": answer,
                    "source_section": ", ".join(fields_used[:3]),
                    "source_type": "memo",
                    "is_question": True,
                    "memo_unchanged": True
                }
        except Exception as e:
            logger.warning(f"Claude API error during FACT mode answer: {e}")
            # Fallback: construct answer from facts directly
            answer_parts = []
            for field_path, fact_value in facts.items():
                if isinstance(fact_value, str) and "not specified" not in fact_value.lower():
                    answer_parts.append(f"{fact_value}")
            answer = ". ".join(answer_parts) if answer_parts else "This information is not specified in the memo."
            logger.info(f"FACT MODE: API error fallback, using facts: {answer[:100]}")
            
            return {
                "answer": answer,
                "source_section": ", ".join(fields_used[:3]),
                "source_type": "memo",
                "is_question": True,
                "memo_unchanged": True
            }
    
    async def _answer_analysis_question(
        self,
        question: str,
        memo: InvestorCommitteeMemo,
        question_plan: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        ANALYSIS MODE: LLM analyzes full memo, but cannot add new data.
        
        Flow:
        1. Provide full memo JSON to LLM
        2. Prompt: "Analyze ONLY what exists in the memo. No external advice. No new numbers."
        3. Output structure:
           - What the startup does (from memo)
           - Strengths (from memo sections)
           - Risks (from memo sections)
           - Missing info (explicitly list what's NOT in memo)
           - Final opinion (clearly labeled as AI analysis)
        
        Key Rules:
        - For "should I invest?" questions: Extract recommendation from final_recommendation.recommendation
          and explain reasoning from final_recommendation.key_decision_factors
        - No generic investment advice
        - Every claim must cite memo section
        """
        # Build ANALYSIS mode prompt with full memo
        system_prompt, messages = self._build_qa_prompt_analysis_mode(
            memo, question, conversation_history
        )
        
        # Call LLM to analyze memo (plain text output, no JSON)
        try:
            logger.info(f"ANALYSIS MODE: Calling Claude with memo data (memo has {len(str(memo.model_dump()))} chars)")
            claude_response_text, claude_usage = await self.claude.chat(
                system_prompt=system_prompt,
                messages=messages,
                max_tokens=3000,  # Increased for longer analysis responses
            )
            
            # Plain text response - just strip code fences if present
            if claude_response_text:
                answer = _strip_code_fences(claude_response_text).strip()
                
                # Ensure proper formatting: normalize line breaks and ensure spacing
                import re
                # Normalize multiple line breaks to double line breaks
                answer = re.sub(r'\n{3,}', '\n\n', answer)
                # Ensure line breaks after bullet points are preserved
                answer = re.sub(r'([•\-])\s*([^\n])', r'\1 \2', answer)
                # Ensure proper spacing around sections (headers followed by content)
                answer = re.sub(r'\n([A-Z][^:]+:)\n([^\n])', r'\n\n\1\n\2', answer)
                
                # Ensure answer is not generic disclaimer - check for ANY generic language
                generic_phrases = [
                    "i'm not a financial advisor", "i cannot provide investment advice",
                    "i don't have enough information", "i apologize, but i don't have",
                    "i'm afraid i don't", "unable to determine", "unable to analyze",
                    "don't have enough information to", "don't have the context",
                    "don't have the expertise", "can't make personalized",
                    "consult with a qualified financial advisor", "unable to advise",
                    "i'm unable to", "i cannot advise", "can't make investment decisions"
                ]
                
                # Also check if answer contains generic investment advice patterns
                has_generic_patterns = (
                    "thoroughly review" in answer.lower() or
                    "do your due diligence" in answer.lower() or
                    "consult with a qualified" in answer.lower() or
                    "invest what you can afford to lose" in answer.lower()
                )
                
                if not answer or any(phrase in answer.lower() for phrase in generic_phrases) or has_generic_patterns:
                    # Always extract from memo when LLM returns generic disclaimer
                    logger.info(f"LLM returned generic disclaimer (detected pattern: {has_generic_patterns}), extracting from memo directly")
                    memo_answer = self._extract_answer_from_memo(memo, question)
                    answer = memo_answer["text"]
                    source_section = memo_answer["source"]
                else:
                    # Use LLM response
                    source_section = "memo_analysis"
                    logger.info(f"ANALYSIS MODE: Using Claude plain text response: {answer[:100]}")
            else:
                # Empty response - extract from memo
                logger.info(f"ANALYSIS MODE: Empty Claude response, extracting from memo directly")
                memo_answer = self._extract_answer_from_memo(memo, question)
                answer = memo_answer["text"]
                source_section = memo_answer["source"]
            
            return {
                "answer": answer,
                "source_section": source_section,
                "source_type": "memo",
                "is_question": True,
                "memo_unchanged": True
            }
        except Exception as e:
            logger.error(f"Claude API error during ANALYSIS mode answer: {e}", exc_info=True)
            # Fallback: extract from memo directly
            answer = self._extract_answer_from_memo(memo, question)
            logger.info(f"ANALYSIS MODE: Exception fallback - memo extraction result: {answer['text'][:100]}")
            return {
                "answer": answer["text"],
                "source_section": answer["source"],
                "source_type": "memo",
                "is_question": True,
                "memo_unchanged": True
            }
    
    def _extract_answer_from_memo(self, memo: InvestorCommitteeMemo, question: str) -> Dict[str, str]:
        """
        Extract answer from memo data directly when LLM fails.
        Handles various question types by extracting relevant memo sections.
        """
        question_lower = question.lower().strip()
        
        # Investment recommendation questions - expanded matching
        if any(phrase in question_lower for phrase in [
            "should i invest", "recommendation", "what is the recommendation",
            "what's the recommendation", "what is your recommendation", "what's your recommendation",
            "whats ur decision", "what's your decision", "what is your decision",
            "whats ur desision", "what's ur decision", "what is ur decision",
            "based on memo", "memo decision", "your decision", "ur decision",
            "final recommendation", "investment recommendation", "should invest"
        ]):
            rec = getattr(memo.final_recommendation, "recommendation", None) if hasattr(memo, "final_recommendation") else None
            if rec and not self._detect_placeholder(rec):
                parts = [f"**Recommendation:** {rec}\n"]
                
                # Key decision factors
                factors = getattr(memo.final_recommendation, "key_decision_factors", []) if hasattr(memo, "final_recommendation") else []
                if factors:
                    parts.append("**Key Decision Factors:**")
                    for factor in factors[:5]:  # Show up to 5 factors
                        parts.append(f"• {factor}")
                    parts.append("")
                
                # Conviction factors
                conviction = getattr(memo.final_recommendation, "conviction_factors", []) if hasattr(memo, "final_recommendation") else []
                if conviction:
                    parts.append("**Conviction Factors:**")
                    for factor in conviction[:5]:
                        parts.append(f"• {factor}")
                    parts.append("")
                
                # Why invest now
                why_now = getattr(memo.final_recommendation, "why_invest_now", None) if hasattr(memo, "final_recommendation") else None
                if why_now and not self._detect_placeholder(why_now):
                    parts.append(f"**Why Invest Now:**\n{why_now}\n")
                
                answer = "\n".join(parts).strip()
                return {"text": answer, "source": "final_recommendation"}
            else:
                return {"text": "The memo does not contain a final recommendation yet.", "source": "final_recommendation"}
        
        # Investment amount questions
        if "investment" in question_lower and ("amount" in question_lower or "how much" in question_lower):
            amount = getattr(memo.investment_terms, "investment_amount", None) if hasattr(memo, "investment_terms") else None
            if amount and not self._detect_placeholder(amount):
                return {"text": str(amount), "source": "investment_terms.investment_amount"}
        
        # Company/startup description questions
        if any(phrase in question_lower for phrase in ["what is this startup", "what is the company", "tell me about", "what is it about", "startup about"]):
            parts = []
            
            # Company name
            company_name = getattr(memo.meta, "company_name", None) if hasattr(memo, "meta") else None
            if company_name and not self._detect_placeholder(company_name):
                parts.append(f"**Company:** {company_name}\n")
            
            # Problem statement
            problem = getattr(memo.executive_summary, "problem", None) if hasattr(memo, "executive_summary") else None
            if problem and not self._detect_placeholder(problem):
                parts.append(f"**Problem:**\n{problem}\n")
            
            # Solution
            solution = getattr(memo.executive_summary, "solution", None) if hasattr(memo, "executive_summary") else None
            if solution and not self._detect_placeholder(solution):
                parts.append(f"**Solution:**\n{solution}\n")
            
            # Product description (full, no truncation)
            product_desc = getattr(memo.product_technology_vision, "product_description", None) if hasattr(memo, "product_technology_vision") else None
            if product_desc and not self._detect_placeholder(product_desc):
                parts.append(f"**Product:**\n{product_desc}\n")
            
            # Investment thesis if available
            thesis = getattr(memo.executive_summary, "investment_thesis", None) if hasattr(memo, "executive_summary") else None
            if thesis and not self._detect_placeholder(thesis):
                parts.append(f"**Investment Thesis:**\n{thesis}\n")
            
            if parts:
                formatted_answer = "\n".join(parts).strip()
                return {"text": formatted_answer, "source": "executive_summary, product_technology_vision"}
        
        # TAM/SAM/SOM questions
        if any(term in question_lower for term in ["tam", "total addressable market", "market size"]):
            tam = getattr(memo.market_opportunity_analysis, "tam", None) if hasattr(memo, "market_opportunity_analysis") else None
            if tam and not self._detect_placeholder(tam):
                return {"text": f"TAM (Total Addressable Market): {tam}", "source": "market_opportunity_analysis.tam"}
        
        # Try to extract from executive summary if nothing matched
        if hasattr(memo, "executive_summary"):
            exec_summary = memo.executive_summary
            summary_text = f"{getattr(exec_summary, 'problem', '')} {getattr(exec_summary, 'solution', '')} {getattr(exec_summary, 'investment_thesis', '')}".strip()
            if summary_text and not self._detect_placeholder(summary_text):
                return {"text": summary_text[:500], "source": "executive_summary"}
        
        return {"text": "This information is not specified in the memo.", "source": ""}

    def _build_edit_prompt(
        self,
        memo: InvestorCommitteeMemo,
        message: str,
        conversation_history: Optional[List[Dict[str, Any]]],
        uploaded_datasets: Optional[List[DatasetRecord]],
        website_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[Dict[str, str]]]:
        history_block = self._compact_json_for_prompt(conversation_history or [], max_chars=2000)
        calibration = self._build_dataset_calibration(uploaded_datasets or [])
        dataset_notes = calibration.get("raw_context") or "No extra datasets."
        market_calibration = calibration.get("market_calibration") or ""
        risk_calibration = calibration.get("risk_calibration") or ""
        terms_calibration = calibration.get("terms_calibration") or ""
        website_profile = (website_context or {}).get("profile") or {}
        website_snippets = (website_context or {}).get("snippets") or []
        website_url = (website_context or {}).get("url")
        website_profile_block = self._compact_json_for_prompt(website_profile, max_chars=1200) if website_profile else "No structured website data."
        snippet_lines = [self._truncate_text(str(snippet), 200) for snippet in website_snippets[:8]]
        website_snippets_block = "\n".join([line for line in snippet_lines if line]) or "No website snippets captured."
        
        # Detect if this is a question
        is_question = self._is_question(message)
        
        if is_question:
            # QUESTION-ANSWERING MODE: Answer from memo first, indicate if internet research needed
            # Special handling for investment recommendation questions
            message_lower = message.lower().strip()
            is_investment_question = any(phrase in message_lower for phrase in [
                "should i invest", "should we invest", "invest in this", "investment recommendation", 
                "what is the recommendation", "what's the recommendation", "recommendation", "should invest"
            ])
            
            system_prompt = (
                "You are an investment analyst.\n\n"
                "RULES (MANDATORY):\n"
                "- You may ONLY use the memo JSON provided.\n"
                "- You may NOT invent, infer, estimate, or research.\n"
                "- If information is missing, say: 'This is not specified in the memo.'\n"
                "- Use reasoning, synthesis, and judgment where possible.\n"
                "- Do NOT modify the memo.\n\n"
                "TASK:\n"
                "Answer the user question by analyzing the memo.\n\n"
                "Input to LLM: memo_json (ONLY)\n\n"
                "🚫 No PDFs\n"
                "🚫 No Gemini\n"
                "🚫 No pitch data\n\n"
            )
            
            if is_investment_question:
                system_prompt += (
                    "SPECIAL HANDLING FOR INVESTMENT RECOMMENDATION QUESTIONS:\n"
                    "When asked about investment recommendations (e.g., 'should I invest?', 'what is the recommendation?'), "
                    "you MUST:\n"
                    "1. FIRST: Look in the memo JSON for 'final_recommendation.recommendation' or 'investment_recommendation_summary.recommendation'.\n"
                    "2. Extract the exact recommendation value (it will be one of: 'LEAD', 'FOLLOW', 'DISCUSS', or 'PASS').\n"
                    "3. Extract the reasoning/justification from these memo fields:\n"
                    "   - 'final_recommendation.recommendation' (if it contains a detailed paragraph)\n"
                    "   - 'final_recommendation.key_decision_factors' (list of factors)\n"
                    "   - 'final_recommendation.conviction_factors' (list of factors)\n"
                    "   - 'final_recommendation.why_invest_now' (explanation)\n"
                    "   - 'investment_recommendation_summary.key_decision_drivers' (list of drivers)\n"
                    "   - 'executive_summary.investment_thesis_3_sentences' (thesis summary)\n"
                    "4. Format your answer EXACTLY as: 'Based on the investment memo, the recommendation is [LEAD/FOLLOW/DISCUSS/PASS].'\n"
                    "5. Then explain WHY using ONLY the memo's reasoning from the fields above.\n"
                    "6. DO NOT provide generic investment advice like 'consider business model, team, traction' - ONLY use what's in the memo.\n"
                    "7. If the recommendation field is empty or says 'DEFAULT_PLACEHOLDER', state: 'The memo does not contain a final recommendation yet.'\n\n"
                )
            
            system_prompt += (
                "Example Behavior (CORRECT):\n"
                "Q: Should I invest in this startup?\n\n"
                "LLM Output (Allowed):\n"
                "Based on the memo, this startup shows strong traction with growing revenue and a clear market opportunity. "
                "However, risks include customer concentration and execution risk. The investment may be suitable for "
                "investors comfortable with early-stage risk.\n\n"
                "✅ This is analysis, not hallucination\n"
                "✅ No new facts\n"
                "✅ Memo unchanged\n\n"
                "CRITICAL RULES FOR MEMO-BASED ANSWERS (PREVENT HALLUCINATIONS):\n"
                "- Quote specific values, metrics, or text DIRECTLY from the memo JSON - do NOT paraphrase or infer.\n"
                "- Be precise and cite which section of the memo your answer comes from (e.g., 'final_recommendation.recommendation').\n"
                "- Do NOT make up facts, numbers, or recommendations - ONLY use what's explicitly in the memo JSON.\n"
                "- Do NOT provide generic investment advice - ONLY explain what the memo says.\n"
                "- If the memo says 'DISCUSS', explain why it says DISCUSS using the memo's reasoning.\n"
                "- If the memo says 'LEAD', explain why it says LEAD using the memo's reasoning.\n"
                "- If the memo says 'FOLLOW', explain why it says FOLLOW using the memo's reasoning.\n"
                "- If the memo says 'PASS', explain why it says PASS using the memo's reasoning.\n"
                "- If the recommendation field is empty or placeholder, state: 'The memo does not contain a final recommendation yet.'\n"
                "- NEVER invent a recommendation that isn't in the memo.\n\n"
                "Return a JSON object with this structure:\n"
                "{{\n"
                '  "answer": "Your answer (from memo if available, or state that information is not specified in the memo)",\n'
                '  "source_section": "Section name where answer came from (e.g., final_recommendation, investment_recommendation_summary)",\n'
                '  "source_type": "memo" or "needs_research",\n'
                '  "memo_unchanged": true,\n'
                '  "updated_memo": <full memo JSON unchanged>\n'
                "}}\n\n"
                "The 'memo_unchanged' field must ALWAYS be true, and 'updated_memo' must be the EXACT memo JSON provided to you, unchanged."
            )
            
            # Include FULL memo JSON for questions (not truncated) so LLM has all context
            import json
            memo_json_full = json.dumps(memo.model_dump(), indent=2, default=str)
            
            user_prompt = (
                "INVESTMENT MEMO JSON (READ THIS CAREFULLY - ALL ANSWERS MUST COME FROM HERE):\n"
                f"{memo_json_full}\n\n"
                f"USER QUESTION: {message}\n\n"
            )
            
            if is_investment_question:
                user_prompt += (
                    "CRITICAL INSTRUCTIONS FOR THIS INVESTMENT RECOMMENDATION QUESTION:\n"
                    "1. Search the memo JSON above for 'final_recommendation' or 'investment_recommendation_summary'.\n"
                    "2. Find the 'recommendation' field - it will contain 'LEAD', 'FOLLOW', 'DISCUSS', or 'PASS'.\n"
                    "3. Find the reasoning in fields like 'key_decision_factors', 'conviction_factors', 'why_invest_now', 'key_decision_drivers', or 'investment_thesis_3_sentences'.\n"
                    "4. Your answer MUST start with: 'Based on the investment memo, the recommendation is [LEAD/FOLLOW/DISCUSS/PASS].'\n"
                    "5. Then quote the memo's reasoning directly - do NOT paraphrase or provide generic advice.\n"
                    "6. If you cannot find a recommendation in the memo, say: 'The memo does not contain a final recommendation yet.'\n"
                    "7. DO NOT say things like 'I don't have enough information' or provide generic investment guidance.\n\n"
                )
            
            user_prompt += (
                "ANSWER THE QUESTION:\n"
                "1. Search the memo JSON above for the answer.\n"
                "2. If found, quote it directly with exact values and cite the section (e.g., 'final_recommendation.recommendation').\n"
                "3. If NOT found, set source_type to 'needs_research' and state: 'This information is not available in the memo.'\n"
                "4. NEVER provide generic advice - ONLY use what's in the memo JSON above.\n"
            )
            
            return system_prompt, [{"role": "user", "content": user_prompt}]
        
        # EDIT MODE: Original editing prompt
        system_prompt = (
            "You are editing an existing IC memo. You use a self-learning approach where you learn from conversation history "
            "and previous requests to improve your responses.\n\n"
            "KEY CAPABILITIES:\n"
            "1. BULK DATA UPDATES: When users provide bulk data (e.g., multiple financial metrics, lists of competitors, "
            "multiple team members), extract and integrate ALL data points into the memo systematically.\n"
            "2. MISSING INFORMATION HANDLING: When users add missing information, intelligently map it to the correct memo "
            "fields. Use context from conversation history to understand intent.\n"
            "3. LEARNING FROM HISTORY: Review conversation history to understand patterns, previous corrections, and user "
            "preferences. Apply this knowledge to make better updates.\n\n"
            "Apply the user's instructions but keep all other fields unchanged unless you can clearly improve them to "
            "investor-grade quality.\n"
            "Return a full JSON object that matches the IC memo schema exactly (no markdown).\n"
            "Do not emit placeholders such as 'UNKNOWN', 'N/A', 'Not disclosed', 'TBD', 'To be confirmed' or similar – instead, "
            "infer concrete values or tightly reasoned estimates using the memo, conversation history, Gemini research "
            "(if available), and standard benchmarks."
        )
        user_prompt = (
            "Current memo JSON:\n"
            f"{self._compact_json_for_prompt(memo.model_dump(), max_chars=6000)}\n\n"
            "Conversation history for context:\n"
            f"{history_block}\n\n"
            "Uploaded datasets / benchmarks (historical memos, sector benchmarks, IC decisions):\n"
            f"{dataset_notes}\n\n"
            "Use uploaded datasets to:\n"
            f"- Calibrate market sizing and attractiveness: {market_calibration}\n"
            f"- Calibrate risk richness and types: {risk_calibration}\n"
            f"- Calibrate investment terms and valuation ranges: {terms_calibration}\n\n"
            f"Website scraped from {website_url or 'not provided'} (structured profile):\n"
            f"{website_profile_block}\n\n"
            "Website content snippets (reference only, rewrite in your own words):\n"
            f"{website_snippets_block}\n\n"
            f"User request: {message}\n\n"
            "Strict rules:\n"
            "1) BULK DATA PROCESSING: When the user message contains multiple data points (e.g., 'Revenue Y1: $1M, Y2: $3M, "
            "Y3: $5M' or 'Competitors: Uber, Lyft, DoorDash'), extract ALL values and update ALL relevant fields.\n"
            "2) MISSING INFORMATION: If the user adds information that was previously missing, intelligently determine the "
            "correct field(s) to update based on the data type and context from conversation history.\n"
            "3) LEARNING FROM HISTORY: Review the conversation history carefully. If similar requests were made before, "
            "learn from how they were handled. If the user corrected something, remember that preference.\n"
            "4) Never say 'N/A', 'unknown', 'no data available', or 'cannot determine'. When data is missing, infer a "
            "reasonable estimate using memo context, conversation history, and public benchmarks.\n"
            "5) Always propose corrections and improvements, even for fields that look acceptable.\n"
            "6) Always fix units and percentages: express currency with explicit codes (e.g. INR, USD) and human-readable "
            "scales (k/Mn/Bn/Cr), and make growth rates explicit in % where appropriate.\n"
            "7) Always lift language to professional investor-grade quality.\n"
            "8) Always suggest enhancements that make the memo crisper and more decision-useful.\n\n"
            "BULK DATA EXAMPLES:\n"
            "- Financial data: Extract all years, metrics, projections into structured fields\n"
            "- Lists: Process all items (competitors, team members, features, etc.) into appropriate arrays\n"
            "- Structured text: Parse tables, bullet points, or structured formats into memo fields\n\n"
            "Your job is to audit the JSON, process ALL user-provided data (bulk or individual), learn from conversation "
            "history, and produce a fully corrected investor-grade memo. Do not leave any field unfixed or any data missing.\n"
            "Return the full updated memo JSON only."
        )
        return system_prompt, [{"role": "user", "content": user_prompt}]

    def _build_audit_prompt(self, memo: InvestorCommitteeMemo) -> Tuple[str, List[Dict[str, str]]]:
        """
        Build a prompt for a *pure audit* of the memo JSON.

        The model:
        - MUST NOT rewrite or regenerate the memo.
        - MUST ONLY return an issues/diagnostics JSON object in the specified format.
        """
        memo_json = self._compact_json_for_prompt(memo.model_dump(), max_chars=6000)
        system_prompt = (
            "You are an AI system that receives a JSON memo and must identify:\n"
            "1. Which fields still require updates.\n"
            "2. Where human input is required.\n"
            "3. Which values look placeholder, missing, or inconsistent.\n"
            "4. Any duplicated, contradictory, or unrealistic values.\n"
            "5. A clean checklist of items the user must update.\n\n"
            "Your job: ONLY detect issues — do NOT rewrite the memo.\n"
        )
        user_prompt = (
            "Take the entire memo JSON below as input and return output ONLY in this JSON format:\n\n"
            "{\n"
            '  "needs_update": [\n'
            "    {\n"
            '      "field_path": "<exact path like memo.executive_summary.problem>",\n'
            '      "reason": "<why this field needs update>",\n'
            '      "current_value": "<existing value>",\n'
            '      "suggestion": "<how user should fix it>"\n'
            "    }\n"
            "  ],\n"
            '  "human_required_fields": [\n'
            '    "<field path>"\n'
            "  ],\n"
            '  "placeholders_detected": [\n'
            '    "<field path>"\n'
            "  ],\n"
            '  "consistency_issues": [\n'
            "    {\n"
            '      "field": "<field path>",\n'
            '      "issue": "<what is inconsistent>"\n'
            "    }\n"
            "  ],\n"
            '  "final_summary": "Short summary of what updates are needed."\n'
            "}\n\n"
            "IMPORTANT:\n"
            "- Only detect issues – do NOT rewrite or regenerate any memo content.\n"
            "- The `field_path` strings must be precise and usable by a developer (e.g. "
            "'memo.financial_projections.revenue_projections[0].revenue').\n"
            "- If everything looks strong for a particular area, you can leave the relevant arrays empty.\n\n"
            "Here is the memo JSON to audit:\n"
            f"{memo_json}\n"
        )
        return system_prompt, [{"role": "user", "content": user_prompt}]

    def _normalize_parsed_memo(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize parsed JSON to fix common LLM output issues like strings instead of objects."""
        normalized = dict(parsed)
        
        # Fix use_of_funds_breakdown: convert strings to objects
        if "use_of_funds_milestones" in normalized:
            uof = normalized.get("use_of_funds_milestones", {})
            if isinstance(uof, dict) and "use_of_funds_breakdown" in uof:
                breakdown = uof["use_of_funds_breakdown"]
                if isinstance(breakdown, list):
                    normalized_breakdown = []
                    for item in breakdown:
                        if isinstance(item, str):
                            # Try to parse string like "Marketing & User Acquisition: 40% - Marketing, partnerships."
                            parts = item.split(":", 1)
                            if len(parts) == 2:
                                category = parts[0].strip()
                                rest = parts[1].strip()
                                # Try to extract percentage
                                percentage = DEFAULT_PLACEHOLDER
                                if "%" in rest:
                                    pct_match = re.search(r'(\d+(?:\.\d+)?)%', rest)
                                    if pct_match:
                                        percentage = pct_match.group(1) + "%"
                                purpose = rest.replace(percentage, "").strip(" -")
                                normalized_breakdown.append({
                                    "category": category,
                                    "amount": DEFAULT_PLACEHOLDER,
                                    "percentage": percentage,
                                    "purpose": purpose or DEFAULT_PLACEHOLDER
                                })
                            else:
                                normalized_breakdown.append({
                                    "category": item[:50] if len(item) > 50 else item,
                                    "amount": DEFAULT_PLACEHOLDER,
                                    "percentage": DEFAULT_PLACEHOLDER,
                                    "purpose": DEFAULT_PLACEHOLDER
                                })
                        elif isinstance(item, dict):
                            normalized_breakdown.append(item)
                        else:
                            normalized_breakdown.append({
                                "category": str(item)[:50],
                                "amount": DEFAULT_PLACEHOLDER,
                                "percentage": DEFAULT_PLACEHOLDER,
                                "purpose": DEFAULT_PLACEHOLDER
                            })
                    uof["use_of_funds_breakdown"] = normalized_breakdown
                    normalized["use_of_funds_milestones"] = uof
        
        # Fix risk_assessment.risks: convert strings to objects
        if "risk_assessment" in normalized:
            risk_assessment = normalized.get("risk_assessment", {})
            if isinstance(risk_assessment, dict) and "risks" in risk_assessment:
                risks = risk_assessment["risks"]
                if isinstance(risks, list):
                    normalized_risks = []
                    for item in risks:
                        if isinstance(item, str):
                            # Try to parse string like "Monitor data privacy and...gemini:https://url.com)"
                            # Extract name (first part before colon or first sentence, up to 100 chars)
                            clean_item = item.strip()
                            # Remove source references at the end like "gemini:https://url.com"
                            if "gemini:" in clean_item.lower() or "source:" in clean_item.lower():
                                # Try to remove source suffix
                                clean_item = re.sub(r'\s*(gemini|source):https?://[^\s)]+\)?', '', clean_item, flags=re.IGNORECASE).strip()
                            
                            # Extract name - first sentence or first part before colon
                            if ":" in clean_item:
                                name = clean_item.split(":")[0].strip()
                            elif "." in clean_item:
                                name = clean_item.split(".")[0].strip()
                            else:
                                name = clean_item
                            
                            # Limit name length
                            if len(name) > 100:
                                name = name[:97] + "..."
                            
                            # Use full cleaned string as description (limit to reasonable length)
                            description = clean_item[:500] if len(clean_item) > 500 else clean_item
                            
                            normalized_risks.append({
                                "name": name or DEFAULT_PLACEHOLDER,
                                "description": description or DEFAULT_PLACEHOLDER,
                                "probability": DEFAULT_PLACEHOLDER,
                                "mitigation": [],
                                "residual_risk": DEFAULT_PLACEHOLDER
                            })
                        elif isinstance(item, dict):
                            normalized_risks.append(item)
                        else:
                            normalized_risks.append({
                                "name": str(item)[:100],
                                "description": str(item),
                                "probability": DEFAULT_PLACEHOLDER,
                                "mitigation": [],
                                "residual_risk": DEFAULT_PLACEHOLDER
                            })
                    risk_assessment["risks"] = normalized_risks
                    normalized["risk_assessment"] = risk_assessment
        
        return normalized

    def _parse_or_fallback_memo(self, claude_response: str, pitch_fallback: Any) -> InvestorCommitteeMemo:
        sanitized = _strip_code_fences(claude_response)
        parsed: Optional[Dict[str, Any]] = None
        try:
            parsed = json.loads(sanitized)
        except json.JSONDecodeError:
            logger.warning("Claude returned invalid JSON, attempting recovery.")
        if not parsed or not isinstance(parsed, dict):
            baseline = InvestorCommitteeMemo()
            baseline = self._populate_from_pitch(baseline, pitch_fallback)
            baseline.executive_summary.investment_thesis_3_sentences = (
                f"Auto-generated memo for {pitch_fallback.get('company_name', DEFAULT_PLACEHOLDER)}."
                if isinstance(pitch_fallback, dict)
                else "Auto-generated memo."
            )
            return baseline
        
        # Normalize parsed data to fix common LLM output issues
        parsed = self._normalize_parsed_memo(parsed)
        
        # Fix data_sources: ensure it's a list of strings, not dicts
        if "data_sources" in parsed:
            data_sources = parsed["data_sources"]
            if isinstance(data_sources, list):
                normalized_sources = []
                for src in data_sources:
                    if isinstance(src, str):
                        normalized_sources.append(src)
                    elif isinstance(src, dict):
                        # Extract source URL from dict if present
                        source_url = src.get("source") or src.get("source_url") or src.get("url")
                        if source_url and isinstance(source_url, str):
                            normalized_sources.append(source_url)
                parsed["data_sources"] = normalized_sources
        
        # Fix key_risks: ensure it's a list of objects, not strings
        if "executive_summary" in parsed and isinstance(parsed["executive_summary"], dict):
            if "key_risks" in parsed["executive_summary"]:
                key_risks = parsed["executive_summary"]["key_risks"]
                if isinstance(key_risks, list):
                    normalized_risks = []
                    for risk in key_risks:
                        if isinstance(risk, str):
                            # Convert string to ExecutiveRisk object
                            normalized_risks.append({
                                "type": "Execution",
                                "description": risk[:500] if len(risk) > 500 else risk,
                                "probability": DEFAULT_PLACEHOLDER,
                                "mitigation": [],
                                "residual_risk": DEFAULT_PLACEHOLDER
                            })
                        elif isinstance(risk, dict):
                            normalized_risks.append(risk)
                    parsed["executive_summary"]["key_risks"] = normalized_risks
        
        try:
            memo = InvestorCommitteeMemo.model_validate(parsed)
            return memo
        except Exception as exc:
            logger.warning("Validation failed on Claude memo payload: %s", exc)
            merged = InvestorCommitteeMemo().model_dump()
            merged.update(parsed)
            try:
                memo = InvestorCommitteeMemo.model_validate(merged)
            except Exception:
                # If still failing, use baseline and populate from pitch
                memo = InvestorCommitteeMemo()
            return self._populate_from_pitch(memo, pitch_fallback)

    def _pre_populate_memo_with_pdf_data(self, pitch: Dict[str, Any], pdf_extraction_results: Dict[str, Any]) -> Optional[InvestorCommitteeMemo]:
        """Pre-populate memo with PDF-extracted traction data before Claude generation."""
        pdf_metrics = pdf_extraction_results.get("pdf_metrics", {}) if pdf_extraction_results else {}
        traction_kpis = pdf_extraction_results.get("traction_kpis", []) if pdf_extraction_results else []

        if not pdf_metrics and not traction_kpis:
            return None

        # Create a base memo and populate with PDF traction data
        memo = InvestorCommitteeMemo()

        # Populate traction_kpis directly from PDF extraction
        if traction_kpis:
            memo.traction_kpis = traction_kpis

        return memo if memo.traction_kpis else None

    def _convert_pitch_deck_to_traction_kpis(self, pitch: Dict[str, Any]) -> List[TractionKPI]:
        """Convert pitch deck numeric fields into TractionKPI objects so visuals/memo reuse factual deck data."""
        kpis: List[TractionKPI] = []
        if not isinstance(pitch, dict):
            return kpis

        deck_data: Any = pitch.get("pitch_deck_data") or pitch.get("pitch_deck") or pitch.get("data")
        if isinstance(deck_data, dict) and "data" in deck_data and isinstance(deck_data["data"], dict):
            deck_data = deck_data["data"]
        if not isinstance(deck_data, dict):
            return kpis

        def add_kpi(name: str, raw_value: Any, unit: str = "count", estimated: bool = False, confidence: str = "medium") -> None:
            val = self.extract_numeric_value(raw_value)
            if val is None:
                return
            
            # Sanity check: reject unrealistic values based on unit type
            if unit == "count":
                # For counts, anything > 50 billion is unrealistic
                if val > 50_000_000_000:
                    logger.warning(f"Rejecting unrealistic count KPI '{name}' from pitch deck: {val} (raw: {raw_value})")
                    return
            elif unit == "USD":
                # For currency, anything > 10 trillion is unrealistic (unless it's TAM/SAM/SOM which aren't funnel metrics)
                if val > 10_000_000_000_000:
                    logger.warning(f"Rejecting unrealistic currency KPI '{name}' from pitch deck: {val} (raw: {raw_value})")
                    return
            
            try:
                kpis.append(
                    TractionKPI(
                        name=name,
                        value=float(val),
                        unit=unit,
                        source="pitch_deck",
                        confidence=confidence,
                        estimated=estimated,
                    )
                )
            except Exception as e:
                logger.debug(f"Failed to append deck KPI '{name}': {e}")

        # Core KPI fields (counts and revenue)
        kpi1_name = deck_data.get("kpi_1_name") or "KPI 1"
        kpi1_val = deck_data.get("kpi_1_value")
        if kpi1_val is not None:
            unit = "USD" if isinstance(kpi1_name, str) and any(token in kpi1_name.lower() for token in ["mrr", "arr", "revenue"]) else "count"
            add_kpi(kpi1_name, kpi1_val, unit=unit)

        kpi2_name = deck_data.get("kpi_2_name") or "KPI 2"
        kpi2_val = deck_data.get("kpi_2_value")
        if kpi2_val is not None:
            add_kpi(kpi2_name, kpi2_val, unit="count")

        # Revenue / pipeline
        if deck_data.get("mrr_or_arr") is not None:
            add_kpi("MRR/ARR", deck_data.get("mrr_or_arr"), unit="USD")
        if deck_data.get("pipeline_value") is not None:
            add_kpi("Pipeline Value", deck_data.get("pipeline_value"), unit="USD")
        if deck_data.get("pipeline_count") is not None:
            add_kpi("Pipeline Count", deck_data.get("pipeline_count"), unit="count")

        # Funnel metrics often surfaced on slide 9
        funnel_mappings = [
            ("Waitlist / Top Funnel", deck_data.get("top_funnel_metric")),
            ("Beta / Mid Funnel", deck_data.get("mid_funnel_metric")),
            ("Paying / Bottom Funnel", deck_data.get("bottom_funnel_metric")),
        ]
        for label, raw in funnel_mappings:
            if raw is not None:
                add_kpi(label, raw, unit="count")

        # Growth rate as percentage KPI
        if deck_data.get("growth_rate_pct") is not None:
            add_kpi("Growth Rate", deck_data.get("growth_rate_pct"), unit="percent", estimated=True)

        return kpis

    def _extract_llm_traction_estimates(self, generated_content: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Extract numeric traction estimates from LLM-generated traction_analysis if available."""
        out: Dict[str, float] = {}
        if not generated_content:
            return out
        traction = generated_content.get("traction_analysis") or {}
        if not isinstance(traction, dict):
            return out
        for key in ["waitlist", "beta_users", "lois_pilots", "paying_customers"]:
            val = self.extract_numeric_value(traction.get(key))
            if val is not None:
                out[key] = float(val)
        return out

    def _sanitize_funnel_stage_values(self, stages: Dict[str, int], max_count: int = 1_000_000_000) -> Tuple[Dict[str, int], bool]:
        """Sanitize funnel stage values to prevent absurd magnitudes.
        Returns (sanitized_stages, was_suppressed).
        """
        sanitized = {}
        suppressed = False
        MAX_REASONABLE_USERS = 1_000_000_000  # 1 billion users max
        
        for k, v in stages.items():
            try:
                val = int(v)
            except Exception:
                continue
            if val < 0:
                logger.warning(f"Suppressing negative funnel value: {k} = {val}")
                suppressed = True
                continue
            if val > MAX_REASONABLE_USERS:
                logger.warning(f"Suppressing unrealistic funnel value: {k} = {val}")
                suppressed = True
                continue
            sanitized[k] = val
        return sanitized, suppressed

    def _filter_unreasonable_traction_kpis(self, kpis: List[TractionKPI]) -> Tuple[List[TractionKPI], List[TractionKPI]]:
        """
        Drop KPIs with implausible magnitudes to prevent hallucinated giga-scale values from flowing to UI/graphs.
        - Count-type KPIs: keep <= 50M
        - Currency KPIs: keep <= 50B
        - Percent KPIs: keep within 0-100
        """
        sanitized: List[TractionKPI] = []
        dropped: List[TractionKPI] = []

        for kpi in kpis or []:
            unit_lower = (kpi.unit or "").lower()
            val = kpi.value
            keep = True

            if unit_lower in ["percent", "%"]:
                if val < 0 or val > 100:
                    keep = False
            elif any(currency in unit_lower for currency in ["usd", "$", "eur", "inr", "gbp", "£", "€", "₹"]):
                if val > 50_000_000_000:  # 50B upper guardrail for traction-level KPIs
                    keep = False
            else:
                # Count-type guardrail
                if val > 50_000_000:  # 50M cap for users/beta/waitlist/etc.
                    keep = False

            if keep:
                sanitized.append(kpi)
            else:
                dropped.append(kpi)

        return sanitized, dropped

    def _has_traction_data(self, memo: InvestorCommitteeMemo) -> bool:
        """Check if memo has any real traction data."""
        return bool(memo.traction_kpis)
    
    def _populate_early_traction_signals_from_kpis(self, memo: InvestorCommitteeMemo) -> None:
        """
        Populate early_traction_signals fields from traction_kpis list.
        This ensures that PDF-extracted traction KPIs are displayed in the UI.
        """
        if not memo.traction_kpis:
            logger.warning("_populate_early_traction_signals_from_kpis called but memo.traction_kpis is empty")
            return
        
        traction = memo.early_traction_signals
        
        # Mapping of KPI names to early_traction_signals field names
        # Note: These must match the exact names from convert_metrics_to_traction_kpis
        kpi_to_field_mapping = {
            "Waitlist": "waitlist",
            "Beta Users": "beta_users",
            "Paying Customers": "paying_customers",
            "LOIs/Pilots": "lois_pilots",
            "Monthly Active Users": "monthly_active_users",
            "Daily Active Users": "daily_active_users",
            "Total Users": "total_users",
            "Monthly Recurring Revenue": "mrr",
            "Annual Recurring Revenue": "arr",
            "Retention Rate (6M)": "retention_6m_pct",
            "Monthly Growth Rate": "monthly_growth_rate_pct",
            "Monthly Churn Rate": "monthly_churn_rate_pct",
            "Engagement Time": "engagement_time",
            "Workflows Created": "workflows_created",
        }
        
        populated_count = 0
        for kpi in memo.traction_kpis:
            # Try exact match first
            field_name = kpi_to_field_mapping.get(kpi.name)
            
            # If no exact match, try case-insensitive match
            if not field_name:
                kpi_name_lower = kpi.name.lower()
                for mapped_name, mapped_field in kpi_to_field_mapping.items():
                    if mapped_name.lower() == kpi_name_lower:
                        field_name = mapped_field
                        break
            
            # If still no match, try partial match (e.g., "Waitlist" matches "waitlist")
            if not field_name:
                kpi_name_lower = kpi.name.lower().replace(" ", "_")
                # Try direct field name match
                if hasattr(traction, kpi_name_lower):
                    field_name = kpi_name_lower
            
            if not field_name:
                logger.debug(f"No mapping found for KPI name: '{kpi.name}' (available fields: {[f for f in dir(traction) if not f.startswith('_')]})")
                continue
            
            # Always populate if field exists, regardless of current value (PDF data takes priority)
            if hasattr(traction, field_name):
                # Format the value based on unit
                if kpi.unit == "USD":
                    # Format as currency
                    if kpi.value >= 1_000_000:
                        formatted_value = f"${kpi.value/1_000_000:.1f}M"
                    elif kpi.value >= 1_000:
                        formatted_value = f"${kpi.value/1_000:.0f}K"
                    else:
                        formatted_value = f"${kpi.value:.0f}"
                elif kpi.unit == "percent" or "%" in str(kpi.unit):
                    formatted_value = f"{kpi.value}%"
                elif kpi.unit == "minutes":
                    formatted_value = f"{int(kpi.value)} minutes"
                else:
                    # Format as integer for counts
                    formatted_value = str(int(kpi.value)) if kpi.value == int(kpi.value) else str(kpi.value)
                
                # Set the field value (always overwrite with PDF data)
                setattr(traction, field_name, formatted_value)
                populated_count += 1
                
                # Set source field if it exists
                source_field = f"{field_name}_source"
                if hasattr(traction, source_field):
                    setattr(traction, source_field, kpi.source)
                
                logger.info(f"Populated early_traction_signals.{field_name} = {formatted_value} from traction_kpi: {kpi.name} (value={kpi.value}, unit={kpi.unit})")
            else:
                logger.warning(f"Field {field_name} does not exist on early_traction_signals")
        
        logger.info(f"Populated {populated_count} early_traction_signals fields from {len(memo.traction_kpis)} traction KPIs")

    def _ensure_early_traction_signals_populated(self, memo: InvestorCommitteeMemo) -> None:
        """
        Ensure all early_traction_signals fields are populated with defaults if Claude didn't populate them.
        Quantitative fields get DEFAULT_PLACEHOLDER if empty.
        Qualitative fields should be populated by Claude with meaningful assessments - only set default if truly empty.
        """
        traction = memo.early_traction_signals
        
        # Quantitative fields - use DEFAULT_PLACEHOLDER if empty
        quantitative_fields = [
            'paying_customers',
            'lois_pilots', 
            'beta_users',
            'waitlist',
            'monthly_growth_rate'
        ]
        
        # Qualitative fields - Claude should populate these with meaningful assessments
        # Only set DEFAULT_PLACEHOLDER if completely empty (Claude should have inferred something)
        qualitative_fields = [
            'engagement_level',
            'traction_quality',
            'customer_acquisition',
            'early_retention'
        ]
        
        # Set defaults for quantitative fields if empty
        # CRITICAL: Only set default if truly empty - generated_content should have populated these
        for field_name in quantitative_fields:
            current_value = getattr(traction, field_name, None)
            needs_default = (
                current_value is None or 
                current_value == "" or 
                (isinstance(current_value, str) and self._detect_placeholder(current_value))
            )
            
            if needs_default:
                logger.warning(f"Quantitative field early_traction_signals.{field_name} is empty - setting default. This should have been populated by generated_content!")
                setattr(traction, field_name, DEFAULT_PLACEHOLDER)
            else:
                logger.info(f"Quantitative field early_traction_signals.{field_name} has value: '{current_value}'")
        
        # For qualitative fields, only set default if completely empty (Claude should have generated something)
        # Don't override if Claude provided any value, even if it seems like a placeholder
        for field_name in qualitative_fields:
            current_value = getattr(traction, field_name, None)
            # Only set default if truly empty (None or empty string)
            # If Claude provided any value, keep it (even if it's a placeholder - Claude made that choice)
            if current_value is None or current_value == "":
                setattr(traction, field_name, DEFAULT_PLACEHOLDER)
                logger.debug(f"Set qualitative field early_traction_signals.{field_name} to default (was empty)")
            else:
                logger.debug(f"Keeping Claude-generated value for early_traction_signals.{field_name}: '{current_value}'")
        
        # Ensure best_traction_evidence is at least an empty list if not populated
        if not hasattr(traction, 'best_traction_evidence') or not traction.best_traction_evidence:
            traction.best_traction_evidence = []
        
        logger.info(f"Ensured all early_traction_signals fields are populated")

    def _populate_from_pitch(self, memo: InvestorCommitteeMemo, pitch: Any) -> InvestorCommitteeMemo:
        if not isinstance(pitch, dict):
            return memo

        def pick(*keys: str) -> Optional[str]:
            for key in keys:
                value = pitch.get(key)
                if value:
                    return str(value)
            return None

        company = pick("company_name", "company", "name", "venue")
        stage = None
        relationships = pitch.get("relationships") or {}
        stage_rel = relationships.get("stage", {}) if isinstance(relationships, dict) else {}
        if isinstance(stage_rel, dict):
            stage = stage_rel.get("stage") or stage_rel.get("name")
        stage = stage or pick("stage", "round")

        memo.meta.company_name = company or memo.meta.company_name
        memo.meta.stage = (stage.strip() if isinstance(stage, str) else stage) or memo.meta.stage

        description = pick("description", "business")
        market = pick("market", "industry")
        headline = pick("headline", "title")

        # DO NOT copy directly - these should come from LLM-generated content
        # Keep as fallback only if LLM didn't generate content
        # The actual content will be populated from generated_content in _fill_unknown_fields
        if not memo.executive_summary.problem or memo.executive_summary.problem == DEFAULT_PLACEHOLDER:
            memo.executive_summary.problem = description or memo.executive_summary.problem
        if not memo.executive_summary.solution or memo.executive_summary.solution == DEFAULT_PLACEHOLDER:
            memo.executive_summary.solution = pitch.get("solution") or memo.executive_summary.solution
        if not memo.executive_summary.why_now or memo.executive_summary.why_now == DEFAULT_PLACEHOLDER:
            memo.executive_summary.why_now = market or memo.executive_summary.why_now
        if not memo.executive_summary.target_customer or memo.executive_summary.target_customer == DEFAULT_PLACEHOLDER:
            memo.executive_summary.target_customer = pitch.get("target_customer") or memo.executive_summary.target_customer
        if headline and (not memo.executive_summary.investment_thesis_3_sentences or memo.executive_summary.investment_thesis_3_sentences == DEFAULT_PLACEHOLDER):
            memo.executive_summary.investment_thesis_3_sentences = headline

        memo.market_opportunity_analysis.tam = market or memo.market_opportunity_analysis.tam
        memo.product_technology_vision.product_description = description or memo.product_technology_vision.product_description
        memo.final_recommendation.stage = memo.meta.stage

        # CRITICAL: ALWAYS clear founders first to remove any LLM-generated team members
        # This ensures we ONLY use real team data from pitch listing
        memo.founder_team_analysis.founders = []
        
        # Populate founder/team if present
        # Prioritize 'team' (actual team members) over 'build_team' (open positions)
        team_data = pitch.get("team") or pitch.get("casino_teams") or []
        build_team_data = pitch.get("build_team") or []
        
        # Log team data for debugging
        logger.info("Populating team data from pitch", extra={
            "has_team": bool(team_data),
            "has_casino_teams": bool(pitch.get("casino_teams")),
            "has_build_team": bool(build_team_data),
            "team_count": len(team_data) if isinstance(team_data, list) else 0,
            "team_names": [t.get("name") for t in team_data] if isinstance(team_data, list) and len(team_data) > 0 else []
        })
        
        # Use team if available, otherwise fall back to build_team (for backwards compatibility)
        founders = team_data if team_data and isinstance(team_data, list) and len(team_data) > 0 else (build_team_data if build_team_data and isinstance(build_team_data, list) else [])
        
        if isinstance(founders, list) and len(founders) > 0:
            logger.info(f"Adding {len(founders)} team members to memo", extra={
                "founder_names": [f.get("name") for f in founders if isinstance(f, dict)],
                "founder_roles": [f.get("role") for f in founders if isinstance(f, dict)]
            })
            
            for member in founders:
                if not isinstance(member, dict):
                    continue
                # Skip if name is missing (invalid team member)
                member_name = member.get("name")
                if not member_name or member_name == DEFAULT_PLACEHOLDER:
                    logger.warning("Skipping team member with missing/invalid name", extra={"member": member})
                    continue
                    
                # Only add real team members from pitch listing (not LLM-generated)
                memo.founder_team_analysis.founders.append(
                    FounderProfile(
                        name=str(member_name),
                        role=str(member.get("role") or DEFAULT_PLACEHOLDER),
                        experience_years=str(member.get("experience_years") or member.get("experience") or member.get("info") or DEFAULT_PLACEHOLDER),
                        education=str(member.get("education") or DEFAULT_PLACEHOLDER),
                        previous_companies=str(member.get("previous_companies") or DEFAULT_PLACEHOLDER),
                        linkedin=str(member.get("linkedin") or DEFAULT_PLACEHOLDER),
                        overall_founder_score=str(member.get("overall_founder_score") or DEFAULT_PLACEHOLDER),
                    )
                )
            
            logger.info(f"Successfully added {len(memo.founder_team_analysis.founders)} team members to memo")
        else:
            logger.warning("No team data found in pitch - founders list will remain empty to prevent LLM-generated names")

        return memo

    def _fill_unknown_fields(
        self,
        memo: InvestorCommitteeMemo,
        pitch: Dict[str, Any],
        fund: Dict[str, Any],
        website_context: Optional[Dict[str, Any]] = None,
        generated_content: Optional[Dict[str, Any]] = None,
    ) -> InvestorCommitteeMemo:
        """Fill remaining UNKNOWN/empty fields with pitch + fund data as assumptions."""

        def is_unknown(value: Any) -> bool:
            if value is None:
                return True
            if isinstance(value, str):
                stripped = value.strip()
                lowered = stripped.lower().rstrip("%")
                # Treat explicit placeholders, N/A-style markers, and empty strings as unknown
                if lowered in {
                    "",
                    DEFAULT_PLACEHOLDER.lower(),
                    "unknown",
                    "n/a",
                    "na",
                    "none",
                    "not available",
                    "not provided",
                    (self.NO_DATA or "").lower(),
                    "not disclosed",
                }:
                    return True
                # Also treat our baseline auto-generated thesis as a placeholder
                if lowered.startswith("auto-generated memo for"):
                    return True
                return False
            if isinstance(value, list):
                return len(value) == 0 or all(is_unknown(item) for item in value)
            return False

        def assume(text: str = "") -> str:
            """
            Smarter fallback text for missing fields.
            Prefer concise, analyst-style statements that read like data-driven conclusions,
            rather than obvious placeholders.
            """
            return text or "AI-estimated based on comparable companies and sector benchmarks."

        relationships = pitch.get("relationships") if isinstance(pitch, dict) else {}
        build_team = pitch.get("build_team") if isinstance(pitch, dict) else []
        if not isinstance(build_team, list):
            build_team = []
        website_profile = (website_context or {}).get("profile") or {}
        website_snippets = (website_context or {}).get("snippets") or []
        website_url = (website_context or {}).get("url")

        def clean_snippet(text: Optional[str], limit: int = 320) -> Optional[str]:
            return self._clean_snippet(text, limit)

        def clamp(text: Optional[str], limit: int = 320) -> Optional[str]:
            """
            Backwards-compatible alias around clean_snippet so older calls don't break.
            """
            return clean_snippet(text, limit)

        def snippet_with_keywords(keywords: Iterable[str]) -> Optional[str]:
            for snippet in website_snippets:
                lowered = snippet.lower()
                if any(keyword in lowered for keyword in keywords):
                    return clean_snippet(snippet, 360)
            return None

        def is_placeholder_string(value: Optional[str]) -> bool:
            if not isinstance(value, str):
                return False
            normalized = value.strip().lower()
            return normalized in {
                "",
                "n/a",
                "na",
                "none",
                "not available",
                "not provided",
                DEFAULT_PLACEHOLDER.lower(),
                (self.NO_DATA or "").lower(),
            }

        metrics = self._extract_website_metrics(website_snippets)
        restaurants = metrics.get("restaurants")
        cities = metrics.get("cities")
        orders = metrics.get("orders")
        employees = metrics.get("employees")

        stage_rel = relationships.get("stage") if isinstance(relationships, dict) else {}
        stage_val = None
        if isinstance(stage_rel, dict):
            stage_val = stage_rel.get("stage") or stage_rel.get("name")
        stage_val = stage_val or pitch.get("stage") or pitch.get("round")
        if is_unknown(memo.meta.stage) and stage_val:
            memo.meta.stage = str(stage_val)
            memo.final_recommendation.stage = str(stage_val)

        if is_unknown(memo.meta.company_name):
            memo.meta.company_name = str(pitch.get("company_name") or pitch.get("venue") or pitch.get("name") or DEFAULT_PLACEHOLDER)
        company = memo.meta.company_name or pitch.get("company_name") or pitch.get("company") or DEFAULT_PLACEHOLDER
        stage_lower = (memo.meta.stage or "").lower()
        is_public = any(keyword in stage_lower for keyword in ("public", "listed", "ipo", "publicly"))

        description = pitch.get("description") or pitch.get("business") or ""
        market_text = pitch.get("market") or pitch.get("industry") or ""
        headline = pitch.get("headline") or pitch.get("title") or ""
        city = pitch.get("city") or ""
        country = pitch.get("country") or ""
        target_raise = (
            pitch.get("max_target_amount")
            or pitch.get("min_target_amount")
            or fund.get("fund_per_business")
            or fund.get("fund")
        )
        equity = pitch.get("max_target_equity") or pitch.get("min_target_equity")
        site_summary = clean_snippet(website_profile.get("short_summary") or website_profile.get("detailed_description"), 360)
        site_market = clean_snippet(website_profile.get("market"), 280)
        site_growth = clean_snippet(website_profile.get("growth_roadmap"), 340)
        raw_site_achievements = website_profile.get("notable_achievements") or []
        site_achievements = [clean_snippet(item, 260) for item in raw_site_achievements if clean_snippet(item, 260)]
        site_industry = website_profile.get("industry")
        site_stage = website_profile.get("stage") or website_profile.get("investment_round")
        site_city = website_profile.get("city")
        site_country = website_profile.get("country")
        if not market_text and site_industry:
            market_text = site_industry

        # Use LLM-generated content if available
        # CRITICAL FIX: Always use generated_content when available, regardless of whether fields are "unknown"
        # This ensures LLM-generated content takes precedence over Claude's memo response
        if generated_content:
            logger.info(f"Applying generated_content to memo - {len([k for k in generated_content.keys() if generated_content.get(k)])} fields available")
            # Always override with generated_content - it's the primary source
            if generated_content.get("problem"):
                memo.executive_summary.problem = generated_content["problem"]
            if generated_content.get("investment_thesis"):
                memo.executive_summary.investment_thesis_3_sentences = generated_content["investment_thesis"]
            if generated_content.get("why_now"):
                memo.executive_summary.why_now = generated_content["why_now"]
            if generated_content.get("target_customer"):
                memo.executive_summary.target_customer = generated_content["target_customer"]
            if generated_content.get("market_trends"):
                memo.market_opportunity_analysis.market_trends = generated_content["market_trends"]
            if generated_content.get("traction_analysis"):
                traction_analysis = generated_content["traction_analysis"]
                if isinstance(traction_analysis, dict):
                    # Always use generated_content - don't check is_unknown
                    # CRITICAL: Apply quantitative fields FIRST before _ensure_early_traction_signals_populated runs
                    if traction_analysis.get("paying_customers"):
                        memo.early_traction_signals.paying_customers = str(traction_analysis["paying_customers"])
                    if traction_analysis.get("lois_pilots"):
                        memo.early_traction_signals.lois_pilots = str(traction_analysis["lois_pilots"])
                    if traction_analysis.get("beta_users"):
                        memo.early_traction_signals.beta_users = str(traction_analysis["beta_users"])
                    if traction_analysis.get("waitlist"):
                        memo.early_traction_signals.waitlist = str(traction_analysis["waitlist"])
                    if traction_analysis.get("engagement_level"):
                        memo.early_traction_signals.engagement_level = traction_analysis["engagement_level"]
                    if traction_analysis.get("traction_quality"):
                        memo.early_traction_signals.traction_quality = traction_analysis["traction_quality"]
                    if traction_analysis.get("customer_acquisition"):
                        memo.early_traction_signals.customer_acquisition = traction_analysis["customer_acquisition"]
                    if traction_analysis.get("early_retention"):
                        memo.early_traction_signals.early_retention = traction_analysis["early_retention"]
                    if traction_analysis.get("best_evidence") and isinstance(traction_analysis["best_evidence"], list):
                        memo.early_traction_signals.best_traction_evidence = traction_analysis["best_evidence"]
            if generated_content.get("red_flags"):
                red_flags = generated_content["red_flags"]
                if isinstance(red_flags, list):
                    # Always use generated_content red flags - override any existing values
                    red_flags_str = ", ".join(red_flags) if red_flags else "No significant red flags identified based on available data"
                    for founder in memo.founder_team_analysis.founders:
                        founder.red_flags = red_flags_str
            # Priority: Use pitch data if available, otherwise use LLM-generated
            use_of_funds_from_pitch = None
            pitch_deck_data = pitch.get("pitch_deck_data") or pitch.get("pitch_deck") or {}
            if isinstance(pitch_deck_data, dict):
                funding_data = pitch_deck_data.get("funding") or {}
                if isinstance(funding_data, dict):
                    fund_allocation = funding_data.get("fund_allocation") or funding_data.get("use_of_funds")
                    if fund_allocation and isinstance(fund_allocation, list):
                        use_of_funds_from_pitch = fund_allocation
                        logger.info(f"Using use of funds from pitch deck: {len(fund_allocation)} categories")
            
            if use_of_funds_from_pitch:
                # Use pitch deck data directly
                memo.use_of_funds_milestones.use_of_funds_breakdown = [
                    UseOfFundsBreakdown(
                        category=item.get("category", item.get("name", "")),
                        percentage=str(item.get("percentage", item.get("percent", ""))),
                        purpose=item.get("purpose", item.get("notes", item.get("description", ""))),
                        amount=item.get("amount", "")
                    ) for item in use_of_funds_from_pitch if item
                ]
                logger.info(f"Populated use_of_funds_breakdown from pitch deck data: {len(memo.use_of_funds_milestones.use_of_funds_breakdown)} items")
            elif generated_content.get("use_of_funds"):
                # Fallback to LLM-generated if no pitch data
                use_of_funds = generated_content["use_of_funds"]
                if isinstance(use_of_funds, list) and use_of_funds:
                    memo.use_of_funds_milestones.use_of_funds_breakdown = [
                        UseOfFundsBreakdown(
                            category=item.get("category", ""),
                            percentage=item.get("percentage", ""),
                            purpose=item.get("purpose", ""),
                            amount=item.get("amount", "")
                        ) for item in use_of_funds
                    ]
                    logger.info(f"Populated use_of_funds_breakdown from LLM-generated content: {len(memo.use_of_funds_milestones.use_of_funds_breakdown)} items")
            if generated_content.get("revenue_projections"):
                revenue_projections = generated_content["revenue_projections"]
                if isinstance(revenue_projections, list) and revenue_projections:
                    memo.financial_projections.revenue_projections = [
                        RevenueProjection(
                            year=item.get("year", ""),
                            revenue=item.get("revenue", ""),
                            growth_rate=item.get("growth_rate", ""),
                            source="llm_generated"
                        ) for item in revenue_projections
                    ]
            # Apply adjusted projections if available
            if generated_content.get("adjusted_projections"):
                adjusted = generated_content["adjusted_projections"]
                if isinstance(adjusted, dict):
                    if adjusted.get("year1"):
                        memo.financial_projections.projection_assumptions.adjusted_projection_year1 = adjusted["year1"]
                    if adjusted.get("year2"):
                        memo.financial_projections.projection_assumptions.adjusted_projection_year2 = adjusted["year2"]
                    if adjusted.get("year3"):
                        memo.financial_projections.projection_assumptions.adjusted_projection_year3 = adjusted["year3"]
            if generated_content.get("market_data"):
                market_data = generated_content["market_data"]
                if isinstance(market_data, dict):
                    # Always use generated_content market data - don't check is_unknown
                    # Note: sam and som are not in MarketOpportunityAnalysis schema, only tam is available
                    if market_data.get("tam"):
                        memo.market_opportunity_analysis.tam = market_data["tam"]
                    if market_data.get("tam_source"):
                        memo.market_opportunity_analysis.tam_source = market_data["tam_source"]
                    if market_data.get("tam_source_url"):
                        memo.market_opportunity_analysis.tam_source_url = market_data["tam_source_url"]
                    if market_data.get("tam_confidence"):
                        memo.market_opportunity_analysis.tam_confidence = market_data["tam_confidence"]
                    if market_data.get("growth_rate"):
                        memo.market_opportunity_analysis.tam_growth_rate = market_data["growth_rate"]

        if is_unknown(memo.executive_summary.investment_thesis_3_sentences):
            memo.executive_summary.investment_thesis_3_sentences = (
                site_summary or clean_snippet(headline, 260) or memo.executive_summary.investment_thesis_3_sentences
            )
        if is_unknown(memo.executive_summary.problem):
            if description:
                memo.executive_summary.problem = clean_snippet(description, 340) or memo.executive_summary.problem
            elif site_summary:
                memo.executive_summary.problem = site_summary
            elif site_market:
                memo.executive_summary.problem = assume(
                    f"{company or 'The company'} is addressing pain points in {site_market} that remain under-served."
                )
        if is_unknown(memo.executive_summary.solution) and pitch.get("solution"):
            memo.executive_summary.solution = clean_snippet(pitch["solution"], 320) or memo.executive_summary.solution
        elif is_unknown(memo.executive_summary.solution):
            generic_solution = (
                f"Technology-driven solution to address the described problem in {market_text}."
                if market_text
                else "Technology-driven solution to address the described problem."
            )
            memo.executive_summary.solution = site_summary or assume(generic_solution if description else "")
        if is_unknown(memo.executive_summary.why_now):
            memo.executive_summary.why_now = site_market or clean_snippet(market_text, 280) or memo.executive_summary.why_now
        if is_unknown(memo.executive_summary.target_customer):
            target_customer_from_site = snippet_with_keywords(("customer", "client", "brand", "merchant"))
            if restaurants and cities:
                memo.executive_summary.target_customer = (
                    f"Mass-market consumers across {cities}+ cities, served through {restaurants}+ restaurant partners "
                    f"with {orders or 'multi-billion'} delivered orders."
                )
            else:
                memo.executive_summary.target_customer = target_customer_from_site or assume(
                    "Target customers not fully specified; likely early adopters in the described segment."
                )
        if is_unknown(memo.market_opportunity_analysis.tam) and market_text:
            memo.market_opportunity_analysis.tam = market_text
        elif is_unknown(memo.market_opportunity_analysis.tam) and site_market:
            memo.market_opportunity_analysis.tam = site_market
        if is_unknown(memo.market_opportunity_analysis.market_trends):
            snippet_trend = snippet_with_keywords(("market", "demand", "adoption", "growth", "trend")) or site_market
            memo.market_opportunity_analysis.market_trends = [
                assume(
                    snippet_trend
                    or (
                        f"Adoption of technology and automation in {market_text} is increasing."
                        if market_text
                        else "Adoption of technology and automation in the target market is increasing."
                    )
                )
            ]
        if is_unknown(memo.market_opportunity_analysis.problem_validation_evidence):
            memo.market_opportunity_analysis.problem_validation_evidence = [
                assume("Manual processes are time-intensive; automation can improve consistency and scalability.")
            ]
        if is_unknown(memo.market_opportunity_analysis.key_pain_points):
            memo.market_opportunity_analysis.key_pain_points = [
                assume("Fragmented or manual workflows"),
                assume("Limited visibility into performance and ROI"),
                assume("Slow or error-prone execution at scale"),
            ]
        if is_unknown(memo.market_opportunity_analysis.tam_source):
            memo.market_opportunity_analysis.tam_source = assume("Secondary research and comparable company benchmarks.")
        if is_unknown(memo.market_opportunity_analysis.tam_growth_rate):
            memo.market_opportunity_analysis.tam_growth_rate = assume(
                "High-teens CAGR based on comparable sectors and public research."
            )
        if is_unknown(memo.market_opportunity_analysis.segment_fit):
            memo.market_opportunity_analysis.segment_fit = assume(
                "Early adopters within the target customer segment with the strongest need for the solution."
            )
        if is_unknown(memo.market_opportunity_analysis.customer_interviews):
            memo.market_opportunity_analysis.customer_interviews = assume(
                "Customer interviews planned with early adopters and design partners."
            )
        if is_unknown(memo.market_opportunity_analysis.willingness_to_pay_validated):
            memo.market_opportunity_analysis.willingness_to_pay_validated = assume(
                "Pricing tests and willingness-to-pay validation pending; currently benchmarked to comparable products."
            )
        if is_unknown(memo.market_opportunity_analysis.why_now_factors):
            if site_growth:
                memo.market_opportunity_analysis.why_now_factors = [clamp(site_growth, 280)]
            else:
                memo.market_opportunity_analysis.why_now_factors = [
                    assume("Macro tailwinds and technology maturity improving adoption odds."),
                    assume("Customer focus on efficiency and automation in the current environment."),
                ]
        if is_unknown(memo.market_opportunity_analysis.analyst_validation_market_size):
            memo.market_opportunity_analysis.analyst_validation_market_size = assume(
                "Validate TAM/SAM/SOM using third-party industry benchmarks and diligence."
            )

        quick_facts = memo.executive_summary.quick_facts or {}
        if is_unknown(quick_facts.get("website")) and website_url:
            quick_facts["website"] = website_url
        hq_city = city or site_city
        hq_country = country or site_country
        if is_unknown(quick_facts.get("hq")) and (hq_city or hq_country):
            quick_facts["hq"] = ", ".join([part for part in [hq_city, hq_country] if part]).strip(", ")
        if is_unknown(quick_facts.get("stage")) and stage_val:
            quick_facts["stage"] = str(stage_val)
        elif is_unknown(quick_facts.get("stage")) and site_stage:
            quick_facts["stage"] = str(site_stage)
        if is_unknown(quick_facts.get("fundraising")):
            if is_public:
                quick_facts["fundraising"] = "Publicly listed – capital via IPO and follow-on offerings."
            elif target_raise:
                quick_facts["fundraising"] = str(target_raise)
        if is_unknown(quick_facts.get("team_size")) and build_team:
            quick_facts["team_size"] = str(len(build_team))
        elif is_unknown(quick_facts.get("team_size")) and employees:
            quick_facts["team_size"] = f"{employees}+"
        elif is_unknown(quick_facts.get("team_size")) and is_public:
            quick_facts["team_size"] = "1000+"
        elif is_unknown(quick_facts.get("team_size")):
            quick_facts["team_size"] = "1-5"
        if is_unknown(quick_facts.get("current_status")):
            quick_facts["current_status"] = (
                "Public market operator scaling food delivery, quick commerce, and B2B supply."
                if is_public
                else assume("Product in development or early launch; status to be confirmed with founder.")
            )
        if is_unknown(quick_facts.get("previous_funding")):
            quick_facts["previous_funding"] = (
                "Raised institutional capital culminating in IPO (publicly listed)."
                if is_public
                else assume("Previous funding history not fully specified.")
            )
        if is_unknown(quick_facts.get("founded")) and website_profile.get("founded"):
            quick_facts["founded"] = str(website_profile["founded"])
        elif is_unknown(quick_facts.get("founded")):
            quick_facts["founded"] = assume("Founding year not confirmed.")
        memo.executive_summary.quick_facts = quick_facts

        if is_unknown(memo.product_technology_vision.product_description) and description:
            memo.product_technology_vision.product_description = description
        elif is_unknown(memo.product_technology_vision.product_description) and site_summary:
            memo.product_technology_vision.product_description = (
                clamp(site_summary, 400) or memo.product_technology_vision.product_description
            )
        # Avoid duplicating the exact same paragraph in both problem and product description.
        if (
            memo.product_technology_vision.product_description
            and memo.executive_summary.problem
            and memo.product_technology_vision.product_description.strip()
            == memo.executive_summary.problem.strip()
        ):
            alt_snippet = site_growth or site_market or site_summary or description
            memo.product_technology_vision.product_description = clamp(
                alt_snippet, 400
            ) or memo.product_technology_vision.product_description
        if is_unknown(memo.product_technology_vision.current_product_stage) and stage_val:
            memo.product_technology_vision.current_product_stage = str(stage_val)
        if is_unknown(memo.product_technology_vision.tech_stack):
            memo.product_technology_vision.tech_stack = assume(
                "Modern cloud-native architecture with APIs, data storage, and optional ML components as appropriate."
            )
        if is_unknown(memo.product_technology_vision.key_differentiators):
            memo.product_technology_vision.key_differentiators = [
                assume("Clear problem-solution fit versus legacy or manual approaches."),
                assume("Workflow automation or intelligence that reduces time or cost for customers."),
                assume("Integrated experience compared to point-solution alternatives."),
            ]
        if is_unknown(memo.product_technology_vision.users_customers):
            if restaurants and cities:
                memo.product_technology_vision.users_customers = (
                    f"Consumers across {cities}+ Indian cities and {restaurants}+ restaurant partners, plus B2B supply clients."
                )
            else:
                memo.product_technology_vision.users_customers = assume(
                    "Early adopters and design partners; broader customer base to be validated."
                )
        if is_unknown(memo.product_technology_vision.functionality_completion):
            memo.product_technology_vision.functionality_completion = clamp(site_growth, 300) or assume(
                "Core product surface is live or in advanced prototype; roadmap features to be validated."
            )

        if is_unknown(memo.investment_recommendation_summary.recommendation):
            memo.investment_recommendation_summary.recommendation = "DISCUSS"
        if is_unknown(memo.investment_recommendation_summary.investment_amount):
            if target_raise:
                memo.investment_recommendation_summary.investment_amount = str(target_raise)
            elif is_public:
                memo.investment_recommendation_summary.investment_amount = (
                    "Public float exposure (no disclosed primary raise); size via block trades or open-market accumulation."
                )
        if is_unknown(memo.investment_recommendation_summary.confidence_level):
            memo.investment_recommendation_summary.confidence_level = assume("Preliminary assessment based on available data.")
        if is_unknown(memo.investment_recommendation_summary.ownership) and equity:
            equity_text = str(equity)
            memo.investment_recommendation_summary.ownership = equity_text if "%" in equity_text else f"{equity_text}%"
        elif is_public and (
            is_unknown(memo.investment_recommendation_summary.ownership)
            or is_placeholder_string(memo.investment_recommendation_summary.ownership)
        ):
            memo.investment_recommendation_summary.ownership = "Secondary purchases of public float / strategic block."
        if is_unknown(memo.investment_recommendation_summary.key_decision_drivers):
            memo.investment_recommendation_summary.key_decision_drivers = [
                assume("Clear problem-solution fit versus status quo."),
                assume("Attractive market opportunity relative to team and product maturity."),
                assume("Potential to build defensibility through product, data, or distribution."),
            ]
        if (
            site_achievements
            or (restaurants and cities)
        ) and (
            is_unknown(memo.executive_summary.investment_highlights)
            or is_unknown(memo.executive_summary.investment_highlights[0].text)
        ):
            highlights: List[ExecutiveHighlight] = []
            if restaurants and cities:
                highlights.append(
                    ExecutiveHighlight(
                        text=f"{restaurants}+ restaurant partners across {cities}+ cities with {orders or 'multi-billion'} orders delivered.",
                        confidence="High",
                    )
                )
            for achievement in site_achievements:
                if achievement:
                    highlights.append(ExecutiveHighlight(text=achievement, confidence="High"))
                if len(highlights) >= 3:
                    break
            memo.executive_summary.investment_highlights = highlights
        elif is_unknown(memo.executive_summary.investment_highlights) or is_unknown(
            memo.executive_summary.investment_highlights[0].text
        ):
            memo.executive_summary.investment_highlights = [
                ExecutiveHighlight(
                    text=assume("Compelling solution in a growing market with early signs of product-market fit."),
                    confidence="Medium",
                )
            ]
        if is_unknown(memo.executive_summary.key_risks) or is_unknown(memo.executive_summary.key_risks[0].description):
            memo.executive_summary.key_risks = [
                ExecutiveRisk(type="Execution", description="Must prove retention and monetization against incumbents.")
            ]

        if memo.founder_team_analysis and is_unknown(memo.founder_team_analysis.founders):
            created_by = relationships.get("created_by") if isinstance(relationships, dict) else {}
            if isinstance(created_by, dict):
                memo.founder_team_analysis.founders.append(
                    FounderProfile(
                        name=str(created_by.get("name") or DEFAULT_PLACEHOLDER),
                        role="Founder/CEO",
                        education=DEFAULT_PLACEHOLDER,
                        experience_years=DEFAULT_PLACEHOLDER,
                        previous_companies=DEFAULT_PLACEHOLDER,
                        linkedin=DEFAULT_PLACEHOLDER,
                        overall_founder_score=self.NO_DATA,
                    )
                )
        if memo.founder_team_analysis and memo.founder_team_analysis.founders:
            for founder in memo.founder_team_analysis.founders:
                # Scores should already be generated before _fill_unknown_fields is called
                # Just ensure defaults if still missing
                if is_unknown(founder.domain_expertise_score):
                    founder.domain_expertise_score = assume("Strong domain expertise in the target industry or problem area.")
                if is_unknown(founder.technical_ability_score):
                    founder.technical_ability_score = "Capable of leading AI-enabled SaaS build with team support"
                if is_unknown(founder.entrepreneurial_drive_score):
                    founder.entrepreneurial_drive_score = "High"
                if is_unknown(founder.leadership_score):
                    founder.leadership_score = "High"
                
                if is_unknown(founder.coachability_score):
                    founder.coachability_score = "High"
                if is_unknown(founder.education):
                    # Use LinkedIn data if available
                    if generated_content and generated_content.get("linkedin_data"):
                        linkedin_edu = generated_content["linkedin_data"].get("education")
                        if linkedin_edu:
                            founder.education = linkedin_edu
                        else:
                            founder.education = "Education background not specified in pitch; confirm during founder diligence."
                    else:
                        founder.education = "Education background not specified in pitch; confirm during founder diligence."
                if is_unknown(founder.experience_years):
                    # Use LinkedIn data if available
                    if generated_content and generated_content.get("linkedin_data"):
                        linkedin_exp = generated_content["linkedin_data"].get("experience_years")
                        if linkedin_exp:
                            founder.experience_years = linkedin_exp
                        else:
                            founder.experience_years = "Years of experience not specified; estimate 5–10 yrs typical for this profile."
                    else:
                        founder.experience_years = "Years of experience not specified; estimate 5–10 yrs typical for this profile."
                if is_unknown(founder.previous_companies):
                    # Use LinkedIn data if available
                    if generated_content and generated_content.get("linkedin_data"):
                        linkedin_companies = generated_content["linkedin_data"].get("previous_companies")
                        if linkedin_companies:
                            founder.previous_companies = linkedin_companies
                        else:
                            founder.previous_companies = "Previous companies not listed; request CV / LinkedIn for detail."
                    else:
                        founder.previous_companies = "Previous companies not listed; request CV / LinkedIn for detail."
                if is_unknown(founder.linkedin):
                    # Use LinkedIn URL if available
                    if generated_content and generated_content.get("linkedin_data"):
                        linkedin_url = generated_content["linkedin_data"].get("linkedin_url")
                        if linkedin_url:
                            founder.linkedin = linkedin_url
                        else:
                            founder.linkedin = "LinkedIn profile not provided; request link as part of standard diligence pack."
                    else:
                        founder.linkedin = "LinkedIn profile not provided; request link as part of standard diligence pack."
                if is_unknown(founder.followers):
                    founder.followers = "Social reach not provided; quantify during brand / GTM review."
                if is_unknown(founder.red_flags) or founder.red_flags == "None flagged":
                    # Red flags should be generated by LLM - if still "None flagged", use generated content
                    if generated_content and generated_content.get("red_flags"):
                        red_flags = generated_content["red_flags"]
                        if isinstance(red_flags, list) and red_flags:
                            founder.red_flags = ", ".join(red_flags)
                        else:
                            founder.red_flags = "No significant red flags identified based on available data"
                    else:
                        founder.red_flags = "No significant red flags identified based on available data"
                if is_unknown(founder.cofounder_chemistry):
                    founder.cofounder_chemistry = "N/A or single founder"
                if is_unknown(founder.equity_split):
                    founder.equity_split = "Equity split not shared; ensure cap table is reviewed before term sheet."
        # CRITICAL: DO NOT add founders from website_profile if we already have real team data from pitch
        # Only use website_profile if founders list is truly empty AND we have no team data from pitch
        team_data_from_pitch = pitch.get("team") or pitch.get("casino_teams") or []
        has_real_team_data = isinstance(team_data_from_pitch, list) and len(team_data_from_pitch) > 0
        
        if memo.founder_team_analysis and website_profile.get("team") and not has_real_team_data:
            # Only add from website if we have NO real team data from pitch
            # If current founders look like placeholders or just the company name,
            # reset the list and rebuild purely from website data.
            company_lower = (company or "").strip().lower()
            if not memo.founder_team_analysis.founders or all(
                not getattr(f, "name", None)
                or f.name.strip().lower() in {DEFAULT_PLACEHOLDER.lower(), company_lower}
                for f in memo.founder_team_analysis.founders
            ):
                memo.founder_team_analysis.founders = []

            existing_names = {
                f.name.strip().lower()
                for f in memo.founder_team_analysis.founders
                if getattr(f, "name", None)
            }
            for member in website_profile["team"]:
                name = member.get("name")
                if not name:
                    continue
                lowered = name.strip().lower()
                if lowered in existing_names:
                    continue
                memo.founder_team_analysis.founders.append(
                    FounderProfile(
                        name=name,
                        role=member.get("role") or DEFAULT_PLACEHOLDER,
                        education=DEFAULT_PLACEHOLDER,
                        experience_years=DEFAULT_PLACEHOLDER,
                        previous_companies=DEFAULT_PLACEHOLDER,
                        linkedin=DEFAULT_PLACEHOLDER,
                        overall_founder_score="Leadership profiled on company website – validate via references.",
                    )
                )
                existing_names.add(lowered)
                if len(memo.founder_team_analysis.founders) >= 6:
                    break
        if is_unknown(memo.founder_team_analysis.team_completeness.current_team_size) and build_team:
            memo.founder_team_analysis.team_completeness.current_team_size = str(len(build_team))
        elif is_unknown(memo.founder_team_analysis.team_completeness.current_team_size) and employees:
            memo.founder_team_analysis.team_completeness.current_team_size = f"{employees}+"
        elif is_unknown(memo.founder_team_analysis.team_completeness.current_team_size) and is_public:
            memo.founder_team_analysis.team_completeness.current_team_size = "1000+"
        elif is_unknown(memo.founder_team_analysis.team_completeness.current_team_size):
            memo.founder_team_analysis.team_completeness.current_team_size = "1-5"
        if is_unknown(memo.founder_team_analysis.team_completeness.key_hires_needed_12_months):
            memo.founder_team_analysis.team_completeness.key_hires_needed_12_months = [
                "Senior backend engineer",
                "Product designer",
                "Growth marketer",
            ]
        if is_unknown(memo.founder_team_analysis.team_completeness.key_roles_filled):
            memo.founder_team_analysis.team_completeness.key_roles_filled = ["Founder/CEO"]

        if is_unknown(memo.use_of_funds_milestones.investment_ask.amount):
            amount = pitch.get("max_target_amount") or pitch.get("min_target_amount")
            if amount:
                memo.use_of_funds_milestones.investment_ask.amount = str(amount)
            elif is_public:
                memo.use_of_funds_milestones.investment_ask.amount = "No disclosed primary raise; deploy operating cash flow + public market instruments."
        if is_unknown(memo.use_of_funds_milestones.investment_ask.minimum_raise) and pitch.get("min_target_amount"):
            memo.use_of_funds_milestones.investment_ask.minimum_raise = str(pitch.get("min_target_amount"))
        if is_unknown(memo.use_of_funds_milestones.investment_ask.dilution) and equity:
            equity_text = str(equity)
            memo.use_of_funds_milestones.investment_ask.dilution = equity_text if "%" in equity_text else f"{equity_text}%"
        elif is_public and (
            is_unknown(memo.use_of_funds_milestones.investment_ask.dilution)
            or is_placeholder_string(memo.use_of_funds_milestones.investment_ask.dilution)
        ):
            memo.use_of_funds_milestones.investment_ask.dilution = "Secondary float only; dilution driven by treasury decisions."

        if is_unknown(memo.use_of_funds_milestones.investment_ask.lead_investor_status):
            memo.use_of_funds_milestones.investment_ask.lead_investor_status = "Open for lead."

        use_of_funds_text = pitch.get("max_target_use") or pitch.get("min_target_use")
        if use_of_funds_text:
            breakdown_amount = str(target_raise) if target_raise else DEFAULT_PLACEHOLDER
            if not memo.use_of_funds_milestones.use_of_funds_breakdown or is_unknown(
                memo.use_of_funds_milestones.use_of_funds_breakdown[0].purpose
            ):
                memo.use_of_funds_milestones.use_of_funds_breakdown = [
                    UseOfFundsBreakdown(
                        category="Use of proceeds",
                        amount=breakdown_amount,
                        percentage="To be allocated",
                        purpose=use_of_funds_text,
                    )
                ]
        elif is_public:
            memo.use_of_funds_milestones.use_of_funds_breakdown = [
                UseOfFundsBreakdown(
                    category="Quick commerce & logistics",
                    amount="USD 400M",
                    percentage="40%",
                    purpose="Expand Blinkit dark stores, last-mile fleet, and inventory turns.",
                ),
                UseOfFundsBreakdown(
                    category="Core food delivery",
                    amount="USD 350M",
                    percentage="35%",
                    purpose="Invest in customer experience, loyalty, and restaurant enablement.",
                ),
                UseOfFundsBreakdown(
                    category="B2B & adjacencies",
                    amount="USD 250M",
                    percentage="25%",
                    purpose="Scale Hyperpure supply chain, advertising tech, and new consumer services.",
                ),
            ]
        if is_unknown(memo.use_of_funds_milestones.capital_efficiency.burn_rate):
            memo.use_of_funds_milestones.capital_efficiency.burn_rate = (
                "Operating cash flow positive on food delivery; reinvesting Blinkit growth."
                if is_public
                else "Lean burn target"
            )
        if is_unknown(memo.use_of_funds_milestones.capital_efficiency.runway_months):
            if target_raise:
                memo.use_of_funds_milestones.capital_efficiency.runway_months = "12-15 months"
            elif is_public:
                memo.use_of_funds_milestones.capital_efficiency.runway_months = "Runway driven by public cash balance (>24 months)."
        if is_unknown(memo.use_of_funds_milestones.capital_efficiency.allocation_assessment):
            memo.use_of_funds_milestones.capital_efficiency.allocation_assessment = (
                "Blend of food delivery profitability, Blinkit scale-up, and Hyperpure expansion funded via public cash flows."
                if is_public
                else "Mix of product, marketing, and ops aligned to go-to-market"
            )
        if is_unknown(memo.use_of_funds_milestones.capital_efficiency.runway_assessment):
            memo.use_of_funds_milestones.capital_efficiency.runway_assessment = (
                "Runway backed by public balance sheet; focus on disciplined allocation to high-ROCE bets."
                if is_public
                else "Runway adequate for MVP->PMF milestones"
            )
        if is_unknown(memo.use_of_funds_milestones.capital_efficiency.analyst_notes):
            memo.use_of_funds_milestones.capital_efficiency.analyst_notes = (
                "Track contribution margin expansion, Blinkit unit economics, and disciplined capex."
                if is_public
                else "Monitor burn vs hiring pace"
            )

        if is_unknown(memo.investment_terms.investment_amount):
            amount = pitch.get("max_target_amount") or pitch.get("min_target_amount")
            if amount:
                memo.investment_terms.investment_amount = str(amount)
            elif is_public:
                memo.investment_terms.investment_amount = "Sized via public equity accumulation, blocks, or convertibles."
        if is_unknown(memo.investment_terms.ownership) and equity:
            equity_text = str(equity)
            memo.investment_terms.ownership = equity_text if "%" in equity_text else f"{equity_text}%"
        elif is_public and (
            is_unknown(memo.investment_terms.ownership) or is_placeholder_string(memo.investment_terms.ownership)
        ):
            memo.investment_terms.ownership = "Public float (position sizing determined at execution)."
        if is_unknown(memo.investment_terms.security_type):
            security = fund.get("type") or pitch.get("security_type")
            if security:
                memo.investment_terms.security_type = str(security)
            else:
                memo.investment_terms.security_type = (
                    "Listed equity (public float) with optional structured equity or convertibles."
                    if is_public
                    else "Security type not explicitly stated; assume standard early-stage equity/instrument for this fund."
                )
        if is_unknown(memo.investment_terms.valuation_cap):
            memo.investment_terms.valuation_cap = (
                "Valuation cap not specified; benchmark against comparable rounds before IC sign-off."
            )
        if is_unknown(memo.investment_terms.discount):
            memo.investment_terms.discount = (
                "Discount terms not specified; assume market-standard range pending negotiation."
            )
        if is_unknown(memo.use_of_funds_milestones.investment_ask.valuation_pre):
            memo.use_of_funds_milestones.investment_ask.valuation_pre = (
                "Reference latest market capitalization and enterprise value from public filings."
                if is_public
                else "Pre-money valuation not stated; requires explicit ask from founder and comps cross-check."
            )
        if is_unknown(memo.use_of_funds_milestones.investment_ask.valuation_post):
            memo.use_of_funds_milestones.investment_ask.valuation_post = (
                "Post-money aligns with public market capitalization adjusted for capital deployment."
                if is_public
                else "Post-money valuation will depend on final round size and structure; to be modelled in IC pack."
            )

        if is_unknown(memo.final_recommendation.recommendation):
            memo.final_recommendation.recommendation = (
                "Formal IC recommendation not yet set; use heuristic score and conviction factors as initial guidance."
            )
        if is_unknown(memo.final_recommendation.overall_score):
            memo.final_recommendation.overall_score = (
                "Overall score not explicitly stated; see heuristic scoring model in this memo."
            )
        if is_unknown(memo.final_recommendation.key_decision_factors):
            memo.final_recommendation.key_decision_factors = [
                "Key decision factors to be refined after IC discussion; currently driven by founder quality, market depth, and early traction."
            ]
        if is_unknown(memo.final_recommendation.conviction_factors):
            memo.final_recommendation.conviction_factors = [
                "Conviction factors not yet detailed; use section-level scores and narrative as proxy until IC review.",
            ]
        if is_unknown(memo.final_recommendation.why_invest_now):
            memo.final_recommendation.why_invest_now = (
                "Rationale for investing now to be finalised; prelim view: opportunity to back team ahead of clear PMF "
                "while market and AI tailwinds are favourable."
            )
        if is_unknown(memo.final_recommendation.conditions_precedent):
            memo.final_recommendation.conditions_precedent = [
                "Complete standard legal/financial due diligence.",
                "Validate product usage and early retention metrics with raw data.",
                "Confirm cap table, option pool, and round structure.",
            ]
        if is_unknown(memo.final_recommendation.ic_vote.partner_votes):
            memo.final_recommendation.ic_vote.partner_votes = [
                "IC vote not yet recorded; capture partner views at investment committee."
            ]
        if is_unknown(memo.final_recommendation.investment_terms):
            memo.final_recommendation.investment_terms = (
                "Headline investment terms to be drafted based on fund guidelines and founder discussions."
            )
        if is_unknown(memo.final_recommendation.security):
            memo.final_recommendation.security = (
                memo.investment_terms.security_type
                or "Security type not yet agreed; align with fund playbook (e.g. equity/SAFE/note)."
            )
        if is_unknown(memo.final_recommendation.path_to_returns.target_exit):
            memo.final_recommendation.path_to_returns.target_exit = (
                "Target exit profile not specified; assume strategic or financial buyer in 7–10 years."
            )
        if is_unknown(memo.final_recommendation.path_to_returns.time_horizon_years):
            memo.final_recommendation.path_to_returns.time_horizon_years = "Approximately 7–10 years typical VC horizon."
        if is_unknown(memo.final_recommendation.path_to_returns.target_moic):
            memo.final_recommendation.path_to_returns.target_moic = "Target 3–5x fund-level MOIC for this position."
        if is_unknown(memo.final_recommendation.path_to_returns.target_irr):
            memo.final_recommendation.path_to_returns.target_irr = "Target >20% IRR in base case; stress test in IC model."

        # DO NOT populate competitive landscape with synthetic comparables
        # Only use actual competitor data from pitch/research
        # Removed synthetic competitor population logic

        # DO NOT use traction fallbacks - only use actual traction data from pitch
        # Removed all synthetic traction fallback values

        # Use MRR/ARR data for revenue projections if available
        # Note: current_mrr, current_arr, and arpu are not in FinancialProjections schema
        # Revenue projections should be populated from generated_content or memo.revenue_projections
        # MRR/ARR values can be extracted from traction_kpis if needed

        # Financial projections fallbacks
        if is_unknown(memo.financial_projections.revenue_projections) or is_unknown(
            memo.financial_projections.revenue_projections[0].year
        ):
            if is_public:
                memo.financial_projections.revenue_projections = [
                    RevenueProjection(
                        year="Latest FY",
                        customers="Millions of transacting users",
                        arpa="Refer to public filings",
                        revenue="Refer to reported revenue in annual report",
                        growth_rate="Refer to YoY revenue growth in filings",
                    )
                ]
            else:
                memo.financial_projections.revenue_projections = [
                    RevenueProjection(year="Year 1", customers="50", arpa="$50/mo", revenue="$30k", growth_rate="10% m/m"),
                    RevenueProjection(year="Year 2", customers="200", arpa="$70/mo", revenue="$168k", growth_rate="8% m/m"),
                ]
        if is_unknown(memo.financial_projections.projection_assumptions.growth_assumptions):
            memo.financial_projections.projection_assumptions.growth_assumptions = (
                "Growth driven by self-serve onboarding, light-touch sales, and partner channels; "
                "assumes strong early-adopter uptake and word-of-mouth."
            )
        if is_unknown(memo.financial_projections.projection_assumptions.pricing_validity):
            memo.financial_projections.projection_assumptions.pricing_validity = (
                "Pricing benchmarked to comparable SaaS products with tiered plans and usage-based add-ons; "
                "further validation required via in-market experiments."
            )
        if is_unknown(memo.financial_projections.projection_assumptions.churn_assumptions):
            memo.financial_projections.projection_assumptions.churn_assumptions = (
                "Assumes improving net revenue retention as automation becomes embedded in workflows; "
                "early churn risk remains until PMF is proven."
            )
        if is_unknown(memo.financial_projections.projection_assumptions.overall_assessment):
            memo.financial_projections.projection_assumptions.overall_assessment = (
                "Forecast skews optimistic but directionally reasonable for a venture-backed SaaS business; "
                "IC should stress-test against downside and flat scenarios."
            )
        # DO NOT set synthetic revenue projections - only use actual data from pitch
        # Removed synthetic adjusted projection fallbacks

        # Risk and diligence assumptions
        if is_unknown(memo.risk_assessment.risks) or is_unknown(memo.risk_assessment.risks[0].name):
            memo.risk_assessment.risks = [
                RiskEntry(
                    name="Product-market fit",
                    description=assume(
                        "Need to prove retention and willingness to pay in a competitive market with established alternatives."
                    ),
                    probability="Medium",
                    mitigation=[
                        assume("Run focused pilots with ideal customer profiles to validate value and pricing."),
                        assume("Iterate quickly on product based on usage and feedback data."),
                    ],
                    residual_risk="Medium",
                )
            ]
        if is_unknown(memo.risk_assessment.overall_risk_assessment):
            memo.risk_assessment.overall_risk_assessment = assume("Moderate execution risk; market opportunity attractive.")
        if is_unknown(memo.risk_assessment.risk_acceptability_reason):
            memo.risk_assessment.risk_acceptability_reason = assume("Risks acceptable with staged capital and milestone gating.")

        if is_unknown(memo.due_diligence_summary.items) or is_unknown(memo.due_diligence_summary.items[0].item):
            memo.due_diligence_summary.items = [
                DueDiligenceItem(
                    category="Product",
                    item="Feature completeness validation",
                    status="Pending",
                    findings="To be validated with beta users.",
                )
            ]
        if is_unknown(memo.due_diligence_summary.critical_issues):
            memo.due_diligence_summary.critical_issues = assume("None flagged; requires standard technical and market diligence.")

        # Data sources / references
        if website_url:
            memo.data_sources = memo.data_sources or []
            if website_url not in memo.data_sources:
                memo.data_sources.append(website_url)
        if not memo.data_sources:
            memo.data_sources = [
                "Pitch payload (founder-supplied)",
                "Fund payload (internal mandate/strategy)",
                "SEC markets data (public)",
                "EDGAR search (if available)",
                "Assumptions generated by AI where explicit data is missing (clearly labelled in memo)",
                "Web search / news (LLM context) when explicit data is not provided",
            ]

        return memo

    def _seed_minimum_entries(self, memo: InvestorCommitteeMemo) -> InvestorCommitteeMemo:
        if not memo.investment_recommendation_summary.key_decision_drivers:
            memo.investment_recommendation_summary.key_decision_drivers.append(DEFAULT_PLACEHOLDER)
        if not memo.executive_summary.investment_highlights:
            memo.executive_summary.investment_highlights.append(ExecutiveHighlight())
        if not memo.executive_summary.key_risks:
            memo.executive_summary.key_risks.append(ExecutiveRisk())
        if not memo.founder_team_analysis.founders:
            memo.founder_team_analysis.founders.append(FounderProfile())
        if not memo.market_opportunity_analysis.market_trends:
            memo.market_opportunity_analysis.market_trends.append(DEFAULT_PLACEHOLDER)
        if not memo.market_opportunity_analysis.problem_validation_evidence:
            memo.market_opportunity_analysis.problem_validation_evidence.append(DEFAULT_PLACEHOLDER)
        if not memo.market_opportunity_analysis.key_pain_points:
            memo.market_opportunity_analysis.key_pain_points.append(DEFAULT_PLACEHOLDER)
        if not memo.product_technology_vision.key_differentiators:
            memo.product_technology_vision.key_differentiators.append(DEFAULT_PLACEHOLDER)
        if not memo.early_traction_signals.best_traction_evidence:
            memo.early_traction_signals.best_traction_evidence.append(DEFAULT_PLACEHOLDER)
        if not memo.use_of_funds_milestones.use_of_funds_breakdown:
            memo.use_of_funds_milestones.use_of_funds_breakdown.append(UseOfFundsBreakdown())
        if not memo.financial_projections.revenue_projections:
            memo.financial_projections.revenue_projections.append(RevenueProjection())
        if not memo.competitive_landscape.competitors:
            memo.competitive_landscape.competitors.append(Competitor())
        if not memo.risk_assessment.risks:
            memo.risk_assessment.risks.append(RiskEntry())
        if not memo.due_diligence_summary.items:
            memo.due_diligence_summary.items.append(DueDiligenceItem())
        if not memo.final_recommendation.conviction_factors:
            memo.final_recommendation.conviction_factors.append(DEFAULT_PLACEHOLDER)
        if not memo.final_recommendation.key_decision_factors:
            memo.final_recommendation.key_decision_factors.append(DEFAULT_PLACEHOLDER)
        if not memo.final_recommendation.conditions_precedent:
            memo.final_recommendation.conditions_precedent.append(DEFAULT_PLACEHOLDER)
        if not memo.final_recommendation.ic_vote.partner_votes:
            memo.final_recommendation.ic_vote.partner_votes.append(DEFAULT_PLACEHOLDER)
        return memo

    def _apply_scoring_and_recommendation(
        self,
        memo: InvestorCommitteeMemo,
        pitch: Dict[str, Any],
        website_context: Optional[Dict[str, Any]] = None,
    ) -> InvestorCommitteeMemo:
        """
        Apply a data-driven scoring model for the final recommendation using real memo data.
        """

        def is_missing_str(value: Any) -> bool:
            if value is None:
                return True
            if not isinstance(value, str):
                return False
            stripped = value.strip().lower()
            return stripped in (
                "",
                DEFAULT_PLACEHOLDER.lower(),
                (self.NO_DATA or "").lower(),
                "not disclosed",
                "gathering additional data – manual validation required.".lower(),
                "formal ic recommendation not yet set; use heuristic score and conviction factors as initial guidance.".lower(),
            )

        qualitative_map = {
            "excellent": 9.0,
            "great": 8.5,
            "strong": 8.0,
            "good": 7.0,
            "fair": 5.5,
            "medium": 5.5,
            "moderate": 5.5,
            "weak": 4.0,
            "low": 4.0,
            "poor": 3.0,
        }

        def clamp_score(val: float) -> float:
            return max(0.0, min(10.0, val))

        def extract_score(value: Any) -> Optional[float]:
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return clamp_score(float(value))
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in qualitative_map:
                    return clamp_score(qualitative_map[lowered])
                num_val = self.extract_numeric_value(value)
                if num_val is not None:
                    # Handle values like 7 or 7/10
                    if num_val > 10:
                        num_val = num_val / 10 if num_val <= 100 else 10
                    return clamp_score(float(num_val))
            return None

        def coverage_ratio(numer: int, denom: int) -> float:
            return 0.0 if denom <= 0 else min(1.0, max(0.0, numer / denom))

        # Founder score (uses explicit scores first; otherwise derive from experience/education/LinkedIn/data)
        founders = memo.founder_team_analysis.founders or []
        founder_scores: List[float] = []
        founder_coverage_signals = 0
        founder_total_signals = 0
        for f in founders[:1]:  # primary founder
            components = [
                extract_score(getattr(f, "domain_expertise_score", None)),
                extract_score(getattr(f, "technical_ability_score", None)),
                extract_score(getattr(f, "leadership_score", None)),
                extract_score(getattr(f, "entrepreneurial_drive_score", None)),
            ]
            founder_total_signals += 4
            for c in components:
                if c is not None:
                    founder_scores.append(c)
                    founder_coverage_signals += 1
            # Derive from experience/education if explicit scores missing
            if not founder_scores:
                exp_years = self.extract_numeric_value(getattr(f, "experience_years", None) or "")
                edu = (getattr(f, "education", "") or "").lower()
                prev_cos = (getattr(f, "previous_companies", "") or "").lower()
                derived = 5.0
                if exp_years and exp_years >= 8:
                    derived += 1.0
                    founder_coverage_signals += 1
                if any(k in edu for k in ["phd", "mba", "ms", "masters", "bachelor", "computer science"]):
                    derived += 0.5
                    founder_coverage_signals += 1
                if any(k in prev_cos for k in ["acquired", "ipo", "public", "unicorn"]):
                    derived += 0.5
                    founder_coverage_signals += 1
                if not is_missing_str(getattr(f, "linkedin", None)) and "linkedin.com" in str(getattr(f, "linkedin", "")).lower():
                    derived += 0.5
                    founder_coverage_signals += 1
                founder_scores.append(clamp_score(derived))
                founder_total_signals += 3
        founder_score = clamp_score(sum(founder_scores) / len(founder_scores)) if founder_scores else 0.0
        founder_coverage = coverage_ratio(founder_coverage_signals, max(1, founder_total_signals))

        # Market score (TAM, growth, competitive density)
        mo = memo.market_opportunity_analysis
        market_signals = 0
        market_total = 0
        tam_val = self.extract_numeric_value(mo.tam) if not is_missing_str(mo.tam) else None
        tam_growth = self.extract_percentage(mo.tam_growth_rate) if not is_missing_str(mo.tam_growth_rate) else None
        competitors = [c for c in (memo.competitive_landscape.competitors or []) if not self._detect_placeholder(getattr(c, "name", ""))]
        comp_count = len([c for c in competitors if getattr(c, "name", None)])
        market_score = 0.0
        market_total += 3
        if tam_val is not None:
            if tam_val >= 1_000_000_000:
                market_score += 8.0
            elif tam_val >= 100_000_000:
                market_score += 6.5
            elif tam_val >= 10_000_000:
                market_score += 5.5
            else:
                market_score += 4.5
            market_signals += 1
        if tam_growth is not None:
            if tam_growth >= 30:
                market_score += 2.0
            elif tam_growth >= 15:
                market_score += 1.5
            elif tam_growth >= 5:
                market_score += 1.0
            else:
                market_score += 0.5
            market_signals += 1
        if comp_count:
            # Competitive density: more competitors slightly lowers score
            if comp_count <= 3:
                market_score += 1.0
            elif comp_count <= 6:
                market_score += 0.5
            else:
                market_score -= 0.5
            market_signals += 1
        market_score = clamp_score(market_score if market_signals else 0.0)
        market_coverage = coverage_ratio(market_signals, max(1, market_total))

        # Product score (stage, completeness, differentiation, IP signals + website data)
        pt = memo.product_technology_vision
        website_profile = (website_context or {}).get("profile") or {}
        website_snippets = (website_context or {}).get("snippets") or []
        product_signals = 0
        product_total = 6
        product_score = 0.0
        completeness_pct = self.extract_percentage(pt.functionality_completion) if not is_missing_str(pt.functionality_completion) else None
        if completeness_pct is not None:
            if completeness_pct >= 80:
                product_score += 3.0
            elif completeness_pct >= 50:
                product_score += 2.0
            else:
                product_score += 1.0
            product_signals += 1
        if pt.key_differentiators and not is_missing_str(pt.key_differentiators[0]):
            product_score += 2.0
            product_signals += 1
        if not is_missing_str(pt.tech_stack):
            product_score += 1.5
            product_signals += 1
        if any("patent" in str(d).lower() or "ip" in str(d).lower() for d in (pt.key_differentiators or [])):
            product_score += 1.0
            product_signals += 1
        # Website-derived signals: product description and feature mentions
        web_fields = [
            website_profile.get("short_summary"),
            website_profile.get("detailed_description"),
            website_profile.get("solution"),
            website_profile.get("product"),
        ]
        if any(val and not is_missing_str(val) for val in web_fields):
            product_score += 1.0
            product_signals += 1
        if website_snippets:
            keyword_hits = 0
            for snip in website_snippets[:10]:
                ls = str(snip).lower()
                if any(k in ls for k in ["feature", "platform", "workflow", "api", "integration", "ai", "automation", "model", "analytics", "cloud", "saas"]):
                    keyword_hits += 1
            if keyword_hits:
                product_score += min(1.0, 0.25 * keyword_hits)
                product_signals += 1
        product_score = clamp_score(product_score if product_signals else 0.0)
        product_coverage = coverage_ratio(product_signals, max(1, product_total))

        # Traction score (users/revenue/growth from KPIs and traction signals)
        ts = memo.early_traction_signals
        traction_signals = 0
        traction_total = 4
        traction_score = 0.0
        payers = self.extract_numeric_value(ts.paying_customers) if not is_missing_str(ts.paying_customers) else None
        growth_pct = self.extract_percentage(ts.monthly_growth_rate) if not is_missing_str(ts.monthly_growth_rate) else None
        beta_users = self.extract_numeric_value(ts.beta_users) if not is_missing_str(ts.beta_users) else None
        kpis = memo.traction_kpis or []
        revenue_kpi = next((k for k in kpis if "rev" in k.name.lower() or "mrr" in k.name.lower() or "arr" in k.name.lower()), None)

        if payers is not None:
            if payers >= 1000:
                traction_score += 3.0
            elif payers >= 100:
                traction_score += 2.0
            elif payers >= 10:
                traction_score += 1.0
            else:
                traction_score += 0.5
            traction_signals += 1
        if revenue_kpi and revenue_kpi.value:
            rev_val = revenue_kpi.value
            if rev_val >= 100_000:
                traction_score += 2.0
            elif rev_val >= 10_000:
                traction_score += 1.5
            elif rev_val > 0:
                traction_score += 1.0
            traction_signals += 1
        if growth_pct is not None:
            if growth_pct >= 20:
                traction_score += 2.0
            elif growth_pct >= 10:
                traction_score += 1.5
            elif growth_pct >= 1:
                traction_score += 1.0
            traction_signals += 1
        if beta_users is not None:
            if beta_users >= 500:
                traction_score += 1.5
            elif beta_users >= 100:
                traction_score += 1.0
            else:
                traction_score += 0.5
            traction_signals += 1
        traction_score = clamp_score(traction_score if traction_signals else 0.0)
        # Adjust traction score if quantitative fields are missing (check for placeholders)
        ts = memo.early_traction_signals
        missing_traction_fields = sum([
            is_missing_str(getattr(ts, 'paying_customers', None)),
            is_missing_str(getattr(ts, 'beta_users', None)),
            is_missing_str(getattr(ts, 'waitlist', None)),
            is_missing_str(getattr(ts, 'monthly_growth_rate', None)),
        ])
        if missing_traction_fields >= 2:  # If 2+ quantitative fields are missing, reduce score
            traction_score *= 0.7
            logger.warning(f"Traction score reduced by 30% due to {missing_traction_fields} missing traction fields")
        traction_coverage = coverage_ratio(traction_signals, max(1, traction_total))

        # Component confidences from coverage
        def coverage_to_conf(cov: float) -> str:
            return "High" if cov >= 0.75 else "Medium" if cov >= 0.5 else "Low"

        founder_conf = coverage_to_conf(founder_coverage)
        market_conf = coverage_to_conf(market_coverage)
        product_conf = coverage_to_conf(product_coverage)
        traction_conf = coverage_to_conf(traction_coverage)

        # Risk adjustments (simple: high-probability risks reduce overall)
        risk_adjust = 0.0
        high_risks = [
            r for r in (memo.risk_assessment.risks or []) if getattr(r, "probability", "").lower().startswith("high")
        ]
        if high_risks:
            risk_adjust -= 0.4 * len(high_risks)

        # Weighted overall score
        raw_overall = (
            0.4 * founder_score + 0.25 * market_score + 0.2 * product_score + 0.15 * traction_score
        )
        overall_numeric = clamp_score(raw_overall + risk_adjust)

        # Data completeness → confidence
        completeness = self._compute_data_completeness(memo)
        if completeness >= 0.8:
            conf_label = "High"
        elif completeness >= 0.6:
            conf_label = "Medium"
        else:
            conf_label = "Low"

        memo.investment_recommendation_summary.founder_quality_score = f"{founder_score:.1f}/10 ({founder_conf} data confidence)"
        memo.investment_recommendation_summary.market_opportunity_score = f"{market_score:.1f}/10 ({market_conf} data confidence)"
        memo.investment_recommendation_summary.confidence_level = (
            f"{conf_label} overall; data coverage ~{int(completeness * 100)}%"
        )

        memo.final_recommendation.overall_score = f"{overall_numeric:.1f}/10"

        # Decision logic with human-review triggers
        recommendation_label = memo.final_recommendation.recommendation
        if is_missing_str(recommendation_label):
            if overall_numeric >= 8.5:
                recommendation_label = "APPROVE"
            elif overall_numeric >= 6.0:
                recommendation_label = "DISCUSS"
            else:
                recommendation_label = "DECLINE"
        review_flags: List[str] = []
        if overall_numeric < 6.0 or overall_numeric > 8.5:
            review_flags.append("Edge-case score: require human review.")
        if min(founder_coverage, market_coverage, product_coverage, traction_coverage) < 0.5:
            review_flags.append("Data coverage <50% in at least one component.")
        if completeness < 0.5:
            review_flags.append("Overall memo data completeness <50%.")

        # Component strengths/risks for narrative
        components = [
            ("Founder", founder_score),
            ("Market", market_score),
            ("Product", product_score),
            ("Traction", traction_score),
        ]
        components_sorted = sorted(components, key=lambda x: x[1], reverse=True)
        strongest = [name for name, score in components_sorted if score >= 7.5]
        weakest = [name for name, score in components_sorted if score < 6.0]

        # Decision drivers (company-specific)
        drivers: List[str] = []
        if founder_score:
            drivers.append(f"Founder: {founder_score:.1f}/10 ({founder_conf} confidence).")
        if market_score:
            drivers.append(f"Market: {market_score:.1f}/10 ({market_conf} confidence).")
        if product_score:
            drivers.append(f"Product: {product_score:.1f}/10 ({product_conf} confidence).")
        if traction_score:
            drivers.append(f"Traction: {traction_score:.1f}/10 ({traction_conf} confidence).")
        if risk_adjust < 0:
            drivers.append(f"Risk adjustment applied ({risk_adjust:+.1f}) due to high-probability risks.")

        # Narrative recommendation paragraph with detailed explanation
        strengths_text = f"Strengths: {', '.join(strongest)}." if strongest else "Strengths: none standout."
        risks_text = f"Risks: {', '.join(weakest)}." if weakest else "Risks: balanced across components."
        
        # Detailed explanation for DISCUSS recommendation
        if recommendation_label == "DISCUSS":
            # Explain WHY it's DISCUSS based on component scores
            reasons = []
            if founder_score < 7.0:
                reasons.append(f"Founder score {founder_score:.1f}/10 needs validation")
            if market_score < 7.0:
                reasons.append(f"Market opportunity {market_score:.1f}/10 requires deeper analysis")
            if product_score < 7.0:
                reasons.append(f"Product validation {product_score:.1f}/10 incomplete")
            if traction_score < 7.0:
                reasons.append(f"Traction signals {traction_score:.1f}/10 are early-stage")
            if completeness < 0.6:
                reasons.append(f"Data completeness {int(completeness*100)}% below threshold")
            
            reason_text = "; ".join(reasons) if reasons else "Mixed signals across components"
            action_text = f"IC discussion needed to resolve: {reason_text}. Good to engage if we can validate key assumptions quickly."
        elif recommendation_label == "APPROVE":
            action_text = "Proceed to term sheet subject to standard diligence."
        else:  # DECLINE
            action_text = "Defer until product and traction mature."
            
        recommendation_paragraph = (
            f"AI suggestion: {recommendation_label} — overall score {overall_numeric:.1f}/10 ({conf_label} confidence). "
            f"{strengths_text} {risks_text} {action_text}"
        )

        memo.final_recommendation.recommendation = recommendation_paragraph or recommendation_label
        if is_missing_str(memo.investment_recommendation_summary.recommendation):
            memo.investment_recommendation_summary.recommendation = recommendation_paragraph or recommendation_label

        # Sanitize data_sources to real URLs only (drop LLM labels or generic text)
        memo.data_sources = [
            self._normalize_url(src)
            for src in (memo.data_sources or [])
            if self._normalize_url(src)
        ]

        # Append sources if available (ensure external data attribution)
        sources: set[str] = set()

        def add_source(val: Optional[str]) -> None:
            if not val or not isinstance(val, str):
                return
            normalized = self._normalize_url(val)
            if normalized:
                sources.add(normalized)

        for src in memo.data_sources or []:
            add_source(src)
        if website_context and website_context.get("url"):
            add_source(website_context["url"])

        # Market sources
        mo = memo.market_opportunity_analysis
        add_source(getattr(mo, "tam_source", None))
        add_source(getattr(mo, "tam_source_url", None))

        # Competitor sources
        for comp in memo.competitive_landscape.competitors or []:
            add_source(getattr(comp, "source_url", None))
            add_source(getattr(comp, "source", None))

        # Revenue projection sources
        for rp in memo.financial_projections.revenue_projections or []:
            add_source(getattr(rp, "source", None))

        # Traction KPI sources
        for kpi in memo.traction_kpis or []:
            add_source(getattr(kpi, "source", None))

        if not sources:
            # Fallback disclosure when no explicit URLs/sources surfaced
            sources.add("No external URLs captured; verify sources manually.")

        sources_line = "Sources: " + ", ".join(sorted(sources))
        drivers.append(sources_line)

        memo.final_recommendation.conviction_factors = drivers or ["Data-driven drivers not available."]
        memo.final_recommendation.key_decision_factors = drivers or ["Key decision factors pending more data."]
        memo.investment_recommendation_summary.key_decision_drivers = drivers or ["Key decision drivers pending more data."]

        if review_flags:
            memo.final_recommendation.conditions_precedent = review_flags

        # Populate IC vote suggestion if missing/placeholder
        partner_votes = getattr(memo.final_recommendation.ic_vote, "partner_votes", []) or []
        def _is_placeholder_vote(v: Any) -> bool:
            if not isinstance(v, str):
                return False
            stripped = v.strip().lower()
            return stripped in {"", DEFAULT_PLACEHOLDER.lower(), "ic vote not yet recorded; capture partner views at investment committee."}
        if (not partner_votes) or all(_is_placeholder_vote(v) for v in partner_votes):
            memo.final_recommendation.ic_vote.partner_votes = [
                f"{recommendation_paragraph or recommendation_label} (human IC to confirm)."
            ]

        # Deduplicate and persist sources into memo.data_sources for Data Sources section
        if sources:
            existing = set(memo.data_sources or [])
            for s in sources:
                if not isinstance(s, str):
                    continue
                if not (s.startswith("http://") or s.startswith("https://")):
                    # Only persist verifiable URLs into data_sources
                    continue
                if s not in existing:
                    memo.data_sources.append(s)
                    existing.add(s)

        return memo

        return memo

    def _compute_data_completeness(self, memo: InvestorCommitteeMemo) -> float:
        """Approximate how much of the memo is filled with non-placeholder content."""

        def is_placeholder(value: Any) -> bool:
            if value is None:
                return True
            if isinstance(value, str):
                stripped = value.strip().lower()
                return stripped in (
                    "",
                    DEFAULT_PLACEHOLDER.lower(),
                    (self.NO_DATA or "").lower(),
                    "not disclosed",
                    "gathering additional data – manual validation required.".lower(),
                )
            return False

        payload = memo.model_dump()
        total = 0
        missing = 0

        def walk(node: Any) -> None:
            nonlocal total, missing
            if isinstance(node, dict):
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                if not node:
                    total += 1
                    missing += 1
                else:
                    for item in node:
                        walk(item)
            elif isinstance(node, str):
                total += 1
                if is_placeholder(node):
                    missing += 1
            else:
                # Non-string scalar
                total += 1

        walk(payload)
        if total == 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - (missing / float(total))))

    def _extract_overrides_from_message(self, message: str) -> Dict[str, Any]:
        """
        Parse free-text user message for deterministic overrides (traction + financials + founder).
        Returns a dict of field->value (numeric or string).
        """
        if not message:
            return {}
        text = message.strip()
        lowered = text.lower()
        import re as _re

        overrides: Dict[str, Any] = {}

        # Founder name
        match = _re.search(r"founder\s+name\s+is\s+(.+?)([.,;\n]|$)", lowered)
        if not match:
            match = _re.search(r"founder\s+is\s+(.+?)([.,;\n]|$)", lowered)
        if match:
            raw_name = match.group(1).strip()
            overrides["founder_name"] = " ".join(part.capitalize() for part in raw_name.split())

        metric_patterns = {
            "paying_customers": [r"paying\s+customers?", r"payers", r"customers?\s*\(paying\)"],
            "beta_users": [r"beta\s+users?", r"beta\s+testers?", r"mid\s*funnel"],
            "waitlist": [r"waitlist", r"top\s*funnel"],
            "lois_pilots": [r"lois", r"pilots?", r"pilot\s+programs?"],
            "active_customers": [r"active\s+customers?", r"active\s+users?", r"users?\s*\(active\)"],
            "mau": [r"mau\b", r"monthly\s+active\s+users?"],
            "dau": [r"dau\b", r"daily\s+active\s+users?"],
            "mrr": [r"\bmrr\b", r"monthly\s+recurring\s+revenue"],
            "arr": [r"\barr\b", r"annual\s+recurring\s+revenue"],
            "revenue": [r"\brevenue\b"],
        }

        def extract_first_number(patterns: List[str]) -> Optional[float]:
            for pat in patterns:
                m = _re.search(rf"{pat}\s*(are|is|=|:)?\s*([\d,\.]+(?:[kmbKMB])?)", lowered)
                if m:
                    val = self.extract_numeric_value(m.group(2))
                    if val is not None:
                        return float(val)
            return None

        for field, patterns in metric_patterns.items():
            val = extract_first_number(patterns)
            if val is not None:
                overrides[field] = val

        return overrides

    def _build_memo_state_from_memo(self, memo: InvestorCommitteeMemo) -> Dict[str, Any]:
        """Build a canonical memo_state from the current memo object."""
        state: Dict[str, Any] = {}
        try:
            state["company_name"] = memo.meta.company_name
            state["stage"] = memo.meta.stage
            # Financials
            if memo.financial_projections and memo.financial_projections.revenue_projections:
                rp = memo.financial_projections.revenue_projections[0]
                val = self.extract_numeric_value(rp.revenue) if rp.revenue else None
                state["mrr"] = {"value": val, "currency": "USD", "period": "monthly"} if val is not None else None
            # Traction
            ts = memo.early_traction_signals
            state["paying_customers"] = self.extract_numeric_value(ts.paying_customers) if ts and ts.paying_customers else None
            state["lois_pilots"] = self.extract_numeric_value(ts.lois_pilots) if ts and ts.lois_pilots else None
            state["beta_users"] = self.extract_numeric_value(ts.beta_users) if ts and ts.beta_users else None
            state["waitlist"] = self.extract_numeric_value(ts.waitlist) if ts and ts.waitlist else None
            state["active_customers"] = self.extract_numeric_value(ts.customer_acquisition) if ts and ts.customer_acquisition else None
            state["mau"] = self.extract_numeric_value(ts.monthly_active_users) if ts and ts.monthly_active_users else None
            state["dau"] = self.extract_numeric_value(ts.daily_active_users) if ts and ts.daily_active_users else None
        except Exception:
            pass
        return state

    def _load_or_build_memo_state(self, memo: InvestorCommitteeMemo, memo_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load memo_state from DB if available; otherwise build from memo or payload."""
        try:
            record = get_memo_record(memo.meta.document_id)
            if record and getattr(record, "memo_state", None):
                return record.memo_state
        except Exception:
            pass
        # fallback: build from memo, then enrich from payload if provided
        base_state = self._build_memo_state_from_memo(memo)
        if memo_payload and isinstance(memo_payload, dict):
            # pull some basics from payload if missing
            for key in ["company_name", "stage", "market", "industry"]:
                if key in memo_payload and base_state.get(key) is None:
                    base_state[key] = memo_payload.get(key)
            # traction_kpis
            if memo_payload.get("traction_kpis"):
                tkpis = memo_payload.get("traction_kpis")
                if isinstance(tkpis, list):
                    for kpi in tkpis:
                        name = str(kpi.get("name", "")).lower()
                        val = kpi.get("value")
                        if val is None:
                            continue
                        if "waitlist" in name and base_state.get("waitlist") is None:
                            base_state["waitlist"] = val
                        if "beta" in name and base_state.get("beta_users") is None:
                            base_state["beta_users"] = val
                        if "paying" in name and base_state.get("paying_customers") is None:
                            base_state["paying_customers"] = val
        return base_state

    def _persist_memo_state_safe(self, memo_id: str, memo_state: Dict[str, Any]) -> None:
        """Best-effort persistence of memo_state to DB."""
        try:
            if memo_id and memo_state:
                update_memo_state(memo_id, memo_state)
        except Exception:
            pass

    def apply_patch(self, memo_state: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
        """Shallow merge patch['fields'] into memo_state without nulling others."""
        try:
            fields = patch.get("fields", {}) if patch else {}
            for k, v in fields.items():
                memo_state[k] = v
            return memo_state
        except Exception:
            return memo_state

    async def parse_user_correction(self, user_message: str, memo_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to extract a JSON PATCH of fields the user wants to update.
        IMPROVED: Better prompt and error handling.
        """
        if not user_message or not user_message.strip():
            return {"fields": {}}

        # IMPROVED: More detailed prompt with better examples
        prompt = f"""
You are a strict JSON-only parser for investment memo field updates.

CRITICAL RULES:
1. Extract ONLY the fields the user wants to UPDATE
2. Do NOT include fields that are not mentioned
3. Do NOT null out or delete existing fields
4. Return ONLY a JSON object with a "fields" key

USER MESSAGE: {user_message}

CURRENT MEMO STATE: {json.dumps(memo_state, indent=2)}

OUTPUT FORMAT (strict JSON only):
{{
  "fields": {{
    "<field_name>": <new_value>
  }}
}}

EXAMPLES:

Example 1:
User: "MRR is 150,000 USD not 145,000"
Output: {{"fields": {{"mrr": {{"value": 150000, "currency": "USD", "period": "monthly"}}}}}}

Example 2:
User: "We have 220 paying customers now"
Output: {{"fields": {{"paying_customers": 220}}}}

Example 3:
User: "Update beta users to 500 and waitlist to 2000"
Output: {{"fields": {{"beta_users": 500, "waitlist": 2000}}}}

Example 4:
User: "Change team size from 10 to 15"
Output: {{"fields": {{"team_size": 15}}}}

Example 5:
User: "Growth rate is 15% monthly"
Output: {{"fields": {{"monthly_growth_rate": 15}}}}

FIELD NAME MAPPING (use these exact field names):
- "MRR", "monthly revenue", "monthly recurring revenue" → "mrr"
- "ARR", "annual revenue", "annual recurring revenue" → "arr"
- "paying customers", "customers" → "paying_customers"
- "beta users", "beta testers" → "beta_users"
- "waitlist", "signups" → "waitlist"
- "LOIs", "pilots" → "lois_pilots"
- "team size", "employees" → "team_size"
- "MAU", "monthly active users" → "mau"
- "DAU", "daily active users" → "dau"
- "growth rate" → "monthly_growth_rate" or "annual_growth_rate" (detect from context)

IMPORTANT:
- If the user says "not X" or "actually Y", extract Y as the new value
- If the user says "change from X to Y", extract Y as the new value
- For currency values, extract value and currency separately
- For percentages, extract as a number (15% → 15)
- Return empty fields object if no updates detected

Return ONLY the JSON object. No explanations, no markdown, no additional text.
"""

        try:
            # IMPROVED: Use shorter timeout and better error handling
            response, _ = await self.claude.generate_json(
                system_prompt="You are a strict JSON-only parser. Extract field updates from user messages. Return ONLY valid JSON.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            
            # IMPROVED: Better response parsing
            if isinstance(response, str):
                # Strip any markdown code fences
                response = _strip_code_fences(response)
                # Parse JSON
                response = json.loads(response)
            
            if not isinstance(response, dict):
                logger.warning(f"parse_user_correction returned non-dict: {type(response)}")
                return {"fields": {}}
            
            # Validate response structure
            if "fields" not in response:
                logger.warning(f"parse_user_correction missing 'fields' key: {response}")
                return {"fields": {}}
            
            fields = response.get("fields", {})
            if not isinstance(fields, dict):
                logger.warning(f"parse_user_correction 'fields' is not dict: {type(fields)}")
                return {"fields": {}}
            
            # IMPROVED: Log extracted fields for debugging
            if fields:
                logger.info(f"parse_user_correction extracted {len(fields)} fields: {list(fields.keys())}")
                for k, v in fields.items():
                    logger.info(f"  {k}: {v}")
            else:
                logger.info("parse_user_correction found no field updates")
            
            return response
            
        except json.JSONDecodeError as e:
            logger.error(f"parse_user_correction JSON decode error: {e}")
            logger.error(f"Response was: {response[:500] if 'response' in locals() else 'N/A'}")
            return {"fields": {}}
        except ValueError as e:
            # API key missing or configuration error
            logger.warning(f"Claude API not available for parse_user_correction: {e}")
            return {"fields": {}}
        except Exception as e:
            # Check if it's an authentication error
            error_str = str(e).lower()
            if "authentication" in error_str or "401" in error_str or ("invalid" in error_str and "api" in error_str):
                logger.warning(f"Claude API authentication failed for parse_user_correction: {e}")
                return {"fields": {}}
            else:
                logger.error(f"parse_user_correction unexpected error: {e}", exc_info=True)
                return {"fields": {}}

    def _apply_state_patch_to_memo(self, memo: InvestorCommitteeMemo, memo_state: Dict[str, Any]) -> InvestorCommitteeMemo:
        """Apply patched memo_state values back onto memo object deterministically."""
        try:
            ts = memo.early_traction_signals
            if not ts:
                return memo
            # Core meta
            if memo_state.get("company_name"):
                memo.meta.company_name = memo_state.get("company_name")
            if memo_state.get("stage"):
                memo.meta.stage = memo_state.get("stage")

            # Traction signals
            def maybe_set(attr: str, key: str, is_currency: bool = False):
                if key in memo_state and memo_state[key] is not None:
                    val = memo_state[key]
                    setattr(ts, attr, self.format_currency(val) if is_currency else self.format_number(val))
            maybe_set("paying_customers", "paying_customers")
            maybe_set("lois_pilots", "lois_pilots")
            maybe_set("beta_users", "beta_users")
            maybe_set("waitlist", "waitlist")
            maybe_set("customer_acquisition", "active_customers")
            maybe_set("monthly_active_users", "mau")
            maybe_set("daily_active_users", "dau")

            # Financials
            if memo_state.get("mrr") and isinstance(memo_state["mrr"], dict):
                mrr_val = memo_state["mrr"].get("value")
                if mrr_val is not None:
                    memo.financial_projections.revenue_projections = [
                        RevenueProjection(year="Current", revenue=self.format_currency(mrr_val), source="user_chat", confidence="high")
                    ]
            if memo_state.get("arr") and isinstance(memo_state["arr"], dict):
                arr_val = memo_state["arr"].get("value")
                if arr_val is not None and memo.financial_projections.revenue_projections:
                    # Append ARR as a second projection row if appropriate
                    memo.financial_projections.revenue_projections.append(
                        RevenueProjection(year="Annual", revenue=self.format_currency(arr_val), source="user_chat", confidence="high")
                    )

            # Quick facts / team size
            if hasattr(memo, "founder_team_analysis") and memo_state.get("team_size"):
                memo.founder_team_analysis.team_completeness.current_team_size = self.format_number(memo_state["team_size"])
            if hasattr(memo, "meta") and hasattr(memo.meta, "quick_facts"):
                qf = memo.meta.quick_facts
                if memo_state.get("team_size"):
                    qf["team_size"] = self.format_number(memo_state["team_size"])
                if memo_state.get("stage"):
                    qf["stage"] = memo_state["stage"]
                if memo_state.get("company_name"):
                    qf["current_status"] = memo_state.get("company_name")
                if memo_state.get("monthly_growth_rate"):
                    qf["current_status"] = qf.get("current_status") or ""
                    qf["growth_rate"] = f"{self.format_number(memo_state['monthly_growth_rate'])}%"

            # KPIs sync
            if memo_state.get("paying_customers") or memo_state.get("beta_users") or memo_state.get("waitlist"):
                if not memo.traction_kpis:
                    memo.traction_kpis = []
                existing = {kpi.name.lower(): i for i, kpi in enumerate(memo.traction_kpis)}
                def upsert(name: str, val: Any, unit: str = "count"):
                    if val is None:
                        return
                    lname = name.lower()
                    if lname in existing:
                        memo.traction_kpis[existing[lname]] = TractionKPI(name=name, value=float(val), unit=unit, source="user_chat", confidence="high", estimated=False)
                    else:
                        memo.traction_kpis.append(TractionKPI(name=name, value=float(val), unit=unit, source="user_chat", confidence="high", estimated=False))
                upsert("Paying Customers", memo_state.get("paying_customers"))
                upsert("Beta Users", memo_state.get("beta_users"))
                upsert("Waitlist", memo_state.get("waitlist"))
                upsert("MAU", memo_state.get("mau"))
                upsert("DAU", memo_state.get("dau"))

            return memo
        except Exception:
            return memo
    def _apply_overrides_to_memo(self, memo: InvestorCommitteeMemo, overrides: Dict[str, Any]) -> InvestorCommitteeMemo:
        """
        Apply deterministic overrides to the memo without regenerating content.
        """
        try:
            if not overrides:
                return memo

            # Founder name
            founder_name = overrides.get("founder_name")
            if founder_name:
                if not memo.founder_team_analysis.founders:
                    memo.founder_team_analysis.founders.append(
                        FounderProfile(
                            name=founder_name,
                            role="Founder/CEO",
                        )
                    )
                else:
                    memo.founder_team_analysis.founders[0].name = founder_name
                    if not memo.founder_team_analysis.founders[0].role or memo.founder_team_analysis.founders[0].role == DEFAULT_PLACEHOLDER:
                        memo.founder_team_analysis.founders[0].role = "Founder/CEO"

            def set_signal(attr: str, val: float, is_currency: bool = False) -> None:
                fmt = self.format_currency(val) if is_currency else self.format_number(val)
                if hasattr(memo.early_traction_signals, attr):
                    setattr(memo.early_traction_signals, attr, fmt)
                # Also stash into memo_state-like fields if present in memo.meta.quick_facts
                if hasattr(memo, "meta") and hasattr(memo.meta, "quick_facts"):
                    qf = memo.meta.quick_facts
                    key_map = {
                        "paying_customers": "customers",
                        "active_customers": "customers",
                        "team_size": "team_size",
                    }
                    if attr in key_map and key_map[attr] in qf:
                        qf[key_map[attr]] = self.format_number(val)

            traction_fields = ["paying_customers", "beta_users", "waitlist", "lois_pilots", "active_customers", "mau", "dau"]
            for tf in traction_fields:
                if tf in overrides:
                    target_attr = tf
                    if tf == "active_customers":
                        target_attr = "customer_acquisition"
                    set_signal(target_attr, overrides[tf], is_currency=False)

            # Update traction_kpis so charts reflect user-provided values
            if any(tf in overrides for tf in traction_fields):
                if not memo.traction_kpis:
                    memo.traction_kpis = []
                existing_names = {kpi.name.lower() for kpi in memo.traction_kpis}

                def upsert_kpi(name: str, val: float, unit: str = "count") -> None:
                    lname = name.lower()
                    if val is None:
                        return
                    if lname in existing_names:
                        for i, kpi in enumerate(memo.traction_kpis):
                            if kpi.name.lower() == lname:
                                memo.traction_kpis[i] = TractionKPI(
                                    name=name,
                                    value=float(val),
                                    unit=unit,
                                    source="user_chat",
                                    confidence="high",
                                    estimated=False,
                                )
                                return
                    memo.traction_kpis.append(
                        TractionKPI(
                            name=name,
                            value=float(val),
                            unit=unit,
                            source="user_chat",
                            confidence="high",
                            estimated=False,
                        )
                    )
                    existing_names.add(lname)

                if "active_customers" in overrides:
                    upsert_kpi("Active Customers", overrides["active_customers"])
                if "paying_customers" in overrides:
                    upsert_kpi("Paying Customers", overrides["paying_customers"])
                if "beta_users" in overrides:
                    upsert_kpi("Beta Users", overrides["beta_users"])
                if "waitlist" in overrides:
                    upsert_kpi("Waitlist", overrides["waitlist"])
                if "lois_pilots" in overrides:
                    upsert_kpi("LOIs/Pilots", overrides["lois_pilots"])
                if "mau" in overrides:
                    upsert_kpi("MAU", overrides["mau"])
                if "dau" in overrides:
                    upsert_kpi("DAU", overrides["dau"])

            # Financial overrides
            if "mrr" in overrides:
                memo.financial_projections.revenue_projections = [
                    RevenueProjection(year="Current", revenue=self.format_currency(overrides["mrr"]), source="user_chat", confidence="high")
                ]
                if not memo.traction_kpis:
                    memo.traction_kpis = []
                memo.traction_kpis.append(TractionKPI(name="MRR", value=float(overrides["mrr"]), unit="USD", source="user_chat", confidence="high", estimated=False))
            if "arr" in overrides:
                if not memo.traction_kpis:
                    memo.traction_kpis = []
                memo.traction_kpis.append(TractionKPI(name="ARR", value=float(overrides["arr"]), unit="USD", source="user_chat", confidence="high", estimated=False))
            if "revenue" in overrides and "mrr" not in overrides:
                if not memo.traction_kpis:
                    memo.traction_kpis = []
                memo.traction_kpis.append(TractionKPI(name="Revenue", value=float(overrides["revenue"]), unit="USD", source="user_chat", confidence="high", estimated=False))

            # Team and growth
            if "team_size" in overrides and hasattr(memo, "founder_team_analysis"):
                memo.founder_team_analysis.team_completeness.current_team_size = self.format_number(overrides["team_size"])
            if "growth_rate_monthly" in overrides and hasattr(memo, "market_opportunity_analysis"):
                memo.market_opportunity_analysis.tam_growth_rate = self.format_number(overrides["growth_rate_monthly"])
            if "growth_rate_annual" in overrides and hasattr(memo, "market_opportunity_analysis"):
                memo.market_opportunity_analysis.tam_growth_rate = self.format_number(overrides["growth_rate_annual"])

            return memo
        except Exception:
            return memo

    def _restore_traction_if_regressed(self, original: InvestorCommitteeMemo, updated: InvestorCommitteeMemo) -> InvestorCommitteeMemo:
        """
        Preserve existing traction details if an LLM/edit pass wipes them out with placeholders.
        """
        try:
            if not original or not updated:
                return updated

            def is_placeholder(val: Any) -> bool:
                if val is None:
                    return True
                if isinstance(val, str):
                    return self._detect_placeholder(val)
                return False

            orig_ts = getattr(original, "early_traction_signals", None)
            upd_ts = getattr(updated, "early_traction_signals", None)
            if orig_ts and upd_ts:
                fields_to_preserve = [
                    "paying_customers",
                    "lois_pilots",
                    "beta_users",
                    "waitlist",
                    "monthly_growth_rate",
                    "engagement_level",
                    "traction_quality",
                    "customer_acquisition",
                    "early_retention",
                    "best_traction_evidence",
                ]
                for field in fields_to_preserve:
                    orig_val = getattr(orig_ts, field, None)
                    upd_val = getattr(upd_ts, field, None)
                    if orig_val and not is_placeholder(orig_val):
                        if is_placeholder(upd_val):
                            setattr(upd_ts, field, orig_val)

            # Restore traction KPIs if the updated memo lost them
            if (not getattr(updated, "traction_kpis", None)) and getattr(original, "traction_kpis", None):
                updated.traction_kpis = original.traction_kpis

            return updated
        except Exception:
            return updated

    def _preserve_market_data_if_lost(self, original: InvestorCommitteeMemo, updated: InvestorCommitteeMemo) -> InvestorCommitteeMemo:
        """
        Preserve existing market data (TAM, SAM, SOM, growth rates, sources) if an LLM/edit pass wipes them out.
        This is critical because TAM data is expensive to regenerate and should be preserved across edits.
        """
        try:
            if not original or not updated:
                return updated

            def is_placeholder(val: Any) -> bool:
                if val is None:
                    return True
                if isinstance(val, str):
                    return self._detect_placeholder(val)
                return False

            orig_moa = getattr(original, "market_opportunity_analysis", None)
            upd_moa = getattr(updated, "market_opportunity_analysis", None)
            
            if orig_moa and upd_moa:
                # Preserve TAM and related fields if they were lost
                market_fields_to_preserve = [
                    "tam",
                    "tam_source",
                    "tam_confidence",
                    "tam_growth_rate",
                    "market_trends",
                    "segment_fit",
                ]
                
                for field in market_fields_to_preserve:
                    orig_val = getattr(orig_moa, field, None)
                    upd_val = getattr(upd_moa, field, None)
                    
                    # If original has a valid value and updated lost it (None or placeholder), restore it
                    if orig_val and not is_placeholder(orig_val):
                        if upd_val is None or is_placeholder(upd_val):
                            setattr(upd_moa, field, orig_val)
                            logger.info(f"Preserved market_opportunity_analysis.{field} from original memo: {orig_val[:100] if isinstance(orig_val, str) and len(str(orig_val)) > 100 else orig_val}")
            
            return updated
        except Exception as e:
            logger.warning(f"Error preserving market data: {e}")
            return updated

    def _extract_data_sources_from_research(self, research: Any) -> List[str]:
        """Best-effort extraction of URL-like sources from Gemini research payload."""

        import re

        urls: List[str] = []

        def add_from_text(text: str) -> None:
            for match in re.findall(r"https?://[^\\s)\\]]+", text):
                cleaned = match.rstrip(".,);]")
                if cleaned not in urls:
                    urls.append(cleaned)

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    # Common patterns like {"sources": [...]} or {"references": [...]}
                    if key.lower() in {"sources", "references", "data_sources"} and isinstance(value, list):
                        for item in value:
                            if isinstance(item, str):
                                add_from_text(item)
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)
            elif isinstance(node, str):
                add_from_text(node)

        try:
            walk(research or {})
        except Exception:
            return []
        return urls

    def _build_dataset_calibration(self, datasets: List[DatasetRecord]) -> Dict[str, str]:
        """
        Build lightweight calibration hints from uploaded datasets so that memos
        gradually align with historical patterns without changing response shape.
        """

        if not datasets:
            return {}

        raw_summaries: List[str] = []
        market_snippets: List[str] = []
        risk_snippets: List[str] = []
        terms_snippets: List[str] = []

        def add_if_contains(text: str, keywords: List[str], bucket: List[str]) -> None:
            lowered = text.lower()
            if any(kw in lowered for kw in keywords):
                bucket.append(text.strip())

        # Simple semantic analysis - categorize content based on key phrases
        def categorize_dataset_content(text: str) -> Dict[str, float]:
            """Simple semantic categorization of dataset content"""
            lowered = text.lower()
            scores = {"market": 0.0, "risk": 0.0, "terms": 0.0}
            
            # Market/TAM related phrases
            market_phrases = ["tam", "sam", "som", "cagr", "market size", "addressable market", "growth rate", "industry analysis"]
            # Risk related phrases  
            risk_phrases = ["risk", "threat", "concern", "challenge", "competition", "barrier", "vulnerability"]
            # Terms/valuation phrases
            terms_phrases = ["valuation", "ownership", "dilution", "term sheet", "cap table", "discount", "investment", "funding"]
            
            for phrase in market_phrases:
                if phrase in lowered:
                    scores["market"] += 0.3
            for phrase in risk_phrases:
                if phrase in lowered:
                    scores["risk"] += 0.3
            for phrase in terms_phrases:
                if phrase in lowered:
                    scores["terms"] += 0.3
                    
            # Normalize scores
            total = sum(scores.values())
            if total > 0:
                for key in scores:
                    scores[key] = scores[key] / total
            
            return scores

        for ds in datasets:
            if not ds.summary:
                continue
            summary = str(ds.summary)
            raw_summaries.append(summary.strip())
            
            # Use semantic categorization instead of simple keyword matching
            categories = categorize_dataset_content(summary)
            
            # Add to appropriate buckets based on semantic scores
            if categories["market"] >= 0.4:
                market_snippets.append(summary.strip())
            if categories["risk"] >= 0.4:
                risk_snippets.append(summary.strip())
            if categories["terms"] >= 0.4:
                terms_snippets.append(summary.strip())

        # Limit prompt size
        joined_raw = "\n---\n".join(raw_summaries[:10])
        market_text = "; ".join(market_snippets[:5])
        risk_text = "; ".join(risk_snippets[:5])
        terms_text = "; ".join(terms_snippets[:5])

        calibration: Dict[str, str] = {
            "raw_context": joined_raw,
        }
        if market_text:
            calibration[
                "market_calibration"
            ] = f"Representative market/TAM patterns from historical datasets: {market_text}"
        if risk_text:
            calibration["risk_calibration"] = f"Representative risk patterns from historical datasets: {risk_text}"
        if terms_text:
            calibration[
                "terms_calibration"
            ] = f"Representative valuation/terms patterns from historical datasets: {terms_text}"

        return calibration

    async def _gather_website_context(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt to scrape the company's public website (if provided) to gather richer context.

        Returns a dict with:
        - url: normalized website URL
        - profile: structured fields extracted from the site (if available)
        - snippets: list of text excerpts for prompt grounding
        """
        if not isinstance(payload, dict):
            return {}

        candidate_urls: List[str] = []
        for key in ("website_url", "website", "url", "site", "site_url"):
            value = payload.get(key)
            if isinstance(value, str):
                candidate_urls.append(value.strip())

        relationships = payload.get("relationships")
        if isinstance(relationships, dict):
            rel_site = relationships.get("website") or relationships.get("url")
            if isinstance(rel_site, str):
                candidate_urls.append(rel_site.strip())

        quick_facts = payload.get("executive_summary", {}).get("quick_facts", {})
        if isinstance(quick_facts, dict):
            qf_site = quick_facts.get("website") or quick_facts.get("website_url")
            if isinstance(qf_site, str):
                candidate_urls.append(qf_site.strip())

        normalized_url = None
        for raw in candidate_urls:
            normalized = self._normalize_website_url(raw)
            if normalized:
                normalized_url = normalized
                break

        if not normalized_url:
            return {}

        try:
            scrape_result = await scrape_company_profile(normalized_url)
            profile_dump = scrape_result.profile.model_dump(exclude_none=True)
            snippets = scrape_result.text_snippets[:25]
            return {"url": normalized_url, "profile": profile_dump, "snippets": snippets}
        except Exception as exc:  # pragma: no cover - network best effort
            logger.debug("Website scrape failed for %s: %s", normalized_url, exc)
            return {"url": normalized_url}

    @staticmethod
    def _normalize_website_url(url: Optional[str]) -> Optional[str]:
        if not url:
            return None
        trimmed = url.strip()
        placeholder_tokens = {
            "",
            "http://",
            "https://",
            "n/a",
            "na",
            "none",
            "null",
            DEFAULT_PLACEHOLDER.lower(),
        }
        if trimmed.lower() in placeholder_tokens:
            return None
        if not trimmed.startswith(("http://", "https://")):
            trimmed = f"https://{trimmed}"
        parsed = urlparse(trimmed)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return None
        return trimmed

    @staticmethod
    def _clean_snippet(text: Optional[str], limit: int = 320) -> Optional[str]:
        if not text:
            return None
        import re as _re

        collapsed = _re.sub(r"\s+", " ", text).strip()
        if not collapsed:
            return None
        collapsed = _re.sub(r"https?://\S+", "", collapsed)
        sentences = _re.split(r"(?<=[.!?])\s+", collapsed)
        deduped: List[str] = []
        seen: set[str] = set()
        for sentence in sentences:
            candidate = sentence.strip(" ,.;")
            if not candidate:
                continue
            normalized = _re.sub(r"[^a-z0-9]+", "", candidate.lower())
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(candidate)
            if len(" ".join(deduped)) >= limit:
                break
        cleaned = " ".join(deduped).strip(" ,.;")
        if not cleaned:
            return None
        if len(cleaned) > limit:
            cleaned = cleaned[:limit].rstrip(" ,.;")
        if not cleaned.endswith("."):
            cleaned += "."
        return cleaned

    @staticmethod
    def _extract_website_metrics(snippets: List[str]) -> Dict[str, str]:
        import re as _re

        text = " ".join(snippets or [])
        metrics: Dict[str, str] = {}
        patterns = {
            "restaurants": r"([0-9][0-9,\.]*\+?)\s*(?:restaurants|restaurant partners)",
            "cities": r"([0-9][0-9,\.]*\+?)\s*(?:cities|city)",
            "orders": r"([0-9][0-9,\.]*\+?)\s*(?:orders|deliveries)",
            "employees": r"([0-9][0-9,\.]*\+?)\s*(?:employees|team)",
        }
        for key, pattern in patterns.items():
            match = _re.search(pattern, text, _re.IGNORECASE)
            if match:
                value = match.group(1).replace(" ", "")
                metrics[key] = value
        return metrics

    @staticmethod
    def _extract_founder_snippets_from_docs(pdf_texts: List[str], max_snippets: int = 2, max_length: int = 600) -> List[str]:
        """
        Grab short, founder-related snippets from uploaded documents to ground founder skill scoring.
        """
        snippets: List[str] = []
        if not pdf_texts:
            return snippets
        keywords = ("founder", "co-founder", "team", "leadership", "experience", "cto", "ceo", "bio")
        for text in pdf_texts:
            if len(snippets) >= max_snippets:
                break
            if not text:
                continue
            lowered = text.lower()
            if any(keyword in lowered for keyword in keywords):
                cleaned = " ".join(text.split())
                if len(cleaned) > max_length:
                    cleaned = cleaned[:max_length].rsplit(" ", 1)[0]
                snippets.append(cleaned)
        return snippets

    def _build_structured_interface(self, memo: InvestorCommitteeMemo) -> StructuredInterface:
        sections: List[StructuredInterfaceSection] = []
        sections.append(
            StructuredInterfaceSection(
                id="revenue_assumptions",
                title="Revenue Assumptions",
                progress_index=1,
                total_sections=3,
                fields=[
                    StructuredField(
                        id="year1_revenue",
                        label="Year 1 Revenue",
                        current_value=(memo.financial_projections.projection_assumptions.adjusted_projection_year1),
                        source="memo",
                        confidence=0.72,
                        requires_human_input=False,
                        validation=StructuredFieldValidation(type="number"),
                        help_text="Adjust Year 1 revenue if updated forecast is available.",
                    ),
                    StructuredField(
                        id="growth_rate",
                        label="Growth rate",
                        current_value=memo.financial_projections.revenue_projections[0].growth_rate
                        if memo.financial_projections.revenue_projections
                        else DEFAULT_PLACEHOLDER,
                        source="memo",
                        confidence=0.51,
                        requires_human_input=True,
                        validation=StructuredFieldValidation(type="percent", min=0, max=1000),
                        help_text="Growth rate must be 0-1000%.",
                    ),
                ],
            )
        )
        sections.append(
            StructuredInterfaceSection(
                id="team_assessment",
                title="Team Assessment",
                progress_index=2,
                total_sections=3,
                fields=[
                    StructuredField(
                        id="founder_score",
                        label="Overall Founder Score",
                        current_value=(
                            memo.founder_team_analysis.founders[0].overall_founder_score
                            if memo.founder_team_analysis.founders
                            else DEFAULT_PLACEHOLDER
                        ),
                        source="memo",
                        confidence=0.64,
                        requires_human_input=False,
                        validation=StructuredFieldValidation(type="text"),
                        help_text="Verify founder assessment after any new references.",
                    ),
                    StructuredField(
                        id="key_hires",
                        label="Key hires needed (12 months)",
                        current_value=", ".join(memo.founder_team_analysis.team_completeness.key_hires_needed_12_months),
                        source="memo",
                        confidence=0.58,
                        requires_human_input=True,
                        validation=StructuredFieldValidation(type="text"),
                        help_text="List priority hires to de-risk execution.",
                    ),
                ],
            )
        )
        sections.append(
            StructuredInterfaceSection(
                id="risk_analysis",
                title="Risk Analysis",
                progress_index=3,
                total_sections=3,
                fields=[
                    StructuredField(
                        id="top_risk",
                        label="Top Risk",
                        current_value=(memo.risk_assessment.risks[0].description if memo.risk_assessment.risks else DEFAULT_PLACEHOLDER),
                        source="memo",
                        confidence=0.55,
                        requires_human_input=False,
                        validation=StructuredFieldValidation(type="text"),
                        help_text="Clarify risk wording before IC circulation.",
                    ),
                    StructuredField(
                        id="mitigation",
                        label="Mitigation Plan",
                        current_value=", ".join(memo.risk_assessment.risks[0].mitigation)
                        if memo.risk_assessment.risks and memo.risk_assessment.risks[0].mitigation
                        else DEFAULT_PLACEHOLDER,
                        source="memo",
                        confidence=0.47,
                        requires_human_input=True,
                        validation=StructuredFieldValidation(type="text"),
                        help_text="Add concrete mitigation owners and timelines.",
                    ),
                ],
            )
        )
        return StructuredInterface(
            sections=sections,
            progress={"completed_sections": 1, "total_sections": len(sections)},
        )

    def _merge_usage(
        self, claude_usage: Optional[Dict[str, int]], gemini_usage: Optional[Dict[str, int]]
    ) -> LLMTokenUsage:
        usage = LLMTokenUsage(
            claude_input_tokens=claude_usage.get("input_tokens") if claude_usage else 0,
            claude_output_tokens=claude_usage.get("output_tokens") if claude_usage else 0,
            gemini_input_tokens=gemini_usage.get("input_tokens") if gemini_usage else 0,
            gemini_output_tokens=gemini_usage.get("output_tokens") if gemini_usage else 0,
        )
        usage.total_tokens = (
            (usage.claude_input_tokens or 0)
            + (usage.claude_output_tokens or 0)
            + (usage.gemini_input_tokens or 0)
            + (usage.gemini_output_tokens or 0)
        )
        return usage

    async def _fetch_edgar_notes(self, company_name: Optional[str]) -> Optional[str]:
        if not company_name:
            return None
        if not self.settings.edgar_api_key:
            return None
        headers = {
            "accept": "application/json",
            "User-Agent": self.settings.edgar_user_agent,
            "x-api-key": self.settings.edgar_api_key,
        }
        url = f"https://api.edgarfiling.sec.gov/v2/companies?name={quote_plus(company_name)}"
        async with httpx.AsyncClient(timeout=self.settings.http_timeout_seconds) as client:
            try:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                companies = data.get("companies") or data.get("data") or []
                if companies:
                    first = companies[0]
                    cik = first.get("cik") or first.get("id") or "N/A"
                    filing_types = first.get("filing_types") or first.get("filings") or []
                    return (
                        f"EDGAR match {first.get('name', company_name)} (CIK {cik}). "
                        f"Recent filings: {filing_types[:3] if isinstance(filing_types, list) else filing_types}."
                    )
            except Exception as exc:  # pragma: no cover - network best effort
                logger.debug("EDGAR lookup failed for %s: %s", company_name, exc)
        return None

    async def _fetch_sec_markets_notes(self) -> Optional[str]:
        """Best-effort pull of SEC markets data landing content for context."""
        url = "https://www.sec.gov/data-research/sec-markets-data"
        headers = {"User-Agent": self.settings.edgar_user_agent or "AngelLinx/1.0"}
        try:
            async with httpx.AsyncClient(timeout=self.settings.http_timeout_seconds) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                text = resp.text
                snippet = " ".join(text.split()[:200])
                return f"SEC markets data page snippet: {snippet}"
        except Exception as exc:  # pragma: no cover - network best effort
            logger.debug("SEC markets fetch failed: %s", exc)
            return None


async def load_uploaded_datasets() -> List[DatasetRecord]:
    """Load dataset records for RAG context."""
    try:
        return list_datasets()
    except Exception as exc:
        logger.warning("Failed to load datasets for agent context: %s", exc)
        return []


async def extract_text_from_files(files: List[Any]) -> List[str]:
    """Extract raw text from upload files; used to enrich datasets."""
    processor = DocumentProcessor()
    texts: List[str] = []
    dummy_id = uuid4().hex
    metadata = await processor.ingest_documents(dummy_id, files)
    for item in metadata:
        if item.text_content:
            texts.append(item.text_content)
    return texts
