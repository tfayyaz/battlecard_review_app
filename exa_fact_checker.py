"""
Exa Fact Checker - Verifies claims using Exa web search + LLM judge.

Uses the Exa API to search for supporting/contradicting evidence for each
claim's evidence rows, then calls an LLM judge to evaluate verdicts.
"""

import json
import logging
import os
from datetime import datetime

from exa_py import Exa
from openai import OpenAI
from sqlalchemy import text

logger = logging.getLogger(__name__)


def _get_openai_client():
    """Build OpenAI client for Databricks Model Serving (same pattern as WorkflowRunner)."""
    host = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")
    if host and token:
        return OpenAI(
            api_key=token,
            base_url=host.rstrip("/") + "/serving-endpoints",
        )
    from databricks.sdk import WorkspaceClient
    profile = os.getenv("DATABRICKS_PROFILE", "fe-vm-pmt")
    w = WorkspaceClient(profile=profile)
    return w.serving_endpoints.get_open_ai_client()


class ExaFactChecker:
    """Verifies claims using Exa web search and an LLM judge."""

    MODEL_NAME = "databricks-claude-haiku-4-5"

    def __init__(self, session_id, engine):
        self.session_id = session_id
        self.engine = engine

        exa_key = os.getenv("EXA_API_KEY")
        if not exa_key:
            raise ValueError("EXA_API_KEY environment variable is not set")
        self.exa = Exa(api_key=exa_key)
        self.llm_client = _get_openai_client()

    def run_fact_checks(self):
        """Main orchestrator: load claims, search Exa, judge with LLM, save results."""
        self._update_step(
            6,
            "in_progress",
            progress_current=0,
            progress_total=4,
            progress_message="[1/4] Preparing fact check run...",
        )

        # 1. Load generation_id from the workflow session
        with self.engine.begin() as conn:
            row = conn.execute(
                text("SELECT generation_id FROM workflow_sessions WHERE session_id::text = :sid"),
                {"sid": self.session_id},
            ).mappings().first()
        if not row or not row["generation_id"]:
            self._update_step(6, "waiting_human", error_message="No generation found for this session.")
            return

        gen_id = row["generation_id"]
        self._update_step(
            6,
            "in_progress",
            progress_current=1,
            progress_total=4,
            progress_message="[2/4] Loading evidence rows for fact checking...",
        )

        # 2. Load all evidence rows for this generation
        with self.engine.begin() as conn:
            evidence_rows = conn.execute(
                text(
                    "SELECT e.evidence_id, e.traces_to_text, e.detail_item_id, e.claim_id, "
                    "c.headline, c.description, co.company_name, co.company_type, "
                    "di.item_text AS detail_item_text "
                    "FROM evidence e "
                    "JOIN claims c ON e.claim_id = c.claim_id "
                    "JOIN companies co ON c.company_id = co.company_id "
                    "LEFT JOIN claim_detail_items di ON e.detail_item_id = di.detail_item_id "
                    "WHERE c.generation_id = :gen_id "
                    "ORDER BY e.evidence_id"
                ),
                {"gen_id": gen_id},
            ).mappings().all()

        if not evidence_rows:
            self._update_step(6, "waiting_human", error_message="No evidence rows found to fact check.")
            return

        total = len(evidence_rows)
        logger.info("Fact checking %d evidence rows for session %s (gen %s)", total, self.session_id, gen_id)
        self._update_step(
            6,
            "in_progress",
            progress_current=2,
            progress_total=total + 3,
            progress_message=f"[3/4] Starting checks for {total} evidence rows...",
        )

        # 3. Process each evidence row
        for idx, ev in enumerate(evidence_rows):
            claim_preview = (ev["traces_to_text"] or "")[:80]
            self._update_step(
                6, "in_progress",
                progress_current=idx + 3,
                progress_total=total + 3,
                progress_message=f"[3/4] Fact checking [{idx + 1}/{total}]: {claim_preview}..."
            )

            try:
                # a. Search Exa — use detail_item_text for richer context when available
                search_text = ev.get("detail_item_text") or ev["traces_to_text"]
                exa_results = self._search_exa(search_text, ev["company_name"])

                # b. Judge with LLM
                verdict = self._judge_with_llm(ev["traces_to_text"], exa_results)

                # c. Upsert source (best Exa result)
                source_id = None
                source_text = None
                best_idx = verdict.get("best_source_index", 0)
                if exa_results and 0 <= best_idx < len(exa_results):
                    best_result = exa_results[best_idx]
                    source_id = self._upsert_exa_source(best_result)
                    source_text = best_result.get("text", "")[:2000]

                # d. Update or insert fact_check row
                self._update_fact_check(
                    evidence_id=ev["evidence_id"],
                    verdict=verdict.get("verdict", "unverified"),
                    source_id=source_id,
                    source_text=source_text,
                    reasoning=verdict.get("reasoning", ""),
                    dispute_details=verdict.get("dispute_details", ""),
                    confidence=verdict.get("confidence", 0),
                )
            except Exception as e:
                logger.warning("Fact check failed for evidence %s: %s", ev["evidence_id"], e)
                self._update_fact_check(
                    evidence_id=ev["evidence_id"],
                    verdict="unverified",
                    source_id=None,
                    source_text=None,
                    reasoning=f"Automated check failed: {e}",
                    dispute_details="",
                    confidence=0,
                )

        # 4. Build summary artifact
        self._update_step(
            6,
            "in_progress",
            progress_current=total + 2,
            progress_total=total + 3,
            progress_message="[4/4] Building fact check summary...",
        )
        summary = self._build_summary(gen_id)
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO workflow_artifacts "
                    "(session_id, step_number, artifact_type, artifact_name, artifact_content) "
                    "VALUES (CAST(:sid AS uuid), 6, 'fact_check_results', 'fact_check_summary', :content)"
                ),
                {"sid": self.session_id, "content": json.dumps(summary)},
            )

        # 5. Set step 6 to waiting_human
        self._update_step(
            6,
            "waiting_human",
            progress_current=total + 3,
            progress_total=total + 3,
            progress_message=f"Fact checked {total} evidence rows. Review results below.",
        )
        logger.info("Fact check complete for session %s: %s", self.session_id, summary)

    def _search_exa(self, claim_text, company_name):
        """Search Exa for evidence supporting or contradicting the claim."""
        query = f"{company_name}: {claim_text}"
        try:
            results = self.exa.search(
                query=query,
                num_results=5,
                type="auto",
                contents={
                    "text": {"max_characters": 1500},
                    "highlights": {
                        "query": claim_text,
                        "num_sentences": 3,
                    },
                },
            )
            parsed = []
            for r in results.results:
                parsed.append({
                    "url": r.url,
                    "title": r.title or "",
                    "text": r.text or "",
                    "highlights": r.highlights or [],
                    "highlight_scores": r.highlight_scores if hasattr(r, "highlight_scores") else [],
                    "score": r.score if hasattr(r, "score") else 0,
                    "published_date": r.published_date if hasattr(r, "published_date") else None,
                })
            return parsed
        except Exception as e:
            logger.warning("Exa search failed for claim: %s — %s", claim_text[:60], e)
            return []

    def _judge_with_llm(self, claim_text, exa_results):
        """Call LLM judge to evaluate fact-check verdict."""
        if not exa_results:
            return {
                "verdict": "unverified",
                "confidence": 0,
                "reasoning": "No web search results found.",
                "best_source_index": 0,
                "dispute_details": "",
            }

        # Format search results for the prompt
        formatted_results = ""
        for i, r in enumerate(exa_results):
            highlights_text = "\n".join(f"  - {h}" for h in (r.get("highlights") or []))
            formatted_results += (
                f"\n--- Result {i + 1} ---\n"
                f"Title: {r.get('title', 'N/A')}\n"
                f"URL: {r.get('url', 'N/A')}\n"
                f"Highlights:\n{highlights_text or '  (none)'}\n"
                f"Text excerpt: {(r.get('text') or '')[:500]}\n"
            )

        prompt = f"""You are a fact-checking judge. Given a factual claim and web search results,
determine if the claim is supported by credible evidence.

Claim: "{claim_text}"

Web Search Results:
{formatted_results}

Evaluate the claim against the search results and return a JSON object with:
- "verdict": one of "verified", "unverified", "disputed", "outdated"
  - verified: credible sources confirm the claim (confidence >= 70)
  - unverified: insufficient evidence to confirm or deny
  - disputed: sources contradict the claim
  - outdated: claim was true but no longer current
- "confidence": integer 0-100
- "reasoning": brief explanation of your verdict
- "best_source_index": 0-based index of the most relevant search result
- "dispute_details": if disputed or outdated, explain what contradicts the claim (empty string otherwise)

Return ONLY the JSON object, no other text."""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.1,
            )
            content = response.choices[0].message.content.strip()
            # Strip markdown code fence if present
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

            verdict = json.loads(content)
            # Validate verdict field
            if verdict.get("verdict") not in ("verified", "unverified", "disputed", "outdated"):
                verdict["verdict"] = "unverified"
            verdict.setdefault("confidence", 0)
            verdict.setdefault("reasoning", "")
            verdict.setdefault("best_source_index", 0)
            verdict.setdefault("dispute_details", "")
            return verdict
        except Exception as e:
            logger.warning("LLM judge failed: %s", e)
            return {
                "verdict": "unverified",
                "confidence": 0,
                "reasoning": f"LLM judge error: {e}",
                "best_source_index": 0,
                "dispute_details": "",
            }

    def _upsert_exa_source(self, exa_result):
        """Insert or find a source by URL. Returns source_id."""
        url = exa_result.get("url", "")
        if not url:
            return None

        with self.engine.begin() as conn:
            # Check if source already exists
            existing = conn.execute(
                text("SELECT source_id FROM sources WHERE source_url = :url LIMIT 1"),
                {"url": url},
            ).scalar()
            if existing:
                return existing

            # Insert new source
            published = exa_result.get("published_date")
            pub_date = None
            if published:
                try:
                    pub_date = datetime.fromisoformat(published.replace("Z", "+00:00")).date()
                except (ValueError, AttributeError):
                    pass

            source_id = conn.execute(
                text(
                    "INSERT INTO sources (source_name, source_url, source_type, publisher, published_date) "
                    "VALUES (:name, :url, 'third-party', :publisher, :pub_date) "
                    "RETURNING source_id"
                ),
                {
                    "name": exa_result.get("title", url)[:500],
                    "url": url,
                    "publisher": exa_result.get("title", "")[:255],
                    "pub_date": pub_date,
                },
            ).scalar()
            return source_id

    def _update_fact_check(self, evidence_id, verdict, source_id, source_text, reasoning, dispute_details, confidence):
        """Update existing fact_check row or insert new one."""
        with self.engine.begin() as conn:
            # Check if row exists
            existing = conn.execute(
                text("SELECT fact_check_id FROM fact_checks WHERE evidence_id = :eid LIMIT 1"),
                {"eid": evidence_id},
            ).scalar()

            if existing:
                conn.execute(
                    text(
                        "UPDATE fact_checks SET "
                        "status = :verdict, fact_check_source_id = :src_id, "
                        "fact_check_source_text = :src_text, reasoning = :reasoning, "
                        "dispute_details = :dispute, checked_at = NOW(), "
                        "checked_by = 'exa_fact_checker', check_method = 'automated', "
                        "confidence_score = :confidence "
                        "WHERE evidence_id = :eid"
                    ),
                    {
                        "verdict": verdict, "src_id": source_id,
                        "src_text": source_text, "reasoning": reasoning,
                        "dispute": dispute_details, "confidence": confidence,
                        "eid": evidence_id,
                    },
                )
            else:
                conn.execute(
                    text(
                        "INSERT INTO fact_checks "
                        "(evidence_id, status, fact_check_source_id, fact_check_source_text, "
                        "reasoning, dispute_details, checked_at, checked_by, check_method, confidence_score) "
                        "VALUES (:eid, :verdict, :src_id, :src_text, :reasoning, :dispute, "
                        "NOW(), 'exa_fact_checker', 'automated', :confidence)"
                    ),
                    {
                        "eid": evidence_id, "verdict": verdict,
                        "src_id": source_id, "src_text": source_text,
                        "reasoning": reasoning, "dispute": dispute_details,
                        "confidence": confidence,
                    },
                )

    def _build_summary(self, gen_id):
        """Build a summary dict of fact check results."""
        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    "SELECT "
                    "COUNT(*) as total, "
                    "SUM(CASE WHEN fc.status = 'verified' THEN 1 ELSE 0 END) as verified, "
                    "SUM(CASE WHEN fc.status = 'unverified' THEN 1 ELSE 0 END) as unverified, "
                    "SUM(CASE WHEN fc.status = 'disputed' THEN 1 ELSE 0 END) as disputed, "
                    "SUM(CASE WHEN fc.status = 'outdated' THEN 1 ELSE 0 END) as outdated, "
                    "ROUND(AVG(fc.confidence_score), 2) as avg_confidence "
                    "FROM fact_checks fc "
                    "JOIN evidence e ON fc.evidence_id = e.evidence_id "
                    "JOIN claims c ON e.claim_id = c.claim_id "
                    "WHERE c.generation_id = :gid"
                ),
                {"gid": gen_id},
            ).mappings().first()

        if not row:
            return {"total": 0, "verified": 0, "unverified": 0, "disputed": 0, "outdated": 0, "avg_confidence": 0}
        return {
            "total": int(row["total"] or 0),
            "verified": int(row["verified"] or 0),
            "unverified": int(row["unverified"] or 0),
            "disputed": int(row["disputed"] or 0),
            "outdated": int(row["outdated"] or 0),
            "avg_confidence": float(row["avg_confidence"] or 0),
        }

    def _update_step(self, step_number, status, progress_current=None, progress_total=None,
                     progress_message=None, error_message=None):
        """Update workflow step status (mirrors app.py _update_step_status)."""
        with self.engine.begin() as conn:
            sets = ["status = :status", "heartbeat_at = NOW()"]
            params = {"sid": self.session_id, "step": step_number, "status": status}

            if status == "in_progress":
                sets.append("started_at = COALESCE(started_at, NOW())")
            if status in ("completed", "failed"):
                sets.append("completed_at = NOW()")
            if progress_current is not None:
                sets.append("progress_current = :pc")
                params["pc"] = progress_current
            if progress_total is not None:
                sets.append("progress_total = :pt")
                params["pt"] = progress_total
            if progress_message is not None:
                sets.append("progress_message = :pm")
                params["pm"] = progress_message
            if error_message is not None:
                sets.append("error_message = :em")
                params["em"] = error_message
            elif status in ("ready", "in_progress", "waiting_human", "completed"):
                sets.append("error_message = NULL")

            conn.execute(
                text(f"UPDATE workflow_steps SET {', '.join(sets)} WHERE session_id::text = :sid AND step_number = :step"),
                params,
            )
            conn.execute(
                text("UPDATE workflow_sessions SET current_step = :step, updated_at = NOW() WHERE session_id::text = :sid"),
                {"sid": self.session_id, "step": step_number},
            )
