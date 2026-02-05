"""
Workflow Runner - Orchestrates battlecard generation passes.

Self-contained module that renders prompt templates, calls the Databricks Model
Serving API via the OpenAI client, and writes results back to Lakebase tables.

No dependency on scripts/ or mlflow — uses openai + databricks-sdk directly.
"""

import hashlib
import json
import logging
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openai import OpenAI
from sqlalchemy import text

try:
    import tiktoken
    _TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _TIKTOKEN_ENCODING = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths (same as app.py — used for context formatting)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_PASS1_PROMPT = os.path.join(
    _PROJECT_ROOT, "generate-battlecards", "2_prompts", "l200_pass1_planning_v1.md",
)
DEFAULT_PASS2_PROMPT = os.path.join(
    _PROJECT_ROOT, "generate-battlecards", "2_prompts", "l200_pass2_detail_v3_factcheck.md",
)

# ---------------------------------------------------------------------------
# JSON Schemas for structured output
# ---------------------------------------------------------------------------

_CITATION_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "citation_id": {"type": "string"},
        "detail_item_index": {"type": "integer"},
        "start_index": {"type": "integer"},
        "end_index": {"type": "integer"},
        "source_index": {"type": "integer"},
        "source_quote": {"type": "string"},
        "verdict": {"type": "string"},
        "confidence": {"type": "number"},
        "verdict_rationale": {"type": "string"},
    },
    "required": [
        "citation_id", "detail_item_index", "start_index", "end_index",
        "source_index", "source_quote", "verdict",
        "confidence", "verdict_rationale",
    ],
}

_CITATIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "databricks_details": {"type": "array", "items": _CITATION_ITEM_SCHEMA},
        "databricks_reasoning": {"type": "array", "items": _CITATION_ITEM_SCHEMA},
        "competitor_details": {"type": "array", "items": _CITATION_ITEM_SCHEMA},
        "competitor_reasoning": {"type": "array", "items": _CITATION_ITEM_SCHEMA},
    },
    "required": [
        "databricks_details", "databricks_reasoning",
        "competitor_details", "competitor_reasoning",
    ],
}

_SOURCE_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "index": {"type": "integer"},
        "title": {"type": "string"},
        "url": {"type": "string"},
        "type": {"type": "string"},
        "accessed_at": {"type": "string"},
    },
    "required": ["index", "title", "url", "type", "accessed_at"],
}

L200_PASS1_JSON_SCHEMA = {
    "name": "l200_pass1_planning",
    "schema": {
        "type": "object",
        "properties": {
            "slides": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "competitor": {"type": "string"},
                        "category": {"type": "string"},
                        "rank": {"type": "integer"},
                        "key_differentiator": {"type": "string"},
                        "description": {"type": "string"},
                        "selection_reasoning": {"type": "string"},
                        "rank_reasoning": {"type": "string"},
                        "directive_alignment": {"type": "string"},
                        "databricks_rating": {"type": "string"},
                        "competitor_rating": {"type": "string"},
                    },
                    "required": [
                        "id", "competitor", "category", "rank",
                        "key_differentiator", "description",
                        "selection_reasoning", "rank_reasoning",
                        "directive_alignment",
                        "databricks_rating", "competitor_rating",
                    ],
                },
            },
        },
        "required": ["slides"],
    },
    "strict": True,
}

L200_PASS2_JSON_SCHEMA = {
    "name": "l200_diff_detail",
    "schema": {
        "type": "object",
        "properties": {
            "databricks_headline": {"type": "string"},
            "databricks_details": {
                "type": "array",
                "items": {"type": "string"},
            },
            "databricks_reasoning": {"type": "string"},
            "competitor_headline": {"type": "string"},
            "competitor_details": {
                "type": "array",
                "items": {"type": "string"},
            },
            "competitor_reasoning": {"type": "string"},
            "citations": {
                "type": "object",
                "properties": {
                    "databricks_details": {"type": "array", "items": _CITATION_ITEM_SCHEMA},
                    "databricks_reasoning": {"type": "array", "items": _CITATION_ITEM_SCHEMA},
                    "competitor_details": {"type": "array", "items": _CITATION_ITEM_SCHEMA},
                    "competitor_reasoning": {"type": "array", "items": _CITATION_ITEM_SCHEMA},
                },
                "required": [
                    "databricks_details", "databricks_reasoning",
                    "competitor_details", "competitor_reasoning",
                ],
            },
            "sources": {
                "type": "array",
                "items": _SOURCE_ITEM_SCHEMA,
            },
            "research_sources": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "databricks_headline", "databricks_details", "databricks_reasoning",
            "competitor_headline", "competitor_details", "competitor_reasoning",
            "citations", "sources", "research_sources",
        ],
    },
    "strict": True,
}

L200_PASS3_JSON_SCHEMA = {
    "name": "l200_slide_update",
    "schema": {
        "type": "object",
        "properties": {
            "key_differentiator": {"type": "string"},
            "description": {"type": "string"},
            "databricks_rating": {"type": "string"},
            "competitor_rating": {"type": "string"},
            "selection_reasoning": {"type": "string"},
            "rank_reasoning": {"type": "string"},
            "directive_alignment": {"type": "string"},
            "databricks_headline": {"type": "string"},
            "databricks_details": {
                "type": "array",
                "items": {"type": "string"},
            },
            "databricks_reasoning": {"type": "string"},
            "competitor_headline": {"type": "string"},
            "competitor_details": {
                "type": "array",
                "items": {"type": "string"},
            },
            "competitor_reasoning": {"type": "string"},
            "citations": _CITATIONS_SCHEMA,
            "sources": {"type": "array", "items": _SOURCE_ITEM_SCHEMA},
            "research_sources": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "key_differentiator", "description",
            "databricks_rating", "competitor_rating",
            "selection_reasoning", "rank_reasoning", "directive_alignment",
            "databricks_headline", "databricks_details", "databricks_reasoning",
            "competitor_headline", "competitor_details", "competitor_reasoning",
            "citations", "sources", "research_sources",
        ],
    },
    "strict": True,
}

# ---------------------------------------------------------------------------
# LLM client + call helpers
# ---------------------------------------------------------------------------


def get_openai_client() -> OpenAI:
    """Return an OpenAI-compatible client pointing at Databricks Model Serving."""
    host = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")
    if host and token:
        return OpenAI(
            api_key=token,
            base_url=host.rstrip("/") + "/serving-endpoints",
        )
    # Fall back to WorkspaceClient OAuth flow
    from databricks.sdk import WorkspaceClient
    profile = os.getenv("DATABRICKS_PROFILE", "fe-vm-pmt")
    w = WorkspaceClient(profile=profile)
    return w.serving_endpoints.get_open_ai_client()


def call_model(
    client: OpenAI,
    model_name: str,
    rendered_prompt: str,
    json_schema: Optional[Dict] = None,
    temperature: float = 0.2,
    max_tokens: int = 16384,
) -> str:
    """Call the model via Databricks Model Serving with optional structured output."""
    kwargs = {
        "model": model_name,
        "messages": [{"role": "user", "content": rendered_prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": json_schema,
        }
        logger.info("  Using structured output (json_schema: %s)", json_schema["name"])

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Prompt template helpers
# ---------------------------------------------------------------------------


def load_prompt_template(path: str) -> str:
    """Read a prompt template file from disk."""
    with open(path) as f:
        return f.read()


def render_template(template: str, **kwargs) -> str:
    """Render a prompt template by replacing {{variable}} placeholders."""
    result = template
    for key, value in kwargs.items():
        result = result.replace("{{" + key + "}}", str(value))
    return result


def load_template_from_db(engine, template_type: str, display_order: int):
    """Load a prompt template from the DB by type and display_order.

    Returns (template_text, template_name) or None if not found / inactive.
    """
    if engine is None:
        return None
    try:
        with engine.begin() as conn:
            row = conn.execute(
                text(
                    "SELECT template_text, template_name FROM prompt_templates "
                    "WHERE template_type = :ttype AND display_order = :dorder AND is_active = TRUE "
                    "LIMIT 1"
                ),
                {"ttype": template_type, "dorder": display_order},
            ).mappings().first()
        if row:
            return row["template_text"], row["template_name"]
    except Exception as e:
        logger.warning("Failed to load template from DB (type=%s, order=%d): %s", template_type, display_order, e)
    return None


def load_directive_template(engine=None) -> str:
    """Load the directive generation prompt template.

    Tries DB first, then filesystem, then hardcoded fallback.
    """
    # Try DB first
    if engine:
        try:
            with engine.begin() as conn:
                row = conn.execute(
                    text(
                        "SELECT template_text FROM prompt_templates "
                        "WHERE template_type = 'directive' AND is_default = TRUE AND is_active = TRUE "
                        "LIMIT 1"
                    ),
                ).scalar()
            if row:
                return row
        except Exception as e:
            logger.warning("Failed to load directive template from DB: %s", e)

    # Filesystem fallback
    try:
        from app import DEFAULT_DIRECTIVE_PROMPT as directive_path
        if os.path.isfile(directive_path):
            return load_prompt_template(directive_path)
    except (ImportError, FileNotFoundError):
        pass

    # Hardcoded fallback
    return (
        "Read the following slides content about competing against {{competitor}}.\n\n"
        "Parse and create a max 10 to 25 bullets on how we (Databricks compete team) should "
        "take what has been taught to AE account executives and SAs on how to compete against {{competitor}}.\n\n"
        "Extract this directive as max 10 to 25 bullets that will be used to create an internal battlecard.\n\n"
        "Write the directive in markdown format."
    )


def load_pass1_template(version: int, engine=None) -> tuple[str, str]:
    """Load the Pass 1 prompt template for the given version.

    Returns (template_text, file_path) tuple.
    Tries DB first (by display_order == version), then falls back to
    filesystem / inline templates.
    """
    # Try DB first
    result = load_template_from_db(engine, "pass1", version)
    if result:
        return result

    # Filesystem / inline fallback
    if version == 1:
        return load_prompt_template(DEFAULT_PASS1_PROMPT), DEFAULT_PASS1_PROMPT
    try:
        from app import PASS1_PROMPT_TEMPLATES
        cfg = PASS1_PROMPT_TEMPLATES.get(version)
        if cfg:
            if "template" in cfg:
                return cfg["template"], f"inline_v{version}"
            if "file" in cfg:
                return load_prompt_template(cfg["file"]), cfg["file"]
    except ImportError:
        pass
    logger.warning("Pass1 prompt version %d not found, falling back to V1", version)
    return load_prompt_template(DEFAULT_PASS1_PROMPT), DEFAULT_PASS1_PROMPT


def load_pass2_template(version: int, engine=None) -> str:
    """Load the Pass 2 prompt template for the given version.

    Tries DB first (by display_order == version), then falls back to
    filesystem / inline templates.
    """
    # Try DB first
    result = load_template_from_db(engine, "pass2", version)
    if result:
        return result[0]  # Only return template_text

    # Filesystem / inline fallback
    if version == 1:
        return load_prompt_template(DEFAULT_PASS2_PROMPT)
    try:
        from app import PASS2_PROMPT_TEMPLATES
        cfg = PASS2_PROMPT_TEMPLATES.get(version)
        if cfg:
            if "template" in cfg:
                return cfg["template"]
            if "file" in cfg:
                return load_prompt_template(cfg["file"])
    except ImportError:
        pass
    # Fallback to V1
    logger.warning("Pass2 prompt version %d not found, falling back to V1", version)
    return load_prompt_template(DEFAULT_PASS2_PROMPT)


def format_context_xml(directive: str, old_battlecard: str, competitor: str,
                       review_feedback: str = "", fact_check_results: str = "") -> str:
    """Wrap directive + old battlecard + optional review/fact-check content in XML tags for the prompt."""
    parts = []
    if directive:
        parts.append(
            f'<competitive_directive name="directive" doc_type="competitive_directive" source="human_provided">\n'
            f'{directive}\n'
            f'</competitive_directive>'
        )
    if old_battlecard:
        parts.append(
            f'<battlecard_archive name="previous_battlecard" doc_type="battlecard_archive" source="human_provided" scope="{competitor}">\n'
            f'{old_battlecard}\n'
            f'</battlecard_archive>'
        )
    if review_feedback:
        parts.append(
            f'<previous_version_reviews name="review_feedback" doc_type="review_feedback" source="human_reviews">\n'
            f'{review_feedback}\n'
            f'</previous_version_reviews>'
        )
    if fact_check_results:
        parts.append(
            f'<previous_version_fact_checks name="fact_check_results" doc_type="fact_check_results" source="automated_fact_check">\n'
            f'{fact_check_results}\n'
            f'</previous_version_fact_checks>'
        )
    return "\n\n".join(parts) if parts else "No additional context provided."


# ---------------------------------------------------------------------------
# Prompt versioning
# ---------------------------------------------------------------------------


def ensure_prompt_version(engine, prompt_name: str, prompt_file: str, prompt_text: str) -> int:
    """Insert a prompt version if it doesn't exist (dedup on content_hash), return prompt_version_id."""
    content_hash = hashlib.sha256(prompt_text.encode()).hexdigest()
    with engine.begin() as conn:
        existing = conn.execute(
            text(
                "SELECT prompt_version_id FROM prompt_versions "
                "WHERE prompt_name = :name AND content_hash = :hash LIMIT 1"
            ),
            {"name": prompt_name, "hash": content_hash},
        ).scalar()
        if existing:
            return existing

        # Compute next version number for this prompt
        max_ver = conn.execute(
            text("SELECT COALESCE(MAX(prompt_version), 0) FROM prompt_versions WHERE prompt_name = :name"),
            {"name": prompt_name},
        ).scalar() or 0

        new_id = conn.execute(
            text(
                "INSERT INTO prompt_versions (prompt_name, prompt_version, prompt_file, prompt_text, content_hash) "
                "VALUES (:name, :ver, :file, :text, :hash) RETURNING prompt_version_id"
            ),
            {
                "name": prompt_name,
                "ver": int(max_ver) + 1,
                "file": prompt_file,
                "text": prompt_text,
                "hash": content_hash,
            },
        ).scalar()
        logger.info("Created prompt version %s v%d (id=%d)", prompt_name, int(max_ver) + 1, new_id)
        return new_id


# ============================================================================
# WorkflowRunner
# ============================================================================


class WorkflowRunner:
    """Manages a single workflow session's generation passes."""

    # Unique worker ID for this runner instance (process + thread + time)
    _worker_id = None

    @classmethod
    def _get_worker_id(cls):
        """Generate a unique worker ID for heartbeat tracking."""
        if cls._worker_id is None:
            import threading
            import uuid
            cls._worker_id = f"{os.getpid()}-{threading.current_thread().ident}-{uuid.uuid4().hex[:8]}"
        return cls._worker_id

    def __init__(self, session_id: str, engine):
        self.session_id = session_id
        self.engine = engine
        self.worker_id = self._get_worker_id()
        self._load_session()
        self.client = get_openai_client()

    def _load_session(self):
        """Load session config from the database."""
        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    "SELECT competitor_name, product_area, model_name, "
                    "diffs_per_category, max_workers, "
                    "COALESCE(pass1_prompt_template_version, 2) AS pass1_prompt_template_version, "
                    "COALESCE(pass2_prompt_template_version, 2) AS pass2_prompt_template_version "
                    "FROM workflow_sessions WHERE session_id::text = :sid"
                ),
                {"sid": self.session_id},
            ).mappings().first()
        if not row:
            raise ValueError(f"Session {self.session_id} not found")
        self.competitor = row["competitor_name"]
        self.product_area = row["product_area"]
        self.model_name = row["model_name"]
        self.diffs_per_category = row["diffs_per_category"]
        self.max_workers = row["max_workers"]
        self.pass1_prompt_template_version = row["pass1_prompt_template_version"]
        self.pass2_prompt_template_version = row["pass2_prompt_template_version"]

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _update_step(self, step_number, status, **kwargs):
        """Update step status in the database."""
        with self.engine.begin() as conn:
            sets = ["status = :status", "heartbeat_at = NOW()"]
            params = {"sid": self.session_id, "step": step_number, "status": status, "wid": self.worker_id}
            if status == "in_progress":
                sets.append("started_at = COALESCE(started_at, NOW())")
                sets.append("worker_id = :wid")
            if status in ("completed", "failed"):
                sets.append("completed_at = NOW()")
            if "progress_current" in kwargs:
                sets.append("progress_current = :pc")
                params["pc"] = kwargs["progress_current"]
            if "progress_total" in kwargs:
                sets.append("progress_total = :pt")
                params["pt"] = kwargs["progress_total"]
            if "progress_message" in kwargs:
                sets.append("progress_message = :pm")
                params["pm"] = kwargs["progress_message"]
            if "error_message" in kwargs:
                sets.append("error_message = :em")
                params["em"] = kwargs["error_message"]
            if "error_details" in kwargs:
                sets.append("error_details = CAST(:ed AS jsonb)")
                params["ed"] = json.dumps(kwargs["error_details"])
            # Track last_error and increment error_count
            if "last_error" in kwargs:
                sets.append("last_error = :le")
                params["le"] = kwargs["last_error"]
                sets.append("error_count = COALESCE(error_count, 0) + 1")

            conn.execute(
                text(f"UPDATE workflow_steps SET {', '.join(sets)} WHERE session_id::text = :sid AND step_number = :step"),
                params,
            )
            conn.execute(
                text("UPDATE workflow_sessions SET current_step = :step, updated_at = NOW() WHERE session_id::text = :sid"),
                {"sid": self.session_id, "step": step_number},
            )

    def _send_heartbeat(self, step_number, progress_message=None, progress_current=None, progress_total=None):
        """Send a heartbeat update without changing status."""
        with self.engine.begin() as conn:
            sets = ["heartbeat_at = NOW()"]
            params = {"sid": self.session_id, "step": step_number, "wid": self.worker_id}
            if progress_message is not None:
                sets.append("progress_message = :pm")
                params["pm"] = progress_message
            if progress_current is not None:
                sets.append("progress_current = :pc")
                params["pc"] = progress_current
            if progress_total is not None:
                sets.append("progress_total = :pt")
                params["pt"] = progress_total
            conn.execute(
                text(f"UPDATE workflow_steps SET {', '.join(sets)} WHERE session_id::text = :sid AND step_number = :step AND worker_id = :wid"),
                params,
            )

    def _save_artifact(self, step_number, artifact_type, artifact_name, content, metadata=None):
        """Save an artifact to the database and return the artifact_id."""
        with self.engine.begin() as conn:
            meta_val = json.dumps(metadata) if metadata else None
            art_id = conn.execute(
                text(
                    "INSERT INTO workflow_artifacts (session_id, step_number, artifact_type, artifact_name, artifact_content, artifact_metadata) "
                    "VALUES (CAST(:sid AS uuid), :step, :atype, :aname, :content, CAST(:meta AS jsonb)) "
                    "RETURNING artifact_id"
                ),
                {
                    "sid": self.session_id,
                    "step": step_number,
                    "atype": artifact_type,
                    "aname": artifact_name,
                    "content": content,
                    "meta": meta_val,
                },
            ).scalar()
        return art_id

    def _advance_step(self, completed_step):
        """Mark the next step as ready."""
        next_step = completed_step + 1
        if next_step > 7:
            return
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE workflow_steps SET status = 'ready' "
                    "WHERE session_id::text = :sid AND step_number = :step AND status = 'pending'"
                ),
                {"sid": self.session_id, "step": next_step},
            )

    def _get_artifact_content(self, artifact_type):
        """Retrieve an artifact's content by type."""
        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    "SELECT artifact_content FROM workflow_artifacts "
                    "WHERE session_id::text = :sid AND artifact_type = :atype "
                    "ORDER BY created_at DESC LIMIT 1"
                ),
                {"sid": self.session_id, "atype": artifact_type},
            ).scalar()
        return row

    def _get_categories(self):
        """Get selected product categories for this session."""
        content = self._get_artifact_content("product_categories")
        if content:
            return [c.strip() for c in content.split("\n") if c.strip()]
        return []

    def _get_category_classifications(self):
        """Get category classifications (core vs cross-platform) from the category_selections artifact.

        Returns a dict with keys: core_product_categories, cross_platform_capabilities, skipped.
        Each value is a list of category name strings.
        Falls back to treating all categories as core if no classification is found.
        """
        content = self._get_artifact_content("category_selections")
        if content:
            try:
                return json.loads(content)
            except (json.JSONDecodeError, TypeError):
                pass
        # Fallback: treat all categories as core
        all_cats = self._get_categories()
        return {"core_product_categories": all_cats, "cross_platform_capabilities": [], "skipped": []}

    def _count_tokens(self, text_content: str) -> int:
        """Return estimated token count using tiktoken (cl100k_base) or char-based fallback."""
        if not text_content:
            return 0
        if _TIKTOKEN_ENCODING:
            return len(_TIKTOKEN_ENCODING.encode(text_content))
        return len(text_content) // 4  # rough char-based estimate

    def _record_turn(self, step_number, turn_type, role, content_type, content, model_name=None, artifact_id=None):
        """Insert a row into agent_turns to track the agent trajectory."""
        preview = (content[:500] if content else "")
        token_count = self._count_tokens(content) if content else 0
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO agent_turns (session_id, step_number, turn_type, role, content_type, "
                    "content_preview, token_count, model_name, artifact_id) "
                    "VALUES (CAST(:sid AS uuid), :step, :ttype, :role, :ctype, :preview, :tokens, :model, :aid)"
                ),
                {
                    "sid": self.session_id,
                    "step": step_number,
                    "ttype": turn_type,
                    "role": role,
                    "ctype": content_type,
                    "preview": preview,
                    "tokens": token_count,
                    "model": model_name,
                    "aid": artifact_id,
                },
            )

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def _get_previous_version_context(self):
        """Load review feedback and fact-check results from the previous generation (if linked).

        Returns a dict with keys 'review_feedback' and 'fact_check_results',
        each a plain-text string (empty if unavailable).
        """
        result = {"review_feedback": "", "fact_check_results": ""}

        with self.engine.begin() as conn:
            # Find previous_generation_id via this session's generation
            row = conn.execute(
                text(
                    "SELECT bg.previous_generation_id "
                    "FROM workflow_sessions ws "
                    "JOIN battlecard_generations bg ON ws.generation_id = bg.generation_id "
                    "WHERE ws.session_id::text = :sid"
                ),
                {"sid": self.session_id},
            ).mappings().first()

        if not row or not row["previous_generation_id"]:
            return result

        prev_gen_id = row["previous_generation_id"]

        # --- Review feedback ---
        with self.engine.begin() as conn:
            reviews = conn.execute(
                text(
                    "SELECT review_type, feedback_text, reviewed_by, reviewed_at "
                    "FROM human_reviews WHERE generation_id = :gid ORDER BY reviewed_at"
                ),
                {"gid": prev_gen_id},
            ).mappings().all()

        if reviews:
            parts = ["## Previous Version Review Feedback\n"]
            for r in reviews:
                review_type = r["review_type"]
                feedback_raw = r["feedback_text"] or ""
                reviewer = r["reviewed_by"] or "anonymous"
                comment = ""
                scope = ""
                try:
                    payload = json.loads(feedback_raw)
                    comment = payload.get("comment", "")
                    scope = payload.get("scope", "")
                except (json.JSONDecodeError, TypeError, ValueError):
                    comment = feedback_raw

                status_label = {
                    "approve": "APPROVED",
                    "request_edit": "NEEDS REVISION",
                    "reject": "REJECTED",
                }.get(review_type, review_type.upper())
                line = f"- [{status_label}]"
                if scope:
                    line += f" (scope: {scope})"
                if comment:
                    line += f": {comment}"
                line += f"  — {reviewer}"
                parts.append(line)
            result["review_feedback"] = "\n".join(parts)

        # --- Fact-check results ---
        with self.engine.begin() as conn:
            fact_checks = conn.execute(
                text(
                    "SELECT fc.status AS fc_status, fc.reasoning, fc.fact_check_source_text, "
                    "fc.confidence_score, e.traces_to_text, "
                    "c.headline AS claim_headline, kd.key_diff_name "
                    "FROM fact_checks fc "
                    "JOIN evidence e ON fc.evidence_id = e.evidence_id "
                    "JOIN claims c ON e.claim_id = c.claim_id "
                    "JOIN key_differentiators kd ON c.key_diff_id = kd.key_diff_id "
                    "WHERE c.generation_id = :gid "
                    "ORDER BY kd.key_diff_name, c.claim_id"
                ),
                {"gid": prev_gen_id},
            ).mappings().all()

        if fact_checks:
            parts = ["## Previous Version Fact-Check Results\n"]
            for fc in fact_checks:
                status = fc["fc_status"] or "pending"
                kd = fc["key_diff_name"] or ""
                claim = fc["claim_headline"] or ""
                reasoning = fc["reasoning"] or ""
                confidence = fc["confidence_score"]
                traced = fc["traces_to_text"] or ""

                line = f"- [{status.upper()}] {kd} / {claim}"
                if traced:
                    line += f' — claim text: "{traced[:200]}"'
                if reasoning:
                    line += f" — reason: {reasoning[:200]}"
                if confidence is not None:
                    line += f" (confidence: {confidence}%)"
                parts.append(line)
            result["fact_check_results"] = "\n".join(parts)

        return result

    def _build_context(self, context_sources=None):
        """Build the XML-tagged context string from selected sources.

        Args:
            context_sources: Optional dict with boolean flags for each source:
                - directive (default True)
                - old_battlecard (default True)
                - review_feedback (default False)
                - fact_checks (default False)
            When None, uses default behavior (directive + old_battlecard only).
        """
        if context_sources is None:
            context_sources = {}

        include_directive = context_sources.get("directive", True)
        include_old_bc = context_sources.get("old_battlecard", True)
        include_reviews = context_sources.get("review_feedback", False)
        include_fact_checks = context_sources.get("fact_checks", False)

        directive = (self._get_artifact_content("directive_generated") or "") if include_directive else ""
        old_battlecard = (self._get_artifact_content("old_battlecard_extracted") or "") if include_old_bc else ""

        review_feedback = ""
        fact_check_results = ""
        if include_reviews or include_fact_checks:
            prev_ctx = self._get_previous_version_context()
            if include_reviews:
                review_feedback = prev_ctx.get("review_feedback", "")
            if include_fact_checks:
                fact_check_results = prev_ctx.get("fact_check_results", "")

        return format_context_xml(
            directive, old_battlecard, self.competitor,
            review_feedback=review_feedback,
            fact_check_results=fact_check_results,
        )

    # ------------------------------------------------------------------
    # Pass 1: Generate Key Differentiators (Skeletons)
    # ------------------------------------------------------------------

    def _process_single_category(self, category, all_core_cats, cross_cats,
                                  template_text, directive, context, feedback=None):
        """Generate Pass 1 skeletons for ONE core product category. Returns list[dict]."""
        other_cats = [c for c in all_core_cats if c != category]
        other_text = "\n".join(f"- {c}" for c in other_cats) if other_cats else "_(none)_"
        cross_text = "\n".join(f"- {c}" for c in cross_cats) if cross_cats else "_(none)_"

        rendered = render_template(
            template_text,
            competitor=self.competitor,
            product_area=self.product_area,
            comparison=f"Databricks vs {self.competitor}",
            target_category=category,
            other_core_categories=other_text,
            cross_platform_capabilities=cross_text,
            diffs_per_category=str(self.diffs_per_category),
            directives=directive,
            context=context,
        )

        if feedback:
            rendered += f"\n\n## Previous Feedback\n{feedback}\n\nPlease regenerate incorporating this feedback."

        raw = call_model(
            client=self.client,
            model_name=self.model_name,
            rendered_prompt=rendered,
            json_schema=L200_PASS1_JSON_SCHEMA,
        )
        return json.loads(raw).get("slides", [])

    def _stub_pass1_category(self, category):
        """Create N stub skeletons for a category that failed (fallback on error)."""
        stubs = []
        for rank in range(1, self.diffs_per_category + 1):
            stubs.append({
                "id": f"{category.replace(' ', '_')}_stub_{rank}",
                "competitor": self.competitor,
                "category": category,
                "rank": rank,
                "key_differentiator": f"Stub Diff {rank}",
                "description": "Generation failed — stub differentiator.",
                "selection_reasoning": "Generation failed — stub reasoning.",
                "rank_reasoning": "Generation failed — stub reasoning.",
                "directive_alignment": "N/A",
                "databricks_rating": "partial",
                "competitor_rating": "partial",
            })
        return stubs

    def run_pass1(self, feedback=None, context_sources=None):
        """Run Pass 1 to generate key differentiator skeletons using the LLM."""
        self._update_step(4, "in_progress",
                          progress_message="[1/6] Preparing Pass 1 — loading categories and directive...",
                          progress_current=0, progress_total=6)

        categories = self._get_categories()
        if not categories:
            self._update_step(4, "failed", error_message="No product categories selected. Complete Step 3 first.")
            return

        directive = self._get_artifact_content("directive_generated") or ""

        try:
            # Stage 2: Load prompt template
            ver = getattr(self, 'pass1_prompt_template_version', 2)

            # For V3+, use category classifications to compute total_diffs from core categories only
            classifications = self._get_category_classifications()
            core_cats = classifications.get("core_product_categories", [])
            cross_cats = classifications.get("cross_platform_capabilities", [])

            if ver >= 3 and core_cats:
                # V3: slides are only generated for core product categories
                total_diffs = len(core_cats) * self.diffs_per_category
            else:
                # V1/V2 or fallback: all categories get slides
                total_diffs = len(categories) * self.diffs_per_category

            # Determine if we should use parallel per-category path
            use_parallel_pass1 = (ver >= 3 and self.max_workers > 1 and len(core_cats) > 1)

            if use_parallel_pass1:
                # Load V4 per-category template for parallel execution
                template_text, template_file = load_pass1_template(4, engine=self.engine)
                prompt_label = f"V4 (parallel per-category, {len(core_cats)} categories)"
            else:
                template_text, template_file = load_pass1_template(ver, engine=self.engine)
                prompt_label = f"V{ver}"

            self._update_step(4, "in_progress",
                              progress_message=f"[2/6] Loading Pass 1 prompt template {prompt_label} ({total_diffs} diffs across {len(core_cats) if ver >= 3 else len(categories)} categories)...",
                              progress_current=1, progress_total=6)

            # Build context
            context = self._build_context(context_sources)

            # Stage 3: Store the prompt version
            self._update_step(4, "in_progress",
                              progress_message=f"[3/6] Registering prompt version...",
                              progress_current=2, progress_total=6)

            pass1_version_id = ensure_prompt_version(
                self.engine,
                prompt_name="l200_pass1_planning",
                prompt_file=template_file,
                prompt_text=template_text,
            )
            with self.engine.begin() as conn:
                conn.execute(
                    text("UPDATE workflow_sessions SET pass1_prompt_version_id = :vid, updated_at = NOW() WHERE session_id::text = :sid"),
                    {"vid": pass1_version_id, "sid": self.session_id},
                )

            # ---------- PARALLEL PER-CATEGORY PATH ----------
            if use_parallel_pass1:
                errors = []
                completed = 0
                workers_label = f"{self.max_workers} parallel workers"

                # Record a representative prompt for the first category
                first_cat = core_cats[0]
                other_text = "\n".join(f"- {c}" for c in core_cats[1:])
                cross_text_repr = "\n".join(f"- {c}" for c in cross_cats) if cross_cats else "_(none)_"
                representative_prompt = render_template(
                    template_text,
                    competitor=self.competitor,
                    product_area=self.product_area,
                    comparison=f"Databricks vs {self.competitor}",
                    target_category=first_cat,
                    other_core_categories=other_text,
                    cross_platform_capabilities=cross_text_repr,
                    diffs_per_category=str(self.diffs_per_category),
                    directives=directive,
                    context=context,
                )
                self._record_turn(4, "system_prompt", "system", "pass1_prompt", representative_prompt, model_name=self.model_name)

                self._update_step(4, "in_progress",
                                  progress_message=f"[4/6] Generating diffs — 0/{len(core_cats)} categories done ({workers_label}, model: {self.model_name})...",
                                  progress_current=3, progress_total=3 + len(core_cats))

                # Submit one task per core category
                futures = {}
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    for cat in core_cats:
                        future = executor.submit(
                            self._process_single_category,
                            cat, core_cats, cross_cats,
                            template_text, directive, context, feedback,
                        )
                        futures[future] = cat

                    # Collect results, saving incremental artifacts as each completes
                    results_by_cat = {}
                    for future in as_completed(futures):
                        cat = futures[future]
                        try:
                            cat_skeletons = future.result()
                            results_by_cat[cat] = cat_skeletons
                        except Exception as e:
                            logger.error("Pass 1 failed for category '%s': %s", cat, e)
                            results_by_cat[cat] = self._stub_pass1_category(cat)
                            errors.append({"category": cat, "error": str(e)})

                        completed += 1
                        error_suffix = f" ({len(errors)} errors)" if errors else ""

                        # Accumulate results in original order for incremental save
                        accumulated = []
                        for c in core_cats:
                            if c in results_by_cat:
                                accumulated.extend(results_by_cat[c])

                        self._update_step(4, "in_progress",
                                          progress_current=3 + completed,
                                          progress_total=3 + len(core_cats),
                                          progress_message=f"[4/6] Generating diffs — {completed}/{len(core_cats)} categories done{error_suffix}: {cat}")

                        # Incremental save: partial artifact so the frontend can poll and display
                        partial_content = json.dumps(accumulated, indent=2)
                        self._save_artifact(
                            4, "pass1_skeletons", "key_differentiators.json",
                            partial_content,
                            metadata={
                                "count": len(accumulated),
                                "partial": completed < len(core_cats),
                                "completed_categories": completed,
                                "total_categories": len(core_cats),
                                "categories": [c for c in core_cats if c in results_by_cat],
                                "model": self.model_name,
                                "prompt_version_id": pass1_version_id,
                            },
                        )

                # Reassemble all skeletons in original category order
                all_skeletons = []
                for c in core_cats:
                    all_skeletons.extend(results_by_cat.get(c, []))

                logger.info("Pass 1 (parallel) returned %d skeletons across %d categories (expected %d)",
                            len(all_skeletons), len(core_cats), total_diffs)

                # Stage 5: Save final artifact
                self._update_step(4, "in_progress",
                                  progress_message=f"[5/6] Saving {len(all_skeletons)} key differentiators...",
                                  progress_current=3 + len(core_cats), progress_total=3 + len(core_cats) + 1)

                skeletons_content = json.dumps(all_skeletons, indent=2)
                art_id = self._save_artifact(
                    4, "pass1_skeletons", "key_differentiators.json",
                    skeletons_content,
                    metadata={
                        "count": len(all_skeletons),
                        "partial": False,
                        "completed_categories": len(core_cats),
                        "total_categories": len(core_cats),
                        "categories": core_cats,
                        "model": self.model_name,
                        "prompt_version_id": pass1_version_id,
                        "parallel": True,
                        "max_workers": self.max_workers,
                        "errors": errors,
                    },
                )

                self._record_turn(4, "model_output", "assistant", "pass1_skeletons", skeletons_content,
                                  model_name=self.model_name, artifact_id=art_id)

                # Stage 6: Save to Lakebase tables
                self._update_step(4, "in_progress",
                                  progress_message=f"[6/6] Writing {len(all_skeletons)} key differentiators to database...",
                                  progress_current=3 + len(core_cats) + 1, progress_total=3 + len(core_cats) + 2)

                try:
                    self._save_skeletons_to_lakebase(all_skeletons)
                except Exception as e:
                    logger.exception("Failed to save skeletons to Lakebase (non-fatal)")

                error_suffix = f" ({len(errors)} had errors — used fallback stubs)" if errors else ""
                self._update_step(4, "waiting_human",
                                  progress_message=f"Generated {len(all_skeletons)} key differentiators across {len(core_cats)} categories (parallel).{error_suffix} Review and approve or provide feedback.",
                                  progress_current=6, progress_total=6,
                                  error_details={"errors": errors} if errors else None)
                return

            # ---------- SINGLE-CALL PATH (V1/V2 or non-parallel) ----------
            # Build template variables based on version
            if ver >= 3 and core_cats:
                # V3+: separate core and cross-platform category lists
                core_text = "\n".join(f"- {c}" for c in core_cats)
                cross_text = "\n".join(f"- {c}" for c in cross_cats) if cross_cats else "_(none selected)_"
                rendered = render_template(
                    template_text,
                    competitor=self.competitor,
                    product_area=self.product_area,
                    comparison=f"Databricks vs {self.competitor}",
                    core_product_categories=core_text,
                    cross_platform_capabilities=cross_text,
                    core_category_count=str(len(core_cats)),
                    diffs_per_category=str(self.diffs_per_category),
                    total_diffs=str(total_diffs),
                    directives=directive,
                    context=context,
                )
            else:
                # V1/V2: flat list of all categories
                categories_text = "\n".join(f"- {c}" for c in categories)
                rendered = render_template(
                    template_text,
                    competitor=self.competitor,
                    product_area=self.product_area,
                    comparison=f"Databricks vs {self.competitor}",
                    product_categories=categories_text,
                    diffs_per_category=str(self.diffs_per_category),
                    total_diffs=str(total_diffs),
                    directives=directive,
                    context=context,
                )

            # Append feedback if provided (regeneration)
            if feedback:
                rendered += f"\n\n## Previous Feedback\n{feedback}\n\nPlease regenerate incorporating this feedback."

            # Record the prompt as an agent turn
            self._record_turn(4, "system_prompt", "system", "pass1_prompt", rendered, model_name=self.model_name)

            # Stage 4: Call the model
            self._update_step(4, "in_progress",
                              progress_message=f"[4/6] Calling {self.model_name} — generating {total_diffs} key differentiators across {len(categories)} categories...",
                              progress_current=3, progress_total=6)

            raw = call_model(
                client=self.client,
                model_name=self.model_name,
                rendered_prompt=rendered,
                json_schema=L200_PASS1_JSON_SCHEMA,
            )

            # Stage 5: Parse and save
            self._update_step(4, "in_progress",
                              progress_message="[5/6] Parsing LLM response and saving artifacts...",
                              progress_current=4, progress_total=6)

            parsed = json.loads(raw)
            skeletons = parsed.get("slides", [])

            logger.info("Pass 1 returned %d skeletons (expected %d)", len(skeletons), total_diffs)

            # Save artifact
            skeletons_content = json.dumps(skeletons, indent=2)
            art_id = self._save_artifact(
                4, "pass1_skeletons", "key_differentiators.json",
                skeletons_content,
                metadata={
                    "count": len(skeletons),
                    "partial": False,
                    "categories": categories,
                    "model": self.model_name,
                    "prompt_version_id": pass1_version_id,
                },
            )

            # Record the output as an agent turn
            self._record_turn(4, "model_output", "assistant", "pass1_skeletons", skeletons_content,
                              model_name=self.model_name, artifact_id=art_id)

            # Stage 6: Save to Lakebase tables
            self._update_step(4, "in_progress",
                              progress_message=f"[6/6] Writing {len(skeletons)} key differentiators to database...",
                              progress_current=5, progress_total=6)

            try:
                self._save_skeletons_to_lakebase(skeletons)
            except Exception as e:
                logger.exception("Failed to save skeletons to Lakebase (non-fatal)")
                # Continue — artifact was saved, Lakebase write is bonus

            self._update_step(4, "waiting_human",
                              progress_message=f"Generated {len(skeletons)} key differentiators across {len(categories)} categories. Review and approve or provide feedback.",
                              progress_current=6, progress_total=6)

        except json.JSONDecodeError as e:
            logger.exception("Pass 1 failed: invalid JSON from LLM")
            self._update_step(4, "failed",
                              error_message=f"LLM returned invalid JSON: {e}",
                              error_details={"stage": "parse_response", "raw_preview": raw[:500] if raw else ""})
        except Exception as e:
            logger.exception("Pass 1 generation failed")
            self._update_step(4, "failed", error_message=f"Pass 1 failed: {e}")

    def _save_skeletons_to_lakebase(self, skeletons):
        """Save skeleton key differentiators to the Lakebase tables."""
        with self.engine.begin() as conn:
            for sk in skeletons:
                category_name = sk.get("category", "")
                cat_id = conn.execute(
                    text("SELECT category_id FROM product_categories WHERE category_name = :name"),
                    {"name": category_name},
                ).scalar()
                if not cat_id:
                    cat_id = conn.execute(
                        text(
                            "INSERT INTO product_categories (category_name, category_description, display_order) "
                            "VALUES (:name, :desc, :order) RETURNING category_id"
                        ),
                        {"name": category_name, "desc": "", "order": 0},
                    ).scalar()

                conn.execute(
                    text(
                        "INSERT INTO key_differentiators (category_id, key_diff_name, key_diff_description, display_order) "
                        "VALUES (:cat, :name, :desc, :order)"
                    ),
                    {
                        "cat": cat_id,
                        "name": sk.get("key_differentiator", ""),
                        "desc": sk.get("description", ""),
                        "order": sk.get("rank", 0),
                    },
                )

    # ------------------------------------------------------------------
    # Pass 2: Generate Claims for each Key Differentiator
    # ------------------------------------------------------------------

    def run_pass2(self, context_sources=None):
        """Run Pass 2 to generate detailed claims for each key differentiator."""
        self._update_step(5, "in_progress",
                          progress_message="[1/5] Loading skeletons from Pass 1...",
                          progress_current=0, progress_total=5)

        skeletons_json = self._get_artifact_content("pass1_skeletons")
        if not skeletons_json:
            self._update_step(5, "failed", error_message="No skeletons found. Complete Step 4 first.")
            return

        skeletons = json.loads(skeletons_json)
        total = len(skeletons)
        directive = self._get_artifact_content("directive_generated") or ""
        context = self._build_context(context_sources)

        try:
            # Stage 2: Load template (version-aware)
            ver = getattr(self, 'pass2_prompt_template_version', 2)
            self._update_step(5, "in_progress",
                              progress_message=f"[2/5] Loading Pass 2 prompt template V{ver} ({total} diffs to process)...",
                              progress_current=1, progress_total=5)
            template_text = load_pass2_template(ver, engine=self.engine)
            logger.info("Using Pass 2 prompt template version %d (%d chars)", ver, len(template_text))

            # Store the prompt version
            prompt_label = f"l200_pass2_detail_v{ver}"
            pass2_version_id = ensure_prompt_version(
                self.engine,
                prompt_name=prompt_label,
                prompt_file=f"pass2_template_v{ver}",
                prompt_text=template_text,
            )
            with self.engine.begin() as conn:
                conn.execute(
                    text("UPDATE workflow_sessions SET pass2_prompt_version_id = :vid, updated_at = NOW() WHERE session_id::text = :sid"),
                    {"vid": pass2_version_id, "sid": self.session_id},
                )

            all_claims = []
            completed = 0
            errors = []  # Track per-skeleton errors

            # Record a representative Pass 2 prompt (using first skeleton)
            if skeletons:
                first_sk = skeletons[0]
                representative_prompt = render_template(
                    template_text,
                    competitor=self.competitor,
                    category=first_sk.get("category", ""),
                    key_differentiator=first_sk.get("key_differentiator", ""),
                    description=first_sk.get("description", ""),
                    databricks_rating=first_sk.get("databricks_rating", ""),
                    competitor_rating=first_sk.get("competitor_rating", ""),
                    selection_reasoning=first_sk.get("selection_reasoning", ""),
                    directives=directive,
                    context=context,
                )
                self._record_turn(5, "system_prompt", "system", "pass2_prompt",
                                  representative_prompt, model_name=self.model_name)

            def _process_single_diff(idx: int, sk: dict) -> dict:
                """Generate claims for a single skeleton diff."""
                rendered = render_template(
                    template_text,
                    competitor=self.competitor,
                    category=sk.get("category", ""),
                    key_differentiator=sk.get("key_differentiator", ""),
                    description=sk.get("description", ""),
                    databricks_rating=sk.get("databricks_rating", ""),
                    competitor_rating=sk.get("competitor_rating", ""),
                    selection_reasoning=sk.get("selection_reasoning", ""),
                    directives=directive,
                    context=context,
                )

                raw = call_model(
                    client=self.client,
                    model_name=self.model_name,
                    rendered_prompt=rendered,
                    json_schema=L200_PASS2_JSON_SCHEMA,
                )
                return json.loads(raw)

            # Stage 3: Generate claims
            workers_label = f"{self.max_workers} parallel workers" if self.max_workers > 1 else "sequential"
            self._update_step(5, "in_progress",
                              progress_message=f"[3/5] Generating claims — 0/{total} done ({workers_label}, model: {self.model_name}, prompt: V{ver})...",
                              progress_current=0, progress_total=total)

            # Process with concurrency using max_workers
            if self.max_workers > 1 and total > 1:
                futures = {}
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    for idx, sk in enumerate(skeletons):
                        future = executor.submit(_process_single_diff, idx, sk)
                        futures[future] = (idx, sk)

                    # Collect results in order
                    results_by_idx = {}
                    for future in as_completed(futures):
                        idx, sk = futures[future]
                        kd_name = sk.get("key_differentiator", "")[:50]
                        try:
                            claim = future.result()
                            results_by_idx[idx] = claim
                        except Exception as e:
                            error_str = str(e)
                            logger.error("Pass 2 failed for skeleton %d (%s): %s", idx, kd_name, error_str)
                            results_by_idx[idx] = self._stub_pass2_claim(sk)
                            errors.append({"index": idx, "key_differentiator": kd_name, "error": error_str})
                            # Store error in step for visibility
                            self._update_step(5, "in_progress", last_error=f"[{kd_name}] {error_str[:500]}")

                        completed += 1
                        error_suffix = f" ({len(errors)} errors)" if errors else ""
                        self._update_step(5, "in_progress",
                                          progress_current=completed, progress_total=total,
                                          progress_message=f"[3/5] Generating claims — {completed}/{total} done{error_suffix}: {kd_name}")

                        # Incremental save: partial artifact so the frontend can poll and display
                        accumulated_claims = [results_by_idx[i] for i in range(total) if i in results_by_idx]
                        partial_content = json.dumps(accumulated_claims, indent=2)
                        self._save_artifact(
                            5, "pass2_claims", "claims.json",
                            partial_content,
                            metadata={
                                "count": len(accumulated_claims),
                                "partial": completed < total,
                                "completed": completed,
                                "total": total,
                                "model": self.model_name,
                                "errors": errors,
                            },
                        )

                # Reassemble in original order
                all_claims = [results_by_idx[i] for i in range(total)]
            else:
                # Sequential processing
                for idx, sk in enumerate(skeletons):
                    kd_name = sk.get("key_differentiator", "")[:50]
                    self._update_step(5, "in_progress",
                                      progress_current=idx, progress_total=total,
                                      progress_message=f"[3/5] Generating claim {idx + 1}/{total}: {kd_name}")

                    try:
                        claim = _process_single_diff(idx, sk)
                    except Exception as e:
                        error_str = str(e)
                        logger.error("Pass 2 failed for skeleton %d (%s): %s", idx, kd_name, error_str)
                        claim = self._stub_pass2_claim(sk)
                        errors.append({"index": idx, "key_differentiator": kd_name, "error": error_str})
                        # Store error in step for visibility
                        self._update_step(5, "in_progress", last_error=f"[{kd_name}] {error_str[:500]}")
                    all_claims.append(claim)

            # Stage 4: Save artifact
            self._update_step(5, "in_progress",
                              progress_message=f"[4/5] Saving {len(all_claims)} claims to artifacts...",
                              progress_current=total, progress_total=total)

            claims_content = json.dumps(all_claims, indent=2)
            art_id = self._save_artifact(
                5, "pass2_claims", "claims.json",
                claims_content,
                metadata={
                    "count": len(all_claims),
                    "model": self.model_name,
                    "prompt_version_id": pass2_version_id,
                    "max_workers": self.max_workers,
                    "errors": errors,
                },
            )

            # Record the output as an agent turn
            self._record_turn(5, "model_output", "assistant", "pass2_claims", claims_content,
                              model_name=self.model_name, artifact_id=art_id)

            # Stage 5: Save to Lakebase tables
            self._update_step(5, "in_progress",
                              progress_message=f"[5/5] Writing {len(all_claims)} claims to database...",
                              progress_current=total, progress_total=total)

            try:
                self._save_claims_to_lakebase(skeletons, all_claims)
            except Exception as e:
                logger.exception("Failed to save claims to Lakebase (non-fatal)")
                errors.append({"index": -1, "key_differentiator": "lakebase_write", "error": str(e)})

            error_suffix = f" ({len(errors)} had errors — used fallback stubs)" if errors else ""
            self._update_step(5, "waiting_human",
                              progress_message=f"Generated {len(all_claims)} claim pairs.{error_suffix} Review and approve or provide feedback.",
                              progress_current=total, progress_total=total,
                              error_details={"errors": errors} if errors else None)

        except Exception as e:
            logger.exception("Pass 2 generation failed")
            self._update_step(5, "failed", error_message=f"Pass 2 failed: {e}",
                              error_details={"errors": errors} if errors else None)

    def _stub_pass2_claim(self, skeleton):
        """Create a stub claim for a skeleton differentiator (fallback on error)."""
        kd_name = skeleton.get("key_differentiator", "Unknown")
        return {
            "databricks_headline": f"Databricks: {kd_name}",
            "databricks_details": [f"Databricks provides strong capabilities in {kd_name}."],
            "databricks_reasoning": "Generation failed — stub reasoning.",
            "competitor_headline": f"{self.competitor}: {kd_name}",
            "competitor_details": [f"{self.competitor} has partial support for {kd_name}."],
            "competitor_reasoning": "Generation failed — stub reasoning.",
            "citations": {
                "databricks_details": [],
                "databricks_reasoning": [],
                "competitor_details": [],
                "competitor_reasoning": [],
            },
            "sources": [],
            "research_sources": [],
        }

    def _save_claims_to_lakebase(self, skeletons, claims):
        """Save Pass 2 claims into the Lakebase claims/evidence/fact_checks tables."""
        with self.engine.begin() as conn:
            # Reuse the generation_id created at workflow start (if it exists),
            # otherwise create a new one for backwards compatibility.
            gen_id = conn.execute(
                text("SELECT generation_id FROM workflow_sessions WHERE session_id::text = :sid"),
                {"sid": self.session_id},
            ).scalar()

            if not gen_id:
                gen_id = conn.execute(
                    text(
                        "INSERT INTO battlecard_generations (trigger_type, generated_by, generation_model, status) "
                        "VALUES ('manual_request', 'workflow_runner', :model, 'draft') RETURNING generation_id"
                    ),
                    {"model": self.model_name},
                ).scalar()
                conn.execute(
                    text("UPDATE workflow_sessions SET generation_id = :gid, updated_at = NOW() WHERE session_id::text = :sid"),
                    {"gid": gen_id, "sid": self.session_id},
                )

            # Clear any existing claims for this generation (in case of regeneration)
            conn.execute(
                text(
                    "DELETE FROM fact_checks WHERE evidence_id IN "
                    "(SELECT e.evidence_id FROM evidence e JOIN claims c ON e.claim_id = c.claim_id WHERE c.generation_id = :gid)"
                ),
                {"gid": gen_id},
            )
            conn.execute(
                text(
                    "DELETE FROM evidence WHERE claim_id IN "
                    "(SELECT claim_id FROM claims WHERE generation_id = :gid)"
                ),
                {"gid": gen_id},
            )
            conn.execute(
                text(
                    "DELETE FROM claim_detail_items WHERE claim_id IN "
                    "(SELECT claim_id FROM claims WHERE generation_id = :gid)"
                ),
                {"gid": gen_id},
            )
            conn.execute(
                text("DELETE FROM claims WHERE generation_id = :gid"),
                {"gid": gen_id},
            )

            # Ensure companies exist
            db_id = conn.execute(
                text("SELECT company_id FROM companies WHERE company_name = 'Databricks' AND company_type = 'databricks' LIMIT 1")
            ).scalar()
            if not db_id:
                db_id = conn.execute(
                    text("INSERT INTO companies (company_name, company_type) VALUES ('Databricks', 'databricks') RETURNING company_id")
                ).scalar()

            comp_id = conn.execute(
                text("SELECT company_id FROM companies WHERE company_name = :name AND company_type = 'competitor' LIMIT 1"),
                {"name": self.competitor},
            ).scalar()
            if not comp_id:
                comp_id = conn.execute(
                    text("INSERT INTO companies (company_name, company_type) VALUES (:name, 'competitor') RETURNING company_id"),
                    {"name": self.competitor},
                ).scalar()

            total_claims = 0
            for sk, claim in zip(skeletons, claims):
                kd_name = sk.get("key_differentiator", "")
                kd_id = conn.execute(
                    text("SELECT key_diff_id FROM key_differentiators WHERE key_diff_name = :name LIMIT 1"),
                    {"name": kd_name},
                ).scalar()
                if not kd_id:
                    continue

                def _rating_to_symbol(rating):
                    if rating in ("advantage", "strong_advantage", "positive"):
                        return "(+)"
                    if rating in ("partial", "neutral"):
                        return "(~)"
                    if rating in ("disadvantage", "negative"):
                        return "(-)"
                    return "(~)"

                def _rating_to_db(rating):
                    if rating in ("advantage", "strong_advantage", "positive"):
                        return "positive"
                    if rating in ("partial", "neutral"):
                        return "neutral"
                    if rating in ("disadvantage", "negative"):
                        return "negative"
                    return "neutral"

                # Databricks claim
                db_rating = _rating_to_db(sk.get("databricks_rating", ""))
                db_details_raw = claim.get("databricks_details", [])
                if isinstance(db_details_raw, str):
                    db_details_raw = [db_details_raw] if db_details_raw else []
                db_desc_flat = " ".join(db_details_raw)

                db_claim_id = conn.execute(
                    text(
                        "INSERT INTO claims (generation_id, key_diff_id, company_id, rating, rating_symbol, headline, description, change_type) "
                        "VALUES (:gen, :kd, :co, :rating, :symbol, :head, :desc, 'new') RETURNING claim_id"
                    ),
                    {
                        "gen": gen_id,
                        "kd": kd_id,
                        "co": db_id,
                        "rating": db_rating,
                        "symbol": _rating_to_symbol(sk.get("databricks_rating", "")),
                        "head": claim.get("databricks_headline", ""),
                        "desc": db_desc_flat,
                    },
                ).scalar()
                total_claims += 1

                # Insert detail items for Databricks claim
                db_detail_item_ids = []
                for order, item_text in enumerate(db_details_raw):
                    did = conn.execute(
                        text(
                            "INSERT INTO claim_detail_items (claim_id, item_order, item_text) "
                            "VALUES (:cid, :ord, :txt) RETURNING detail_item_id"
                        ),
                        {"cid": db_claim_id, "ord": order, "txt": item_text},
                    ).scalar()
                    db_detail_item_ids.append(did)

                # Competitor claim
                comp_rating = _rating_to_db(sk.get("competitor_rating", ""))
                comp_details_raw = claim.get("competitor_details", [])
                if isinstance(comp_details_raw, str):
                    comp_details_raw = [comp_details_raw] if comp_details_raw else []
                comp_desc_flat = " ".join(comp_details_raw)

                comp_claim_id = conn.execute(
                    text(
                        "INSERT INTO claims (generation_id, key_diff_id, company_id, rating, rating_symbol, headline, description, change_type) "
                        "VALUES (:gen, :kd, :co, :rating, :symbol, :head, :desc, 'new') RETURNING claim_id"
                    ),
                    {
                        "gen": gen_id,
                        "kd": kd_id,
                        "co": comp_id,
                        "rating": comp_rating,
                        "symbol": _rating_to_symbol(sk.get("competitor_rating", "")),
                        "head": claim.get("competitor_headline", ""),
                        "desc": comp_desc_flat,
                    },
                ).scalar()
                total_claims += 1

                # Insert detail items for competitor claim
                comp_detail_item_ids = []
                for order, item_text in enumerate(comp_details_raw):
                    did = conn.execute(
                        text(
                            "INSERT INTO claim_detail_items (claim_id, item_order, item_text) "
                            "VALUES (:cid, :ord, :txt) RETURNING detail_item_id"
                        ),
                        {"cid": comp_claim_id, "ord": order, "txt": item_text},
                    ).scalar()
                    comp_detail_item_ids.append(did)

                # Save evidence + fact checks from citations
                self._save_evidence_and_fact_checks(
                    conn, gen_id, db_claim_id, claim, "databricks", db_detail_item_ids
                )
                self._save_evidence_and_fact_checks(
                    conn, gen_id, comp_claim_id, claim, "competitor", comp_detail_item_ids
                )

            conn.execute(
                text("UPDATE battlecard_generations SET total_claims = :tc WHERE generation_id = :gid"),
                {"tc": total_claims, "gid": gen_id},
            )

    def _save_evidence_and_fact_checks(self, conn, gen_id, claim_id, claim_data, side, detail_item_ids=None):
        """Save evidence rows and fact checks from the citations in the claim."""
        citations = claim_data.get("citations", {})
        sources_list = claim_data.get("sources", [])
        if detail_item_ids is None:
            detail_item_ids = []

        # Build source_index -> source mapping
        source_map = {}
        for src in sources_list:
            source_map[src.get("index")] = src

        for field_suffix in ("details", "reasoning"):
            field_key = f"{side}_{field_suffix}"
            field_citations = citations.get(field_key, [])

            for cite in field_citations:
                source_index = cite.get("source_index")
                src = source_map.get(source_index, {})

                # Find or create source
                source_id = None
                if src.get("url") or src.get("title"):
                    source_id = conn.execute(
                        text("SELECT source_id FROM sources WHERE source_url = :url LIMIT 1"),
                        {"url": src.get("url", "")},
                    ).scalar()
                    if not source_id and src.get("title"):
                        source_id = conn.execute(
                            text(
                                "INSERT INTO sources (source_name, source_url, source_type, publisher) "
                                "VALUES (:name, :url, :stype, :pub) RETURNING source_id"
                            ),
                            {
                                "name": src.get("title", "Source"),
                                "url": src.get("url", ""),
                                "stype": self._map_source_type(src.get("type", "context")),
                                "pub": "",
                            },
                        ).scalar()

                # Resolve detail_item_id for details citations
                detail_item_id = None
                if field_suffix == "details" and detail_item_ids:
                    detail_idx = cite.get("detail_item_index", 0)
                    if 0 <= detail_idx < len(detail_item_ids):
                        detail_item_id = detail_item_ids[detail_idx]

                traces_to_field = "detail_item" if detail_item_id else ("description" if field_suffix == "details" else "headline")

                # Insert evidence
                traces_text = cite.get("source_quote", "")[:500]
                evidence_id = conn.execute(
                    text(
                        "INSERT INTO evidence (claim_id, detail_item_id, traces_to_field, traces_to_start_index, traces_to_end_index, "
                        "traces_to_text, generation_source_id, generation_source_text) "
                        "VALUES (:claim, :did, :field, :start, :end, :trace, :src, :src_text) RETURNING evidence_id"
                    ),
                    {
                        "claim": claim_id,
                        "did": detail_item_id,
                        "field": traces_to_field,
                        "start": cite.get("start_index", 0),
                        "end": cite.get("end_index", 0),
                        "trace": traces_text,
                        "src": source_id,
                        "src_text": traces_text,
                    },
                ).scalar()

    def _map_source_type(self, src_type: str) -> str:
        """Map a source type string to the DB enum value."""
        mapping = {
            "documentation": "vendor",
            "blog": "vendor",
            "directive": "internal",
            "analyst_report": "analyst",
            "news": "third-party",
            "context": "internal",
        }
        return mapping.get(src_type, "third-party")

    # ------------------------------------------------------------------
    # Pass 3: Regenerate Claims from Feedback
    # ------------------------------------------------------------------

    def run_pass3(self, feedback="", context_sources=None):
        """Run Pass 3 to regenerate claims incorporating feedback."""
        self._update_step(5, "in_progress",
                          progress_message="[1/4] Loading existing claims and skeletons for regeneration...",
                          progress_current=0, progress_total=4)

        claims_json = self._get_artifact_content("pass2_claims")
        skeletons_json = self._get_artifact_content("pass1_skeletons")

        if not claims_json or not skeletons_json:
            self._update_step(5, "failed", error_message="No existing claims found. Complete Step 5 generation first.")
            return

        skeletons = json.loads(skeletons_json)
        claims = json.loads(claims_json)
        total = len(skeletons)
        directive = self._get_artifact_content("directive_generated") or ""
        context = self._build_context(context_sources)
        errors = []

        try:
            # Stage 2: Load template
            ver = getattr(self, 'pass2_prompt_template_version', 2)
            self._update_step(5, "in_progress",
                              progress_message=f"[2/4] Loading Pass 2 prompt template V{ver} for regeneration ({total} claims)...",
                              progress_current=1, progress_total=4)
            template_text = load_pass2_template(ver, engine=self.engine)

            # Stage 3: Regenerate claims sequentially
            updated_claims = []
            for idx, (sk, claim) in enumerate(zip(skeletons, claims)):
                kd_name = sk.get('key_differentiator', '')[:50]
                self._update_step(5, "in_progress",
                                  progress_current=idx, progress_total=total,
                                  progress_message=f"[3/4] Regenerating claim {idx + 1}/{total}: {kd_name}")

                # Build prompt: Pass 2 template + current content + feedback
                rendered = render_template(
                    template_text,
                    competitor=self.competitor,
                    category=sk.get("category", ""),
                    key_differentiator=sk.get("key_differentiator", ""),
                    description=sk.get("description", ""),
                    databricks_rating=sk.get("databricks_rating", ""),
                    competitor_rating=sk.get("competitor_rating", ""),
                    selection_reasoning=sk.get("selection_reasoning", ""),
                    directives=directive,
                    context=context,
                )

                # Append current content and feedback for regeneration
                current_content = json.dumps({**sk, **claim}, indent=2)
                rendered += (
                    f"\n\n## Previous Output (to improve upon)\n```json\n{current_content}\n```"
                    f"\n\n## Reviewer Feedback\n{feedback}"
                    f"\n\nIncorporate the feedback above while regenerating this differentiator's claims. "
                    f"Maintain or improve citation quality."
                )

                try:
                    raw = call_model(
                        client=self.client,
                        model_name=self.model_name,
                        rendered_prompt=rendered,
                        json_schema=L200_PASS2_JSON_SCHEMA,
                    )
                    updated = json.loads(raw)
                except Exception as e:
                    error_str = str(e)
                    logger.error("Pass 3 regen failed for %d (%s): %s", idx, kd_name, error_str)
                    updated = claim  # Keep original on failure
                    errors.append({"index": idx, "key_differentiator": kd_name, "error": error_str})
                    self._update_step(5, "in_progress", last_error=f"[Regen {kd_name}] {error_str[:500]}")

                updated_claims.append(updated)

            # Stage 4: Save artifacts and to Lakebase
            error_suffix = f" ({len(errors)} errors)" if errors else ""
            self._update_step(5, "in_progress",
                              progress_message=f"[4/4] Saving {len(updated_claims)} regenerated claims{error_suffix}...",
                              progress_current=total, progress_total=total)

            regen_content = json.dumps(updated_claims, indent=2)
            art_id = self._save_artifact(
                5, "pass3_regenerated", "claims_regenerated.json",
                regen_content,
                metadata={"count": len(updated_claims), "feedback": feedback[:200], "errors": errors},
            )

            # Record the regenerated output as an agent turn
            self._record_turn(5, "model_output", "assistant", "pass3_regenerated", regen_content,
                              model_name=self.model_name, artifact_id=art_id)

            # Update pass2_claims artifact so it becomes the "current" version
            self._save_artifact(
                5, "pass2_claims", "claims.json",
                regen_content,
                metadata={"count": len(updated_claims), "regenerated": True},
            )

            # Re-save to Lakebase
            try:
                self._save_claims_to_lakebase(skeletons, updated_claims)
            except Exception as e:
                logger.exception("Failed to save regenerated claims to Lakebase (non-fatal)")
                errors.append({"index": -1, "key_differentiator": "lakebase_write", "error": str(e)})

            error_suffix = f" ({len(errors)} had errors — kept originals)" if errors else ""
            self._update_step(5, "waiting_human",
                              progress_message=f"Regenerated {len(updated_claims)} claims.{error_suffix} Review again.",
                              progress_current=total, progress_total=total,
                              error_details={"errors": errors} if errors else None)

        except Exception as e:
            logger.exception("Pass 3 generation failed")
            self._update_step(5, "failed", error_message=f"Pass 3 failed: {e}",
                              error_details={"errors": errors} if errors else None)
