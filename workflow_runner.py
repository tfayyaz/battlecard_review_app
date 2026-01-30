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
        "start_index": {"type": "integer"},
        "end_index": {"type": "integer"},
        "source_index": {"type": "integer"},
        "source_quote": {"type": "string"},
        "verdict": {"type": "string"},
        "confidence": {"type": "number"},
        "verdict_rationale": {"type": "string"},
    },
    "required": [
        "citation_id", "start_index", "end_index",
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
            "databricks_details": {"type": "string"},
            "databricks_reasoning": {"type": "string"},
            "competitor_headline": {"type": "string"},
            "competitor_details": {"type": "string"},
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
            "databricks_details": {"type": "string"},
            "databricks_reasoning": {"type": "string"},
            "competitor_headline": {"type": "string"},
            "competitor_details": {"type": "string"},
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


def load_pass1_template(version: int) -> tuple[str, str]:
    """Load the Pass 1 prompt template for the given version.

    Returns (template_text, file_path) tuple.
    V1 = original file on disk (DEFAULT_PASS1_PROMPT).
    V2+ = loaded at runtime from app.PASS1_PROMPT_TEMPLATES.
    Falls back to V1 file if version not found.
    """
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


def load_pass2_template(version: int) -> str:
    """Load the Pass 2 prompt template for the given version.

    V1 = original file on disk (DEFAULT_PASS2_PROMPT).
    V2+ = loaded at runtime from app.PASS2_PROMPT_TEMPLATES (inline template string).
    Falls back to V1 file if version not found.
    """
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


def format_context_xml(directive: str, old_battlecard: str, competitor: str) -> str:
    """Wrap directive + old battlecard content in XML tags for the prompt."""
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

    def __init__(self, session_id: str, engine):
        self.session_id = session_id
        self.engine = engine
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
            sets = ["status = :status"]
            params = {"sid": self.session_id, "step": step_number, "status": status}
            if status == "in_progress":
                sets.append("started_at = COALESCE(started_at, NOW())")
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

            conn.execute(
                text(f"UPDATE workflow_steps SET {', '.join(sets)} WHERE session_id::text = :sid AND step_number = :step"),
                params,
            )
            conn.execute(
                text("UPDATE workflow_sessions SET current_step = :step, updated_at = NOW() WHERE session_id::text = :sid"),
                {"sid": self.session_id, "step": step_number},
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

    def _build_context(self):
        """Build the XML-tagged context string from directive + old battlecard artifacts."""
        directive = self._get_artifact_content("directive_generated") or ""
        old_battlecard = self._get_artifact_content("old_battlecard_extracted") or ""
        return format_context_xml(directive, old_battlecard, self.competitor)

    # ------------------------------------------------------------------
    # Pass 1: Generate Key Differentiators (Skeletons)
    # ------------------------------------------------------------------

    def run_pass1(self, feedback=None):
        """Run Pass 1 to generate key differentiator skeletons using the LLM."""
        self._update_step(5, "in_progress",
                          progress_message="[1/6] Preparing Pass 1 — loading categories and directive...",
                          progress_current=0, progress_total=6)

        categories = self._get_categories()
        if not categories:
            self._update_step(5, "failed", error_message="No product categories selected. Complete Step 3 first.")
            return

        directive = self._get_artifact_content("directive_generated") or ""
        total_diffs = len(categories) * self.diffs_per_category

        try:
            # Stage 2: Load prompt template
            ver = getattr(self, 'pass1_prompt_template_version', 2)
            self._update_step(5, "in_progress",
                              progress_message=f"[2/6] Loading Pass 1 prompt template V{ver} ({total_diffs} diffs across {len(categories)} categories)...",
                              progress_current=1, progress_total=6)

            template_text, template_file = load_pass1_template(ver)

            # Build context
            context = self._build_context()

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

            # Stage 3: Store the prompt version
            prompt_chars = len(rendered)
            self._update_step(5, "in_progress",
                              progress_message=f"[3/6] Rendered prompt ({prompt_chars:,} chars). Registering prompt version...",
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

            # Record the prompt as an agent turn
            self._record_turn(5, "system_prompt", "system", "pass1_prompt", rendered, model_name=self.model_name)

            # Stage 4: Call the model
            self._update_step(5, "in_progress",
                              progress_message=f"[4/6] Calling {self.model_name} — generating {total_diffs} key differentiators across {len(categories)} categories...",
                              progress_current=3, progress_total=6)

            raw = call_model(
                client=self.client,
                model_name=self.model_name,
                rendered_prompt=rendered,
                json_schema=L200_PASS1_JSON_SCHEMA,
            )

            # Stage 5: Parse and save
            self._update_step(5, "in_progress",
                              progress_message="[5/6] Parsing LLM response and saving artifacts...",
                              progress_current=4, progress_total=6)

            parsed = json.loads(raw)
            skeletons = parsed.get("slides", [])

            logger.info("Pass 1 returned %d skeletons (expected %d)", len(skeletons), total_diffs)

            # Save artifact
            skeletons_content = json.dumps(skeletons, indent=2)
            art_id = self._save_artifact(
                5, "pass1_skeletons", "key_differentiators.json",
                skeletons_content,
                metadata={
                    "count": len(skeletons),
                    "categories": categories,
                    "model": self.model_name,
                    "prompt_version_id": pass1_version_id,
                },
            )

            # Record the output as an agent turn
            self._record_turn(5, "model_output", "assistant", "pass1_skeletons", skeletons_content,
                              model_name=self.model_name, artifact_id=art_id)

            # Stage 6: Save to Lakebase tables
            self._update_step(5, "in_progress",
                              progress_message=f"[6/6] Writing {len(skeletons)} key differentiators to database...",
                              progress_current=5, progress_total=6)

            try:
                self._save_skeletons_to_lakebase(skeletons)
            except Exception as e:
                logger.exception("Failed to save skeletons to Lakebase (non-fatal)")
                # Continue — artifact was saved, Lakebase write is bonus

            self._update_step(5, "waiting_human",
                              progress_message=f"Generated {len(skeletons)} key differentiators across {len(categories)} categories. Review and approve or provide feedback.",
                              progress_current=6, progress_total=6)

        except json.JSONDecodeError as e:
            logger.exception("Pass 1 failed: invalid JSON from LLM")
            self._update_step(5, "failed",
                              error_message=f"LLM returned invalid JSON: {e}",
                              error_details={"stage": "parse_response", "raw_preview": raw[:500] if raw else ""})
        except Exception as e:
            logger.exception("Pass 1 generation failed")
            self._update_step(5, "failed", error_message=f"Pass 1 failed: {e}")

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

    def run_pass2(self):
        """Run Pass 2 to generate detailed claims for each key differentiator."""
        self._update_step(6, "in_progress",
                          progress_message="[1/5] Loading skeletons from Pass 1...",
                          progress_current=0, progress_total=5)

        skeletons_json = self._get_artifact_content("pass1_skeletons")
        if not skeletons_json:
            self._update_step(6, "failed", error_message="No skeletons found. Complete Step 5 first.")
            return

        skeletons = json.loads(skeletons_json)
        total = len(skeletons)
        directive = self._get_artifact_content("directive_generated") or ""
        context = self._build_context()

        try:
            # Stage 2: Load template (version-aware)
            ver = getattr(self, 'pass2_prompt_template_version', 2)
            self._update_step(6, "in_progress",
                              progress_message=f"[2/5] Loading Pass 2 prompt template V{ver} ({total} diffs to process)...",
                              progress_current=1, progress_total=5)
            template_text = load_pass2_template(ver)
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
                self._record_turn(6, "system_prompt", "system", "pass2_prompt",
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
            self._update_step(6, "in_progress",
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
                            logger.error("Pass 2 failed for skeleton %d (%s): %s", idx, kd_name, e)
                            results_by_idx[idx] = self._stub_pass2_claim(sk)
                            errors.append({"index": idx, "key_differentiator": kd_name, "error": str(e)})

                        completed += 1
                        error_suffix = f" ({len(errors)} errors)" if errors else ""
                        self._update_step(6, "in_progress",
                                          progress_current=completed, progress_total=total,
                                          progress_message=f"[3/5] Generating claims — {completed}/{total} done{error_suffix}: {kd_name}")

                # Reassemble in original order
                all_claims = [results_by_idx[i] for i in range(total)]
            else:
                # Sequential processing
                for idx, sk in enumerate(skeletons):
                    kd_name = sk.get("key_differentiator", "")[:50]
                    self._update_step(6, "in_progress",
                                      progress_current=idx, progress_total=total,
                                      progress_message=f"[3/5] Generating claim {idx + 1}/{total}: {kd_name}")

                    try:
                        claim = _process_single_diff(idx, sk)
                    except Exception as e:
                        logger.error("Pass 2 failed for skeleton %d (%s): %s", idx, kd_name, e)
                        claim = self._stub_pass2_claim(sk)
                        errors.append({"index": idx, "key_differentiator": kd_name, "error": str(e)})
                    all_claims.append(claim)

            # Stage 4: Save artifact
            self._update_step(6, "in_progress",
                              progress_message=f"[4/5] Saving {len(all_claims)} claims to artifacts...",
                              progress_current=total, progress_total=total)

            claims_content = json.dumps(all_claims, indent=2)
            art_id = self._save_artifact(
                6, "pass2_claims", "claims.json",
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
            self._record_turn(6, "model_output", "assistant", "pass2_claims", claims_content,
                              model_name=self.model_name, artifact_id=art_id)

            # Stage 5: Save to Lakebase tables
            self._update_step(6, "in_progress",
                              progress_message=f"[5/5] Writing {len(all_claims)} claims to database...",
                              progress_current=total, progress_total=total)

            try:
                self._save_claims_to_lakebase(skeletons, all_claims)
            except Exception as e:
                logger.exception("Failed to save claims to Lakebase (non-fatal)")
                errors.append({"index": -1, "key_differentiator": "lakebase_write", "error": str(e)})

            error_suffix = f" ({len(errors)} had errors — used fallback stubs)" if errors else ""
            self._update_step(6, "waiting_human",
                              progress_message=f"Generated {len(all_claims)} claim pairs.{error_suffix} Review and approve or provide feedback.",
                              progress_current=total, progress_total=total,
                              error_details={"errors": errors} if errors else None)

        except Exception as e:
            logger.exception("Pass 2 generation failed")
            self._update_step(6, "failed", error_message=f"Pass 2 failed: {e}",
                              error_details={"errors": errors} if errors else None)

    def _stub_pass2_claim(self, skeleton):
        """Create a stub claim for a skeleton differentiator (fallback on error)."""
        kd_name = skeleton.get("key_differentiator", "Unknown")
        return {
            "databricks_headline": f"Databricks: {kd_name}",
            "databricks_details": f"Databricks provides strong capabilities in {kd_name}.",
            "databricks_reasoning": "Generation failed — stub reasoning.",
            "competitor_headline": f"{self.competitor}: {kd_name}",
            "competitor_details": f"{self.competitor} has partial support for {kd_name}.",
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
                        "desc": claim.get("databricks_details", ""),
                    },
                ).scalar()
                total_claims += 1

                # Competitor claim
                comp_rating = _rating_to_db(sk.get("competitor_rating", ""))
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
                        "desc": claim.get("competitor_details", ""),
                    },
                ).scalar()
                total_claims += 1

                # Save evidence + fact checks from citations
                self._save_evidence_and_fact_checks(
                    conn, gen_id, db_claim_id, claim, "databricks"
                )
                self._save_evidence_and_fact_checks(
                    conn, gen_id, comp_claim_id, claim, "competitor"
                )

            conn.execute(
                text("UPDATE battlecard_generations SET total_claims = :tc WHERE generation_id = :gid"),
                {"tc": total_claims, "gid": gen_id},
            )

    def _save_evidence_and_fact_checks(self, conn, gen_id, claim_id, claim_data, side):
        """Save evidence rows and fact checks from the citations in the claim."""
        citations = claim_data.get("citations", {})
        sources_list = claim_data.get("sources", [])

        # Build source_index -> source mapping
        source_map = {}
        for src in sources_list:
            source_map[src.get("index")] = src

        for field_suffix in ("details", "reasoning"):
            field_key = f"{side}_{field_suffix}"
            field_citations = citations.get(field_key, [])
            traces_to_field = "description" if field_suffix == "details" else "headline"

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

                # Insert evidence
                traces_text = cite.get("source_quote", "")[:500]
                evidence_id = conn.execute(
                    text(
                        "INSERT INTO evidence (claim_id, traces_to_field, traces_to_start_index, traces_to_end_index, "
                        "traces_to_text, generation_source_id, generation_source_text) "
                        "VALUES (:claim, :field, :start, :end, :trace, :src, :src_text) RETURNING evidence_id"
                    ),
                    {
                        "claim": claim_id,
                        "field": traces_to_field,
                        "start": cite.get("start_index", 0),
                        "end": cite.get("end_index", 0),
                        "trace": traces_text,
                        "src": source_id,
                        "src_text": traces_text,
                    },
                ).scalar()

                # Insert fact check from verdict
                verdict = cite.get("verdict", "unverified")
                if verdict not in ("verified", "unverified", "disputed", "outdated"):
                    verdict = "unverified"
                confidence = cite.get("confidence", 0)
                if isinstance(confidence, float) and confidence <= 1.0:
                    confidence = int(confidence * 100)
                else:
                    confidence = int(confidence) if confidence else 0

                conn.execute(
                    text(
                        "INSERT INTO fact_checks (evidence_id, status, fact_check_source_id, fact_check_source_text, reasoning, "
                        "checked_by, check_method, confidence_score) "
                        "VALUES (:ev, :status, :src, :text, :reason, 'workflow_runner', 'llm_assisted', :conf)"
                    ),
                    {
                        "ev": evidence_id,
                        "status": verdict,
                        "src": source_id,
                        "text": traces_text,
                        "reason": cite.get("verdict_rationale", ""),
                        "conf": confidence,
                    },
                )

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

    def run_pass3(self, feedback=""):
        """Run Pass 3 to regenerate claims incorporating feedback."""
        self._update_step(6, "in_progress",
                          progress_message="[1/4] Loading existing claims and skeletons for regeneration...",
                          progress_current=0, progress_total=4)

        claims_json = self._get_artifact_content("pass2_claims")
        skeletons_json = self._get_artifact_content("pass1_skeletons")

        if not claims_json or not skeletons_json:
            self._update_step(6, "failed", error_message="No existing claims found. Complete Step 6 generation first.")
            return

        skeletons = json.loads(skeletons_json)
        claims = json.loads(claims_json)
        total = len(skeletons)
        directive = self._get_artifact_content("directive_generated") or ""
        context = self._build_context()
        errors = []

        try:
            # Stage 2: Load template
            ver = getattr(self, 'pass2_prompt_template_version', 2)
            self._update_step(6, "in_progress",
                              progress_message=f"[2/4] Loading Pass 2 prompt template V{ver} for regeneration ({total} claims)...",
                              progress_current=1, progress_total=4)
            template_text = load_pass2_template(ver)

            # Stage 3: Regenerate claims sequentially
            updated_claims = []
            for idx, (sk, claim) in enumerate(zip(skeletons, claims)):
                kd_name = sk.get('key_differentiator', '')[:50]
                self._update_step(6, "in_progress",
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
                    logger.error("Pass 3 regen failed for %d (%s): %s", idx, kd_name, e)
                    updated = claim  # Keep original on failure
                    errors.append({"index": idx, "key_differentiator": kd_name, "error": str(e)})

                updated_claims.append(updated)

            # Stage 4: Save artifacts and to Lakebase
            error_suffix = f" ({len(errors)} errors)" if errors else ""
            self._update_step(6, "in_progress",
                              progress_message=f"[4/4] Saving {len(updated_claims)} regenerated claims{error_suffix}...",
                              progress_current=total, progress_total=total)

            regen_content = json.dumps(updated_claims, indent=2)
            art_id = self._save_artifact(
                6, "pass3_regenerated", "claims_regenerated.json",
                regen_content,
                metadata={"count": len(updated_claims), "feedback": feedback[:200], "errors": errors},
            )

            # Record the regenerated output as an agent turn
            self._record_turn(6, "model_output", "assistant", "pass3_regenerated", regen_content,
                              model_name=self.model_name, artifact_id=art_id)

            # Update pass2_claims artifact so it becomes the "current" version
            self._save_artifact(
                6, "pass2_claims", "claims.json",
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
            self._update_step(6, "waiting_human",
                              progress_message=f"Regenerated {len(updated_claims)} claims.{error_suffix} Review again.",
                              progress_current=total, progress_total=total,
                              error_details={"errors": errors} if errors else None)

        except Exception as e:
            logger.exception("Pass 3 generation failed")
            self._update_step(6, "failed", error_message=f"Pass 3 failed: {e}",
                              error_details={"errors": errors} if errors else None)
