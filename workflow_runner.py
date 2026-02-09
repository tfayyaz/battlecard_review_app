"""
Workflow Runner - Orchestrates battlecard generation passes.

Self-contained module that renders prompt templates, calls the Databricks Model
Serving API via the OpenAI client, and writes results back to Lakebase tables.
"""

import hashlib
import json
import logging
import os
import random
import threading
import time
import traceback
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openai import OpenAI
from sqlalchemy import text

try:
    import mlflow
except ImportError:
    mlflow = None

try:
    import tiktoken
    _TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _TIKTOKEN_ENCODING = None

logger = logging.getLogger(__name__)

# Use deterministic defaults unless explicitly overridden.
DEFAULT_MODEL_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
DEFAULT_MODEL_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "16384"))
DEFAULT_MODEL_TIMEOUT_SECONDS = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "180"))
DEFAULT_MODEL_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "6"))
DEFAULT_MODEL_RETRY_BASE_SECONDS = float(os.getenv("LLM_RETRY_BASE_SECONDS", "0.75"))
DEFAULT_MODEL_RETRY_MAX_SECONDS = float(os.getenv("LLM_RETRY_MAX_SECONDS", "12.0"))

_MLFLOW_AUTOLOG_LOCK = threading.Lock()
_MLFLOW_AUTOLOG_INITIALIZED = False
_MLFLOW_AUTOLOG_DISABLED = False


def _normalize_mlflow_tag_value(value: Any) -> str:
    """Normalize tag values to bounded strings for MLflow."""
    if value is None:
        return ""
    text_val = str(value)
    # Keep values modest to avoid oversized tag payloads.
    return text_val[:500]


def _ensure_mlflow_openai_autolog() -> bool:
    """Configure MLflow OpenAI autologging once per process."""
    global _MLFLOW_AUTOLOG_INITIALIZED, _MLFLOW_AUTOLOG_DISABLED

    if _MLFLOW_AUTOLOG_DISABLED:
        return False
    if _MLFLOW_AUTOLOG_INITIALIZED:
        return True
    if mlflow is None:
        _MLFLOW_AUTOLOG_DISABLED = True
        logger.info("MLflow not installed; OpenAI trace autologging disabled.")
        return False

    with _MLFLOW_AUTOLOG_LOCK:
        if _MLFLOW_AUTOLOG_INITIALIZED:
            return True
        if _MLFLOW_AUTOLOG_DISABLED:
            return False
        try:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
            experiment_path = os.getenv(
                "MLFLOW_EXPERIMENT_PATH",
                "/Shared/battlecards-review-pg-llm-traces",
            )
            mlflow.set_tracking_uri(tracking_uri)
            if experiment_path:
                mlflow.set_experiment(experiment_path)

            mlflow.openai.autolog(
                log_models=False,
                log_traces=True,
                disable_for_unsupported_versions=True,
                silent=True,
            )
            _MLFLOW_AUTOLOG_INITIALIZED = True
            logger.info(
                "Enabled MLflow OpenAI autologging (tracking_uri=%s, experiment=%s)",
                tracking_uri,
                experiment_path,
            )
            return True
        except Exception as e:
            _MLFLOW_AUTOLOG_DISABLED = True
            logger.warning("Failed to initialize MLflow OpenAI autologging: %s", e)
            return False


@contextmanager
def _mlflow_step_run(
    model_name: str,
    rendered_prompt: str,
    trace_context: Optional[Dict[str, Any]] = None,
):
    """Start a short-lived MLflow run for one model call (if tracing is enabled)."""
    if not trace_context or not _ensure_mlflow_openai_autolog():
        yield None
        return

    try:
        step_number = trace_context.get("step_number", "unknown")
        operation = trace_context.get("operation", "model_call")
        run_name = f"workflow-step{step_number}-{operation}"
        run_tags = {
            "battlecards.trace": "true",
            "battlecards.session_id": _normalize_mlflow_tag_value(trace_context.get("session_id")),
            "battlecards.step_number": _normalize_mlflow_tag_value(step_number),
            "battlecards.operation": _normalize_mlflow_tag_value(operation),
            "battlecards.model_name": _normalize_mlflow_tag_value(model_name),
            "battlecards.workflow_phase": _normalize_mlflow_tag_value(trace_context.get("phase", "")),
            "source": "battlecards-review-pg",
        }
        with mlflow.start_run(run_name=run_name, nested=bool(mlflow.active_run())) as run:
            mlflow.set_tags(run_tags)
            mlflow.log_params(
                {
                    "model_name": model_name,
                    "prompt_chars": len(rendered_prompt or ""),
                    "step_number": trace_context.get("step_number", ""),
                    "operation": operation,
                }
            )
            yield run.info.run_id
    except Exception as e:
        logger.warning("MLflow trace run setup failed (continuing without tracing): %s", e)
        yield None

# ---------------------------------------------------------------------------
# Default paths (same as app.py — used for context formatting)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

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
        "verbose_item_index": {"type": "integer"},
        "start_index": {"type": "integer"},
        "end_index": {"type": "integer"},
        "source_index": {"type": "integer"},
        "source_quote": {"type": "string"},
        "verdict": {"type": "string"},
        "confidence": {"type": "number"},
        "verdict_rationale": {"type": "string"},
        "claim_subfield": {"type": "string"},
    },
    "required": [
        "citation_id", "detail_item_index", "start_index", "end_index",
        "source_index", "source_quote", "verdict",
        "confidence", "verdict_rationale",
    ],
    "additionalProperties": False,
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
    "additionalProperties": False,
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
    "additionalProperties": False,
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
                    "additionalProperties": False,
                },
            },
        },
        "required": ["slides"],
        "additionalProperties": False,
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
                "additionalProperties": False,
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
        "additionalProperties": False,
    },
    "strict": True,
}

_V10_BULLET_CITATION_SCHEMA = {
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
    "required": ["citation_id", "start_index", "end_index", "source_index", "source_quote"],
    "additionalProperties": False,
}

_V10_L300_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "citations": {"type": "array", "items": _V10_BULLET_CITATION_SCHEMA},
    },
    "required": ["text", "citations"],
    "additionalProperties": False,
}

_V10_L200_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "citations": {"type": "array", "items": _V10_BULLET_CITATION_SCHEMA},
        "l300": {"type": "array", "items": _V10_L300_ITEM_SCHEMA},
    },
    "required": ["text", "citations", "l300"],
    "additionalProperties": False,
}

_V10_TEXT_WITH_CITATIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "citations": {"type": "array", "items": _V10_BULLET_CITATION_SCHEMA},
    },
    "required": ["text", "citations"],
    "additionalProperties": False,
}

_V10_SIDE_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": _V10_TEXT_WITH_CITATIONS_SCHEMA,
        "l200": {"type": "array", "items": _V10_L200_ITEM_SCHEMA},
        "reasoning": _V10_TEXT_WITH_CITATIONS_SCHEMA,
    },
    "required": ["headline", "l200", "reasoning"],
    "additionalProperties": False,
}

L200_PASS2_V10_JSON_SCHEMA = {
    "name": "l200_diff_detail_v10",
    "schema": {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "databricks": _V10_SIDE_SCHEMA,
                        "competitor": _V10_SIDE_SCHEMA,
                        "sources": {"type": "array", "items": _SOURCE_ITEM_SCHEMA},
                        "research_sources": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["id", "databricks", "competitor", "sources", "research_sources"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["claims"],
        "additionalProperties": False,
    },
    "strict": True,
}

_BATCH_LEGACY_CLAIM_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "key_differentiator": {"type": "string"},
        "databricks_headline": {"type": "string"},
        "databricks_l200": {"type": "array", "items": {"type": "string"}},
        "databricks_l300": {"type": "array", "items": {"type": "string"}},
        "databricks_details": {"type": "array", "items": {"type": "string"}},
        "databricks_reasoning": {"type": "string"},
        "competitor_headline": {"type": "string"},
        "competitor_l200": {"type": "array", "items": {"type": "string"}},
        "competitor_l300": {"type": "array", "items": {"type": "string"}},
        "competitor_details": {"type": "array", "items": {"type": "string"}},
        "competitor_reasoning": {"type": "string"},
        "citations": _CITATIONS_SCHEMA,
        "sources": {"type": "array", "items": _SOURCE_ITEM_SCHEMA},
        "research_sources": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "id",
        "databricks_headline",
        "databricks_l200",
        "databricks_reasoning",
        "competitor_headline",
        "competitor_l200",
        "competitor_reasoning",
        "sources",
    ],
    "additionalProperties": False,
}

L200_PASS2_BATCH_JSON_SCHEMA = {
    "name": "l200_diff_detail_batch",
    "schema": {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": _BATCH_LEGACY_CLAIM_ITEM_SCHEMA,
            }
        },
        "required": ["claims"],
        "additionalProperties": False,
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
        "additionalProperties": False,
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


def _extract_status_code(exc: Exception) -> Optional[int]:
    """Best-effort extraction of HTTP status code from SDK exceptions."""
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int):
            return status_code
    return None


def _is_retryable_model_error(exc: Exception) -> bool:
    """Return True for transient model-serving failures worth retrying."""
    status_code = _extract_status_code(exc)
    if status_code in (429, 500, 502, 503, 504):
        return True

    name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    transient_markers = (
        "ratelimit",
        "rate limit",
        "too many requests",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "connection reset",
        "service unavailable",
    )
    if any(marker in name for marker in ("ratelimit", "timeout", "connection")):
        return True
    if any(marker in message for marker in transient_markers):
        return True
    return False


def _call_databricks_model_with_backoff(
    fn,
    *,
    model_name: str,
    operation_name: str,
    max_retries: int = DEFAULT_MODEL_MAX_RETRIES,
    base_delay_seconds: float = DEFAULT_MODEL_RETRY_BASE_SECONDS,
    max_delay_seconds: float = DEFAULT_MODEL_RETRY_MAX_SECONDS,
):
    """Call Databricks model endpoint with exponential backoff + jitter on 429/transient failures."""
    attempts = max(1, int(max_retries))
    base_delay = max(0.05, float(base_delay_seconds))
    max_delay = max(base_delay, float(max_delay_seconds))

    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            retryable = _is_retryable_model_error(exc)
            if not retryable or attempt >= attempts:
                raise

            backoff_cap = min(max_delay, base_delay * (2 ** (attempt - 1)))
            sleep_seconds = random.uniform(0, backoff_cap)
            status_code = _extract_status_code(exc)
            logger.warning(
                "Model call retry for %s (%s): attempt %d/%d failed%s; backing off %.2fs",
                operation_name,
                model_name,
                attempt,
                attempts,
                f" with status {status_code}" if status_code else "",
                sleep_seconds,
            )
            time.sleep(sleep_seconds)
    if last_exc is not None:
        raise last_exc


def call_model(
    client: OpenAI,
    model_name: str,
    rendered_prompt: str,
    json_schema: Optional[Dict] = None,
    temperature: float = DEFAULT_MODEL_TEMPERATURE,
    max_tokens: int = DEFAULT_MODEL_MAX_TOKENS,
    timeout_seconds: float = DEFAULT_MODEL_TIMEOUT_SECONDS,
    trace_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Call the model via Databricks Model Serving with optional structured output.

    Returns the raw response content as a string. If the SDK returns a parsed
    object (list/dict), it's re-serialized to JSON string for consistent handling.
    """
    kwargs = {
        "model": model_name,
        "messages": [{"role": "user", "content": rendered_prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout_seconds,
    }
    if json_schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": json_schema,
        }
        logger.info("  Using structured output (json_schema: %s)", json_schema["name"])

    with _mlflow_step_run(model_name, rendered_prompt, trace_context=trace_context):
        response = _call_databricks_model_with_backoff(
            lambda: client.chat.completions.create(**kwargs),
            model_name=model_name,
            operation_name="chat.completions.create",
        )
    content = response.choices[0].message.content

    # Handle case where SDK returns already-parsed JSON (list or dict)
    # instead of a raw string - convert back to string for consistent handling
    if isinstance(content, (list, dict)):
        logger.debug("Model response was already parsed, re-serializing to JSON string")
        return json.dumps(content)

    return content


def call_model_with_debug(
    client: OpenAI,
    model_name: str,
    rendered_prompt: str,
    json_schema: Optional[Dict] = None,
    temperature: float = DEFAULT_MODEL_TEMPERATURE,
    max_tokens: int = DEFAULT_MODEL_MAX_TOKENS,
    timeout_seconds: float = DEFAULT_MODEL_TIMEOUT_SECONDS,
    trace_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Call the model and return detailed debug info including request/response.

    Returns a dict with:
      - content: The processed response content (string)
      - api_request: The kwargs sent to the API
      - api_response_raw: The raw response object as dict
      - response_was_parsed: Whether SDK returned pre-parsed JSON
    """
    kwargs = {
        "model": model_name,
        "messages": [{"role": "user", "content": rendered_prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout_seconds,
    }
    if json_schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": json_schema,
        }
        logger.info("  Using structured output (json_schema: %s)", json_schema["name"])

    with _mlflow_step_run(model_name, rendered_prompt, trace_context=trace_context):
        response = _call_databricks_model_with_backoff(
            lambda: client.chat.completions.create(**kwargs),
            model_name=model_name,
            operation_name="chat.completions.create(debug)",
        )

    # Capture raw response as dict for debugging
    try:
        api_response_raw = response.model_dump() if hasattr(response, 'model_dump') else str(response)
    except Exception:
        api_response_raw = str(response)

    content = response.choices[0].message.content
    response_was_parsed = False

    # Handle case where SDK returns already-parsed JSON (list or dict)
    if isinstance(content, (list, dict)):
        logger.debug("Model response was already parsed, re-serializing to JSON string")
        response_was_parsed = True
        content = json.dumps(content)

    return {
        "content": content,
        "api_request": {k: v for k, v in kwargs.items() if k != "messages"},  # Exclude large prompt
        "api_request_full": kwargs,  # Full request including messages
        "api_response_raw": api_response_raw,
        "response_was_parsed": response_was_parsed,
    }


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


def parse_model_json(raw: str):
    """Parse model JSON with light cleanup for markdown fences/wrappers."""
    if raw is None:
        raise ValueError("Empty model response")

    text_resp = raw.strip()
    if not text_resp:
        raise ValueError("Empty model response")

    try:
        return json.loads(text_resp)
    except Exception:
        pass

    if text_resp.startswith("```"):
        lines = text_resp.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text_resp = "\n".join(lines).strip()
        try:
            return json.loads(text_resp)
        except Exception:
            pass

    arr_start = text_resp.find("[")
    arr_end = text_resp.rfind("]")
    obj_start = text_resp.find("{")
    obj_end = text_resp.rfind("}")

    candidates = []
    if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
        candidates.append(text_resp[arr_start : arr_end + 1])
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        candidates.append(text_resp[obj_start : obj_end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue

    raise ValueError("Model response is not valid JSON")


def build_fallback_citations(databricks_details, competitor_details, db_reasoning, comp_reasoning, sources):
    """Build synthetic citations when V5/V6 outputs omit structured citations."""
    src_idx = 1 if sources else 0

    def _mk_details(prefix, details):
        out = []
        if src_idx <= 0:
            return out
        for i, text_item in enumerate(details or []):
            txt = str(text_item or "")
            out.append(
                {
                    "citation_id": f"auto_{prefix}_{i}",
                    "detail_item_index": i,
                    "source_index": src_idx,
                    "start_index": 0,
                    "end_index": min(len(txt), 240),
                    "source_quote": txt[:500],
                    "verdict": "unverified",
                    "confidence": 40,
                    "verdict_rationale": "Auto-linked from model-provided source list.",
                    "claim_subfield": "l200",
                }
            )
        return out

    def _mk_reasoning(prefix, reasoning):
        if src_idx <= 0:
            return []
        txt = str(reasoning or "")
        if not txt:
            return []
        return [
            {
                "citation_id": f"auto_{prefix}_reasoning",
                "source_index": src_idx,
                "start_index": 0,
                "end_index": min(len(txt), 240),
                "source_quote": txt[:500],
                "verdict": "unverified",
                "confidence": 40,
                "verdict_rationale": "Auto-linked from model-provided source list.",
                "claim_subfield": "reasoning",
            }
        ]

    return {
        "databricks_headline": [],
        "databricks_details": _mk_details("db_details", databricks_details),
        "databricks_detail_verbose": [],
        "databricks_reasoning": _mk_reasoning("db", db_reasoning),
        "competitor_headline": [],
        "competitor_details": _mk_details("comp_details", competitor_details),
        "competitor_detail_verbose": [],
        "competitor_reasoning": _mk_reasoning("comp", comp_reasoning),
    }


def load_template_from_db(engine, template_type: str, display_order: int):
    """Load a prompt template from the DB by type and display_order.

    Returns (template_text, template_name) or None if not found.
    Note: does NOT filter by is_active — that flag controls admin UI visibility,
    not whether the workflow runner can use the template.
    """
    if engine is None:
        return None
    try:
        with engine.begin() as conn:
            row = conn.execute(
                text(
                    "SELECT template_text, template_name FROM prompt_templates "
                    "WHERE template_type = :ttype AND display_order = :dorder "
                    "LIMIT 1"
                ),
                {"ttype": template_type, "dorder": display_order},
            ).mappings().first()
        if row and row["template_text"] and not row["template_text"].startswith("[Placeholder"):
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
                        "WHERE template_type = 'directive' AND is_default = TRUE "
                        "LIMIT 1"
                    ),
                ).scalar()
            if row and not row.startswith("[Placeholder"):
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


def resolve_context_source_flags(context_sources=None) -> dict:
    """Normalize context source toggles with runtime defaults.

    Defaults match existing workflow behavior:
      - directive: True
      - old_battlecard: True
      - review_feedback: False
      - fact_checks: False
    """
    if context_sources is None:
        context_sources = {}
    return {
        "directive": bool(context_sources.get("directive", True)),
        "old_battlecard": bool(context_sources.get("old_battlecard", True)),
        "review_feedback": bool(context_sources.get("review_feedback", False)),
        "fact_checks": bool(context_sources.get("fact_checks", False)),
    }


def get_pass1_request_spec(version: int, engine=None) -> Dict[str, Any]:
    """Return request behavior and JSON schema for a Pass 1 template version.

    Defaults to structured output with the legacy Pass 1 schema. If a DB prompt
    template provides request/schema config, use that instead.
    """
    default = {
        "use_structured_output": True,
        "response_format_type": "json_schema",
        "schema": L200_PASS1_JSON_SCHEMA,
        "request_preview": {"response_format": {"type": "json_schema", "json_schema": L200_PASS1_JSON_SCHEMA}},
    }
    if engine is None:
        return default
    try:
        with engine.begin() as conn:
            row = conn.execute(
                text(
                    "SELECT response_schema_json, request_config_json "
                    "FROM prompt_templates "
                    "WHERE template_type = 'pass1' AND display_order = :ver "
                    "LIMIT 1"
                ),
                {"ver": int(version or 0)},
            ).mappings().first()
        if not row:
            return default

        request_cfg = row.get("request_config_json") or {}
        if isinstance(request_cfg, str):
            try:
                request_cfg = json.loads(request_cfg)
            except Exception:
                request_cfg = {}
        response_schema = row.get("response_schema_json")
        if isinstance(response_schema, str):
            try:
                response_schema = json.loads(response_schema)
            except Exception:
                response_schema = None

        structured = bool(request_cfg.get("structured_output", True))
        response_format_type = str(request_cfg.get("response_format_type") or ("json_schema" if structured else "none"))
        if not structured or response_format_type == "none":
            return {
                "use_structured_output": False,
                "response_format_type": "none",
                "schema": None,
                "request_preview": {"response_format": None},
            }

        schema = response_schema if isinstance(response_schema, dict) and response_schema else L200_PASS1_JSON_SCHEMA
        return {
            "use_structured_output": True,
            "response_format_type": "json_schema",
            "schema": schema,
            "request_preview": {"response_format": {"type": "json_schema", "json_schema": schema}},
        }
    except Exception as e:
        logger.warning("Failed to load Pass 1 request spec from DB (v%s): %s", version, e)
        return default


def get_pass2_request_spec(version: int, engine=None) -> Dict[str, Any]:
    """Return request behavior and JSON schema for a Pass 2 template version."""
    return get_pass2_request_spec_with_engine(version, engine=engine)


def _resolve_pass2_schema_alias(schema_name: str) -> Optional[Dict[str, Any]]:
    name = str(schema_name or "").strip()
    if not name:
        return None
    lookup = {
        L200_PASS2_JSON_SCHEMA["name"]: L200_PASS2_JSON_SCHEMA,
        L200_PASS2_V10_JSON_SCHEMA["name"]: L200_PASS2_V10_JSON_SCHEMA,
        L200_PASS2_BATCH_JSON_SCHEMA["name"]: L200_PASS2_BATCH_JSON_SCHEMA,
    }
    return lookup.get(name)


def _default_pass2_request_spec(version: int) -> Dict[str, Any]:
    ver = int(version or 0)
    if ver == 10:
        schema = L200_PASS2_V10_JSON_SCHEMA
    elif ver in (5, 6, 7, 8, 9):
        schema = L200_PASS2_BATCH_JSON_SCHEMA
    else:
        schema = L200_PASS2_JSON_SCHEMA
    return {
        "use_structured_output": True,
        "response_format_type": "json_schema",
        "schema": schema,
        "request_preview": {"response_format": {"type": "json_schema", "json_schema": schema}},
        "request_config": {
            "structured_output": True,
            "response_format_type": "json_schema",
            "supports_category_batch": ver in (5, 6, 7, 8, 9, 10),
            "supports_single_call": ver in (8,),
        },
        "binding_source": "builtin",
    }


def get_pass2_request_spec_with_engine(version: int, engine=None) -> Dict[str, Any]:
    """Return request behavior and JSON schema for a Pass 2 template version.

    DB mapping (prompt_templates.request_config_json/response_schema_json) is authoritative
    when available; builtin defaults are used as fallback.
    """
    ver = int(version or 0)
    default = _default_pass2_request_spec(ver)
    if engine is None:
        return default

    try:
        with engine.begin() as conn:
            row = conn.execute(
                text(
                    "SELECT template_name, response_schema_json, request_config_json "
                    "FROM prompt_templates "
                    "WHERE template_type = 'pass2' AND display_order = :ver "
                    "LIMIT 1"
                ),
                {"ver": ver},
            ).mappings().first()
        if not row:
            return default

        request_cfg = row.get("request_config_json") or {}
        if isinstance(request_cfg, str):
            try:
                request_cfg = json.loads(request_cfg)
            except Exception:
                request_cfg = {}

        stored_schema = row.get("response_schema_json")
        if isinstance(stored_schema, str):
            try:
                stored_schema = json.loads(stored_schema)
            except Exception:
                stored_schema = None

        schema = None
        if isinstance(stored_schema, dict):
            # Full json_schema payload persisted directly.
            if isinstance(stored_schema.get("schema"), dict) and stored_schema.get("name"):
                schema = stored_schema
            else:
                # Legacy config shape: {"schema_name": "...", "strict": true}
                schema_name = stored_schema.get("schema_name")
                if schema_name:
                    schema = _resolve_pass2_schema_alias(schema_name)

        structured = bool(request_cfg.get("structured_output", schema is not None))
        response_format_type = str(
            request_cfg.get("response_format_type") or ("json_schema" if structured else "none")
        ).strip().lower()
        if structured and response_format_type != "json_schema":
            response_format_type = "json_schema"

        # If structured output is expected but DB schema is missing, use builtin fallback
        # and mark the source accordingly. UI verification can enforce stricter behavior.
        binding_source = "db"
        if structured and not schema:
            schema = default.get("schema")
            binding_source = "db_with_builtin_schema_fallback"
        if not structured:
            schema = None

        request_preview = (
            {"response_format": {"type": "json_schema", "json_schema": schema}}
            if structured and schema
            else {"response_format": None}
        )
        return {
            "use_structured_output": structured,
            "response_format_type": response_format_type,
            "schema": schema,
            "request_preview": request_preview,
            "request_config": request_cfg,
            "binding_source": binding_source,
            "template_name": row.get("template_name"),
        }
    except Exception as e:
        logger.warning("Failed to load Pass 2 request spec from DB (v%s): %s", ver, e)
        return default


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
                    "COALESCE(step4_execution_mode, 'auto') AS step4_execution_mode, "
                    "COALESCE(step5_execution_mode, 'auto') AS step5_execution_mode, "
                    "COALESCE(step5_inline_fact_check, FALSE) AS step5_inline_fact_check, "
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
        self.step4_execution_mode = (row.get("step4_execution_mode") or "auto").strip().lower()
        self.step5_execution_mode = (row.get("step5_execution_mode") or "auto").strip().lower()
        self.step5_inline_fact_check = bool(row.get("step5_inline_fact_check"))
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
            elif status in ("ready", "in_progress", "waiting_human", "completed"):
                sets.append("error_message = NULL")
            if "error_details" in kwargs:
                sets.append("error_details = CAST(:ed AS jsonb)")
                params["ed"] = json.dumps(kwargs["error_details"])
            elif status in ("ready", "in_progress", "waiting_human", "completed"):
                sets.append("error_details = NULL")
            # Track last_error and increment error_count
            if "last_error" in kwargs:
                sets.append("last_error = :le")
                params["le"] = kwargs["last_error"]
                sets.append("error_count = COALESCE(error_count, 0) + 1")
            elif status in ("ready", "in_progress", "waiting_human", "completed"):
                sets.append("last_error = NULL")

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

    def _save_pass2_debug_log(self, debug_log: dict):
        """Save a Pass 2 debug log entry to the database (upsert by session_id + skeleton_index)."""
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    text("""
                        INSERT INTO pass2_debug_logs (
                            session_id, skeleton_index, key_differentiator, category,
                            rendered_prompt, api_request_json, api_response_raw,
                            structured_output, response_type, was_list_fixed,
                            lakebase_saved, lakebase_error, error_message, processing_time_ms
                        ) VALUES (
                            CAST(:session_id AS uuid), :skeleton_index, :key_differentiator, :category,
                            :rendered_prompt, :api_request_json, :api_response_raw,
                            :structured_output, :response_type, :was_list_fixed,
                            :lakebase_saved, :lakebase_error, :error_message, :processing_time_ms
                        )
                        ON CONFLICT (session_id, skeleton_index)
                        DO UPDATE SET
                            key_differentiator = EXCLUDED.key_differentiator,
                            category = EXCLUDED.category,
                            rendered_prompt = EXCLUDED.rendered_prompt,
                            api_request_json = EXCLUDED.api_request_json,
                            api_response_raw = EXCLUDED.api_response_raw,
                            structured_output = EXCLUDED.structured_output,
                            response_type = EXCLUDED.response_type,
                            was_list_fixed = EXCLUDED.was_list_fixed,
                            lakebase_saved = EXCLUDED.lakebase_saved,
                            lakebase_error = EXCLUDED.lakebase_error,
                            error_message = EXCLUDED.error_message,
                            processing_time_ms = EXCLUDED.processing_time_ms,
                            created_at = NOW()
                    """),
                    debug_log,
                )
        except Exception as e:
            logger.error("Failed to save Pass 2 debug log for skeleton %d: %s", debug_log.get("skeleton_index"), e)

    def _update_pass2_debug_log_lakebase(self, skeleton_index: int, saved: bool, error: str = None):
        """Update the lakebase_saved status for a Pass 2 debug log entry."""
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    text("""
                        UPDATE pass2_debug_logs
                        SET lakebase_saved = :saved, lakebase_error = :error
                        WHERE session_id = CAST(:session_id AS uuid) AND skeleton_index = :skeleton_index
                    """),
                    {"session_id": self.session_id, "skeleton_index": skeleton_index, "saved": saved, "error": error},
                )
        except Exception as e:
            logger.error("Failed to update Pass 2 debug log lakebase status for skeleton %d: %s", skeleton_index, e)

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

    def _get_or_create_generation_id(self, conn):
        """Return workflow generation_id, creating/linking one if missing."""
        gen_id = conn.execute(
            text("SELECT generation_id FROM workflow_sessions WHERE session_id::text = :sid"),
            {"sid": self.session_id},
        ).scalar()
        if gen_id:
            return int(gen_id)

        gen_id = conn.execute(
            text(
                "INSERT INTO battlecard_generations (trigger_type, generated_by, generation_model, status) "
                "VALUES ('manual_request', 'workflow_runner', :model, 'draft') "
                "RETURNING generation_id"
            ),
            {"model": self.model_name},
        ).scalar()
        conn.execute(
            text("UPDATE workflow_sessions SET generation_id = :gid, updated_at = NOW() WHERE session_id::text = :sid"),
            {"gid": int(gen_id), "sid": self.session_id},
        )
        logger.warning(
            "Session %s had no generation_id; created fallback generation_id=%s",
            self.session_id,
            gen_id,
        )
        return int(gen_id)

    def _hydrate_skeleton_key_diff_ids(self, skeletons):
        """Populate missing key_diff_id values from generation/category/name lookup."""
        missing = [s for s in skeletons if not s.get("key_diff_id")]
        if not missing:
            return skeletons, 0

        hydrated = 0
        with self.engine.begin() as conn:
            gen_id = self._get_or_create_generation_id(conn)
            session_created_at = conn.execute(
                text("SELECT created_at FROM workflow_sessions WHERE session_id::text = :sid"),
                {"sid": self.session_id},
            ).scalar()

            for sk in skeletons:
                if sk.get("key_diff_id"):
                    continue
                kd_name = sk.get("key_differentiator", "")
                category_name = sk.get("category", "")
                if not kd_name or not category_name:
                    continue

                kd_id = conn.execute(
                    text(
                        "SELECT kd.key_diff_id "
                        "FROM key_differentiators kd "
                        "JOIN product_category_catalog pcc ON kd.category_id = pcc.catalog_id "
                        "WHERE kd.generation_id = :gid "
                        "AND kd.key_diff_name = :name "
                        "AND pcc.category_name = :category "
                        "ORDER BY kd.display_order, kd.key_diff_id DESC "
                        "LIMIT 1"
                    ),
                    {"gid": gen_id, "name": kd_name, "category": category_name},
                ).scalar()

                if not kd_id and session_created_at:
                    # Recovery path for older runs where generation_id was not bound in Step 4.
                    kd_id = conn.execute(
                        text(
                            "SELECT kd.key_diff_id "
                            "FROM key_differentiators kd "
                            "JOIN product_category_catalog pcc ON kd.category_id = pcc.catalog_id "
                            "WHERE kd.generation_id IS NULL "
                            "AND kd.key_diff_name = :name "
                            "AND pcc.category_name = :category "
                            "AND kd.created_at >= :session_ts - INTERVAL '2 hours' "
                            "AND kd.created_at <= NOW() "
                            "ORDER BY kd.key_diff_id DESC "
                            "LIMIT 1"
                        ),
                        {"name": kd_name, "category": category_name, "session_ts": session_created_at},
                    ).scalar()

                if kd_id:
                    sk["key_diff_id"] = int(kd_id)
                    hydrated += 1

        return skeletons, hydrated

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

    def _get_categories_with_ids(self):
        """Get selected categories with their catalog_ids from the database.

        Returns a list of dicts: [{"catalog_id": 4, "category_name": "Data Engineering..."}, ...]
        Only returns categories with inclusion_type = 'core_product_category'.
        """
        with self.engine.begin() as conn:
            rows = conn.execute(
                text("""
                    SELECT pcc.catalog_id, pcc.category_name
                    FROM session_category_selections scs
                    JOIN product_category_catalog pcc ON scs.catalog_id = pcc.catalog_id
                    WHERE scs.session_id = CAST(:sid AS uuid)
                      AND scs.inclusion_type = 'core_product_category'
                    ORDER BY scs.display_order
                """),
                {"sid": self.session_id},
            ).mappings().all()
            return [{"catalog_id": r["catalog_id"], "category_name": r["category_name"]} for r in rows]

    def _count_tokens(self, text_content: str) -> int:
        """Return estimated token count using tiktoken (cl100k_base) or char-based fallback."""
        if not text_content:
            return 0
        if _TIKTOKEN_ENCODING:
            return len(_TIKTOKEN_ENCODING.encode(text_content))
        return len(text_content) // 4  # rough char-based estimate

    def _pass2_supports_category_batch(self, version: int) -> bool:
        return int(version or 0) in (5, 6, 7, 8, 9, 10)

    def _pass2_supports_single_call(self, version: int) -> bool:
        return int(version or 0) in (8,)

    def _resolve_step4_execution(self, pass1_version: int, core_count: int) -> dict:
        mode = (getattr(self, "step4_execution_mode", "auto") or "auto").strip().lower()
        workers = max(1, int(getattr(self, "max_workers", 1) or 1))
        core_count = max(0, int(core_count or 0))

        if mode == "single_call":
            return {"runtime_mode": "single_call", "parallel": False, "workers": 1}

        if mode in ("category_parallel", "category_sequential"):
            can_batch = pass1_version >= 3 and core_count > 1
            if not can_batch:
                return {"runtime_mode": "single_call", "parallel": False, "workers": 1}
            if mode == "category_parallel" and workers > 1:
                return {
                    "runtime_mode": "category_parallel",
                    "parallel": True,
                    "workers": min(workers, core_count, 6),
                }
            return {"runtime_mode": "category_sequential", "parallel": False, "workers": 1}

        # auto
        use_parallel = bool(pass1_version >= 3 and workers > 1 and core_count > 1)
        if use_parallel:
            return {
                "runtime_mode": "category_parallel",
                "parallel": True,
                "workers": min(workers, core_count, 6),
            }
        return {"runtime_mode": "single_call", "parallel": False, "workers": 1}

    def _resolve_step5_execution(self, pass2_version: int, core_count: int, total_diffs: int) -> dict:
        mode = (getattr(self, "step5_execution_mode", "auto") or "auto").strip().lower()
        workers = max(1, int(getattr(self, "max_workers", 1) or 1))
        core_count = max(0, int(core_count or 0))
        total_diffs = max(0, int(total_diffs or 0))
        supports_category = self._pass2_supports_category_batch(pass2_version)
        supports_single = self._pass2_supports_single_call(pass2_version)

        def _per_diff(parallel: bool, fallback_reason: Optional[str] = None):
            use_parallel = parallel and workers > 1 and total_diffs > 1
            return {
                "runtime_mode": "per_diff_parallel" if use_parallel else "per_diff_sequential",
                "parallel": use_parallel,
                "workers": min(workers, total_diffs, 8) if use_parallel else 1,
                "fallback_reason": fallback_reason,
            }

        def _category(parallel: bool, fallback_reason: Optional[str] = None):
            use_parallel = parallel and workers > 1 and core_count > 1
            return {
                "runtime_mode": "category_parallel" if use_parallel else "category_sequential",
                "parallel": use_parallel,
                "workers": min(workers, core_count, 4) if use_parallel else 1,
                "fallback_reason": fallback_reason,
            }

        if mode == "single_call":
            if supports_single:
                return {"runtime_mode": "single_call", "parallel": False, "workers": 1, "fallback_reason": None}
            return _per_diff(False, f"Prompt V{pass2_version} does not support single-call batching")
        if mode == "per_diff_parallel":
            return _per_diff(True)
        if mode == "per_diff_sequential":
            return _per_diff(False)
        if mode == "category_parallel":
            if supports_category:
                return _category(True)
            return _per_diff(False, f"Prompt V{pass2_version} does not support category batching")
        if mode == "category_sequential":
            if supports_category:
                return _category(False)
            return _per_diff(False, f"Prompt V{pass2_version} does not support category batching")

        # auto
        if pass2_version in (5, 6, 7, 8, 9, 10):
            return _category(True)
        return _per_diff(True)

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
        flags = resolve_context_source_flags(context_sources)
        include_directive = flags["directive"]
        include_old_bc = flags["old_battlecard"]
        include_reviews = flags["review_feedback"]
        include_fact_checks = flags["fact_checks"]

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

    def _process_single_category(self, category_info, all_core_cats, cross_cats,
                                  template_text, directive, context, pass1_json_schema, feedback=None):
        """Generate Pass 1 skeletons for ONE core product category. Returns list[dict].

        Args:
            category_info: Either a dict with {"catalog_id": int, "category_name": str}
                          or a plain string (for backward compatibility).
            all_core_cats: List of all core category infos (same format as category_info).
            cross_cats: List of cross-platform category names (strings).
            template_text: The prompt template.
            directive: The directive text.
            context: The context XML.
            feedback: Optional feedback for regeneration.
        """
        # Handle both dict and string input for backward compatibility
        if isinstance(category_info, dict):
            catalog_id = category_info["catalog_id"]
            category = category_info["category_name"]
        else:
            catalog_id = None
            category = category_info

        # Extract category names for template rendering
        if all_core_cats and isinstance(all_core_cats[0], dict):
            all_cat_names = [c["category_name"] for c in all_core_cats]
        else:
            all_cat_names = all_core_cats

        other_cats = [c for c in all_cat_names if c != category]
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
            json_schema=pass1_json_schema,
        )
        skeletons = json.loads(raw).get("slides", [])

        # CRITICAL: Force the category name and ID from input - don't rely on LLM output
        # Use catalog_id as the authoritative identifier
        for sk in skeletons:
            sk["category"] = category  # Use the exact category name we passed in
            if catalog_id is not None:
                sk["catalog_id"] = catalog_id  # Store the catalog_id for later use
            # Fix the ID to use the correct category
            old_id = sk.get("id", "")
            key_diff_part = old_id.split("_", 1)[-1] if "_" in old_id else sk.get("key_differentiator", "unknown").replace(" ", "_")
            sk["id"] = f"{category.replace(' ', '_')}_{key_diff_part}"

        return skeletons

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

    def _match_category(self, generated_cat: str, expected_categories: list) -> str | None:
        """
        Match a generated category name to an expected category.
        Handles cases where LLM shortens category names.

        Returns the matched expected category name, or None if no match.
        """
        gen_norm = generated_cat.lower().strip()

        # First try exact match
        for exp_cat in expected_categories:
            if gen_norm == exp_cat.lower().strip():
                return exp_cat

        # Try prefix match: "Data Engineering" matches "Data Engineering (ETL, ...)"
        for exp_cat in expected_categories:
            exp_norm = exp_cat.lower().strip()
            # Generated is a prefix of expected (LLM shortened the name)
            if exp_norm.startswith(gen_norm):
                return exp_cat
            # Or the first significant part matches (before parenthesis or dash)
            exp_base = exp_norm.split('(')[0].split(' - ')[0].strip()
            if gen_norm == exp_base:
                return exp_cat

        # Try substring match for compound names
        for exp_cat in expected_categories:
            exp_norm = exp_cat.lower().strip()
            # Generated contains key words from expected
            gen_words = set(gen_norm.replace('&', 'and').split())
            exp_words = set(exp_norm.replace('&', 'and').split())
            # If generated words are mostly in expected (80%+ overlap)
            if gen_words and len(gen_words & exp_words) / len(gen_words) >= 0.8:
                return exp_cat

        return None

    def _validate_and_filter_skeletons(self, skeletons: list, expected_categories: list, diffs_per_category: int) -> tuple:
        """
        Validate LLM output matches expected categories. Filter out invalid entries.
        Uses fuzzy matching to handle LLM shortening category names.

        Returns:
            (filtered_skeletons, validation_warnings)
        """
        warnings = []

        # Check what categories the LLM actually generated
        generated_categories = set(sk.get("category", "") for sk in skeletons)

        # Build mapping from generated -> expected category
        category_mapping = {}
        unmapped_generated = []
        for gen_cat in generated_categories:
            matched = self._match_category(gen_cat, expected_categories)
            if matched:
                category_mapping[gen_cat] = matched
                logger.debug("Category match: '%s' -> '%s'", gen_cat, matched)
            else:
                unmapped_generated.append(gen_cat)

        # Find expected categories that weren't matched
        matched_expected = set(category_mapping.values())
        missing_expected = [cat for cat in expected_categories if cat not in matched_expected]

        if unmapped_generated:
            warnings.append(f"LLM generated unexpected categories (filtered out): {unmapped_generated}")
            logger.warning("Pass 1 validation: unexpected categories: %s", unmapped_generated)

        if missing_expected:
            warnings.append(f"LLM missed expected categories: {missing_expected}")
            logger.warning("Pass 1 validation: missing categories: %s", missing_expected)

        # Filter and normalize skeletons
        filtered = []
        for sk in skeletons:
            sk_cat = sk.get("category", "")
            if sk_cat in category_mapping:
                # Normalize category name to the full expected name
                sk["category"] = category_mapping[sk_cat]
                filtered.append(sk)
            else:
                logger.debug("Filtering out skeleton with unexpected category '%s': %s",
                            sk_cat, sk.get("key_differentiator"))

        # Validate count per category
        cat_counts = {}
        for sk in filtered:
            cat = sk.get("category", "")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        for cat, count in cat_counts.items():
            if count != diffs_per_category:
                warnings.append(f"Category '{cat}' has {count} diffs (expected {diffs_per_category})")

        total_expected = len(expected_categories) * diffs_per_category
        if len(filtered) != total_expected:
            warnings.append(f"Total diff count mismatch: got {len(filtered)}, expected {total_expected}")

        logger.info("Pass 1 validation: %d/%d skeletons kept, %d categories matched, warnings: %s",
                    len(filtered), len(skeletons), len(cat_counts), warnings if warnings else "none")

        return filtered, warnings

    def run_pass1(self, feedback=None, context_sources=None):
        """Run Pass 1 to generate key differentiator skeletons using the LLM."""
        self._update_step(4, "in_progress",
                          progress_message="[1/6] Preparing Pass 1 — loading categories and directive...",
                          progress_current=0, progress_total=6)

        categories = self._get_categories()
        if not categories:
            self._update_step(4, "failed", error_message="No product categories selected. Complete Step 3 first.")
            return

        ctx_flags = resolve_context_source_flags(context_sources)
        directive = (self._get_artifact_content("directive_generated") or "") if ctx_flags["directive"] else ""

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

            step4_exec = self._resolve_step4_execution(ver, len(core_cats))
            use_parallel_pass1 = step4_exec["runtime_mode"] in ("category_parallel", "category_sequential")
            per_category_parallel = step4_exec["runtime_mode"] == "category_parallel"
            step4_workers = int(step4_exec.get("workers", 1))

            if use_parallel_pass1:
                # Load V4 per-category template for parallel execution
                template_text, template_file = load_pass1_template(4, engine=self.engine)
                prompt_label = f"V4 ({step4_exec['runtime_mode']}, {len(core_cats)} categories)"
            else:
                template_text, template_file = load_pass1_template(ver, engine=self.engine)
                prompt_label = f"V{ver}"

            pass1_req_spec = get_pass1_request_spec((4 if use_parallel_pass1 else ver), engine=self.engine)
            pass1_json_schema = pass1_req_spec.get("schema") if pass1_req_spec.get("use_structured_output") else None

            self._update_step(4, "in_progress",
                              progress_message=f"[2/6] Loading Pass 1 prompt template {prompt_label} ({total_diffs} diffs across {len(core_cats) if ver >= 3 else len(categories)} categories)...",
                              progress_current=1, progress_total=6)

            # Build context
            context = self._build_context(ctx_flags)

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
                workers_label = (
                    f"{step4_workers} parallel workers"
                    if per_category_parallel and step4_workers > 1
                    else "sequential"
                )

                # Fetch categories with their catalog_ids from the database
                # This is the authoritative source of truth for category names and IDs
                core_cats_with_ids = self._get_categories_with_ids()
                if not core_cats_with_ids:
                    # Fallback to string-based categories if DB lookup fails
                    logger.warning("No categories with IDs found in DB, falling back to string-based categories")
                    core_cats_with_ids = [{"catalog_id": None, "category_name": c} for c in core_cats]

                num_cats = len(core_cats_with_ids)
                core_cat_names = [c["category_name"] for c in core_cats_with_ids]

                # Record a representative prompt for the first category
                first_cat_name = core_cats_with_ids[0]["category_name"]
                other_text = "\n".join(f"- {c['category_name']}" for c in core_cats_with_ids[1:])
                cross_text_repr = "\n".join(f"- {c}" for c in cross_cats) if cross_cats else "_(none)_"
                representative_prompt = render_template(
                    template_text,
                    competitor=self.competitor,
                    product_area=self.product_area,
                    comparison=f"Databricks vs {self.competitor}",
                    target_category=first_cat_name,
                    other_core_categories=other_text,
                    cross_platform_capabilities=cross_text_repr,
                    diffs_per_category=str(self.diffs_per_category),
                    directives=directive,
                    context=context,
                )
                self._record_turn(4, "system_prompt", "system", "pass1_prompt", representative_prompt, model_name=self.model_name)

                self._update_step(4, "in_progress",
                                  progress_message=f"[4/6] Generating diffs — 0/{num_cats} categories done ({workers_label}, model: {self.model_name})...",
                                  progress_current=3, progress_total=3 + num_cats)

                # Submit one task per core category, using catalog_id as the key
                futures = {}
                executor_workers = step4_workers if per_category_parallel and step4_workers > 1 else 1
                with ThreadPoolExecutor(max_workers=executor_workers) as executor:
                    for cat_info in core_cats_with_ids:
                        future = executor.submit(
                            self._process_single_category,
                            cat_info, core_cats_with_ids, cross_cats,
                            template_text, directive, context, pass1_json_schema, feedback,
                        )
                        # Use catalog_id as key (or category_name as fallback)
                        key = cat_info["catalog_id"] if cat_info["catalog_id"] is not None else cat_info["category_name"]
                        futures[future] = (key, cat_info)

                    # Collect results, saving incremental artifacts as each completes
                    # Key by catalog_id for consistent ordering
                    results_by_id = {}
                    for future in as_completed(futures):
                        key, cat_info = futures[future]
                        cat_name = cat_info["category_name"]
                        try:
                            cat_skeletons = future.result()
                            results_by_id[key] = cat_skeletons
                        except Exception as e:
                            logger.error("Pass 1 failed for category '%s' (id=%s): %s", cat_name, key, e)
                            results_by_id[key] = self._stub_pass1_category(cat_name)
                            errors.append({"category": cat_name, "catalog_id": cat_info.get("catalog_id"), "error": str(e)})

                        completed += 1
                        error_suffix = f" ({len(errors)} errors)" if errors else ""

                        # Accumulate results in original order for incremental save
                        accumulated = []
                        for ci in core_cats_with_ids:
                            ci_key = ci["catalog_id"] if ci["catalog_id"] is not None else ci["category_name"]
                            if ci_key in results_by_id:
                                accumulated.extend(results_by_id[ci_key])

                        self._update_step(4, "in_progress",
                                          progress_current=3 + completed,
                                          progress_total=3 + num_cats,
                                          progress_message=f"[4/6] Generating diffs — {completed}/{num_cats} categories done{error_suffix}: {cat_name}")

                        # Incremental save: partial artifact so the frontend can poll and display
                        partial_content = json.dumps(accumulated, indent=2)
                        completed_cat_names = [ci["category_name"] for ci in core_cats_with_ids
                                               if (ci["catalog_id"] if ci["catalog_id"] is not None else ci["category_name"]) in results_by_id]
                        self._save_artifact(
                            4, "pass1_skeletons", "key_differentiators.json",
                            partial_content,
                            metadata={
                                "count": len(accumulated),
                                "partial": completed < num_cats,
                                "completed_categories": completed,
                                "total_categories": num_cats,
                                "categories": completed_cat_names,
                                "model": self.model_name,
                                "prompt_version_id": pass1_version_id,
                            },
                        )

                # Reassemble all skeletons in original category order
                all_skeletons_raw = []
                for ci in core_cats_with_ids:
                    ci_key = ci["catalog_id"] if ci["catalog_id"] is not None else ci["category_name"]
                    all_skeletons_raw.extend(results_by_id.get(ci_key, []))

                logger.info("Pass 1 (parallel) returned %d skeletons across %d categories (expected %d)",
                            len(all_skeletons_raw), num_cats, total_diffs)

                # Validate and filter skeletons to match expected categories
                all_skeletons, validation_warnings = self._validate_and_filter_skeletons(
                    all_skeletons_raw, core_cat_names, self.diffs_per_category
                )

                # Stage 5: Save final artifact
                self._update_step(4, "in_progress",
                                  progress_message=f"[5/6] Saving {len(all_skeletons)} key differentiators...",
                                  progress_current=3 + num_cats, progress_total=3 + num_cats + 1)

                skeletons_content = json.dumps(all_skeletons, indent=2)
                art_id = self._save_artifact(
                    4, "pass1_skeletons", "key_differentiators.json",
                    skeletons_content,
                    metadata={
                        "count": len(all_skeletons),
                        "partial": False,
                        "completed_categories": num_cats,
                        "total_categories": num_cats,
                        "categories": core_cat_names,
                        "model": self.model_name,
                        "prompt_version_id": pass1_version_id,
                        "parallel": per_category_parallel,
                        "max_workers": executor_workers,
                        "execution_mode": step4_exec["runtime_mode"],
                        "errors": errors,
                        "validation_warnings": validation_warnings,
                        "raw_count": len(all_skeletons_raw),
                    },
                )

                self._record_turn(4, "model_output", "assistant", "pass1_skeletons", skeletons_content,
                                  model_name=self.model_name, artifact_id=art_id)

                # Stage 6: Save to Lakebase tables
                self._update_step(4, "in_progress",
                                  progress_message=f"[6/6] Writing {len(all_skeletons)} key differentiators to database...",
                                  progress_current=3 + num_cats + 1, progress_total=3 + num_cats + 2)

                try:
                    all_skeletons = self._save_skeletons_to_lakebase(all_skeletons)
                    # Persist key_diff_id bindings for deterministic Pass 2 linking.
                    enriched_content = json.dumps(all_skeletons, indent=2)
                    self._save_artifact(
                        4, "pass1_skeletons", "key_differentiators.json",
                        enriched_content,
                        metadata={
                            "count": len(all_skeletons),
                            "partial": False,
                            "completed_categories": num_cats,
                            "total_categories": num_cats,
                            "categories": core_cat_names,
                            "model": self.model_name,
                            "prompt_version_id": pass1_version_id,
                            "parallel": per_category_parallel,
                            "max_workers": executor_workers,
                            "execution_mode": step4_exec["runtime_mode"],
                            "errors": errors,
                            "validation_warnings": validation_warnings,
                            "raw_count": len(all_skeletons_raw),
                            "has_key_diff_ids": True,
                        },
                    )
                except Exception as e:
                    logger.exception("Failed to save skeletons to Lakebase (non-fatal)")

                # Build final message with any warnings
                error_suffix = f" ({len(errors)} had errors — used fallback stubs)" if errors else ""
                warning_suffix = ""
                if validation_warnings:
                    warning_suffix = f" ⚠️ Validation: {'; '.join(validation_warnings)}"

                self._update_step(4, "waiting_human",
                                  progress_message=f"Generated {len(all_skeletons)} key differentiators across {num_cats} categories ({step4_exec['runtime_mode']}).{error_suffix}{warning_suffix} Review and approve or provide feedback.",
                                  progress_current=6, progress_total=6,
                                  error_details={"errors": errors, "validation_warnings": validation_warnings} if (errors or validation_warnings) else None)
                return

            # ---------- SINGLE-CALL PATH (V1/V2 or non-parallel) ----------
            # Build template variables based on version
            if ver >= 3 and core_cats:
                # V3+: use only core product categories (cross-platform weaved in)
                # The template uses {{product_categories}} so we pass that variable name
                core_text = "\n".join(f"- {c}" for c in core_cats)
                cross_text = "\n".join(f"- {c}" for c in cross_cats) if cross_cats else "_(none selected)_"
                rendered = render_template(
                    template_text,
                    competitor=self.competitor,
                    product_area=self.product_area,
                    comparison=f"Databricks vs {self.competitor}",
                    # Pass both old and new variable names for compatibility
                    product_categories=core_text,  # V3 template uses this
                    core_product_categories=core_text,  # Future templates may use this
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
                json_schema=pass1_json_schema,
            )

            # Stage 5: Parse and save
            self._update_step(4, "in_progress",
                              progress_message="[5/6] Parsing LLM response and saving artifacts...",
                              progress_current=4, progress_total=6)

            parsed = json.loads(raw)
            skeletons_raw = parsed.get("slides", [])

            logger.info("Pass 1 returned %d skeletons (expected %d)", len(skeletons_raw), total_diffs)

            # Validate and filter skeletons to match expected categories
            expected_cats = core_cats if (ver >= 3 and core_cats) else categories
            skeletons, validation_warnings = self._validate_and_filter_skeletons(
                skeletons_raw, expected_cats, self.diffs_per_category
            )

            # Save artifact (filtered skeletons)
            skeletons_content = json.dumps(skeletons, indent=2)
            art_id = self._save_artifact(
                4, "pass1_skeletons", "key_differentiators.json",
                skeletons_content,
                metadata={
                    "count": len(skeletons),
                    "partial": False,
                    "categories": expected_cats,
                    "model": self.model_name,
                    "prompt_version_id": pass1_version_id,
                    "validation_warnings": validation_warnings,
                    "raw_count": len(skeletons_raw),
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
                skeletons = self._save_skeletons_to_lakebase(skeletons)
                # Persist key_diff_id bindings for deterministic Pass 2 linking.
                enriched_content = json.dumps(skeletons, indent=2)
                self._save_artifact(
                    4, "pass1_skeletons", "key_differentiators.json",
                    enriched_content,
                    metadata={
                        "count": len(skeletons),
                        "partial": False,
                        "categories": expected_cats,
                        "model": self.model_name,
                        "prompt_version_id": pass1_version_id,
                        "validation_warnings": validation_warnings,
                        "raw_count": len(skeletons_raw),
                        "has_key_diff_ids": True,
                    },
                )
            except Exception as e:
                logger.exception("Failed to save skeletons to Lakebase (non-fatal)")
                # Continue — artifact was saved, Lakebase write is bonus

            # Build final message with validation warnings if any
            warning_suffix = ""
            if validation_warnings:
                warning_suffix = f" ⚠️ Validation issues: {'; '.join(validation_warnings)}"

            self._update_step(4, "waiting_human",
                              progress_message=f"Generated {len(skeletons)} key differentiators across {len(expected_cats)} categories.{warning_suffix} Review and approve or provide feedback.",
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
        """Save skeleton key differentiators to Lakebase and return enriched skeletons."""
        with self.engine.begin() as conn:
            gen_id = self._get_or_create_generation_id(conn)

            # Regeneration path: replace generation-scoped differentiators instead of
            # accumulating duplicates across repeated Step 4 runs.
            if gen_id:
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
                conn.execute(
                    text("DELETE FROM key_differentiators WHERE generation_id = :gid"),
                    {"gid": gen_id},
                )

            selected_rows = conn.execute(
                text(
                    "SELECT pcc.catalog_id, pcc.category_name "
                    "FROM session_category_selections scs "
                    "JOIN product_category_catalog pcc ON scs.catalog_id = pcc.catalog_id "
                    "WHERE scs.session_id = CAST(:sid AS uuid)"
                ),
                {"sid": self.session_id},
            ).mappings().all()
            selected_by_name = {r["category_name"]: int(r["catalog_id"]) for r in selected_rows}

            catalog_rows = conn.execute(
                text("SELECT catalog_id, category_name FROM product_category_catalog")
            ).mappings().all()
            catalog_by_name = {r["category_name"]: int(r["catalog_id"]) for r in catalog_rows}
            catalog_names = list(catalog_by_name.keys())

            unresolved_categories = []
            for sk in skeletons:
                category_name = (sk.get("category") or "").strip()

                cat_id = sk.get("catalog_id")
                if cat_id is not None:
                    try:
                        cat_id = int(cat_id)
                    except (TypeError, ValueError):
                        cat_id = None

                if cat_id is None and category_name:
                    cat_id = selected_by_name.get(category_name) or catalog_by_name.get(category_name)

                if cat_id is None and category_name:
                    matched = self._match_category(category_name, catalog_names)
                    if matched:
                        category_name = matched
                        sk["category"] = matched
                        cat_id = selected_by_name.get(matched) or catalog_by_name.get(matched)

                if cat_id is None:
                    unresolved_categories.append(category_name or "<blank>")
                    continue

                key_diff_id = conn.execute(
                    text(
                        "INSERT INTO key_differentiators (category_id, key_diff_name, key_diff_description, display_order, generation_id) "
                        "VALUES (:cat, :name, :desc, :order, :gen_id) RETURNING key_diff_id"
                    ),
                    {
                        "cat": int(cat_id),
                        "name": sk.get("key_differentiator", ""),
                        "desc": sk.get("description", ""),
                        "order": sk.get("rank", 0),
                        "gen_id": int(gen_id),
                    },
                ).scalar()
                sk["catalog_id"] = int(cat_id)
                sk["key_diff_id"] = key_diff_id
                sk["generation_id"] = int(gen_id)

            if unresolved_categories:
                logger.warning(
                    "Skipped %d skeletons with unresolved categories: %s",
                    len(unresolved_categories),
                    sorted(set(unresolved_categories)),
                )

        return skeletons

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
        skeletons, hydrated_ids = self._hydrate_skeleton_key_diff_ids(skeletons)
        if hydrated_ids:
            try:
                self._save_artifact(
                    4,
                    "pass1_skeletons",
                    "key_differentiators.json",
                    json.dumps(skeletons, indent=2),
                    metadata={
                        "count": len(skeletons),
                        "partial": False,
                        "hydrated_key_diff_ids": hydrated_ids,
                        "has_key_diff_ids": True,
                    },
                )
            except Exception:
                logger.exception("Failed to persist hydrated key_diff_id bindings for session %s", self.session_id)
        total = len(skeletons)
        ctx_flags = resolve_context_source_flags(context_sources)
        directive = (self._get_artifact_content("directive_generated") or "") if ctx_flags["directive"] else ""
        context = self._build_context(ctx_flags)

        try:
            # Stage 2: Load template (version-aware)
            ver = getattr(self, 'pass2_prompt_template_version', 2)
            self._update_step(5, "in_progress",
                              progress_message=f"[2/5] Loading Pass 2 prompt template V{ver} ({total} diffs to process)...",
                              progress_current=1, progress_total=5)
            template_text = load_pass2_template(ver, engine=self.engine)
            logger.info("Using Pass 2 prompt template version %d (%d chars)", ver, len(template_text))
            pass2_request_spec = get_pass2_request_spec_with_engine(ver, engine=self.engine)
            pass2_json_schema = pass2_request_spec.get("schema")

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

            # Per-batch prompts already receive directive content
            # through the context XML; avoid injecting directives twice.
            directives_for_template = "" if ver in (5, 6, 7, 8, 9, 10) else directive

            # Build category payloads for templates that expect key_diffs_json.
            category_payloads = {}
            for sk in skeletons:
                cat = sk.get("category", "")
                category_payloads.setdefault(cat, []).append(
                    {
                        "id": sk.get("id", ""),
                        "category": sk.get("category", ""),
                        "key_differentiator": sk.get("key_differentiator", ""),
                        "description": sk.get("description", ""),
                        "databricks_rating": sk.get("databricks_rating", ""),
                        "competitor_rating": sk.get("competitor_rating", ""),
                        "selection_reasoning": sk.get("selection_reasoning", ""),
                    }
                )

            def _normalize_sources(sources):
                if not isinstance(sources, list):
                    return []
                normalized = []
                for i, src in enumerate(sources):
                    if isinstance(src, dict):
                        normalized.append(
                            {
                                "index": int(src.get("index", i + 1)),
                                "title": str(src.get("title", f"Source {i + 1}")),
                                "url": str(src.get("url", "")),
                                "type": str(src.get("type", "context")),
                                "accessed_at": str(src.get("accessed_at", datetime.now(timezone.utc).isoformat())),
                            }
                        )
                    else:
                        src_str = str(src)
                        normalized.append(
                            {
                                "index": i + 1,
                                "title": src_str[:120] if src_str else f"Source {i + 1}",
                                "url": src_str if src_str.startswith("http") else "",
                                "type": "context",
                                "accessed_at": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                return normalized

            def _normalize_batched_result(parsed_result, skeleton, *, strict_match: bool = False):
                """Map batched output objects into the standard Pass 2 shape."""
                # Some model SDK paths return wrapper blocks like:
                # {"type":"text","text":"{...json...}"}.
                # Unwrap these before candidate matching.
                for _ in range(3):
                    if isinstance(parsed_result, dict):
                        wrapper_type = str(parsed_result.get("type", "")).strip().lower()
                        wrapper_text = parsed_result.get("text")
                        if isinstance(wrapper_text, str) and (
                            wrapper_type in ("text", "output_text")
                            or set(parsed_result.keys()).issubset({"type", "text"})
                        ):
                            try:
                                parsed_result = parse_model_json(wrapper_text)
                                continue
                            except Exception:
                                break
                    if (
                        isinstance(parsed_result, list)
                        and len(parsed_result) == 1
                        and isinstance(parsed_result[0], dict)
                    ):
                        first = parsed_result[0]
                        wrapper_type = str(first.get("type", "")).strip().lower()
                        wrapper_text = first.get("text")
                        if isinstance(wrapper_text, str) and wrapper_type in ("text", "output_text"):
                            try:
                                parsed_result = parse_model_json(wrapper_text)
                                continue
                            except Exception:
                                break
                    break

                target_id = skeleton.get("id", "")
                target_name = skeleton.get("key_differentiator", "").strip().lower()

                if isinstance(parsed_result, dict):
                    claims_list = parsed_result.get("claims")
                    if isinstance(claims_list, str):
                        try:
                            claims_list = json.loads(claims_list)
                        except Exception:
                            try:
                                claims_list = parse_model_json(claims_list)
                            except Exception:
                                claims_list = None
                    if isinstance(claims_list, list):
                        candidates = [c for c in claims_list if isinstance(c, dict)]
                    else:
                        candidates = [parsed_result]
                elif isinstance(parsed_result, list):
                    candidates = [c for c in parsed_result if isinstance(c, dict)]
                else:
                    candidates = []

                chosen = None
                if target_id:
                    id_matches = [c for c in candidates if c.get("id") == target_id]
                    if len(id_matches) == 1:
                        chosen = id_matches[0]
                    elif len(id_matches) > 1 and strict_match:
                        raise ValueError(f"Batched response had duplicate id matches for {target_id}")
                if chosen is None and target_name:
                    name_matches = [
                        c
                        for c in candidates
                        if str(c.get("key_differentiator", "")).strip().lower() == target_name
                    ]
                    if len(name_matches) == 1:
                        chosen = name_matches[0]
                    elif len(name_matches) > 1 and strict_match:
                        raise ValueError(
                            f"Batched response had ambiguous key_differentiator matches for '{target_name}'"
                        )
                if chosen is None and candidates and not strict_match:
                    chosen = candidates[0]
                if chosen is None:
                    raise ValueError(
                        f"Batched response did not contain a usable object for id='{target_id}' "
                        f"name='{target_name}' (candidates={len(candidates)})"
                    )

                def _to_text_list(value):
                    if value is None:
                        return []
                    if isinstance(value, str):
                        return [value] if value else []
                    if isinstance(value, list):
                        out = []
                        for item in value:
                            if isinstance(item, str):
                                if item:
                                    out.append(item)
                            elif isinstance(item, dict):
                                txt = str(item.get("text", "") or "").strip()
                                if txt:
                                    out.append(txt)
                        return out
                    return [str(value)]

                def _normalize_inline_citations(raw_citations, *, detail_item_index=None, verbose_item_index=None, claim_subfield="l200"):
                    out = []
                    if not isinstance(raw_citations, list):
                        return out
                    for i, cite in enumerate(raw_citations):
                        if not isinstance(cite, dict):
                            continue
                        cid = str(cite.get("citation_id") or f"cite_{claim_subfield}_{detail_item_index or 0}_{verbose_item_index or 0}_{i}")
                        try:
                            conf = float(cite.get("confidence", 0.0) or 0.0)
                        except Exception:
                            conf = 0.0
                        entry = {
                            "citation_id": cid,
                            "detail_item_index": int(detail_item_index) if detail_item_index is not None else 0,
                            "start_index": int(cite.get("start_index", 0) or 0),
                            "end_index": int(cite.get("end_index", 0) or 0),
                            "source_index": int(cite.get("source_index", 1) or 1),
                            "source_quote": str(cite.get("source_quote", "") or ""),
                            "verdict": str(cite.get("verdict", "pending") or "pending"),
                            "confidence": conf,
                            "verdict_rationale": str(cite.get("verdict_rationale", "") or ""),
                            "claim_subfield": claim_subfield,
                        }
                        if verbose_item_index is not None:
                            entry["verbose_item_index"] = int(verbose_item_index)
                        out.append(entry)
                    return out

                def _extract_side(chosen_obj, side_key, top_prefix):
                    side = chosen_obj.get(side_key, {}) if isinstance(chosen_obj.get(side_key), dict) else {}
                    l200_items = []
                    l300_map = []
                    citations_map = {
                        f"{top_prefix}_headline": [],
                        f"{top_prefix}_details": [],
                        f"{top_prefix}_detail_verbose": [],
                        f"{top_prefix}_reasoning": [],
                    }

                    headline_raw = side.get("headline", chosen_obj.get(f"{top_prefix}_headline", ""))
                    if isinstance(headline_raw, dict):
                        headline_text = str(headline_raw.get("text", "") or "")
                        citations_map[f"{top_prefix}_headline"] = _normalize_inline_citations(
                            headline_raw.get("citations", []),
                            claim_subfield="headline",
                        )
                    else:
                        headline_text = str(headline_raw or "")

                    reasoning_raw = side.get("reasoning", chosen_obj.get(f"{top_prefix}_reasoning", ""))
                    if isinstance(reasoning_raw, dict):
                        reasoning_text = str(reasoning_raw.get("text", "") or "")
                        citations_map[f"{top_prefix}_reasoning"] = _normalize_inline_citations(
                            reasoning_raw.get("citations", []),
                            claim_subfield="reasoning",
                        )
                    else:
                        reasoning_text = str(reasoning_raw or "")

                    side_l200 = side.get("l200")
                    if isinstance(side_l200, list) and side_l200 and isinstance(side_l200[0], dict):
                        for detail_idx, l200_item in enumerate(side_l200):
                            bullet_text = str(l200_item.get("text", "") or "").strip()
                            if bullet_text:
                                l200_items.append(bullet_text)
                            else:
                                l200_items.append(f"{side_key.title()} bullet {detail_idx + 1}")
                            citations_map[f"{top_prefix}_details"].extend(
                                _normalize_inline_citations(
                                    l200_item.get("citations", []),
                                    detail_item_index=detail_idx,
                                    claim_subfield="l200",
                                )
                            )
                            l300_items = []
                            raw_l300 = l200_item.get("l300", [])
                            if isinstance(raw_l300, list):
                                for verbose_idx, verbose_item in enumerate(raw_l300):
                                    if isinstance(verbose_item, dict):
                                        verbose_text = str(verbose_item.get("text", "") or "").strip()
                                        verbose_cites = verbose_item.get("citations", [])
                                    else:
                                        verbose_text = str(verbose_item or "").strip()
                                        verbose_cites = []
                                    if not verbose_text:
                                        continue
                                    l300_items.append(verbose_text)
                                    citations_map[f"{top_prefix}_detail_verbose"].extend(
                                        _normalize_inline_citations(
                                            verbose_cites,
                                            detail_item_index=detail_idx,
                                            verbose_item_index=verbose_idx,
                                            claim_subfield="l300",
                                        )
                                    )
                            l300_map.append(l300_items)
                    else:
                        l200_items = _to_text_list(chosen_obj.get(f"{top_prefix}_l200", chosen_obj.get(f"{top_prefix}_details", [])))
                        raw_l300 = chosen_obj.get(f"{top_prefix}_l300", [])
                        if isinstance(raw_l300, list) and raw_l300 and isinstance(raw_l300[0], list):
                            l300_map = [list(map(str, row)) for row in raw_l300]
                        else:
                            as_list = _to_text_list(raw_l300)
                            if l200_items:
                                if len(as_list) == len(l200_items):
                                    l300_map = [[v] if v else [] for v in as_list]
                                else:
                                    l300_map = [as_list] + [[] for _ in range(max(len(l200_items) - 1, 0))]
                            else:
                                l300_map = [as_list] if as_list else []

                    if len(l300_map) < len(l200_items):
                        l300_map.extend([[] for _ in range(len(l200_items) - len(l300_map))])
                    elif len(l300_map) > len(l200_items):
                        l300_map = l300_map[:len(l200_items)]

                    # Enforce at least one citation per L200/L300 bullet for deterministic evidence linkage.
                    existing_l200 = citations_map[f"{top_prefix}_details"]
                    existing_l300 = citations_map[f"{top_prefix}_detail_verbose"]
                    for detail_idx, bullet_text in enumerate(l200_items):
                        if not str(bullet_text or "").strip():
                            continue
                        has_detail_cite = any(int(c.get("detail_item_index", -1)) == detail_idx for c in existing_l200)
                        if not has_detail_cite:
                            txt = str(bullet_text)
                            existing_l200.append(
                                {
                                    "citation_id": f"auto_{top_prefix}_l200_{detail_idx}",
                                    "detail_item_index": detail_idx,
                                    "start_index": 0,
                                    "end_index": min(len(txt), 120),
                                    "source_index": 1,
                                    "source_quote": txt[:500],
                                    "verdict": "unverified",
                                    "confidence": 40.0,
                                    "verdict_rationale": "Auto-citation added because model omitted L200 citations.",
                                    "claim_subfield": "l200",
                                }
                            )
                        for verbose_idx, verbose_text in enumerate(l300_map[detail_idx] if detail_idx < len(l300_map) else []):
                            if not str(verbose_text or "").strip():
                                continue
                            has_verbose_cite = any(
                                int(c.get("detail_item_index", -1)) == detail_idx
                                and int(c.get("verbose_item_index", -1)) == verbose_idx
                                for c in existing_l300
                            )
                            if not has_verbose_cite:
                                vtxt = str(verbose_text)
                                existing_l300.append(
                                    {
                                        "citation_id": f"auto_{top_prefix}_l300_{detail_idx}_{verbose_idx}",
                                        "detail_item_index": detail_idx,
                                        "verbose_item_index": verbose_idx,
                                        "start_index": 0,
                                        "end_index": min(len(vtxt), 120),
                                        "source_index": 1,
                                        "source_quote": vtxt[:500],
                                        "verdict": "unverified",
                                        "confidence": 40.0,
                                        "verdict_rationale": "Auto-citation added because model omitted L300 citations.",
                                        "claim_subfield": "l300",
                                    }
                                )

                    return headline_text, l200_items, l300_map, reasoning_text, citations_map

                db_head, db_details, db_l300_map, db_reasoning, db_citations = _extract_side(chosen, "databricks", "databricks")
                comp_head, comp_details, comp_l300_map, comp_reasoning, comp_citations = _extract_side(chosen, "competitor", "competitor")

                sources = _normalize_sources(chosen.get("sources", []))
                if not sources:
                    sources = [
                        {
                            "index": 1,
                            "title": "Model output (no explicit source URL)",
                            "url": "",
                            "type": "context",
                            "accessed_at": datetime.now(timezone.utc).isoformat(),
                        }
                    ]

                citations = chosen.get("citations")
                if not isinstance(citations, dict):
                    citations = build_fallback_citations(
                        db_details,
                        comp_details,
                        db_reasoning,
                        comp_reasoning,
                        sources,
                    )
                else:
                    for key in (
                        "databricks_headline",
                        "databricks_details",
                        "databricks_detail_verbose",
                        "databricks_reasoning",
                        "competitor_headline",
                        "competitor_details",
                        "competitor_detail_verbose",
                        "competitor_reasoning",
                    ):
                        if key not in citations or not isinstance(citations.get(key), list):
                            citations[key] = []
                        else:
                            citations[key] = [c for c in citations.get(key, []) if isinstance(c, dict)]

                    if not any(len(v) for v in citations.values()):
                        citations = build_fallback_citations(
                            db_details,
                            comp_details,
                            db_reasoning,
                            comp_reasoning,
                            sources,
                        )

                for k, v in db_citations.items():
                    citations.setdefault(k, [])
                    citations[k].extend(v)
                for k, v in comp_citations.items():
                    citations.setdefault(k, [])
                    citations[k].extend(v)

                return {
                    "databricks_headline": db_head or chosen.get("databricks_headline", ""),
                    "databricks_details": db_details,
                    "databricks_details_verbose": db_l300_map,
                    "databricks_reasoning": db_reasoning or chosen.get("databricks_reasoning", ""),
                    "competitor_headline": comp_head or chosen.get("competitor_headline", ""),
                    "competitor_details": comp_details,
                    "competitor_details_verbose": comp_l300_map,
                    "competitor_reasoning": comp_reasoning or chosen.get("competitor_reasoning", ""),
                    "citations": citations,
                    "sources": sources,
                    "research_sources": chosen.get("research_sources", []),
                }

            all_claims = [None] * total
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
                    num_diffs=len(category_payloads.get(first_sk.get("category", ""), [])),
                    key_diffs_json=json.dumps(category_payloads.get(first_sk.get("category", ""), []), indent=2),
                    directives=directives_for_template,
                    context=context,
                )
                self._record_turn(5, "system_prompt", "system", "pass2_prompt",
                                  representative_prompt, model_name=self.model_name)

            def _process_single_diff(idx: int, sk: dict) -> dict:
                """Generate claims for a single skeleton diff."""
                import time as _time
                start_time = _time.time()

                kd_name = sk.get("key_differentiator", "")
                category = sk.get("category", "")
                logger.info("Pass 2 processing skeleton %d: %s", idx, kd_name)

                rendered = render_template(
                    template_text,
                    competitor=self.competitor,
                    category=category,
                    key_differentiator=kd_name,
                    description=sk.get("description", ""),
                    databricks_rating=sk.get("databricks_rating", ""),
                    competitor_rating=sk.get("competitor_rating", ""),
                    selection_reasoning=sk.get("selection_reasoning", ""),
                    num_diffs=len(category_payloads.get(category, [])),
                    key_diffs_json=json.dumps(category_payloads.get(category, []), indent=2),
                    directives=directives_for_template,
                    context=context,
                )

                # Log a snippet of the rendered prompt to verify key_differentiator is included
                logger.info("Pass 2 skeleton %d prompt snippet: ...%s...", idx, rendered[200:400])

                # Initialize debug log data
                debug_log = {
                    "session_id": self.session_id,
                    "skeleton_index": idx,
                    "key_differentiator": kd_name[:500] if kd_name else None,
                    "category": category[:255] if category else None,
                    "rendered_prompt": rendered,
                    "api_request_json": None,
                    "api_response_raw": None,
                    "structured_output": None,
                    "response_type": "unknown",
                    "was_list_fixed": False,
                    "lakebase_saved": False,
                    "lakebase_error": None,
                    "error_message": None,
                    "processing_time_ms": None,
                }

                try:
                    use_structured_output = bool(pass2_request_spec.get("use_structured_output"))
                    # Use debug version to capture request/response
                    debug_result = call_model_with_debug(
                        client=self.client,
                        model_name=self.model_name,
                        rendered_prompt=rendered,
                        json_schema=pass2_json_schema if use_structured_output else None,
                    )

                    raw = debug_result["content"]
                    debug_log["api_request_json"] = json.dumps(debug_result["api_request"], indent=2)
                    debug_log["api_response_raw"] = json.dumps(debug_result["api_response_raw"], indent=2) if isinstance(debug_result["api_response_raw"], dict) else str(debug_result["api_response_raw"])

                    result = parse_model_json(raw)

                    if ver in (5, 6, 7, 8, 9, 10):
                        debug_log["response_type"] = type(result).__name__ if type(result).__name__ in ("dict", "list") else "unknown"
                        debug_log["was_list_fixed"] = False
                        debug_log["structured_output"] = json.dumps(result, indent=2)
                        result = _normalize_batched_result(result, sk, strict_match=True)
                    else:
                        # Determine response type and handle list case (potentially nested)
                        extraction_depth = 0
                        while isinstance(result, list) and extraction_depth < 5:
                            extraction_depth += 1
                            logger.warning("Pass 2 skeleton %d: model returned list (depth %d), extracting first element", idx, extraction_depth)
                            if result and len(result) > 0:
                                result = result[0]
                            else:
                                raise ValueError(f"Model returned empty list at depth {extraction_depth}")

                        if extraction_depth > 0:
                            debug_log["response_type"] = "list"
                            debug_log["was_list_fixed"] = isinstance(result, dict)
                            if not isinstance(result, dict):
                                raise ValueError(f"After extracting {extraction_depth} levels, result is {type(result).__name__}, not dict")
                        elif isinstance(result, dict):
                            debug_log["response_type"] = "dict"
                        else:
                            debug_log["response_type"] = "unknown"
                            raise ValueError(f"Model returned {type(result).__name__}, expected dict")

                        debug_log["structured_output"] = json.dumps(result, indent=2)

                    logger.info("Pass 2 skeleton %d result headline: %s", idx, result.get("databricks_headline", "N/A")[:50])

                    # Calculate processing time
                    debug_log["processing_time_ms"] = int((_time.time() - start_time) * 1000)

                    # Save debug log to database
                    self._save_pass2_debug_log(debug_log)

                    return result

                except Exception as e:
                    debug_log["response_type"] = "error"
                    debug_log["error_message"] = str(e)[:2000]
                    debug_log["processing_time_ms"] = int((_time.time() - start_time) * 1000)
                    self._save_pass2_debug_log(debug_log)
                    raise

            # Stage 3: Generate claims
            step5_exec = self._resolve_step5_execution(ver, len(category_payloads), total)
            runtime_mode = step5_exec["runtime_mode"]
            if step5_exec.get("fallback_reason"):
                logger.warning("Step 5 execution fallback: %s", step5_exec["fallback_reason"])
                errors.append({"index": -1, "key_differentiator": "execution_mode", "error": step5_exec["fallback_reason"]})

            if runtime_mode in ("category_parallel", "category_sequential"):
                category_to_indexes = {}
                for idx, sk in enumerate(skeletons):
                    category = sk.get("category", "")
                    category_to_indexes.setdefault(category, []).append(idx)
                category_groups = list(category_to_indexes.items())
                total_groups = len(category_groups)
                workers_to_use = max(1, min(step5_exec.get("workers", 1), total_groups, 4))
                workers_label = f"{workers_to_use} parallel workers" if workers_to_use > 1 else "sequential"

                self._update_step(
                    5,
                    "in_progress",
                    progress_message=(
                        f"[3/5] Generating claims by category — 0/{total_groups} categories "
                        f"({workers_label}, model: {self.model_name}, prompt: V{ver})..."
                    ),
                    progress_current=0,
                    progress_total=max(total_groups, 1),
                )

                def _process_category_batch(category: str, idx_list: list[int]) -> dict[int, dict]:
                    import time as _time

                    start_time = _time.time()
                    first_sk = skeletons[idx_list[0]]
                    rendered = render_template(
                        template_text,
                        competitor=self.competitor,
                        category=category,
                        key_differentiator=first_sk.get("key_differentiator", ""),
                        description=first_sk.get("description", ""),
                        databricks_rating=first_sk.get("databricks_rating", ""),
                        competitor_rating=first_sk.get("competitor_rating", ""),
                        selection_reasoning=first_sk.get("selection_reasoning", ""),
                        num_diffs=len(category_payloads.get(category, [])),
                        key_diffs_json=json.dumps(category_payloads.get(category, []), indent=2),
                        directives=directives_for_template,
                        context=context,
                    )

                    try:
                        debug_result = call_model_with_debug(
                            client=self.client,
                            model_name=self.model_name,
                            rendered_prompt=rendered,
                            json_schema=pass2_json_schema if pass2_request_spec.get("use_structured_output") else None,
                        )
                        raw = debug_result["content"]
                        parsed = parse_model_json(raw)
                        elapsed_ms = int((_time.time() - start_time) * 1000)
                        batch_results = {}

                        for idx in idx_list:
                            sk = skeletons[idx]
                            batch_results[idx] = _normalize_batched_result(parsed, sk, strict_match=True)
                            self._save_pass2_debug_log(
                                {
                                    "session_id": self.session_id,
                                    "skeleton_index": idx,
                                    "key_differentiator": (sk.get("key_differentiator", "") or "")[:500],
                                    "category": (category or "")[:255],
                                    "rendered_prompt": rendered,
                                    "api_request_json": json.dumps(debug_result["api_request"], indent=2),
                                    "api_response_raw": json.dumps(debug_result["api_response_raw"], indent=2)
                                    if isinstance(debug_result["api_response_raw"], dict)
                                    else str(debug_result["api_response_raw"]),
                                    "structured_output": json.dumps(parsed, indent=2),
                                    "response_type": type(parsed).__name__
                                    if type(parsed).__name__ in ("dict", "list")
                                    else "unknown",
                                    "was_list_fixed": False,
                                    "lakebase_saved": False,
                                    "lakebase_error": None,
                                    "error_message": None,
                                    "processing_time_ms": elapsed_ms,
                                }
                            )
                        return batch_results
                    except Exception as e:
                        elapsed_ms = int((_time.time() - start_time) * 1000)
                        for idx in idx_list:
                            sk = skeletons[idx]
                            self._save_pass2_debug_log(
                                {
                                    "session_id": self.session_id,
                                    "skeleton_index": idx,
                                    "key_differentiator": (sk.get("key_differentiator", "") or "")[:500],
                                    "category": (category or "")[:255],
                                    "rendered_prompt": rendered,
                                    "api_request_json": None,
                                    "api_response_raw": None,
                                    "structured_output": None,
                                    "response_type": "error",
                                    "was_list_fixed": False,
                                    "lakebase_saved": False,
                                    "lakebase_error": None,
                                    "error_message": str(e)[:2000],
                                    "processing_time_ms": elapsed_ms,
                                }
                            )
                        raise

                completed_groups = 0
                completed_claims = 0

                if runtime_mode == "category_parallel" and workers_to_use > 1 and total_groups > 1:
                    futures = {}
                    with ThreadPoolExecutor(max_workers=workers_to_use) as executor:
                        for category, idx_list in category_groups:
                            futures[executor.submit(_process_category_batch, category, idx_list)] = (category, idx_list)

                        for future in as_completed(futures):
                            category, idx_list = futures[future]
                            try:
                                batch_results = future.result()
                                for idx, claim in batch_results.items():
                                    all_claims[idx] = claim
                            except Exception as e:
                                error_str = str(e)
                                logger.error("Pass 2 failed for category '%s': %s", category, error_str)
                                for idx in idx_list:
                                    sk = skeletons[idx]
                                    kd_name = sk.get("key_differentiator", "")[:50]
                                    all_claims[idx] = self._stub_pass2_claim(sk)
                                    errors.append({"index": idx, "key_differentiator": kd_name, "error": error_str})
                                self._update_step(5, "in_progress", last_error=f"[{category}] {error_str[:500]}")

                            completed_groups += 1
                            completed_claims += len(idx_list)
                            error_suffix = f" ({len(errors)} errors)" if errors else ""
                            self._update_step(
                                5,
                                "in_progress",
                                progress_current=completed_groups,
                                progress_total=max(total_groups, 1),
                                progress_message=(
                                    f"[3/5] Generating claims by category — {completed_groups}/{total_groups} "
                                    f"categories ({completed_claims}/{total} diffs){error_suffix}: {category}"
                                ),
                            )

                            self._save_artifact(
                                5,
                                "pass2_claims",
                                "claims.json",
                                json.dumps(all_claims, indent=2),
                                metadata={
                                    "count": total,
                                    "partial": completed_groups < total_groups,
                                    "completed": completed_claims,
                                    "total": total,
                                    "completed_categories": completed_groups,
                                    "total_categories": total_groups,
                                    "model": self.model_name,
                                    "errors": errors,
                                    "mode": "category_batched",
                                },
                            )
                else:
                    for category, idx_list in category_groups:
                        try:
                            batch_results = _process_category_batch(category, idx_list)
                            for idx, claim in batch_results.items():
                                all_claims[idx] = claim
                        except Exception as e:
                            error_str = str(e)
                            logger.error("Pass 2 failed for category '%s': %s", category, error_str)
                            for idx in idx_list:
                                sk = skeletons[idx]
                                kd_name = sk.get("key_differentiator", "")[:50]
                                all_claims[idx] = self._stub_pass2_claim(sk)
                                errors.append({"index": idx, "key_differentiator": kd_name, "error": error_str})
                            self._update_step(5, "in_progress", last_error=f"[{category}] {error_str[:500]}")

                        completed_groups += 1
                        completed_claims += len(idx_list)
                        error_suffix = f" ({len(errors)} errors)" if errors else ""
                        self._update_step(
                            5,
                            "in_progress",
                            progress_current=completed_groups,
                            progress_total=max(total_groups, 1),
                            progress_message=(
                                f"[3/5] Generating claims by category — {completed_groups}/{total_groups} "
                                f"categories ({completed_claims}/{total} diffs){error_suffix}: {category}"
                            ),
                        )
            elif runtime_mode == "single_call":
                self._update_step(
                    5,
                    "in_progress",
                    progress_message=(
                        f"[3/5] Generating claims in single-call mode — 0/1 "
                        f"(model: {self.model_name}, prompt: V{ver})..."
                    ),
                    progress_current=0,
                    progress_total=1,
                )

                def _process_single_call() -> dict[int, dict]:
                    import time as _time

                    start_time = _time.time()
                    first_sk = skeletons[0] if skeletons else {}
                    rendered = render_template(
                        template_text,
                        competitor=self.competitor,
                        category="ALL_CATEGORIES",
                        key_differentiator=first_sk.get("key_differentiator", ""),
                        description=first_sk.get("description", ""),
                        databricks_rating=first_sk.get("databricks_rating", ""),
                        competitor_rating=first_sk.get("competitor_rating", ""),
                        selection_reasoning=first_sk.get("selection_reasoning", ""),
                        num_diffs=total,
                        key_diffs_json=json.dumps(
                            [
                                {
                                    "id": sk.get("id", ""),
                                    "category": sk.get("category", ""),
                                    "key_differentiator": sk.get("key_differentiator", ""),
                                    "description": sk.get("description", ""),
                                    "databricks_rating": sk.get("databricks_rating", ""),
                                    "competitor_rating": sk.get("competitor_rating", ""),
                                    "selection_reasoning": sk.get("selection_reasoning", ""),
                                }
                                for sk in skeletons
                            ],
                            indent=2,
                        ),
                        directives=directives_for_template,
                        context=context,
                    )
                    debug_result = call_model_with_debug(
                        client=self.client,
                        model_name=self.model_name,
                        rendered_prompt=rendered,
                        json_schema=pass2_json_schema if pass2_request_spec.get("use_structured_output") else None,
                    )
                    raw = debug_result["content"]
                    parsed = parse_model_json(raw)
                    elapsed_ms = int((_time.time() - start_time) * 1000)
                    out = {}
                    for idx, sk in enumerate(skeletons):
                        out[idx] = _normalize_batched_result(parsed, sk, strict_match=True)
                        self._save_pass2_debug_log(
                            {
                                "session_id": self.session_id,
                                "skeleton_index": idx,
                                "key_differentiator": (sk.get("key_differentiator", "") or "")[:500],
                                "category": (sk.get("category", "") or "")[:255],
                                "rendered_prompt": rendered,
                                "api_request_json": json.dumps(debug_result["api_request"], indent=2),
                                "api_response_raw": json.dumps(debug_result["api_response_raw"], indent=2)
                                if isinstance(debug_result["api_response_raw"], dict)
                                else str(debug_result["api_response_raw"]),
                                "structured_output": json.dumps(parsed, indent=2),
                                "response_type": type(parsed).__name__
                                if type(parsed).__name__ in ("dict", "list")
                                else "unknown",
                                "was_list_fixed": False,
                                "lakebase_saved": False,
                                "lakebase_error": None,
                                "error_message": None,
                                "processing_time_ms": elapsed_ms,
                            }
                        )
                    return out

                try:
                    result_map = _process_single_call()
                    for idx, val in result_map.items():
                        all_claims[idx] = val
                    self._update_step(
                        5,
                        "in_progress",
                        progress_message=f"[3/5] Generating claims in single-call mode — 1/1 complete ({total} diffs).",
                        progress_current=1,
                        progress_total=1,
                    )
                except Exception as e:
                    err = str(e)
                    logger.error("Pass 2 single-call generation failed: %s", err)
                    for idx, sk in enumerate(skeletons):
                        all_claims[idx] = self._stub_pass2_claim(sk)
                        errors.append({"index": idx, "key_differentiator": sk.get("key_differentiator", "")[:50], "error": err})
                    self._update_step(5, "in_progress", last_error=f"[single_call] {err[:500]}")
            else:
                workers_to_use = max(1, min(step5_exec.get("workers", 1), total, 8))
                workers_label = f"{workers_to_use} parallel workers" if workers_to_use > 1 else "sequential"
                self._update_step(
                    5,
                    "in_progress",
                    progress_message=f"[3/5] Generating claims — 0/{total} done ({workers_label}, model: {self.model_name}, prompt: V{ver})...",
                    progress_current=0,
                    progress_total=max(total, 1),
                )

                if runtime_mode == "per_diff_parallel" and workers_to_use > 1 and total > 1:
                    futures = {}
                    with ThreadPoolExecutor(max_workers=workers_to_use) as executor:
                        for idx, sk in enumerate(skeletons):
                            future = executor.submit(_process_single_diff, idx, sk)
                            futures[future] = (idx, sk)

                        for future in as_completed(futures):
                            idx, sk = futures[future]
                            kd_name = sk.get("key_differentiator", "")[:50]
                            try:
                                all_claims[idx] = future.result()
                            except Exception as e:
                                error_str = str(e)
                                logger.error("Pass 2 failed for skeleton %d (%s): %s", idx, kd_name, error_str)
                                all_claims[idx] = self._stub_pass2_claim(sk)
                                errors.append({"index": idx, "key_differentiator": kd_name, "error": error_str})
                                self._update_step(5, "in_progress", last_error=f"[{kd_name}] {error_str[:500]}")

                            completed += 1
                            error_suffix = f" ({len(errors)} errors)" if errors else ""
                            self._update_step(
                                5,
                                "in_progress",
                                progress_current=completed,
                                progress_total=max(total, 1),
                                progress_message=f"[3/5] Generating claims — {completed}/{total} done{error_suffix}: {kd_name}",
                            )

                            self._save_artifact(
                                5,
                                "pass2_claims",
                                "claims.json",
                                json.dumps(all_claims, indent=2),
                                metadata={
                                    "count": total,
                                    "partial": completed < total,
                                    "completed": completed,
                                    "total": total,
                                    "model": self.model_name,
                                    "errors": errors,
                                    "mode": "per_diff",
                                },
                            )
                else:
                    for idx, sk in enumerate(skeletons):
                        kd_name = sk.get("key_differentiator", "")[:50]
                        self._update_step(
                            5,
                            "in_progress",
                            progress_current=idx,
                            progress_total=max(total, 1),
                            progress_message=f"[3/5] Generating claim {idx + 1}/{total}: {kd_name}",
                        )

                        try:
                            all_claims[idx] = _process_single_diff(idx, sk)
                        except Exception as e:
                            error_str = str(e)
                            logger.error("Pass 2 failed for skeleton %d (%s): %s", idx, kd_name, error_str)
                            all_claims[idx] = self._stub_pass2_claim(sk)
                            errors.append({"index": idx, "key_differentiator": kd_name, "error": error_str})
                            self._update_step(5, "in_progress", last_error=f"[{kd_name}] {error_str[:500]}")

            for idx, claim in enumerate(all_claims):
                if claim is None:
                    all_claims[idx] = self._stub_pass2_claim(skeletons[idx])
                    errors.append(
                        {
                            "index": idx,
                            "key_differentiator": skeletons[idx].get("key_differentiator", "")[:50],
                            "error": "Missing model output; inserted fallback stub.",
                        }
                    )

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
                    "execution_mode": runtime_mode,
                    "step5_inline_fact_check": bool(self.step5_inline_fact_check),
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
                self._save_claims_to_lakebase(
                    skeletons,
                    all_claims,
                    inline_fact_check=bool(self.step5_inline_fact_check),
                )
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
            "databricks_details_verbose": [[]],
            "databricks_reasoning": "Generation failed — stub reasoning.",
            "competitor_headline": f"{self.competitor}: {kd_name}",
            "competitor_details": [f"{self.competitor} has partial support for {kd_name}."],
            "competitor_details_verbose": [[]],
            "competitor_reasoning": "Generation failed — stub reasoning.",
            "citations": {
                "databricks_headline": [],
                "databricks_details": [],
                "databricks_detail_verbose": [],
                "databricks_reasoning": [],
                "competitor_headline": [],
                "competitor_details": [],
                "competitor_detail_verbose": [],
                "competitor_reasoning": [],
            },
            "sources": [],
            "research_sources": [],
        }

    def _save_claims_to_lakebase(self, skeletons, claims, inline_fact_check: bool = False):
        """Save Pass 2 claims into the Lakebase claims/evidence/fact_checks tables."""
        with self.engine.begin() as conn:
            # Reuse the generation created at workflow creation.
            # If missing, recover deterministically and link it before writes.
            gen_id = self._get_or_create_generation_id(conn)
            session_created_at = conn.execute(
                text("SELECT created_at FROM workflow_sessions WHERE session_id::text = :sid"),
                {"sid": self.session_id},
            ).scalar()

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
            for idx, (sk, claim) in enumerate(zip(skeletons, claims)):
                kd_name = sk.get("key_differentiator", "")
                category_name = sk.get("category", "")
                kd_id = sk.get("key_diff_id")

                # Validate provided key_diff_id belongs to this category/generation.
                if kd_id:
                    valid = conn.execute(
                        text(
                            "SELECT 1 "
                            "FROM key_differentiators kd "
                            "JOIN product_category_catalog pcc ON kd.category_id = pcc.catalog_id "
                            "WHERE kd.key_diff_id = :kid "
                            "AND pcc.category_name = :category "
                            "AND (kd.generation_id = :gid OR kd.generation_id IS NULL) "
                            "LIMIT 1"
                        ),
                        {"kid": int(kd_id), "category": category_name, "gid": int(gen_id)},
                    ).scalar()
                    if not valid:
                        kd_id = None

                if not kd_id and category_name:
                    kd_id = conn.execute(
                        text(
                            "SELECT kd.key_diff_id "
                            "FROM key_differentiators kd "
                            "JOIN product_category_catalog pcc ON kd.category_id = pcc.catalog_id "
                            "WHERE kd.generation_id = :gid "
                            "AND kd.key_diff_name = :name "
                            "AND pcc.category_name = :category "
                            "ORDER BY kd.key_diff_id DESC LIMIT 1"
                        ),
                        {"gid": int(gen_id), "name": kd_name, "category": category_name},
                    ).scalar()

                # Recovery path for older Step 4 writes that missed generation_id.
                if not kd_id and category_name and session_created_at:
                    kd_id = conn.execute(
                        text(
                            "SELECT kd.key_diff_id "
                            "FROM key_differentiators kd "
                            "JOIN product_category_catalog pcc ON kd.category_id = pcc.catalog_id "
                            "WHERE kd.generation_id IS NULL "
                            "AND kd.key_diff_name = :name "
                            "AND pcc.category_name = :category "
                            "AND kd.created_at >= :session_ts - INTERVAL '2 hours' "
                            "AND kd.created_at <= NOW() "
                            "ORDER BY kd.key_diff_id DESC LIMIT 1"
                        ),
                        {"name": kd_name, "category": category_name, "session_ts": session_created_at},
                    ).scalar()
                if not kd_id:
                    self._update_pass2_debug_log_lakebase(
                        idx,
                        False,
                        f"key_diff_id not found for '{kd_name}' in category '{category_name}'",
                    )
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

                # Track Lakebase save for debug log
                lakebase_error = None
                try:
                    def _normalize_verbose_map(raw_value, detail_count):
                        if detail_count <= 0:
                            return []
                        if raw_value is None:
                            return [[] for _ in range(detail_count)]
                        if isinstance(raw_value, str):
                            raw_value = [raw_value]
                        if isinstance(raw_value, list):
                            if raw_value and isinstance(raw_value[0], list):
                                out = []
                                for row in raw_value:
                                    row_items = []
                                    if isinstance(row, list):
                                        for cell in row:
                                            txt = str(cell or "").strip()
                                            if txt:
                                                row_items.append(txt)
                                    elif isinstance(row, str) and row.strip():
                                        row_items.append(row.strip())
                                    out.append(row_items)
                            else:
                                flat = [str(v or "").strip() for v in raw_value if str(v or "").strip()]
                                if not flat:
                                    out = [[] for _ in range(detail_count)]
                                elif len(flat) == detail_count:
                                    out = [[v] for v in flat]
                                else:
                                    out = [flat] + [[] for _ in range(max(detail_count - 1, 0))]
                        else:
                            out = [[] for _ in range(detail_count)]
                        if len(out) < detail_count:
                            out.extend([[] for _ in range(detail_count - len(out))])
                        return out[:detail_count]

                    # Databricks claim
                    db_rating = _rating_to_db(sk.get("databricks_rating", ""))
                    db_details_raw = claim.get("databricks_details", [])
                    if isinstance(db_details_raw, str):
                        db_details_raw = [db_details_raw] if db_details_raw else []
                    db_verbose_map = _normalize_verbose_map(claim.get("databricks_details_verbose"), len(db_details_raw))
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
                    db_verbose_item_ids_by_detail = {}
                    for order, item_text in enumerate(db_details_raw):
                        verbose_items = db_verbose_map[order] if order < len(db_verbose_map) else []
                        did = conn.execute(
                            text(
                                "INSERT INTO claim_detail_items (claim_id, item_order, item_text, item_text_verbose) "
                                "VALUES (:cid, :ord, :txt, :txtv) RETURNING detail_item_id"
                            ),
                            {
                                "cid": db_claim_id,
                                "ord": order,
                                "txt": item_text,
                                "txtv": "\n".join(verbose_items) if verbose_items else None,
                            },
                        ).scalar()
                        db_detail_item_ids.append(did)
                        db_verbose_item_ids_by_detail[order] = []
                        for v_order, v_text in enumerate(verbose_items):
                            vid = conn.execute(
                                text(
                                    "INSERT INTO claim_detail_verbose_items (detail_item_id, item_order, item_text) "
                                    "VALUES (:did, :ord, :txt) RETURNING verbose_item_id"
                                ),
                                {"did": did, "ord": v_order, "txt": v_text},
                            ).scalar()
                            db_verbose_item_ids_by_detail[order].append(vid)

                    # Competitor claim
                    comp_rating = _rating_to_db(sk.get("competitor_rating", ""))
                    comp_details_raw = claim.get("competitor_details", [])
                    if isinstance(comp_details_raw, str):
                        comp_details_raw = [comp_details_raw] if comp_details_raw else []
                    comp_verbose_map = _normalize_verbose_map(claim.get("competitor_details_verbose"), len(comp_details_raw))
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
                    comp_verbose_item_ids_by_detail = {}
                    for order, item_text in enumerate(comp_details_raw):
                        verbose_items = comp_verbose_map[order] if order < len(comp_verbose_map) else []
                        did = conn.execute(
                            text(
                                "INSERT INTO claim_detail_items (claim_id, item_order, item_text, item_text_verbose) "
                                "VALUES (:cid, :ord, :txt, :txtv) RETURNING detail_item_id"
                            ),
                            {
                                "cid": comp_claim_id,
                                "ord": order,
                                "txt": item_text,
                                "txtv": "\n".join(verbose_items) if verbose_items else None,
                            },
                        ).scalar()
                        comp_detail_item_ids.append(did)
                        comp_verbose_item_ids_by_detail[order] = []
                        for v_order, v_text in enumerate(verbose_items):
                            vid = conn.execute(
                                text(
                                    "INSERT INTO claim_detail_verbose_items (detail_item_id, item_order, item_text) "
                                    "VALUES (:did, :ord, :txt) RETURNING verbose_item_id"
                                ),
                                {"did": did, "ord": v_order, "txt": v_text},
                            ).scalar()
                            comp_verbose_item_ids_by_detail[order].append(vid)

                    # Save evidence + fact checks from citations
                    self._save_evidence_and_fact_checks(
                        conn,
                        gen_id,
                        db_claim_id,
                        claim,
                        "databricks",
                        db_detail_item_ids,
                        db_verbose_item_ids_by_detail,
                        inline_fact_check=inline_fact_check,
                    )
                    self._save_evidence_and_fact_checks(
                        conn,
                        gen_id,
                        comp_claim_id,
                        claim,
                        "competitor",
                        comp_detail_item_ids,
                        comp_verbose_item_ids_by_detail,
                        inline_fact_check=inline_fact_check,
                    )

                    # Mark as successfully saved to Lakebase
                    self._update_pass2_debug_log_lakebase(idx, True, None)

                except Exception as e:
                    lakebase_error = str(e)[:500]
                    logger.error("Lakebase save failed for skeleton %d: %s", idx, lakebase_error)
                    self._update_pass2_debug_log_lakebase(idx, False, lakebase_error)
                    raise

            conn.execute(
                text("UPDATE battlecard_generations SET total_claims = :tc WHERE generation_id = :gid"),
                {"tc": total_claims, "gid": gen_id},
            )

    def _save_evidence_and_fact_checks(
        self,
        conn,
        gen_id,
        claim_id,
        claim_data,
        side,
        detail_item_ids=None,
        verbose_item_ids_by_detail=None,
        inline_fact_check: bool = False,
    ):
        """Save evidence rows and optionally seed fact-check rows from citation metadata."""
        citations = claim_data.get("citations", {})
        sources_list = claim_data.get("sources", [])
        if detail_item_ids is None:
            detail_item_ids = []
        if verbose_item_ids_by_detail is None:
            verbose_item_ids_by_detail = {}

        # Build source_index -> source mapping
        source_map = {}
        for i, src in enumerate(sources_list or []):
            if isinstance(src, dict):
                source_map[src.get("index")] = src
            else:
                src_text = str(src or "")
                source_map[i + 1] = {
                    "index": i + 1,
                    "title": src_text[:120] if src_text else f"Source {i + 1}",
                    "url": src_text if src_text.startswith("http") else "",
                    "type": "context",
                }

        field_specs = (
            ("headline", "headline"),
            ("details", "l200"),
            ("detail_verbose", "l300"),
            ("reasoning", "reasoning"),
        )

        for field_suffix, claim_subfield in field_specs:
            field_key = f"{side}_{field_suffix}"
            field_citations = citations.get(field_key, [])
            if not isinstance(field_citations, list):
                field_citations = []

            for cite_order, cite in enumerate(field_citations):
                if not isinstance(cite, dict):
                    continue
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
                detail_idx = int(cite.get("detail_item_index", 0) or 0)
                if field_suffix in ("details", "detail_verbose") and detail_item_ids:
                    if 0 <= detail_idx < len(detail_item_ids):
                        detail_item_id = detail_item_ids[detail_idx]

                verbose_item_id = None
                if field_suffix == "detail_verbose" and detail_item_id is not None:
                    verbose_idx = int(cite.get("verbose_item_index", 0) or 0)
                    verbose_ids = verbose_item_ids_by_detail.get(detail_idx, [])
                    if 0 <= verbose_idx < len(verbose_ids):
                        verbose_item_id = verbose_ids[verbose_idx]

                if claim_subfield == "headline":
                    traces_to_field = "headline"
                elif detail_item_id:
                    traces_to_field = "detail_item"
                else:
                    traces_to_field = "description"
                claim_subfield_value = str(cite.get("claim_subfield", claim_subfield) or claim_subfield).strip().lower()
                if claim_subfield_value not in ("headline", "reasoning", "l200", "l300"):
                    claim_subfield_value = claim_subfield

                # Insert evidence
                traces_text = cite.get("source_quote", "")[:500]
                evidence_id = conn.execute(
                    text(
                        "INSERT INTO evidence (claim_id, detail_item_id, verbose_item_id, traces_to_field, claim_subfield, citation_id, citation_order, "
                        "traces_to_start_index, traces_to_end_index, traces_to_text, generation_source_id, generation_source_text) "
                        "VALUES (:claim, :did, :vid, :field, :subfield, :cid, :corder, :start, :end, :trace, :src, :src_text) RETURNING evidence_id"
                    ),
                    {
                        "claim": claim_id,
                        "did": detail_item_id,
                        "vid": verbose_item_id,
                        "field": traces_to_field,
                        "subfield": claim_subfield_value,
                        "cid": str(cite.get("citation_id", "") or None),
                        "corder": int(cite.get("citation_order", cite_order) or cite_order),
                        "start": cite.get("start_index", 0),
                        "end": cite.get("end_index", 0),
                        "trace": traces_text,
                        "src": source_id,
                        "src_text": traces_text,
                    },
                ).scalar()

                if inline_fact_check:
                    verdict = str(cite.get("verdict", "pending") or "pending").strip().lower()
                    if verdict not in ("pending", "verified", "unverified", "disputed", "outdated", "not_applicable"):
                        verdict = "pending"
                    confidence_raw = cite.get("confidence")
                    confidence = None
                    if confidence_raw is not None:
                        try:
                            conf_val = float(confidence_raw)
                            if conf_val <= 1:
                                conf_val *= 100
                            confidence = max(0, min(100, int(round(conf_val))))
                        except Exception:
                            confidence = None
                    conn.execute(
                        text(
                            "INSERT INTO fact_checks "
                            "(evidence_id, status, fact_check_source_id, fact_check_source_text, reasoning, dispute_details, "
                            "checked_at, checked_by, check_method, confidence_score) "
                            "VALUES (:eid, :status, :src_id, :src_text, :reasoning, :dispute, NOW(), :checked_by, :method, :confidence)"
                        ),
                        {
                            "eid": evidence_id,
                            "status": verdict,
                            "src_id": source_id,
                            "src_text": traces_text,
                            "reasoning": str(cite.get("verdict_rationale", "") or ""),
                            "dispute": "",
                            "checked_by": "workflow_runner_inline",
                            "method": "llm_assisted",
                            "confidence": confidence,
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
        ctx_flags = resolve_context_source_flags(context_sources)
        directive = (self._get_artifact_content("directive_generated") or "") if ctx_flags["directive"] else ""
        context = self._build_context(ctx_flags)
        errors = []

        try:
            # Stage 2: Load template
            ver = getattr(self, 'pass2_prompt_template_version', 2)
            self._update_step(5, "in_progress",
                              progress_message=f"[2/4] Loading Pass 2 prompt template V{ver} for regeneration ({total} claims)...",
                              progress_current=1, progress_total=4)
            template_text = load_pass2_template(ver, engine=self.engine)
            directives_for_template = "" if ver in (5, 6, 7, 8, 9, 10) else directive
            pass2_request_spec = get_pass2_request_spec_with_engine(ver, engine=self.engine)
            pass2_json_schema = pass2_request_spec.get("schema")

            def _normalize_sources(sources):
                if not isinstance(sources, list):
                    return []
                normalized = []
                for i, src in enumerate(sources):
                    if isinstance(src, dict):
                        normalized.append(
                            {
                                "index": int(src.get("index", i + 1)),
                                "title": str(src.get("title", f"Source {i + 1}")),
                                "url": str(src.get("url", "")),
                                "type": str(src.get("type", "context")),
                                "accessed_at": str(src.get("accessed_at", datetime.now(timezone.utc).isoformat())),
                            }
                        )
                    else:
                        src_str = str(src)
                        normalized.append(
                            {
                                "index": i + 1,
                                "title": src_str[:120] if src_str else f"Source {i + 1}",
                                "url": src_str if src_str.startswith("http") else "",
                                "type": "context",
                                "accessed_at": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                return normalized

            def _normalize_v5_v6_result(parsed_result, skeleton, *, strict_match: bool = False):
                # Some model SDK paths return wrapper blocks like:
                # {"type":"text","text":"{...json...}"}.
                # Unwrap these before candidate matching.
                for _ in range(3):
                    if isinstance(parsed_result, dict):
                        wrapper_type = str(parsed_result.get("type", "")).strip().lower()
                        wrapper_text = parsed_result.get("text")
                        if isinstance(wrapper_text, str) and (
                            wrapper_type in ("text", "output_text")
                            or set(parsed_result.keys()).issubset({"type", "text"})
                        ):
                            try:
                                parsed_result = parse_model_json(wrapper_text)
                                continue
                            except Exception:
                                break
                    if (
                        isinstance(parsed_result, list)
                        and len(parsed_result) == 1
                        and isinstance(parsed_result[0], dict)
                    ):
                        first = parsed_result[0]
                        wrapper_type = str(first.get("type", "")).strip().lower()
                        wrapper_text = first.get("text")
                        if isinstance(wrapper_text, str) and wrapper_type in ("text", "output_text"):
                            try:
                                parsed_result = parse_model_json(wrapper_text)
                                continue
                            except Exception:
                                break
                    break

                target_id = skeleton.get("id", "")
                target_name = skeleton.get("key_differentiator", "").strip().lower()

                if isinstance(parsed_result, dict):
                    claims_list = parsed_result.get("claims")
                    if isinstance(claims_list, str):
                        try:
                            claims_list = json.loads(claims_list)
                        except Exception:
                            try:
                                claims_list = parse_model_json(claims_list)
                            except Exception:
                                claims_list = None
                    if isinstance(claims_list, list):
                        candidates = [c for c in claims_list if isinstance(c, dict)]
                    else:
                        candidates = [parsed_result]
                elif isinstance(parsed_result, list):
                    candidates = [c for c in parsed_result if isinstance(c, dict)]
                else:
                    candidates = []

                chosen = None
                if target_id:
                    id_matches = [c for c in candidates if c.get("id") == target_id]
                    if len(id_matches) == 1:
                        chosen = id_matches[0]
                    elif len(id_matches) > 1 and strict_match:
                        raise ValueError(f"Batched regeneration response had duplicate id matches for {target_id}")
                if chosen is None and target_name:
                    name_matches = [
                        c
                        for c in candidates
                        if str(c.get("key_differentiator", "")).strip().lower() == target_name
                    ]
                    if len(name_matches) == 1:
                        chosen = name_matches[0]
                    elif len(name_matches) > 1 and strict_match:
                        raise ValueError(
                            f"Batched regeneration response had ambiguous key_differentiator matches for '{target_name}'"
                        )
                if chosen is None and candidates and not strict_match:
                    chosen = candidates[0]
                if chosen is None:
                    raise ValueError(
                        f"Batched regeneration response did not contain a usable object for id='{target_id}' "
                        f"name='{target_name}' (candidates={len(candidates)})"
                    )

                def _to_text_list(value):
                    if value is None:
                        return []
                    if isinstance(value, str):
                        return [value] if value else []
                    if isinstance(value, list):
                        out = []
                        for item in value:
                            if isinstance(item, str):
                                if item:
                                    out.append(item)
                            elif isinstance(item, dict):
                                txt = str(item.get("text", "") or "").strip()
                                if txt:
                                    out.append(txt)
                        return out
                    return [str(value)]

                def _extract_side(chosen_obj, side_key, top_prefix):
                    side = chosen_obj.get(side_key, {}) if isinstance(chosen_obj.get(side_key), dict) else {}
                    headline_raw = side.get("headline", chosen_obj.get(f"{top_prefix}_headline", ""))
                    reasoning_raw = side.get("reasoning", chosen_obj.get(f"{top_prefix}_reasoning", ""))
                    headline_text = str(headline_raw.get("text", "") or "") if isinstance(headline_raw, dict) else str(headline_raw or "")
                    reasoning_text = str(reasoning_raw.get("text", "") or "") if isinstance(reasoning_raw, dict) else str(reasoning_raw or "")
                    l200_items = []
                    l300_map = []
                    side_l200 = side.get("l200")
                    if isinstance(side_l200, list) and side_l200 and isinstance(side_l200[0], dict):
                        for l200_item in side_l200:
                            l200_items.append(str(l200_item.get("text", "") or "").strip())
                            raw_l300 = l200_item.get("l300", [])
                            l300_items = []
                            if isinstance(raw_l300, list):
                                for v in raw_l300:
                                    if isinstance(v, dict):
                                        txt = str(v.get("text", "") or "").strip()
                                    else:
                                        txt = str(v or "").strip()
                                    if txt:
                                        l300_items.append(txt)
                            l300_map.append(l300_items)
                    else:
                        l200_items = _to_text_list(chosen_obj.get(f"{top_prefix}_l200", chosen_obj.get(f"{top_prefix}_details", [])))
                        raw_l300 = chosen_obj.get(f"{top_prefix}_l300", [])
                        if isinstance(raw_l300, list) and raw_l300 and isinstance(raw_l300[0], list):
                            l300_map = [list(map(str, row)) for row in raw_l300]
                        else:
                            as_list = _to_text_list(raw_l300)
                            if l200_items:
                                if len(as_list) == len(l200_items):
                                    l300_map = [[v] if v else [] for v in as_list]
                                else:
                                    l300_map = [as_list] + [[] for _ in range(max(len(l200_items) - 1, 0))]
                            else:
                                l300_map = [as_list] if as_list else []
                    if len(l300_map) < len(l200_items):
                        l300_map.extend([[] for _ in range(len(l200_items) - len(l300_map))])
                    return headline_text, l200_items, l300_map, reasoning_text

                db_head, db_details, db_l300, db_reasoning = _extract_side(chosen, "databricks", "databricks")
                comp_head, comp_details, comp_l300, comp_reasoning = _extract_side(chosen, "competitor", "competitor")

                sources = _normalize_sources(chosen.get("sources", []))
                if not sources:
                    sources = [
                        {
                            "index": 1,
                            "title": "Model output (no explicit source URL)",
                            "url": "",
                            "type": "context",
                            "accessed_at": datetime.now(timezone.utc).isoformat(),
                        }
                    ]

                citations = chosen.get("citations")
                if not isinstance(citations, dict):
                    citations = build_fallback_citations(
                        db_details,
                        comp_details,
                        db_reasoning,
                        comp_reasoning,
                        sources,
                    )
                else:
                    for key in (
                        "databricks_headline",
                        "databricks_details",
                        "databricks_detail_verbose",
                        "databricks_reasoning",
                        "competitor_headline",
                        "competitor_details",
                        "competitor_detail_verbose",
                        "competitor_reasoning",
                    ):
                        if key not in citations or not isinstance(citations.get(key), list):
                            citations[key] = []
                        else:
                            citations[key] = [c for c in citations.get(key, []) if isinstance(c, dict)]

                    if not any(len(v) for v in citations.values()):
                        citations = build_fallback_citations(
                            db_details,
                            comp_details,
                            db_reasoning,
                            comp_reasoning,
                            sources,
                        )

                return {
                    "databricks_headline": db_head or chosen.get("databricks_headline", ""),
                    "databricks_details": db_details,
                    "databricks_details_verbose": db_l300 if isinstance(db_l300, list) else [],
                    "databricks_reasoning": db_reasoning or chosen.get("databricks_reasoning", ""),
                    "competitor_headline": comp_head or chosen.get("competitor_headline", ""),
                    "competitor_details": comp_details,
                    "competitor_details_verbose": comp_l300 if isinstance(comp_l300, list) else [],
                    "competitor_reasoning": comp_reasoning or chosen.get("competitor_reasoning", ""),
                    "citations": citations,
                    "sources": sources,
                    "research_sources": chosen.get("research_sources", []),
                }

            category_payloads = {}
            for sk in skeletons:
                cat = sk.get("category", "")
                category_payloads.setdefault(cat, []).append(
                    {
                        "id": sk.get("id", ""),
                        "category": sk.get("category", ""),
                        "key_differentiator": sk.get("key_differentiator", ""),
                        "description": sk.get("description", ""),
                        "databricks_rating": sk.get("databricks_rating", ""),
                        "competitor_rating": sk.get("competitor_rating", ""),
                        "selection_reasoning": sk.get("selection_reasoning", ""),
                    }
                )

            updated_claims = list(claims)
            step5_exec = self._resolve_step5_execution(ver, len(category_payloads), total)
            runtime_mode = step5_exec["runtime_mode"]

            if runtime_mode in ("category_parallel", "category_sequential"):
                category_to_indexes = {}
                for idx, sk in enumerate(skeletons):
                    category_to_indexes.setdefault(sk.get("category", ""), []).append(idx)
                category_groups = list(category_to_indexes.items())
                total_groups = len(category_groups)
                workers_to_use = max(1, min(step5_exec.get("workers", 1), total_groups, 4))
                workers_label = f"{workers_to_use} parallel workers" if workers_to_use > 1 else "sequential"

                self._update_step(
                    5,
                    "in_progress",
                    progress_current=0,
                    progress_total=max(total_groups, 1),
                    progress_message=(
                        f"[3/4] Regenerating by category — 0/{total_groups} categories "
                        f"({workers_label}, prompt: V{ver})"
                    ),
                )

                def _process_category_regen(category: str, idx_list: list[int]) -> dict[int, dict]:
                    first_sk = skeletons[idx_list[0]]
                    rendered = render_template(
                        template_text,
                        competitor=self.competitor,
                        category=category,
                        key_differentiator=first_sk.get("key_differentiator", ""),
                        description=first_sk.get("description", ""),
                        databricks_rating=first_sk.get("databricks_rating", ""),
                        competitor_rating=first_sk.get("competitor_rating", ""),
                        selection_reasoning=first_sk.get("selection_reasoning", ""),
                        num_diffs=len(category_payloads.get(category, [])),
                        key_diffs_json=json.dumps(category_payloads.get(category, []), indent=2),
                        directives=directives_for_template,
                        context=context,
                    )
                    previous_outputs = [{**skeletons[i], **claims[i]} for i in idx_list]
                    rendered += (
                        f"\n\n## Previous Output (to improve upon)\n```json\n{json.dumps(previous_outputs, indent=2)}\n```"
                        f"\n\n## Reviewer Feedback\n{feedback}"
                        "\n\nRegenerate every differentiator in this category while applying the feedback. "
                        "Keep outputs concise and improve factual grounding."
                    )
                    raw = call_model(
                        client=self.client,
                        model_name=self.model_name,
                        rendered_prompt=rendered,
                        json_schema=pass2_json_schema if pass2_request_spec.get("use_structured_output") else None,
                    )
                    parsed = parse_model_json(raw)
                    return {idx: _normalize_v5_v6_result(parsed, skeletons[idx], strict_match=True) for idx in idx_list}

                completed_groups = 0
                if runtime_mode == "category_parallel" and workers_to_use > 1 and total_groups > 1:
                    futures = {}
                    with ThreadPoolExecutor(max_workers=workers_to_use) as executor:
                        for category, idx_list in category_groups:
                            futures[executor.submit(_process_category_regen, category, idx_list)] = (category, idx_list)
                        for future in as_completed(futures):
                            category, idx_list = futures[future]
                            try:
                                result_map = future.result()
                                for idx, val in result_map.items():
                                    updated_claims[idx] = val
                            except Exception as e:
                                error_str = str(e)
                                for idx in idx_list:
                                    kd_name = skeletons[idx].get("key_differentiator", "")[:50]
                                    errors.append({"index": idx, "key_differentiator": kd_name, "error": error_str})
                                self._update_step(5, "in_progress", last_error=f"[Regen {category}] {error_str[:500]}")
                            completed_groups += 1
                            self._update_step(
                                5,
                                "in_progress",
                                progress_current=completed_groups,
                                progress_total=max(total_groups, 1),
                                progress_message=f"[3/4] Regenerating by category — {completed_groups}/{total_groups}: {category}",
                            )
                else:
                    for category, idx_list in category_groups:
                        try:
                            result_map = _process_category_regen(category, idx_list)
                            for idx, val in result_map.items():
                                updated_claims[idx] = val
                        except Exception as e:
                            error_str = str(e)
                            for idx in idx_list:
                                kd_name = skeletons[idx].get("key_differentiator", "")[:50]
                                errors.append({"index": idx, "key_differentiator": kd_name, "error": error_str})
                            self._update_step(5, "in_progress", last_error=f"[Regen {category}] {error_str[:500]}")
                        completed_groups += 1
                        self._update_step(
                            5,
                            "in_progress",
                            progress_current=completed_groups,
                            progress_total=max(total_groups, 1),
                            progress_message=f"[3/4] Regenerating by category — {completed_groups}/{total_groups}: {category}",
                        )
            elif runtime_mode == "single_call":
                self._update_step(
                    5,
                    "in_progress",
                    progress_current=0,
                    progress_total=1,
                    progress_message=f"[3/4] Regenerating in single-call mode (prompt: V{ver})",
                )

                first_sk = skeletons[0] if skeletons else {}
                rendered = render_template(
                    template_text,
                    competitor=self.competitor,
                    category="ALL_CATEGORIES",
                    key_differentiator=first_sk.get("key_differentiator", ""),
                    description=first_sk.get("description", ""),
                    databricks_rating=first_sk.get("databricks_rating", ""),
                    competitor_rating=first_sk.get("competitor_rating", ""),
                    selection_reasoning=first_sk.get("selection_reasoning", ""),
                    num_diffs=total,
                    key_diffs_json=json.dumps(
                        [
                            {
                                "id": sk.get("id", ""),
                                "category": sk.get("category", ""),
                                "key_differentiator": sk.get("key_differentiator", ""),
                                "description": sk.get("description", ""),
                                "databricks_rating": sk.get("databricks_rating", ""),
                                "competitor_rating": sk.get("competitor_rating", ""),
                                "selection_reasoning": sk.get("selection_reasoning", ""),
                            }
                            for sk in skeletons
                        ],
                        indent=2,
                    ),
                    directives=directives_for_template,
                    context=context,
                )
                previous_outputs = [{**skeletons[i], **claims[i]} for i in range(len(skeletons))]
                rendered += (
                    f"\n\n## Previous Output (to improve upon)\n```json\n{json.dumps(previous_outputs, indent=2)}\n```"
                    f"\n\n## Reviewer Feedback\n{feedback}"
                    "\n\nRegenerate every differentiator while applying the feedback. "
                    "Keep outputs concise and improve factual grounding."
                )
                try:
                    raw = call_model(
                        client=self.client,
                        model_name=self.model_name,
                        rendered_prompt=rendered,
                        json_schema=pass2_json_schema if pass2_request_spec.get("use_structured_output") else None,
                    )
                    parsed = parse_model_json(raw)
                    for idx, sk in enumerate(skeletons):
                        updated_claims[idx] = _normalize_v5_v6_result(parsed, sk, strict_match=True)
                    self._update_step(
                        5,
                        "in_progress",
                        progress_current=1,
                        progress_total=1,
                        progress_message=f"[3/4] Regenerating in single-call mode — complete ({total} diffs)",
                    )
                except Exception as e:
                    err = str(e)
                    logger.error("Pass 3 single-call regeneration failed: %s", err)
                    for idx, sk in enumerate(skeletons):
                        kd_name = sk.get("key_differentiator", "")[:50]
                        errors.append({"index": idx, "key_differentiator": kd_name, "error": err})
                    self._update_step(5, "in_progress", last_error=f"[Regen single_call] {err[:500]}")
            else:
                # Stage 3: Regenerate claims sequentially
                for idx, (sk, claim) in enumerate(zip(skeletons, claims)):
                    kd_name = sk.get('key_differentiator', '')[:50]
                    self._update_step(5, "in_progress",
                                      progress_current=idx, progress_total=max(total, 1),
                                      progress_message=f"[3/4] Regenerating claim {idx + 1}/{total}: {kd_name}")

                    rendered = render_template(
                        template_text,
                        competitor=self.competitor,
                        category=sk.get("category", ""),
                        key_differentiator=sk.get("key_differentiator", ""),
                        description=sk.get("description", ""),
                        databricks_rating=sk.get("databricks_rating", ""),
                        competitor_rating=sk.get("competitor_rating", ""),
                        selection_reasoning=sk.get("selection_reasoning", ""),
                        num_diffs=len(category_payloads.get(sk.get("category", ""), [])),
                        key_diffs_json=json.dumps(category_payloads.get(sk.get("category", ""), []), indent=2),
                        directives=directives_for_template,
                        context=context,
                    )

                    current_content = json.dumps({**sk, **claim}, indent=2)
                    rendered += (
                        f"\n\n## Previous Output (to improve upon)\n```json\n{current_content}\n```"
                        f"\n\n## Reviewer Feedback\n{feedback}"
                        "\n\nIncorporate the feedback above while regenerating this differentiator's claims. "
                        "Maintain or improve citation quality."
                    )

                    try:
                        raw = call_model(
                            client=self.client,
                            model_name=self.model_name,
                            rendered_prompt=rendered,
                            json_schema=pass2_json_schema if pass2_request_spec.get("use_structured_output") else None,
                        )
                        updated_claims[idx] = _normalize_v5_v6_result(
                            parse_model_json(raw),
                            sk,
                            strict_match=ver in (5, 6, 7, 8, 9, 10),
                        )
                    except Exception as e:
                        error_str = str(e)
                        logger.error("Pass 3 regen failed for %d (%s): %s", idx, kd_name, error_str)
                        errors.append({"index": idx, "key_differentiator": kd_name, "error": error_str})
                        self._update_step(5, "in_progress", last_error=f"[Regen {kd_name}] {error_str[:500]}")

            # Stage 4: Save artifacts and to Lakebase
            error_suffix = f" ({len(errors)} errors)" if errors else ""
            self._update_step(5, "in_progress",
                              progress_message=f"[4/4] Saving {len(updated_claims)} regenerated claims{error_suffix}...",
                              progress_current=total, progress_total=total)

            regen_content = json.dumps(updated_claims, indent=2)
            art_id = self._save_artifact(
                5, "pass3_regenerated", "claims_regenerated.json",
                regen_content,
                metadata={
                    "count": len(updated_claims),
                    "feedback": feedback[:200],
                    "errors": errors,
                    "execution_mode": runtime_mode,
                    "step5_inline_fact_check": bool(self.step5_inline_fact_check),
                },
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
                self._save_claims_to_lakebase(
                    skeletons,
                    updated_claims,
                    inline_fact_check=bool(self.step5_inline_fact_check),
                )
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
