"""
Two-Pass L200 Platform Battlecard Generation.

Replaces single-pass generation (which fails on large platform battlecards due
to token limits) with a two-pass architecture:

  Pass 1 (Planning): Generate skeleton key differentiators — name, description,
      category, rank — for all product categories in one call.  Save to UC
      immediately so they are visible in the review app.

  Pass 2 (Detail): Fill in full details (headlines, details, ratings, reasoning,
      citations, sources) for each key diff via individual concurrent model calls.
      Uses a session ID to link both passes.

Usage:
    # Full run (both passes)
    python -m scripts.generate_l200_twopass \
        --competitor "Microsoft Fabric" \
        --product-area "Data Platform" \
        --prompt-dir context-and-prompts/prompts \
        --product-categories context-and-prompts/context/platform_product_categories.md \
        --directives context-and-prompts/context/fabric/DIRECTIVE.md

    # Pass 1 only (skeleton)
    python -m scripts.generate_l200_twopass --pass1-only [same args]

    # Pass 2 only (fill details for existing battlecard)
    python -m scripts.generate_l200_twopass --pass2-only --battlecard-id <UUID>

    # Control concurrency
    python -m scripts.generate_l200_twopass --max-workers 5 [same args]

Environment variables:
    DATABRICKS_PROFILE  - Databricks CLI profile (default: fe-vm-pmt)
    MODEL_NAME          - Model serving endpoint (default: databricks-claude-sonnet-4)
    UC_CATALOG          - Unity Catalog catalog (default: pm_technical)
    UC_SCHEMA           - Unity Catalog schema (default: battlecards)
    PROMPT_CATALOG      - Prompt registry catalog (default: pm_technical)
    PROMPT_SCHEMA       - Prompt registry schema (default: battlecards)
"""

import argparse
import hashlib
import json
import os
import re
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.genai
from databricks import sql
from databricks.sdk import WorkspaceClient

# Reuse helpers from the single-pass script
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.linked_ids import GenerationIDs, new_id
from scripts.generate_l200 import (
    get_openai_client,
    get_sql_connection,
    get_next_version,
    classify_context_file,
    load_context_files,
    format_context_for_prompt,
    register_prompt_from_file,
    call_model,
    parse_slides,
    save_context_documents,
    _escape,
)


# ---------------------------------------------------------------------------
# JSON Schemas for structured output
# ---------------------------------------------------------------------------

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

# Citation item schema (reused across all four citation fields)
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
                    "databricks_details": {
                        "type": "array",
                        "items": _CITATION_ITEM_SCHEMA,
                    },
                    "databricks_reasoning": {
                        "type": "array",
                        "items": _CITATION_ITEM_SCHEMA,
                    },
                    "competitor_details": {
                        "type": "array",
                        "items": _CITATION_ITEM_SCHEMA,
                    },
                    "competitor_reasoning": {
                        "type": "array",
                        "items": _CITATION_ITEM_SCHEMA,
                    },
                },
                "required": [
                    "databricks_details", "databricks_reasoning",
                    "competitor_details", "competitor_reasoning",
                ],
            },
            "sources": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "type": {"type": "string"},
                        "accessed_at": {"type": "string"},
                    },
                    "required": ["index", "title", "url", "type", "accessed_at"],
                },
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

# Pass 3 Update schema: complete slide object (skeleton + detail fields)
_CITATIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "databricks_details": {
            "type": "array",
            "items": _CITATION_ITEM_SCHEMA,
        },
        "databricks_reasoning": {
            "type": "array",
            "items": _CITATION_ITEM_SCHEMA,
        },
        "competitor_details": {
            "type": "array",
            "items": _CITATION_ITEM_SCHEMA,
        },
        "competitor_reasoning": {
            "type": "array",
            "items": _CITATION_ITEM_SCHEMA,
        },
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

L200_PASS3_JSON_SCHEMA = {
    "name": "l200_slide_update",
    "schema": {
        "type": "object",
        "properties": {
            # Skeleton fields
            "key_differentiator": {"type": "string"},
            "description": {"type": "string"},
            "databricks_rating": {"type": "string"},
            "competitor_rating": {"type": "string"},
            "selection_reasoning": {"type": "string"},
            "rank_reasoning": {"type": "string"},
            "directive_alignment": {"type": "string"},
            # Detail fields
            "databricks_headline": {"type": "string"},
            "databricks_details": {"type": "string"},
            "databricks_reasoning": {"type": "string"},
            "competitor_headline": {"type": "string"},
            "competitor_details": {"type": "string"},
            "competitor_reasoning": {"type": "string"},
            "citations": _CITATIONS_SCHEMA,
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
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_unicode(obj):
    """Recursively sanitize unicode characters in a dict/list structure.

    LLMs sometimes produce curly/smart quotes (\u201c \u201d \u2018 \u2019)
    and other non-ASCII punctuation inside string values.  These cause
    MALFORMED_RECORD_IN_PARSING errors when passed to Databricks SQL
    PARSE_JSON().  We must fix them at the Python object level (before
    json.dumps) so that json.dumps properly escapes the replacements.
    """
    if isinstance(obj, str):
        return (
            obj
            .replace("\u201c", '"')   # left double curly quote
            .replace("\u201d", '"')   # right double curly quote
            .replace("\u2018", "'")   # left single curly quote
            .replace("\u2019", "'")   # right single curly quote
            .replace("\u2013", "-")   # en dash
            .replace("\u2014", "-")   # em dash
        )
    elif isinstance(obj, dict):
        return {k: _sanitize_unicode(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_unicode(item) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# Prompt loading helpers
# ---------------------------------------------------------------------------

def load_prompt_template(prompt_dir: str, filename: str) -> str:
    """Load a prompt template from a file."""
    path = os.path.join(prompt_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path) as f:
        return f.read()


def render_pass1_prompt(
    template: str,
    competitor: str,
    product_area: str,
    product_categories_text: str,
    directives: str,
    context: str,
    diffs_per_category: int,
    total_diffs: int,
) -> str:
    """Render the Pass 1 planning prompt with variables."""
    rendered = template.replace("{{competitor}}", competitor)
    rendered = rendered.replace("{{product_area}}", product_area)
    rendered = rendered.replace("{{comparison}}", f"Databricks vs {competitor}")
    rendered = rendered.replace("{{product_categories}}", product_categories_text or "No product categories provided.")
    rendered = rendered.replace("{{directives}}", directives or "No specific directives provided.")
    rendered = rendered.replace("{{context}}", context or "No additional context provided.")
    rendered = rendered.replace("{{diffs_per_category}}", str(diffs_per_category))
    rendered = rendered.replace("{{total_diffs}}", str(total_diffs))
    return rendered


def render_pass2_prompt(
    template: str,
    competitor: str,
    category: str,
    key_differentiator: str,
    description: str,
    databricks_rating: str,
    competitor_rating: str,
    selection_reasoning: str,
    directives: str,
    context: str,
) -> str:
    """Render the Pass 2 detail prompt for a single differentiator."""
    rendered = template.replace("{{competitor}}", competitor)
    rendered = rendered.replace("{{category}}", category)
    rendered = rendered.replace("{{key_differentiator}}", key_differentiator)
    rendered = rendered.replace("{{description}}", description)
    rendered = rendered.replace("{{databricks_rating}}", databricks_rating)
    rendered = rendered.replace("{{competitor_rating}}", competitor_rating)
    rendered = rendered.replace("{{selection_reasoning}}", selection_reasoning)
    rendered = rendered.replace("{{directives}}", directives or "No specific directives provided.")
    rendered = rendered.replace("{{context}}", context or "No additional context provided.")
    return rendered


def render_pass3_prompt(
    template: str,
    competitor: str,
    category: str,
    current_content: str,
    feedback: str,
    directives: str,
    context: str,
) -> str:
    """Render the Pass 3 update prompt with current content and feedback."""
    rendered = template.replace("{{competitor}}", competitor)
    rendered = rendered.replace("{{category}}", category)
    rendered = rendered.replace("{{current_content}}", current_content)
    rendered = rendered.replace("{{feedback}}", feedback)
    rendered = rendered.replace("{{directives}}", directives or "No specific directives provided.")
    rendered = rendered.replace("{{context}}", context or "No additional context provided.")
    return rendered


# ---------------------------------------------------------------------------
# UC storage helpers
# ---------------------------------------------------------------------------

def save_generation_twopass(
    conn, catalog: str, schema: str, ids: GenerationIDs,
    generation_pass: str, **kwargs
):
    """Save a generation record with generation_pass column."""
    prefix = f"{catalog}.{schema}"

    rendered = kwargs.get("prompt_rendered", "")
    raw = kwargs.get("raw_response", "")
    slide_count = kwargs.get("slide_count", 0)
    context_json = kwargs.get("generation_context")

    context_sql = "NULL"
    if context_json:
        context_sql = f"PARSE_JSON('{_escape(json.dumps(_sanitize_unicode(context_json), ensure_ascii=True))}')"

    session_id_sql = f"'{_escape(ids.session_id)}'" if ids.session_id else "NULL"

    query = f"""
    INSERT INTO {prefix}.battlecard_generations
    (battlecard_id, battlecard_version_id, prompt_name, prompt_version,
     experiment_id, agent_run_id, competitor, product_area, product_category,
     model_name, slide_count, prompt_rendered, raw_response,
     generation_context, generated_at, session_id, generation_pass)
    VALUES (
        '{ids.battlecard_id}',
        {ids.battlecard_version_id},
        '{_escape(ids.prompt_name)}',
        {ids.prompt_version},
        '{ids.experiment_id}',
        '{ids.agent_run_id}',
        '{_escape(ids.competitor)}',
        '{_escape(ids.product_area)}',
        '{_escape(kwargs.get("product_category", ""))}',
        '{_escape(ids.model_name)}',
        {slide_count},
        '{_escape(rendered)}',
        '{_escape(raw)}',
        {context_sql},
        '{ids.generated_at.isoformat()}',
        {session_id_sql},
        '{_escape(generation_pass)}'
    )
    """
    with conn.cursor() as cursor:
        cursor.execute(query)


def save_skeleton_slides(
    conn, catalog: str, schema: str, ids: GenerationIDs,
    slides: List[Dict[str, Any]],
):
    """Save skeleton slides with detail_status = 'skeleton'."""
    prefix = f"{catalog}.{schema}"

    with conn.cursor() as cursor:
        for i, slide in enumerate(slides):
            slide_id = slide.get("id", f"slide_{i}")
            slide_type = "l200_critical_differentiator"
            category = slide.get("category", "")
            slide_title = slide.get("key_differentiator", "")
            db_rating = slide.get("databricks_rating", "")
            comp_rating = slide.get("competitor_rating", "")
            content_json = _escape(json.dumps(_sanitize_unicode(slide), ensure_ascii=True))

            query = f"""
            INSERT INTO {prefix}.battlecard_slides
            (slide_id, slide_order, slide_type,
             battlecard_id, battlecard_version_id,
             agent_run_id, prompt_name, prompt_version,
             competitor, product_area, category, slide_title,
             slide_content, databricks_rating, competitor_rating,
             generated_at, detail_status)
            VALUES (
                '{_escape(slide_id)}',
                {i + 1},
                '{_escape(slide_type)}',
                '{ids.battlecard_id}',
                {ids.battlecard_version_id},
                '{ids.agent_run_id}',
                '{_escape(ids.prompt_name)}',
                {ids.prompt_version},
                '{_escape(ids.competitor)}',
                '{_escape(ids.product_area)}',
                '{_escape(category)}',
                '{_escape(slide_title)}',
                PARSE_JSON('{content_json}'),
                '{_escape(db_rating)}',
                '{_escape(comp_rating)}',
                '{ids.generated_at.isoformat()}',
                'skeleton'
            )
            """
            cursor.execute(query)


def update_slide_status(conn, catalog: str, schema: str,
                        slide_id: str, battlecard_id: str,
                        status: str):
    """Update the detail_status of a slide."""
    prefix = f"{catalog}.{schema}"
    query = f"""
    UPDATE {prefix}.battlecard_slides
    SET detail_status = '{_escape(status)}'
    WHERE slide_id = '{_escape(slide_id)}'
      AND battlecard_id = '{_escape(battlecard_id)}'
    """
    with conn.cursor() as cursor:
        cursor.execute(query)


def update_slide_with_details(
    conn, catalog: str, schema: str,
    slide_id: str, battlecard_id: str,
    full_content: Dict[str, Any],
    db_rating: str, comp_rating: str,
):
    """Update a skeleton slide with full detail content."""
    prefix = f"{catalog}.{schema}"
    sanitized = _sanitize_unicode(full_content)
    content_json = json.dumps(sanitized, ensure_ascii=True)
    escaped_json = _escape(content_json)
    query = f"""
    UPDATE {prefix}.battlecard_slides
    SET slide_content = PARSE_JSON('{escaped_json}'),
        detail_status = 'complete',
        databricks_rating = '{_escape(db_rating)}',
        competitor_rating = '{_escape(comp_rating)}'
    WHERE slide_id = '{_escape(slide_id)}'
      AND battlecard_id = '{_escape(battlecard_id)}'
    """
    with conn.cursor() as cursor:
        cursor.execute(query)


def load_skeleton_slides_from_uc(
    conn, catalog: str, schema: str, battlecard_id: str,
) -> List[Dict[str, Any]]:
    """Load skeleton slides from UC for Pass 2 processing."""
    prefix = f"{catalog}.{schema}"
    query = f"""
    SELECT slide_id, slide_order, category, slide_title,
           slide_content, databricks_rating, competitor_rating,
           detail_status
    FROM {prefix}.battlecard_slides
    WHERE battlecard_id = '{_escape(battlecard_id)}'
    ORDER BY slide_order
    """
    with conn.cursor() as cursor:
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

    slides = []
    for row in rows:
        row_dict = dict(zip(columns, row))
        content = row_dict["slide_content"]
        if isinstance(content, str):
            content = json.loads(content)
        slides.append({
            "slide_id": row_dict["slide_id"],
            "slide_order": row_dict["slide_order"],
            "detail_status": row_dict.get("detail_status", "skeleton"),
            **content,
        })
    return slides


def load_generation_info(conn, catalog: str, schema: str, battlecard_id: str) -> Dict:
    """Load generation metadata for an existing battlecard."""
    prefix = f"{catalog}.{schema}"
    query = f"""
    SELECT battlecard_id, battlecard_version_id, competitor, product_area,
           model_name, prompt_name, prompt_version, experiment_id, agent_run_id,
           session_id
    FROM {prefix}.battlecard_generations
    WHERE battlecard_id = '{_escape(battlecard_id)}'
    ORDER BY generated_at DESC
    LIMIT 1
    """
    with conn.cursor() as cursor:
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        row = cursor.fetchone()

    if not row:
        raise ValueError(f"No generation found for battlecard_id: {battlecard_id}")
    return dict(zip(columns, row))


def save_session(conn, catalog: str, schema: str, ids: GenerationIDs,
                 status: str, conversation_history: Dict):
    """Create or update a session in battlecard_agent_sessions."""
    prefix = f"{catalog}.{schema}"
    now = datetime.utcnow().isoformat()
    history_json = _escape(json.dumps(conversation_history))

    query = f"""
    MERGE INTO {prefix}.battlecard_agent_sessions AS target
    USING (SELECT '{_escape(ids.session_id)}' AS session_id) AS source
    ON target.session_id = source.session_id
    WHEN MATCHED THEN
        UPDATE SET
            status = '{_escape(status)}',
            conversation_history = '{history_json}',
            updated_at = '{now}',
            battlecard_id = '{ids.battlecard_id}',
            agent_run_id = '{ids.agent_run_id}'
    WHEN NOT MATCHED THEN
        INSERT (session_id, battlecard_id, agent_run_id, competitor, product_area,
                status, message_count, conversation_history, created_at, updated_at)
        VALUES (
            '{_escape(ids.session_id)}',
            '{ids.battlecard_id}',
            '{ids.agent_run_id}',
            '{_escape(ids.competitor)}',
            '{_escape(ids.product_area)}',
            '{_escape(status)}',
            0,
            '{history_json}',
            '{now}',
            '{now}'
        )
    """
    with conn.cursor() as cursor:
        cursor.execute(query)


def load_session(conn, catalog: str, schema: str,
                 battlecard_id: str, session_id: str) -> Optional[Dict]:
    """Load a session from battlecard_agent_sessions."""
    prefix = f"{catalog}.{schema}"
    query = f"""
    SELECT session_id, battlecard_id, status, conversation_history
    FROM {prefix}.battlecard_agent_sessions
    WHERE battlecard_id = '{_escape(battlecard_id)}'
       OR session_id = '{_escape(session_id)}'
    ORDER BY updated_at DESC
    LIMIT 1
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            row = cursor.fetchone()
            if row:
                result = dict(zip(columns, row))
                # Parse conversation_history from string if needed
                ch = result.get("conversation_history", "{}")
                if isinstance(ch, str):
                    try:
                        result["conversation_history"] = json.loads(ch)
                    except (json.JSONDecodeError, TypeError):
                        result["conversation_history"] = {}
                return result
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Pass 3 helpers
# ---------------------------------------------------------------------------

def load_reviews_for_battlecard(
    conn, catalog: str, schema: str, battlecard_id: str,
) -> Dict[str, Dict[str, Any]]:
    """Load reviews for a battlecard, grouped by slide_id with latest review per scope.

    Returns: {slide_id: {key_diff: {status, comment}, databricks: {...}, fabric: {...}}}
    """
    prefix = f"{catalog}.{schema}"
    query = f"""
    SELECT slide_id, review_scope, status, comment, reviewed_at
    FROM {prefix}.battlecard_reviews
    WHERE battlecard_id = '{_escape(battlecard_id)}'
    ORDER BY reviewed_at DESC
    """
    with conn.cursor() as cursor:
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

    feedback: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        r = dict(zip(columns, row))
        slide_id = r["slide_id"]
        scope = r["review_scope"]
        if not slide_id or scope == "reorder":
            continue

        if slide_id not in feedback:
            feedback[slide_id] = {}

        # Keep only the most recent review per scope (first row = newest)
        if scope not in feedback[slide_id]:
            feedback[slide_id][scope] = {
                "status": r["status"],
                "comment": r["comment"] or "",
            }

    return feedback


def update_slide_from_feedback(
    conn, catalog: str, schema: str,
    slide_id: str, battlecard_id: str,
    full_content: Dict[str, Any],
    db_rating: str, comp_rating: str,
):
    """Update slide with regenerated content and increment update_count."""
    prefix = f"{catalog}.{schema}"
    sanitized = _sanitize_unicode(full_content)
    content_json = json.dumps(sanitized, ensure_ascii=True)
    escaped_json = _escape(content_json)
    slide_title = _escape(full_content.get("key_differentiator", ""))

    query = f"""
    UPDATE {prefix}.battlecard_slides
    SET slide_content = PARSE_JSON('{escaped_json}'),
        detail_status = 'complete',
        databricks_rating = '{_escape(db_rating)}',
        competitor_rating = '{_escape(comp_rating)}',
        slide_title = '{slide_title}',
        update_count = COALESCE(update_count, 0) + 1
    WHERE slide_id = '{_escape(slide_id)}'
      AND battlecard_id = '{_escape(battlecard_id)}'
    """
    with conn.cursor() as cursor:
        cursor.execute(query)


# ---------------------------------------------------------------------------
# Pass 3 worker
# ---------------------------------------------------------------------------

def process_single_update(
    openai_client,
    model_name: str,
    pass3_template: str,
    slide: Dict[str, Any],
    feedback: Dict[str, Any],
    competitor: str,
    directives: str,
    context_text: str,
    conn,
    catalog: str,
    schema: str,
    battlecard_id: str,
    dry_run: bool,
    no_structured_output: bool,
) -> Dict[str, Any]:
    """Process a single slide update in Pass 3: regenerate based on feedback.

    Returns a result dict with status and info.
    """
    slide_id = slide.get("slide_id", slide.get("id", "unknown"))
    category = slide.get("category", "")
    key_diff = slide.get("key_differentiator", "")

    result = {
        "slide_id": slide_id,
        "category": category,
        "key_differentiator": key_diff,
        "status": "pending",
    }

    try:
        # Mark as generating
        if not dry_run and conn:
            update_slide_status(conn, catalog, schema, slide_id, battlecard_id, "generating")

        # Build current content JSON (exclude internal fields)
        current_content = {k: v for k, v in slide.items()
                          if k not in ("slide_id", "slide_order", "detail_status")}
        current_content_json = json.dumps(_sanitize_unicode(current_content), indent=2, ensure_ascii=True)
        feedback_json = json.dumps(feedback, indent=2, ensure_ascii=True)

        # Render prompt
        rendered = render_pass3_prompt(
            template=pass3_template,
            competitor=competitor,
            category=category,
            current_content=current_content_json,
            feedback=feedback_json,
            directives=directives,
            context=context_text,
        )

        # Call model
        json_schema = None if no_structured_output else L200_PASS3_JSON_SCHEMA
        raw_response = call_model(
            client=openai_client,
            model_name=model_name,
            rendered_prompt=rendered,
            json_schema=json_schema,
            max_tokens=8192,
        )

        # Parse response
        text = raw_response.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```\s*$", "", text)

        updated_slide = json.loads(text)

        # Preserve fields from original that aren't in the update
        full_content = {**current_content}
        full_content.update(updated_slide)
        # Add feedback stub
        full_content["feedback"] = {
            "key_diff": {},
            "databricks": {},
            "competitor": {},
            "reorder": None,
        }

        # Update UC
        if not dry_run and conn:
            update_slide_from_feedback(
                conn, catalog, schema,
                slide_id=slide_id,
                battlecard_id=battlecard_id,
                full_content=full_content,
                db_rating=updated_slide.get("databricks_rating", ""),
                comp_rating=updated_slide.get("competitor_rating", ""),
            )

        result["status"] = "complete"
        result["updated_slide"] = updated_slide
        result["full_content"] = full_content
        result["raw_response"] = raw_response
        result["rendered_prompt"] = rendered

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

        # Mark as failed in UC
        if not dry_run and conn:
            try:
                update_slide_status(conn, catalog, schema, slide_id, battlecard_id, "failed")
            except Exception:
                pass

    return result


# ---------------------------------------------------------------------------
# Pass 2 worker
# ---------------------------------------------------------------------------

def process_single_diff(
    openai_client,
    model_name: str,
    pass2_template: str,
    skeleton: Dict[str, Any],
    competitor: str,
    directives: str,
    context_text: str,
    conn,
    catalog: str,
    schema: str,
    battlecard_id: str,
    dry_run: bool,
    no_structured_output: bool,
) -> Dict[str, Any]:
    """Process a single differentiator in Pass 2: generate details and update UC.

    Returns a result dict with status and info.
    """
    slide_id = skeleton.get("id", skeleton.get("slide_id", "unknown"))
    category = skeleton.get("category", "")
    key_diff = skeleton.get("key_differentiator", "")

    result = {
        "slide_id": slide_id,
        "category": category,
        "key_differentiator": key_diff,
        "status": "pending",
    }

    try:
        # Mark as generating
        if not dry_run and conn:
            update_slide_status(conn, catalog, schema, slide_id, battlecard_id, "generating")

        # Render prompt
        rendered = render_pass2_prompt(
            template=pass2_template,
            competitor=competitor,
            category=category,
            key_differentiator=key_diff,
            description=skeleton.get("description", ""),
            databricks_rating=skeleton.get("databricks_rating", ""),
            competitor_rating=skeleton.get("competitor_rating", ""),
            selection_reasoning=skeleton.get("selection_reasoning", ""),
            directives=directives,
            context=context_text,
        )

        # Call model
        json_schema = None if no_structured_output else L200_PASS2_JSON_SCHEMA
        raw_response = call_model(
            client=openai_client,
            model_name=model_name,
            rendered_prompt=rendered,
            json_schema=json_schema,
            max_tokens=8192,
        )

        # Parse response
        text = raw_response.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```\s*$", "", text)

        detail = json.loads(text)

        # Merge skeleton + detail into full slide content
        full_content = {**skeleton}
        # Remove UC-internal fields
        full_content.pop("slide_id", None)
        full_content.pop("slide_order", None)
        full_content.pop("detail_status", None)
        # Overlay detail fields
        full_content.update(detail)
        # Add feedback stub
        full_content["feedback"] = {
            "key_diff": {},
            "databricks": {},
            "competitor": {},
            "reorder": None,
        }

        # Update UC
        if not dry_run and conn:
            update_slide_with_details(
                conn, catalog, schema,
                slide_id=slide_id,
                battlecard_id=battlecard_id,
                full_content=full_content,
                db_rating=skeleton.get("databricks_rating", ""),
                comp_rating=skeleton.get("competitor_rating", ""),
            )

        result["status"] = "complete"
        result["detail"] = detail
        result["full_content"] = full_content
        result["raw_response"] = raw_response
        result["rendered_prompt"] = rendered

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

        # Mark as failed in UC
        if not dry_run and conn:
            try:
                update_slide_status(conn, catalog, schema, slide_id, battlecard_id, "failed")
            except Exception:
                pass

    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Two-pass L200 platform battlecard generation")
    parser.add_argument("--competitor", required=True, help="Competitor name")
    parser.add_argument("--product-area", required=True, help="Product area")
    parser.add_argument("--prompt-dir", default="context-and-prompts/prompts",
                        help="Directory containing pass1 and pass2 prompt files")
    parser.add_argument("--product-categories", default="",
                        help="Path to product categories .md file")
    parser.add_argument("--directives", default="", help="Directives text or path to .md file")
    parser.add_argument("--context", action="append", default=[],
                        help="Context file paths (repeatable)")
    parser.add_argument("--diffs-per-category", type=int, default=10,
                        help="Number of key differentiators per category (default: 10)")
    parser.add_argument("--pass1-only", action="store_true",
                        help="Only run Pass 1 (skeleton generation)")
    parser.add_argument("--pass2-only", action="store_true",
                        help="Only run Pass 2 (requires --battlecard-id)")
    parser.add_argument("--update", action="store_true",
                        help="Run Pass 3: update flagged slides based on feedback (requires --battlecard-id)")
    parser.add_argument("--slide-ids", default="",
                        help="Comma-separated slide IDs to update (optional, defaults to all flagged)")
    parser.add_argument("--battlecard-id", default="",
                        help="Existing battlecard ID for Pass 2 only or --update mode")
    parser.add_argument("--only-categories", default="",
                        help="Comma-separated product category names to process")
    parser.add_argument("--max-workers", type=int, default=5,
                        help="ThreadPoolExecutor concurrency for Pass 2 (default: 5)")
    parser.add_argument("--no-structured-output", action="store_true",
                        help="Disable JSON schema structured output enforcement")
    parser.add_argument("--dry-run", action="store_true", help="Skip UC writes, print output")
    parser.add_argument("--pass2-prompt", default="l200_pass2_detail_v1.md",
                        help="Pass 2 prompt filename (default: l200_pass2_detail_v1.md)")
    args = parser.parse_args()

    if args.pass2_only and not args.battlecard_id:
        parser.error("--pass2-only requires --battlecard-id")
    if args.update and not args.battlecard_id:
        parser.error("--update requires --battlecard-id")

    # -----------------------------------------------------------------------
    # Config
    # -----------------------------------------------------------------------
    profile = os.getenv("DATABRICKS_PROFILE", "fe-vm-pmt")
    model_name = os.getenv("MODEL_NAME", "databricks-claude-sonnet-4")
    uc_catalog = os.getenv("UC_CATALOG", "pm_technical")
    uc_schema = os.getenv("UC_SCHEMA", "battlecards")
    prompt_catalog = os.getenv("PROMPT_CATALOG", "pm_technical")
    prompt_schema = os.getenv("PROMPT_SCHEMA", "battlecards")

    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "/Users/tahir.fayyaz@databricks.com/l200-generation",
    )

    # Session ID links Pass 1 and Pass 2
    session_id = str(uuid.uuid4())

    # -----------------------------------------------------------------------
    # Load directives
    # -----------------------------------------------------------------------
    directives = args.directives
    directives_file_info = None
    if directives and os.path.isfile(directives):
        directives_path = directives
        with open(directives) as f:
            directives = f.read()
        filename = os.path.basename(directives_path)
        directives_file_info = {
            "name": filename,
            "path": directives_path,
            "content": directives,
            "content_hash": hashlib.sha256(directives.encode()).hexdigest(),
            "source_type": "human_provided",
            **classify_context_file(filename),
        }
        if directives_file_info["doc_type"] == "general_context":
            directives_file_info["doc_type"] = "competitive_directive"
            directives_file_info["xml_tag"] = "competitive_directive"
        print(f"  Loaded directives: {directives_path}")

    # -----------------------------------------------------------------------
    # Load product categories
    # -----------------------------------------------------------------------
    product_categories_text = ""
    num_categories = 0
    product_categories_file_info = None
    if args.product_categories and os.path.isfile(args.product_categories):
        with open(args.product_categories) as f:
            product_categories_text = f.read()
        pc_filename = os.path.basename(args.product_categories)
        product_categories_file_info = {
            "name": pc_filename,
            "path": args.product_categories,
            "content": product_categories_text,
            "content_hash": hashlib.sha256(product_categories_text.encode()).hexdigest(),
            "source_type": "human_provided",
            **classify_context_file(pc_filename),
        }
        category_lines = set()
        for line in product_categories_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                category_lines.add(stripped.lstrip("- ").strip())
        num_categories = len(category_lines)
        print(f"  Loaded product categories: {pc_filename} ({num_categories} categories)")

    total_diffs = num_categories * args.diffs_per_category if num_categories else 0

    # -----------------------------------------------------------------------
    # Load context files
    # -----------------------------------------------------------------------
    context_files = load_context_files(args.context)
    context_text = format_context_for_prompt(context_files)

    all_context_docs = list(context_files)
    if directives_file_info:
        all_context_docs.insert(0, directives_file_info)
    if product_categories_file_info:
        all_context_docs.append(product_categories_file_info)

    # -----------------------------------------------------------------------
    # Setup MLflow
    # -----------------------------------------------------------------------
    os.environ.setdefault("DATABRICKS_CONFIG_PROFILE", profile)
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")
    mlflow.set_experiment(experiment_name)
    mlflow.openai.autolog()

    # -----------------------------------------------------------------------
    # Initialize clients
    # -----------------------------------------------------------------------
    openai_client = get_openai_client(profile)

    conn = None
    if not args.dry_run:
        conn = get_sql_connection(profile)

    # -----------------------------------------------------------------------
    # Handle Pass 2 only mode
    # -----------------------------------------------------------------------
    if args.pass2_only:
        print(f"\n{'='*60}")
        print(f"  PASS 2 ONLY — Filling details for {args.battlecard_id}")
        print(f"{'='*60}\n")

        gen_info = load_generation_info(conn, uc_catalog, uc_schema, args.battlecard_id)
        skeleton_slides = load_skeleton_slides_from_uc(conn, uc_catalog, uc_schema, args.battlecard_id)

        # Filter to skeleton/failed slides only
        pending_slides = [
            s for s in skeleton_slides
            if s.get("detail_status") in ("skeleton", "failed", None)
        ]

        if args.only_categories:
            category_filter = {
                c.strip().lower() for c in args.only_categories.split(",") if c.strip()
            }
            if category_filter:
                before_count = len(pending_slides)
                pending_slides = [
                    s for s in pending_slides
                    if (s.get("category") or "").strip().lower() in category_filter
                ]
                print(f"  Category filter: {sorted(category_filter)}")
                print(f"  Pending after filter: {len(pending_slides)} (was {before_count})")
        print(f"  Total slides: {len(skeleton_slides)}")
        print(f"  Pending (skeleton/failed): {len(pending_slides)}")

        if not pending_slides:
            print("  No pending slides to process. Exiting.")
            return

        # Use existing session_id or generate new one
        session_id = gen_info.get("session_id") or session_id
        competitor = gen_info.get("competitor", args.competitor)

        # Create IDs for pass 2
        ids = GenerationIDs(
            battlecard_id=args.battlecard_id,
            battlecard_version_id=gen_info.get("battlecard_version_id", 1),
            prompt_name=gen_info.get("prompt_name", ""),
            prompt_version=gen_info.get("prompt_version", 0),
            model_name=model_name,
            competitor=competitor,
            product_area=gen_info.get("product_area", args.product_area),
            session_id=session_id,
        )

        # Load pass2 prompt
        pass2_template = load_prompt_template(args.prompt_dir, args.pass2_prompt)

        _run_pass2(
            args=args,
            ids=ids,
            openai_client=openai_client,
            model_name=model_name,
            pass2_template=pass2_template,
            pending_slides=pending_slides,
            competitor=competitor,
            directives=directives,
            context_text=context_text,
            conn=conn,
            uc_catalog=uc_catalog,
            uc_schema=uc_schema,
            session_id=session_id,
        )

        if conn:
            conn.close()
        return

    # -----------------------------------------------------------------------
    # Handle Pass 3 Update mode
    # -----------------------------------------------------------------------
    if args.update:
        print(f"\n{'='*60}")
        print(f"  PASS 3: UPDATE — Regenerating flagged slides for {args.battlecard_id}")
        print(f"{'='*60}\n")

        gen_info = load_generation_info(conn, uc_catalog, uc_schema, args.battlecard_id)
        all_slides = load_skeleton_slides_from_uc(conn, uc_catalog, uc_schema, args.battlecard_id)
        reviews = load_reviews_for_battlecard(conn, uc_catalog, uc_schema, args.battlecard_id)

        # Filter to slides with needs_revision or rejected feedback at any level
        flagged_statuses = {"needs_revision", "rejected"}
        flagged_slides = []
        flagged_feedback = {}
        for slide in all_slides:
            sid = slide.get("slide_id", slide.get("id", ""))
            slide_reviews = reviews.get(sid, {})
            is_flagged = any(
                slide_reviews.get(scope, {}).get("status") in flagged_statuses
                for scope in ("key_diff", "databricks", "fabric", "competitor")
            )
            if is_flagged:
                flagged_slides.append(slide)
                flagged_feedback[sid] = slide_reviews

        # Optionally filter by --slide-ids
        if args.slide_ids:
            requested_ids = set(s.strip() for s in args.slide_ids.split(",") if s.strip())
            flagged_slides = [s for s in flagged_slides
                              if s.get("slide_id", s.get("id", "")) in requested_ids]
            flagged_feedback = {sid: fb for sid, fb in flagged_feedback.items()
                                if sid in requested_ids}

        print(f"  Total slides: {len(all_slides)}")
        print(f"  Flagged slides: {len(flagged_slides)}")

        if not flagged_slides:
            print("  No flagged slides to update. Exiting.")
            if conn:
                conn.close()
            return

        # Use existing session_id or generate new one
        session_id = gen_info.get("session_id") or session_id
        competitor = gen_info.get("competitor", args.competitor)

        ids = GenerationIDs(
            battlecard_id=args.battlecard_id,
            battlecard_version_id=gen_info.get("battlecard_version_id", 1),
            prompt_name=gen_info.get("prompt_name", ""),
            prompt_version=gen_info.get("prompt_version", 0),
            model_name=model_name,
            competitor=competitor,
            product_area=gen_info.get("product_area", args.product_area),
            session_id=session_id,
        )

        _run_pass3_update(
            args=args,
            ids=ids,
            openai_client=openai_client,
            model_name=model_name,
            flagged_slides=flagged_slides,
            flagged_feedback=flagged_feedback,
            competitor=competitor,
            directives=directives,
            context_text=context_text,
            conn=conn,
            uc_catalog=uc_catalog,
            uc_schema=uc_schema,
            session_id=session_id,
        )

        if conn:
            conn.close()
        return

    # -----------------------------------------------------------------------
    # Full run or Pass 1 only
    # -----------------------------------------------------------------------
    if not args.dry_run and conn:
        next_version = get_next_version(conn, uc_catalog, uc_schema,
                                        args.competitor, args.product_area)
    else:
        next_version = 1

    # Prompt names for registry
    pass1_prompt_name = f"{prompt_catalog}.{prompt_schema}.l200_pass1_planning_v1"
    pass2_prompt_basename = args.pass2_prompt.replace('.md', '')
    pass2_prompt_name = f"{prompt_catalog}.{prompt_schema}.{pass2_prompt_basename}"

    ids = GenerationIDs(
        battlecard_version_id=next_version,
        prompt_name=pass1_prompt_name,
        prompt_alias="latest",
        model_name=model_name,
        competitor=args.competitor,
        product_area=args.product_area,
        session_id=session_id,
    )

    pass_mode = "pass1" if args.pass1_only else "both"

    print(f"\n{'='*60}")
    print(f"  Two-Pass L200 Generation")
    print(f"{'='*60}")
    print(f"  Competitor:      {args.competitor}")
    print(f"  Product area:    {args.product_area}")
    print(f"  Model:           {model_name}")
    print(f"  Battlecard ID:   {ids.battlecard_id}")
    print(f"  Session ID:      {session_id}")
    print(f"  Version:         {next_version}")
    print(f"  Mode:            {pass_mode}")
    print(f"  Categories:      {num_categories} x {args.diffs_per_category} = {total_diffs} total")
    print(f"  Context files:   {len(context_files)}")
    print(f"  Structured:      {not args.no_structured_output}")
    print(f"{'='*60}\n")

    # Session tracking data
    session_history = {
        "pass1": {"status": "pending", "slide_count": 0},
        "pass2": {"status": "pending", "completed": 0, "failed": 0, "pending": 0},
    }

    # ===================================================================
    # PASS 1: Planning
    # ===================================================================
    print("=" * 60)
    print("  PASS 1: Planning — Generating skeleton differentiators")
    print("=" * 60)

    # Load and register pass1 prompt
    pass1_file = os.path.join(args.prompt_dir, "l200_pass1_planning_v1.md")
    pass1_template, pass1_version = register_prompt_from_file(
        pass1_file, pass1_prompt_name, "latest",
    )
    ids.prompt_version = pass1_version

    # Render
    rendered_pass1 = render_pass1_prompt(
        template=pass1_template,
        competitor=args.competitor,
        product_area=args.product_area,
        product_categories_text=product_categories_text,
        directives=directives,
        context=context_text,
        diffs_per_category=args.diffs_per_category,
        total_diffs=total_diffs,
    )

    # JSON schema for structured output
    pass1_schema = None if args.no_structured_output else L200_PASS1_JSON_SCHEMA

    # Build generation context
    generation_context = {
        "directives_source": args.directives or None,
        "prompt_file": pass1_file,
        "product_categories_file": args.product_categories or None,
        "num_categories": num_categories,
        "diffs_per_category": args.diffs_per_category,
        "total_diffs_requested": total_diffs,
        "generation_mode": "twopass",
        "pass": "pass1_planning",
        "session_id": session_id,
        "context_files": [
            {
                "name": c["name"],
                "path": c["path"],
                "doc_type": c["doc_type"],
                "doc_scope": c.get("doc_scope"),
                "source_type": c["source_type"],
                "content_hash": c["content_hash"],
            }
            for c in all_context_docs
        ],
        "structured_output": not args.no_structured_output,
    }

    # Run inside MLflow
    run_name = f"l200_twopass_{args.competitor}_{args.product_area}_v{next_version}"
    with mlflow.start_run(run_name=run_name) as parent_run:
        ids.agent_run_id = parent_run.info.run_id
        ids.experiment_id = parent_run.info.experiment_id

        mlflow.log_params(ids.to_mlflow_params())
        mlflow.log_params({
            "generation_mode": "twopass",
            "pass_mode": pass_mode,
            "structured_output": not args.no_structured_output,
            "context_file_count": len(context_files),
            "num_categories": num_categories,
            "diffs_per_category": args.diffs_per_category,
            "total_diffs_requested": total_diffs,
            "max_workers": args.max_workers,
        })
        mlflow.log_text(rendered_pass1, "pass1/rendered_prompt.txt")

        # Log context files as artifacts
        for ctx in context_files:
            mlflow.log_text(ctx["content"], f"context/{ctx['name']}")
        if product_categories_text:
            mlflow.log_text(product_categories_text,
                            f"context/{os.path.basename(args.product_categories)}")

        # Call model for Pass 1
        max_tokens = 16384 if total_diffs <= 30 else 32768
        print(f"\n  Calling {model_name} for Pass 1 (max_tokens={max_tokens})...")
        raw_pass1 = call_model(
            client=openai_client,
            model_name=model_name,
            rendered_prompt=rendered_pass1,
            json_schema=pass1_schema,
            max_tokens=max_tokens,
        )
        mlflow.log_text(raw_pass1, "pass1/raw_response.txt")

        # Parse skeleton slides
        print("  Parsing Pass 1 response...")
        skeleton_slides = parse_slides(raw_pass1)
        print(f"  Parsed {len(skeleton_slides)} skeleton differentiators")
        mlflow.log_metric("pass1_slide_count", len(skeleton_slides))

        # Log each skeleton slide
        for i, slide in enumerate(skeleton_slides):
            mlflow.log_dict(slide, f"pass1/slides/slide_{i+1}_{slide.get('id', 'unknown')}.json")
        mlflow.log_dict(skeleton_slides, "pass1/slides/all_skeletons.json")

        # Save to UC
        if not args.dry_run and conn:
            print("  Saving to Unity Catalog...")

            save_generation_twopass(
                conn, uc_catalog, uc_schema, ids,
                generation_pass="pass1_planning",
                prompt_rendered=rendered_pass1,
                raw_response=raw_pass1,
                slide_count=len(skeleton_slides),
                product_category="",
                generation_context=generation_context,
            )
            print(f"    Saved generation record: {ids.battlecard_id}")

            save_skeleton_slides(conn, uc_catalog, uc_schema, ids, skeleton_slides)
            print(f"    Saved {len(skeleton_slides)} skeleton slides")

            if all_context_docs:
                save_context_documents(conn, uc_catalog, uc_schema, ids, all_context_docs)
                print(f"    Saved {len(all_context_docs)} context documents")

            # Update session
            session_history["pass1"] = {
                "status": "complete",
                "slide_count": len(skeleton_slides),
                "completed_at": datetime.utcnow().isoformat(),
            }
            save_session(conn, uc_catalog, uc_schema, ids,
                         status="pass1_complete", conversation_history=session_history)

            mlflow.log_param("saved_to_uc", True)
        else:
            print("\n  Dry run — Pass 1 skeleton slides:")
            print(json.dumps(skeleton_slides, indent=2))

        # Pass 1 summary
        print(f"\n  Pass 1 Complete!")
        print(f"    Skeleton slides: {len(skeleton_slides)}")
        print(f"    Battlecard ID:   {ids.battlecard_id}")

        if args.pass1_only:
            print(f"\n  Pass 1 only mode — skipping Pass 2.")
            print(f"  To run Pass 2 later:")
            print(f"    python -m scripts.generate_l200_twopass \\")
            print(f"      --pass2-only --battlecard-id {ids.battlecard_id} \\")
            print(f"      --competitor \"{args.competitor}\" --product-area \"{args.product_area}\" \\")
            print(f"      --prompt-dir {args.prompt_dir}")
            if args.directives:
                print(f"      --directives {args.directives}")

            if conn:
                conn.close()
            return

        # ===================================================================
        # PASS 2: Detail (inside same MLflow parent run)
        # ===================================================================
        print(f"\n{'='*60}")
        print(f"  PASS 2: Detail — Filling in {len(skeleton_slides)} differentiators")
        print(f"{'='*60}")

        pass2_template = load_prompt_template(args.prompt_dir, args.pass2_prompt)

        # Register pass2 prompt
        try:
            pass2_tmpl, pass2_ver = register_prompt_from_file(
                os.path.join(args.prompt_dir, args.pass2_prompt),
                pass2_prompt_name, "latest",
            )
        except Exception as e:
            print(f"  WARNING: Could not register pass2 prompt ({e})")

        _run_pass2_in_mlflow(
            args=args,
            ids=ids,
            openai_client=openai_client,
            model_name=model_name,
            pass2_template=pass2_template,
            skeleton_slides=skeleton_slides,
            competitor=args.competitor,
            directives=directives,
            context_text=context_text,
            conn=conn,
            uc_catalog=uc_catalog,
            uc_schema=uc_schema,
            session_id=session_id,
            session_history=session_history,
        )

    if conn:
        conn.close()


def _run_pass3_update(
    args, ids, openai_client, model_name,
    flagged_slides, flagged_feedback, competitor, directives, context_text,
    conn, uc_catalog, uc_schema, session_id,
):
    """Run Pass 3: update flagged slides based on reviewer feedback."""
    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "/Users/tahir.fayyaz@databricks.com/l200-generation",
    )
    profile = os.getenv("DATABRICKS_PROFILE", "fe-vm-pmt")
    os.environ.setdefault("DATABRICKS_CONFIG_PROFILE", profile)
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")
    mlflow.set_experiment(experiment_name)

    # Load pass3 prompt
    pass3_template = load_prompt_template(args.prompt_dir, "l200_pass3_update_v1.md")

    # Register prompt in MLflow
    prompt_catalog = os.getenv("DATABRICKS_PROMPT_CATALOG", uc_catalog)
    prompt_schema = os.getenv("DATABRICKS_PROMPT_SCHEMA", uc_schema)
    pass3_prompt_name = f"{prompt_catalog}.{prompt_schema}.l200_pass3_update_v1"
    try:
        registered_pass3 = mlflow.register_prompt(
            name=pass3_prompt_name,
            template=pass3_template,
        )
        print(f"  Registered prompt: {pass3_prompt_name} v{registered_pass3.version}")
    except Exception as e:
        print(f"  Warning: Could not register pass3 prompt: {e}")

    run_name = f"l200_pass3_update_{competitor}_{ids.product_area}_v{ids.battlecard_version_id}"
    with mlflow.start_run(run_name=run_name) as parent_run:
        ids.agent_run_id = parent_run.info.run_id
        ids.experiment_id = parent_run.info.experiment_id

        mlflow.log_params(ids.to_mlflow_params())
        mlflow.log_params({
            "generation_mode": "twopass",
            "pass_mode": "pass3_update",
            "max_workers": args.max_workers,
            "flagged_slides": len(flagged_slides),
        })

        completed = 0
        failed = 0
        slide_ids_updated = []

        print(f"\n  PASS 3: Updating {len(flagged_slides)} flagged slides...")
        print(f"  Workers: {args.max_workers}")
        print(f"  Model: {model_name}\n")

        futures = {}
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            for slide in flagged_slides:
                sid = slide.get("slide_id", slide.get("id", ""))
                fb = flagged_feedback.get(sid, {})
                future = executor.submit(
                    process_single_update,
                    openai_client=openai_client,
                    model_name=model_name,
                    pass3_template=pass3_template,
                    slide=slide,
                    feedback=fb,
                    competitor=competitor,
                    directives=directives,
                    context_text=context_text,
                    conn=conn,
                    catalog=uc_catalog,
                    schema=uc_schema,
                    battlecard_id=args.battlecard_id,
                    dry_run=args.dry_run,
                    no_structured_output=args.no_structured_output,
                )
                futures[future] = sid

            for future in as_completed(futures):
                sid = futures[future]
                try:
                    result = future.result()
                    status = result.get("status", "failed")
                    key_diff = result.get("key_differentiator", "")
                    if status == "complete":
                        completed += 1
                        slide_ids_updated.append(sid)
                        print(f"    [{completed + failed}/{len(flagged_slides)}] "
                              f"UPDATED: {key_diff[:40]}")

                        # Log to MLflow child run
                        try:
                            with mlflow.start_run(
                                run_name=f"update_{key_diff[:30]}",
                                nested=True,
                            ) as child_run:
                                mlflow.log_params({
                                    "slide_id": sid,
                                    "key_differentiator": key_diff[:200],
                                    "category": result.get("category", ""),
                                })
                                if result.get("rendered_prompt"):
                                    mlflow.log_text(result["rendered_prompt"], "pass3_prompt.txt")
                                if result.get("raw_response"):
                                    mlflow.log_text(result["raw_response"], "pass3_response.json")
                        except Exception as e:
                            print(f"    Warning: MLflow logging failed: {e}")
                    else:
                        failed += 1
                        err = result.get("error", "unknown")
                        print(f"    [{completed + failed}/{len(flagged_slides)}] "
                              f"FAILED: {key_diff[:40]} — {err[:60]}")
                except Exception as e:
                    failed += 1
                    print(f"    [{completed + failed}/{len(flagged_slides)}] "
                          f"EXCEPTION: {sid} — {e}")

        # Save generation record for pass 3
        if not args.dry_run and conn:
            try:
                save_generation_twopass(
                    conn, uc_catalog, uc_schema, ids,
                    generation_pass="pass3_update",
                    raw_response=json.dumps({
                        "slides_updated": completed,
                        "slides_failed": failed,
                        "slide_ids": slide_ids_updated,
                    }),
                    slide_count=len(flagged_slides),
                    product_category="",
                    generation_context={
                        "flagged_count": len(flagged_slides),
                        "updated_count": completed,
                        "failed_count": failed,
                    },
                )
            except Exception as e:
                print(f"  Warning: Could not save generation record: {e}")

            # Update session history with pass3_updates entry
            try:
                session = load_session(conn, uc_catalog, uc_schema, ids.battlecard_id, session_id)
                session_history = session.get("conversation_history", {}) if session else {}
                if isinstance(session_history, str):
                    session_history = json.loads(session_history) if session_history else {}
                if "pass3_updates" not in session_history:
                    session_history["pass3_updates"] = []
                session_history["pass3_updates"].append({
                    "triggered_at": datetime.utcnow().isoformat(),
                    "completed_at": datetime.utcnow().isoformat(),
                    "slides_targeted": len(flagged_slides),
                    "completed": completed,
                    "failed": failed,
                    "slide_ids": slide_ids_updated,
                })
                save_session(conn, uc_catalog, uc_schema, ids,
                             status="pass3_complete", conversation_history=session_history)
            except Exception as e:
                print(f"  Warning: Could not update session: {e}")

        # Summary
        mlflow.log_metrics({
            "pass3_completed": completed,
            "pass3_failed": failed,
            "pass3_total": len(flagged_slides),
        })

        print(f"\n  Pass 3 Update Complete!")
        print(f"    Total flagged: {len(flagged_slides)}")
        print(f"    Updated: {completed}")
        print(f"    Failed: {failed}")
        print(f"    MLflow Run: {parent_run.info.run_id}")


def _run_pass2(
    args, ids, openai_client, model_name, pass2_template,
    pending_slides, competitor, directives, context_text,
    conn, uc_catalog, uc_schema, session_id,
):
    """Run Pass 2 as a standalone operation (--pass2-only mode)."""
    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "/Users/tahir.fayyaz@databricks.com/l200-generation",
    )
    profile = os.getenv("DATABRICKS_PROFILE", "fe-vm-pmt")
    os.environ.setdefault("DATABRICKS_CONFIG_PROFILE", profile)
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")
    mlflow.set_experiment(experiment_name)

    run_name = f"l200_twopass_pass2_{competitor}_{ids.product_area}_v{ids.battlecard_version_id}"
    with mlflow.start_run(run_name=run_name) as parent_run:
        ids.agent_run_id = parent_run.info.run_id
        ids.experiment_id = parent_run.info.experiment_id

        mlflow.log_params(ids.to_mlflow_params())
        mlflow.log_params({
            "generation_mode": "twopass",
            "pass_mode": "pass2_only",
            "max_workers": args.max_workers,
            "pending_slides": len(pending_slides),
        })

        session_history = {
            "pass1": {"status": "complete", "slide_count": len(pending_slides)},
            "pass2": {"status": "in_progress", "completed": 0, "failed": 0,
                      "pending": len(pending_slides)},
        }

        _run_pass2_in_mlflow(
            args=args,
            ids=ids,
            openai_client=openai_client,
            model_name=model_name,
            pass2_template=pass2_template,
            skeleton_slides=pending_slides,
            competitor=competitor,
            directives=directives,
            context_text=context_text,
            conn=conn,
            uc_catalog=uc_catalog,
            uc_schema=uc_schema,
            session_id=session_id,
            session_history=session_history,
        )


def _run_pass2_in_mlflow(
    args, ids, openai_client, model_name, pass2_template,
    skeleton_slides, competitor, directives, context_text,
    conn, uc_catalog, uc_schema, session_id, session_history,
):
    """Execute Pass 2 inside an active MLflow run context."""
    completed = 0
    failed = 0
    results = []

    print(f"\n  Processing {len(skeleton_slides)} diffs with {args.max_workers} workers...")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_slide = {}
        for skeleton in skeleton_slides:
            future = executor.submit(
                process_single_diff,
                openai_client=openai_client,
                model_name=model_name,
                pass2_template=pass2_template,
                skeleton=skeleton,
                competitor=competitor,
                directives=directives,
                context_text=context_text,
                conn=conn,
                catalog=uc_catalog,
                schema=uc_schema,
                battlecard_id=ids.battlecard_id,
                dry_run=args.dry_run,
                no_structured_output=args.no_structured_output,
            )
            future_to_slide[future] = skeleton

        for future in as_completed(future_to_slide):
            result = future.result()
            results.append(result)

            if result["status"] == "complete":
                completed += 1
                print(f"    [{completed + failed}/{len(skeleton_slides)}] "
                      f"COMPLETE: {result['category']} / {result['key_differentiator']}")

                # Log to MLflow as child run
                try:
                    child_name = f"pass2_{result['slide_id']}"
                    with mlflow.start_run(run_name=child_name, nested=True) as child_run:
                        mlflow.log_params({
                            "slide_id": result["slide_id"],
                            "category": result["category"],
                            "key_differentiator": result["key_differentiator"],
                            "pass": "pass2_detail",
                        })
                        if result.get("rendered_prompt"):
                            mlflow.log_text(result["rendered_prompt"], "rendered_prompt.txt")
                        if result.get("raw_response"):
                            mlflow.log_text(result["raw_response"], "raw_response.txt")
                        if result.get("full_content"):
                            mlflow.log_dict(result["full_content"], "merged_slide.json")
                except Exception as e:
                    print(f"    WARNING: Could not log child run: {e}")

            else:
                failed += 1
                print(f"    [{completed + failed}/{len(skeleton_slides)}] "
                      f"FAILED: {result['category']} / {result['key_differentiator']}")
                print(f"      Error: {result.get('error', 'unknown')}")

    # Save Pass 2 generation record
    if not args.dry_run and conn:
        save_generation_twopass(
            conn, uc_catalog, uc_schema, ids,
            generation_pass="pass2_detail",
            prompt_rendered="[per-diff prompts logged in child runs]",
            raw_response=f"[{completed} complete, {failed} failed out of {len(skeleton_slides)}]",
            slide_count=completed,
            generation_context={
                "generation_mode": "twopass",
                "pass": "pass2_detail",
                "session_id": session_id,
                "total_diffs": len(skeleton_slides),
                "completed": completed,
                "failed": failed,
            },
        )

        # Update session
        session_history["pass2"] = {
            "status": "complete" if failed == 0 else "partial",
            "completed": completed,
            "failed": failed,
            "pending": 0,
            "completed_at": datetime.utcnow().isoformat(),
        }
        final_status = "completed" if failed == 0 else "pass2_partial"
        save_session(conn, uc_catalog, uc_schema, ids,
                     status=final_status, conversation_history=session_history)

    # Log metrics
    mlflow.log_metric("pass2_completed", completed)
    mlflow.log_metric("pass2_failed", failed)
    mlflow.log_metric("pass2_total", len(skeleton_slides))

    # Final summary
    print(f"\n{'='*60}")
    print(f"  PASS 2 COMPLETE")
    print(f"{'='*60}")
    print(f"  Battlecard ID:   {ids.battlecard_id}")
    print(f"  Session ID:      {session_id}")
    print(f"  Total diffs:     {len(skeleton_slides)}")
    print(f"  Completed:       {completed}")
    print(f"  Failed:          {failed}")
    if failed > 0:
        print(f"\n  Failed diffs:")
        for r in results:
            if r["status"] == "failed":
                print(f"    - {r['category']} / {r['key_differentiator']}: {r.get('error', '?')}")
        print(f"\n  To retry failed diffs:")
        print(f"    python -m scripts.generate_l200_twopass \\")
        print(f"      --pass2-only --battlecard-id {ids.battlecard_id} \\")
        print(f"      --competitor \"{ids.competitor}\" --product-area \"{ids.product_area}\" \\")
        print(f"      --prompt-dir {args.prompt_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
