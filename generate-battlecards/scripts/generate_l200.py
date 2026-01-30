"""
Generate L200 Slides using a registered MLflow prompt.

This script:
1. Loads the prompt from MLflow Prompt Registry
2. Renders it with competitor-specific variables + any context files
3. Calls Databricks Model Serving (OpenAI-compatible) with structured output
4. Parses the JSON response into individual slides with v3 citations
5. Saves each slide to the l200_slides UC table
6. Creates a generation record in battlecard_generations
7. Logs everything to MLflow with linked IDs

Usage:
    python -m scripts.generate_l200 --competitor "Microsoft Fabric" --product-area "Data Platform"
    python -m scripts.generate_l200 --competitor "Microsoft Fabric" --product-area "Data Platform" \
        --directives directives/general.md --context context.md --context competitor_notes.txt

Environment variables:
    DATABRICKS_PROFILE  - Databricks CLI profile (default: fe-vm-pmt)
    MODEL_NAME          - Model serving endpoint (default: databricks-claude-sonnet-4)
    UC_CATALOG          - Unity Catalog catalog (default: pm_technical)
    UC_SCHEMA           - Unity Catalog schema (default: battlecards)
    PROMPT_CATALOG      - Prompt registry catalog (default: pm_technical)
    PROMPT_SCHEMA       - Prompt registry schema (default: battlecards)
    PROMPT_NAME         - Prompt name (default: l200_slide_v1)
"""

import argparse
import hashlib
import json
import os
import re
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

import mlflow
import mlflow.genai
from databricks import sql
from databricks.sdk import WorkspaceClient

# Import the linked ID system
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.linked_ids import GenerationIDs, new_id


# ---------------------------------------------------------------------------
# Databricks client helpers
# ---------------------------------------------------------------------------

def get_openai_client(profile: str):
    """Get an OpenAI-compatible client from Databricks."""
    from openai import OpenAI

    host = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")
    if host and token:
        return OpenAI(
            api_key=token,
            base_url=host.rstrip("/") + "/serving-endpoints",
        )
    w = WorkspaceClient(profile=profile)
    return w.serving_endpoints.get_open_ai_client()


def get_sql_connection(profile: str):
    """Get a SQL connection for Unity Catalog writes."""
    w = WorkspaceClient(profile=profile)
    host = w.config.host.replace("https://", "")

    http_path = os.getenv("DATABRICKS_SQL_WAREHOUSE_HTTP_PATH")
    if not http_path:
        warehouses = list(w.warehouses.list())
        for wh in warehouses:
            if wh.state and wh.state.value == "RUNNING":
                http_path = f"/sql/1.0/warehouses/{wh.id}"
                break
        if not http_path and warehouses:
            http_path = f"/sql/1.0/warehouses/{warehouses[0].id}"

    if not http_path:
        raise ValueError("No SQL warehouse found")

    return sql.connect(
        server_hostname=host,
        http_path=http_path,
        auth_type="databricks-oauth",
    )


# ---------------------------------------------------------------------------
# Version resolution
# ---------------------------------------------------------------------------

def get_next_version(conn, catalog: str, schema: str, competitor: str, product_area: str) -> int:
    """Get the next battlecard_version_id for a competitor+product_area."""
    query = f"""
    SELECT COALESCE(MAX(battlecard_version_id), 0) + 1
    FROM {catalog}.{schema}.battlecard_generations
    WHERE competitor = '{_escape(competitor)}'
      AND product_area = '{_escape(product_area)}'
    """
    with conn.cursor() as cursor:
        cursor.execute(query)
        result = cursor.fetchone()
        return result[0] if result else 1


def _escape(s: str) -> str:
    """Escape single quotes for SQL."""
    return s.replace("'", "''")


# ---------------------------------------------------------------------------
# Context loading
# ---------------------------------------------------------------------------

def classify_context_file(filename: str) -> Dict[str, Optional[str]]:
    """Classify a context file by its filename pattern.

    Supported patterns:
        competitive_directive_<scope>.md  -> doc_type=competitive_directive, doc_scope from suffix
        battlecard_archive_<scope>_<date>.md -> doc_type=battlecard_archive
        product_categories_<scope>.md     -> doc_type=product_categories
        key_differentiator_themes.md      -> doc_type=key_differentiator_themes
        anything else                     -> doc_type=general_context

    Returns dict with doc_type, doc_scope, xml_tag.
    """
    name_lower = filename.lower()
    base = os.path.splitext(filename)[0]

    if name_lower.startswith("competitive_directive"):
        # competitive_directive_<scope>.md
        parts = base.split("_", 2)  # ["competitive", "directive", "<scope>"]
        scope = parts[2] if len(parts) > 2 else None
        return {"doc_type": "competitive_directive", "doc_scope": scope, "xml_tag": "competitive_directive"}
    elif name_lower.startswith("battlecard_archive"):
        # battlecard_archive_<scope>_<date>.md
        parts = base.split("_", 2)  # ["battlecard", "archive", "<scope>_<date>"]
        scope = parts[2] if len(parts) > 2 else None
        return {"doc_type": "battlecard_archive", "doc_scope": scope, "xml_tag": "battlecard_archive"}
    elif name_lower.startswith("product_categories"):
        # product_categories_<scope>.md
        parts = base.split("_", 2)  # ["product", "categories", "<scope>"]
        scope = parts[2] if len(parts) > 2 else None
        return {"doc_type": "product_categories", "doc_scope": scope, "xml_tag": "product_categories"}
    elif name_lower.startswith("key_differentiator_themes"):
        return {"doc_type": "key_differentiator_themes", "doc_scope": None, "xml_tag": "key_differentiator_themes"}
    else:
        return {"doc_type": "general_context", "doc_scope": None, "xml_tag": "context"}


def load_context_files(paths: List[str], source_type: str = "human_provided") -> List[Dict[str, Any]]:
    """Load context from file paths. Returns list of classified context dicts.

    Each dict has: name, path, content, content_hash, doc_type, doc_scope, xml_tag, source_type.
    """
    contexts = []
    for p in paths:
        if not p or not os.path.isfile(p):
            print(f"  Warning: context file not found: {p}")
            continue
        with open(p) as f:
            content = f.read()
        filename = os.path.basename(p)
        classification = classify_context_file(filename)
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        contexts.append({
            "name": filename,
            "path": p,
            "content": content,
            "content_hash": content_hash,
            "source_type": source_type,
            **classification,
        })
        print(f"  Loaded context: {filename} (type={classification['doc_type']}, "
              f"scope={classification['doc_scope']}, {len(content)} chars)")
    return contexts


def format_context_for_prompt(contexts: List[Dict[str, Any]]) -> str:
    """Format loaded context files into XML-tagged prompt sections.

    Each context document is wrapped in an XML tag based on its doc_type,
    with attributes for name, doc_type, doc_scope, and source_type.
    """
    if not contexts:
        return "No additional context provided."
    parts = []
    for ctx in contexts:
        tag = ctx.get("xml_tag", "context")
        attrs = f'name="{ctx["name"]}" doc_type="{ctx["doc_type"]}" source="{ctx["source_type"]}"'
        if ctx.get("doc_scope"):
            attrs += f' scope="{ctx["doc_scope"]}"'
        parts.append(f"<{tag} {attrs}>\n{ctx['content']}\n</{tag}>")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def register_prompt_from_file(
    prompt_file: str,
    prompt_full_name: str,
    prompt_alias: str = "latest",
) -> tuple:
    """Load prompt template from a .md file and register it to MLflow Prompt Registry.

    Returns (template_text, version_number).
    """
    with open(prompt_file) as f:
        template = f.read()

    print(f"  Loaded prompt from file: {prompt_file} ({len(template)} chars)")

    # Register to MLflow Prompt Registry (creates a new version)
    try:
        prompt_version = mlflow.genai.register_prompt(
            prompt_full_name,
            template,
        )
        version = prompt_version.version
        print(f"  Registered prompt to MLflow: {prompt_full_name} v{version}")

        # Set alias
        mlflow.genai.set_prompt_alias(
            prompt_full_name,
            prompt_alias,
            version,
        )
        print(f"  Set alias '{prompt_alias}' -> v{version}")
    except Exception as e:
        print(f"  WARNING: Could not register prompt to MLflow ({e.__class__.__name__}: {e})")
        version = 0

    return template, version


def load_and_render_prompt(
    prompt_full_name: str,
    prompt_alias: str,
    competitor: str,
    product_area: str,
    product_category: str,
    directives: str = "",
    context: str = "",
    prompt_file: Optional[str] = None,
    product_categories_text: str = "",
    total_diffs: int = 0,
) -> tuple:
    """Load prompt from file, registry, or local fallback, and render with variables.

    If prompt_file is provided, loads from that file and registers to MLflow.
    Otherwise loads from MLflow registry with local fallback.

    Returns (rendered_prompt, prompt_version_number).
    """
    version = 0

    if prompt_file:
        # Load from file and register to MLflow
        template, version = register_prompt_from_file(
            prompt_file, prompt_full_name, prompt_alias,
        )
    else:
        try:
            prompt = mlflow.genai.load_prompt(f"prompts:/{prompt_full_name}@{prompt_alias}")
            version = getattr(prompt, "version", 0)
            template = prompt.template
            print(f"  Loaded prompt from registry: v{version}")
        except Exception as e:
            print(f"  Prompt registry unavailable ({e.__class__.__name__}), using local template")
            from scripts.register_l200_prompt import L200_PROMPT_TEMPLATE
            template = L200_PROMPT_TEMPLATE

    # Render with {{variable}} syntax
    rendered = template.replace("{{competitor}}", competitor)
    rendered = rendered.replace("{{product_area}}", product_area)
    rendered = rendered.replace("{{product_category}}", product_category)
    rendered = rendered.replace("{{comparison}}", f"Databricks vs {competitor}")
    rendered = rendered.replace("{{directives}}", directives or "No specific directives provided.")
    rendered = rendered.replace("{{context}}", context or "No additional context provided.")
    rendered = rendered.replace("{{product_categories}}", product_categories_text or "No product categories provided.")
    rendered = rendered.replace("{{total_diffs}}", str(total_diffs) if total_diffs else "50")

    return rendered, version


def call_model(
    client,
    model_name: str,
    rendered_prompt: str,
    json_schema: Optional[Dict] = None,
    temperature: float = 0.2,
    max_tokens: int = 16384,
) -> str:
    """Call the model via Databricks Model Serving with optional structured output."""
    kwargs = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": rendered_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if json_schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": json_schema,
        }
        print(f"  Using structured output (json_schema: {json_schema['name']})")

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def parse_slides(raw_response: str) -> List[Dict[str, Any]]:
    """Parse the model response into a list of slide dicts.

    Handles both:
    - Structured output: {"slides": [...]} wrapper
    - Plain JSON array: [...]
    """
    text = raw_response.strip()

    # Remove markdown fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)

    # Try parsing directly
    try:
        data = json.loads(text)
        # Structured output wraps in {"slides": [...]}
        if isinstance(data, dict) and "slides" in data:
            return data["slides"]
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    # Try finding array in text
    start = text.find("[")
    end = text.rfind("]") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse slides from response. First 200 chars: {text[:200]}")


# ---------------------------------------------------------------------------
# Unity Catalog storage
# ---------------------------------------------------------------------------

def save_generation(conn, catalog: str, schema: str, ids: GenerationIDs, **kwargs):
    """Save generation record to battlecard_generations."""
    prefix = f"{catalog}.{schema}"

    # Build optional fields
    rendered = kwargs.get("prompt_rendered", "")
    raw = kwargs.get("raw_response", "")
    slide_count = kwargs.get("slide_count", 0)
    context_json = kwargs.get("generation_context")

    context_sql = "NULL"
    if context_json:
        context_sql = f"PARSE_JSON('{_escape(json.dumps(context_json))}')"

    query = f"""
    INSERT INTO {prefix}.battlecard_generations
    (battlecard_id, battlecard_version_id, prompt_name, prompt_version,
     experiment_id, agent_run_id, competitor, product_area, product_category,
     model_name, slide_count, prompt_rendered, raw_response,
     generation_context, generated_at)
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
        '{ids.generated_at.isoformat()}'
    )
    """
    with conn.cursor() as cursor:
        cursor.execute(query)


def save_context_documents(
    conn,
    catalog: str,
    schema: str,
    ids: GenerationIDs,
    context_files: List[Dict[str, Any]],
):
    """Save context documents to battlecard_context_documents."""
    prefix = f"{catalog}.{schema}"

    with conn.cursor() as cursor:
        for ctx in context_files:
            doc_id = str(uuid.uuid4())
            query = f"""
            INSERT INTO {prefix}.battlecard_context_documents
            (context_doc_id, battlecard_id, agent_run_id,
             doc_type, doc_name, doc_scope, competitor, product_area,
             source_type, content_md, content_hash, created_at)
            VALUES (
                '{doc_id}',
                '{_escape(ids.battlecard_id)}',
                '{_escape(ids.agent_run_id)}',
                '{_escape(ctx.get("doc_type", "general_context"))}',
                '{_escape(ctx["name"])}',
                {("'" + _escape(ctx["doc_scope"]) + "'") if ctx.get("doc_scope") else "NULL"},
                '{_escape(ids.competitor)}',
                '{_escape(ids.product_area)}',
                '{_escape(ctx.get("source_type", "human_provided"))}',
                '{_escape(ctx["content"])}',
                '{_escape(ctx.get("content_hash", ""))}',
                '{datetime.utcnow().isoformat()}'
            )
            """
            cursor.execute(query)
            print(f"  Saved context doc: {ctx['name']} ({ctx.get('doc_type')}) -> {doc_id}")


def save_slides(
    conn,
    catalog: str,
    schema: str,
    ids: GenerationIDs,
    slides: List[Dict[str, Any]],
    slide_type_override: Optional[str] = None,
):
    """Save individual slide rows to battlecard_slides."""
    prefix = f"{catalog}.{schema}"

    with conn.cursor() as cursor:
        for i, slide in enumerate(slides):
            slide_id = slide.get("id", f"slide_{i}")
            slide_type = slide_type_override or "l200_critical_differentiator"
            category = slide.get("category", "")
            slide_title = slide.get("key_differentiator", slide.get("headline", ""))
            db_rating = slide.get("databricks_rating", "")
            comp_rating = slide.get("competitor_rating", "")
            content_json = _escape(json.dumps(slide))

            query = f"""
            INSERT INTO {prefix}.battlecard_slides
            (slide_id, slide_order, slide_type,
             battlecard_id, battlecard_version_id,
             agent_run_id, prompt_name, prompt_version,
             competitor, product_area, category, slide_title,
             slide_content, databricks_rating, competitor_rating, generated_at)
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
                '{ids.generated_at.isoformat()}'
            )
            """
            cursor.execute(query)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate L200 slides")
    parser.add_argument("--competitor", required=True, help="Competitor name")
    parser.add_argument("--product-area", required=True, help="Product area")
    parser.add_argument("--product-category", default="Data Platform", help="Product category")
    parser.add_argument("--directives", default="", help="Directives text or path to .md file")
    parser.add_argument("--context", action="append", default=[],
                        help="Context file paths (can specify multiple: --context a.md --context b.md)")
    parser.add_argument("--prompt-file", default="",
                        help="Path to prompt template .md file (registers to MLflow on use)")
    parser.add_argument("--product-categories", default="",
                        help="Path to product categories .md file (parsed for category list & injected into prompt)")
    parser.add_argument("--diffs-per-category", type=int, default=10,
                        help="Number of key differentiators per product category (default: 10)")
    parser.add_argument("--no-structured-output", action="store_true",
                        help="Disable JSON schema structured output enforcement")
    parser.add_argument("--dry-run", action="store_true", help="Skip UC save, print slides")
    args = parser.parse_args()

    # Config
    profile = os.getenv("DATABRICKS_PROFILE", "fe-vm-pmt")
    model_name = os.getenv("MODEL_NAME", "databricks-claude-sonnet-4")
    uc_catalog = os.getenv("UC_CATALOG", "pm_technical")
    uc_schema = os.getenv("UC_SCHEMA", "battlecards")
    prompt_catalog = os.getenv("PROMPT_CATALOG", "pm_technical")
    prompt_schema = os.getenv("PROMPT_SCHEMA", "battlecards")
    prompt_name = os.getenv("PROMPT_NAME", "l200_slide_v1")
    prompt_name_clean = re.sub(r"[^A-Za-z0-9_]", "_", prompt_name)
    prompt_full_name = f"{prompt_catalog}.{prompt_schema}.{prompt_name_clean}"
    prompt_alias = "latest"

    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "/Users/tahir.fayyaz@databricks.com/l200-generation",
    )

    # Load directives from file if path provided
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
        # Override doc_type to competitive_directive if it doesn't already match
        if directives_file_info["doc_type"] == "general_context":
            directives_file_info["doc_type"] = "competitive_directive"
            directives_file_info["xml_tag"] = "competitive_directive"
        print(f"  Loaded directives: {directives_path} (type={directives_file_info['doc_type']})")

    # Load product categories file if provided
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
        # Count unique categories (lines starting with "- ", deduplicated)
        category_lines = set()
        for line in product_categories_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                # Normalize: strip bullet, leading/trailing whitespace
                category_lines.add(stripped.lstrip("- ").strip())
        num_categories = len(category_lines)
        print(f"  Loaded product categories: {pc_filename} ({num_categories} categories)")

    total_diffs = num_categories * args.diffs_per_category if num_categories else 0

    # Load context files
    context_files = load_context_files(args.context)
    context_text = format_context_for_prompt(context_files)

    # Combine all context docs for UC storage (directives + context files + product categories)
    all_context_docs = list(context_files)
    if directives_file_info:
        all_context_docs.insert(0, directives_file_info)
    if product_categories_file_info:
        all_context_docs.append(product_categories_file_info)

    # Setup MLflow
    os.environ.setdefault("DATABRICKS_CONFIG_PROFILE", profile)
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")
    mlflow.set_experiment(experiment_name)
    mlflow.openai.autolog()

    # Initialize clients
    openai_client = get_openai_client(profile)

    # Get next version
    conn = None
    if not args.dry_run:
        conn = get_sql_connection(profile)
        next_version = get_next_version(
            conn, uc_catalog, uc_schema,
            args.competitor, args.product_area,
        )
    else:
        next_version = 1

    # Create linked IDs
    ids = GenerationIDs(
        battlecard_version_id=next_version,
        prompt_name=prompt_full_name,
        prompt_alias=prompt_alias,
        model_name=model_name,
        competitor=args.competitor,
        product_area=args.product_area,
    )

    print(f"Generating L200 slides:")
    print(f"  Competitor:    {args.competitor}")
    print(f"  Product area:  {args.product_area}")
    print(f"  Prompt:        {prompt_full_name}@{prompt_alias}")
    if args.prompt_file:
        print(f"  Prompt file:   {args.prompt_file}")
    print(f"  Model:         {model_name}")
    print(f"  Battlecard ID: {ids.battlecard_id}")
    print(f"  Version:       {next_version}")
    print(f"  Structured:    {not args.no_structured_output}")
    print(f"  Context files: {len(context_files)}")
    if num_categories:
        print(f"  Categories:    {num_categories} x {args.diffs_per_category} diffs = {total_diffs} total")
    print()

    # Load and render prompt
    if args.prompt_file:
        print(f"Loading prompt from file: {args.prompt_file}")
    else:
        print("Loading prompt from registry...")
    rendered_prompt, prompt_version = load_and_render_prompt(
        prompt_full_name=prompt_full_name,
        prompt_alias=prompt_alias,
        competitor=args.competitor,
        product_area=args.product_area,
        product_category=args.product_category,
        directives=directives,
        context=context_text,
        prompt_file=args.prompt_file or None,
        product_categories_text=product_categories_text,
        total_diffs=total_diffs,
    )
    ids.prompt_version = prompt_version

    # Load structured output schema
    json_schema = None
    if not args.no_structured_output:
        from scripts.register_l200_prompt import L200_SLIDE_JSON_SCHEMA
        json_schema = L200_SLIDE_JSON_SCHEMA

    # Build generation context for UC storage
    generation_context = {
        "directives_source": args.directives or None,
        "prompt_file": args.prompt_file or None,
        "product_categories_file": args.product_categories or None,
        "num_categories": num_categories,
        "diffs_per_category": args.diffs_per_category,
        "total_diffs_requested": total_diffs,
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

    # Run generation inside MLflow tracking
    with mlflow.start_run(run_name=f"l200_{args.competitor}_{args.product_area}_v{next_version}") as run:
        ids.agent_run_id = run.info.run_id
        ids.experiment_id = run.info.experiment_id

        # Log all linked IDs as params (prompt_version is now set correctly)
        mlflow.log_params(ids.to_mlflow_params())
        mlflow.log_params({
            "structured_output": not args.no_structured_output,
            "context_file_count": len(context_files),
            "prompt_file": os.path.basename(args.prompt_file) if args.prompt_file else "registry",
            "num_categories": num_categories,
            "diffs_per_category": args.diffs_per_category,
            "total_diffs_requested": total_diffs,
        })
        mlflow.log_text(rendered_prompt, "rendered_prompt.txt")

        # Log prompt file as artifact
        if args.prompt_file:
            with open(args.prompt_file) as f:
                mlflow.log_text(f.read(), f"prompt/{os.path.basename(args.prompt_file)}")

        # Log product categories file as artifact
        if product_categories_text:
            mlflow.log_text(product_categories_text, f"context/{os.path.basename(args.product_categories)}")

        # Log context files as artifacts
        for ctx in context_files:
            mlflow.log_text(ctx["content"], f"context/{ctx['name']}")

        # 2. Call model
        # Use higher max_tokens for platform battlecards with many categories
        max_tokens = 32768 if total_diffs > 15 else 16384
        print(f"Calling {model_name} (max_tokens={max_tokens})...")
        raw_response = call_model(
            client=openai_client,
            model_name=model_name,
            rendered_prompt=rendered_prompt,
            json_schema=json_schema,
            max_tokens=max_tokens,
        )
        mlflow.log_text(raw_response, "raw_response.txt")

        # 3. Parse slides
        print("Parsing response...")
        slides = parse_slides(raw_response)
        print(f"  Parsed {len(slides)} slides")
        mlflow.log_metric("slide_count", len(slides))

        # Count citations across all slides
        total_citations = 0
        for slide in slides:
            citations = slide.get("citations", {})
            for field_cites in citations.values():
                if isinstance(field_cites, list):
                    total_citations += len(field_cites)
        print(f"  Total citations: {total_citations}")
        mlflow.log_metric("total_citations", total_citations)

        # Log each slide
        for i, slide in enumerate(slides):
            mlflow.log_dict(slide, f"slides/slide_{i+1}_{slide.get('id', 'unknown')}.json")

        # Log the full array
        mlflow.log_dict(slides, "slides/all_slides.json")

        # 4. Save to Unity Catalog
        if not args.dry_run and conn:
            print("Saving to Unity Catalog...")

            save_generation(
                conn, uc_catalog, uc_schema, ids,
                prompt_rendered=rendered_prompt,
                raw_response=raw_response,
                slide_count=len(slides),
                product_category=args.product_category,
                generation_context=generation_context,
            )
            print(f"  Saved generation record: {ids.battlecard_id}")

            save_slides(conn, uc_catalog, uc_schema, ids, slides)
            print(f"  Saved {len(slides)} slide rows")

            if all_context_docs:
                save_context_documents(conn, uc_catalog, uc_schema, ids, all_context_docs)
                print(f"  Saved {len(all_context_docs)} context documents")

            mlflow.log_param("saved_to_uc", True)
        else:
            print("Dry run - skipping UC save")
            print(json.dumps(slides, indent=2))

    # Summary
    print()
    print("=" * 60)
    print("Generation complete!")
    print(f"  Battlecard ID:      {ids.battlecard_id}")
    print(f"  Version:            {ids.battlecard_version_id}")
    print(f"  MLflow Run ID:      {ids.agent_run_id}")
    print(f"  MLflow Experiment:  {ids.experiment_id}")
    print(f"  Prompt:             {ids.prompt_name} v{ids.prompt_version}")
    print(f"  Slides generated:   {len(slides)}")
    print(f"  Total citations:    {total_citations}")
    print(f"  Competitor:         {ids.competitor}")
    print(f"  Product area:       {ids.product_area}")
    print("=" * 60)

    if conn:
        conn.close()


if __name__ == "__main__":
    main()
