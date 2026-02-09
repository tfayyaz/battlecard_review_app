"""
Postgres-backed Battlecard Review App (Databricks Postgres)

Uses the Databricks SDK + SQLAlchemy ``do_connect`` event pattern from
https://www.databricks.com/blog/how-use-lakebase-transactional-data-layer-databricks-apps

Local dev:  set DATABRICKS_PROFILE (e.g. "fe-vm-pmt") so WorkspaceClient
            authenticates via the Databricks CLI OAuth cache.
Databricks Apps:  DATABRICKS_HOST / DATABRICKS_CLIENT_ID / PGHOST are
            injected automatically by the platform.
"""

import json
import logging
import os
import subprocess
import threading
import time
from datetime import datetime
from difflib import SequenceMatcher
from uuid import uuid4

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; env vars must be set externally

import tiktoken
from databricks.sdk import WorkspaceClient
from flask import Flask, jsonify, redirect, render_template, request
from flask_cors import CORS
from sqlalchemy import create_engine, event, text
from werkzeug.exceptions import HTTPException

app = Flask(__name__)
CORS(app)

logger = logging.getLogger(__name__)


@app.errorhandler(Exception)
def handle_exception(e):
    """Return JSON for API errors while preserving normal HTTP exceptions."""
    if isinstance(e, HTTPException):
        return e
    logger.exception("Unhandled exception: %s", e)
    if request.path.startswith("/api/"):
        return jsonify({"error": str(e)}), 500
    return "Internal server error", 500

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Databricks CLI profile for local development (ignored in Databricks Apps).
DATABRICKS_PROFILE = os.getenv("DATABRICKS_PROFILE", "fe-vm-pmt")

# Postgres connection details
DB_HOST = os.getenv(
    "PGHOST",
    "instance-98170129-d87f-4357-b0ce-8991c128dea8.database.cloud.databricks.com",
)
DB_PORT = int(os.getenv("PGPORT", "5432"))
DB_NAME = os.getenv("PGDATABASE", "databricks_postgres")
DEFAULT_LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))


def _build_engine():
    """Build a SQLAlchemy engine with Databricks OAuth token injection.

    If DATABASE_URL is set, use it directly (escape-hatch for any env).
    Otherwise, follow the Lakebase blog pattern:
      1. Create a WorkspaceClient (uses CLI profile locally, platform creds in Apps).
      2. Use the current user's email (local) or app client_id (Apps) as PG username.
      3. Inject the OAuth access_token as the PG password on every connection via
         the SQLAlchemy ``do_connect`` event.
    """
    explicit = os.getenv("DATABASE_URL")
    if explicit:
        return create_engine(explicit, pool_pre_ping=True)

    # Initialise WorkspaceClient — picks up DATABRICKS_PROFILE for local dev,
    # or platform-injected env vars inside Databricks Apps.
    # Wrapped in a list so nested functions can reassign it after re-auth.
    def _init_workspace_client(profile):
        """Create a WorkspaceClient, triggering CLI re-auth if the refresh token is expired."""
        try:
            return WorkspaceClient(profile=profile)
        except ValueError as e:
            err_msg = str(e)
            if "refresh token is invalid" in err_msg or "access token could not be retrieved" in err_msg:
                logger.warning("OAuth refresh token expired at startup — triggering 'databricks auth login' (will open browser)...")
                # Extract host from the error message or use env/default
                host = os.getenv("DATABRICKS_HOST", "https://fe-vm-pmt.cloud.databricks.com")
                try:
                    subprocess.run(
                        ["databricks", "auth", "login", "--host", host, "--profile", profile],
                        check=True,
                        timeout=120,
                    )
                    logger.info("Re-authentication successful, creating WorkspaceClient...")
                    return WorkspaceClient(profile=profile)
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as auth_err:
                    logger.error("databricks auth login failed: %s", auth_err)
                    raise e from auth_err
            raise

    _ws = [_init_workspace_client(DATABRICKS_PROFILE)]

    # Determine the Postgres username.
    #   - In Databricks Apps the DATABRICKS_CLIENT_ID env var is the PG role.
    #   - Locally, fall back to the authenticated user's email.
    pg_user = os.getenv("DATABRICKS_CLIENT_ID") or os.getenv("PGUSER")
    if not pg_user:
        try:
            me = _ws[0].current_user.me()
            pg_user = me.user_name  # e.g. "tahir.fayyaz@databricks.com"
        except Exception:
            pg_user = "databricks"
    logger.info("Lakebase PG user: %s  host: %s", pg_user, DB_HOST)

    engine = create_engine(
        f"postgresql+psycopg://{pg_user}:@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        pool_pre_ping=True,
        pool_size=5,
        pool_recycle=1800,
    )

    def _get_fresh_token():
        """Get a fresh OAuth token, triggering CLI re-auth if the refresh token has expired."""
        try:
            return _ws[0].config.oauth_token().access_token
        except Exception as token_err:
            err_msg = str(token_err)
            if "refresh token is invalid" in err_msg or "access token could not be retrieved" in err_msg:
                logger.warning("OAuth refresh token expired — triggering 'databricks auth login' (will open browser)...")
                host = _ws[0].config.host
                profile = DATABRICKS_PROFILE
                try:
                    subprocess.run(
                        ["databricks", "auth", "login", "--host", host, "--profile", profile],
                        check=True,
                        timeout=120,
                    )
                    logger.info("Re-authentication successful, retrying token fetch...")
                    # Re-create WorkspaceClient to pick up new token cache
                    _ws[0] = WorkspaceClient(profile=profile)
                    return _ws[0].config.oauth_token().access_token
                except subprocess.TimeoutExpired:
                    logger.error("databricks auth login timed out after 120s")
                    raise token_err
                except subprocess.CalledProcessError as e:
                    logger.error("databricks auth login failed: %s", e)
                    raise token_err
            raise

    @event.listens_for(engine, "do_connect")
    def _provide_token(dialect, conn_rec, cargs, cparams):
        """Inject a fresh OAuth token as the PG password on each connection."""
        cparams["password"] = _get_fresh_token()

    return engine


ENGINE = _build_engine()


# ---------------------------------------------------------------------------
# Default context file paths (relative to project root)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DIRECTIVE_PATH = os.path.join(
    _PROJECT_ROOT, "generate-battlecards", "0_context", "directives",
    "FABRIC_COMPETE_PITCHDECKS_FY26.md",
)
DEFAULT_OLD_BATTLECARD_PATH = os.path.join(
    _PROJECT_ROOT, "generate-battlecards", "0_context", "previous_battlecards",
    "MULTIPLE_USE_CASE_BATTLECARDS_FABRIC_FY26_SLIDES.md",
)
DEFAULT_PROMPTS_DIR = os.path.join(
    _PROJECT_ROOT, "generate-battlecards", "2_prompts",
)
DEFAULT_PASS1_PROMPT = os.path.join(DEFAULT_PROMPTS_DIR, "l200_pass1_planning_v1.md")
DEFAULT_PASS1_PROMPT_V2 = os.path.join(DEFAULT_PROMPTS_DIR, "l200_pass1_planning_v2.md")
DEFAULT_PASS1_PROMPT_V3 = os.path.join(DEFAULT_PROMPTS_DIR, "l200_pass1_planning_v3.md")
DEFAULT_PASS1_PROMPT_V4 = os.path.join(DEFAULT_PROMPTS_DIR, "l200_pass1_planning_v4_per_category.md")
DEFAULT_PASS2_PROMPT = os.path.join(DEFAULT_PROMPTS_DIR, "l200_pass2_detail_v3_factcheck.md")
DEFAULT_PASS2_PROMPT_V10 = os.path.join(DEFAULT_PROMPTS_DIR, "l200_pass2_detail_v10_granular.md")
DEFAULT_DIRECTIVE_PROMPT = os.path.join(
    os.path.expanduser("~"), "databricks-dev", "compete_automation",
    "battlecard-skill", "battlecards", "fabric-platform",
    "fabric-platform-0.0.2", "prompts", "STEP_A_001_GEN_DIRECTIVE_PROMPT.md",
)

# ---------------------------------------------------------------------------
# Pass 1 prompt template versions (selectable at workflow creation)
# ---------------------------------------------------------------------------
PASS1_PROMPT_TEMPLATES = {
    1: {
        "label": "V1 — Original (3-6 word diffs)",
        "description": "Original prompt. Key diffs are 3-6 words, descriptions up to 15 words.",
        "file": DEFAULT_PASS1_PROMPT,
    },
    2: {
        "label": "V2 — Snappy (2-4 word diffs)",
        "description": "Shorter key diff names (2-4 words). Benefit-focused descriptions (max 12 words). Theme guidance included.",
        "file": DEFAULT_PASS1_PROMPT_V2,
    },
    3: {
        "label": "V3 — Core vs Cross-Platform",
        "description": "Like V2 but treats Cross-Platform Capabilities as context woven into Core Product Category slides, not separate slides.",
        "file": DEFAULT_PASS1_PROMPT_V3,
    },
    4: {
        "label": "V4 — Parallel Per-Category",
        "description": "Per-category parallel execution. One LLM call per core product category with cross-platform context.",
        "file": DEFAULT_PASS1_PROMPT_V4,
    },
}

# ---------------------------------------------------------------------------
# Pass 2 prompt template versions (selectable at workflow creation)
# ---------------------------------------------------------------------------
PASS2_PROMPT_TEMPLATES = {
    1: {
        "label": "V1 — Technical Detail",
        "description": "Original prompt. Leads with technology and architecture details.",
        "file": DEFAULT_PASS2_PROMPT,  # loaded from disk at runtime
    },
    2: {
        "label": "V2 — Outcome-Focused",
        "description": "Improved prompt. Leads with outcomes/limitations, not technology. 10-15 words max.",
        "template": """\
You fill in detailed competitive analysis for a single key differentiator on a Databricks vs {{competitor}} platform battlecard.

## Key Differentiator
- **Category**: {{category}}
- **Key Differentiator**: {{key_differentiator}}
- **Description**: {{description}}
- **Databricks Rating**: {{databricks_rating}}
- **Competitor Rating**: {{competitor_rating}}
- **Selection Reasoning**: {{selection_reasoning}}

## Audience
This battlecard is for designed for the sales team at Databricks who are selling to **C-suite executives** (CIO, CTO, CDO, VP Data/AI) and **data/ML/AI practitioners** (data engineers, ML engineers, analytics engineers, data scientists, AI engineers, data analysts, etc).
- C-suite cares about: strategic platform direction, total cost of ownership, vendor risk, governance posture, time-to-value, AI/ML readiness, etc.
- Practitioners care about: performance benchmarks, developer experience, tooling maturity, open standards, operational reliability, etc.

## Directives
{{directives}}

## Additional Context
{{context}}

## Task
Generate the full detail for this single differentiator. The descriptions must be **outcome-focused, not technology-focused**. Lead with the business outcome or limitation, NOT the underlying technology. Include compelling headlines, reasoning for each rating, and properly cited sources.

This prompt combines Pass 2 detail generation with Pass 3-style rigor and inline fact-checking. That means:
- MULTIPLE citations per field (details + reasoning)
- Multiple distinct sources (docs, blogs, analyst reports, directives, context)
- Populate citation verdicts + rationales (do NOT leave all as unverified)

### Writing style for details fields — CRITICAL

**Lead with the outcome or limitation, NOT the technology.**

Each details field must be **10-15 words maximum**. Use this formula:
- **For strengths:** `[Benefit statement]. [Proof point or quantification].`
- **For weaknesses:** `[Limitation]. [Consequence].`

Save technical details (engine names, protocols, architectures) for the reasoning field. The details field is for instant comprehension of "why this rating."

### Examples of GOOD details (outcome-focused, concise):
- "Purpose-built for analytics. 2-3x faster in benchmarks."
- "No waiting. Queries run immediately without capacity planning."
- "Works with all major open formats. No conversion needed."
- "Optimized for Power BI reporting. Performance gaps emerge at scale."
- "Requires pre-purchased capacity. Cold starts possible."
- "Primarily proprietary storage. Limited open format support."
- "Pay only for what you use. No idle capacity costs."
- "Single pane of glass for all data assets. No extra tools needed."

### Examples of BAD details (too technical — DO NOT write like this):
- "Photon C++ vectorized engine with SIMD processing delivers 2-3x faster query performance than row-based engines on analytical workloads."
- "Microsoft Fabric uses SQL Server engine with Columnstore Index technology and VertiPaq compression. Provides columnar processing capabilities optimized for analytical queries."
- "Serverless SQL warehouses start instantly without cluster pre-warming, delivering sub-second query execution with automatic scaling."
- "Azure Databricks provides instant query execution through serverless SQL warehouses that automatically scale from zero to thousands of nodes without any cluster management or capacity planning."

## Output Format
Return ONLY a JSON object with these fields:

```json
{
  "databricks_headline": "<3-8 word headline for Databricks position>",
  "databricks_details": [
    "<Detail item 1: ONE sentence, 10-15 words max — lead with OUTCOME/BENEFIT, not technology.>",
    "<Detail item 2: ONE sentence covering a different capability or fact.>"
  ],
  "databricks_reasoning": "<why Databricks gets this rating — reference specific features/benchmarks/architecture>",
  "competitor_headline": "<3-8 word headline for competitor position>",
  "competitor_details": [
    "<Detail item 1: ONE sentence, 10-15 words max — lead with LIMITATION or CAPABILITY.>",
    "<Detail item 2: ONE sentence covering a different capability or fact.>"
  ],
  "competitor_reasoning": "<why competitor gets this rating — reference specific limitations or strengths>",
  "citations": {
    "databricks_details": [
      {
        "citation_id": "cite_databricks_details_0_1",
        "detail_item_index": 0,
        "start_index": 0,
        "end_index": 42,
        "source_index": 1,
        "source_quote": "<exact passage from source that supports this claim>",
        "verdict": "verified|unverified|disputed|outdated",
        "confidence": 0.0,
        "verdict_rationale": "<why this citation supports (or disputes) the claim>"
      }
    ],
    "databricks_reasoning": [],
    "competitor_details": [],
    "competitor_reasoning": []
  },
  "sources": [
    {
      "index": 1,
      "title": "<source title>",
      "url": "<URL or internal://path>",
      "type": "documentation|blog|directive|analyst_report|news|context",
      "accessed_at": "<ISO timestamp>"
    }
  ],
  "research_sources": ["<url1>", "<url2>"]
}
```

## Citation + Fact-Check Rules
- Every distinct factual claim in databricks_details, databricks_reasoning, competitor_details, and competitor_reasoning MUST have its own citation entry.
- Each details field should have 2-3 citations. Each reasoning field should have 2-4 citations.
- The start_index/end_index range must exactly match a substring of the parent field's text value.
- Use MULTIPLE different sources (minimum 3). Do not cite everything from one source.
- Include a "source_quote" — the exact passage that supports the cited text.
- **Populate verdict/confidence/verdict_rationale now** based on the evidence in the source_quote.
  - verified: clear, direct support in the quoted evidence
  - unverified: insufficient evidence in the quote
  - disputed: quote conflicts with claim
  - outdated: quote indicates information is old or superseded
- Prefer context + official docs for evidence; use web sources if necessary.

## Rules
1. **Detail items must be outcome-focused** — ONE sentence per item, 10-15 words max. Produce 2-4 detail items per vendor. Lead with WHAT IT MEANS for the user (benefit/limitation), then one proof point. Do NOT lead with technology names, engine names, or architecture details. Do NOT repeat information from the headline.
2. **Reasoning fields carry the technical depth** — put architecture details, engine names, and technical specifics here.
3. Keep claims concrete and verifiable — avoid hype or vague superlatives.
4. Include real URLs in citations where possible. Use documentation links.
5. Be fair — if the competitor genuinely excels in this area, reflect that honestly.
6. Ground claims in product capabilities, benchmarks, and architecture.
7. Each detail item should have at least 1 citation. Use `detail_item_index` (0-based) to link citations to the correct item in the details array.

Return ONLY the JSON object. No markdown fences, no explanation text.
""",
    },
    3: {
        "label": "V3 — Technical Detail with Fact-Check",
        "description": "Single key diff with inline citations, verdicts, and source quotes. High rigor.",
        "file": DEFAULT_PASS2_PROMPT,  # l200_pass2_detail_v3_factcheck.md
    },
    5: {
        "label": "V5 — Per-Category L200 (8 parallel calls)",
        "description": "Processes ALL key diffs for one category in a single call. L200-only (1-2 concise bullets). 8 calls instead of 80.",
        "template": "inline",  # Will be loaded from DB or inline fallback
    },
    6: {
        "label": "V6 — Per-Category L200+L300 (8 parallel calls)",
        "description": "Processes ALL key diffs for one category in a single call. L200 (1-2 bullets) + L300 (2-5 detailed bullets). 8 calls instead of 80.",
        "template": "inline",  # Will be loaded from DB or inline fallback
    },
    7: {
        "label": "V7 — Per-Category Strict JSON",
        "description": "Per-category batched generation with stricter JSON and citation requirements for improved determinism.",
        "template": "inline",
    },
    8: {
        "label": "V8 — Single-Call Claims + Inline Fact Check",
        "description": "Generates all key diff claims in one response and can seed fact-check verdicts inline.",
        "template": "inline",
    },
    9: {
        "label": "V9 — Per-Category L200+L300 + Inline Fact Check",
        "description": "Category-batched generation like V6, but requires inline citation verdict metadata for direct fact-check seeding.",
        "template": "inline",
    },
    10: {
        "label": "V10 — Granular L200/L300 + Per-Item Citations",
        "description": "Category-batched strict JSON. Each L200 bullet and each nested L300 bullet carries explicit citations.",
        "file": DEFAULT_PASS2_PROMPT_V10,
    },
}

STEP4_EXECUTION_OPTIONS = {
    "auto": {
        "label": "Auto (Template-Driven)",
        "description": "Use runtime defaults from selected template and worker count.",
    },
    "single_call": {
        "label": "Single Call",
        "description": "Generate all key diffs in one request.",
    },
    "category_parallel": {
        "label": "Category Parallel",
        "description": "One request per category using parallel workers.",
    },
    "category_sequential": {
        "label": "Category Sequential",
        "description": "One request per category, processed sequentially.",
    },
}

STEP5_EXECUTION_OPTIONS = {
    "auto": {
        "label": "Auto (Template-Driven)",
        "description": "Use runtime defaults from selected template and worker count.",
    },
    "per_diff_parallel": {
        "label": "Per-Diff Parallel",
        "description": "One request per key diff in parallel.",
    },
    "per_diff_sequential": {
        "label": "Per-Diff Sequential",
        "description": "One request per key diff sequentially.",
    },
    "category_parallel": {
        "label": "Category Parallel",
        "description": "One batched request per category in parallel.",
    },
    "category_sequential": {
        "label": "Category Sequential",
        "description": "One batched request per category sequentially.",
    },
    "single_call": {
        "label": "Single Call",
        "description": "Generate all key diff claims in one request.",
    },
}

# ---------------------------------------------------------------------------
# Context Window pill configuration
# Maps content_type -> display properties for the agent trajectory view
# ---------------------------------------------------------------------------
PILL_CONFIG = {
    "directive_upload":        {"color_index": 0, "label": "Directive Upload",     "turn_type": "user_input",     "role": "user"},
    "directive_generated":     {"color_index": 1, "label": "Generated Directive",  "turn_type": "tool_result",    "role": "system"},
    "old_battlecard_upload":   {"color_index": 2, "label": "Old Battlecard",       "turn_type": "user_input",     "role": "user"},
    "old_battlecard_extracted":{"color_index": 3, "label": "Extracted Battlecard", "turn_type": "tool_result",    "role": "system"},
    "product_categories":      {"color_index": 4, "label": "Categories",           "turn_type": "user_input",     "role": "user"},
    "category_selections":     {"color_index": 10, "label": "Category Selections",  "turn_type": "user_input",     "role": "user"},
    "pass1_prompt":            {"color_index": 5, "label": "Pass 1 Prompt",        "turn_type": "system_prompt",  "role": "system"},
    "pass1_skeletons":         {"color_index": 6, "label": "Key Diffs (Output)",   "turn_type": "model_output",   "role": "assistant"},
    "pass2_prompt":            {"color_index": 7, "label": "Pass 2 Prompt",        "turn_type": "system_prompt",  "role": "system"},
    "pass2_claims":            {"color_index": 8, "label": "Claims (Output)",      "turn_type": "model_output",   "role": "assistant"},
    "pass3_regenerated":       {"color_index": 8, "label": "Regenerated Claims",   "turn_type": "model_output",   "role": "assistant"},
    "google_slides_url":       {"color_index": 9, "label": "Slides URL",           "turn_type": "tool_result",    "role": "system"},
    "fact_check_results":      {"color_index": 11, "label": "Fact Checks",          "turn_type": "model_output",   "role": "assistant"},
}


# =============================================================================
# Schema setup + seed
# =============================================================================


SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS companies (
    company_id SERIAL PRIMARY KEY,
    company_name VARCHAR(255) NOT NULL,
    company_type VARCHAR(50) NOT NULL CHECK (company_type IN ('databricks', 'competitor')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS product_category_catalog (
    catalog_id SERIAL PRIMARY KEY,
    category_name VARCHAR(255) NOT NULL UNIQUE,
    category_description TEXT,
    is_core_product_category BOOLEAN NOT NULL DEFAULT FALSE,
    is_cross_platform_capability BOOLEAN NOT NULL DEFAULT FALSE,
    display_order INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS key_differentiators (
    key_diff_id SERIAL PRIMARY KEY,
    category_id INT REFERENCES product_category_catalog(catalog_id),
    generation_id INT,
    key_diff_name VARCHAR(255) NOT NULL,
    key_diff_description TEXT,
    display_order INT DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sources (
    source_id SERIAL PRIMARY KEY,
    source_name VARCHAR(500) NOT NULL,
    source_url TEXT,
    source_type VARCHAR(50) NOT NULL CHECK (source_type IN ('vendor', 'third-party', 'field-validated', 'internal', 'analyst')),
    publisher VARCHAR(255),
    published_date DATE,
    content_hash VARCHAR(64),
    last_crawled_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS battlecard_generations (
    generation_id SERIAL PRIMARY KEY,
    generation_uuid UUID DEFAULT gen_random_uuid(),
    trigger_type VARCHAR(50) NOT NULL CHECK (trigger_type IN (
        'initial',
        'scheduled',
        'manual_request',
        'fact_check_failed',
        'feedback_applied',
        'source_updated'
    )),
    trigger_details JSONB,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    generated_by VARCHAR(255) NOT NULL,
    generation_model VARCHAR(255),
    generation_prompt_version VARCHAR(50),
    status VARCHAR(50) DEFAULT 'draft' CHECK (status IN (
        'draft',
        'pending_review',
        'approved',
        'rejected',
        'superseded'
    )),
    previous_generation_id INT REFERENCES battlecard_generations(generation_id),
    total_claims INT DEFAULT 0,
    verified_claims INT DEFAULT 0,
    unverified_claims INT DEFAULT 0,
    disputed_claims INT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id SERIAL PRIMARY KEY,
    claim_uuid UUID DEFAULT gen_random_uuid(),
    generation_id INT REFERENCES battlecard_generations(generation_id),
    key_diff_id INT REFERENCES key_differentiators(key_diff_id),
    company_id INT REFERENCES companies(company_id),
    rating VARCHAR(20) NOT NULL CHECK (rating IN ('positive', 'neutral', 'negative')),
    rating_symbol VARCHAR(10) NOT NULL,
    headline VARCHAR(500) NOT NULL,
    description TEXT NOT NULL,
    change_type VARCHAR(50) CHECK (change_type IN ('new', 'unchanged', 'modified', 'removed')),
    previous_claim_id INT REFERENCES claims(claim_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS claim_detail_items (
    detail_item_id SERIAL PRIMARY KEY,
    claim_id INT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    item_order INT NOT NULL,
    item_text TEXT NOT NULL,
    item_text_verbose TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_detail_items_claim ON claim_detail_items(claim_id);

CREATE TABLE IF NOT EXISTS claim_detail_verbose_items (
    verbose_item_id SERIAL PRIMARY KEY,
    detail_item_id INT NOT NULL REFERENCES claim_detail_items(detail_item_id) ON DELETE CASCADE,
    item_order INT NOT NULL,
    item_text TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_verbose_items_detail ON claim_detail_verbose_items(detail_item_id);

CREATE TABLE IF NOT EXISTS evidence (
    evidence_id SERIAL PRIMARY KEY,
    claim_id INT REFERENCES claims(claim_id),
    detail_item_id INT REFERENCES claim_detail_items(detail_item_id),
    verbose_item_id INT REFERENCES claim_detail_verbose_items(verbose_item_id),
    traces_to_field VARCHAR(50) NOT NULL CHECK (traces_to_field IN ('headline', 'description', 'detail_item')),
    claim_subfield VARCHAR(30) NOT NULL DEFAULT 'l200' CHECK (claim_subfield IN ('headline', 'reasoning', 'l200', 'l300')),
    citation_id VARCHAR(120),
    citation_order INT,
    traces_to_start_index INT NOT NULL,
    traces_to_end_index INT NOT NULL,
    traces_to_text TEXT NOT NULL,
    generation_source_id INT REFERENCES sources(source_id),
    generation_source_text TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fact_checks (
    fact_check_id SERIAL PRIMARY KEY,
    evidence_id INT REFERENCES evidence(evidence_id),
    status VARCHAR(50) NOT NULL CHECK (status IN (
        'pending',
        'verified',
        'unverified',
        'disputed',
        'outdated',
        'not_applicable'
    )),
    fact_check_source_id INT REFERENCES sources(source_id),
    fact_check_source_text TEXT,
    reasoning TEXT,
    dispute_details TEXT,
    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    checked_by VARCHAR(255) NOT NULL,
    check_method VARCHAR(100) NOT NULL CHECK (check_method IN (
        'automated',
        'llm_assisted',
        'manual',
        'human_override'
    )),
    valid_until DATE,
    confidence_score INT CHECK (confidence_score BETWEEN 0 AND 100)
);

CREATE TABLE IF NOT EXISTS human_reviews (
    review_id SERIAL PRIMARY KEY,
    generation_id INT REFERENCES battlecard_generations(generation_id),
    claim_id INT REFERENCES claims(claim_id),
    detail_item_id INT REFERENCES claim_detail_items(detail_item_id),
    evidence_id INT REFERENCES evidence(evidence_id),
    fact_check_id INT REFERENCES fact_checks(fact_check_id),
    review_type VARCHAR(50) NOT NULL CHECK (review_type IN (
        'approve',
        'reject',
        'request_edit',
        'flag_for_review',
        'provide_feedback',
        'override_fact_check'
    )),
    feedback_text TEXT,
    suggested_headline VARCHAR(500),
    suggested_description TEXT,
    suggested_rating VARCHAR(20),
    issue_category VARCHAR(100) CHECK (issue_category IN (
        'factually_incorrect',
        'outdated',
        'missing_context',
        'too_technical',
        'too_vague',
        'wrong_rating',
        'wrong_source',
        'tone_issue',
        'competitive_concern',
        'legal_concern',
        'other'
    )),
    priority VARCHAR(20) DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high', 'critical')),
    resolution_status VARCHAR(50) DEFAULT 'open' CHECK (resolution_status IN (
        'open',
        'acknowledged',
        'in_progress',
        'resolved',
        'wont_fix',
        'duplicate'
    )),
    resolved_in_generation_id INT REFERENCES battlecard_generations(generation_id),
    resolved_at TIMESTAMP,
    resolved_by VARCHAR(255),
    reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reviewed_by VARCHAR(255) NOT NULL,
    reviewer_role VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS regeneration_requests (
    request_id SERIAL PRIMARY KEY,
    scope VARCHAR(50) NOT NULL CHECK (scope IN (
        'full_battlecard',
        'single_claim',
        'category',
        'company'
    )),
    generation_id INT REFERENCES battlecard_generations(generation_id),
    claim_id INT REFERENCES claims(claim_id),
    category_id INT REFERENCES product_category_catalog(catalog_id),
    company_id INT REFERENCES companies(company_id),
    reason VARCHAR(100) NOT NULL CHECK (reason IN (
        'scheduled_refresh',
        'human_feedback',
        'fact_check_failures',
        'source_updates',
        'new_competitor_announcement',
        'manual_request'
    )),
    instructions TEXT,
    feedback_ids INT[],
    status VARCHAR(50) DEFAULT 'queued' CHECK (status IN (
        'queued',
        'in_progress',
        'completed',
        'failed',
        'cancelled'
    )),
    result_generation_id INT REFERENCES battlecard_generations(generation_id),
    error_message TEXT,
    requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    requested_by VARCHAR(255) NOT NULL,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    processed_by VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS audit_log (
    log_id SERIAL PRIMARY KEY,
    entity_type VARCHAR(50) NOT NULL,
    entity_id INT NOT NULL,
    action VARCHAR(50) NOT NULL,
    old_values JSONB,
    new_values JSONB,
    actor VARCHAR(255) NOT NULL,
    actor_type VARCHAR(50) NOT NULL CHECK (actor_type IN ('human', 'agent', 'system')),
    logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prompt_versions (
    prompt_version_id SERIAL PRIMARY KEY,
    prompt_name VARCHAR(255) NOT NULL,
    prompt_version INT NOT NULL DEFAULT 1,
    prompt_file VARCHAR(500),
    prompt_text TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(prompt_name, content_hash)
);

CREATE TABLE IF NOT EXISTS workflow_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    competitor_name VARCHAR(255) NOT NULL,
    product_area VARCHAR(255) NOT NULL DEFAULT 'Data Platform',
    current_step INT NOT NULL DEFAULT 1,
    status VARCHAR(50) NOT NULL DEFAULT 'in_progress' CHECK (status IN (
        'in_progress', 'completed', 'failed', 'cancelled'
    )),
    generation_id INT REFERENCES battlecard_generations(generation_id),
    model_name VARCHAR(255) DEFAULT 'databricks-claude-sonnet-4',
    diffs_per_category INT DEFAULT 10,
    max_workers INT DEFAULT 5,
    step4_execution_mode VARCHAR(50) NOT NULL DEFAULT 'auto',
    step5_execution_mode VARCHAR(50) NOT NULL DEFAULT 'auto',
    step5_inline_fact_check BOOLEAN NOT NULL DEFAULT FALSE,
    pass1_prompt_version_id INT REFERENCES prompt_versions(prompt_version_id),
    pass2_prompt_version_id INT REFERENCES prompt_versions(prompt_version_id),
    pass1_prompt_template_version INT NOT NULL DEFAULT 3,
    pass2_prompt_template_version INT NOT NULL DEFAULT 2,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS workflow_steps (
    step_id SERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES workflow_sessions(session_id) ON DELETE CASCADE,
    step_number INT NOT NULL,
    step_name VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending', 'ready', 'in_progress', 'waiting_human', 'completed', 'failed', 'skipped'
    )),
    progress_current INT DEFAULT 0,
    progress_total INT DEFAULT 0,
    progress_message TEXT,
    error_message TEXT,
    error_details JSONB,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    heartbeat_at TIMESTAMP,
    worker_id VARCHAR(100),
    UNIQUE(session_id, step_number)
);

CREATE TABLE IF NOT EXISTS workflow_artifacts (
    artifact_id SERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES workflow_sessions(session_id) ON DELETE CASCADE,
    step_number INT NOT NULL,
    artifact_type VARCHAR(100) NOT NULL,
    artifact_name VARCHAR(500),
    artifact_content TEXT,
    artifact_metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS agent_turns (
    turn_id SERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES workflow_sessions(session_id) ON DELETE CASCADE,
    step_number INT NOT NULL,
    turn_type VARCHAR(50) NOT NULL CHECK (turn_type IN (
        'user_input', 'system_prompt', 'model_output', 'tool_result'
    )),
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content_type VARCHAR(100) NOT NULL,
    content_preview VARCHAR(500),
    token_count INT,
    model_name VARCHAR(255),
    artifact_id INT REFERENCES workflow_artifacts(artifact_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS product_mappings (
    mapping_id SERIAL PRIMARY KEY,
    catalog_id INT NOT NULL REFERENCES product_category_catalog(catalog_id),
    vendor VARCHAR(50) NOT NULL CHECK (vendor IN ('databricks', 'competitor')),
    competitor_name VARCHAR(255),
    product_name VARCHAR(500) NOT NULL,
    product_description TEXT,
    display_order INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK (
        (vendor = 'databricks' AND competitor_name IS NULL) OR
        (vendor = 'competitor' AND competitor_name IS NOT NULL)
    )
);

CREATE TABLE IF NOT EXISTS session_category_selections (
    selection_id SERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES workflow_sessions(session_id) ON DELETE CASCADE,
    catalog_id INT NOT NULL REFERENCES product_category_catalog(catalog_id),
    inclusion_type VARCHAR(30) NOT NULL CHECK (inclusion_type IN (
        'core_product_category', 'cross_platform_capability', 'skip'
    )),
    display_order INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, catalog_id)
);

CREATE TABLE IF NOT EXISTS prompt_templates (
    template_id SERIAL PRIMARY KEY,
    template_name VARCHAR(255) NOT NULL,
    template_type VARCHAR(50) NOT NULL,
    version_label VARCHAR(100),
    description TEXT,
    template_text TEXT NOT NULL,
    variables TEXT,
    response_schema_json JSONB,
    request_config_json JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    is_default BOOLEAN DEFAULT FALSE,
    display_order INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_prompt_templates_name ON prompt_templates(template_name);

CREATE TABLE IF NOT EXISTS pass2_debug_logs (
    log_id SERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES workflow_sessions(session_id) ON DELETE CASCADE,
    skeleton_index INT NOT NULL,
    key_differentiator VARCHAR(500),
    category VARCHAR(255),
    rendered_prompt TEXT,
    api_request_json TEXT,
    api_response_raw TEXT,
    structured_output TEXT,
    response_type VARCHAR(20) CHECK (response_type IN ('dict', 'list', 'error', 'unknown')),
    was_list_fixed BOOLEAN DEFAULT FALSE,
    lakebase_saved BOOLEAN DEFAULT FALSE,
    lakebase_error TEXT,
    error_message TEXT,
    processing_time_ms INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, skeleton_index)
);
CREATE INDEX IF NOT EXISTS idx_pass2_debug_session ON pass2_debug_logs(session_id);

CREATE TABLE IF NOT EXISTS eval_datasets (
    dataset_id SERIAL PRIMARY KEY,
    dataset_name VARCHAR(255) NOT NULL UNIQUE,
    source_generation_id INT REFERENCES battlecard_generations(generation_id),
    source_battlecard_uuid UUID,
    notes TEXT,
    created_by VARCHAR(255) NOT NULL DEFAULT 'system',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS eval_dataset_items (
    item_id SERIAL PRIMARY KEY,
    dataset_id INT NOT NULL REFERENCES eval_datasets(dataset_id) ON DELETE CASCADE,
    key_diff_name VARCHAR(500) NOT NULL,
    category_name VARCHAR(255) NOT NULL,
    company_type VARCHAR(50) NOT NULL CHECK (company_type IN ('databricks', 'competitor')),
    rating VARCHAR(20),
    headline TEXT,
    details_text TEXT,
    verdict_label VARCHAR(50),
    source_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_eval_dataset_items_dataset ON eval_dataset_items(dataset_id);

CREATE TABLE IF NOT EXISTS eval_runs (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status VARCHAR(50) NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'completed', 'failed')),
    benchmark_config JSONB,
    requested_by VARCHAR(255) NOT NULL DEFAULT 'system',
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS eval_run_results (
    result_id SERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES eval_runs(run_id) ON DELETE CASCADE,
    session_id UUID REFERENCES workflow_sessions(session_id) ON DELETE SET NULL,
    dataset_id INT REFERENCES eval_datasets(dataset_id),
    scenario_label VARCHAR(100) NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    pass1_prompt_template_version INT NOT NULL,
    pass2_prompt_template_version INT NOT NULL,
    step4_execution_mode VARCHAR(50) NOT NULL,
    step5_execution_mode VARCHAR(50) NOT NULL,
    step5_inline_fact_check BOOLEAN NOT NULL DEFAULT FALSE,
    core_categories INT NOT NULL,
    diffs_per_category INT NOT NULL,
    expected_diffs INT NOT NULL,
    generated_diffs INT NOT NULL,
    duration_seconds NUMERIC(12,3),
    total_tokens INT,
    input_tokens INT,
    output_tokens INT,
    eval_claim_match_rate NUMERIC(6,3),
    eval_rating_match_rate NUMERIC(6,3),
    eval_verdict_match_rate NUMERIC(6,3),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_eval_run_results_run ON eval_run_results(run_id);
"""


PRODUCT_CATEGORY_CATALOG_SEED = [
    ("Data Engineering (Ingestion, ETL, Orchestration)", "Full data engineering stack covering data ingestion pipelines, ETL/ELT transformations, and workflow orchestration", True, False),
    ("Data Engineering (ETL & Orchestration - excluding ingestion)", "ETL/ELT transformations and workflow orchestration without native ingestion capabilities", True, False),
    ("Data Engineering Platform (Ingestion, ETL, Real-time and Orchestration)", "Comprehensive data engineering platform covering ingestion, ETL/ELT, real-time streaming, and workflow orchestration", True, False),
    ("Data Engineering (ETL, Real-time and Orchestration)", "Data engineering capabilities for ETL/ELT, real-time streaming, and orchestration without native ingestion", True, False),
    ("Data Ingestion", "Dedicated data ingestion capabilities including batch loading, file ingestion, and source connectivity", True, True),
    ("Data Ingestion & Integration", "Connectors, CDC, streaming ingestion, and data integration from diverse sources", True, True),
    ("ETL", "Extract, transform, and load capabilities for batch data processing and transformation pipelines", True, False),
    ("Real-time & Streaming", "Real-time data processing, stream ingestion, event-driven architectures, and continuous data pipelines", True, True),
    ("Orchestration", "Workflow scheduling, pipeline orchestration, dependency management, and job coordination", True, True),
    ("Analytics Engineering", "Data modeling, transformation logic, semantic layers, and metrics definitions for analytics-ready datasets", True, False),
    ("Data Warehousing & Lakehouse (including Open Table Formats)", "Structured data storage, analytics warehousing, and lakehouse architectures with open table format support", True, False),
    ("Data Science & ML", "Machine learning model development, training, experimentation, and deployment", True, False),
    ("Business Intelligence & Reporting", "Dashboards, visualizations, semantic layers, and business reporting tools", True, True),
    ("Internal AI Assistants", "Conversational AI, copilots, coding agents, and AI-powered assistants embedded within the platform for interactive user support", False, False),
    ("ETL / SQL Gen AI Features", "AI functions callable directly in SQL queries, AI model inference in Spark DataFrames, and GenAI-powered batch data extraction and transformation", False, False),
    ("Catalog & Governance", "Data cataloging, metadata management, lineage, access control, and compliance", True, True),
    ("Open Table Formats", "Delta, Iceberg, Hudi table format support for interoperability", False, True),
    ("Serverless Compute", "On-demand, auto-scaling compute resources for SQL queries and data processing without infrastructure management", True, False),
    ("Serverless GPUs", "On-demand GPU resources for ML training, inference, and AI workloads without provisioning infrastructure", False, False),
    ("Agent Development Platform", "Tools and frameworks for building, deploying, and managing autonomous AI agents with tool use, memory, and orchestration capabilities", False, False),
    ("Databases (OLTP & Vector Search)", "Transactional databases with integrated vector search capabilities for hybrid operational and AI workloads", False, True),
    ("Databases (OLTP only)", "Transactional databases for operational workloads supporting ACID transactions and low-latency reads/writes", False, True),
    ("Databases (Vector Search only)", "Specialized vector databases for similarity search, embeddings storage, and AI/ML retrieval workloads", False, True),
    ("Internal Data & AI Apps", "Low-code/no-code tools for building internal applications, dashboards, and AI-powered workflows for business users", False, False),
    ("MCP (Model Context Protocol)", "Standardized protocol for connecting AI models to external tools, data sources, and services with consistent context sharing", False, True),
    ("Monitoring & Observability", "Platform monitoring, query performance tracking, cost management, resource utilization, and system health observability", True, False),
    ("General Platform Capabilities", "Cross-cutting platform features including security, identity management, multi-cloud support, APIs, SDKs, and foundational infrastructure services", True, False),
]


def init_db():
    # First, try running the full schema SQL in one go (fastest path for fresh DBs).
    try:
        with ENGINE.begin() as conn:
            conn.execute(text(SCHEMA_SQL))
    except Exception:
        # If the monolithic schema fails (e.g. ownership issues on existing tables),
        # split into individual statements and run each in its own transaction.
        logger.info("Full schema SQL failed; running statements individually...")
        for stmt in SCHEMA_SQL.split(";"):
            stmt = stmt.strip()
            if not stmt:
                continue
            try:
                with ENGINE.begin() as conn:
                    conn.execute(text(stmt))
            except Exception as e:
                logger.warning("Schema init (non-fatal): %s", e)

    # Migrations: add columns that may not exist on older schemas.
    # Each runs in its own transaction to avoid one failure aborting all.
    migrations = [
        "ALTER TABLE workflow_sessions ADD COLUMN IF NOT EXISTS pass2_prompt_template_version INT NOT NULL DEFAULT 2",
        "ALTER TABLE workflow_sessions ADD COLUMN IF NOT EXISTS pass1_prompt_template_version INT NOT NULL DEFAULT 2",
        "ALTER TABLE workflow_sessions ADD COLUMN IF NOT EXISTS step4_execution_mode VARCHAR(50) NOT NULL DEFAULT 'auto'",
        "ALTER TABLE workflow_sessions ADD COLUMN IF NOT EXISTS step5_execution_mode VARCHAR(50) NOT NULL DEFAULT 'auto'",
        "ALTER TABLE workflow_sessions ADD COLUMN IF NOT EXISTS step5_inline_fact_check BOOLEAN NOT NULL DEFAULT FALSE",
        """CREATE TABLE IF NOT EXISTS claim_detail_verbose_items (
            verbose_item_id SERIAL PRIMARY KEY,
            detail_item_id INT NOT NULL REFERENCES claim_detail_items(detail_item_id) ON DELETE CASCADE,
            item_order INT NOT NULL,
            item_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        "CREATE INDEX IF NOT EXISTS idx_verbose_items_detail ON claim_detail_verbose_items(detail_item_id)",
        "ALTER TABLE evidence ADD COLUMN IF NOT EXISTS detail_item_id INT REFERENCES claim_detail_items(detail_item_id)",
        "ALTER TABLE evidence ADD COLUMN IF NOT EXISTS verbose_item_id INT REFERENCES claim_detail_verbose_items(verbose_item_id)",
        "ALTER TABLE evidence ADD COLUMN IF NOT EXISTS claim_subfield VARCHAR(30) NOT NULL DEFAULT 'l200'",
        "ALTER TABLE evidence ADD COLUMN IF NOT EXISTS citation_id VARCHAR(120)",
        "ALTER TABLE evidence ADD COLUMN IF NOT EXISTS citation_order INT",
        "ALTER TABLE human_reviews ADD COLUMN IF NOT EXISTS detail_item_id INT REFERENCES claim_detail_items(detail_item_id)",
        "ALTER TABLE key_differentiators ADD COLUMN IF NOT EXISTS generation_id INT REFERENCES battlecard_generations(generation_id)",
        # Migrate category references from legacy product_categories -> product_category_catalog
        "ALTER TABLE key_differentiators DROP CONSTRAINT IF EXISTS key_differentiators_category_id_fkey",
        "ALTER TABLE regeneration_requests DROP CONSTRAINT IF EXISTS regeneration_requests_category_id_fkey",
        """DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'product_categories'
            ) THEN
                UPDATE key_differentiators kd
                SET category_id = pcc.catalog_id
                FROM product_categories pc
                JOIN product_category_catalog pcc
                  ON lower(trim(pc.category_name)) = lower(trim(pcc.category_name))
                WHERE kd.category_id = pc.category_id;

                UPDATE regeneration_requests rr
                SET category_id = pcc.catalog_id
                FROM product_categories pc
                JOIN product_category_catalog pcc
                  ON lower(trim(pc.category_name)) = lower(trim(pcc.category_name))
                WHERE rr.category_id = pc.category_id;
            END IF;
        END $$""",
        "UPDATE key_differentiators SET category_id = NULL WHERE category_id IS NOT NULL AND category_id NOT IN (SELECT catalog_id FROM product_category_catalog)",
        "UPDATE regeneration_requests SET category_id = NULL WHERE category_id IS NOT NULL AND category_id NOT IN (SELECT catalog_id FROM product_category_catalog)",
        "ALTER TABLE key_differentiators ADD CONSTRAINT key_differentiators_category_id_fkey FOREIGN KEY (category_id) REFERENCES product_category_catalog(catalog_id)",
        "ALTER TABLE regeneration_requests ADD CONSTRAINT regeneration_requests_category_id_fkey FOREIGN KEY (category_id) REFERENCES product_category_catalog(catalog_id)",
        "DROP TABLE IF EXISTS product_categories",
        "CREATE INDEX IF NOT EXISTS idx_key_diffs_generation ON key_differentiators(generation_id)",
        "ALTER TABLE evidence DROP CONSTRAINT IF EXISTS evidence_traces_to_field_check",
        "ALTER TABLE evidence ADD CONSTRAINT evidence_traces_to_field_check CHECK (traces_to_field IN ('headline', 'description', 'detail_item'))",
        "ALTER TABLE evidence DROP CONSTRAINT IF EXISTS evidence_claim_subfield_check",
        "ALTER TABLE evidence ADD CONSTRAINT evidence_claim_subfield_check CHECK (claim_subfield IN ('headline', 'reasoning', 'l200', 'l300'))",
        "ALTER TABLE workflow_steps ADD COLUMN IF NOT EXISTS heartbeat_at TIMESTAMP",
        "ALTER TABLE workflow_steps ADD COLUMN IF NOT EXISTS worker_id VARCHAR(100)",
        # L200/L300 detail text support
        "ALTER TABLE claim_detail_items ADD COLUMN IF NOT EXISTS item_text_verbose TEXT",
        "ALTER TABLE prompt_templates ADD COLUMN IF NOT EXISTS response_schema_json JSONB",
        "ALTER TABLE prompt_templates ADD COLUMN IF NOT EXISTS request_config_json JSONB",
        # Error storage for workflow steps
        "ALTER TABLE workflow_steps ADD COLUMN IF NOT EXISTS last_error TEXT",
        "ALTER TABLE workflow_steps ADD COLUMN IF NOT EXISTS error_count INT DEFAULT 0",
        # Pass 2 debug logs table for debugging LLM responses
        """CREATE TABLE IF NOT EXISTS pass2_debug_logs (
            log_id SERIAL PRIMARY KEY,
            session_id UUID NOT NULL REFERENCES workflow_sessions(session_id) ON DELETE CASCADE,
            skeleton_index INT NOT NULL,
            key_differentiator VARCHAR(500),
            category VARCHAR(255),
            rendered_prompt TEXT,
            api_request_json TEXT,
            api_response_raw TEXT,
            structured_output TEXT,
            response_type VARCHAR(20) CHECK (response_type IN ('dict', 'list', 'error', 'unknown')),
            was_list_fixed BOOLEAN DEFAULT FALSE,
            lakebase_saved BOOLEAN DEFAULT FALSE,
            lakebase_error TEXT,
            error_message TEXT,
            processing_time_ms INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(session_id, skeleton_index)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_pass2_debug_session ON pass2_debug_logs(session_id)",
        """CREATE TABLE IF NOT EXISTS eval_datasets (
            dataset_id SERIAL PRIMARY KEY,
            dataset_name VARCHAR(255) NOT NULL UNIQUE,
            source_generation_id INT REFERENCES battlecard_generations(generation_id),
            source_battlecard_uuid UUID,
            notes TEXT,
            created_by VARCHAR(255) NOT NULL DEFAULT 'system',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS eval_dataset_items (
            item_id SERIAL PRIMARY KEY,
            dataset_id INT NOT NULL REFERENCES eval_datasets(dataset_id) ON DELETE CASCADE,
            key_diff_name VARCHAR(500) NOT NULL,
            category_name VARCHAR(255) NOT NULL,
            company_type VARCHAR(50) NOT NULL CHECK (company_type IN ('databricks', 'competitor')),
            rating VARCHAR(20),
            headline TEXT,
            details_text TEXT,
            verdict_label VARCHAR(50),
            source_count INT DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        "CREATE INDEX IF NOT EXISTS idx_eval_dataset_items_dataset ON eval_dataset_items(dataset_id)",
        """CREATE TABLE IF NOT EXISTS eval_runs (
            run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            status VARCHAR(50) NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'completed', 'failed')),
            benchmark_config JSONB,
            requested_by VARCHAR(255) NOT NULL DEFAULT 'system',
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS eval_run_results (
            result_id SERIAL PRIMARY KEY,
            run_id UUID NOT NULL REFERENCES eval_runs(run_id) ON DELETE CASCADE,
            session_id UUID REFERENCES workflow_sessions(session_id) ON DELETE SET NULL,
            dataset_id INT REFERENCES eval_datasets(dataset_id),
            scenario_label VARCHAR(100) NOT NULL,
            model_name VARCHAR(255) NOT NULL,
            pass1_prompt_template_version INT NOT NULL,
            pass2_prompt_template_version INT NOT NULL,
            step4_execution_mode VARCHAR(50) NOT NULL,
            step5_execution_mode VARCHAR(50) NOT NULL,
            step5_inline_fact_check BOOLEAN NOT NULL DEFAULT FALSE,
            core_categories INT NOT NULL,
            diffs_per_category INT NOT NULL,
            expected_diffs INT NOT NULL,
            generated_diffs INT NOT NULL,
            duration_seconds NUMERIC(12,3),
            total_tokens INT,
            input_tokens INT,
            output_tokens INT,
            eval_claim_match_rate NUMERIC(6,3),
            eval_rating_match_rate NUMERIC(6,3),
            eval_verdict_match_rate NUMERIC(6,3),
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        "CREATE INDEX IF NOT EXISTS idx_eval_run_results_run ON eval_run_results(run_id)",
    ]
    for migration in migrations:
        try:
            with ENGINE.begin() as conn:
                conn.execute(text(migration))
        except Exception:
            pass  # column already exists or DB doesn't support IF NOT EXISTS

    # Seed product_category_catalog if empty
    _seed_product_category_catalog()

    # Seed prompt_templates if empty
    _seed_prompt_templates()


def _seed_product_category_catalog():
    """Populate product_category_catalog with default rows if empty."""
    with ENGINE.begin() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM product_category_catalog")).scalar()
        if count and int(count) > 0:
            return
        for idx, (name, desc, is_core, is_cross) in enumerate(PRODUCT_CATEGORY_CATALOG_SEED):
            conn.execute(
                text(
                    "INSERT INTO product_category_catalog (category_name, category_description, is_core_product_category, is_cross_platform_capability, display_order) "
                    "VALUES (:name, :desc, :is_core, :is_cross, :ord)"
                ),
                {"name": name, "desc": desc, "is_core": is_core, "is_cross": is_cross, "ord": idx},
            )
    logger.info("Seeded product_category_catalog with %d rows", len(PRODUCT_CATEGORY_CATALOG_SEED))


def _seed_prompt_templates():
    """Populate prompt_templates with built-in templates if empty.

    Also repairs existing placeholder rows (from earlier seeds where template
    files were missing) by replacing them with inline fallback content.
    """
    import re

    def _read_file_safe(path):
        """Read file contents or return None if missing."""
        try:
            if os.path.isfile(path):
                with open(path) as f:
                    return f.read()
        except Exception:
            pass
        return None

    # ── Inline fallback templates ──
    # These are embedded directly so the seed works even when template files
    # don't exist on disk (e.g. on the Databricks Apps platform).

    _PASS1_V1_INLINE = """\
You plan L200 critical differentiator slides for a **platform battlecard** comparing Databricks to {{competitor}}.

## Context
Competitor: {{competitor}}
Product area: {{product_area}}
Comparison: {{comparison}}

## Audience
This battlecard is for **C-suite executives** (CIO, CTO, CDO, VP Data/AI) and **data/ML/AI practitioners** (data engineers, ML engineers, analytics engineers, data scientists). Key differentiators must resonate with both audiences:
- C-suite cares about: strategic platform direction, total cost of ownership, vendor risk, governance posture, time-to-value, and AI/ML readiness.
- Practitioners care about: performance benchmarks, developer experience, tooling maturity, open standards, and operational reliability.

## Product Categories
The following product categories define the structure of this platform battlecard. You MUST generate exactly **{{diffs_per_category}} key differentiators per category**, for a total of {{total_diffs}} differentiators.

{{product_categories}}

## Directives
{{directives}}

## Additional Context
The following context documents are provided as XML-tagged sections.
Each tag indicates the document type (competitive_directive, battlecard_archive, product_categories, key_differentiator_themes, or context).
Attributes include the document name, type classification, scope, and whether it was human_provided or agent_generated.
Use these documents to inform your analysis. Cite them as sources where relevant.

{{context}}

## Task
Generate exactly **{{diffs_per_category}} key differentiators per product category** ({{total_diffs}} total).
For each, provide ONLY the planning skeleton — detailed headlines, details, reasoning, citations, and sources will be generated separately in a second pass.

## Output Format
Return ONLY a JSON array. Each object must have this exact shape:

```json
{
  "id": "<Category>_<Differentiator> with underscores",
  "competitor": "{{competitor}}",
  "category": "<product category from the list above>",
  "rank": 1,
  "key_differentiator": "<3-6 word differentiator name — NO brand names or product names>",
  "description": "<short buyer-focused description: why this capability matters to end users/buyers — NO brand names, NO vendor comparisons>",
  "selection_reasoning": "<why this differentiator was selected — reference audience needs>",
  "rank_reasoning": "<why this rank position within its category>",
  "directive_alignment": "<which directive points this aligns with, or 'N/A'>",
  "databricks_rating": "strong_advantage|advantage|partial|disadvantage",
  "competitor_rating": "strong_advantage|advantage|partial|disadvantage"
}
```

## Rules
1. Generate exactly **{{diffs_per_category}} differentiators per product category**. The total count MUST equal {{total_diffs}}.
2. The `category` field MUST be one of the product categories listed above. Every category must have exactly {{diffs_per_category}} differentiators.
3. Within each category, rank differentiators 1-{{diffs_per_category}} by importance. The `rank` field resets to 1 for each category.
4. Databricks rating must be >= competitor rating for the majority of differentiators.
5. Be fair — include 1-2 areas per category where the competitor has genuine strengths.
6. Balance differentiators between what matters to C-suite (TCO, governance, strategic direction, vendor risk) and practitioners (performance, DX, tooling, open standards).
7. Align differentiators with the provided directives where applicable. Reference specific directive points in the `directive_alignment` field.
8. Each differentiator must have a unique `id` (format: Category_KeyDiff with underscores replacing spaces).
9. **NO BRAND NAMES**: The `key_differentiator` and `description` fields must NOT contain any brand names, product names, or branded feature names. Use generic capability descriptions instead.
10. **BUYER-FOCUSED descriptions**: The `description` field should be short (under 15 words) and explain why this capability matters to the end buyer/user.

Return ONLY the JSON array. No markdown fences, no explanation text.
"""

    _PASS1_V2_INLINE = """\
You plan L200 critical differentiator slides for a **platform battlecard** comparing Databricks to {{competitor}}.

## Context
Competitor: {{competitor}}
Product area: {{product_area}}
Comparison: {{comparison}}

## Audience
This battlecard is for **C-suite executives** (CIO, CTO, CDO, VP Data/AI) and **data/ML/AI practitioners** (data engineers, ML engineers, analytics engineers, data scientists). Key differentiators must resonate with both audiences:
- C-suite cares about: strategic platform direction, total cost of ownership, vendor risk, governance posture, time-to-value, and AI/ML readiness.
- Practitioners care about: performance benchmarks, developer experience, tooling maturity, open standards, and operational reliability.

## Product Categories
The following product categories define the structure of this platform battlecard. You MUST generate exactly **{{diffs_per_category}} key differentiators per category**, for a total of {{total_diffs}} differentiators.

{{product_categories}}

## Key Differentiator Theme Guidance
Use the following themes as guidance when selecting key differentiators. Most differentiators should map to one of these themes, but you MAY choose differentiators outside these themes for specific product categories where it makes sense — justify with clear reasoning about who cares and why.

| Theme | Who Cares | Why It Matters |
|-------|-----------|----------------|
| Serverless | Both | Operational simplicity, no capacity planning, instant scale |
| Open Formats & Standards | Both | Avoid lock-in, portability, future-proof investments |
| AI Assistants | Both | Productivity, natural language access to data |
| Intelligent Optimization | Both | Self-tuning, auto-scaling, ML-driven performance |
| Performance & TCO | C-suite + Practitioners | Query speed and cost efficiency together |
| Unified Platform | C-suite | Fewer tools, lower integration complexity |
| Interoperability | Practitioners | Works with existing tools and data sources |
| Governance & Security | C-suite | Access controls, lineage, compliance (HIPAA, SOC2, GDPR) |
| Real-time & Streaming | Practitioners | Handle streaming alongside batch workloads |
| Data Sharing | Both | Cross-org collaboration without copying data |
| Developer Experience | Practitioners | IDE support, debugging, CI/CD, daily productivity |
| GenAI & LLM Support | Both | RAG, vector search, fine-tuning, compound AI systems |
| Pricing Transparency | C-suite | Predictable costs, no surprise bills |
| Multi-cloud | C-suite | Vendor risk mitigation, cloud flexibility |

## Directives
{{directives}}

## Additional Context
The following context documents are provided as XML-tagged sections.
Each tag indicates the document type (competitive_directive, battlecard_archive, product_categories, key_differentiator_themes, or context).
Attributes include the document name, type classification, scope, and whether it was human_provided or agent_generated.
Use these documents to inform your analysis. Cite them as sources where relevant.

{{context}}

## Task
Generate exactly **{{diffs_per_category}} key differentiators per product category** ({{total_diffs}} total).
For each, provide ONLY the planning skeleton — detailed headlines, details, reasoning, citations, and sources will be generated separately in a second pass.

## Output Format
Return ONLY a JSON array. Each object must have this exact shape:

```json
{
  "id": "<Category>_<Differentiator> with underscores",
  "competitor": "{{competitor}}",
  "category": "<product category from the list above>",
  "rank": 1,
  "key_differentiator": "<2-4 word differentiator name — see naming rules below>",
  "description": "<1 sentence, max 12 words — the benefit or value this capability delivers to buyers>",
  "selection_reasoning": "<who cares about this (C-suite, practitioners, or both) and WHY it matters for platform selection>",
  "rank_reasoning": "<why this rank position within its category>",
  "directive_alignment": "<which directive points this aligns with, or 'N/A'>",
  "databricks_rating": "strong_advantage|advantage|partial|disadvantage",
  "competitor_rating": "strong_advantage|advantage|partial|disadvantage"
}
```

## Key Differentiator Naming Rules — CRITICAL

The `key_differentiator` field is the title shown on the battlecard slide. It must be:

1. **2-4 words maximum** (ideally 2-3). This is a category label, not a sentence.
2. **NO brand names, product names, or feature names** (e.g. NOT "Delta Live Tables", NOT "Fabric Capacity", NOT "Databricks Unity Catalog").
3. **Generic capability labels** that any platform could be evaluated against.
4. **Title case** formatting.

### Examples of GOOD key_differentiator names:
- "Ease of Use"
- "Data Connectors"
- "Query Performance"
- "Cost Transparency"
- "Streaming Support"
- "Data Governance"
- "Open Standards"
- "Auto-Optimization"
- "Multi-Cloud Support"
- "Developer Tooling"

### Examples of BAD key_differentiator names — DO NOT write like this:
- "Native Change Data Capture Pipeline Support" (too long, too specific)
- "Serverless SQL Warehouse Auto-Scaling" (too long, includes branded concepts)
- "Delta Lake Open Table Format" (brand name)

## Description Rules

The `description` field explains WHY this differentiator matters. It must be:

1. **One sentence, max 12 words.**
2. **Benefit-focused** — what value does this deliver to the buyer?
3. **No brand names or vendor comparisons.**

## Selection Reasoning Rules

The `selection_reasoning` field must clearly state:
1. **WHO cares** — C-suite, practitioners, or both
2. **WHY it matters** for choosing a data platform

## Rules
1. Generate exactly **{{diffs_per_category}} differentiators per product category**. The total count MUST equal {{total_diffs}}.
2. The `category` field MUST be one of the product categories listed above. Every category must have exactly {{diffs_per_category}} differentiators.
3. Within each category, rank differentiators 1-{{diffs_per_category}} by importance. The `rank` field resets to 1 for each category.
4. Databricks rating must be >= competitor rating for the majority of differentiators.
5. Be fair — include 1-2 areas per category where the competitor has genuine strengths.
6. Balance differentiators between what matters to C-suite and practitioners.
7. Align differentiators with the provided directives where applicable.
8. Each differentiator must have a unique `id` (format: Category_KeyDiff with underscores replacing spaces).
9. **Most differentiators should map to the theme guidance table above**, but you may introduce themes outside that list when clearly important.
10. **NO BRAND NAMES** in `key_differentiator` or `description` fields.
11. **KEY DIFF NAMES must be 2-4 words** (ideally 2-3).
12. **DESCRIPTIONS must be benefit-focused**, max 12 words.

Return ONLY the JSON array. No markdown fences, no explanation text.
"""

    # V3 is like V2 but treats Cross-Platform Capabilities as context woven into
    # Core Product Category slides, not separate slides.
    _PASS1_V3_INLINE = _PASS1_V2_INLINE.replace(
        "## Task\nGenerate exactly",
        "## Cross-Platform Context\n"
        "Cross-Platform Capabilities (multi-cloud, governance, security, pricing) should NOT be treated as a separate product category. "
        "Instead, weave cross-platform capabilities into the core product category slides where they naturally fit. "
        "For example, governance differentiators belong in the relevant product category (e.g., Data Warehousing governance vs. ML governance), "
        "not in a standalone 'Cross-Platform' section.\n\n"
        "## Task\nGenerate exactly"
    )

    # V4 adds per-category parallel execution instruction
    _PASS1_V4_INLINE = _PASS1_V3_INLINE.replace(
        "## Task\nGenerate exactly",
        "## Execution Mode\n"
        "This prompt will be executed **once per product category** in parallel. "
        "You will receive a single product category and its cross-platform context. "
        "Generate exactly **{{diffs_per_category}} key differentiators** for the given category only.\n\n"
        "## Task\nGenerate exactly"
    )

    _PASS1_INLINE_FALLBACKS = {
        1: _PASS1_V1_INLINE,
        2: _PASS1_V2_INLINE,
        3: _PASS1_V3_INLINE,
        4: _PASS1_V4_INLINE,
    }

    # ── Pass 2 inline fallbacks ──
    _PASS2_V1_INLINE = """\
You fill in detailed competitive analysis for a single key differentiator on a Databricks vs {{competitor}} platform battlecard.

## Key Differentiator
- **Category**: {{category}}
- **Key Differentiator**: {{key_differentiator}}
- **Description**: {{description}}
- **Databricks Rating**: {{databricks_rating}}
- **Competitor Rating**: {{competitor_rating}}
- **Selection Reasoning**: {{selection_reasoning}}

## Audience
This battlecard is for **C-suite executives** and **data/ML/AI practitioners**.
- C-suite cares about: strategic platform direction, total cost of ownership, vendor risk, governance posture, time-to-value.
- Practitioners care about: performance benchmarks, developer experience, tooling maturity, open standards, operational reliability.

## Directives
{{directives}}

## Additional Context
{{context}}

## Task
Generate the full detail for this single differentiator. Include compelling headlines, detailed reasoning for each rating, and properly cited sources.

### Writing style for details fields
Each details field must be a concise technical description explaining the capability or limitation. Include specific technical details, version numbers, benchmarks, or architecture details where available.

### Output Format
Return a JSON object with these fields:
- `databricks_headline`: 6-10 word headline for Databricks position
- `databricks_details`: 2-3 sentences of technical detail with citations
- `databricks_reasoning`: Why this rating for Databricks
- `competitor_headline`: 6-10 word headline for competitor position
- `competitor_details`: 2-3 sentences of technical detail with citations
- `competitor_reasoning`: Why this rating for competitor
- `citations`: Array of citation objects with source, url, verdict, rationale

Return ONLY the JSON object. No markdown fences, no explanation text.
"""

    # V5 — Per-Category L200 Only (8 parallel calls instead of 80)
    _PASS2_V5_INLINE = """\
You generate L200 details for ALL key differentiators in the **{{category}}** category for a Databricks vs {{competitor}} battlecard.

## Execution Mode
This prompt processes **ALL key differentiators for one category** in a single call.
You will receive {{num_diffs}} key differentiators for the {{category}} category.
Generate L200 details (1-2 concise bullets) for EACH differentiator.

## Category: {{category}}

## Key Differentiators to Process
{{key_diffs_json}}

## Audience
This battlecard is for **C-suite executives** (CIO, CTO, CDO, VP Data/AI) and **data/ML/AI practitioners**.
- C-suite cares about: strategic direction, TCO, vendor risk, governance, time-to-value, AI readiness.
- Practitioners care about: performance, developer experience, tooling, open standards, reliability.

## Additional Context
{{context}}

## L200 Writing Style — CRITICAL

L200 bullets are **executive-ready sound bites**. They must be:
- **Outcome-focused**: Lead with the benefit or limitation, NOT technology
- **Concise**: 1-2 bullets, max 10 words each (aim for 6-8)
- **No jargon**: Write for a VP, not an engineer
- **Specific**: Include proof points where possible (benchmarks, percentages)

### Examples of GOOD L200 bullets:
**Semantic Understanding** (Databricks advantage):
- Genie leverages Unity Catalog semantics for accurate, context-aware answers without extensive configuration.

**Cost Model** (Databricks advantage):
- Scale analytics to every employee without seat licensing.
- Pay only for compute consumed.

**Query Performance** (Databricks advantage):
- 2-3x faster query execution in independent benchmarks.
- Photon engine optimizes automatically without tuning.

**BI Integration** (Competitor advantage):
- Native Power BI integration with no additional connectors needed.
- Familiar Excel-like experience for business users.

### Examples of BAD L200 bullets — DO NOT write like this:
- "Databricks uses Delta Lake with ACID transactions and time travel capabilities for reliable data management." (too technical, too long)
- "The platform provides comprehensive data governance features." (vague, no proof point)

## Output Format
Return a JSON array with one object per key differentiator. Each object must have:

```json
{
  "id": "<same id from input>",
  "databricks_headline": "<3-8 word headline for Databricks position>",
  "databricks_l200": ["<bullet 1: 6-8 words, max 10, outcome-focused>", "<bullet 2 (optional)>"],
  "databricks_reasoning": "<why this rating — 1-2 sentences with technical depth>",
  "competitor_headline": "<3-8 word headline for competitor position>",
  "competitor_l200": ["<bullet 1: 6-8 words, max 10, outcome-focused>", "<bullet 2 (optional)>"],
  "competitor_reasoning": "<why this rating — 1-2 sentences with technical depth>",
  "sources": ["<source 1>", "<source 2>"]
}
```

## Rules
1. Process ALL {{num_diffs}} key differentiators provided in the input.
2. **L200 bullets must be outcome-focused**, not technology-focused.
3. Each L200 array should have 1-2 bullets (not more).
4. Headlines should be punchy and memorable (3-8 words).
5. Put technical depth in the reasoning field, not in L200 bullets.
6. Be fair — if the competitor genuinely excels, reflect that honestly.

Return ONLY the JSON array. No markdown fences, no explanation text.
"""

    # V6 — Per-Category L200 + L300 (8 parallel calls instead of 80)
    _PASS2_V6_INLINE = """\
You generate L200 and L300 details for ALL key differentiators in the **{{category}}** category for a Databricks vs {{competitor}} battlecard.

## Execution Mode
This prompt processes **ALL key differentiators for one category** in a single call.
You will receive {{num_diffs}} key differentiators for the {{category}} category.
Generate L200 (executive summary) and L300 (technical detail) for EACH differentiator.

## Category: {{category}}

## Key Differentiators to Process
{{key_diffs_json}}

## Audience
This battlecard is for **C-suite executives** (CIO, CTO, CDO, VP Data/AI) and **data/ML/AI practitioners**.
- C-suite cares about: strategic direction, TCO, vendor risk, governance, time-to-value, AI readiness.
- Practitioners care about: performance, developer experience, tooling, open standards, reliability.

## Additional Context
{{context}}

## L200 vs L300 Writing Style — CRITICAL

**L200 (Executive Summary)** — For C-suite and quick scanning:
- Outcome-focused: Lead with benefit or limitation
- 1-2 bullets, max 10 words each (aim for 6-8)
- No technical jargon
- Include proof points (benchmarks, percentages)

**L300 (Technical Detail)** — For practitioners who want depth:
- 2-5 bullets with technical specifics
- Architecture details, version numbers, benchmarks
- Comparison points with specific features
- Can include product names (Delta Lake, Photon, etc.)

### L200 Examples (GOOD):
**Semantic Understanding**:
- Genie leverages Unity Catalog semantics for accurate, context-aware answers without extensive configuration.

**Cost Model**:
- Scale analytics to every employee without seat licensing.
- Pay only for compute consumed.

### L300 Examples (GOOD):
**Semantic Understanding**:
- Unity Catalog provides table/column descriptions, tags, and business glossary terms that Genie uses for disambiguation.
- AI-generated suggestions for common questions based on schema analysis.
- Natural language to SQL with 95%+ accuracy on trained domains.
- Supports follow-up questions with conversation context retention.

**Cost Model**:
- Serverless SQL warehouses auto-scale from 0 with sub-second startup.
- Per-query pricing with no minimum commits or idle charges.
- Predictive autoscaling reduces over-provisioning by 40-60%.

## Output Format
Return a JSON array with one object per key differentiator. Each object must have:

```json
{
  "id": "<same id from input>",
  "databricks_headline": "<3-8 word headline for Databricks position>",
  "databricks_l200": ["<bullet 1: 6-8 words, max 10, outcome-focused>", "<bullet 2 (optional)>"],
  "databricks_l300": ["<detail 1>", "<detail 2>", "...up to 5 technical bullets"],
  "databricks_reasoning": "<why this rating — 1-2 sentences>",
  "competitor_headline": "<3-8 word headline for competitor position>",
  "competitor_l200": ["<bullet 1: 6-8 words, max 10, outcome-focused>", "<bullet 2 (optional)>"],
  "competitor_l300": ["<detail 1>", "<detail 2>", "...up to 5 technical bullets"],
  "competitor_reasoning": "<why this rating — 1-2 sentences>",
  "sources": ["<source 1>", "<source 2>"]
}
```

## Rules
1. Process ALL {{num_diffs}} key differentiators provided in the input.
2. **L200 = outcome-focused** (1-2 bullets), **L300 = technical depth** (2-5 bullets).
3. L300 CAN include product names and technical details.
4. Headlines should be punchy and memorable (3-8 words).
5. Be fair — if the competitor genuinely excels, reflect that honestly.
6. Cite sources where possible.

Return ONLY the JSON array. No markdown fences, no explanation text.
"""

    _PASS2_V7_INLINE = """\
You generate L200 details for ALL key differentiators in the **{{category}}** category for a Databricks vs {{competitor}} battlecard.

## Execution Mode
This prompt is category-batched. Process every key differentiator provided in one deterministic JSON response.

## Category: {{category}}

## Key Differentiators to Process
{{key_diffs_json}}

## Additional Context
{{context}}

## Requirements
1. Return one object per input differentiator `id` and keep the same ordering.
2. Keep L200 bullets concise: max 10 words (aim for 6-8), 1-2 bullets each side.
3. Include citations and sources for every factual point in details and reasoning.
4. Use stable phrasing and avoid random stylistic variation.

## Output Format
Return ONLY a JSON array:
[
  {
    "id": "<same id from input>",
    "databricks_headline": "<3-8 words>",
    "databricks_l200": ["<bullet>", "<optional bullet>"],
    "databricks_reasoning": "<1-2 sentences>",
    "competitor_headline": "<3-8 words>",
    "competitor_l200": ["<bullet>", "<optional bullet>"],
    "competitor_reasoning": "<1-2 sentences>",
    "citations": {
      "databricks_details": [],
      "databricks_reasoning": [],
      "competitor_details": [],
      "competitor_reasoning": []
    },
    "sources": [
      {"index": 1, "title": "<source>", "url": "<url>", "type": "context", "accessed_at": "<ISO timestamp>"}
    ],
    "research_sources": []
  }
]
"""

    _PASS2_V8_INLINE = """\
You generate final claim content for ALL provided key differentiators in a Databricks vs {{competitor}} battlecard.

## Execution Mode
Single-call batched generation. Process every differentiator in one response.

## Key Differentiators to Process
{{key_diffs_json}}

## Additional Context
{{context}}

## Task
For each differentiator:
1. Generate concise L200 bullets for Databricks and competitor (max 10 words each, aim 6-8).
2. Add short reasoning for both sides.
3. Provide citations and sources.
4. Provide inline fact-check verdict metadata per citation (`verdict`, `confidence`, `verdict_rationale`).

## Output Format
Return ONLY a JSON array with one object per differentiator id:
[
  {
    "id": "<same id from input>",
    "databricks_headline": "<3-8 words>",
    "databricks_l200": ["<bullet>", "<optional bullet>"],
    "databricks_reasoning": "<1-2 sentences>",
    "competitor_headline": "<3-8 words>",
    "competitor_l200": ["<bullet>", "<optional bullet>"],
    "competitor_reasoning": "<1-2 sentences>",
    "citations": {
      "databricks_details": [
        {
          "citation_id": "cite_databricks_details_0_1",
          "detail_item_index": 0,
          "start_index": 0,
          "end_index": 20,
          "source_index": 1,
          "source_quote": "<exact support quote>",
          "verdict": "verified|unverified|disputed|outdated",
          "confidence": 0.0,
          "verdict_rationale": "<why verdict applies>"
        }
      ],
      "databricks_reasoning": [],
      "competitor_details": [],
      "competitor_reasoning": []
    },
    "sources": [
      {"index": 1, "title": "<source>", "url": "<url>", "type": "documentation", "accessed_at": "<ISO timestamp>"}
    ],
    "research_sources": []
  }
]
"""

    _PASS2_V9_INLINE = """\
You generate L200 and L300 details for ALL key differentiators in the **{{category}}** category for a Databricks vs {{competitor}} battlecard.

## Execution Mode
Category-batched generation. Process all differentiators for this category in one call.
You will receive {{num_diffs}} key differentiators for {{category}}.

## Category: {{category}}

## Key Differentiators to Process
{{key_diffs_json}}

## Additional Context
{{context}}

## Task
For each differentiator:
1. Generate concise L200 bullets for Databricks and competitor (max 10 words, aim 6-8).
2. Generate L300 technical detail bullets (2-5 bullets per side).
3. Add short reasoning for both sides.
4. Provide citations and sources.
5. Include inline fact-check metadata on each citation: `verdict`, `confidence`, `verdict_rationale`.

## Output Format
Return ONLY a JSON array with one object per differentiator id:
[
  {
    "id": "<same id from input>",
    "databricks_headline": "<3-8 words>",
    "databricks_l200": ["<bullet>", "<optional bullet>"],
    "databricks_l300": ["<technical detail>", "<optional detail>"],
    "databricks_reasoning": "<1-2 sentences>",
    "competitor_headline": "<3-8 words>",
    "competitor_l200": ["<bullet>", "<optional bullet>"],
    "competitor_l300": ["<technical detail>", "<optional detail>"],
    "competitor_reasoning": "<1-2 sentences>",
    "citations": {
      "databricks_details": [
        {
          "citation_id": "cite_databricks_details_0_1",
          "detail_item_index": 0,
          "start_index": 0,
          "end_index": 20,
          "source_index": 1,
          "source_quote": "<exact support quote>",
          "verdict": "verified|unverified|disputed|outdated",
          "confidence": 0.0,
          "verdict_rationale": "<why verdict applies>"
        }
      ],
      "databricks_reasoning": [],
      "competitor_details": [],
      "competitor_reasoning": []
    },
    "sources": [
      {"index": 1, "title": "<source>", "url": "<url>", "type": "documentation", "accessed_at": "<ISO timestamp>"}
    ],
    "research_sources": []
  }
]
"""

    _PASS2_V10_INLINE = """\
You generate L200 and L300 details with granular citations for ALL key differentiators in the **{{category}}** category for Databricks vs {{competitor}}.

## Execution Mode
Category-batched structured generation. Process all differentiators for this category in one call.
You will receive {{num_diffs}} differentiators for {{category}}.

## Category
{{category}}

## Key Differentiators to Process
{{key_diffs_json}}

## Additional Context
{{context}}

## Critical Rules
1. L200 bullets: max 10 words, target 6-8 words.
2. L300 bullets: technical proof, 2-5 bullets per L200 bullet when needed.
3. Every factual statement in headline, each L200 bullet, each L300 bullet, and reasoning needs citations.
4. Keep ids stable and return one object for each input id.

## Output Format
Return ONLY a JSON object:
{
  "claims": [
    {
      "id": "<same id from input>",
      "databricks": {
        "headline": {"text": "<3-8 words>", "citations": []},
        "l200": [
          {
            "text": "<6-8 words, max 10>",
            "citations": [],
            "l300": [
              {"text": "<technical detail>", "citations": []}
            ]
          }
        ],
        "reasoning": {"text": "<1-2 sentences>", "citations": []}
      },
      "competitor": {
        "headline": {"text": "<3-8 words>", "citations": []},
        "l200": [
          {
            "text": "<6-8 words, max 10>",
            "citations": [],
            "l300": [
              {"text": "<technical detail>", "citations": []}
            ]
          }
        ],
        "reasoning": {"text": "<1-2 sentences>", "citations": []}
      },
      "sources": [
        {"index": 1, "title": "<source>", "url": "<url>", "type": "documentation", "accessed_at": "<ISO timestamp>"}
      ],
      "research_sources": []
    }
  ]
}
"""

    _PASS2_INLINE_FALLBACKS = {
        1: _PASS2_V1_INLINE,
        5: _PASS2_V5_INLINE,
        6: _PASS2_V6_INLINE,
        7: _PASS2_V7_INLINE,
        8: _PASS2_V8_INLINE,
        9: _PASS2_V9_INLINE,
        10: _PASS2_V10_INLINE,
    }

    def _pass2_schema_config(version_num: int):
        if version_num == 10:
            return {
                "response_format": "json_schema",
                "schema_name": "l200_diff_detail_v10",
                "strict": True,
            }
        if version_num not in (5, 6, 7, 8, 9):
            return {
                "response_format": "json_schema",
                "schema_name": "l200_diff_detail",
                "strict": True,
            }
        return None

    def _pass2_request_config(version_num: int):
        return {
            "structured_output": version_num not in (5, 6, 7, 8, 9),
            "supports_category_batch": version_num in (5, 6, 7, 8, 9, 10),
            "supports_single_call": version_num in (8,),
        }

    _DIRECTIVE_INLINE = (
        "Read the following slides content about competing against {{competitor}}.\n\n"
        "Parse and create a max 10 to 25 bullets on how we (Databricks compete team) should "
        "take what has been taught to AE account executives and SAs on how to compete against {{competitor}}.\n\n"
        "Extract this directive as max 10 to 25 bullets that will be used to create an internal battlecard.\n\n"
        "Write the directive in markdown format."
    )

    # ── Build seed list ──
    seeds = []

    # Pass 1 templates
    for ver_num, cfg in PASS1_PROMPT_TEMPLATES.items():
        # Try file first, then inline fallback
        tpl_text = None
        if "template" in cfg:
            tpl_text = cfg["template"]
        elif "file" in cfg:
            tpl_text = _read_file_safe(cfg["file"])
        if not tpl_text:
            tpl_text = _PASS1_INLINE_FALLBACKS.get(ver_num)

        variables = sorted(set(re.findall(r'\{\{(\w+)\}\}', tpl_text))) if tpl_text else []

        seeds.append({
            "template_name": f"pass1_v{ver_num}",
            "template_type": "pass1",
            "version_label": cfg.get("label", f"V{ver_num}"),
            "description": cfg.get("description", ""),
            "template_text": tpl_text or f"[Placeholder — template v{ver_num} not available]",
            "variables": json.dumps(variables),
            "response_schema_json": None,
            "request_config_json": None,
            "is_active": tpl_text is not None,
            "is_default": ver_num == 3,
            "display_order": ver_num,
        })

    # Pass 2 templates
    for ver_num, cfg in PASS2_PROMPT_TEMPLATES.items():
        tpl_text = None
        if "template" in cfg and cfg["template"] != "inline":
            # Real template text provided in config
            tpl_text = cfg["template"]
        elif "file" in cfg:
            tpl_text = _read_file_safe(cfg["file"])
        # Always try inline fallback if no valid text yet
        if not tpl_text or tpl_text == "inline":
            tpl_text = _PASS2_INLINE_FALLBACKS.get(ver_num)

        variables = sorted(set(re.findall(r'\{\{(\w+)\}\}', tpl_text))) if tpl_text else []

        seeds.append({
            "template_name": f"pass2_v{ver_num}",
            "template_type": "pass2",
            "version_label": cfg.get("label", f"V{ver_num}"),
            "description": cfg.get("description", ""),
            "template_text": tpl_text or f"[Placeholder — template v{ver_num} not available]",
            "variables": json.dumps(variables),
            "response_schema_json": json.dumps(_pass2_schema_config(ver_num)) if _pass2_schema_config(ver_num) else None,
            "request_config_json": json.dumps(_pass2_request_config(ver_num)),
            "is_active": tpl_text is not None,
            "is_default": ver_num == 2,
            "display_order": ver_num,
        })

    # Directive template
    directive_text = _read_file_safe(DEFAULT_DIRECTIVE_PROMPT) or _DIRECTIVE_INLINE
    directive_vars = sorted(set(re.findall(r'\{\{(\w+)\}\}', directive_text)))
    seeds.append({
        "template_name": "directive_v1",
        "template_type": "directive",
        "version_label": "V1 — Default Directive",
        "description": "Default directive generation prompt. Extracts competitive positioning bullets from slides content.",
        "template_text": directive_text,
        "variables": json.dumps(directive_vars),
        "response_schema_json": None,
        "request_config_json": None,
        "is_active": True,
        "is_default": True,
        "display_order": 1,
    })

    # ── Insert or repair ──
    with ENGINE.begin() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM prompt_templates")).scalar()

        if count and int(count) > 0:
            # DB already has rows — repair any placeholders or incomplete templates
            for seed in seeds:
                if seed["template_text"].startswith("[Placeholder"):
                    continue
                try:
                    # Repair rows with placeholder text OR 'inline' stub OR short invalid content
                    conn.execute(
                        text(
                            "UPDATE prompt_templates "
                            "SET template_text = :template_text, "
                            "    variables = :variables, "
                            "    response_schema_json = CAST(:response_schema_json AS jsonb), "
                            "    request_config_json = CAST(:request_config_json AS jsonb), "
                            "    is_active = :is_active, "
                            "    updated_at = CURRENT_TIMESTAMP "
                            "WHERE template_name = :template_name "
                            "  AND ("
                            "template_text LIKE '[Placeholder%' "
                            "OR template_text = 'inline' "
                            "OR LENGTH(template_text) < 100 "
                            "OR template_name IN ('pass2_v5', 'pass2_v6', 'pass2_v7', 'pass2_v8', 'pass2_v9', 'pass2_v10')"
                            ")"
                        ),
                        {
                            "template_text": seed["template_text"],
                            "variables": seed["variables"],
                            "response_schema_json": seed["response_schema_json"],
                            "request_config_json": seed["request_config_json"],
                            "is_active": seed["is_active"],
                            "template_name": seed["template_name"],
                        },
                    )
                except Exception as e:
                    logger.warning("Repair prompt_templates row %s (non-fatal): %s", seed["template_name"], e)

                try:
                    conn.execute(
                        text(
                            "INSERT INTO prompt_templates "
                            "(template_name, template_type, version_label, description, template_text, variables, response_schema_json, request_config_json, is_active, is_default, display_order) "
                            "VALUES (:template_name, :template_type, :version_label, :description, :template_text, :variables, CAST(:response_schema_json AS jsonb), CAST(:request_config_json AS jsonb), :is_active, :is_default, :display_order) "
                            "ON CONFLICT (template_name) DO NOTHING"
                        ),
                        seed,
                    )
                except Exception as e:
                    logger.warning("Upsert prompt_templates row %s (non-fatal): %s", seed["template_name"], e)
            logger.info("Repaired any placeholder prompt_templates rows")
            return

        # Fresh insert
        for seed in seeds:
            try:
                conn.execute(
                    text(
                        "INSERT INTO prompt_templates "
                        "(template_name, template_type, version_label, description, template_text, variables, response_schema_json, request_config_json, is_active, is_default, display_order) "
                        "VALUES (:template_name, :template_type, :version_label, :description, :template_text, :variables, CAST(:response_schema_json AS jsonb), CAST(:request_config_json AS jsonb), :is_active, :is_default, :display_order)"
                    ),
                    seed,
                )
            except Exception as e:
                logger.warning("Seed prompt_templates (non-fatal): %s", e)

    logger.info("Seeded prompt_templates with %d rows", len(seeds))


def cleanup_stale_data():
    """Wipe all workflow sessions and battlecard generations that don't have
    proper 1:1 linkage.  Run once at startup so the user starts fresh with the
    unified dashboard flow."""
    with ENGINE.begin() as conn:
        conn.execute(text("DELETE FROM human_reviews"))
        conn.execute(text("DELETE FROM fact_checks"))
        conn.execute(text("DELETE FROM evidence"))
        conn.execute(text("DELETE FROM claim_detail_items"))
        conn.execute(text("DELETE FROM claims"))
        conn.execute(text("DELETE FROM agent_turns"))
        conn.execute(text("DELETE FROM workflow_artifacts"))
        conn.execute(text("DELETE FROM workflow_steps"))
        conn.execute(text("DELETE FROM workflow_sessions"))
        conn.execute(text("DELETE FROM battlecard_generations"))
        conn.execute(text("DELETE FROM regeneration_requests"))
    logger.info("Cleaned up stale data for unified dashboard")


def seed_data():
    with ENGINE.begin() as conn:
        exists = conn.execute(text("SELECT COUNT(*) FROM battlecard_generations")).scalar()
        if exists and int(exists) > 0:
            return

        databricks_id = conn.execute(
            text("INSERT INTO companies (company_name, company_type) VALUES (:name, 'databricks') RETURNING company_id"),
            {"name": "Databricks"},
        ).scalar()
        fabric_id = conn.execute(
            text("INSERT INTO companies (company_name, company_type) VALUES (:name, 'competitor') RETURNING company_id"),
            {"name": "Microsoft Fabric"},
        ).scalar()

        dw_cat = conn.execute(
            text(
                "SELECT catalog_id FROM product_category_catalog "
                "WHERE category_name = 'Data Warehousing & Lakehouse (including Open Table Formats)' "
                "LIMIT 1"
            )
        ).scalar()
        di_cat = conn.execute(
            text(
                "SELECT catalog_id FROM product_category_catalog "
                "WHERE category_name = 'Data Ingestion & Integration' "
                "LIMIT 1"
            )
        ).scalar()

        dw_diff = conn.execute(
            text(
                "INSERT INTO key_differentiators (category_id, key_diff_name, key_diff_description, display_order) "
                "VALUES (:cat, :name, :desc, 1) RETURNING key_diff_id"
            ),
            {
                "cat": dw_cat,
                "name": "Vectorized Query Engine Performance",
                "desc": "2–3x faster query execution through optimized columnar processing",
            },
        ).scalar()
        di_diff = conn.execute(
            text(
                "INSERT INTO key_differentiators (category_id, key_diff_name, key_diff_description, display_order) "
                "VALUES (:cat, :name, :desc, 1) RETURNING key_diff_id"
            ),
            {
                "cat": di_cat,
                "name": "Incremental Ingestion & CDC",
                "desc": "Reduce cost by only processing new and changed data",
            },
        ).scalar()

        photon_doc = conn.execute(
            text(
                "INSERT INTO sources (source_name, source_url, source_type, publisher) "
                "VALUES (:name, :url, 'vendor', 'Databricks') RETURNING source_id"
            ),
            {
                "name": "Photon Engine Documentation",
                "url": "https://docs.databricks.com/en/optimizations/photon.html",
            },
        ).scalar()
        gigao = conn.execute(
            text(
                "INSERT INTO sources (source_name, source_url, source_type, publisher) "
                "VALUES (:name, :url, 'third-party', 'GigaOm') RETURNING source_id"
            ),
            {
                "name": "GigaOm Data Lake Report 2025",
                "url": "https://gigaom.com/report/data-lake-analytics/",
            },
        ).scalar()
        fabric_arch = conn.execute(
            text(
                "INSERT INTO sources (source_name, source_url, source_type, publisher) "
                "VALUES (:name, :url, 'vendor', 'Microsoft') RETURNING source_id"
            ),
            {
                "name": "Microsoft Fabric Architecture Overview",
                "url": "https://learn.microsoft.com/fabric/",
            },
        ).scalar()

        gen_id = conn.execute(
            text(
                "INSERT INTO battlecard_generations (trigger_type, generated_by, generation_model, status) "
                "VALUES ('initial', 'agent:battlecard-generator', 'gpt-4', 'approved') RETURNING generation_id"
            )
        ).scalar()

        db_dw_claim = conn.execute(
            text(
                "INSERT INTO claims (generation_id, key_diff_id, company_id, rating, rating_symbol, headline, description, change_type) "
                "VALUES (:gen, :kd, :co, 'positive', '(+)', :head, :desc, 'new') RETURNING claim_id"
            ),
            {
                "gen": gen_id,
                "kd": dw_diff,
                "co": databricks_id,
                "head": "Photon Vectorized Query Engine",
                "desc": "Purpose-built engine optimized for analytical workloads at scale.",
            },
        ).scalar()
        fabric_dw_claim = conn.execute(
            text(
                "INSERT INTO claims (generation_id, key_diff_id, company_id, rating, rating_symbol, headline, description, change_type) "
                "VALUES (:gen, :kd, :co, 'neutral', '(~)', :head, :desc, 'new') RETURNING claim_id"
            ),
            {
                "gen": gen_id,
                "kd": dw_diff,
                "co": fabric_id,
                "head": "Adequate for Standard Workloads",
                "desc": "Performs well for Power BI and simple queries, gaps emerge at scale.",
            },
        ).scalar()

        db_di_claim = conn.execute(
            text(
                "INSERT INTO claims (generation_id, key_diff_id, company_id, rating, rating_symbol, headline, description, change_type) "
                "VALUES (:gen, :kd, :co, 'positive', '(+)', :head, :desc, 'new') RETURNING claim_id"
            ),
            {
                "gen": gen_id,
                "kd": di_diff,
                "co": databricks_id,
                "head": "Incremental Ingestion at Scale",
                "desc": "Auto Loader and LakeFlow support incremental and CDC ingestion.",
            },
        ).scalar()
        fabric_di_claim = conn.execute(
            text(
                "INSERT INTO claims (generation_id, key_diff_id, company_id, rating, rating_symbol, headline, description, change_type) "
                "VALUES (:gen, :kd, :co, 'negative', '(-)', :head, :desc, 'new') RETURNING claim_id"
            ),
            {
                "gen": gen_id,
                "kd": di_diff,
                "co": fabric_id,
                "head": "Incremental Ingestion Gaps",
                "desc": "Mirroring supports limited sources and CDC scenarios; other tools lack full incremental ingestion.",
            },
        ).scalar()

        def insert_evidence(claim_id, field_name, text_value, source_id, source_text):
            return conn.execute(
                text(
                    "INSERT INTO evidence (claim_id, traces_to_field, traces_to_start_index, traces_to_end_index, "
                    "traces_to_text, generation_source_id, generation_source_text) "
                    "VALUES (:claim, :field, 0, :end, :trace, :src, :src_text) RETURNING evidence_id"
                ),
                {
                    "claim": claim_id,
                    "field": field_name,
                    "end": len(text_value),
                    "trace": text_value,
                    "src": source_id,
                    "src_text": source_text,
                },
            ).scalar()

        # Evidence rows
        db_dw_head_ev = insert_evidence(
            db_dw_claim,
            "headline",
            "Photon Vectorized Query Engine",
            photon_doc,
            "Photon is written natively in C++ and uses vectorized execution.",
        )
        db_dw_desc_ev = insert_evidence(
            db_dw_claim,
            "description",
            "Purpose-built engine optimized for analytical workloads at scale.",
            photon_doc,
            "Photon delivers faster query performance than traditional engines.",
        )
        fab_dw_head_ev = insert_evidence(
            fabric_dw_claim,
            "headline",
            "Adequate for Standard Workloads",
            fabric_arch,
            "Fabric SQL engine uses SQL Server architecture with columnstore optimizations.",
        )
        fab_dw_desc_ev = insert_evidence(
            fabric_dw_claim,
            "description",
            "Performs well for Power BI and simple queries, gaps emerge at scale.",
            fabric_arch,
            "Fabric targets BI workloads and uses capacity-based execution.",
        )

        db_di_head_ev = insert_evidence(
            db_di_claim,
            "headline",
            "Incremental Ingestion at Scale",
            photon_doc,
            "Auto Loader supports incremental ingestion at scale.",
        )
        db_di_desc_ev = insert_evidence(
            db_di_claim,
            "description",
            "Auto Loader and LakeFlow support incremental and CDC ingestion.",
            photon_doc,
            "LakeFlow supports incremental and CDC ingestion patterns.",
        )
        fab_di_head_ev = insert_evidence(
            fabric_di_claim,
            "headline",
            "Incremental Ingestion Gaps",
            fabric_arch,
            "Mirroring supports limited sources and CDC scenarios.",
        )
        fab_di_desc_ev = insert_evidence(
            fabric_di_claim,
            "description",
            "Mirroring supports limited sources and CDC scenarios; other tools lack full incremental ingestion.",
            fabric_arch,
            "Data Factory and Dataflow Gen2 do not support incremental ingestion for all sources.",
        )

        def insert_fact_check(evidence_id, status, source_id, source_text, reasoning, confidence):
            conn.execute(
                text(
                    "INSERT INTO fact_checks (evidence_id, status, fact_check_source_id, fact_check_source_text, reasoning, "
                    "checked_by, check_method, confidence_score) "
                    "VALUES (:ev, :status, :src, :text, :reason, 'agent:fact-checker', 'automated', :conf)"
                ),
                {
                    "ev": evidence_id,
                    "status": status,
                    "src": source_id,
                    "text": source_text,
                    "reason": reasoning,
                    "conf": confidence,
                },
            )

        insert_fact_check(
            db_dw_head_ev,
            "verified",
            gigao,
            "TPC-DS benchmark results show 2.4x improvement at 30TB scale.",
            "Third-party benchmark confirms performance claims within stated range.",
            85,
        )
        insert_fact_check(
            db_dw_desc_ev,
            "verified",
            gigao,
            "Photon improves analytics performance with vectorized execution.",
            "Independent report confirms performance gains.",
            82,
        )
        insert_fact_check(
            db_di_head_ev,
            "verified",
            gigao,
            "Auto Loader provides incremental ingestion with CDC options.",
            "Report validates incremental ingestion support.",
            80,
        )
        insert_fact_check(
            db_di_desc_ev,
            "verified",
            gigao,
            "LakeFlow supports incremental ingestion patterns.",
            "Third-party report verifies ingestion support.",
            78,
        )

        insert_fact_check(
            fab_dw_head_ev,
            "unverified",
            None,
            None,
            "No independent verification source found.",
            30,
        )
        insert_fact_check(
            fab_dw_desc_ev,
            "unverified",
            None,
            None,
            "Claim based on vendor documentation only.",
            30,
        )
        insert_fact_check(
            fab_di_head_ev,
            "unverified",
            None,
            None,
            "No independent verification source found.",
            30,
        )
        insert_fact_check(
            fab_di_desc_ev,
            "unverified",
            None,
            None,
            "Claim based on vendor documentation only.",
            30,
        )

        conn.execute(
            text(
                "UPDATE battlecard_generations SET total_claims = 4, verified_claims = 2, unverified_claims = 2 "
                "WHERE generation_id = :gid"
            ),
            {"gid": gen_id},
        )


# =============================================================================
# Token estimation helpers (tiktoken)
# =============================================================================

# Use cl100k_base encoding as a reasonable proxy for Claude / GPT-4 class models.
# Actual Claude tokenisation differs slightly but cl100k_base gives a good estimate.
_TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return the estimated token count for *text* using cl100k_base."""
    if not text:
        return 0
    return len(_TIKTOKEN_ENCODING.encode(text))


def estimate_prompt_tokens(session_id: str, step_number: int) -> dict:
    """Build the same rendered prompt that WorkflowRunner would use and return a
    token-count breakdown.

    Returns a dict like::

        {
            "step": 4,
            "input_breakdown": {
                "template": 820,
                "directive": 4500,
                "old_battlecard": 12000,
                "categories": 45,
                "total_rendered_prompt": 18200,
            },
            "output_estimate": {
                "max_tokens": 16384,
                "description": "Max output tokens configured for the model call",
            },
            "total_input_tokens": 18200,
        }
    """
    from workflow_runner import (
        load_pass1_template,
        load_pass2_template,
        render_template as render_prompt,
        format_context_xml,
    )

    with ENGINE.begin() as conn:
        session = conn.execute(
            text(
                "SELECT competitor_name, product_area, model_name, diffs_per_category, "
                "COALESCE(pass1_prompt_template_version, 2) AS pass1_prompt_template_version, "
                "COALESCE(pass2_prompt_template_version, 2) AS pass2_prompt_template_version "
                "FROM workflow_sessions WHERE session_id::text = :sid"
            ),
            {"sid": session_id},
        ).mappings().first()
        if not session:
            return {"error": "Session not found"}

    def _get_artifact(atype):
        with ENGINE.begin() as conn:
            return conn.execute(
                text(
                    "SELECT artifact_content FROM workflow_artifacts "
                    "WHERE session_id::text = :sid AND artifact_type = :atype "
                    "ORDER BY created_at DESC LIMIT 1"
                ),
                {"sid": session_id, "atype": atype},
            ).scalar()

    competitor = session["competitor_name"]
    product_area = session["product_area"]
    diffs_per_category = session["diffs_per_category"]
    max_output_tokens = 16384  # matches call_model default

    directive = _get_artifact("directive_generated") or ""
    old_battlecard = _get_artifact("old_battlecard_extracted") or ""
    context = format_context_xml(directive, old_battlecard, competitor)

    breakdown = {}

    if step_number == 4:
        # Pass 1 prompt
        p1_ver = int(session.get("pass1_prompt_template_version") or 2)
        try:
            template_text, _ = load_pass1_template(p1_ver, engine=ENGINE)
        except FileNotFoundError:
            return {"error": f"Pass 1 prompt template V{p1_ver} not found"}

        categories_content = _get_artifact("product_categories") or ""
        categories = [c.strip() for c in categories_content.split("\n") if c.strip()]
        categories_text = "\n".join(f"- {c}" for c in categories) if categories else ""
        total_diffs = len(categories) * diffs_per_category

        rendered = render_prompt(
            template_text,
            competitor=competitor,
            product_area=product_area,
            comparison=f"Databricks vs {competitor}",
            product_categories=categories_text,
            diffs_per_category=str(diffs_per_category),
            total_diffs=str(total_diffs),
            directives=directive,
            context=context,
        )

        breakdown = {
            "template": count_tokens(template_text),
            "directive": count_tokens(directive),
            "old_battlecard": count_tokens(old_battlecard),
            "categories": count_tokens(categories_text),
            "context_xml_wrapper": count_tokens(context) - count_tokens(directive) - count_tokens(old_battlecard),
            "total_rendered_prompt": count_tokens(rendered),
        }

    elif step_number == 5:
        # Pass 2 prompt — estimate for a single skeleton (they all share the same context)
        p2_ver = int(session.get("pass2_prompt_template_version") or 2)
        try:
            template_text = load_pass2_template(p2_ver, engine=ENGINE)
        except FileNotFoundError:
            return {"error": f"Pass 2 prompt template V{p2_ver} not found"}

        skeletons_json = _get_artifact("pass1_skeletons")
        num_skeletons = 0
        sample_rendered = ""

        if skeletons_json:
            skeletons = json.loads(skeletons_json)
            num_skeletons = len(skeletons)
            if skeletons:
                sk = skeletons[0]
                sample_rendered = render_prompt(
                    template_text,
                    competitor=competitor,
                    category=sk.get("category", ""),
                    key_differentiator=sk.get("key_differentiator", ""),
                    description=sk.get("description", ""),
                    databricks_rating=sk.get("databricks_rating", ""),
                    competitor_rating=sk.get("competitor_rating", ""),
                    selection_reasoning=sk.get("selection_reasoning", ""),
                    directives="" if p2_ver in (5, 6, 7, 8, 9, 10) else directive,
                    context=context,
                )

        per_call_tokens = count_tokens(sample_rendered) if sample_rendered else count_tokens(template_text)

        breakdown = {
            "template": count_tokens(template_text),
            "directive": count_tokens(directive),
            "old_battlecard": count_tokens(old_battlecard),
            "context_xml_wrapper": count_tokens(context) - count_tokens(directive) - count_tokens(old_battlecard),
            "per_call_rendered_prompt": per_call_tokens,
            "num_skeleton_calls": num_skeletons,
            "total_all_calls": per_call_tokens * max(num_skeletons, 1),
        }
        max_output_tokens = max_output_tokens * max(num_skeletons, 1)

    else:
        return {
            "step": step_number,
            "input_breakdown": {},
            "output_estimate": {"max_tokens": 0, "description": "No LLM call for this step"},
            "total_input_tokens": 0,
        }

    total_input = breakdown.get("total_rendered_prompt") or breakdown.get("total_all_calls") or 0

    return {
        "step": step_number,
        "input_breakdown": breakdown,
        "output_estimate": {
            "max_tokens": max_output_tokens,
            "description": "Max output tokens configured for the model call(s)",
        },
        "total_input_tokens": total_input,
    }


# =============================================================================
# Bi-directional workflow <-> battlecard linking helpers
# =============================================================================


def _lookup_workflow_session_for_battlecard(battlecard_id):
    """Given a battlecard UUID, return the linked workflow session_id (or None)."""
    with ENGINE.begin() as conn:
        row = conn.execute(
            text(
                "SELECT ws.session_id::text FROM workflow_sessions ws "
                "JOIN battlecard_generations bg ON ws.generation_id = bg.generation_id "
                "WHERE bg.generation_uuid::text = :bid LIMIT 1"
            ),
            {"bid": battlecard_id},
        ).scalar()
    return row


def _lookup_battlecard_for_workflow(session_id):
    """Given a workflow session_id, return the linked battlecard UUID (or None)."""
    with ENGINE.begin() as conn:
        row = conn.execute(
            text(
                "SELECT bg.generation_uuid::text FROM battlecard_generations bg "
                "JOIN workflow_sessions ws ON ws.generation_id = bg.generation_id "
                "WHERE ws.session_id::text = :sid LIMIT 1"
            ),
            {"sid": session_id},
        ).scalar()
    return row


def _get_review_feedback_summary(generation_id):
    """Count approve/revision/reject reviews for a generation."""
    with ENGINE.begin() as conn:
        row = conn.execute(
            text(
                "SELECT COUNT(*) AS total, "
                "SUM(CASE WHEN review_type='approve' THEN 1 ELSE 0 END) AS approved, "
                "SUM(CASE WHEN review_type='request_edit' THEN 1 ELSE 0 END) AS needs_revision, "
                "SUM(CASE WHEN review_type='reject' THEN 1 ELSE 0 END) AS rejected "
                "FROM human_reviews WHERE generation_id = :gid"
            ),
            {"gid": generation_id},
        ).mappings().first()
    if not row:
        return {"total": 0, "approved": 0, "needs_revision": 0, "rejected": 0}
    return {
        "total": int(row["total"] or 0),
        "approved": int(row["approved"] or 0),
        "needs_revision": int(row["needs_revision"] or 0),
        "rejected": int(row["rejected"] or 0),
    }


def _collect_review_feedback_text(generation_id):
    """Gather all review comments into a structured text block for regeneration prompts."""
    with ENGINE.begin() as conn:
        rows = conn.execute(
            text(
                "SELECT review_type, feedback_text, reviewed_by, reviewed_at "
                "FROM human_reviews WHERE generation_id = :gid ORDER BY reviewed_at"
            ),
            {"gid": generation_id},
        ).mappings().all()

    if not rows:
        return ""

    parts = ["## Battlecard Review Feedback\n"]
    for r in rows:
        review_type = r["review_type"]
        feedback_raw = r["feedback_text"] or ""
        reviewer = r["reviewed_by"] or "anonymous"

        # Parse JSON feedback payload
        comment = ""
        scope = ""
        try:
            payload = json.loads(feedback_raw)
            comment = payload.get("comment", "")
            scope = payload.get("scope", "")
        except (json.JSONDecodeError, TypeError):
            comment = feedback_raw

        status_label = {"approve": "APPROVED", "request_edit": "NEEDS REVISION", "reject": "REJECTED"}.get(review_type, review_type.upper())
        line = f"- [{status_label}]"
        if scope:
            line += f" (scope: {scope})"
        if comment:
            line += f": {comment}"
        line += f"  — {reviewer}"
        parts.append(line)

    return "\n".join(parts)


# =============================================================================
# Data loaders
# =============================================================================


def load_battlecard_generations():
    with ENGINE.begin() as conn:
        rows = conn.execute(
            text(
                "SELECT bg.generation_id, bg.generation_uuid::text AS battlecard_id, "
                "bg.generated_at, bg.status, bg.generation_model, "
                "ws.session_id::text AS session_id "
                "FROM battlecard_generations bg "
                "LEFT JOIN workflow_sessions ws ON ws.generation_id = bg.generation_id "
                "ORDER BY bg.generated_at DESC"
            )
        ).mappings().all()

    generations = []
    for r in rows:
        gen_id = r["generation_id"]
        competitor = ""
        with ENGINE.begin() as conn:
            comp = conn.execute(
                text(
                    "SELECT DISTINCT c.company_name FROM claims cl "
                    "JOIN companies c ON cl.company_id = c.company_id "
                    "WHERE cl.generation_id = :gid AND c.company_type = 'competitor' LIMIT 1"
                ),
                {"gid": gen_id},
            ).scalar()
            competitor = comp or "Competitor"

        generations.append(
            {
                "battlecard_id": r["battlecard_id"],
                "battlecard_version_id": gen_id,
                "competitor": competitor,
                "product_area": "Data Platform",
                "model_name": r.get("generation_model"),
                "prompt_name": "",
                "prompt_version": 0,
                "experiment_id": "",
                "agent_run_id": "",
                "status": r.get("status"),
                "generated_at": r.get("generated_at"),
                "context_doc_count": 0,
                "session_id": r.get("session_id"),
            }
        )
    return generations


def _fetch_generation_by_uuid(battlecard_id):
    with ENGINE.begin() as conn:
        row = conn.execute(
            text(
                "SELECT generation_id, generation_uuid::text AS battlecard_id, generated_at, status, generation_model "
                "FROM battlecard_generations WHERE generation_uuid::text = :bid"
            ),
            {"bid": battlecard_id},
        ).mappings().first()
    return row


def _build_detail_items(claim, detail_items_by_claim, verbose_items_by_detail=None):
    """Build structured detail items list for a claim.
    Returns list of dicts with detail_item_id, item_order, text (L200), and verbose_items (L300).
    Falls back to wrapping description in a single-item list for legacy claims.
    """
    if not claim:
        return []
    verbose_items_by_detail = verbose_items_by_detail or {}
    items = detail_items_by_claim.get(claim["claim_id"], [])
    if items:
        out = []
        for di in items:
            v_items = verbose_items_by_detail.get(di["detail_item_id"], [])
            text_verbose = di.get("item_text_verbose")
            if not v_items and text_verbose:
                v_items = [
                    {"verbose_item_id": None, "item_order": 0, "text": text_verbose}
                ]
            out.append(
                {
                    "detail_item_id": di["detail_item_id"],
                    "item_order": di["item_order"],
                    "text": di["item_text"],
                    "text_verbose": "\n".join([v.get("text", "") for v in v_items if v.get("text")]) if v_items else None,
                    "verbose_items": v_items,
                }
            )
        return out
    # Legacy fallback: no detail items, wrap description
    desc = claim.get("description") or ""
    if desc:
        return [{
            "detail_item_id": None,
            "item_order": 0,
            "text": desc,
            "text_verbose": None,
            "verbose_items": [],
        }]
    return []


def load_battlecard_slides(battlecard_id):
    gen = _fetch_generation_by_uuid(battlecard_id)
    if not gen:
        return [], {}

    gen_id = gen["generation_id"]
    selected_core_categories = set()

    with ENGINE.begin() as conn:
        selected_rows = conn.execute(
            text(
                "SELECT DISTINCT pcc.category_name "
                "FROM workflow_sessions ws "
                "JOIN session_category_selections scs ON scs.session_id = ws.session_id "
                "JOIN product_category_catalog pcc ON pcc.catalog_id = scs.catalog_id "
                "WHERE ws.generation_id = :gid "
                "AND scs.inclusion_type = 'core_product_category'"
            ),
            {"gid": gen_id},
        ).scalars().all()
        selected_core_categories = set(selected_rows or [])

        diffs = conn.execute(
            text(
                "SELECT kd.key_diff_id, kd.key_diff_name, kd.key_diff_description, kd.display_order, "
                "pcc.category_name "
                "FROM key_differentiators kd "
                "JOIN product_category_catalog pcc ON kd.category_id = pcc.catalog_id "
                "WHERE kd.is_active = TRUE "
                "AND kd.key_diff_id IN (SELECT DISTINCT cl.key_diff_id FROM claims cl WHERE cl.generation_id = :gid) "
                "ORDER BY pcc.display_order, kd.display_order"
            ),
            {"gid": gen_id},
        ).mappings().all()
        if selected_core_categories:
            diffs = [d for d in diffs if d["category_name"] in selected_core_categories]

        claims = conn.execute(
            text(
                "SELECT cl.*, c.company_type, c.company_name "
                "FROM claims cl "
                "JOIN companies c ON cl.company_id = c.company_id "
                "WHERE cl.generation_id = :gid"
            ),
            {"gid": gen_id},
        ).mappings().all()

        evidences = conn.execute(
            text(
                "SELECT e.*, e.detail_item_id, s.source_name, s.source_url, s.source_type "
                "FROM evidence e "
                "JOIN claims cl ON e.claim_id = cl.claim_id "
                "LEFT JOIN sources s ON e.generation_source_id = s.source_id "
                "WHERE cl.generation_id = :gid"
            ),
            {"gid": gen_id},
        ).mappings().all()

        fact_checks = conn.execute(
            text(
                "SELECT fc.*, e.claim_id, e.detail_item_id, e.verbose_item_id, e.traces_to_field, e.claim_subfield, e.citation_id, e.traces_to_start_index, e.traces_to_end_index, "
                "s.source_name, s.source_url, s.source_type "
                "FROM fact_checks fc "
                "JOIN evidence e ON fc.evidence_id = e.evidence_id "
                "LEFT JOIN sources s ON fc.fact_check_source_id = s.source_id "
                "JOIN claims cl ON e.claim_id = cl.claim_id "
                "WHERE cl.generation_id = :gid "
                "ORDER BY fc.checked_at"
            ),
            {"gid": gen_id},
        ).mappings().all()

        detail_items = conn.execute(
            text(
                "SELECT di.* FROM claim_detail_items di "
                "JOIN claims cl ON di.claim_id = cl.claim_id "
                "WHERE cl.generation_id = :gid "
                "ORDER BY di.claim_id, di.item_order"
            ),
            {"gid": gen_id},
        ).mappings().all()

        verbose_items = conn.execute(
            text(
                "SELECT vi.* FROM claim_detail_verbose_items vi "
                "JOIN claim_detail_items di ON vi.detail_item_id = di.detail_item_id "
                "JOIN claims cl ON di.claim_id = cl.claim_id "
                "WHERE cl.generation_id = :gid "
                "ORDER BY vi.detail_item_id, vi.item_order"
            ),
            {"gid": gen_id},
        ).mappings().all()

    # Build detail_items lookup by claim_id
    detail_items_by_claim = {}
    for di in detail_items:
        detail_items_by_claim.setdefault(di["claim_id"], []).append(di)
    verbose_items_by_detail = {}
    for vi in verbose_items:
        verbose_items_by_detail.setdefault(vi["detail_item_id"], []).append(
            {
                "verbose_item_id": vi["verbose_item_id"],
                "item_order": vi["item_order"],
                "text": vi["item_text"],
            }
        )

    claims_by_keydiff = {}
    for c in claims:
        claims_by_keydiff.setdefault(c["key_diff_id"], {})[c["company_type"]] = c

    evidence_by_claim = {}
    for e in evidences:
        evidence_by_claim.setdefault(e["claim_id"], []).append(e)

    fact_checks_by_claim = {}
    for fc in fact_checks:
        fact_checks_by_claim.setdefault(fc["claim_id"], []).append(fc)

    slides = []
    for d in diffs:
        key_diff_id = d["key_diff_id"]
        db_claim = claims_by_keydiff.get(key_diff_id, {}).get("databricks")
        fab_claim = claims_by_keydiff.get(key_diff_id, {}).get("competitor")

        diff_id = f"kd-{key_diff_id}"
        sources = []
        source_index_map = {}

        def add_source(source_id, name, url, source_type):
            if not source_id:
                return None
            if source_id not in source_index_map:
                source_index_map[source_id] = len(source_index_map) + 1
                sources.append(
                    {
                        "index": source_index_map[source_id],
                        "title": name or "Source",
                        "url": url or "",
                        "type": source_type or "",
                    }
                )
            return source_index_map[source_id]

        def build_citations(claim, prefix):
            citations = {}
            if not claim:
                return citations
            for e in evidence_by_claim.get(claim["claim_id"], []):
                subfield = (e.get("claim_subfield") or "").strip().lower()
                if subfield == "headline":
                    field = "headline"
                elif subfield == "reasoning":
                    field = "reasoning"
                elif subfield == "l300":
                    field = "detail_verbose"
                else:
                    field = "details"
                field_name = f"{prefix}_{field}"
                idx = add_source(e["generation_source_id"], e.get("source_name"), e.get("source_url"), e.get("source_type"))
                entry = {
                    "evidence_id": e.get("evidence_id"),
                    "generation_source_id": e.get("generation_source_id"),
                    "citation_id": e.get("citation_id") or f"ev-{e['evidence_id']}",
                    "detail_item_id": e.get("detail_item_id"),
                    "verbose_item_id": e.get("verbose_item_id"),
                    "claim_subfield": subfield or ("headline" if e["traces_to_field"] == "headline" else "l200"),
                    "citation_order": e.get("citation_order"),
                    "start_index": int(e["traces_to_start_index"]),
                    "end_index": int(e["traces_to_end_index"]),
                    "source_index": idx,
                    "source_quote": e.get("generation_source_text") or "",
                    "source_title": e.get("source_name") or "",
                    "source_url": e.get("source_url") or "",
                    "source_type": e.get("source_type") or "",
                }
                citations.setdefault(field_name, []).append(entry)
            return citations

        citations = {}
        citations.update(build_citations(db_claim, "databricks"))
        citations.update(build_citations(fab_claim, "fabric"))

        def build_fact_checks(claim, prefix):
            output = []
            if not claim:
                return output
            for fc in fact_checks_by_claim.get(claim["claim_id"], []):
                subfield = (fc.get("claim_subfield") or "").strip().lower()
                if subfield == "headline":
                    field = "headline"
                elif subfield == "reasoning":
                    field = "reasoning"
                elif subfield == "l300":
                    field = "detail_verbose"
                else:
                    field = "details"
                claim_field = f"{prefix}_{field}"
                confidence = None
                if fc.get("confidence_score") is not None:
                    confidence = float(fc["confidence_score"]) / 100.0

                citations_list = []
                if fc.get("source_name") or fc.get("source_url") or fc.get("fact_check_source_text"):
                    citations_list.append(
                        {
                            "title": fc.get("source_name") or "Source",
                            "url": fc.get("source_url") or "",
                            "quote": fc.get("fact_check_source_text") or "",
                            "source_type": fc.get("source_type") or "",
                        }
                    )

                output.append(
                    {
                        "fact_check_id": str(fc["fact_check_id"]),
                        "claim_field": claim_field,
                        "detail_item_id": fc.get("detail_item_id"),
                        "verbose_item_id": fc.get("verbose_item_id"),
                        "claim_start_index": int(fc["traces_to_start_index"]),
                        "claim_end_index": int(fc["traces_to_end_index"]),
                        "verdict": fc.get("status"),
                        "confidence": confidence,
                        "rationale": fc.get("reasoning") or "",
                        "citations": citations_list,
                        "checked_at": fc.get("checked_at"),
                    }
                )
            return output

        def map_rating(value):
            if value == "positive":
                return "advantage"
            if value == "neutral":
                return "partial"
            if value == "negative":
                return "disadvantage"
            return value or ""

        slide_fact_checks = []
        slide_fact_checks.extend(build_fact_checks(db_claim, "databricks"))
        slide_fact_checks.extend(build_fact_checks(fab_claim, "fabric"))

        db_detail_items = _build_detail_items(db_claim, detail_items_by_claim, verbose_items_by_detail)
        fab_detail_items = _build_detail_items(fab_claim, detail_items_by_claim, verbose_items_by_detail)

        # Flat text fallback for backward compatibility
        db_details_flat = " ".join(it["text"] for it in db_detail_items) if db_detail_items else ""
        fab_details_flat = " ".join(it["text"] for it in fab_detail_items) if fab_detail_items else ""

        diff = {
            "id": diff_id,
            "slide_type": "L200",
            "category": d.get("category_name") or "",
            "key_differentiator": d.get("key_diff_name") or "",
            "description": d.get("key_diff_description") or "",
            "rank": d.get("display_order") or 0,
            "databricks_headline": (db_claim.get("headline") if db_claim else ""),
            "databricks_details": db_details_flat,
            "databricks_detail_items": db_detail_items,
            "databricks_rating": (map_rating(db_claim.get("rating")) if db_claim else ""),
            "databricks_reasoning": "",
            "fabric_headline": (fab_claim.get("headline") if fab_claim else ""),
            "fabric_details": fab_details_flat,
            "fabric_detail_items": fab_detail_items,
            "fabric_rating": (map_rating(fab_claim.get("rating")) if fab_claim else ""),
            "fabric_reasoning": "",
            "selection_reasoning": "",
            "rank_reasoning": "",
            "directive_alignment": "",
            "citations": citations,
            "sources": sources,
            "competitor": "Microsoft Fabric",
            "key_diff_theme": "",
            "detail_status": "complete",
            "update_count": 0,
            "fact_checks": slide_fact_checks,
        }

        slides.append(diff)

    gen_info = {
        "competitor": "Microsoft Fabric",
        "product_area": "Data Platform",
        "battlecard_version_id": gen_id,
        "agent_run_id": "",
        "experiment_id": "",
        "prompt_name": "",
        "prompt_version": 0,
    }

    return slides, gen_info


def load_battlecard_reviews(battlecard_id):
    gen = _fetch_generation_by_uuid(battlecard_id)
    if not gen:
        return {}

    gen_id = gen["generation_id"]
    with ENGINE.begin() as conn:
        rows = conn.execute(
            text(
                "SELECT * FROM human_reviews WHERE generation_id = :gid ORDER BY reviewed_at DESC"
            ),
            {"gid": gen_id},
        ).mappings().all()

    feedback = {}
    for r in rows:
        slide_id = "__battlecard__"
        scope = None
        comment = ""
        reorder = None
        key_diff_id = None

        if r.get("feedback_text"):
            try:
                payload = json.loads(r.get("feedback_text"))
                scope = payload.get("scope")
                comment = payload.get("comment") or ""
                reorder = payload.get("reorder")
                key_diff_id = payload.get("key_diff_id")
                # Use stored diff_id as slide key when available
                stored_diff_id = payload.get("diff_id")
                if stored_diff_id:
                    slide_id = stored_diff_id
            except json.JSONDecodeError:
                comment = r.get("feedback_text")

        if key_diff_id:
            slide_id = f"kd-{key_diff_id}"

        status_map = {
            "approve": "approved",
            "request_edit": "needs_revision",
            "reject": "rejected",
        }
        status = status_map.get(r.get("review_type"), "")

        feedback.setdefault(slide_id, {})

        fb_entry = {
            "status": status,
            "comment": comment,
            "timestamp": r.get("reviewed_at") or "",
            "human_review_id": r.get("review_id"),
        }

        if scope == "reorder" and reorder:
            # Only keep the newest reorder (first seen since rows are DESC)
            if "reorder" not in feedback[slide_id]:
                feedback[slide_id]["reorder"] = {
                    "original_rank": reorder.get("original_rank"),
                    "new_rank": reorder.get("new_rank"),
                    "timestamp": fb_entry["timestamp"],
                }
        elif scope:
            # Only keep the newest review per scope (first seen since rows are DESC)
            if scope not in feedback[slide_id]:
                feedback[slide_id][scope] = fb_entry

    return feedback


def load_battlecard_fact_checks(battlecard_id):
    gen = _fetch_generation_by_uuid(battlecard_id)
    if not gen:
        return {}

    gen_id = gen["generation_id"]
    with ENGINE.begin() as conn:
        rows = conn.execute(
            text(
                "SELECT fc.*, e.claim_id, e.detail_item_id, e.verbose_item_id, e.claim_subfield, "
                "e.citation_id, e.traces_to_field, e.traces_to_start_index, e.traces_to_end_index, "
                "e.traces_to_text, e.generation_source_text, "
                "s.source_name AS fc_source_name, s.source_url AS fc_source_url, s.source_type AS fc_source_type, "
                "gs.source_name AS generation_source_name, gs.source_url AS generation_source_url, gs.source_type AS generation_source_type, "
                "cl.key_diff_id, co.company_type "
                "FROM fact_checks fc "
                "JOIN evidence e ON fc.evidence_id = e.evidence_id "
                "LEFT JOIN sources s ON fc.fact_check_source_id = s.source_id "
                "LEFT JOIN sources gs ON e.generation_source_id = gs.source_id "
                "JOIN claims cl ON e.claim_id = cl.claim_id "
                "JOIN companies co ON cl.company_id = co.company_id "
                "WHERE cl.generation_id = :gid "
                "ORDER BY fc.checked_at"
            ),
            {"gid": gen_id},
        ).mappings().all()

    by_slide = {}
    for r in rows:
        slide_id = f"kd-{r.get('key_diff_id')}"
        by_slide.setdefault(slide_id, [])
        confidence = None
        if r.get("confidence_score") is not None:
            confidence = float(r.get("confidence_score")) / 100.0

        # Build vendor-prefixed claim_field (e.g. databricks_headline, fabric_detail_verbose)
        vendor_prefix = "databricks" if r.get("company_type") == "databricks" else "fabric"
        subfield = (r.get("claim_subfield") or "").strip().lower()
        if subfield == "headline":
            field = "headline"
        elif subfield == "reasoning":
            field = "reasoning"
        elif subfield == "l300":
            field = "detail_verbose"
        else:
            field = "details"
        claim_field = f"{vendor_prefix}_{field}"

        entry = {
            "fact_check_id": str(r.get("fact_check_id")),
            "evidence_id": r.get("evidence_id"),
            "citation_id": r.get("citation_id"),
            "claim": r.get("traces_to_text") or "",
            "claim_field": claim_field,
            "detail_item_id": r.get("detail_item_id"),
            "verbose_item_id": r.get("verbose_item_id"),
            "claim_start_index": int(r.get("traces_to_start_index")),
            "claim_end_index": int(r.get("traces_to_end_index")),
            "verdict": r.get("status"),
            "confidence": confidence,
            "rationale": r.get("reasoning") or "",
            "claim_quote": r.get("generation_source_text") or "",
            "claim_source_title": r.get("generation_source_name") or "Source",
            "claim_source_url": r.get("generation_source_url") or "",
            "claim_source_type": r.get("generation_source_type") or "",
            "citations": [],
            "checked_at": r.get("checked_at"),
        }

        if r.get("fc_source_name") or r.get("fc_source_url") or r.get("fact_check_source_text"):
            entry["citations"].append(
                {
                    "title": r.get("fc_source_name") or "Source",
                    "url": r.get("fc_source_url") or "",
                    "quote": r.get("fact_check_source_text") or "",
                    "source_type": r.get("fc_source_type") or "",
                }
            )

        by_slide[slide_id].append(entry)

    return by_slide


def load_context_documents(battlecard_id):
    return []


def load_agent_session(battlecard_id):
    return None


def get_key_diff_theme_coverage(battlecard_id):
    return []


def _find_internal_source_context(session_id, source_text):
    """Locate source_text in workflow context artifacts and return a focused excerpt."""
    if not session_id or not source_text:
        return None

    with ENGINE.begin() as conn:
        arts = conn.execute(
            text(
                "SELECT artifact_id, step_number, artifact_type, artifact_name, artifact_content "
                "FROM workflow_artifacts "
                "WHERE session_id::text = :sid "
                "AND artifact_type IN ('directive_generated', 'directive_upload', 'old_battlecard_extracted', "
                "'old_battlecard_upload', 'product_categories', 'category_selections') "
                "ORDER BY step_number, created_at DESC"
            ),
            {"sid": session_id},
        ).mappings().all()
        try:
            dbg_rows = conn.execute(
                text(
                    "SELECT log_id, skeleton_index, key_differentiator, rendered_prompt "
                    "FROM pass2_debug_logs "
                    "WHERE session_id::text = :sid "
                    "ORDER BY created_at DESC LIMIT 40"
                ),
                {"sid": session_id},
            ).mappings().all()
        except Exception:
            dbg_rows = []

    needle = (source_text or "").strip()
    if not needle:
        return None

    for a in arts:
        content = a.get("artifact_content") or ""
        if not content:
            continue

        idx = content.find(needle)
        if idx < 0:
            short = needle[:120]
            if short:
                idx = content.lower().find(short.lower())
        if idx < 0:
            words = [w for w in needle.split() if len(w) >= 5][:6]
            if words:
                probe = " ".join(words)
                idx = content.lower().find(probe.lower())

        if idx >= 0:
            before = max(0, idx - 500)
            after = min(len(content), idx + len(needle) + 500)
            return {
                "artifact_id": a["artifact_id"],
                "artifact_type": a["artifact_type"],
                "artifact_name": a["artifact_name"],
                "step_number": a["step_number"],
                "match_start": idx,
                "match_end": idx + len(needle),
                "excerpt_start": before,
                "excerpt_end": after,
                "excerpt_text": content[before:after],
            }

    for d in dbg_rows:
        content = d.get("rendered_prompt") or ""
        if not content:
            continue
        idx = content.find(needle)
        if idx < 0:
            short = needle[:120]
            if short:
                idx = content.lower().find(short.lower())
        if idx < 0:
            continue
        before = max(0, idx - 500)
        after = min(len(content), idx + len(needle) + 500)
        return {
            "artifact_id": None,
            "artifact_type": "pass2_debug_prompt",
            "artifact_name": f"pass2_debug_log_{d.get('log_id')}",
            "step_number": 5,
            "match_start": idx,
            "match_end": idx + len(needle),
            "excerpt_start": before,
            "excerpt_end": after,
            "excerpt_text": content[before:after],
            "debug_log_id": d.get("log_id"),
            "skeleton_index": d.get("skeleton_index"),
            "key_differentiator": d.get("key_differentiator"),
        }

    return None


def get_fact_check_summary_by_gen(gen_id):
    """Return fact check summary counts for a generation_id (used in workflow view)."""
    if not gen_id:
        return {}
    with ENGINE.begin() as conn:
        row = conn.execute(
            text(
                "SELECT "
                "COUNT(*) as total_claims, "
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
    if not row or not row["total_claims"]:
        return {}
    result = dict(row)
    for k in ("total_claims", "verified", "unverified", "disputed", "outdated"):
        result[k] = int(result.get(k) or 0)
    avg = result.get("avg_confidence")
    result["avg_confidence"] = float(avg) if avg else 0.0
    return result


def get_fact_check_details_by_gen(gen_id):
    """Return per-evidence fact check details for the workflow UI table."""
    if not gen_id:
        return []
    with ENGINE.begin() as conn:
        rows = conn.execute(
            text(
                "SELECT fc.fact_check_id, fc.evidence_id, fc.status AS verdict, "
                "fc.confidence_score, fc.reasoning, fc.dispute_details, "
                "fc.fact_check_source_text, "
                "s.source_url, s.source_name, "
                "e.traces_to_text, e.claim_id, "
                "c.headline AS claim_headline, c.description AS claim_description, "
                "co.company_name, co.company_type "
                "FROM fact_checks fc "
                "JOIN evidence e ON fc.evidence_id = e.evidence_id "
                "JOIN claims c ON e.claim_id = c.claim_id "
                "JOIN companies co ON c.company_id = co.company_id "
                "LEFT JOIN sources s ON fc.fact_check_source_id = s.source_id "
                "WHERE c.generation_id = :gid "
                "ORDER BY fc.fact_check_id"
            ),
            {"gid": gen_id},
        ).mappings().all()
    return [dict(r) for r in rows]


def get_fact_check_summary(battlecard_id):
    gen = _fetch_generation_by_uuid(battlecard_id)
    if not gen:
        return {}

    gen_id = gen["generation_id"]
    with ENGINE.begin() as conn:
        row = conn.execute(
            text(
                "SELECT "
                "COUNT(*) as total_claims, "
                "SUM(CASE WHEN status = 'verified' THEN 1 ELSE 0 END) as verified, "
                "SUM(CASE WHEN status = 'unverified' THEN 1 ELSE 0 END) as unverified, "
                "SUM(CASE WHEN status = 'disputed' THEN 1 ELSE 0 END) as disputed, "
                "SUM(CASE WHEN status = 'outdated' THEN 1 ELSE 0 END) as outdated, "
                "ROUND(AVG(confidence_score), 2) as avg_confidence "
                "FROM fact_checks fc "
                "JOIN evidence e ON fc.evidence_id = e.evidence_id "
                "JOIN claims c ON e.claim_id = c.claim_id "
                "WHERE c.generation_id = :gid"
            ),
            {"gid": gen_id},
        ).mappings().first()

    if not row:
        return {}

    result = dict(row)
    for k in ("total_claims", "verified", "unverified", "disputed", "outdated"):
        result[k] = int(result.get(k) or 0)
    avg = result.get("avg_confidence")
    if avg is None:
        result["avg_confidence"] = 0.0
    else:
        result["avg_confidence"] = float(avg) / 100.0 if avg > 1 else float(avg)
    return result


def save_review_to_db(battlecard_id, diff_id, review_scope, status, comment="", rating_value=None, reorder=None, reviewer_email="anonymous"):
    gen = _fetch_generation_by_uuid(battlecard_id)
    if not gen:
        return None

    gen_id = gen["generation_id"]
    key_diff_id = None
    if diff_id and diff_id.startswith("kd-"):
        try:
            key_diff_id = int(diff_id.replace("kd-", ""))
        except ValueError:
            key_diff_id = None

    # Parse detail_item_id from scope like "databricks_detail_42" or "fabric_detail_42"
    detail_item_id = None
    if review_scope and ("_detail_" in review_scope):
        parts = review_scope.rsplit("_", 1)
        if len(parts) == 2:
            try:
                detail_item_id = int(parts[1])
            except ValueError:
                pass

    status_map = {
        "approved": "approve",
        "needs_revision": "request_edit",
        "rejected": "reject",
    }
    review_type = status_map.get(status, "provide_feedback")

    payload = {
        "scope": review_scope,
        "comment": comment,
        "key_diff_id": key_diff_id,
        "diff_id": diff_id,
    }
    if detail_item_id:
        payload["detail_item_id"] = detail_item_id
    if reorder:
        payload["reorder"] = reorder

    with ENGINE.begin() as conn:
        review_id = conn.execute(
            text(
                "INSERT INTO human_reviews (generation_id, claim_id, detail_item_id, review_type, feedback_text, "
                "reviewed_by, reviewer_role) "
                "VALUES (:gen, :claim, :detail_item, :rtype, :feedback, :by, :role) RETURNING review_id"
            ),
            {
                "gen": gen_id,
                "claim": None,
                "detail_item": detail_item_id,
                "rtype": review_type,
                "feedback": json.dumps(payload),
                "by": reviewer_email,
                "role": "reviewer",
            },
        ).scalar()

    return review_id


# =============================================================================
# Utility
# =============================================================================


def calculate_trial_score(battlecard_id, differentiators, feedback):
    keydiff_reviewed = 0
    keydiff_approved = 0
    keydiff_revision = 0

    databricks_reviewed = 0
    databricks_approved = 0
    databricks_revision = 0

    fabric_reviewed = 0
    fabric_approved = 0
    fabric_revision = 0

    for diff in differentiators:
        diff_id = diff["id"]
        diff_fb = feedback.get(diff_id, {})

        keydiff_status = diff_fb.get("key_diff", {}).get("status")
        if keydiff_status:
            keydiff_reviewed += 1
            if keydiff_status == "approved":
                keydiff_approved += 1
            elif keydiff_status == "needs_revision":
                keydiff_revision += 1

        db_status = diff_fb.get("databricks", {}).get("status")
        if db_status:
            databricks_reviewed += 1
            if db_status == "approved":
                databricks_approved += 1
            elif db_status == "needs_revision":
                databricks_revision += 1

        fabric_status = diff_fb.get("fabric", {}).get("status")
        if fabric_status:
            fabric_reviewed += 1
            if fabric_status == "approved":
                fabric_approved += 1
            elif fabric_status == "needs_revision":
                fabric_revision += 1

    def calc_score(reviewed, approved, revision):
        if reviewed == 0:
            return 0
        return int(((approved + (revision * 0.5)) / reviewed) * 100)

    keydiff_score = calc_score(keydiff_reviewed, keydiff_approved, keydiff_revision)
    databricks_score = calc_score(databricks_reviewed, databricks_approved, databricks_revision)
    fabric_score = calc_score(fabric_reviewed, fabric_approved, fabric_revision)

    total_reviewed = keydiff_reviewed + databricks_reviewed + fabric_reviewed
    total_items = len(differentiators)

    scores = [s for s in (keydiff_score, databricks_score, fabric_score) if s > 0]
    overall_score = int(sum(scores) / len(scores)) if scores else 0

    reorder_count = 0
    for diff in differentiators:
        diff_id = diff["id"]
        if feedback.get(diff_id, {}).get("reorder"):
            reorder_count += 1

    return {
        "score": overall_score,
        "keydiff_score": keydiff_score,
        "databricks_score": databricks_score,
        "fabric_score": fabric_score,
        "reviewed": total_reviewed,
        "total": total_items,
        "reordered": reorder_count,
    }


# =============================================================================
# Routes
# =============================================================================


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/design-system')
def design_system():
    return render_template('design_system.html', rubrics=[])


@app.route('/battlecards')
def list_battlecards():
    with ENGINE.begin() as conn:
        rows = conn.execute(
            text(
                "SELECT "
                "  ws.session_id::text, "
                "  ws.competitor_name, "
                "  ws.product_area, "
                "  ws.current_step, "
                "  ws.status AS workflow_status, "
                "  ws.model_name, "
                "  ws.created_at, "
                "  ws.updated_at, "
                "  bg.generation_id, "
                "  bg.generation_uuid::text AS battlecard_id, "
                "  bg.status AS battlecard_status, "
                "  bg.total_claims, "
                "  bg.verified_claims, "
                "  bg.unverified_claims, "
                "  bg.disputed_claims, "
                "  ROW_NUMBER() OVER ( "
                "      PARTITION BY ws.competitor_name, ws.product_area "
                "      ORDER BY ws.created_at "
                "  ) AS version_number "
                "FROM workflow_sessions ws "
                "JOIN battlecard_generations bg ON ws.generation_id = bg.generation_id "
                "ORDER BY ws.created_at DESC"
            )
        ).mappings().all()

    entries = []
    for r in rows:
        review = _get_review_feedback_summary(r["generation_id"])
        entries.append({
            "session_id": r["session_id"],
            "competitor_name": r["competitor_name"],
            "product_area": r["product_area"],
            "current_step": r["current_step"],
            "workflow_status": r["workflow_status"],
            "model_name": r["model_name"],
            "created_at": str(r["created_at"] or ""),
            "updated_at": str(r["updated_at"] or ""),
            "battlecard_id": r["battlecard_id"],
            "battlecard_status": r["battlecard_status"],
            "total_claims": r["total_claims"] or 0,
            "verified_claims": r["verified_claims"] or 0,
            "unverified_claims": r["unverified_claims"] or 0,
            "disputed_claims": r["disputed_claims"] or 0,
            "version_number": r["version_number"],
            "review": review,
        })

    # Collect distinct competitors for the "new version" flow
    competitors = []
    seen = set()
    for e in entries:
        key = (e["competitor_name"], e["product_area"])
        if key not in seen:
            seen.add(key)
            competitors.append({"competitor_name": e["competitor_name"], "product_area": e["product_area"]})

    return render_template('battlecards.html', entries=entries, competitors=competitors)


@app.route('/workflow')
def workflow_list():
    return redirect('/battlecards')


def _load_prompt_templates_for_ui():
    """Load prompt templates from DB for use in workflow creation dropdowns.

    Returns (pass1_templates, pass2_templates) as dicts keyed by display_order,
    matching the format expected by workflow_new.html and workflow_session.html.
    Falls back to the Python dict constants if the DB table is empty.
    """
    try:
        with ENGINE.begin() as conn:
            rows = [
                dict(r) for r in conn.execute(
                    text(
                        "SELECT template_id, template_name, template_type, version_label, "
                        "description, display_order, is_active, is_default "
                        "FROM prompt_templates WHERE is_active = TRUE "
                        "ORDER BY template_type, display_order"
                    )
                ).mappings().all()
            ]
        if rows:
            p1 = {}
            p2 = {}
            for r in rows:
                entry = {
                    "label": r["version_label"] or r["template_name"],
                    "description": r["description"] or "",
                    "template_id": r["template_id"],
                    "is_default": r["is_default"],
                }
                if r["template_type"] == "pass1":
                    p1[r["display_order"]] = entry
                elif r["template_type"] == "pass2":
                    p2[r["display_order"]] = entry
            if p1 or p2:
                return p1 or PASS1_PROMPT_TEMPLATES, p2 or PASS2_PROMPT_TEMPLATES
    except Exception as e:
        logger.warning("Failed to load prompt templates from DB: %s", e)
    return PASS1_PROMPT_TEMPLATES, PASS2_PROMPT_TEMPLATES


def _pass2_supports_category_batch(version: int) -> bool:
    return int(version or 0) in (5, 6, 7, 8, 9, 10)


def _pass2_supports_single_call(version: int) -> bool:
    return int(version or 0) in (8,)


def _resolve_step4_execution(pass1_ver: int, mode: str, workers: int, core_count: int) -> dict:
    mode = (mode or "auto").strip().lower()
    workers = max(1, int(workers or 1))
    core_count = max(0, int(core_count or 0))

    if mode == "single_call":
        return {
            "mode": "sequential",
            "summary": "Single-call planning",
            "details": f"Prompt V{pass1_ver} with one request",
            "runtime_mode": "single_call",
            "workers": 1,
            "batches": 1 if core_count > 0 else 0,
        }
    if mode in ("category_parallel", "category_sequential"):
        can_batch = core_count > 1
        use_parallel = mode == "category_parallel" and can_batch and workers > 1
        worker_cap = min(workers, max(core_count, 1), 6) if use_parallel else 1
        return {
            "mode": "parallel" if use_parallel else "sequential",
            "summary": (
                f"Parallel by category ({core_count} calls, up to {worker_cap} workers)"
                if use_parallel
                else f"Sequential by category ({core_count} calls)"
            ),
            "details": f"Prompt V4 runtime path ({'parallel' if use_parallel else 'sequential'} category mode)",
            "runtime_mode": "category_parallel" if use_parallel else "category_sequential",
            "workers": worker_cap,
            "batches": core_count,
        }

    # auto
    pass1_parallel = bool(pass1_ver >= 3 and workers > 1 and core_count > 1)
    pass1_effective_ver = 4 if pass1_parallel else pass1_ver
    pass1_batches = core_count if pass1_parallel else (1 if core_count > 0 else 0)
    pass1_workers = min(workers, pass1_batches, 6) if pass1_parallel else 1
    return {
        "mode": "parallel" if pass1_parallel else "sequential",
        "summary": (
            f"Parallel by category ({pass1_batches} calls, up to {pass1_workers} workers)"
            if pass1_parallel
            else "Sequential single-call planning"
        ),
        "details": f"Prompt V{pass1_effective_ver} runtime behavior (auto)",
        "runtime_mode": "category_parallel" if pass1_parallel else "single_call",
        "workers": pass1_workers,
        "batches": pass1_batches,
    }


def _resolve_step5_execution(pass2_ver: int, mode: str, workers: int, core_count: int, diffs_per_category: int) -> dict:
    mode = (mode or "auto").strip().lower()
    workers = max(1, int(workers or 1))
    core_count = max(0, int(core_count or 0))
    diffs_per_category = max(0, int(diffs_per_category or 0))
    total_diffs = core_count * diffs_per_category
    supports_category = _pass2_supports_category_batch(pass2_ver)
    supports_single = _pass2_supports_single_call(pass2_ver)

    def _per_diff(parallel: bool, explicit: bool = False) -> dict:
        use_parallel = parallel and workers > 1 and total_diffs > 1
        wc = min(workers, max(total_diffs, 1), 8) if use_parallel else 1
        return {
            "mode": "parallel" if use_parallel else "sequential",
            "summary": (
                f"Parallel per-diff calls (~{total_diffs} calls, up to {wc} workers)"
                if use_parallel
                else "Sequential per-diff generation"
            ),
            "details": f"Prompt V{pass2_ver} (per-diff{' explicit' if explicit else ''})",
            "runtime_mode": "per_diff_parallel" if use_parallel else "per_diff_sequential",
            "workers": wc,
            "batches": total_diffs,
            "fallback_reason": None,
        }

    def _category(parallel: bool, explicit: bool = False) -> dict:
        if not supports_category:
            out = _per_diff(parallel=False, explicit=True)
            out["fallback_reason"] = f"Prompt V{pass2_ver} does not support category batching."
            return out
        use_parallel = parallel and workers > 1 and core_count > 1
        wc = min(workers, max(core_count, 1), 4) if use_parallel else 1
        return {
            "mode": "parallel" if use_parallel else "sequential",
            "summary": (
                f"Parallel category batches ({core_count} calls, up to {wc} workers)"
                if use_parallel
                else "Sequential category batches"
            ),
            "details": f"Prompt V{pass2_ver} (category-batched{' explicit' if explicit else ''})",
            "runtime_mode": "category_parallel" if use_parallel else "category_sequential",
            "workers": wc,
            "batches": core_count,
            "fallback_reason": None,
        }

    if mode == "single_call":
        if not supports_single:
            out = _per_diff(parallel=False, explicit=True)
            out["fallback_reason"] = f"Prompt V{pass2_ver} does not support single-call batching."
            return out
        return {
            "mode": "sequential",
            "summary": f"Single-call generation ({total_diffs} diffs in one request)",
            "details": f"Prompt V{pass2_ver} (single-call)",
            "runtime_mode": "single_call",
            "workers": 1,
            "batches": 1 if total_diffs > 0 else 0,
            "fallback_reason": None,
        }
    if mode == "per_diff_parallel":
        return _per_diff(parallel=True, explicit=True)
    if mode == "per_diff_sequential":
        return _per_diff(parallel=False, explicit=True)
    if mode == "category_parallel":
        return _category(parallel=True, explicit=True)
    if mode == "category_sequential":
        return _category(parallel=False, explicit=True)

    # auto
    if pass2_ver in (5, 6, 7, 8, 9, 10):
        return _category(parallel=True, explicit=False)
    return _per_diff(parallel=True, explicit=False)


@app.route('/workflow/new')
def workflow_new():
    # Fetch existing competitors so the user can create a new version
    with ENGINE.begin() as conn:
        existing = conn.execute(
            text(
                "SELECT DISTINCT ws.competitor_name, ws.product_area "
                "FROM workflow_sessions ws "
                "JOIN battlecard_generations bg ON ws.generation_id = bg.generation_id "
                "ORDER BY ws.competitor_name"
            )
        ).mappings().all()
    competitors = [dict(r) for r in existing]
    p1_templates, p2_templates = _load_prompt_templates_for_ui()
    return render_template('workflow_new.html', competitors=competitors,
                           pass1_prompt_templates=p1_templates,
                           pass2_prompt_templates=p2_templates,
                           step4_execution_options=STEP4_EXECUTION_OPTIONS,
                           step5_execution_options=STEP5_EXECUTION_OPTIONS)


@app.route('/workflow/<session_id>')
def workflow_session(session_id):
    with ENGINE.begin() as conn:
        session = conn.execute(
            text(
                "SELECT session_id::text, competitor_name, product_area, current_step, status, "
                "model_name, diffs_per_category, max_workers, generation_id, "
                "COALESCE(step4_execution_mode, 'auto') AS step4_execution_mode, "
                "COALESCE(step5_execution_mode, 'auto') AS step5_execution_mode, "
                "COALESCE(step5_inline_fact_check, FALSE) AS step5_inline_fact_check, "
                "pass1_prompt_template_version, pass2_prompt_template_version, created_at "
                "FROM workflow_sessions WHERE session_id::text = :sid"
            ),
            {"sid": session_id},
        ).mappings().first()
        if not session:
            return "Workflow session not found", 404

        steps = conn.execute(
            text(
                "SELECT step_id, step_number, step_name, status, progress_current, progress_total, "
                "progress_message, error_message, started_at, completed_at "
                "FROM workflow_steps WHERE session_id::text = :sid ORDER BY step_number"
            ),
            {"sid": session_id},
        ).mappings().all()

        # Get only the LATEST artifact per (step_number, artifact_type) to avoid duplicates
        # from incremental saves during parallel processing
        artifacts = conn.execute(
            text(
                "SELECT DISTINCT ON (step_number, artifact_type) "
                "artifact_id, step_number, artifact_type, artifact_name, "
                "artifact_content, artifact_metadata, created_at "
                "FROM workflow_artifacts WHERE session_id::text = :sid "
                "ORDER BY step_number, artifact_type, created_at DESC"
            ),
            {"sid": session_id},
        ).mappings().all()

    steps_list = [dict(s) for s in steps]
    artifacts_list = [dict(a) for a in artifacts]
    # Parse JSONB metadata and JSON artifact content
    for a in artifacts_list:
        if a.get("artifact_metadata") and isinstance(a["artifact_metadata"], str):
            try:
                a["artifact_metadata"] = json.loads(a["artifact_metadata"])
            except (json.JSONDecodeError, TypeError):
                pass
        # Parse artifact_content that is stored as JSON string (e.g., category_selections)
        if a.get("artifact_content") and isinstance(a["artifact_content"], str) and a.get("artifact_type") in ("category_selections",):
            try:
                a["artifact_content"] = json.loads(a["artifact_content"])
            except (json.JSONDecodeError, TypeError):
                pass

    artifacts_by_step = {}
    for a in artifacts_list:
        artifacts_by_step.setdefault(a["step_number"], []).append(a)

    # Look up linked battlecard for header link (always available if generation exists)
    battlecard_link_id = _lookup_battlecard_for_workflow(session_id)

    # Only show the battlecard preview panel if Step 5 claims exist (meaningful content to preview)
    battlecard_id = None
    has_claims = any(
        a["artifact_type"] in ("pass2_claims", "pass3_regenerated")
        for a in artifacts_list
        if a["step_number"] == 5
    )
    if has_claims:
        battlecard_id = battlecard_link_id

    review_summary = None
    if session["generation_id"]:
        review_summary = _get_review_feedback_summary(session["generation_id"])

    # Load product category catalog for Step 3
    with ENGINE.begin() as conn:
        catalog_rows = conn.execute(
            text(
                "SELECT catalog_id, category_name, category_description, "
                "is_core_product_category, is_cross_platform_capability, display_order "
                "FROM product_category_catalog ORDER BY display_order"
            )
        ).mappings().all()
    category_catalog = [dict(r) for r in catalog_rows]

    # Load existing session category selections (if any)
    with ENGINE.begin() as conn:
        sel_rows = conn.execute(
            text(
                "SELECT scs.catalog_id, scs.inclusion_type, scs.display_order "
                "FROM session_category_selections scs "
                "WHERE scs.session_id::text = :sid ORDER BY scs.display_order"
            ),
            {"sid": session_id},
        ).mappings().all()
    session_selections = {r["catalog_id"]: r["inclusion_type"] for r in sel_rows}

    # Load previous version Step 3 selections for baseline comparison.
    previous_session_selections = {}
    previous_selection_source = {
        "previous_generation_id": None,
        "session_id": None,
        "selection_count": 0,
    }
    with ENGINE.begin() as conn:
        prev_gen_id = conn.execute(
            text(
                "SELECT bg.previous_generation_id "
                "FROM workflow_sessions ws "
                "JOIN battlecard_generations bg ON ws.generation_id = bg.generation_id "
                "WHERE ws.session_id::text = :sid"
            ),
            {"sid": session_id},
        ).scalar()
        previous_selection_source["previous_generation_id"] = prev_gen_id
        if prev_gen_id:
            prev_session_id = conn.execute(
                text(
                    "SELECT ws.session_id::text "
                    "FROM workflow_sessions ws "
                    "WHERE ws.generation_id = :gid "
                    "AND EXISTS ("
                    "  SELECT 1 FROM session_category_selections scs "
                    "  WHERE scs.session_id = ws.session_id"
                    ") "
                    "ORDER BY ws.created_at DESC LIMIT 1"
                ),
                {"gid": prev_gen_id},
            ).scalar()
            previous_selection_source["session_id"] = prev_session_id
            if prev_session_id:
                prev_rows = conn.execute(
                    text(
                        "SELECT catalog_id, inclusion_type "
                        "FROM session_category_selections "
                        "WHERE session_id::text = :sid"
                    ),
                    {"sid": prev_session_id},
                ).mappings().all()
                previous_session_selections = {r["catalog_id"]: r["inclusion_type"] for r in prev_rows}
                previous_selection_source["selection_count"] = len(previous_session_selections)

    # Load product mappings for the session's competitor (for display context)
    competitor_name = session.get("competitor_name", "")
    with ENGINE.begin() as conn:
        pm_rows = conn.execute(
            text(
                "SELECT pm.catalog_id, pm.vendor, pm.product_name, pm.product_description "
                "FROM product_mappings pm "
                "WHERE pm.vendor = 'databricks' OR (pm.vendor = 'competitor' AND pm.competitor_name = :comp) "
                "ORDER BY pm.catalog_id, pm.vendor, pm.display_order"
            ),
            {"comp": competitor_name},
        ).mappings().all()
    product_mappings_by_catalog = {}
    for pm in pm_rows:
        cid = pm["catalog_id"]
        product_mappings_by_catalog.setdefault(cid, {"databricks": [], "competitor": []})
        product_mappings_by_catalog[cid][pm["vendor"]].append({
            "name": pm["product_name"],
            "description": pm["product_description"],
        })

    effective_step3_selections = session_selections or previous_session_selections
    core_count = sum(1 for v in effective_step3_selections.values() if v == "core_product_category")
    pass1_ver = int(session.get("pass1_prompt_template_version") or 3)
    pass2_ver = int(session.get("pass2_prompt_template_version") or 2)
    configured_workers = int(session.get("max_workers") or 1)
    diffs_per_category = int(session.get("diffs_per_category") or 0)
    step4_mode_cfg = (session.get("step4_execution_mode") or "auto").strip().lower()
    step5_mode_cfg = (session.get("step5_execution_mode") or "auto").strip().lower()
    step4_exec = _resolve_step4_execution(pass1_ver, step4_mode_cfg, configured_workers, core_count)
    step5_exec = _resolve_step5_execution(pass2_ver, step5_mode_cfg, configured_workers, core_count, diffs_per_category)
    pass1_parallel = step4_exec["mode"] == "parallel"
    pass2_parallel = step5_exec["mode"] == "parallel"

    step_execution_modes = {
        4: {
            "mode": step4_exec["mode"],
            "summary": step4_exec["summary"],
            "details": step4_exec["details"],
            "runtime_mode": step4_exec["runtime_mode"],
            "configured_mode": step4_mode_cfg,
        },
        5: {
            "mode": step5_exec["mode"],
            "summary": step5_exec["summary"],
            "details": step5_exec["details"],
            "runtime_mode": step5_exec["runtime_mode"],
            "configured_mode": step5_mode_cfg,
            "fallback_reason": step5_exec.get("fallback_reason"),
        },
    }
    fastest_strategy_hint = (
        "Fastest balanced setup: Step 3 select only core categories you need, "
        "then use category-batched Step 5 with 3-4 workers and inline fact-check disabled."
    )

    # Load fact check summary for step 6
    fact_check_summary = {}
    fact_check_details = []
    if session.get("generation_id"):
        fact_check_summary = get_fact_check_summary_by_gen(session["generation_id"])
        fact_check_details = get_fact_check_details_by_gen(session["generation_id"])

    # Check if generation steps (4+) have started — locks context steps 1-3
    context_steps_locked = any(
        s.get("status") not in ("pending", None)
        for s in steps_list if s.get("step_number", 0) >= 4
    )

    p1_templates, p2_templates = _load_prompt_templates_for_ui()

    return render_template(
        'workflow_session.html',
        session=dict(session),
        steps=steps_list,
        artifacts=artifacts_list,
        artifacts_by_step=artifacts_by_step,
        category_catalog=category_catalog,
        session_selections=session_selections,
        previous_session_selections=previous_session_selections,
        previous_selection_source=previous_selection_source,
        product_mappings_by_catalog=product_mappings_by_catalog,
        battlecard_id=battlecard_id,
        battlecard_link_id=battlecard_link_id,
        review_summary=review_summary,
        fact_check_summary=fact_check_summary,
        fact_check_details=fact_check_details,
        pill_config=PILL_CONFIG,
        pass1_prompt_templates=p1_templates,
        pass2_prompt_templates=p2_templates,
        step4_execution_options=STEP4_EXECUTION_OPTIONS,
        step5_execution_options=STEP5_EXECUTION_OPTIONS,
        context_steps_locked=context_steps_locked,
        step_execution_modes=step_execution_modes,
        fastest_strategy_hint=fastest_strategy_hint,
    )


@app.route('/battlecard/<battlecard_id>')
def view_battlecard(battlecard_id):
    slides, gen_info = load_battlecard_slides(battlecard_id)
    if not slides:
        return "Battlecard not found", 404

    feedback = load_battlecard_reviews(battlecard_id)
    fact_checks = load_battlecard_fact_checks(battlecard_id)
    fact_check_summary = get_fact_check_summary(battlecard_id)
    context_documents = load_context_documents(battlecard_id)
    agent_session = load_agent_session(battlecard_id)
    theme_coverage = get_key_diff_theme_coverage(battlecard_id)

    categories = {}
    for diff in slides:
        cat = diff["category"]
        categories.setdefault(cat, [])

        diff_fb = feedback.get(diff["id"], {})
        # Start with all scopes from loaded reviews (includes detail-level
        # scopes like databricks_detail_203, fabric_detail_204, etc.)
        diff["feedback"] = dict(diff_fb)
        # Ensure known top-level keys always exist so templates don't error
        diff["feedback"].setdefault("key_diff", {})
        diff["feedback"].setdefault("databricks", {})
        diff["feedback"].setdefault("fabric", {})
        diff["feedback"].setdefault("reorder", None)

        slide_fact_checks = fact_checks.get(diff["id"], [])
        diff["fact_checks"] = slide_fact_checks

        content_for_matching = {
            "citations": diff.get("citations", {}),
            "databricks_details": diff.get("databricks_details", ""),
            "competitor_details": diff.get("fabric_details", ""),
            "description": diff.get("description", ""),
            "databricks_reasoning": diff.get("databricks_reasoning", ""),
            "competitor_reasoning": diff.get("fabric_reasoning", ""),
        }
        diff["merged_citations"] = match_fact_checks_to_citations(
            content_for_matching, slide_fact_checks
        )

        categories[cat].append(diff)

    for cat in categories:
        categories[cat].sort(
            key=lambda x: x["feedback"]["reorder"]["new_rank"]
            if x["feedback"].get("reorder") else x["rank"]
        )

    has_flagged = False
    flagged_statuses = {"needs_revision", "rejected"}
    for diff in slides:
        diff_fb = feedback.get(diff["id"], {})
        for scope in ("key_diff", "databricks", "fabric"):
            if diff_fb.get(scope, {}).get("status") in flagged_statuses:
                has_flagged = True
                break
        if has_flagged:
            break

    embedded = request.args.get('embedded', '').lower() in ('true', '1', 'yes')

    trial = {
        "name": f"{gen_info.get('competitor', 'Unknown')} vs Databricks (v{gen_info.get('battlecard_version_id', '?')})",
        "battlecard_id": battlecard_id,
        "competitor": gen_info.get("competitor", ""),
        "product_area": gen_info.get("product_area", ""),
        "agent_run_id": gen_info.get("agent_run_id", ""),
        "experiment_id": gen_info.get("experiment_id", ""),
        "prompt_name": gen_info.get("prompt_name", ""),
        "prompt_version": gen_info.get("prompt_version", 0),
        "is_uc_battlecard": True,
        "has_flagged_slides": has_flagged,
        "session_id": _lookup_workflow_session_for_battlecard(battlecard_id),
        "mlflow_experiment_url": "",
        "mlflow_run_url": "",
        "embedded": embedded,
    }

    slide_categories = list(categories.keys())
    # Load category ordering from product_category_catalog
    with ENGINE.begin() as conn:
        _cat_order_rows = conn.execute(
            text("SELECT category_name FROM product_category_catalog ORDER BY display_order")
        ).scalars().all()
    dynamic_category_order = list(_cat_order_rows)
    for cat in slide_categories:
        if cat not in dynamic_category_order:
            dynamic_category_order.append(cat)

    scores = calculate_trial_score(battlecard_id, slides, feedback)

    return render_template(
        "trial.html",
        trial=trial,
        categories=categories,
        scores=scores,
        category_order=dynamic_category_order,
        rubrics=[],
        fact_check_summary=fact_check_summary,
        context_documents=context_documents,
        agent_session=agent_session,
        theme_coverage=theme_coverage,
    )


# =============================================================================
# API Routes
# =============================================================================


@app.route('/api/uc/feedback', methods=['POST'])
def submit_uc_feedback():
    data = request.json
    battlecard_id = data.get('battlecard_id')
    diff_id = data.get('diff_id')
    level = data.get('level')
    status = data.get('status')
    comment = data.get('comment', '')

    logger.info(
        "FEEDBACK SAVE: battlecard=%s diff_id=%s level=%s status=%s comment=%r",
        battlecard_id, diff_id, level, status, comment,
    )

    if not all([battlecard_id, diff_id, level]):
        logger.warning("FEEDBACK SAVE REJECTED: missing required fields – battlecard=%s diff_id=%s level=%s", battlecard_id, diff_id, level)
        return jsonify({"error": "Missing required fields"}), 400

    review_id = save_review_to_db(
        battlecard_id=battlecard_id,
        diff_id=diff_id,
        review_scope=level,
        status=status,
        comment=comment,
        reviewer_email="anonymous",
    )

    logger.info("FEEDBACK SAVED: review_id=%s battlecard=%s diff_id=%s level=%s status=%s", review_id, battlecard_id, diff_id, level, status)

    slides, _ = load_battlecard_slides(battlecard_id)
    feedback = load_battlecard_reviews(battlecard_id)
    scores = calculate_trial_score(battlecard_id, slides, feedback)

    return jsonify({"success": True, "scores": scores, "human_review_id": review_id})


@app.route('/api/uc/citation-feedback', methods=['POST'])
def submit_uc_citation_feedback():
    """Persist human review for a single citation/evidence row."""
    data = request.json or {}
    battlecard_id = data.get("battlecard_id")
    diff_id = data.get("diff_id")
    evidence_id = data.get("evidence_id")
    fact_check_id = data.get("fact_check_id")
    status = (data.get("status") or "").strip()
    comment = (data.get("comment") or "").strip()

    if not battlecard_id or not diff_id or not evidence_id:
        return jsonify({"error": "battlecard_id, diff_id, and evidence_id are required"}), 400

    status_map = {
        "approved": "approve",
        "needs_revision": "request_edit",
        "rejected": "reject",
    }
    review_type = status_map.get(status, "provide_feedback")

    gen = _fetch_generation_by_uuid(battlecard_id)
    if not gen:
        return jsonify({"error": "Battlecard not found"}), 404

    payload = {
        "scope": f"citation:{evidence_id}",
        "diff_id": diff_id,
        "comment": comment,
        "status": status,
        "evidence_id": int(evidence_id),
        "fact_check_id": int(fact_check_id) if fact_check_id else None,
    }

    with ENGINE.begin() as conn:
        review_id = conn.execute(
            text(
                "INSERT INTO human_reviews (generation_id, claim_id, detail_item_id, evidence_id, fact_check_id, "
                "review_type, feedback_text, reviewed_by, reviewer_role) "
                "VALUES (:gen, NULL, NULL, :eid, :fcid, :rtype, :feedback, :by, :role) "
                "RETURNING review_id"
            ),
            {
                "gen": gen["generation_id"],
                "eid": int(evidence_id),
                "fcid": int(fact_check_id) if fact_check_id else None,
                "rtype": review_type,
                "feedback": json.dumps(payload),
                "by": "anonymous",
                "role": "reviewer",
            },
        ).scalar()

    return jsonify({"success": True, "review_id": review_id})


@app.route('/api/uc/fact-check-feedback', methods=['POST'])
def submit_uc_fact_check_feedback():
    """Persist human feedback for a fact-check verdict row."""
    data = request.json or {}
    battlecard_id = data.get("battlecard_id")
    diff_id = data.get("diff_id")
    fact_check_id = data.get("fact_check_id")
    status = (data.get("status") or "").strip()
    comment = (data.get("comment") or "").strip()
    if not battlecard_id or not fact_check_id:
        return jsonify({"error": "battlecard_id and fact_check_id are required"}), 400

    gen = _fetch_generation_by_uuid(battlecard_id)
    if not gen:
        return jsonify({"error": "Battlecard not found"}), 404

    status_map = {
        "approved": "approve",
        "needs_revision": "request_edit",
        "rejected": "reject",
    }
    review_type = status_map.get(status, "override_fact_check")
    payload = {
        "scope": f"fact_check:{fact_check_id}",
        "diff_id": diff_id,
        "comment": comment,
        "status": status,
        "fact_check_id": int(fact_check_id),
    }

    with ENGINE.begin() as conn:
        review_id = conn.execute(
            text(
                "INSERT INTO human_reviews (generation_id, claim_id, detail_item_id, evidence_id, fact_check_id, "
                "review_type, feedback_text, reviewed_by, reviewer_role) "
                "VALUES (:gen, NULL, NULL, NULL, :fcid, :rtype, :feedback, :by, :role) "
                "RETURNING review_id"
            ),
            {
                "gen": gen["generation_id"],
                "fcid": int(fact_check_id),
                "rtype": review_type,
                "feedback": json.dumps(payload),
                "by": "anonymous",
                "role": "reviewer",
            },
        ).scalar()

    return jsonify({"success": True, "review_id": review_id})


@app.route('/api/uc/reorder', methods=['POST'])
def reorder_key_diffs():
    data = request.json
    battlecard_id = data.get('battlecard_id')
    category = data.get('category')
    new_order = data.get('new_order')

    if not battlecard_id or not category or not new_order:
        return jsonify({"error": "Missing required fields"}), 400

    for idx, diff_id in enumerate(new_order, start=1):
        save_review_to_db(
            battlecard_id=battlecard_id,
            diff_id=diff_id,
            review_scope="reorder",
            status="approved",
            comment="",
            reorder={"original_rank": None, "new_rank": idx},
            reviewer_email="anonymous",
        )

    return jsonify({"success": True})


@app.route('/api/battlecard/<battlecard_id>/prompt')
def get_prompt_details(battlecard_id):
    return jsonify({"prompt_name": "", "prompt_version": 0, "prompt_text": ""})


@app.route('/api/battlecard/<battlecard_id>/status')
def get_battlecard_status(battlecard_id):
    gen = _fetch_generation_by_uuid(battlecard_id)
    if not gen:
        return jsonify({"status": "unknown"})
    return jsonify({"status": gen.get("status", "unknown")})


@app.route('/api/battlecard/<battlecard_id>/citation/<int:evidence_id>/context')
def get_battlecard_citation_context(battlecard_id, evidence_id):
    """Resolve an internal citation to a context-document excerpt and location."""
    gen = _fetch_generation_by_uuid(battlecard_id)
    if not gen:
        return jsonify({"error": "Battlecard not found"}), 404

    with ENGINE.begin() as conn:
        row = conn.execute(
            text(
                "SELECT e.evidence_id, e.traces_to_text, e.generation_source_text, "
                "e.traces_to_start_index, e.traces_to_end_index, e.citation_id, "
                "s.source_id, s.source_name, s.source_url, s.source_type "
                "FROM evidence e "
                "JOIN claims c ON c.claim_id = e.claim_id "
                "LEFT JOIN sources s ON s.source_id = e.generation_source_id "
                "WHERE e.evidence_id = :eid AND c.generation_id = :gid"
            ),
            {"eid": evidence_id, "gid": gen["generation_id"]},
        ).mappings().first()
        if not row:
            return jsonify({"error": "Citation evidence not found"}), 404

    session_id = _lookup_workflow_session_for_battlecard(battlecard_id)
    source_text = (row.get("generation_source_text") or row.get("traces_to_text") or "").strip()
    context_match = _find_internal_source_context(session_id, source_text)
    if not context_match:
        context_match = {
            "artifact_id": None,
            "artifact_type": "evidence_snapshot",
            "artifact_name": row.get("source_name") or "Internal evidence",
            "step_number": None,
            "match_start": int(row.get("traces_to_start_index") or 0),
            "match_end": int(row.get("traces_to_end_index") or 0),
            "excerpt_start": int(row.get("traces_to_start_index") or 0),
            "excerpt_end": int(row.get("traces_to_end_index") or 0),
            "excerpt_text": source_text,
        }

    result = {
        "battlecard_id": battlecard_id,
        "session_id": session_id,
        "evidence_id": row["evidence_id"],
        "citation_id": row.get("citation_id"),
        "source_id": row.get("source_id"),
        "source_name": row.get("source_name") or "Internal document",
        "source_url": row.get("source_url") or "",
        "source_type": row.get("source_type") or "",
        "source_quote": source_text,
        "traces_start_index": row.get("traces_to_start_index"),
        "traces_end_index": row.get("traces_to_end_index"),
        "context_match": context_match,
    }
    return jsonify(result)


@app.route('/api/context-document/<doc_id>')
def get_context_document(doc_id):
    return jsonify({"content": ""})


# =============================================================================
# Token estimation API
# =============================================================================


@app.route('/api/workflow/<session_id>/step/<int:step_number>/token-estimate')
def workflow_token_estimate(session_id, step_number):
    """Return token usage estimates (input breakdown + output) for a workflow step."""
    result = estimate_prompt_tokens(session_id, step_number)
    return jsonify(result)


# =============================================================================
# Workflow <-> Battlecard linking API
# =============================================================================


@app.route('/api/workflow/<session_id>/review-feedback')
def workflow_review_feedback(session_id):
    """Return collected review feedback from the linked battlecard."""
    with ENGINE.begin() as conn:
        session = conn.execute(
            text(
                "SELECT generation_id FROM workflow_sessions WHERE session_id::text = :sid"
            ),
            {"sid": session_id},
        ).mappings().first()

    if not session or not session["generation_id"]:
        return jsonify({"feedback_text": "", "summary": {"total": 0, "approved": 0, "needs_revision": 0, "rejected": 0}})

    gen_id = session["generation_id"]
    feedback_text = _collect_review_feedback_text(gen_id)
    summary = _get_review_feedback_summary(gen_id)

    return jsonify({"feedback_text": feedback_text, "summary": summary})


# =============================================================================
# Workflow API Routes
# =============================================================================

WORKFLOW_STEPS = [
    (1, "Generate Directive"),
    (2, "Upload Old Battlecards"),
    (3, "Classify Product Categories"),
    (4, "Generate Key Differentiators"),
    (5, "Generate Key Diff Claims"),
    (6, "Fact Check Claims"),
    (7, "Generate Google Slides"),
]


@app.route('/api/workflow/create', methods=['POST'])
def create_workflow():
    data = request.json or {}
    competitor_name = data.get('competitor_name', '').strip()
    product_area = data.get('product_area', 'Data Platform').strip()
    model_name = data.get('model_name', 'databricks-claude-sonnet-4').strip()
    diffs_per_category = int(data.get('diffs_per_category', 10))
    max_workers = int(data.get('max_workers', 5))
    step4_execution_mode = str(data.get('step4_execution_mode', 'auto') or 'auto')
    step5_execution_mode = str(data.get('step5_execution_mode', 'auto') or 'auto')
    step5_inline_fact_check_raw = data.get('step5_inline_fact_check', False)
    if isinstance(step5_inline_fact_check_raw, str):
        step5_inline_fact_check = step5_inline_fact_check_raw.strip().lower() in ("1", "true", "yes", "on")
    else:
        step5_inline_fact_check = bool(step5_inline_fact_check_raw)
    pass1_prompt_template_version = int(data.get('pass1_prompt_template_version', 3))
    pass2_prompt_template_version = int(data.get('pass2_prompt_template_version', 2))
    previous_generation_id = data.get('previous_generation_id')
    if previous_generation_id in ("", None):
        previous_generation_id = None
    elif not isinstance(previous_generation_id, int):
        try:
            previous_generation_id = int(previous_generation_id)
        except (TypeError, ValueError):
            previous_generation_id = None

    if not competitor_name:
        return jsonify({"error": "competitor_name is required"}), 400
    if step4_execution_mode not in STEP4_EXECUTION_OPTIONS:
        step4_execution_mode = "auto"
    if step5_execution_mode not in STEP5_EXECUTION_OPTIONS:
        step5_execution_mode = "auto"

    with ENGINE.begin() as conn:
        # If no explicit previous_generation_id was provided but the competitor
        # already has workflows, auto-link to the latest generation for
        # version chaining.
        if not previous_generation_id:
            prev = conn.execute(
                text(
                    "SELECT bg.generation_id FROM workflow_sessions ws "
                    "JOIN battlecard_generations bg ON ws.generation_id = bg.generation_id "
                    "WHERE ws.competitor_name = :comp AND ws.product_area = :area "
                    "ORDER BY ws.created_at DESC LIMIT 1"
                ),
                {"comp": competitor_name, "area": product_area},
            ).scalar()
            if prev:
                previous_generation_id = prev

        # Create the battlecard generation record up front so the feedback
        # loop between the battlecard review page and the workflow is
        # available from the very start.
        gen_id = conn.execute(
            text(
                "INSERT INTO battlecard_generations "
                "(trigger_type, generated_by, generation_model, status, previous_generation_id) "
                "VALUES ('manual_request', 'workflow_runner', :model, 'draft', :prev) "
                "RETURNING generation_id"
            ),
            {"model": model_name, "prev": previous_generation_id},
        ).scalar()

        row = conn.execute(
            text(
                "INSERT INTO workflow_sessions ("
                "competitor_name, product_area, model_name, diffs_per_category, max_workers, "
                "step4_execution_mode, step5_execution_mode, step5_inline_fact_check, "
                "generation_id, pass1_prompt_template_version, pass2_prompt_template_version"
                ") VALUES ("
                ":comp, :area, :model, :diffs, :workers, "
                ":s4mode, :s5mode, :s5fc, "
                ":gid, :p1tv, :p2tv"
                ") RETURNING session_id::text"
            ),
            {
                "comp": competitor_name,
                "area": product_area,
                "model": model_name,
                "diffs": diffs_per_category,
                "workers": max_workers,
                "s4mode": step4_execution_mode,
                "s5mode": step5_execution_mode,
                "s5fc": step5_inline_fact_check,
                "gid": gen_id,
                "p1tv": pass1_prompt_template_version,
                "p2tv": pass2_prompt_template_version,
            },
        ).scalar()
        session_id = row

        for step_num, step_name in WORKFLOW_STEPS:
            status = "ready" if step_num == 1 else "pending"
            conn.execute(
                text(
                    "INSERT INTO workflow_steps (session_id, step_number, step_name, status) "
                    "VALUES (CAST(:sid AS uuid), :num, :name, :status)"
                ),
                {"sid": session_id, "num": step_num, "name": step_name, "status": status},
            )

        # For new versions, preload Step 3 defaults from the linked previous generation.
        seeded_category_defaults = 0
        if previous_generation_id:
            source_session_id = conn.execute(
                text(
                    "SELECT ws.session_id::text "
                    "FROM workflow_sessions ws "
                    "WHERE ws.generation_id = :gid "
                    "AND EXISTS ("
                    "  SELECT 1 FROM session_category_selections scs "
                    "  WHERE scs.session_id = ws.session_id"
                    ") "
                    "ORDER BY ws.created_at DESC LIMIT 1"
                ),
                {"gid": previous_generation_id},
            ).scalar()
            if source_session_id:
                seeded_category_defaults = conn.execute(
                    text(
                        "INSERT INTO session_category_selections (session_id, catalog_id, inclusion_type, display_order) "
                        "SELECT CAST(:sid AS uuid), scs.catalog_id, scs.inclusion_type, scs.display_order "
                        "FROM session_category_selections scs "
                        "WHERE scs.session_id::text = :src "
                        "ORDER BY scs.display_order, scs.catalog_id"
                    ),
                    {"sid": session_id, "src": source_session_id},
                ).rowcount or 0

    return jsonify({
        "success": True,
        "session_id": session_id,
        "seeded_category_defaults": seeded_category_defaults,
    })


# Heartbeat timeout threshold in seconds - steps in_progress longer than this are considered stuck
HEARTBEAT_TIMEOUT_SECONDS = 300  # 5 minutes without heartbeat = stuck


@app.route('/api/workflow/<session_id>/status')
def workflow_status(session_id):
    with ENGINE.begin() as conn:
        session = conn.execute(
            text(
                "SELECT session_id::text, competitor_name, product_area, current_step, status "
                "FROM workflow_sessions WHERE session_id::text = :sid"
            ),
            {"sid": session_id},
        ).mappings().first()
        if not session:
            return jsonify({"error": "not found"}), 404

        steps = conn.execute(
            text(
                "SELECT step_number, step_name, status, progress_current, progress_total, "
                "progress_message, error_message, error_details, started_at, heartbeat_at, worker_id, "
                "last_error, error_count, "
                "EXTRACT(EPOCH FROM (NOW() - heartbeat_at)) AS seconds_since_heartbeat "
                "FROM workflow_steps WHERE session_id::text = :sid ORDER BY step_number"
            ),
            {"sid": session_id},
        ).mappings().all()

    # Process steps to add stuck detection
    processed_steps = []
    for s in steps:
        step_dict = dict(s)
        # Convert timestamps to ISO strings for JSON
        if step_dict.get("started_at"):
            step_dict["started_at"] = step_dict["started_at"].isoformat() if hasattr(step_dict["started_at"], 'isoformat') else str(step_dict["started_at"])
        if step_dict.get("heartbeat_at"):
            step_dict["heartbeat_at"] = step_dict["heartbeat_at"].isoformat() if hasattr(step_dict["heartbeat_at"], 'isoformat') else str(step_dict["heartbeat_at"])

        # Detect stuck steps: in_progress but no heartbeat for > threshold
        step_dict["is_stuck"] = False
        if step_dict["status"] == "in_progress":
            seconds_since = step_dict.get("seconds_since_heartbeat")
            if seconds_since is not None and float(seconds_since) > HEARTBEAT_TIMEOUT_SECONDS:
                step_dict["is_stuck"] = True
                step_dict["stuck_reason"] = f"No heartbeat for {int(seconds_since)} seconds (worker may have crashed)"
            elif step_dict.get("heartbeat_at") is None and step_dict.get("started_at"):
                # No heartbeat ever recorded but step started - likely old data before heartbeat feature
                step_dict["is_stuck"] = True
                step_dict["stuck_reason"] = "Step started but no heartbeat recorded (worker may have crashed)"

        processed_steps.append(step_dict)

    return jsonify({
        "session": dict(session),
        "steps": processed_steps,
    })


@app.route('/api/workflow/<session_id>/partial-artifacts')
def workflow_partial_artifacts(session_id):
    """Return latest pass1_skeletons, pass2_claims, and category_selections for progressive rendering.

    Lightweight endpoint — 3 queries (one per artifact type), each ORDER BY created_at DESC LIMIT 1.
    """
    result = {}
    with ENGINE.begin() as conn:
        for atype, key, meta_key in [
            ("pass1_skeletons", "pass1_skeletons", "pass1_meta"),
            ("pass2_claims", "pass2_claims", "pass2_meta"),
            ("category_selections", "category_selections", None),
        ]:
            row = conn.execute(
                text(
                    "SELECT artifact_content, artifact_metadata "
                    "FROM workflow_artifacts "
                    "WHERE session_id::text = :sid AND artifact_type = :atype "
                    "ORDER BY created_at DESC LIMIT 1"
                ),
                {"sid": session_id, "atype": atype},
            ).mappings().first()

            if row:
                content = row["artifact_content"]
                metadata = row["artifact_metadata"]
                # Parse JSON content
                try:
                    parsed = json.loads(content) if isinstance(content, str) else content
                except (json.JSONDecodeError, TypeError):
                    parsed = content
                # Parse metadata
                if metadata and isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}
                elif not metadata:
                    metadata = {}

                result[key] = parsed
                if meta_key:
                    result[meta_key] = metadata
            else:
                result[key] = None
                if meta_key:
                    result[meta_key] = {}

    return jsonify(result)


@app.route('/api/workflow/<session_id>/pass2-debug-logs')
def workflow_pass2_debug_logs(session_id):
    """Return Pass 2 debug logs for a workflow session.

    Returns detailed debug information for each skeleton processed in Pass 2, including:
    - Rendered prompt sent to the model
    - API request JSON (excluding the large prompt)
    - Raw API response
    - Structured output after parsing
    - Response type (dict/list/error) and whether a list was auto-fixed
    - Lakebase save status
    - Processing time
    """
    with ENGINE.begin() as conn:
        logs = conn.execute(
            text("""
                SELECT
                    log_id, skeleton_index, key_differentiator, category,
                    rendered_prompt, api_request_json, api_response_raw,
                    structured_output, response_type, was_list_fixed,
                    lakebase_saved, lakebase_error, error_message,
                    processing_time_ms, created_at
                FROM pass2_debug_logs
                WHERE session_id::text = :sid
                ORDER BY skeleton_index
            """),
            {"sid": session_id},
        ).mappings().all()

    result = []
    for log in logs:
        d = dict(log)
        d["created_at"] = str(d.get("created_at", ""))
        result.append(d)

    return jsonify({
        "session_id": session_id,
        "count": len(result),
        "logs": result,
    })


@app.route('/api/workflow/<session_id>/pass2-debug-logs/<int:skeleton_index>')
def workflow_pass2_debug_log_single(session_id, skeleton_index):
    """Return a single Pass 2 debug log entry by skeleton index."""
    with ENGINE.begin() as conn:
        log = conn.execute(
            text("""
                SELECT
                    log_id, skeleton_index, key_differentiator, category,
                    rendered_prompt, api_request_json, api_response_raw,
                    structured_output, response_type, was_list_fixed,
                    lakebase_saved, lakebase_error, error_message,
                    processing_time_ms, created_at
                FROM pass2_debug_logs
                WHERE session_id::text = :sid AND skeleton_index = :idx
            """),
            {"sid": session_id, "idx": skeleton_index},
        ).mappings().first()

    if not log:
        return jsonify({"error": "Debug log not found"}), 404

    d = dict(log)
    d["created_at"] = str(d.get("created_at", ""))
    return jsonify(d)


@app.route('/api/workflow/<session_id>/step/<int:step_number>/artifacts')
def workflow_step_artifacts(session_id, step_number):
    with ENGINE.begin() as conn:
        artifacts = conn.execute(
            text(
                "SELECT artifact_id, step_number, artifact_type, artifact_name, "
                "artifact_content, artifact_metadata, created_at "
                "FROM workflow_artifacts "
                "WHERE session_id::text = :sid AND step_number = :step "
                "ORDER BY created_at"
            ),
            {"sid": session_id, "step": step_number},
        ).mappings().all()
    result = []
    for a in artifacts:
        d = dict(a)
        if d.get("artifact_metadata") and isinstance(d["artifact_metadata"], str):
            try:
                d["artifact_metadata"] = json.loads(d["artifact_metadata"])
            except (json.JSONDecodeError, TypeError):
                pass
        d["created_at"] = str(d.get("created_at", ""))
        result.append(d)
    return jsonify({"artifacts": result})


@app.route('/api/workflow/<session_id>/context')
def workflow_context(session_id):
    """Return all context window pills for the session.

    Primary path: read from agent_turns table (has richer metadata).
    Fallback: derive pills from workflow_artifacts (always works).
    """
    # Try agent_turns first (may not exist in older DBs)
    turns = []
    try:
        with ENGINE.begin() as conn:
            turns = conn.execute(
                text(
                    "SELECT turn_id, step_number, turn_type, role, content_type, "
                    "content_preview, token_count, model_name, artifact_id, created_at "
                    "FROM agent_turns WHERE session_id::text = :sid "
                    "ORDER BY step_number, created_at"
                ),
                {"sid": session_id},
            ).mappings().all()
    except Exception:
        logger.debug("agent_turns table not available, using artifacts fallback")

    if turns:
        pills = []
        for t in turns:
            cfg = PILL_CONFIG.get(t["content_type"], {})
            pills.append({
                "turn_id": t["turn_id"],
                "step_number": t["step_number"],
                "turn_type": t["turn_type"],
                "role": t["role"],
                "content_type": t["content_type"],
                "content_preview": t["content_preview"],
                "token_count": t["token_count"],
                "model_name": t["model_name"],
                "artifact_id": t["artifact_id"],
                "created_at": str(t["created_at"]),
                "color_index": cfg.get("color_index", 0),
                "label": cfg.get("label", t["content_type"]),
            })
        return jsonify({"pills": pills, "source": "agent_turns"})

    # Fallback: derive from workflow_artifacts (always available)
    with ENGINE.begin() as conn:
        artifacts = conn.execute(
            text(
                "SELECT artifact_id, step_number, artifact_type, artifact_name, "
                "artifact_content, created_at "
                "FROM workflow_artifacts WHERE session_id::text = :sid "
                "ORDER BY step_number, created_at"
            ),
            {"sid": session_id},
        ).mappings().all()

    pills = []
    for a in artifacts:
        cfg = PILL_CONFIG.get(a["artifact_type"])
        if not cfg:
            continue
        content = a["artifact_content"] or ""
        pills.append({
            "turn_id": None,
            "step_number": a["step_number"],
            "turn_type": cfg["turn_type"],
            "role": cfg["role"],
            "content_type": a["artifact_type"],
            "content_preview": content[:200] if content else "",
            "token_count": count_tokens(content),
            "model_name": None,
            "artifact_id": a["artifact_id"],
            "created_at": str(a["created_at"]),
            "color_index": cfg["color_index"],
            "label": cfg["label"],
        })
    return jsonify({"pills": pills, "source": "artifacts_fallback"})


@app.route('/api/workflow/<session_id>/artifact/<int:artifact_id>')
def workflow_artifact_detail(session_id, artifact_id):
    """Return full artifact content + token count for the context detail modal."""
    with ENGINE.begin() as conn:
        row = conn.execute(
            text(
                "SELECT artifact_id, step_number, artifact_type, artifact_name, "
                "artifact_content, artifact_metadata, created_at "
                "FROM workflow_artifacts "
                "WHERE session_id::text = :sid AND artifact_id = :aid"
            ),
            {"sid": session_id, "aid": artifact_id},
        ).mappings().first()

    if not row:
        return jsonify({"error": "Artifact not found"}), 404

    content = row["artifact_content"] or ""
    meta = row["artifact_metadata"]
    if meta and isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            pass

    cfg = PILL_CONFIG.get(row["artifact_type"], {})

    return jsonify({
        "artifact_id": row["artifact_id"],
        "step_number": row["step_number"],
        "artifact_type": row["artifact_type"],
        "artifact_name": row["artifact_name"],
        "content": content,
        "token_count": count_tokens(content),
        "metadata": meta,
        "created_at": str(row["created_at"]),
        "label": cfg.get("label", row["artifact_type"]),
        "color_index": cfg.get("color_index", 0),
    })


def _generation_steps_started(conn, session_id):
    """Return True if any generation step (4+) has progressed beyond 'pending'."""
    row = conn.execute(
        text(
            "SELECT COUNT(*) FROM workflow_steps "
            "WHERE session_id::text = :sid AND step_number >= 4 "
            "AND status NOT IN ('pending')"
        ),
        {"sid": session_id},
    ).scalar()
    return (row or 0) > 0


@app.route('/api/workflow/<session_id>/step/1/upload', methods=['POST'])
def workflow_step1_upload(session_id):
    """Upload sales pitch files for directive generation."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files['file']
    raw = f.read()
    content = raw.decode('utf-8', errors='replace').replace('\x00', '')
    with ENGINE.begin() as conn:
        if _generation_steps_started(conn, session_id):
            return jsonify({"error": "Cannot modify context steps after generation has started (step 4+)"}), 400
        art_id = conn.execute(
            text(
                "INSERT INTO workflow_artifacts (session_id, step_number, artifact_type, artifact_name, artifact_content) "
                "VALUES (CAST(:sid AS uuid), 1, 'directive_upload', :name, :content) "
                "RETURNING artifact_id"
            ),
            {"sid": session_id, "name": f.filename, "content": content},
        ).scalar()
        conn.execute(
            text(
                "INSERT INTO agent_turns (session_id, step_number, turn_type, role, content_type, content_preview, token_count, artifact_id) "
                "VALUES (CAST(:sid AS uuid), 1, 'user_input', 'user', 'directive_upload', :preview, :tokens, :aid)"
            ),
            {"sid": session_id, "preview": content[:500], "tokens": count_tokens(content), "aid": art_id},
        )
    return jsonify({"success": True, "filename": f.filename})


@app.route('/api/workflow/<session_id>/step/1/generate', methods=['POST'])
def workflow_step1_generate(session_id):
    """Generate a competitive directive using the LLM.

    Takes the uploaded slides content (or default slides file) and sends it
    to the LLM with the directive generation prompt to produce a structured
    competitive directive.
    """
    with ENGINE.connect() as conn:
        if _generation_steps_started(conn, session_id):
            return jsonify({"error": "Cannot modify context steps after generation has started (step 4+)"}), 400

    import threading

    def _run_generate(sid):
        try:
            _update_step_status(sid, 1, "in_progress", progress_message="Loading slides content...")

            with ENGINE.begin() as conn:
                session = conn.execute(
                    text("SELECT competitor_name, product_area, model_name FROM workflow_sessions WHERE session_id::text = :sid"),
                    {"sid": sid},
                ).mappings().first()
            competitor = session["competitor_name"] if session else "Unknown"
            model_name = session["model_name"] if session else "databricks-claude-sonnet-4"

            # 1. Get slides content: uploaded file first, then default file
            slides_content = ""
            with ENGINE.begin() as conn:
                uploaded = conn.execute(
                    text(
                        "SELECT artifact_content FROM workflow_artifacts "
                        "WHERE session_id::text = :sid AND artifact_type = 'directive_upload' "
                        "ORDER BY created_at DESC LIMIT 1"
                    ),
                    {"sid": sid},
                ).scalar()
            if uploaded:
                slides_content = uploaded
                logger.info("Using uploaded slides content (%d chars)", len(slides_content))
            elif os.path.isfile(DEFAULT_DIRECTIVE_PATH):
                with open(DEFAULT_DIRECTIVE_PATH) as f:
                    slides_content = f.read()
                logger.info("Using default slides file: %s (%d chars)", DEFAULT_DIRECTIVE_PATH, len(slides_content))
            else:
                slides_content = f"No slides content available for {competitor}."

            # 2. Load the directive generation prompt (DB-first with fallback)
            _update_step_status(sid, 1, "in_progress", progress_message="Loading directive prompt template...")
            from workflow_runner import load_directive_template
            prompt_text = load_directive_template(engine=ENGINE)
            logger.info("Loaded directive prompt (%d chars)", len(prompt_text))

            # 3. Build the full prompt: directive prompt + slides content
            # Replace SLIDES_EXTRACT.md reference with actual content
            full_prompt = prompt_text.replace("Read SLIDES_EXTRACT.md", "Read the following slides content:")
            full_prompt = full_prompt.replace("{{competitor}}", competitor)
            full_prompt += f"\n\n---\n\n## SLIDES_EXTRACT.md\n\n{slides_content}"

            # 4. Call the LLM
            _update_step_status(sid, 1, "in_progress", progress_message=f"Generating directive with {model_name}...")
            logger.info("Calling LLM for directive generation (%d chars prompt)", len(full_prompt))

            from workflow_runner import get_openai_client
            client = get_openai_client()
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=DEFAULT_LLM_TEMPERATURE,
                max_tokens=4096,
            )
            directive_content = response.choices[0].message.content
            logger.info("LLM generated directive (%d chars)", len(directive_content))

            # 5. Save the generated directive
            with ENGINE.begin() as conn:
                art_id = conn.execute(
                    text(
                        "INSERT INTO workflow_artifacts (session_id, step_number, artifact_type, artifact_name, artifact_content) "
                        "VALUES (CAST(:sid AS uuid), 1, 'directive_generated', :name, :content) "
                        "RETURNING artifact_id"
                    ),
                    {"sid": sid, "name": "generated_directive.md", "content": directive_content},
                ).scalar()
                conn.execute(
                    text(
                        "INSERT INTO agent_turns (session_id, step_number, turn_type, role, content_type, content_preview, token_count, artifact_id) "
                        "VALUES (CAST(:sid AS uuid), 1, 'tool_result', 'system', 'directive_generated', :preview, :tokens, :aid)"
                    ),
                    {"sid": sid, "preview": directive_content[:500], "tokens": count_tokens(directive_content), "aid": art_id},
                )

            _update_step_status(sid, 1, "completed")
            _advance_workflow(sid, 1)
            logger.info("Step 1 directive generation completed for session %s", sid)

        except Exception as e:
            logger.exception("Step 1 directive generation failed: %s", e)
            _update_step_status(sid, 1, "failed", error_message=str(e))

    # Run in background thread so the UI gets an immediate response
    t = threading.Thread(target=_run_generate, args=(session_id,), daemon=True)
    t.start()
    return jsonify({"success": True, "message": "Generating directive..."})


@app.route('/api/workflow/<session_id>/step/1/skip', methods=['POST'])
def workflow_step1_skip(session_id):
    """Skip LLM generation — load the default directive file directly."""
    with ENGINE.connect() as conn:
        if _generation_steps_started(conn, session_id):
            return jsonify({"error": "Cannot modify context steps after generation has started (step 4+)"}), 400
    _update_step_status(session_id, 1, "in_progress", progress_message="Loading default directive...")
    with ENGINE.begin() as conn:
        session = conn.execute(
            text("SELECT competitor_name FROM workflow_sessions WHERE session_id::text = :sid"),
            {"sid": session_id},
        ).mappings().first()
    competitor = session["competitor_name"] if session else "Unknown"

    directive_content = ""
    if os.path.isfile(DEFAULT_DIRECTIVE_PATH):
        with open(DEFAULT_DIRECTIVE_PATH) as f:
            directive_content = f.read()
        logger.info("Loaded default directive from %s (%d chars)", DEFAULT_DIRECTIVE_PATH, len(directive_content))
    else:
        directive_content = f"# Compete Directive: Databricks vs {competitor}\n\nFocus on key technical differentiators across the data platform.\n"

    with ENGINE.begin() as conn:
        art_id = conn.execute(
            text(
                "INSERT INTO workflow_artifacts (session_id, step_number, artifact_type, artifact_name, artifact_content) "
                "VALUES (CAST(:sid AS uuid), 1, 'directive_generated', :name, :content) "
                "RETURNING artifact_id"
            ),
            {"sid": session_id, "name": "default_directive.md", "content": directive_content},
        ).scalar()
        conn.execute(
            text(
                "INSERT INTO agent_turns (session_id, step_number, turn_type, role, content_type, content_preview, token_count, artifact_id) "
                "VALUES (CAST(:sid AS uuid), 1, 'tool_result', 'system', 'directive_generated', :preview, :tokens, :aid)"
            ),
            {"sid": session_id, "preview": directive_content[:500], "tokens": count_tokens(directive_content), "aid": art_id},
        )
    _update_step_status(session_id, 1, "completed")
    _advance_workflow(session_id, 1)
    return jsonify({"success": True})


@app.route('/api/workflow/<session_id>/step/2/upload', methods=['POST'])
def workflow_step2_upload(session_id):
    """Upload old battlecard PDF."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files['file']
    raw = f.read()
    # Decode text files; strip null bytes for PDFs (Postgres text columns reject \x00)
    content = raw.decode('utf-8', errors='replace').replace('\x00', '')
    with ENGINE.begin() as conn:
        if _generation_steps_started(conn, session_id):
            return jsonify({"error": "Cannot modify context steps after generation has started (step 4+)"}), 400
        art_id = conn.execute(
            text(
                "INSERT INTO workflow_artifacts (session_id, step_number, artifact_type, artifact_name, artifact_content) "
                "VALUES (CAST(:sid AS uuid), 2, 'old_battlecard_upload', :name, :content) "
                "RETURNING artifact_id"
            ),
            {"sid": session_id, "name": f.filename, "content": content},
        ).scalar()
        conn.execute(
            text(
                "INSERT INTO agent_turns (session_id, step_number, turn_type, role, content_type, content_preview, token_count, artifact_id) "
                "VALUES (CAST(:sid AS uuid), 2, 'user_input', 'user', 'old_battlecard_upload', :preview, :tokens, :aid)"
            ),
            {"sid": session_id, "preview": content[:500], "tokens": count_tokens(content), "aid": art_id},
        )
    return jsonify({"success": True, "filename": f.filename})


@app.route('/api/workflow/<session_id>/step/2/extract', methods=['POST'])
def workflow_step2_extract(session_id):
    """Load the default old battlecard file as extracted content."""
    with ENGINE.connect() as conn:
        if _generation_steps_started(conn, session_id):
            return jsonify({"error": "Cannot modify context steps after generation has started (step 4+)"}), 400
    _update_step_status(session_id, 2, "in_progress", progress_message="Loading old battlecard...")

    extracted_content = ""
    source_name = "extracted_battlecard.md"
    if os.path.isfile(DEFAULT_OLD_BATTLECARD_PATH):
        with open(DEFAULT_OLD_BATTLECARD_PATH) as f:
            extracted_content = f.read()
        source_name = os.path.basename(DEFAULT_OLD_BATTLECARD_PATH)
        logger.info("Loaded old battlecard from %s (%d chars)", DEFAULT_OLD_BATTLECARD_PATH, len(extracted_content))
    else:
        # Fallback: check uploaded artifacts
        with ENGINE.begin() as conn:
            uploaded = conn.execute(
                text(
                    "SELECT artifact_content FROM workflow_artifacts "
                    "WHERE session_id::text = :sid AND artifact_type = 'old_battlecard_upload' "
                    "ORDER BY created_at DESC LIMIT 1"
                ),
                {"sid": session_id},
            ).scalar()
        if uploaded:
            extracted_content = uploaded
        else:
            extracted_content = "# No Old Battlecard\n\nNo previous battlecard was provided."

    with ENGINE.begin() as conn:
        art_id = conn.execute(
            text(
                "INSERT INTO workflow_artifacts (session_id, step_number, artifact_type, artifact_name, artifact_content) "
                "VALUES (CAST(:sid AS uuid), 2, 'old_battlecard_extracted', :name, :content) "
                "RETURNING artifact_id"
            ),
            {"sid": session_id, "name": source_name, "content": extracted_content},
        ).scalar()
        conn.execute(
            text(
                "INSERT INTO agent_turns (session_id, step_number, turn_type, role, content_type, content_preview, token_count, artifact_id) "
                "VALUES (CAST(:sid AS uuid), 2, 'tool_result', 'system', 'old_battlecard_extracted', :preview, :tokens, :aid)"
            ),
            {"sid": session_id, "preview": extracted_content[:500], "tokens": count_tokens(extracted_content), "aid": art_id},
        )
    _update_step_status(session_id, 2, "completed")
    _advance_workflow(session_id, 2)
    return jsonify({"success": True})


@app.route('/api/workflow/<session_id>/step/2/use-as-is', methods=['POST'])
def workflow_step2_use_as_is(session_id):
    """Use the uploaded text file directly as extracted content (no LLM)."""
    with ENGINE.connect() as conn:
        if _generation_steps_started(conn, session_id):
            return jsonify({"error": "Cannot modify context steps after generation has started (step 4+)"}), 400
    _update_step_status(session_id, 2, "in_progress", progress_message="Using uploaded content as-is...")

    # Get the most recent uploaded file
    with ENGINE.begin() as conn:
        row = conn.execute(
            text(
                "SELECT artifact_name, artifact_content FROM workflow_artifacts "
                "WHERE session_id::text = :sid AND artifact_type = 'old_battlecard_upload' "
                "ORDER BY created_at DESC LIMIT 1"
            ),
            {"sid": session_id},
        ).mappings().first()

    if not row:
        _update_step_status(session_id, 2, "ready")
        return jsonify({"error": "No file uploaded yet"}), 400

    content = row["artifact_content"]
    source_name = row["artifact_name"]

    with ENGINE.begin() as conn:
        art_id = conn.execute(
            text(
                "INSERT INTO workflow_artifacts (session_id, step_number, artifact_type, artifact_name, artifact_content) "
                "VALUES (CAST(:sid AS uuid), 2, 'old_battlecard_extracted', :name, :content) "
                "RETURNING artifact_id"
            ),
            {"sid": session_id, "name": source_name, "content": content},
        ).scalar()
        conn.execute(
            text(
                "INSERT INTO agent_turns (session_id, step_number, turn_type, role, content_type, content_preview, token_count, artifact_id) "
                "VALUES (CAST(:sid AS uuid), 2, 'tool_result', 'system', 'old_battlecard_extracted', :preview, :tokens, :aid)"
            ),
            {"sid": session_id, "preview": content[:500], "tokens": count_tokens(content), "aid": art_id},
        )
    _update_step_status(session_id, 2, "completed")
    _advance_workflow(session_id, 2)
    logger.info("Step 2 use-as-is completed for session %s (%s, %d chars)", session_id, source_name, len(content))
    return jsonify({"success": True})


@app.route('/api/workflow/<session_id>/step/<int:step_number>/delete-upload', methods=['POST'])
def workflow_step_delete_upload(session_id, step_number):
    """Delete an uploaded file artifact by name for step 1 or 2."""
    if step_number not in (1, 2):
        return jsonify({"error": "Delete upload only supported for steps 1 and 2"}), 400

    artifact_type = {1: 'directive_upload', 2: 'old_battlecard_upload'}[step_number]

    data = request.json or {}
    filename = data.get('filename', '').strip()
    if not filename:
        return jsonify({"error": "filename is required"}), 400

    with ENGINE.begin() as conn:
        # Block modifications once generation steps (4+) have started
        if _generation_steps_started(conn, session_id):
            return jsonify({"error": "Cannot modify context steps after generation has started (step 4+)"}), 400

        # Verify step is in a state that allows editing
        step_status = conn.execute(
            text(
                "SELECT status FROM workflow_steps "
                "WHERE session_id::text = :sid AND step_number = :step"
            ),
            {"sid": session_id, "step": step_number},
        ).scalar()
        if step_status not in ('ready', 'pending'):
            return jsonify({"error": f"Cannot delete uploads when step is '{step_status}'"}), 400

        # Delete the artifact and any linked agent_turns
        art_id = conn.execute(
            text(
                "SELECT artifact_id FROM workflow_artifacts "
                "WHERE session_id::text = :sid AND step_number = :step "
                "AND artifact_type = :atype AND artifact_name = :name "
                "ORDER BY created_at DESC LIMIT 1"
            ),
            {"sid": session_id, "step": step_number, "atype": artifact_type, "name": filename},
        ).scalar()

        if not art_id:
            return jsonify({"error": "File not found"}), 404

        conn.execute(
            text("DELETE FROM agent_turns WHERE artifact_id = :aid"),
            {"aid": art_id},
        )
        conn.execute(
            text("DELETE FROM workflow_artifacts WHERE artifact_id = :aid"),
            {"aid": art_id},
        )

    return jsonify({"success": True, "deleted": filename})


@app.route('/api/workflow/<session_id>/step/3/save', methods=['POST'])
def workflow_step3_save(session_id):
    """Save category classification selections (core / cross-platform / skip).

    Expects JSON: {"selections": {"<catalog_id>": "<inclusion_type>", ...}}
    where inclusion_type is one of: core_product_category, cross_platform_capability, skip
    """
    with ENGINE.connect() as conn:
        if _generation_steps_started(conn, session_id):
            return jsonify({"error": "Cannot modify context steps after generation has started (step 4+)"}), 400
    data = request.json or {}
    selections_dict = data.get('selections', {})

    if not selections_dict:
        return jsonify({"error": "No selections provided"}), 400

    # Validate at least one non-skip selection
    active = {k: v for k, v in selections_dict.items() if v != 'skip'}
    if not active:
        return jsonify({"error": "At least one category must be included (not skipped)"}), 400

    _update_step_status(session_id, 3, "in_progress", progress_message="Saving category selections...")

    with ENGINE.begin() as conn:
        # Load category names from catalog
        catalog_rows = conn.execute(
            text("SELECT catalog_id, category_name FROM product_category_catalog ORDER BY display_order")
        ).mappings().all()
        catalog_lookup = {str(r["catalog_id"]): r["category_name"] for r in catalog_rows}

        # Clear existing selections for this session
        conn.execute(
            text("DELETE FROM session_category_selections WHERE session_id::text = :sid"),
            {"sid": session_id},
        )

        # Insert new selections
        for idx, (cat_id_str, inclusion_type) in enumerate(selections_dict.items()):
            conn.execute(
                text(
                    "INSERT INTO session_category_selections (session_id, catalog_id, inclusion_type, display_order) "
                    "VALUES (CAST(:sid AS uuid), :cid, :itype, :ord)"
                ),
                {
                    "sid": session_id,
                    "cid": int(cat_id_str),
                    "itype": inclusion_type,
                    "ord": idx,
                },
            )

        # Build lists by classification
        core_names = [catalog_lookup.get(k, k) for k, v in selections_dict.items() if v == 'core_product_category']
        cross_names = [catalog_lookup.get(k, k) for k, v in selections_dict.items() if v == 'cross_platform_capability']
        skip_names = [catalog_lookup.get(k, k) for k, v in selections_dict.items() if v == 'skip']

        # Save product_categories artifact (newline-separated non-skipped names for WorkflowRunner._get_categories())
        non_skipped = core_names + cross_names
        categories_content = "\n".join(non_skipped)

        # Remove old product_categories artifacts and turns
        conn.execute(
            text("DELETE FROM agent_turns WHERE session_id::text = :sid AND content_type = 'product_categories'"),
            {"sid": session_id},
        )
        conn.execute(
            text("DELETE FROM workflow_artifacts WHERE session_id::text = :sid AND artifact_type = 'product_categories'"),
            {"sid": session_id},
        )

        cat_art_id = conn.execute(
            text(
                "INSERT INTO workflow_artifacts (session_id, step_number, artifact_type, artifact_name, artifact_content, artifact_metadata) "
                "VALUES (CAST(:sid AS uuid), 3, 'product_categories', 'selected_categories', :content, CAST(:meta AS jsonb)) "
                "RETURNING artifact_id"
            ),
            {
                "sid": session_id,
                "content": categories_content,
                "meta": json.dumps({"categories": non_skipped}),
            },
        ).scalar()
        conn.execute(
            text(
                "INSERT INTO agent_turns (session_id, step_number, turn_type, role, content_type, content_preview, token_count, artifact_id) "
                "VALUES (CAST(:sid AS uuid), 3, 'user_input', 'user', 'product_categories', :preview, :tokens, :aid)"
            ),
            {"sid": session_id, "preview": categories_content[:500], "tokens": count_tokens(categories_content), "aid": cat_art_id},
        )

        # Save category_selections artifact (JSON summary with core/cross/skipped lists)
        summary = {
            "core_product_categories": core_names,
            "cross_platform_capabilities": cross_names,
            "skipped": skip_names,
        }
        artifact_content = json.dumps(summary, indent=2)

        # Remove old category_selections artifacts and turns
        conn.execute(
            text("DELETE FROM agent_turns WHERE session_id::text = :sid AND content_type = 'category_selections'"),
            {"sid": session_id},
        )
        conn.execute(
            text("DELETE FROM workflow_artifacts WHERE session_id::text = :sid AND artifact_type = 'category_selections'"),
            {"sid": session_id},
        )

        art_id = conn.execute(
            text(
                "INSERT INTO workflow_artifacts (session_id, step_number, artifact_type, artifact_name, artifact_content, artifact_metadata) "
                "VALUES (CAST(:sid AS uuid), 3, 'category_selections', 'category_selections.json', :content, CAST(:meta AS jsonb)) "
                "RETURNING artifact_id"
            ),
            {
                "sid": session_id,
                "content": artifact_content,
                "meta": json.dumps({
                    "core_count": len(core_names),
                    "cross_platform_count": len(cross_names),
                    "skipped_count": len(skip_names),
                }),
            },
        ).scalar()

        conn.execute(
            text(
                "INSERT INTO agent_turns (session_id, step_number, turn_type, role, content_type, content_preview, token_count, artifact_id) "
                "VALUES (CAST(:sid AS uuid), 3, 'user_input', 'user', 'category_selections', :preview, :tokens, :aid)"
            ),
            {"sid": session_id, "preview": artifact_content[:500], "tokens": count_tokens(artifact_content), "aid": art_id},
        )

    _update_step_status(session_id, 3, "completed")
    _advance_workflow(session_id, 3)
    return jsonify({"success": True, "summary": summary})


@app.route('/api/workflow/<session_id>/context-sources', methods=['GET'])
def workflow_context_sources(session_id):
    """Return available context sources for a workflow session.

    Each source includes: key, label, available (bool), description, and preview snippet.
    """
    sources = []

    with ENGINE.begin() as conn:
        # Check what artifacts exist for this session
        artifacts = conn.execute(
            text(
                "SELECT artifact_type FROM workflow_artifacts "
                "WHERE session_id::text = :sid "
                "GROUP BY artifact_type"
            ),
            {"sid": session_id},
        ).scalars().all()

        # Get the generation and previous_generation_id
        gen_row = conn.execute(
            text(
                "SELECT ws.generation_id, bg.previous_generation_id "
                "FROM workflow_sessions ws "
                "LEFT JOIN battlecard_generations bg ON ws.generation_id = bg.generation_id "
                "WHERE ws.session_id::text = :sid"
            ),
            {"sid": session_id},
        ).mappings().first()

    artifact_types = set(artifacts) if artifacts else set()
    prev_gen_id = gen_row["previous_generation_id"] if gen_row else None

    # 1. Directive
    has_directive = "directive_generated" in artifact_types
    directive_preview = ""
    if has_directive:
        with ENGINE.begin() as conn:
            dp = conn.execute(
                text(
                    "SELECT LEFT(artifact_content, 200) FROM workflow_artifacts "
                    "WHERE session_id::text = :sid AND artifact_type = 'directive_generated' "
                    "ORDER BY created_at DESC LIMIT 1"
                ),
                {"sid": session_id},
            ).scalar()
            directive_preview = dp or ""
    sources.append({
        "key": "directive",
        "label": "Competitive Directive",
        "available": has_directive,
        "description": "Strategic competitive directive generated in Step 1",
        "preview": directive_preview[:200],
        "default": True,
    })

    # 2. Old Battlecard
    has_old_bc = "old_battlecard_extracted" in artifact_types
    old_bc_preview = ""
    if has_old_bc:
        with ENGINE.begin() as conn:
            bp = conn.execute(
                text(
                    "SELECT LEFT(artifact_content, 200) FROM workflow_artifacts "
                    "WHERE session_id::text = :sid AND artifact_type = 'old_battlecard_extracted' "
                    "ORDER BY created_at DESC LIMIT 1"
                ),
                {"sid": session_id},
            ).scalar()
            old_bc_preview = bp or ""
    sources.append({
        "key": "old_battlecard",
        "label": "Previous Battlecard Content",
        "available": has_old_bc,
        "description": "Extracted content from the uploaded old battlecard (Step 2)",
        "preview": old_bc_preview[:200],
        "default": True,
    })

    # 3. Previous Version Review Feedback
    has_reviews = False
    review_description = "Review feedback from the previous version"
    if prev_gen_id:
        with ENGINE.begin() as conn:
            review_counts = conn.execute(
                text(
                    "SELECT review_type, COUNT(*) AS cnt "
                    "FROM human_reviews WHERE generation_id = :gid "
                    "GROUP BY review_type"
                ),
                {"gid": prev_gen_id},
            ).mappings().all()
        if review_counts:
            has_reviews = True
            counts = {r["review_type"]: r["cnt"] for r in review_counts}
            total = sum(counts.values())
            parts = []
            if counts.get("approve"):
                parts.append(f"{counts['approve']} approved")
            if counts.get("request_edit"):
                parts.append(f"{counts['request_edit']} needs revision")
            if counts.get("reject"):
                parts.append(f"{counts['reject']} rejected")
            review_description = f"Previous version reviews ({total} total: {', '.join(parts)})"

    sources.append({
        "key": "review_feedback",
        "label": "Previous Version Reviews",
        "available": has_reviews,
        "description": review_description,
        "preview": "",
        "default": False,
    })

    # 4. Previous Version Fact-Check Results
    has_fact_checks = False
    fc_description = "Fact-check results from the previous version"
    if prev_gen_id:
        with ENGINE.begin() as conn:
            fc_counts = conn.execute(
                text(
                    "SELECT fc.status, COUNT(*) AS cnt "
                    "FROM fact_checks fc "
                    "JOIN evidence e ON fc.evidence_id = e.evidence_id "
                    "JOIN claims c ON e.claim_id = c.claim_id "
                    "WHERE c.generation_id = :gid "
                    "GROUP BY fc.status"
                ),
                {"gid": prev_gen_id},
            ).mappings().all()
        if fc_counts:
            has_fact_checks = True
            counts = {r["status"]: r["cnt"] for r in fc_counts}
            total = sum(counts.values())
            parts = []
            for status in ("verified", "disputed", "unverified", "outdated"):
                if counts.get(status):
                    parts.append(f"{counts[status]} {status}")
            fc_description = f"Previous version fact-checks ({total} total: {', '.join(parts)})"

    sources.append({
        "key": "fact_checks",
        "label": "Previous Version Fact-Checks",
        "available": has_fact_checks,
        "description": fc_description,
        "preview": "",
        "default": False,
    })

    return jsonify({"sources": sources})


@app.route('/api/workflow/<session_id>/step/4/generate', methods=['POST'])
def workflow_step4_generate(session_id):
    """Trigger Pass 1 - Generate Key Differentiators."""
    data = request.json or {}
    context_sources = data.get('context_sources')
    _update_step_status(session_id, 4, "in_progress", progress_message="Starting Pass 1 generation...")

    def _run():
        try:
            from workflow_runner import WorkflowRunner
            runner = WorkflowRunner(session_id, ENGINE)
            runner.run_pass1(context_sources=context_sources)
        except Exception as e:
            logger.exception("Pass 1 failed for session %s", session_id)
            _update_step_status(session_id, 4, "failed", error_message=str(e))

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({"success": True, "message": "Pass 1 generation started"})


@app.route('/api/workflow/<session_id>/step/4/regenerate', methods=['POST'])
def workflow_step4_regenerate(session_id):
    """Regenerate Key Differentiators from feedback."""
    data = request.json or {}
    feedback_text = data.get('feedback', '')
    context_sources = data.get('context_sources')

    # Optionally prepend battlecard review feedback
    if data.get('include_review_feedback'):
        with ENGINE.begin() as conn:
            gen_id = conn.execute(
                text("SELECT generation_id FROM workflow_sessions WHERE session_id::text = :sid"),
                {"sid": session_id},
            ).scalar()
        if gen_id:
            review_fb = _collect_review_feedback_text(gen_id)
            if review_fb:
                feedback_text = review_fb + "\n\n" + feedback_text

    _update_step_status(session_id, 4, "in_progress", progress_message="Regenerating with feedback...")

    def _run():
        try:
            from workflow_runner import WorkflowRunner
            runner = WorkflowRunner(session_id, ENGINE)
            runner.run_pass1(feedback=feedback_text, context_sources=context_sources)
        except Exception as e:
            logger.exception("Pass 1 regeneration failed for session %s", session_id)
            _update_step_status(session_id, 4, "failed", error_message=str(e))

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({"success": True, "message": "Regeneration started"})


@app.route('/api/workflow/<session_id>/step/5/generate', methods=['POST'])
def workflow_step5_generate(session_id):
    """Trigger Pass 2 - Generate Claims."""
    data = request.json or {}
    context_sources = data.get('context_sources')
    _update_step_status(session_id, 5, "in_progress", progress_message="Starting Pass 2 generation...")

    def _run():
        try:
            from workflow_runner import WorkflowRunner
            runner = WorkflowRunner(session_id, ENGINE)
            runner.run_pass2(context_sources=context_sources)
        except Exception as e:
            logger.exception("Pass 2 failed for session %s", session_id)
            _update_step_status(session_id, 5, "failed", error_message=str(e))

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({"success": True, "message": "Pass 2 generation started"})


@app.route('/api/workflow/<session_id>/step/5/regenerate', methods=['POST'])
def workflow_step5_regenerate(session_id):
    """Trigger Pass 3 - Regenerate claims from feedback."""
    data = request.json or {}
    feedback_text = data.get('feedback', '')
    context_sources = data.get('context_sources')

    # Optionally prepend battlecard review feedback
    if data.get('include_review_feedback'):
        with ENGINE.begin() as conn:
            gen_id = conn.execute(
                text("SELECT generation_id FROM workflow_sessions WHERE session_id::text = :sid"),
                {"sid": session_id},
            ).scalar()
        if gen_id:
            review_fb = _collect_review_feedback_text(gen_id)
            if review_fb:
                feedback_text = review_fb + "\n\n" + feedback_text

    _update_step_status(session_id, 5, "in_progress", progress_message="Starting Pass 3 regeneration...")

    def _run():
        try:
            from workflow_runner import WorkflowRunner
            runner = WorkflowRunner(session_id, ENGINE)
            runner.run_pass3(feedback=feedback_text, context_sources=context_sources)
        except Exception as e:
            logger.exception("Pass 3 failed for session %s", session_id)
            _update_step_status(session_id, 5, "failed", error_message=str(e))

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({"success": True, "message": "Pass 3 regeneration started"})


@app.route('/api/workflow/<session_id>/step/<int:step_number>/approve', methods=['POST'])
def workflow_step_approve(session_id, step_number):
    """Approve a step that is waiting for human review."""
    _update_step_status(session_id, step_number, "completed")
    _advance_workflow(session_id, step_number)
    return jsonify({"success": True})


@app.route('/api/workflow/<session_id>/step/<int:step_number>/reset-stuck', methods=['POST'])
def workflow_step_reset_stuck(session_id, step_number):
    """Reset a stuck step so it can be retried.

    A stuck step is one that's in 'in_progress' status but hasn't received a heartbeat
    for longer than HEARTBEAT_TIMEOUT_SECONDS. This happens when the worker process
    crashes or the server restarts.
    """
    if step_number < 1 or step_number > 7:
        return jsonify({"success": False, "error": "Invalid step number"}), 400

    with ENGINE.begin() as conn:
        # Verify the step is actually stuck (in_progress with old/no heartbeat)
        row = conn.execute(
            text(
                "SELECT status, heartbeat_at, worker_id, "
                "EXTRACT(EPOCH FROM (NOW() - heartbeat_at)) AS seconds_since_heartbeat "
                "FROM workflow_steps WHERE session_id::text = :sid AND step_number = :step"
            ),
            {"sid": session_id, "step": step_number},
        ).mappings().first()

        if not row:
            return jsonify({"success": False, "error": "Step not found"}), 404

        if row["status"] != "in_progress":
            return jsonify({"success": False, "error": f"Step is {row['status']}, not in_progress"}), 400

        # Check if actually stuck (either no heartbeat, or heartbeat too old)
        is_stuck = False
        reason = ""
        if row["heartbeat_at"] is None:
            is_stuck = True
            reason = "No heartbeat recorded"
        elif row["seconds_since_heartbeat"] and float(row["seconds_since_heartbeat"]) > HEARTBEAT_TIMEOUT_SECONDS:
            is_stuck = True
            reason = f"No heartbeat for {int(row['seconds_since_heartbeat'])} seconds"

        if not is_stuck:
            return jsonify({
                "success": False,
                "error": "Step does not appear to be stuck - worker may still be running"
            }), 400

        # Reset the step to 'ready' so it can be re-triggered
        conn.execute(
            text(
                "UPDATE workflow_steps SET status = 'ready', "
                "progress_current = 0, progress_total = 0, "
                "progress_message = :msg, "
                "error_message = NULL, error_details = NULL, "
                "heartbeat_at = NULL, worker_id = NULL "
                "WHERE session_id::text = :sid AND step_number = :step"
            ),
            {
                "sid": session_id,
                "step": step_number,
                "msg": f"Reset from stuck state ({reason}). Ready to retry.",
            },
        )

    return jsonify({
        "success": True,
        "message": f"Step {step_number} reset from stuck state. Click 'Run' to retry."
    })


@app.route('/api/workflow/<session_id>/step/<int:step_number>/stop', methods=['POST'])
def workflow_step_stop(session_id, step_number):
    """Force stop an in-progress step.

    This resets the step to 'ready' status so it can be restarted.
    Note: This doesn't actually kill the running thread (Python limitation),
    but it prevents the worker from saving results and allows a restart.
    """
    if step_number < 1 or step_number > 7:
        return jsonify({"success": False, "error": "Invalid step number"}), 400

    with ENGINE.begin() as conn:
        row = conn.execute(
            text("SELECT status FROM workflow_steps WHERE session_id::text = :sid AND step_number = :step"),
            {"sid": session_id, "step": step_number},
        ).mappings().first()

        if not row:
            return jsonify({"success": False, "error": "Step not found"}), 404

        if row["status"] not in ("in_progress", "failed"):
            return jsonify({"success": False, "error": f"Step is {row['status']}, can only stop in_progress or failed steps"}), 400

        # Reset the step to 'ready' so it can be re-triggered
        conn.execute(
            text(
                "UPDATE workflow_steps SET status = 'ready', "
                "started_at = NULL, completed_at = NULL, "
                "progress_current = 0, progress_total = 0, "
                "progress_message = 'Stopped by user. Ready to restart.', "
                "error_message = NULL, error_details = NULL, "
                "heartbeat_at = NULL, worker_id = NULL, "
                "last_error = NULL, error_count = 0 "
                "WHERE session_id::text = :sid AND step_number = :step"
            ),
            {"sid": session_id, "step": step_number},
        )

        # Update session current_step to the previous step
        conn.execute(
            text("UPDATE workflow_sessions SET current_step = :prev, updated_at = NOW() WHERE session_id::text = :sid"),
            {"sid": session_id, "prev": step_number - 1},
        )

    return jsonify({
        "success": True,
        "message": f"Step {step_number} stopped. Click the generate button to restart."
    })


@app.route('/api/workflow/<session_id>/step/<int:step_number>/reopen', methods=['POST'])
def workflow_step_reopen(session_id, step_number):
    """Reopen a completed step for editing. Steps 1-3 go back to 'ready', steps 4-6 go to 'waiting_human'."""
    if step_number < 1 or step_number > 7:
        return jsonify({"success": False, "error": "Invalid step number"}), 400

    # Block reopening context steps (1-3) once generation steps (4+) have started
    if step_number <= 3:
        with ENGINE.connect() as conn:
            if _generation_steps_started(conn, session_id):
                return jsonify({"success": False, "error": "Cannot reopen context steps after generation has started (step 4+)"}), 400

    # Steps 1-3 are input/config steps → reopen to 'ready'
    # Steps 4-6 are generation/review steps → reopen to 'waiting_human'
    target_status = "ready" if step_number <= 3 else "waiting_human"

    with ENGINE.begin() as conn:
        # Verify the step is actually completed
        row = conn.execute(
            text("SELECT status FROM workflow_steps WHERE session_id::text = :sid AND step_number = :step"),
            {"sid": session_id, "step": step_number},
        ).fetchone()
        if not row:
            return jsonify({"success": False, "error": "Step not found"}), 404
        if row.status != "completed":
            return jsonify({"success": False, "error": f"Step is {row.status}, not completed"}), 400

        # Reopen the step
        conn.execute(
            text("UPDATE workflow_steps SET status = :status, completed_at = NULL WHERE session_id::text = :sid AND step_number = :step"),
            {"sid": session_id, "step": step_number, "status": target_status},
        )
        # Reset any later steps that are 'ready' back to 'pending'
        conn.execute(
            text("UPDATE workflow_steps SET status = 'pending' WHERE session_id::text = :sid AND step_number > :step AND status = 'ready'"),
            {"sid": session_id, "step": step_number},
        )
        # Update current step pointer
        conn.execute(
            text("UPDATE workflow_sessions SET current_step = :step, updated_at = NOW() WHERE session_id::text = :sid"),
            {"sid": session_id, "step": step_number},
        )

    return jsonify({"success": True, "message": f"Step {step_number} reopened for editing"})


@app.route('/api/workflow/<session_id>/step/6/generate', methods=['POST'])
def workflow_step6_fact_check(session_id):
    """Trigger fact checking in a background thread."""
    _update_step_status(
        session_id,
        6,
        "in_progress",
        progress_current=0,
        progress_total=4,
        progress_message="[1/4] Starting fact checks...",
    )

    # Fast-fail preflight so users get immediate feedback instead of a silent stall.
    if not os.getenv("EXA_API_KEY"):
        err = "Fact check unavailable: EXA_API_KEY is not configured for this app deployment."
        _update_step_status(
            session_id,
            6,
            "waiting_human",
            progress_current=0,
            progress_total=0,
            progress_message="Fact check could not start.",
            error_message=err,
        )
        return jsonify({"success": False, "error": err}), 400

    def _run():
        try:
            from exa_fact_checker import ExaFactChecker
            checker = ExaFactChecker(session_id, ENGINE)
            checker.run_fact_checks()
        except Exception as e:
            logger.exception("Fact check failed for session %s", session_id)
            _update_step_status(
                session_id,
                6,
                "waiting_human",
                progress_message="Fact check stopped due to an error.",
                error_message=f"Fact check error: {e}",
            )

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({"success": True, "message": "Fact checking started"})


@app.route('/api/workflow/<session_id>/step/6/approve', methods=['POST'])
def workflow_step6_approve(session_id):
    """Approve fact check results and advance to Step 7."""
    _update_step_status(session_id, 6, "completed")
    _advance_workflow(session_id, 6)
    return jsonify({"success": True})


@app.route('/api/workflow/<session_id>/step/7/generate', methods=['POST'])
def workflow_step7_generate(session_id):
    """Generate Google Slides (stub)."""
    _update_step_status(session_id, 7, "in_progress", progress_message="Generating slides...")

    # Stub implementation
    with ENGINE.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO workflow_artifacts (session_id, step_number, artifact_type, artifact_name, artifact_content) "
                "VALUES (CAST(:sid AS uuid), 7, 'google_slides_url', 'slides_link', 'https://docs.google.com/presentation/d/placeholder')"
            ),
            {"sid": session_id},
        )

    _update_step_status(session_id, 7, "completed")
    with ENGINE.begin() as conn:
        conn.execute(
            text("UPDATE workflow_sessions SET status = 'completed', updated_at = NOW() WHERE session_id::text = :sid"),
            {"sid": session_id},
        )
    return jsonify({"success": True})


def _update_step_status(session_id, step_number, status, progress_current=None, progress_total=None, progress_message=None, error_message=None):
    """Update a workflow step's status."""
    with ENGINE.begin() as conn:
        sets = ["status = :status"]
        params = {"sid": session_id, "step": step_number, "status": status}

        if status == "in_progress":
            sets.append("started_at = COALESCE(started_at, NOW())")
            sets.append("heartbeat_at = NOW()")
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
            {"sid": session_id, "step": step_number},
        )


def _advance_workflow(session_id, completed_step):
    """Mark next step as ready after completing a step.

    If subsequent steps are already completed (e.g. after editing an earlier step),
    skip over them and advance the first pending step in the chain.
    """
    with ENGINE.begin() as conn:
        # Find all steps after the completed one, ordered
        rows = conn.execute(
            text(
                "SELECT step_number, status FROM workflow_steps "
                "WHERE session_id::text = :sid AND step_number > :step "
                "ORDER BY step_number"
            ),
            {"sid": session_id, "step": completed_step},
        ).fetchall()

        for row in rows:
            if row.status == "completed":
                # Already completed — skip and keep looking
                continue
            if row.status == "pending":
                # First pending step — mark it ready
                conn.execute(
                    text(
                        "UPDATE workflow_steps SET status = 'ready' "
                        "WHERE session_id::text = :sid AND step_number = :step AND status = 'pending'"
                    ),
                    {"sid": session_id, "step": row.step_number},
                )
            # Stop at the first non-completed step (whether we set it to ready or it's in another state)
            break


# =============================================================================
# Prompt preview API
# =============================================================================


@app.route('/api/workflow/<session_id>/step/<int:step_number>/prompt', methods=['GET', 'POST'])
def workflow_step_prompt_preview(session_id, step_number):
    """Return the rendered prompt that will be (or was) used for a given step."""
    from workflow_runner import (
        DEFAULT_PASS1_PROMPT, DEFAULT_PASS2_PROMPT,
        load_prompt_template, render_template as render_prompt,
        format_context_xml, load_pass1_template, load_pass2_template,
        load_directive_template,
        get_pass2_request_spec,
        resolve_context_source_flags,
    )

    # Optional context source overrides for prompt preview (used by Step 4/5 UI).
    context_sources = None
    payload = request.get_json(silent=True) or {}
    if isinstance(payload.get("context_sources"), dict):
        context_sources = payload["context_sources"]
    elif request.args.get("context_sources"):
        try:
            parsed = json.loads(request.args.get("context_sources") or "{}")
            if isinstance(parsed, dict):
                context_sources = parsed
        except (json.JSONDecodeError, TypeError):
            context_sources = None
    context_flags = resolve_context_source_flags(context_sources)

    # Load session config
    with ENGINE.begin() as conn:
        session = conn.execute(
            text(
                "SELECT competitor_name, product_area, model_name, diffs_per_category, "
                "max_workers, "
                "COALESCE(step4_execution_mode, 'auto') AS step4_execution_mode, "
                "COALESCE(step5_execution_mode, 'auto') AS step5_execution_mode, "
                "COALESCE(step5_inline_fact_check, FALSE) AS step5_inline_fact_check, "
                "COALESCE(pass1_prompt_template_version, 2) AS pass1_prompt_template_version, "
                "COALESCE(pass2_prompt_template_version, 2) AS pass2_prompt_template_version "
                "FROM workflow_sessions WHERE session_id::text = :sid"
            ),
            {"sid": session_id},
        ).mappings().first()
        if not session:
            return jsonify({"error": "Session not found"}), 404

    # Load artifacts
    def _get_artifact(atype):
        with ENGINE.begin() as conn:
            return conn.execute(
                text(
                    "SELECT artifact_content FROM workflow_artifacts "
                    "WHERE session_id::text = :sid AND artifact_type = :atype "
                    "ORDER BY created_at DESC LIMIT 1"
                ),
                {"sid": session_id, "atype": atype},
            ).scalar()

    competitor = session["competitor_name"]
    product_area = session["product_area"]
    model_name = session["model_name"]
    diffs_per_category = session["diffs_per_category"]

    def _build_preview_context():
        directive_text = (_get_artifact("directive_generated") or "") if context_flags["directive"] else ""
        old_battlecard_text = (_get_artifact("old_battlecard_extracted") or "") if context_flags["old_battlecard"] else ""
        review_feedback = ""
        fact_check_results = ""

        if context_flags["review_feedback"] or context_flags["fact_checks"]:
            with ENGINE.begin() as conn:
                prev_gen_id = conn.execute(
                    text(
                        "SELECT bg.previous_generation_id "
                        "FROM workflow_sessions ws "
                        "JOIN battlecard_generations bg ON ws.generation_id = bg.generation_id "
                        "WHERE ws.session_id::text = :sid"
                    ),
                    {"sid": session_id},
                ).scalar()
            if prev_gen_id:
                if context_flags["review_feedback"]:
                    review_feedback = _collect_review_feedback_text(prev_gen_id)
                if context_flags["fact_checks"]:
                    fc_rows = get_fact_check_details_by_gen(prev_gen_id)
                    if fc_rows:
                        lines = ["## Previous Version Fact-Check Results", ""]
                        for r in fc_rows:
                            verdict = str((r.get("verdict") or "unknown")).upper()
                            headline = str(r.get("claim_headline") or "")
                            traced = str(r.get("traces_to_text") or r.get("claim_description") or "")
                            reasoning = str(r.get("reasoning") or "")
                            confidence = r.get("confidence_score")
                            line = f"- [{verdict}] {headline}"
                            if traced:
                                line += f' — claim text: "{traced[:200]}"'
                            if reasoning:
                                line += f" — reason: {reasoning[:200]}"
                            if confidence is not None:
                                line += f" (confidence: {confidence}%)"
                            lines.append(line)
                        fact_check_results = "\n".join(lines)

        return format_context_xml(
            directive_text,
            old_battlecard_text,
            competitor,
            review_feedback=review_feedback,
            fact_check_results=fact_check_results,
        )

    if step_number == 1:
        # Step 1: Show the directive generation prompt (DB-first)
        directive_prompt = load_directive_template(engine=ENGINE)

        # Get uploaded slides or default
        slides_content = _get_artifact("directive_upload") or ""
        if not slides_content and os.path.isfile(DEFAULT_DIRECTIVE_PATH):
            with open(DEFAULT_DIRECTIVE_PATH) as f:
                slides_content = f.read()

        slides_preview = slides_content[:500] + ("..." if len(slides_content) > 500 else "") if slides_content else "[No slides uploaded yet]"

        full_prompt = directive_prompt.replace("Read SLIDES_EXTRACT.md", "Read the following slides content:")
        full_prompt = full_prompt.replace("{{competitor}}", competitor)
        full_prompt += f"\n\n---\n\n## SLIDES_EXTRACT.md\n\n{slides_content}"

        variables = {
            "competitor": competitor,
            "slides_content": slides_preview,
        }

        return jsonify({
            "step": 1,
            "title": "Generate Directive (LLM)",
            "description": "Sends uploaded slides + directive prompt to LLM to generate a competitive directive.",
            "template_file": DEFAULT_DIRECTIVE_PROMPT,
            "model": model_name,
            "template": directive_prompt,
            "variables": variables,
            "prompt": full_prompt,
        })

    elif step_number == 2:
        # Step 2: Show what old battlecard file will be loaded
        return jsonify({
            "step": 2,
            "title": "Upload Old Battlecards",
            "description": "This step loads the previous battlecard archive from the default file.",
            "source_file": DEFAULT_OLD_BATTLECARD_PATH,
            "prompt": f"[No LLM prompt — loads old battlecard file directly]\n\nSource: {DEFAULT_OLD_BATTLECARD_PATH}",
        })

    elif step_number == 3:
        # Step 3: Classify Product Categories — no LLM prompt
        return jsonify({
            "step": 3,
            "title": "Classify Product Categories",
            "description": "This step lets the user classify each product category as a core product category (gets its own slide), cross-platform capability, or skip.",
            "prompt": "[No LLM prompt — human classification of categories]",
        })

    elif step_number == 4:
        # Step 4: Pass 1 prompt
        p1_ver = session["pass1_prompt_template_version"]

        directive = (_get_artifact("directive_generated") or "[Directive not yet generated]") if context_flags["directive"] else ""
        context = _build_preview_context()

        categories_content = _get_artifact("product_categories") or ""
        categories = [c.strip() for c in categories_content.split("\n") if c.strip()]

        # Load category classifications for V3+
        classifications_content = _get_artifact("category_selections")
        classifications = {}
        if classifications_content:
            try:
                classifications = json.loads(classifications_content)
            except (json.JSONDecodeError, TypeError):
                pass
        core_cats = classifications.get("core_product_categories", [])
        cross_cats = classifications.get("cross_platform_capabilities", [])

        # Match runtime behavior: V3 with multiple core categories and multiple
        # workers executes using the V4 per-category prompt.
        step4_exec = _resolve_step4_execution(
            p1_ver,
            session.get("step4_execution_mode"),
            session.get("max_workers", 1),
            len(core_cats),
        )
        use_parallel_pass1 = step4_exec["runtime_mode"] in ("category_parallel", "category_sequential")
        effective_p1_ver = 4 if use_parallel_pass1 else p1_ver
        try:
            template_text, template_file = load_pass1_template(effective_p1_ver, engine=ENGINE)
        except FileNotFoundError:
            return jsonify({"error": f"Pass 1 prompt template V{effective_p1_ver} not found"}), 404

        if p1_ver >= 3 and core_cats:
            # V3+: separate core and cross-platform category lists
            core_text = "\n".join(f"- {c}" for c in core_cats)
            cross_text = "\n".join(f"- {c}" for c in cross_cats) if cross_cats else "_(none selected)_"
            total_diffs = len(core_cats) * diffs_per_category
            target_category = core_cats[0] if core_cats else "[Category]"
            other_core = "\n".join(f"- {c}" for c in core_cats[1:]) if len(core_cats) > 1 else "_(none)_"
            variables = {
                "competitor": competitor,
                "product_area": product_area,
                "comparison": f"Databricks vs {competitor}",
                # Keep both names populated for template compatibility.
                "product_categories": core_text,
                "core_product_categories": core_text,
                "target_category": target_category,
                "other_core_categories": other_core,
                "cross_platform_capabilities": cross_text,
                "core_category_count": str(len(core_cats)),
                "diffs_per_category": str(diffs_per_category),
                "total_diffs": str(total_diffs),
                "directives": directive,
                "context": context,
            }
        else:
            # V1/V2: flat list of all categories
            categories_text = "\n".join(f"- {c}" for c in categories) if categories else "[No categories selected yet]"
            total_diffs = len(categories) * diffs_per_category
            variables = {
                "competitor": competitor,
                "product_area": product_area,
                "comparison": f"Databricks vs {competitor}",
                "product_categories": categories_text,
                "diffs_per_category": str(diffs_per_category),
                "total_diffs": str(total_diffs),
                "directives": directive,
                "context": context,
            }

        rendered = render_prompt(template_text, **variables)

        return jsonify({
            "step": 4,
            "title": f"Pass 1: Generate Key Differentiators (V{effective_p1_ver})",
            "template_file": template_file,
            "model": model_name,
            "template": template_text,
            "variables": variables,
            "prompt": rendered,
            "selected_prompt_version": p1_ver,
            "effective_prompt_version": effective_p1_ver,
            "note": f"Preview uses runtime execution mode `{step4_exec['runtime_mode']}`.",
            "context_sources": context_flags,
        })

    elif step_number == 5:
        # Step 5: Pass 2 prompt (show for the first skeleton as example)
        p2_ver = session["pass2_prompt_template_version"]
        try:
            template_text = load_pass2_template(p2_ver, engine=ENGINE)
        except FileNotFoundError:
            return jsonify({"error": f"Prompt template V{p2_ver} not found"}), 404

        directive = (_get_artifact("directive_generated") or "[Directive not yet generated]") if context_flags["directive"] else ""
        context = _build_preview_context()
        directives_for_template = "" if p2_ver in (5, 6, 7, 8, 9, 10) else directive

        # Try to get first skeleton for a realistic preview
        skeletons_json = _get_artifact("pass1_skeletons")
        if skeletons_json:
            import json as _json
            skeletons = _json.loads(skeletons_json)
            sk = skeletons[0] if skeletons else {}
        else:
            sk = {
                "category": "[Category]",
                "key_differentiator": "[Key Differentiator]",
                "description": "[Description]",
                "databricks_rating": "[Rating]",
                "competitor_rating": "[Rating]",
                "selection_reasoning": "[Reasoning]",
            }
            skeletons = [sk]

        step5_exec = _resolve_step5_execution(
            p2_ver,
            session.get("step5_execution_mode"),
            session.get("max_workers", 1),
            len({s.get("category", "") for s in skeletons if s.get("category")}),
            session.get("diffs_per_category", 0),
        )
        if step5_exec["runtime_mode"] == "single_call":
            category_diffs = skeletons
        else:
            category_diffs = [s for s in skeletons if s.get("category") == sk.get("category")] if skeletons else [sk]

        variables = {
            "competitor": competitor,
            "category": sk.get("category", ""),
            "key_differentiator": sk.get("key_differentiator", ""),
            "description": sk.get("description", ""),
            "databricks_rating": sk.get("databricks_rating", ""),
            "competitor_rating": sk.get("competitor_rating", ""),
            "selection_reasoning": sk.get("selection_reasoning", ""),
            "num_diffs": len(category_diffs),
            "key_diffs_json": json.dumps(
                [
                    {
                        "id": s.get("id", ""),
                        "category": s.get("category", ""),
                        "key_differentiator": s.get("key_differentiator", ""),
                        "description": s.get("description", ""),
                        "databricks_rating": s.get("databricks_rating", ""),
                        "competitor_rating": s.get("competitor_rating", ""),
                        "selection_reasoning": s.get("selection_reasoning", ""),
                    }
                    for s in category_diffs
                ],
                indent=2,
            ),
            "directives": directives_for_template,
            "context": context,
        }

        rendered = render_prompt(template_text, **variables)
        ver_label = PASS2_PROMPT_TEMPLATES.get(p2_ver, {}).get("label", f"V{p2_ver}")

        return jsonify({
            "step": 5,
            "title": f"Pass 2: Generate Claims — {ver_label}",
            "template_file": f"pass2_template_v{p2_ver}",
            "model": model_name,
            "template": template_text,
            "variables": variables,
            "prompt": rendered,
            "prompt_version": p2_ver,
            "request_preview": get_pass2_request_spec(p2_ver).get("request_preview"),
            "response_schema_json": get_pass2_request_spec(p2_ver).get("schema"),
            "note": (
                f"Preview uses runtime execution mode `{step5_exec['runtime_mode']}`. "
                f"Inline fact-check seeding is {'enabled' if session.get('step5_inline_fact_check') else 'disabled'}."
            ),
            "context_sources": context_flags,
        })

    elif step_number == 6:
        return jsonify({
            "step": 6,
            "title": "Fact Check Claims",
            "description": "Uses Exa web search to verify each claim, then an LLM judge to evaluate verdicts.",
            "prompt": "[Exa search + LLM judge — queries constructed from claim text]",
        })

    elif step_number == 7:
        return jsonify({
            "step": 7,
            "title": "Generate Google Slides",
            "description": "This step generates Google Slides from the approved claims.",
            "prompt": "[Slides generation — not yet implemented with LLM]",
        })

    return jsonify({"error": "Unknown step"}), 400


# =============================================================================
# Evidence matching helper (from sqlite app)
# =============================================================================


def match_fact_checks_to_citations(content, fact_checks):
    citations_obj = content.get("citations", {}) or {}
    merged = {}
    cite_by_id = {}

    for field_name, field_citations in citations_obj.items():
        if not isinstance(field_citations, list):
            continue
        merged[field_name] = []
        for cite in field_citations:
            entry = {**cite, "fact_check": None}
            merged[field_name].append(entry)
            cid = cite.get("citation_id")
            if cid:
                cite_by_id[cid] = entry

    orphan_fact_checks = []
    matched_fc_ids = set()

    for fc in fact_checks or []:
        fc_id = fc.get("fact_check_id")
        if fc_id in matched_fc_ids:
            continue

        matched = False
        claim_field = fc.get("claim_field")
        claim_start = fc.get("claim_start_index")
        claim_end = fc.get("claim_end_index")
        if claim_field and claim_start is not None and claim_end is not None:
            field_cites = merged.get(claim_field, [])
            for cite_entry in field_cites:
                cs = cite_entry.get("start_index", -1)
                ce = cite_entry.get("end_index", -1)
                if cs < claim_end and ce > claim_start:
                    cite_entry["fact_check"] = fc
                    matched_fc_ids.add(fc_id)
                    matched = True
                    break

        if not matched:
            orphan_fact_checks.append(fc)

    return {"merged": merged, "orphan_fact_checks": orphan_fact_checks}


# =============================================================================
# Admin: Product Category & Mappings Management
# =============================================================================


@app.route('/admin/categories')
def admin_categories():
    """Admin page for managing product categories and mappings."""
    with ENGINE.begin() as conn:
        categories = [
            dict(r) for r in conn.execute(
                text(
                    "SELECT catalog_id, category_name, category_description, "
                    "is_core_product_category, is_cross_platform_capability, display_order "
                    "FROM product_category_catalog ORDER BY display_order"
                )
            ).mappings().all()
        ]
        mappings = [
            dict(r) for r in conn.execute(
                text(
                    "SELECT mapping_id, catalog_id, vendor, competitor_name, "
                    "product_name, product_description, display_order "
                    "FROM product_mappings ORDER BY catalog_id, vendor, display_order"
                )
            ).mappings().all()
        ]
    return render_template(
        'admin_categories.html',
        categories=categories,
        mappings=mappings,
    )


@app.route('/api/admin/categories', methods=['POST'])
def create_category():
    """Create a new product category."""
    data = request.json
    name = data.get('name', '').strip()
    description = data.get('description', '').strip()
    is_core = bool(data.get('is_core', False))
    is_cross_platform = bool(data.get('is_cross_platform', False))

    if not name:
        return jsonify({"error": "Category name is required"}), 400

    with ENGINE.begin() as conn:
        max_order = conn.execute(
            text("SELECT COALESCE(MAX(display_order), -1) FROM product_category_catalog")
        ).scalar()
        catalog_id = conn.execute(
            text(
                "INSERT INTO product_category_catalog "
                "(category_name, category_description, is_core_product_category, "
                "is_cross_platform_capability, display_order) "
                "VALUES (:name, :desc, :is_core, :is_cross, :ord) "
                "RETURNING catalog_id"
            ),
            {
                "name": name,
                "desc": description,
                "is_core": is_core,
                "is_cross": is_cross_platform,
                "ord": max_order + 1,
            },
        ).scalar()

    return jsonify({"success": True, "catalog_id": catalog_id})


@app.route('/api/admin/categories/<int:catalog_id>', methods=['PUT'])
def update_category(catalog_id):
    """Update an existing product category."""
    data = request.json
    name = data.get('name', '').strip()
    description = data.get('description', '').strip()
    is_core = bool(data.get('is_core', False))
    is_cross_platform = bool(data.get('is_cross_platform', False))

    if not name:
        return jsonify({"error": "Category name is required"}), 400

    with ENGINE.begin() as conn:
        conn.execute(
            text(
                "UPDATE product_category_catalog SET "
                "category_name = :name, category_description = :desc, "
                "is_core_product_category = :is_core, "
                "is_cross_platform_capability = :is_cross "
                "WHERE catalog_id = :cid"
            ),
            {
                "cid": catalog_id,
                "name": name,
                "desc": description,
                "is_core": is_core,
                "is_cross": is_cross_platform,
            },
        )

    return jsonify({"success": True})


@app.route('/api/admin/categories/<int:catalog_id>', methods=['DELETE'])
def delete_category(catalog_id):
    """Delete a product category and its related data."""
    with ENGINE.begin() as conn:
        # Delete related session_category_selections first
        conn.execute(
            text("DELETE FROM session_category_selections WHERE catalog_id = :cid"),
            {"cid": catalog_id},
        )
        # Delete related product_mappings
        conn.execute(
            text("DELETE FROM product_mappings WHERE catalog_id = :cid"),
            {"cid": catalog_id},
        )
        # Delete the category itself
        conn.execute(
            text("DELETE FROM product_category_catalog WHERE catalog_id = :cid"),
            {"cid": catalog_id},
        )

    return jsonify({"success": True})


@app.route('/api/admin/categories/reorder', methods=['POST'])
def reorder_categories():
    """Reorder product categories."""
    data = request.json
    order = data.get('order', [])

    if not order:
        return jsonify({"error": "Order list is required"}), 400

    with ENGINE.begin() as conn:
        for idx, catalog_id in enumerate(order):
            conn.execute(
                text(
                    "UPDATE product_category_catalog SET display_order = :ord "
                    "WHERE catalog_id = :cid"
                ),
                {"ord": idx, "cid": catalog_id},
            )

    return jsonify({"success": True})


@app.route('/api/admin/mappings', methods=['POST'])
def create_mapping():
    """Create a new product mapping."""
    data = request.json
    catalog_id = data.get('catalog_id')
    vendor = data.get('vendor', '').strip()
    competitor_name = data.get('competitor_name', '').strip() or None
    product_name = data.get('product_name', '').strip()
    product_description = data.get('product_description', '').strip()

    if not catalog_id or not vendor or not product_name:
        return jsonify({"error": "catalog_id, vendor, and product_name are required"}), 400

    if vendor not in ('databricks', 'competitor'):
        return jsonify({"error": "vendor must be 'databricks' or 'competitor'"}), 400

    if vendor == 'competitor' and not competitor_name:
        return jsonify({"error": "competitor_name is required for competitor mappings"}), 400

    if vendor == 'databricks':
        competitor_name = None

    with ENGINE.begin() as conn:
        max_order = conn.execute(
            text(
                "SELECT COALESCE(MAX(display_order), -1) FROM product_mappings "
                "WHERE catalog_id = :cid AND vendor = :v"
            ),
            {"cid": catalog_id, "v": vendor},
        ).scalar()
        mapping_id = conn.execute(
            text(
                "INSERT INTO product_mappings "
                "(catalog_id, vendor, competitor_name, product_name, product_description, display_order) "
                "VALUES (:cid, :vendor, :comp, :pname, :pdesc, :ord) "
                "RETURNING mapping_id"
            ),
            {
                "cid": catalog_id,
                "vendor": vendor,
                "comp": competitor_name,
                "pname": product_name,
                "pdesc": product_description,
                "ord": max_order + 1,
            },
        ).scalar()

    return jsonify({"success": True, "mapping_id": mapping_id})


@app.route('/api/admin/mappings/<int:mapping_id>', methods=['PUT'])
def update_mapping(mapping_id):
    """Update an existing product mapping."""
    data = request.json
    product_name = data.get('product_name', '').strip()
    product_description = data.get('product_description', '').strip()

    if not product_name:
        return jsonify({"error": "product_name is required"}), 400

    with ENGINE.begin() as conn:
        conn.execute(
            text(
                "UPDATE product_mappings SET "
                "product_name = :pname, product_description = :pdesc, "
                "updated_at = CURRENT_TIMESTAMP "
                "WHERE mapping_id = :mid"
            ),
            {
                "mid": mapping_id,
                "pname": product_name,
                "pdesc": product_description,
            },
        )

    return jsonify({"success": True})


@app.route('/api/admin/mappings/<int:mapping_id>', methods=['DELETE'])
def delete_mapping(mapping_id):
    """Delete a product mapping."""
    with ENGINE.begin() as conn:
        conn.execute(
            text("DELETE FROM product_mappings WHERE mapping_id = :mid"),
            {"mid": mapping_id},
        )

    return jsonify({"success": True})


# =============================================================================
# Prompt Template Admin
# =============================================================================


@app.route('/admin/prompts')
def admin_prompts():
    """Admin page for managing prompt templates."""
    with ENGINE.begin() as conn:
        templates = [
            dict(r) for r in conn.execute(
                text(
                    "SELECT template_id, template_name, template_type, version_label, "
                    "description, LENGTH(template_text) AS char_count, variables, "
                    "response_schema_json, request_config_json, "
                    "is_active, is_default, display_order, created_at, updated_at "
                    "FROM prompt_templates ORDER BY template_type, display_order"
                )
            ).mappings().all()
        ]
    # Compute var_count from variables JSON
    for t in templates:
        try:
            t["var_count"] = len(json.loads(t["variables"])) if t.get("variables") else 0
        except (json.JSONDecodeError, TypeError):
            t["var_count"] = 0

    with ENGINE.begin() as conn:
        # Fetch recent workflow sessions for test panel
        sessions = [
            dict(r) for r in conn.execute(
                text(
                    "SELECT session_id::text AS session_id, competitor_name, product_area, "
                    "status, created_at "
                    "FROM workflow_sessions ORDER BY created_at DESC LIMIT 20"
                )
            ).mappings().all()
        ]
    return render_template(
        'admin_prompts.html',
        templates=templates,
        sessions=sessions,
    )


@app.route('/api/admin/prompts')
def api_list_prompts():
    """List all prompt templates."""
    with ENGINE.begin() as conn:
        templates = [
            dict(r) for r in conn.execute(
                text(
                    "SELECT template_id, template_name, template_type, version_label, "
                    "description, LENGTH(template_text) AS char_count, variables, "
                    "response_schema_json, request_config_json, "
                    "is_active, is_default, display_order, created_at, updated_at "
                    "FROM prompt_templates ORDER BY template_type, display_order"
                )
            ).mappings().all()
        ]
    # Serialize datetimes
    for t in templates:
        for k in ("created_at", "updated_at"):
            if t.get(k):
                t[k] = str(t[k])
    return jsonify({"templates": templates})


@app.route('/api/admin/prompts/<int:template_id>')
def api_get_prompt(template_id):
    """Get a single prompt template with full text."""
    with ENGINE.begin() as conn:
        row = conn.execute(
            text(
                "SELECT template_id, template_name, template_type, version_label, "
                "description, template_text, variables, response_schema_json, request_config_json, is_active, is_default, "
                "display_order, created_at, updated_at "
                "FROM prompt_templates WHERE template_id = :tid"
            ),
            {"tid": template_id},
        ).mappings().first()
    if not row:
        return jsonify({"error": "Template not found"}), 404
    result = dict(row)
    for k in ("created_at", "updated_at"):
        if result.get(k):
            result[k] = str(result[k])
    return jsonify(result)


@app.route('/api/admin/prompts', methods=['POST'])
def api_create_prompt():
    """Create a new prompt template."""
    import re
    data = request.json
    name = data.get('template_name', '').strip()
    ttype = data.get('template_type', '').strip()
    template_text = data.get('template_text', '')

    if not name or not ttype or not template_text:
        return jsonify({"error": "template_name, template_type, and template_text are required"}), 400

    if ttype not in ('pass1', 'pass2', 'directive'):
        return jsonify({"error": "template_type must be 'pass1', 'pass2', or 'directive'"}), 400

    # Auto-extract variables
    variables = sorted(set(re.findall(r'\{\{(\w+)\}\}', template_text)))
    schema_payload = data.get("response_schema_json")
    request_cfg_payload = data.get("request_config_json")
    if isinstance(schema_payload, str):
        try:
            schema_payload = json.loads(schema_payload)
        except json.JSONDecodeError:
            schema_payload = None
    if isinstance(request_cfg_payload, str):
        try:
            request_cfg_payload = json.loads(request_cfg_payload)
        except json.JSONDecodeError:
            request_cfg_payload = None

    with ENGINE.begin() as conn:
        # Determine next display_order for this type
        max_order = conn.execute(
            text("SELECT COALESCE(MAX(display_order), 0) FROM prompt_templates WHERE template_type = :ttype"),
            {"ttype": ttype},
        ).scalar()

        is_default = bool(data.get('is_default', False))
        if is_default:
            conn.execute(
                text("UPDATE prompt_templates SET is_default = FALSE WHERE template_type = :ttype"),
                {"ttype": ttype},
            )

        tid = conn.execute(
            text(
                "INSERT INTO prompt_templates "
                "(template_name, template_type, version_label, description, template_text, "
                "variables, response_schema_json, request_config_json, is_active, is_default, display_order) "
                "VALUES (:name, :ttype, :label, :desc, :text, :vars, CAST(:schema AS jsonb), CAST(:request_cfg AS jsonb), :active, :default, :order) "
                "RETURNING template_id"
            ),
            {
                "name": name,
                "ttype": ttype,
                "label": data.get('version_label', ''),
                "desc": data.get('description', ''),
                "text": template_text,
                "vars": json.dumps(variables),
                "schema": json.dumps(schema_payload) if schema_payload is not None else None,
                "request_cfg": json.dumps(request_cfg_payload) if request_cfg_payload is not None else None,
                "active": bool(data.get('is_active', True)),
                "default": is_default,
                "order": int(max_order) + 1,
            },
        ).scalar()

    return jsonify({"success": True, "template_id": tid})


@app.route('/api/admin/prompts/<int:template_id>', methods=['PUT'])
def api_update_prompt(template_id):
    """Update an existing prompt template."""
    import re
    data = request.json
    schema_payload = data.get("response_schema_json") if isinstance(data, dict) else None
    request_cfg_payload = data.get("request_config_json") if isinstance(data, dict) else None
    if isinstance(schema_payload, str):
        try:
            schema_payload = json.loads(schema_payload)
        except json.JSONDecodeError:
            schema_payload = None
    if isinstance(request_cfg_payload, str):
        try:
            request_cfg_payload = json.loads(request_cfg_payload)
        except json.JSONDecodeError:
            request_cfg_payload = None

    with ENGINE.begin() as conn:
        existing = conn.execute(
            text("SELECT template_type FROM prompt_templates WHERE template_id = :tid"),
            {"tid": template_id},
        ).mappings().first()
        if not existing:
            return jsonify({"error": "Template not found"}), 404

        ttype = existing["template_type"]

        sets = ["updated_at = NOW()"]
        params = {"tid": template_id}

        if "template_name" in data:
            sets.append("template_name = :name")
            params["name"] = data["template_name"].strip()
        if "version_label" in data:
            sets.append("version_label = :label")
            params["label"] = data["version_label"]
        if "description" in data:
            sets.append("description = :desc")
            params["desc"] = data["description"]
        if "template_text" in data:
            sets.append("template_text = :text")
            params["text"] = data["template_text"]
            # Auto-update variables
            variables = sorted(set(re.findall(r'\{\{(\w+)\}\}', data["template_text"])))
            sets.append("variables = :vars")
            params["vars"] = json.dumps(variables)
        if "is_active" in data:
            sets.append("is_active = :active")
            params["active"] = bool(data["is_active"])
        if "response_schema_json" in data:
            sets.append("response_schema_json = CAST(:schema AS jsonb)")
            params["schema"] = json.dumps(schema_payload) if schema_payload is not None else None
        if "request_config_json" in data:
            sets.append("request_config_json = CAST(:request_cfg AS jsonb)")
            params["request_cfg"] = json.dumps(request_cfg_payload) if request_cfg_payload is not None else None
        if "display_order" in data:
            sets.append("display_order = :order")
            params["order"] = int(data["display_order"])
        if "is_default" in data and data["is_default"]:
            # Enforce exclusivity: only one default per type
            conn.execute(
                text("UPDATE prompt_templates SET is_default = FALSE WHERE template_type = :ttype"),
                {"ttype": ttype},
            )
            sets.append("is_default = TRUE")
        elif "is_default" in data:
            sets.append("is_default = FALSE")

        conn.execute(
            text(f"UPDATE prompt_templates SET {', '.join(sets)} WHERE template_id = :tid"),
            params,
        )

    return jsonify({"success": True})


@app.route('/api/admin/prompts/<int:template_id>', methods=['DELETE'])
def api_delete_prompt(template_id):
    """Delete a prompt template."""
    with ENGINE.begin() as conn:
        conn.execute(
            text("DELETE FROM prompt_templates WHERE template_id = :tid"),
            {"tid": template_id},
        )
    return jsonify({"success": True})


@app.route('/api/admin/prompts/<int:template_id>/test', methods=['POST'])
def api_test_prompt(template_id):
    """Render a prompt template with real session data for testing."""
    from workflow_runner import (
        render_template as render_prompt,
        format_context_xml,
        get_pass2_request_spec,
    )

    data = request.json or {}
    session_id = data.get('session_id')

    with ENGINE.begin() as conn:
        tpl = conn.execute(
            text(
                "SELECT template_name, template_text, template_type, variables, display_order, "
                "response_schema_json, request_config_json "
                "FROM prompt_templates WHERE template_id = :tid"
            ),
            {"tid": template_id},
        ).mappings().first()
        if not tpl:
            return jsonify({"error": "Template not found"}), 404

    template_name = tpl["template_name"]
    template_text = tpl["template_text"]
    template_type = tpl["template_type"]
    variables_json = tpl["variables"]
    variables_list = json.loads(variables_json) if variables_json else []
    stored_schema = tpl.get("response_schema_json")
    stored_request_cfg = tpl.get("request_config_json")

    # Build variables from session data (if session_id provided)
    variables = {}
    if session_id:
        with ENGINE.begin() as conn:
            session = conn.execute(
                text(
                    "SELECT competitor_name, product_area, model_name, diffs_per_category "
                    "FROM workflow_sessions WHERE session_id::text = :sid"
                ),
                {"sid": session_id},
            ).mappings().first()

        if session:
            competitor = session["competitor_name"]
            product_area = session["product_area"]
            variables["competitor"] = competitor
            variables["product_area"] = product_area
            variables["comparison"] = f"Databricks vs {competitor}"
            variables["diffs_per_category"] = str(session["diffs_per_category"])

            # Load artifacts
            def _get_art(atype):
                with ENGINE.begin() as conn:
                    return conn.execute(
                        text(
                            "SELECT artifact_content FROM workflow_artifacts "
                            "WHERE session_id::text = :sid AND artifact_type = :atype "
                            "ORDER BY created_at DESC LIMIT 1"
                        ),
                        {"sid": session_id, "atype": atype},
                    ).scalar()

            directive = _get_art("directive_generated") or ""
            old_battlecard = _get_art("old_battlecard_extracted") or ""
            context = format_context_xml(directive, old_battlecard, competitor)

            variables["directives"] = directive
            variables["context"] = context

            if template_type == "pass1":
                cats_content = _get_art("product_categories") or ""
                categories = [c.strip() for c in cats_content.split("\n") if c.strip()]
                variables["product_categories"] = "\n".join(f"- {c}" for c in categories)
                variables["total_diffs"] = str(len(categories) * session["diffs_per_category"])
                # V3+ fields
                classifications_content = _get_art("category_selections")
                if classifications_content:
                    try:
                        cls_data = json.loads(classifications_content)
                        core_cats = cls_data.get("core_product_categories", [])
                        cross_cats = cls_data.get("cross_platform_capabilities", [])
                        variables["core_product_categories"] = "\n".join(f"- {c}" for c in core_cats)
                        variables["cross_platform_capabilities"] = "\n".join(f"- {c}" for c in cross_cats) if cross_cats else "_(none)_"
                        variables["core_category_count"] = str(len(core_cats))
                        variables["total_diffs"] = str(len(core_cats) * session["diffs_per_category"])
                        # V4 per-category fields
                        if core_cats:
                            variables["target_category"] = core_cats[0]
                            variables["other_core_categories"] = "\n".join(f"- {c}" for c in core_cats[1:]) if len(core_cats) > 1 else "_(none)_"
                    except (json.JSONDecodeError, TypeError):
                        pass

            elif template_type == "pass2":
                skeletons_json = _get_art("pass1_skeletons")
                if skeletons_json:
                    try:
                        skeletons = json.loads(skeletons_json)
                        if skeletons:
                            first = skeletons[0]
                            variables["category"] = first.get("category", "")
                            variables["key_differentiator"] = first.get("key_differentiator", "")
                            variables["description"] = first.get("description", "")
                            variables["databricks_rating"] = first.get("databricks_rating", "")
                            variables["competitor_rating"] = first.get("competitor_rating", "")
                            variables["selection_reasoning"] = first.get("selection_reasoning", "")
                            category = first.get("category", "")
                            category_diffs = [s for s in skeletons if s.get("category") == category]
                            variables["num_diffs"] = len(category_diffs)
                            variables["key_diffs_json"] = json.dumps(
                                [
                                    {
                                        "id": s.get("id", ""),
                                        "category": s.get("category", ""),
                                        "key_differentiator": s.get("key_differentiator", ""),
                                        "description": s.get("description", ""),
                                        "databricks_rating": s.get("databricks_rating", ""),
                                        "competitor_rating": s.get("competitor_rating", ""),
                                        "selection_reasoning": s.get("selection_reasoning", ""),
                                    }
                                    for s in category_diffs
                                ],
                                indent=2,
                            )
                    except (json.JSONDecodeError, TypeError):
                        pass

                # V5/V6 templates already get directive content inside {{context}}.
                if "key_diffs_json" in variables_list:
                    variables["directives"] = ""

    # Fill in any missing variables with placeholders
    for var in variables_list:
        if var not in variables:
            variables[var] = f"[{var}]"

    rendered = render_prompt(template_text, **variables)

    # Token count estimate
    token_count = 0
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        token_count = len(enc.encode(rendered))
    except Exception:
        token_count = len(rendered) // 4

    response_schema_json = stored_schema
    request_config_json = stored_request_cfg
    request_preview = None
    if template_type == "pass2":
        pass2_ver = None
        if template_name and template_name.startswith("pass2_v"):
            try:
                pass2_ver = int(template_name.split("pass2_v", 1)[1])
            except (TypeError, ValueError):
                pass
        if pass2_ver is None:
            try:
                pass2_ver = int(tpl.get("display_order"))
            except (TypeError, ValueError):
                pass2_ver = None
        if pass2_ver is not None:
            req_spec = get_pass2_request_spec(pass2_ver)
            if response_schema_json is None:
                response_schema_json = req_spec.get("schema")
            request_preview = req_spec.get("request_preview")
            if request_config_json is None:
                request_config_json = {
                    "structured_output": bool(req_spec.get("use_structured_output")),
                    "response_format_type": req_spec.get("response_format_type"),
                }

    return jsonify({
        "rendered_prompt": rendered,
        "variables": variables,
        "token_count": token_count,
        "char_count": len(rendered),
        "response_schema_json": response_schema_json,
        "request_config_json": request_config_json,
        "request_preview": request_preview,
    })


# =============================================================================
# Eval / Benchmark
# =============================================================================


def _read_text_file(path: str) -> str:
    try:
        if path and os.path.isfile(path):
            with open(path) as f:
                return f.read()
    except Exception:
        pass
    return ""


def _select_core_categories(limit_count: int) -> list[str]:
    with ENGINE.begin() as conn:
        rows = conn.execute(
            text(
                "SELECT category_name FROM product_category_catalog "
                "WHERE is_core_product_category = TRUE "
                "ORDER BY display_order LIMIT :lim"
            ),
            {"lim": max(1, int(limit_count or 1))},
        ).scalars().all()
    return [r for r in rows if r]


def _create_workflow_session_internal(
    competitor_name: str,
    product_area: str,
    model_name: str,
    diffs_per_category: int,
    max_workers: int,
    pass1_prompt_template_version: int,
    pass2_prompt_template_version: int,
    step4_execution_mode: str,
    step5_execution_mode: str,
    step5_inline_fact_check: bool,
) -> tuple[str, int]:
    with ENGINE.begin() as conn:
        gen_id = conn.execute(
            text(
                "INSERT INTO battlecard_generations "
                "(trigger_type, generated_by, generation_model, status) "
                "VALUES ('manual_request', 'eval_runner', :model, 'draft') "
                "RETURNING generation_id"
            ),
            {"model": model_name},
        ).scalar()

        session_id = conn.execute(
            text(
                "INSERT INTO workflow_sessions ("
                "competitor_name, product_area, model_name, diffs_per_category, max_workers, "
                "step4_execution_mode, step5_execution_mode, step5_inline_fact_check, "
                "generation_id, pass1_prompt_template_version, pass2_prompt_template_version"
                ") VALUES ("
                ":comp, :area, :model, :diffs, :workers, "
                ":s4mode, :s5mode, :s5fc, "
                ":gid, :p1v, :p2v"
                ") RETURNING session_id::text"
            ),
            {
                "comp": competitor_name,
                "area": product_area,
                "model": model_name,
                "diffs": int(diffs_per_category),
                "workers": int(max_workers),
                "s4mode": step4_execution_mode,
                "s5mode": step5_execution_mode,
                "s5fc": bool(step5_inline_fact_check),
                "gid": gen_id,
                "p1v": int(pass1_prompt_template_version),
                "p2v": int(pass2_prompt_template_version),
            },
        ).scalar()

        for step_num, step_name in WORKFLOW_STEPS:
            status = "ready" if step_num == 1 else "pending"
            conn.execute(
                text(
                    "INSERT INTO workflow_steps (session_id, step_number, step_name, status) "
                    "VALUES (CAST(:sid AS uuid), :num, :name, :status)"
                ),
                {"sid": session_id, "num": step_num, "name": step_name, "status": status},
            )
    return session_id, gen_id


def _seed_context_for_benchmark_session(session_id: str, core_categories: list[str]):
    directive_content = _read_text_file(DEFAULT_DIRECTIVE_PATH)
    old_battlecard_content = _read_text_file(DEFAULT_OLD_BATTLECARD_PATH)
    product_categories_content = "\n".join(core_categories)
    selections = {
        "core_product_categories": core_categories,
        "cross_platform_capabilities": [],
        "skipped": [],
    }

    with ENGINE.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO workflow_artifacts (session_id, step_number, artifact_type, artifact_name, artifact_content, artifact_metadata) "
                "VALUES (CAST(:sid AS uuid), 1, 'directive_generated', 'directive.md', :content, CAST(:meta AS jsonb))"
            ),
            {"sid": session_id, "content": directive_content, "meta": json.dumps({"seeded": True})},
        )
        conn.execute(
            text(
                "INSERT INTO workflow_artifacts (session_id, step_number, artifact_type, artifact_name, artifact_content, artifact_metadata) "
                "VALUES (CAST(:sid AS uuid), 2, 'old_battlecard_extracted', 'old_battlecard.md', :content, CAST(:meta AS jsonb))"
            ),
            {"sid": session_id, "content": old_battlecard_content, "meta": json.dumps({"seeded": True})},
        )
        conn.execute(
            text(
                "INSERT INTO workflow_artifacts (session_id, step_number, artifact_type, artifact_name, artifact_content, artifact_metadata) "
                "VALUES (CAST(:sid AS uuid), 3, 'product_categories', 'categories.md', :content, CAST(:meta AS jsonb))"
            ),
            {"sid": session_id, "content": product_categories_content, "meta": json.dumps({"seeded": True})},
        )
        conn.execute(
            text(
                "INSERT INTO workflow_artifacts (session_id, step_number, artifact_type, artifact_name, artifact_content, artifact_metadata) "
                "VALUES (CAST(:sid AS uuid), 3, 'category_selections', 'category_selections.json', :content, CAST(:meta AS jsonb))"
            ),
            {"sid": session_id, "content": json.dumps(selections, indent=2), "meta": json.dumps({"seeded": True})},
        )

        # Keep relational selections in sync with the seeded artifact so Step 4
        # can resolve canonical catalog IDs (instead of string-only fallback).
        conn.execute(
            text("DELETE FROM session_category_selections WHERE session_id::text = :sid"),
            {"sid": session_id},
        )
        catalog_rows = conn.execute(
            text("SELECT catalog_id, category_name FROM product_category_catalog ORDER BY display_order")
        ).mappings().all()
        catalog_by_name = {r["category_name"]: r["catalog_id"] for r in catalog_rows}

        missing_core_categories = []
        display_order = 0
        for category_name in core_categories:
            catalog_id = catalog_by_name.get(category_name)
            if catalog_id is None:
                missing_core_categories.append(category_name)
                continue
            conn.execute(
                text(
                    "INSERT INTO session_category_selections (session_id, catalog_id, inclusion_type, display_order) "
                    "VALUES (CAST(:sid AS uuid), :cid, 'core_product_category', :ord)"
                ),
                {"sid": session_id, "cid": int(catalog_id), "ord": display_order},
            )
            display_order += 1

        if missing_core_categories:
            logger.warning(
                "Benchmark seed skipped %d core categories missing from product_category_catalog (session %s): %s",
                len(missing_core_categories),
                session_id,
                ", ".join(missing_core_categories),
            )

        conn.execute(
            text(
                "UPDATE workflow_steps SET status = 'completed', completed_at = NOW(), "
                "progress_current = 1, progress_total = 1, progress_message = 'Seeded by eval runner' "
                "WHERE session_id::text = :sid AND step_number IN (1,2,3)"
            ),
            {"sid": session_id},
        )
        conn.execute(
            text(
                "UPDATE workflow_steps SET status = 'ready', progress_message = 'Ready for benchmark generation' "
                "WHERE session_id::text = :sid AND step_number = 4"
            ),
            {"sid": session_id},
        )
        conn.execute(
            text("UPDATE workflow_sessions SET current_step = 4, updated_at = NOW() WHERE session_id::text = :sid"),
            {"sid": session_id},
        )


def _load_generation_eval_items(generation_id: int) -> list[dict]:
    with ENGINE.begin() as conn:
        rows = conn.execute(
            text(
                "SELECT cl.claim_id, co.company_type, kd.key_diff_name, pcc.category_name, cl.rating, cl.headline, "
                "COALESCE(string_agg(di.item_text, ' ' ORDER BY di.item_order), cl.description) AS details_text "
                "FROM claims cl "
                "JOIN companies co ON cl.company_id = co.company_id "
                "JOIN key_differentiators kd ON cl.key_diff_id = kd.key_diff_id "
                "JOIN product_category_catalog pcc ON kd.category_id = pcc.catalog_id "
                "LEFT JOIN claim_detail_items di ON di.claim_id = cl.claim_id "
                "WHERE cl.generation_id = :gid "
                "GROUP BY cl.claim_id, co.company_type, kd.key_diff_name, kd.display_order, pcc.category_name, cl.rating, cl.headline, cl.description "
                "ORDER BY pcc.category_name, kd.display_order, co.company_type"
            ),
            {"gid": generation_id},
        ).mappings().all()

        verdict_rows = conn.execute(
            text(
                "SELECT e.claim_id, fc.status, COUNT(*) AS cnt "
                "FROM fact_checks fc "
                "JOIN evidence e ON fc.evidence_id = e.evidence_id "
                "JOIN claims cl ON e.claim_id = cl.claim_id "
                "WHERE cl.generation_id = :gid "
                "GROUP BY e.claim_id, fc.status"
            ),
            {"gid": generation_id},
        ).mappings().all()

    verdict_by_claim = {}
    for row in verdict_rows:
        cid = row["claim_id"]
        candidate = {"status": row["status"], "cnt": int(row["cnt"] or 0)}
        current = verdict_by_claim.get(cid)
        if current is None or candidate["cnt"] > current["cnt"]:
            verdict_by_claim[cid] = candidate

    items = []
    for row in rows:
        cid = row["claim_id"]
        items.append(
            {
                "claim_id": cid,
                "company_type": row["company_type"] or "",
                "key_diff_name": row["key_diff_name"] or "",
                "category_name": row["category_name"] or "",
                "rating": row["rating"] or "",
                "headline": row["headline"] or "",
                "details_text": row["details_text"] or "",
                "verdict_label": (verdict_by_claim.get(cid) or {}).get("status"),
                "source_count": int((verdict_by_claim.get(cid) or {}).get("cnt") or 0),
            }
        )
    return items


def _upsert_eval_dataset(dataset_name: str, generation_id: int, source_battlecard_uuid: str = "", notes: str = "") -> dict:
    items = _load_generation_eval_items(generation_id)
    with ENGINE.begin() as conn:
        dataset_id = conn.execute(
            text(
                "INSERT INTO eval_datasets (dataset_name, source_generation_id, source_battlecard_uuid, notes, created_by) "
                "VALUES (:name, :gid, CAST(NULLIF(:bid, '') AS uuid), :notes, 'user') "
                "ON CONFLICT (dataset_name) DO UPDATE SET "
                "source_generation_id = EXCLUDED.source_generation_id, "
                "source_battlecard_uuid = EXCLUDED.source_battlecard_uuid, "
                "notes = EXCLUDED.notes, "
                "created_at = CURRENT_TIMESTAMP "
                "RETURNING dataset_id"
            ),
            {"name": dataset_name, "gid": generation_id, "bid": source_battlecard_uuid or "", "notes": notes},
        ).scalar()

        conn.execute(text("DELETE FROM eval_dataset_items WHERE dataset_id = :did"), {"did": dataset_id})
        for item in items:
            conn.execute(
                text(
                    "INSERT INTO eval_dataset_items ("
                    "dataset_id, key_diff_name, category_name, company_type, rating, headline, details_text, verdict_label, source_count"
                    ") VALUES ("
                    ":did, :kd, :cat, :ctype, :rating, :headline, :details, :verdict, :sc"
                    ")"
                ),
                {
                    "did": dataset_id,
                    "kd": item["key_diff_name"],
                    "cat": item["category_name"],
                    "ctype": item["company_type"],
                    "rating": item["rating"],
                    "headline": item["headline"],
                    "details": item["details_text"],
                    "verdict": item["verdict_label"],
                    "sc": item["source_count"],
                },
            )
    return {"dataset_id": dataset_id, "dataset_name": dataset_name, "item_count": len(items)}


def _session_token_usage(session_id: str) -> dict:
    with ENGINE.begin() as conn:
        row = conn.execute(
            text(
                "SELECT "
                "COALESCE(SUM(token_count), 0) AS total_tokens, "
                "COALESCE(SUM(CASE WHEN turn_type IN ('system_prompt', 'user_input') THEN token_count ELSE 0 END), 0) AS input_tokens, "
                "COALESCE(SUM(CASE WHEN turn_type = 'model_output' THEN token_count ELSE 0 END), 0) AS output_tokens "
                "FROM agent_turns WHERE session_id::text = :sid AND step_number IN (4,5,6)"
            ),
            {"sid": session_id},
        ).mappings().first()
    return {
        "total_tokens": int((row or {}).get("total_tokens") or 0),
        "input_tokens": int((row or {}).get("input_tokens") or 0),
        "output_tokens": int((row or {}).get("output_tokens") or 0),
    }


def _score_generation_against_dataset(generation_id: int, dataset_id: int) -> dict:
    with ENGINE.begin() as conn:
        refs = conn.execute(
            text(
                "SELECT key_diff_name, category_name, company_type, rating, headline, details_text, verdict_label "
                "FROM eval_dataset_items WHERE dataset_id = :did"
            ),
            {"did": dataset_id},
        ).mappings().all()

    if not refs:
        return {"claim_match_rate": None, "rating_match_rate": None, "verdict_match_rate": None}

    generated = _load_generation_eval_items(generation_id)
    grouped = {}
    for idx, item in enumerate(generated):
        key = (item["category_name"], item["company_type"])
        grouped.setdefault(key, []).append((idx, item))

    used = set()
    claim_scores, rating_scores, verdict_scores = [], [], []

    for ref in refs:
        key = (ref["category_name"], ref["company_type"])
        candidates = [(idx, g) for idx, g in grouped.get(key, []) if idx not in used]
        if not candidates:
            continue
        best_idx, best = max(
            candidates,
            key=lambda pair: SequenceMatcher(
                None,
                (ref["key_diff_name"] or "").lower(),
                (pair[1]["key_diff_name"] or "").lower(),
            ).ratio(),
        )
        used.add(best_idx)
        head_sim = SequenceMatcher(None, (ref["headline"] or "").lower(), (best["headline"] or "").lower()).ratio()
        details_sim = SequenceMatcher(None, (ref["details_text"] or "").lower(), (best["details_text"] or "").lower()).ratio()
        claim_scores.append((head_sim + details_sim) / 2.0)
        rating_scores.append(1.0 if (ref["rating"] or "") == (best["rating"] or "") else 0.0)
        if ref.get("verdict_label") and best.get("verdict_label"):
            verdict_scores.append(1.0 if ref["verdict_label"] == best["verdict_label"] else 0.0)

    def _avg(vals):
        return float(sum(vals) / len(vals)) if vals else None

    return {
        "claim_match_rate": _avg(claim_scores),
        "rating_match_rate": _avg(rating_scores),
        "verdict_match_rate": _avg(verdict_scores),
    }


def _default_benchmark_techniques():
    return [
        {
            "name": "v3_per_diff_parallel",
            "pass1_prompt_template_version": 4,
            "pass2_prompt_template_version": 3,
            "step4_execution_mode": "category_parallel",
            "step5_execution_mode": "per_diff_parallel",
            "step5_inline_fact_check": False,
            "max_workers": 4,
        },
        {
            "name": "v6_category_parallel",
            "pass1_prompt_template_version": 4,
            "pass2_prompt_template_version": 6,
            "step4_execution_mode": "category_parallel",
            "step5_execution_mode": "category_parallel",
            "step5_inline_fact_check": False,
            "max_workers": 4,
        },
        {
            "name": "v7_category_parallel",
            "pass1_prompt_template_version": 4,
            "pass2_prompt_template_version": 7,
            "step4_execution_mode": "category_parallel",
            "step5_execution_mode": "category_parallel",
            "step5_inline_fact_check": False,
            "max_workers": 4,
        },
        {
            "name": "v8_single_call_inline_fc",
            "pass1_prompt_template_version": 4,
            "pass2_prompt_template_version": 8,
            "step4_execution_mode": "category_parallel",
            "step5_execution_mode": "single_call",
            "step5_inline_fact_check": True,
            "max_workers": 4,
        },
    ]


def _run_benchmark_matrix(run_id: str, config: dict):
    from workflow_runner import WorkflowRunner

    with ENGINE.begin() as conn:
        conn.execute(
            text("UPDATE eval_runs SET status = 'running', started_at = NOW(), error_message = NULL WHERE run_id::text = :rid"),
            {"rid": run_id},
        )

    dataset_id = config.get("dataset_id")
    competitor_name = config.get("competitor_name", "Microsoft Fabric")
    product_area = config.get("product_area", "Data Platform")
    models = config.get("models") or ["databricks-claude-sonnet-4-5", "databricks-claude-opus-4-6"]
    scenarios = config.get("scenarios") or [
        {"label": "2c_x_2d", "core_categories": 2, "diffs_per_category": 2},
        {"label": "4c_x_5d", "core_categories": 4, "diffs_per_category": 5},
        {"label": "8c_x_10d", "core_categories": 8, "diffs_per_category": 10},
    ]
    techniques = config.get("techniques") or _default_benchmark_techniques()

    try:
        for scenario in scenarios:
            core_count = int(scenario.get("core_categories", 2))
            diffs_per_category = int(scenario.get("diffs_per_category", 2))
            scenario_label = scenario.get("label") or f"{core_count}c_x_{diffs_per_category}d"
            core_categories = _select_core_categories(core_count)
            if len(core_categories) < core_count:
                raise RuntimeError(f"Only found {len(core_categories)} core categories, expected {core_count}")

            for model_name in models:
                for technique in techniques:
                    t0 = time.time()
                    session_id = None
                    gen_id = None
                    generated_diffs = 0
                    tokens = {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}
                    score = {"claim_match_rate": None, "rating_match_rate": None, "verdict_match_rate": None}
                    notes = ""

                    p1v = int(technique.get("pass1_prompt_template_version", 4))
                    p2v = int(technique.get("pass2_prompt_template_version", 6))
                    s4mode = technique.get("step4_execution_mode", "auto")
                    s5mode = technique.get("step5_execution_mode", "auto")
                    s5fc = bool(technique.get("step5_inline_fact_check", False))
                    workers = int(technique.get("max_workers", 4))

                    try:
                        session_id, gen_id = _create_workflow_session_internal(
                            competitor_name=competitor_name,
                            product_area=product_area,
                            model_name=model_name,
                            diffs_per_category=diffs_per_category,
                            max_workers=workers,
                            pass1_prompt_template_version=p1v,
                            pass2_prompt_template_version=p2v,
                            step4_execution_mode=s4mode,
                            step5_execution_mode=s5mode,
                            step5_inline_fact_check=s5fc,
                        )
                        _seed_context_for_benchmark_session(session_id, core_categories)

                        runner = WorkflowRunner(session_id, ENGINE)
                        runner.run_pass1()
                        runner.run_pass2()

                        with ENGINE.begin() as conn:
                            generated_diffs = int(
                                conn.execute(
                                    text("SELECT COUNT(*) FROM key_differentiators WHERE generation_id = :gid"),
                                    {"gid": gen_id},
                                ).scalar() or 0
                            )

                        tokens = _session_token_usage(session_id)
                        if dataset_id:
                            score = _score_generation_against_dataset(gen_id, int(dataset_id))
                    except Exception as combo_err:
                        notes = str(combo_err)
                        logger.exception("Benchmark combo failed")

                    duration = time.time() - t0
                    with ENGINE.begin() as conn:
                        conn.execute(
                            text(
                                "INSERT INTO eval_run_results ("
                                "run_id, session_id, dataset_id, scenario_label, model_name, "
                                "pass1_prompt_template_version, pass2_prompt_template_version, "
                                "step4_execution_mode, step5_execution_mode, step5_inline_fact_check, "
                                "core_categories, diffs_per_category, expected_diffs, generated_diffs, "
                                "duration_seconds, total_tokens, input_tokens, output_tokens, "
                                "eval_claim_match_rate, eval_rating_match_rate, eval_verdict_match_rate, notes"
                                ") VALUES ("
                                "CAST(:rid AS uuid), CAST(NULLIF(:sid, '') AS uuid), :did, :scenario, :model, "
                                ":p1v, :p2v, :s4mode, :s5mode, :s5fc, :core, :diffs, :expected, :generated, "
                                ":dur, :tot, :tin, :tout, :claim, :rating, :verdict, :notes"
                                ")"
                            ),
                            {
                                "rid": run_id,
                                "sid": session_id or "",
                                "did": dataset_id,
                                "scenario": scenario_label,
                                "model": model_name,
                                "p1v": p1v,
                                "p2v": p2v,
                                "s4mode": s4mode,
                                "s5mode": s5mode,
                                "s5fc": s5fc,
                                "core": core_count,
                                "diffs": diffs_per_category,
                                "expected": core_count * diffs_per_category,
                                "generated": generated_diffs,
                                "dur": round(duration, 3),
                                "tot": tokens["total_tokens"],
                                "tin": tokens["input_tokens"],
                                "tout": tokens["output_tokens"],
                                "claim": score["claim_match_rate"],
                                "rating": score["rating_match_rate"],
                                "verdict": score["verdict_match_rate"],
                                "notes": notes or technique.get("name", ""),
                            },
                        )

        with ENGINE.begin() as conn:
            conn.execute(
                text("UPDATE eval_runs SET status = 'completed', completed_at = NOW() WHERE run_id::text = :rid"),
                {"rid": run_id},
            )
    except Exception as e:
        logger.exception("Benchmark run failed")
        with ENGINE.begin() as conn:
            conn.execute(
                text(
                    "UPDATE eval_runs SET status = 'failed', completed_at = NOW(), error_message = :err "
                    "WHERE run_id::text = :rid"
                ),
                {"rid": run_id, "err": str(e)[:4000]},
            )


@app.route('/evals')
def evals_page():
    with ENGINE.begin() as conn:
        datasets = conn.execute(
            text(
                "SELECT dataset_id, dataset_name, source_generation_id, source_battlecard_uuid::text AS source_battlecard_uuid, "
                "notes, created_by, created_at FROM eval_datasets ORDER BY created_at DESC"
            )
        ).mappings().all()
        runs = conn.execute(
            text(
                "SELECT run_id::text AS run_id, status, created_at, started_at, completed_at, error_message "
                "FROM eval_runs ORDER BY created_at DESC LIMIT 20"
            )
        ).mappings().all()
        recent_results = conn.execute(
            text(
                "SELECT result_id, run_id::text AS run_id, scenario_label, model_name, pass2_prompt_template_version, "
                "step5_execution_mode, step5_inline_fact_check, duration_seconds, total_tokens, "
                "eval_claim_match_rate, eval_rating_match_rate, eval_verdict_match_rate, notes "
                "FROM eval_run_results ORDER BY created_at DESC LIMIT 200"
            )
        ).mappings().all()
    return render_template(
        'evals.html',
        datasets=[dict(r) for r in datasets],
        runs=[dict(r) for r in runs],
        recent_results=[dict(r) for r in recent_results],
        default_techniques=_default_benchmark_techniques(),
    )


@app.route('/api/evals/datasets', methods=['POST'])
def create_eval_dataset():
    data = request.json or {}
    generation_id = data.get("generation_id")
    battlecard_id = data.get("battlecard_id")
    dataset_name = (data.get("dataset_name") or "").strip()
    notes = (data.get("notes") or "").strip()

    if not generation_id and not battlecard_id:
        return jsonify({"error": "generation_id or battlecard_id is required"}), 400

    if not generation_id and battlecard_id:
        with ENGINE.begin() as conn:
            generation_id = conn.execute(
                text("SELECT generation_id FROM battlecard_generations WHERE generation_uuid::text = :bid"),
                {"bid": battlecard_id},
            ).scalar()
    if not generation_id:
        return jsonify({"error": "Could not resolve generation"}), 404

    if not dataset_name:
        dataset_name = f"gen_{generation_id}_eval"

    result = _upsert_eval_dataset(dataset_name, int(generation_id), source_battlecard_uuid=battlecard_id or "", notes=notes)
    return jsonify({"success": True, **result})


@app.route('/api/evals/runs', methods=['POST'])
def start_eval_run():
    data = request.json or {}
    config = {
        "dataset_id": data.get("dataset_id"),
        "competitor_name": data.get("competitor_name", "Microsoft Fabric"),
        "product_area": data.get("product_area", "Data Platform"),
        "models": data.get("models") or ["databricks-claude-sonnet-4-5", "databricks-claude-opus-4-6"],
        "scenarios": data.get("scenarios") or [
            {"label": "2c_x_2d", "core_categories": 2, "diffs_per_category": 2},
            {"label": "4c_x_5d", "core_categories": 4, "diffs_per_category": 5},
            {"label": "8c_x_10d", "core_categories": 8, "diffs_per_category": 10},
        ],
        "techniques": data.get("techniques") or _default_benchmark_techniques(),
    }

    run_id = str(uuid4())
    with ENGINE.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO eval_runs (run_id, status, benchmark_config, requested_by) "
                "VALUES (CAST(:rid AS uuid), 'queued', CAST(:cfg AS jsonb), 'user')"
            ),
            {"rid": run_id, "cfg": json.dumps(config)},
        )

    t = threading.Thread(target=_run_benchmark_matrix, args=(run_id, config), daemon=True)
    t.start()
    return jsonify({"success": True, "run_id": run_id})


@app.route('/api/evals/runs/<run_id>')
def eval_run_status(run_id):
    with ENGINE.begin() as conn:
        run = conn.execute(
            text(
                "SELECT run_id::text AS run_id, status, created_at, started_at, completed_at, error_message "
                "FROM eval_runs WHERE run_id::text = :rid"
            ),
            {"rid": run_id},
        ).mappings().first()
        if not run:
            return jsonify({"error": "Run not found"}), 404
        results = conn.execute(
            text(
                "SELECT result_id, run_id::text AS run_id, session_id::text AS session_id, scenario_label, model_name, "
                "pass1_prompt_template_version, pass2_prompt_template_version, step4_execution_mode, step5_execution_mode, "
                "step5_inline_fact_check, core_categories, diffs_per_category, expected_diffs, generated_diffs, "
                "duration_seconds, total_tokens, input_tokens, output_tokens, eval_claim_match_rate, "
                "eval_rating_match_rate, eval_verdict_match_rate, notes "
                "FROM eval_run_results WHERE run_id::text = :rid ORDER BY result_id"
            ),
            {"rid": run_id},
        ).mappings().all()
    return jsonify({"run": dict(run), "results": [dict(r) for r in results]})


# =============================================================================
# Main
# =============================================================================


if __name__ == '__main__':
    init_db()
    # cleanup_stale_data()  # disabled: preserve data across restarts

    import argparse

    parser = argparse.ArgumentParser(description="Postgres Battlecard Review App")
    default_port = int(
        os.getenv("APP_PORT")
        or os.getenv("PORT")
        or os.getenv("DATABRICKS_APP_PORT")
        or "8000"
    )
    parser.add_argument("--port", type=int, default=default_port, help="Port to run on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=True, use_reloader=False)
