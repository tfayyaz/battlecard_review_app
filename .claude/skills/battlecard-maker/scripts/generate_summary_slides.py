#!/usr/bin/env python3
# /// script
# dependencies = [
#   "google-auth>=2.0.0",
#   "google-auth-oauthlib>=1.0.0",
#   "google-api-python-client>=2.0.0",
# ]
# ///
"""
Generate Executive Summary, Product Portfolio, and Technical Summary slides
from an existing battlecard UUID in Lakebase.

Reuses the app's load_battlecard_slides() to get data, then:
  - Uses an LLM to synthesize exec summary content in the right tone/style
  - Transforms L200 data into product portfolio and tech summary formats
  - Renders all 3 slide types to a single Google Slides presentation

Auth: google.auth.default() (ADC)
  gcloud auth application-default login \
    --scopes=https://www.googleapis.com/auth/presentations,https://www.googleapis.com/auth/drive

Example:
  uv run python scripts/generate_summary_slides_local.py \
    --battlecard-id b8501f8d-824c-4bca-93cd-15a99b49d9e1 \
    --output-dir /tmp/battlecard-slides/summaries
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import google.auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import load_battlecard_slides  # noqa: E402
from workflow_runner import get_openai_client, call_model  # noqa: E402

PRESENTATION_SCOPE = "https://www.googleapis.com/auth/presentations"
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
EMU_PER_PX = 12700


def px(v: int | float) -> int:
    return int(v * EMU_PER_PX)


def _rgb(hex_color: str) -> dict[str, float]:
    value = hex_color.lstrip("#")
    return {
        "red": int(value[0:2], 16) / 255.0,
        "green": int(value[2:4], 16) / 255.0,
        "blue": int(value[4:6], 16) / 255.0,
    }


def _ensure_rgb(color_like):
    if isinstance(color_like, str):
        return tuple(_rgb(color_like).values())
    if isinstance(color_like, (list, tuple)) and len(color_like) == 3:
        r, g, b = color_like
        return (r / 255.0, g / 255.0, b / 255.0) if max(r, g, b) > 1 else (float(r), float(g), float(b))
    raise ValueError(f"Unsupported color: {color_like}")


def parse_markup(text: str):
    """Parse **bold** and __underline__ markup, return plain text and spans."""
    i, n = 0, len(text)
    out, spans = [], []
    bold_open, underline_open = False, False

    while i < n:
        if text.startswith("**", i):
            bold_open = not bold_open
            i += 2
            continue
        if text.startswith("__", i):
            underline_open = not underline_open
            i += 2
            continue
        start = len(out)
        out.append(text[i])
        spans.append({"start": start, "end": start + 1, "bold": bold_open or None, "underline": underline_open or None})
        i += 1

    merged = []
    for s in spans:
        if not merged:
            merged.append(s)
            continue
        last = merged[-1]
        if last["end"] == s["start"] and last["bold"] == s["bold"] and last["underline"] == s["underline"]:
            last["end"] = s["end"]
        else:
            merged.append(s)
    return "".join(out), merged


def extract_battlecard_id(url_or_id: str) -> str:
    value = (url_or_id or "").strip()
    if not value:
        raise ValueError("Battlecard URL/ID is required")
    m = re.search(r"/battlecard/([0-9a-fA-F-]{36})", value)
    if m:
        return m.group(1)
    if re.fullmatch(r"[0-9a-fA-F-]{36}", value):
        return value
    raise ValueError(f"Could not parse battlecard UUID from: {value}")


def get_google_services():
    creds, _ = google.auth.default(scopes=[PRESENTATION_SCOPE, DRIVE_SCOPE])
    slides_service = build("slides", "v1", credentials=creds, cache_discovery=False)
    drive_service = build("drive", "v3", credentials=creds, cache_discovery=False)
    return slides_service, drive_service


def create_presentation(slides_service, title: str) -> str:
    body = {
        "title": title,
        "pageSize": {
            "width": {"magnitude": 9144000, "unit": "EMU"},
            "height": {"magnitude": 6858000, "unit": "EMU"},
        },
    }
    presentation = slides_service.presentations().create(body=body).execute()
    presentation_id = presentation["presentationId"]
    initial_slides = (
        slides_service.presentations().get(presentationId=presentation_id).execute().get("slides", [])
    )
    if initial_slides:
        slides_service.presentations().batchUpdate(
            presentationId=presentation_id,
            body={"requests": [{"deleteObject": {"objectId": initial_slides[0]["objectId"]}}]},
        ).execute()
    return presentation_id


def export_presentation_pdf(drive_service, presentation_id: str, out_pdf: Path):
    request = drive_service.files().export_media(fileId=presentation_id, mimeType="application/pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fh = io.FileIO(out_pdf, "wb")
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.close()


# ─── Data transformation helpers ──────────────────────────────────────────


def group_slides_by_category(slides: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for s in slides:
        grouped[(s.get("category") or "Uncategorized").strip()].append(s)
    for cat in grouped:
        grouped[cat].sort(key=lambda x: int(x.get("rank") or 0))
    return dict(grouped)


# ─── Executive Summary: LLM-powered synthesis ────────────────────────────

EXEC_SUMMARY_SYSTEM_PROMPT = """\
You are an expert competitive intelligence writer for Databricks sales battlecards.

## Tone and Style
- **Assertive and direct** — make strong, confident claims about competitor weaknesses without hedging.
- **Conversational but authoritative** — reads like an experienced sales leader coaching a rep.
- **Confrontational** — deliberately provocative phrasing that reframes the competitor's narrative.
- **Pro-Databricks bias is explicit** — unapologetically one-sided; exists to arm sellers with Databricks-favorable talking points.
- No filler or fluff — every sentence carries a specific claim or proof point.
- No neutral framing — this is persuasion, not analysis.

## Formatting Rules
- Use **bold** markup (double asterisks) for key claims, product names, and emphatic phrases.
- Use __underline__ markup (double underscores) for emphatic negatives: __not__, __no__, __lacks__, __only__, __don't__, __still requires__.
- Short, punchy sentences. Max 2–3 sentences per paragraph.
- Use rhetorical questions to challenge the competitor.
- Quantify claims where possible (2x, 3X, >10 years, etc.).
- Lead with what the competitor claims, then immediately counter with reality.
- Bullet points sparingly — only when listing distinct sub-items (prefix with •).

## Output Format
Return valid JSON matching this schema exactly:
{
  "intro_line": "Should [Competitor products] come up as a competitor, reinforce these 3 things:",
  "points": [
    {
      "quote": "Short punchy headline with **bold** and __underline__ markup",
      "body": "2-4 paragraphs of content. Separate paragraphs with double newlines. Use **bold** for key claims and __underline__ for negative emphasis."
    },
    ...exactly 3 points
  ]
}

IMPORTANT:
- "quote" is the italic coral headline at the top of each card. Keep it short (5-15 words), provocative, quotable. Use **bold** and __underline__ for emphasis on key words.
- "body" contains ALL content for the card — competitor weakness paragraphs AND Databricks counter-argument. Write 2-4 paragraphs. First 1-2 paragraphs describe the competitor's weakness/problem with specific details. Final paragraph describes Databricks advantage.
- Each point should cover a distinct strategic theme (e.g., rebundling/immaturity, lock-in/openness, cost/pricing traps, governance, AI readiness, fragmented architecture).
- Use actual product names, features, and technical details from the battlecard data.
- Match the style of these examples:
  * "Fabric is a **rebundling** of Power BI, Synapse, Data Factory and Azure Data Explorer..."
  * "__Lock-in__: OneLake is an expensive and vendor-locked storage. Access requires Fabric compute running and it costs 3X to access data from external services."
  * "__Expensive__: Fabric capacities are use-it-or-lose-it subscriptions:\\n• Too big a subscription = paying for what you __don't__ use\\n• Too small a subscription = throttled workloads..."
"""


def _prepare_exec_summary_context(slides: list[dict], competitor: str) -> str:
    """Build a compact context string from battlecard data for the LLM."""
    grouped = group_slides_by_category(slides)
    parts = [f"Competitor: {competitor}\n"]

    for cat_name, cat_slides in grouped.items():
        parts.append(f"\n## Category: {cat_name}")
        for s in cat_slides[:4]:
            diff_name = (s.get("key_differentiator") or "").strip()
            db_hl = (s.get("databricks_headline") or "").strip()
            fab_hl = (s.get("fabric_headline") or "").strip()
            db_rating = (s.get("databricks_rating") or "").strip()
            fab_rating = (s.get("fabric_rating") or "").strip()

            parts.append(f"\n### {diff_name}")
            if db_hl:
                parts.append(f"  Databricks [{db_rating}]: {db_hl}")
            if fab_hl:
                parts.append(f"  {competitor} [{fab_rating}]: {fab_hl}")

            for item in (s.get("databricks_detail_items") or [])[:2]:
                text = (item.get("text") or "").strip()
                if text:
                    parts.append(f"    DB detail: {text}")
            for item in (s.get("fabric_detail_items") or [])[:2]:
                text = (item.get("text") or "").strip()
                if text:
                    parts.append(f"    {competitor} detail: {text}")

            for item in (s.get("databricks_detail_items") or []):
                for vi in (item.get("verbose_items") or [])[:1]:
                    vtext = (vi.get("text") or "").strip()
                    if vtext:
                        parts.append(f"    DB L300: {vtext}")
            for item in (s.get("fabric_detail_items") or []):
                for vi in (item.get("verbose_items") or [])[:1]:
                    vtext = (vi.get("text") or "").strip()
                    if vtext:
                        parts.append(f"    {competitor} L300: {vtext}")

    return "\n".join(parts)


def build_exec_summary_from_slides(
    slides: list[dict], competitor: str, *, use_llm: bool = True, model: str = "databricks-claude-sonnet-4"
) -> dict:
    """
    Build exec summary content from battlecard slides.

    If use_llm=True (default), calls an LLM to synthesize the data into
    assertive, provocative executive summary content matching the battlecard
    style guide. Falls back to naive extraction if LLM call fails.
    """
    if use_llm:
        try:
            return _build_exec_summary_with_llm(slides, competitor, model=model)
        except Exception as e:
            print(f"  WARNING: LLM exec summary generation failed ({e}), falling back to naive extraction...")

    return _build_exec_summary_naive(slides, competitor)


def _build_exec_summary_with_llm(slides: list[dict], competitor: str, model: str) -> dict:
    """Use LLM to synthesize battlecard data into executive summary content."""
    context = _prepare_exec_summary_context(slides, competitor)

    user_prompt = f"""Based on the battlecard data below, write an Executive Summary with exactly 3 talking points for competing against {competitor}.

The executive summary should identify the 3 most impactful strategic themes from the data — think about what a senior sales leader would want their team to hammer home in every deal against {competitor}.

Good themes to consider: rebundling/immaturity, vendor lock-in, cost/pricing traps, governance gaps, AI/ML readiness, fragmented architecture, open vs proprietary, performance limitations.

Use specific product names, feature names, and technical details from the data. Be concrete, not generic. Each point should have 2-4 substantial paragraphs.

BATTLECARD DATA:
{context}"""

    print(f"  Calling {model} for exec summary synthesis...")
    client = get_openai_client()

    json_schema = {
        "name": "exec_summary",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "intro_line": {"type": "string"},
                "points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "quote": {"type": "string"},
                            "body": {"type": "string"},
                        },
                        "required": ["quote", "body"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["intro_line", "points"],
            "additionalProperties": False,
        },
    }

    full_prompt = f"{EXEC_SUMMARY_SYSTEM_PROMPT}\n\n---\n\n{user_prompt}"
    raw = call_model(client, model, full_prompt, json_schema=json_schema, temperature=0.3, max_tokens=4000)

    result = json.loads(raw)
    points = result.get("points", [])[:3]

    while len(points) < 3:
        points.append({"quote": "Databricks leads", "body": f"{competitor} falls short."})

    # Add empty response field for compatibility with drawing function
    for p in points:
        p.setdefault("response", "")

    return {
        "competitor": competitor,
        "intro_line": result.get("intro_line", ""),
        "points": points,
    }


def _build_exec_summary_naive(slides: list[dict], competitor: str) -> dict:
    """Fallback: build exec summary by naively extracting top differentiator data."""
    grouped = group_slides_by_category(slides)
    points = []
    for cat_name, cat_slides in grouped.items():
        if len(points) >= 3:
            break
        top_slide = cat_slides[0]
        db_headline = (top_slide.get("databricks_headline") or "").strip()
        fab_headline = (top_slide.get("fabric_headline") or "").strip()
        quote = fab_headline or f"{competitor} falls short on {cat_name}"
        fab_details = []
        for item in (top_slide.get("fabric_detail_items") or [])[:3]:
            text = (item.get("text") or "").strip()
            if text:
                fab_details.append(text)
        body = " ".join(fab_details) if fab_details else (top_slide.get("fabric_details") or "")
        db_details = []
        for item in (top_slide.get("databricks_detail_items") or [])[:3]:
            text = (item.get("text") or "").strip()
            if text:
                db_details.append(text)
        response = " ".join(db_details) if db_details else (top_slide.get("databricks_details") or "")
        if not body:
            body = f"**{fab_headline}** — {competitor} has gaps in {cat_name}."
        if not response:
            response = f"**{db_headline}** — Databricks provides a superior solution."
        points.append({"quote": quote, "body": body, "response": response})
    while len(points) < 3:
        points.append({"quote": "Databricks leads in key areas", "body": f"{competitor} lacks unified capabilities.", "response": ""})
    return {"competitor": competitor, "points": points}


# ─── Product Portfolio ────────────────────────────────────────────────────


def build_product_portfolio_from_slides(slides: list[dict], competitor: str) -> list[dict]:
    """Build product portfolio cards (2x4 grid = 8 cards) from battlecard categories."""
    grouped = group_slides_by_category(slides)
    cards = []
    for cat_name, cat_slides in grouped.items():
        lines = []
        for s in cat_slides[:4]:
            diff_name = (s.get("key_differentiator") or "").strip()
            db_headline = (s.get("databricks_headline") or "").strip()
            if diff_name:
                lines.append(f"**{diff_name}.** {db_headline}" if db_headline else f"**{diff_name}**")
        body = "\n".join(lines) if lines else "N/A"
        cards.append({"header": re.sub(r"\s*\([^)]*\)\s*$", "", cat_name).strip(), "body": body})
    while len(cards) < 8:
        cards.append({"header": "", "body": ""})
    return cards[:8]


# ─── Technical Summary ────────────────────────────────────────────────────


def build_tech_summary_from_slides(slides: list[dict], competitor: str) -> dict:
    """Build technical summary (3 columns) from battlecard data."""
    grouped = group_slides_by_category(slides)
    scored = []
    for cat_name, cat_slides in grouped.items():
        total_details = 0
        for s in cat_slides:
            for items_key in ("databricks_detail_items", "fabric_detail_items"):
                for item in (s.get(items_key) or []):
                    total_details += 1
                    total_details += len(item.get("verbose_items") or [])
        scored.append((total_details, cat_name, cat_slides))
    scored.sort(key=lambda x: -x[0])

    columns = []
    for _, cat_name, cat_slides in scored[:3]:
        header = re.sub(r"\s*\([^)]*\)\s*$", "", cat_name).strip()
        body_parts = []
        for s in cat_slides[:3]:
            fab_headline = (s.get("fabric_headline") or "").strip()
            db_headline = (s.get("databricks_headline") or "").strip()
            if fab_headline:
                part = f"**{fab_headline}** — "
                for item in (s.get("fabric_detail_items") or [])[:2]:
                    text = (item.get("text") or "").strip()
                    if text:
                        part += text + " "
                body_parts.append(part)
            if db_headline:
                part = f"Databricks: **{db_headline}** — "
                for item in (s.get("databricks_detail_items") or [])[:2]:
                    text = (item.get("text") or "").strip()
                    if text:
                        part += text + " "
                body_parts.append(part)
        body = "\n\n".join(p.strip() for p in body_parts if p.strip())
        columns.append({"header": header, "body": body})

    while len(columns) < 3:
        columns.append({"header": "Additional Insights", "body": "See detailed battlecard for more."})
    return {"title": f"What does {competitor} offer? Technical Deep-Dive", "columns": columns[:3]}


# ─── Slide drawing functions ──────────────────────────────────────────────


def _add_banner(reqs: list, slide_id: str, slide_w_px: int, font_family: str, banner_text: str, banner_bg_color: str, banner_height_px: int):
    """Add 'INTERNAL USE ONLY' banner to top of slide."""
    banner_id = f"banner_{uuid.uuid4().hex[:8]}"
    reqs.append({
        "createShape": {
            "objectId": banner_id,
            "shapeType": "RECTANGLE",
            "elementProperties": {
                "pageObjectId": slide_id,
                "size": {"width": {"magnitude": px(slide_w_px), "unit": "EMU"}, "height": {"magnitude": px(banner_height_px), "unit": "EMU"}},
                "transform": {"scaleX": 1, "scaleY": 1, "translateX": 0, "translateY": 0, "unit": "EMU"},
            },
        }
    })
    br, bg, bb = _ensure_rgb(banner_bg_color)
    reqs.append({
        "updateShapeProperties": {
            "objectId": banner_id,
            "shapeProperties": {
                "shapeBackgroundFill": {"solidFill": {"color": {"rgbColor": {"red": br, "green": bg, "blue": bb}}}},
                "outline": {"propertyState": "NOT_RENDERED"},
                "contentAlignment": "MIDDLE",
            },
            "fields": "shapeBackgroundFill,outline,contentAlignment",
        }
    })
    reqs.append({"insertText": {"objectId": banner_id, "insertionIndex": 0, "text": banner_text}})
    reqs.append({"updateParagraphStyle": {"objectId": banner_id, "textRange": {"type": "ALL"}, "style": {"alignment": "CENTER"}, "fields": "alignment"}})
    btr, btg, btb = _ensure_rgb("#ffffff")
    reqs.append({
        "updateTextStyle": {
            "objectId": banner_id,
            "textRange": {"type": "ALL"},
            "style": {
                "fontFamily": font_family, "bold": True,
                "fontSize": {"magnitude": 8, "unit": "PT"},
                "foregroundColor": {"opaqueColor": {"rgbColor": {"red": btr, "green": btg, "blue": btb}}},
            },
            "fields": "fontFamily,bold,fontSize,foregroundColor",
        }
    })
    return banner_id


def _apply_markup_spans(reqs: list, object_id: str, body_start: int, body_spans: list, *, cell_location: dict | None = None):
    """Apply bold/underline markup spans to text in a shape or table cell."""
    for sp in body_spans:
        style, fields = {}, []
        if sp.get("bold"):
            style["bold"] = True
            fields.append("bold")
        if sp.get("underline"):
            style["underline"] = True
            fields.append("underline")
        if fields:
            req = {
                "updateTextStyle": {
                    "objectId": object_id,
                    "textRange": {"type": "FIXED_RANGE", "startIndex": body_start + sp["start"], "endIndex": body_start + sp["end"]},
                    "style": style,
                    "fields": ",".join(fields),
                }
            }
            if cell_location:
                req["updateTextStyle"]["cellLocation"] = cell_location
            reqs.append(req)


def draw_exec_summary_slide(
    slides_svc,
    presentation_id: str,
    content: dict,
    *,
    insertion_index: int = 0,
    slide_bg_color="#323639",
    card_bg_color=(1, 0.37254903, 0.27058825),
    card_bg_alpha=0.1006,
    card_border_color="#FF5F45",
    number_bg_color="#FF5F45",
    number_text_color="#ffffff",
    quote_color="#FF5F45",
    body_color="#ffffff",
    title_color="#ffffff",
    font_family="DM Sans",
    title_font_pt=20,
    number_font_pt=14,
    quote_font_pt=10,
    body_font_pt=10,
):
    """Draw the executive summary slide: dark bg, 3 coral cards with numbered circles."""
    slide_id = f"slide_{uuid.uuid4().hex[:8]}"
    reqs = [{"createSlide": {"objectId": slide_id, "insertionIndex": insertion_index, "slideLayoutReference": {"predefinedLayout": "BLANK"}}}]

    # Background
    rr, gg, bb = _ensure_rgb(slide_bg_color)
    reqs.append({
        "updatePageProperties": {
            "objectId": slide_id,
            "pageProperties": {"pageBackgroundFill": {"solidFill": {"color": {"rgbColor": {"red": rr, "green": gg, "blue": bb}}}}},
            "fields": "pageBackgroundFill.solidFill.color",
        }
    })

    # Banner
    _add_banner(reqs, slide_id, 720, font_family, "INTERNAL USE ONLY", "#FF5F45", 12)

    # Confidential sub-banner
    sub_banner_id = f"subbanner_{uuid.uuid4().hex[:8]}"
    reqs.append({
        "createShape": {
            "objectId": sub_banner_id,
            "shapeType": "TEXT_BOX",
            "elementProperties": {
                "pageObjectId": slide_id,
                "size": {"width": {"magnitude": px(400), "unit": "EMU"}, "height": {"magnitude": px(18), "unit": "EMU"}},
                "transform": {"scaleX": 1, "scaleY": 1, "translateX": px(260), "translateY": px(14), "unit": "EMU"},
            },
        }
    })
    reqs.append({"insertText": {"objectId": sub_banner_id, "insertionIndex": 0, "text": "DATABRICKS CONFIDENTIAL – NDA REQUIRED – DO NOT DISTRIBUTE"}})
    sr, sg, sb = _ensure_rgb("#aaaaaa")
    reqs.append({
        "updateTextStyle": {
            "objectId": sub_banner_id, "textRange": {"type": "ALL"},
            "style": {"fontFamily": font_family, "fontSize": {"magnitude": 6, "unit": "PT"}, "foregroundColor": {"opaqueColor": {"rgbColor": {"red": sr, "green": sg, "blue": sb}}}},
            "fields": "fontFamily,fontSize,foregroundColor",
        }
    })
    reqs.append({"updateParagraphStyle": {"objectId": sub_banner_id, "textRange": {"type": "ALL"}, "style": {"alignment": "CENTER"}, "fields": "alignment"}})

    # Title: "EXECUTIVE SUMMARY"
    title_id = f"title_{uuid.uuid4().hex[:8]}"
    reqs.append({
        "createShape": {
            "objectId": title_id,
            "shapeType": "TEXT_BOX",
            "elementProperties": {
                "pageObjectId": slide_id,
                "size": {"width": {"magnitude": px(350), "unit": "EMU"}, "height": {"magnitude": px(36), "unit": "EMU"}},
                "transform": {"scaleX": 1, "scaleY": 1, "translateX": px(15), "translateY": px(18), "unit": "EMU"},
            },
        }
    })
    reqs.append({"insertText": {"objectId": title_id, "insertionIndex": 0, "text": "EXECUTIVE SUMMARY"}})
    tr, tg, tb = _ensure_rgb(title_color)
    reqs.append({
        "updateTextStyle": {
            "objectId": title_id, "textRange": {"type": "ALL"},
            "style": {"fontFamily": font_family, "bold": True, "fontSize": {"magnitude": title_font_pt, "unit": "PT"}, "foregroundColor": {"opaqueColor": {"rgbColor": {"red": tr, "green": tg, "blue": tb}}}},
            "fields": "fontFamily,bold,fontSize,foregroundColor",
        }
    })

    # Intro line: "Main talking points" + intro
    intro_line = content.get("intro_line", f"Should {content['competitor']} come up as a competitor, reinforce these 3 things:")
    intro_id = f"intro_{uuid.uuid4().hex[:8]}"
    reqs.append({
        "createShape": {
            "objectId": intro_id,
            "shapeType": "TEXT_BOX",
            "elementProperties": {
                "pageObjectId": slide_id,
                "size": {"width": {"magnitude": px(690), "unit": "EMU"}, "height": {"magnitude": px(40), "unit": "EMU"}},
                "transform": {"scaleX": 1, "scaleY": 1, "translateX": px(15), "translateY": px(50), "unit": "EMU"},
            },
        }
    })
    intro_text = f"Main talking points\n{intro_line}"
    reqs.append({"insertText": {"objectId": intro_id, "insertionIndex": 0, "text": intro_text}})
    # Style "Main talking points" in coral
    qr, qg, qb = _ensure_rgb(quote_color)
    reqs.append({
        "updateTextStyle": {
            "objectId": intro_id, "textRange": {"type": "FIXED_RANGE", "startIndex": 0, "endIndex": len("Main talking points")},
            "style": {"fontFamily": font_family, "fontSize": {"magnitude": 12, "unit": "PT"}, "foregroundColor": {"opaqueColor": {"rgbColor": {"red": qr, "green": qg, "blue": qb}}}},
            "fields": "fontFamily,fontSize,foregroundColor",
        }
    })
    # Style intro line in white
    reqs.append({
        "updateTextStyle": {
            "objectId": intro_id, "textRange": {"type": "FIXED_RANGE", "startIndex": len("Main talking points\n"), "endIndex": len(intro_text)},
            "style": {"fontFamily": font_family, "fontSize": {"magnitude": 10, "unit": "PT"}, "foregroundColor": {"opaqueColor": {"rgbColor": {"red": tr, "green": tg, "blue": tb}}}},
            "fields": "fontFamily,fontSize,foregroundColor",
        }
    })
    reqs.append({"updateParagraphStyle": {"objectId": intro_id, "textRange": {"type": "ALL"}, "style": {"lineSpacing": 115, "spaceAbove": {"magnitude": 0, "unit": "PT"}, "spaceBelow": {"magnitude": 0, "unit": "PT"}}, "fields": "lineSpacing,spaceAbove,spaceBelow"}})

    # Card positions (3 rows)
    card_configs = [
        {"translateY": 635525, "scaleY": 0.4166},
        {"translateY": 2045372, "scaleY": 0.3899},
        {"translateY": 3436500, "scaleY": 0.4166},
    ]
    circle_configs = [
        {"translateX": 240022, "translateY": 618353, "scaleY": 0.1492},
        {"translateX": 240022, "translateY": 1951661, "scaleY": 0.1492},
        {"translateX": 240025, "translateY": 3325968, "scaleY": 0.1685},
    ]

    points = content["points"]
    for i, (point, card_cfg, circle_cfg) in enumerate(zip(points, card_configs, circle_configs)):
        card_id = f"card_{uuid.uuid4().hex[:8]}"

        reqs.append({
            "createShape": {
                "objectId": card_id, "shapeType": "ROUND_RECTANGLE",
                "elementProperties": {
                    "pageObjectId": slide_id,
                    "size": {"width": {"magnitude": 3000000, "unit": "EMU"}, "height": {"magnitude": 3000000, "unit": "EMU"}},
                    "transform": {"scaleX": 2.7354, "scaleY": card_cfg["scaleY"], "translateX": 480475, "translateY": card_cfg["translateY"], "unit": "EMU"},
                },
            }
        })

        bg_r, bg_g, bg_b = _ensure_rgb(card_bg_color)
        br_r, br_g, br_b = _ensure_rgb(card_border_color)
        reqs.append({
            "updateShapeProperties": {
                "objectId": card_id,
                "shapeProperties": {
                    "shapeBackgroundFill": {"solidFill": {"color": {"rgbColor": {"red": bg_r, "green": bg_g, "blue": bg_b}}, "alpha": card_bg_alpha}},
                    "outline": {"outlineFill": {"solidFill": {"color": {"rgbColor": {"red": br_r, "green": br_g, "blue": br_b}}}}, "weight": {"magnitude": 9525, "unit": "EMU"}, "dashStyle": "DOT"},
                    "contentAlignment": "TOP",
                },
                "fields": "shapeBackgroundFill,outline,contentAlignment",
            }
        })

        # Build card text: quote + body + response
        quote_text = point.get("quote", "")
        body_text = point.get("body", "")
        response_text = point.get("response", "")

        plain_quote, quote_spans = parse_markup(quote_text)
        plain_body, body_spans = parse_markup(body_text)
        plain_response, response_spans = parse_markup(response_text)

        parts = []
        positions = {}
        if plain_quote:
            positions["quote_start"] = 0
            positions["quote_end"] = len(plain_quote)
            parts.append(plain_quote)
            parts.append(" ")
        if plain_body:
            positions["body_start"] = sum(len(p) for p in parts)
            parts.append(plain_body)
            positions["body_end"] = sum(len(p) for p in parts)
        if plain_response:
            parts.append("\n\n")
            positions["response_start"] = sum(len(p) for p in parts)
            parts.append(plain_response)
            positions["response_end"] = sum(len(p) for p in parts)

        full_text = "".join(parts)
        reqs.append({"insertText": {"objectId": card_id, "insertionIndex": 0, "text": full_text}})
        reqs.append({
            "updateParagraphStyle": {
                "objectId": card_id, "textRange": {"type": "ALL"},
                "style": {"alignment": "START", "lineSpacing": 110, "indentStart": {"magnitude": 13.5, "unit": "PT"}, "indentFirstLine": {"magnitude": 13.5, "unit": "PT"}, "spaceAbove": {"magnitude": 0, "unit": "PT"}, "spaceBelow": {"magnitude": 0, "unit": "PT"}},
                "fields": "alignment,lineSpacing,indentStart,indentFirstLine,spaceAbove,spaceBelow",
            }
        })

        # Base style: white body text
        body_r, body_g, body_b = _ensure_rgb(body_color)
        reqs.append({
            "updateTextStyle": {
                "objectId": card_id, "textRange": {"type": "ALL"},
                "style": {"fontFamily": font_family, "fontSize": {"magnitude": body_font_pt, "unit": "PT"}, "foregroundColor": {"opaqueColor": {"rgbColor": {"red": body_r, "green": body_g, "blue": body_b}}}},
                "fields": "fontFamily,fontSize,foregroundColor",
            }
        })

        # Style quote (coral, italic)
        if "quote_start" in positions:
            reqs.append({
                "updateTextStyle": {
                    "objectId": card_id,
                    "textRange": {"type": "FIXED_RANGE", "startIndex": positions["quote_start"], "endIndex": positions["quote_end"]},
                    "style": {"italic": True, "fontSize": {"magnitude": quote_font_pt, "unit": "PT"}, "foregroundColor": {"opaqueColor": {"rgbColor": {"red": qr, "green": qg, "blue": qb}}}},
                    "fields": "italic,fontSize,foregroundColor",
                }
            })
            # Apply bold/underline from quote markup
            _apply_markup_spans(reqs, card_id, positions["quote_start"], quote_spans)

        # Apply body markup
        if "body_start" in positions:
            _apply_markup_spans(reqs, card_id, positions["body_start"], body_spans)

        # Apply response markup
        if "response_start" in positions:
            _apply_markup_spans(reqs, card_id, positions["response_start"], response_spans)

        # Number circle
        circle_id = f"circle_{uuid.uuid4().hex[:8]}"
        reqs.append({
            "createShape": {
                "objectId": circle_id, "shapeType": "ELLIPSE",
                "elementProperties": {
                    "pageObjectId": slide_id,
                    "size": {"width": {"magnitude": 3000000, "unit": "EMU"}, "height": {"magnitude": 3000000, "unit": "EMU"}},
                    "transform": {"scaleX": 0.1492, "scaleY": circle_cfg["scaleY"], "translateX": circle_cfg["translateX"], "translateY": circle_cfg["translateY"], "unit": "EMU"},
                },
            }
        })
        nr, ng, nb = _ensure_rgb(number_bg_color)
        reqs.append({
            "updateShapeProperties": {
                "objectId": circle_id,
                "shapeProperties": {
                    "shapeBackgroundFill": {"solidFill": {"color": {"rgbColor": {"red": nr, "green": ng, "blue": nb}}}},
                    "outline": {"propertyState": "NOT_RENDERED"}, "contentAlignment": "MIDDLE",
                },
                "fields": "shapeBackgroundFill.solidFill.color,outline.propertyState,contentAlignment",
            }
        })
        reqs.append({"insertText": {"objectId": circle_id, "insertionIndex": 0, "text": str(i + 1)}})
        reqs.append({"updateParagraphStyle": {"objectId": circle_id, "textRange": {"type": "ALL"}, "style": {"alignment": "CENTER"}, "fields": "alignment"}})
        ntr, ntg, ntb = _ensure_rgb(number_text_color)
        reqs.append({
            "updateTextStyle": {
                "objectId": circle_id, "textRange": {"type": "ALL"},
                "style": {"fontFamily": font_family, "bold": True, "fontSize": {"magnitude": number_font_pt, "unit": "PT"}, "foregroundColor": {"opaqueColor": {"rgbColor": {"red": ntr, "green": ntg, "blue": ntb}}}},
                "fields": "fontFamily,bold,fontSize,foregroundColor",
            }
        })

    slides_svc.presentations().batchUpdate(presentationId=presentation_id, body={"requests": reqs}).execute()
    return slide_id


def draw_product_portfolio_slide(
    slides_svc, presentation_id: str, cards_data: list[dict], competitor: str,
    *, insertion_index: int = 1, font_family="DM Sans",
):
    """Draw a 2x4 product portfolio grid."""
    slide_w_px, num_rows, num_cols = 720, 2, 4
    side_margin_px, top_px = 25, 90
    card_gap_x_px, card_gap_y_px, card_height_px = 12, 12, 190
    header_font_pt, body_font_pt, title_font_pt = 14, 11, 28

    slide_id = f"slide_{uuid.uuid4().hex[:8]}"
    reqs = [{"createSlide": {"objectId": slide_id, "insertionIndex": insertion_index, "slideLayoutReference": {"predefinedLayout": "BLANK"}}}]

    # White background
    reqs.append({"updatePageProperties": {"objectId": slide_id, "pageProperties": {"pageBackgroundFill": {"solidFill": {"color": {"rgbColor": _rgb("#ffffff")}}}}, "fields": "pageBackgroundFill.solidFill.color"}})

    # Banner
    _add_banner(reqs, slide_id, slide_w_px, font_family, "INTERNAL USE ONLY", "#FF5F46", 12)

    # Title
    title = f"PRODUCT PORTFOLIO - Databricks vs {competitor}"
    t_id = f"title_{uuid.uuid4().hex[:8]}"
    reqs.append({
        "createShape": {
            "objectId": t_id, "shapeType": "TEXT_BOX",
            "elementProperties": {
                "pageObjectId": slide_id,
                "size": {"width": {"magnitude": px(slide_w_px - 50), "unit": "EMU"}, "height": {"magnitude": px(40), "unit": "EMU"}},
                "transform": {"scaleX": 1, "scaleY": 1, "translateX": px(25), "translateY": px(22), "unit": "EMU"},
            },
        }
    })
    reqs.append({"insertText": {"objectId": t_id, "insertionIndex": 0, "text": title}})
    reqs.append({"updateTextStyle": {"objectId": t_id, "textRange": {"type": "ALL"}, "style": {"fontFamily": font_family, "bold": True, "fontSize": {"magnitude": title_font_pt, "unit": "PT"}, "foregroundColor": {"opaqueColor": {"rgbColor": _rgb("#000000")}}}, "fields": "fontFamily,bold,fontSize,foregroundColor"}})
    reqs.append({"updateShapeProperties": {"objectId": t_id, "shapeProperties": {"outline": {"propertyState": "NOT_RENDERED"}, "contentAlignment": "MIDDLE"}, "fields": "outline.propertyState,contentAlignment"}})

    # Grid
    content_width = slide_w_px - 2 * side_margin_px
    card_width_px = (content_width - (num_cols - 1) * card_gap_x_px) // num_cols

    for idx, card in enumerate(cards_data[:num_rows * num_cols]):
        row, col = idx // num_cols, idx % num_cols
        x = side_margin_px + col * (card_width_px + card_gap_x_px)
        y = top_px + row * (card_height_px + card_gap_y_px)
        card_id = f"card_{uuid.uuid4().hex[:8]}"

        reqs.append({
            "createShape": {
                "objectId": card_id, "shapeType": "RECTANGLE",
                "elementProperties": {
                    "pageObjectId": slide_id,
                    "size": {"width": {"magnitude": px(card_width_px), "unit": "EMU"}, "height": {"magnitude": px(card_height_px), "unit": "EMU"}},
                    "transform": {"scaleX": 1, "scaleY": 1, "translateX": px(x), "translateY": px(y), "unit": "EMU"},
                },
            }
        })
        reqs.append({
            "updateShapeProperties": {
                "objectId": card_id,
                "shapeProperties": {
                    "shapeBackgroundFill": {"solidFill": {"color": {"rgbColor": _rgb("#ffffff")}}},
                    "outline": {"outlineFill": {"solidFill": {"color": {"rgbColor": _rgb("#FF5F46")}}}, "weight": {"magnitude": 1, "unit": "PT"}, "dashStyle": "SOLID"},
                    "contentAlignment": "TOP",
                },
                "fields": "shapeBackgroundFill,outline,contentAlignment",
            }
        })

        header = card.get("header", "")
        body = card.get("body", "")
        plain_body, body_spans = parse_markup(body)
        full_text = header + "\n\n" + plain_body if plain_body else header
        header_end = len(header)
        body_start = header_end + 2

        if full_text.strip():
            reqs.append({"insertText": {"objectId": card_id, "insertionIndex": 0, "text": full_text}})
            reqs.append({"updateParagraphStyle": {"objectId": card_id, "textRange": {"type": "ALL"}, "style": {"alignment": "START"}, "fields": "alignment"}})
            reqs.append({"updateTextStyle": {"objectId": card_id, "textRange": {"type": "FIXED_RANGE", "startIndex": 0, "endIndex": header_end}, "style": {"fontFamily": font_family, "bold": True, "fontSize": {"magnitude": header_font_pt, "unit": "PT"}, "foregroundColor": {"opaqueColor": {"rgbColor": _rgb("#FF5F46")}}}, "fields": "fontFamily,bold,fontSize,foregroundColor"}})
            if plain_body:
                reqs.append({"updateTextStyle": {"objectId": card_id, "textRange": {"type": "FIXED_RANGE", "startIndex": body_start, "endIndex": body_start + len(plain_body)}, "style": {"fontFamily": font_family, "fontSize": {"magnitude": body_font_pt, "unit": "PT"}, "foregroundColor": {"opaqueColor": {"rgbColor": _rgb("#000000")}}}, "fields": "fontFamily,fontSize,foregroundColor"}})
                _apply_markup_spans(reqs, card_id, body_start, body_spans)

    slides_svc.presentations().batchUpdate(presentationId=presentation_id, body={"requests": reqs}).execute()
    return slide_id


def draw_tech_summary_slide(
    slides_svc, presentation_id: str, content: dict,
    *, insertion_index: int = 2, font_family="DM Sans",
):
    """Draw the technical summary slide with 3-column dashed-border cards."""
    slide_w_px = 720
    side_margin_px, top_px, card_gap_px, card_height_px = 30, 70, 20, 380
    header_font_pt, body_font_pt, title_font_pt = 16, 10, 28

    slide_id = f"slide_{uuid.uuid4().hex[:8]}"
    reqs = [{"createSlide": {"objectId": slide_id, "insertionIndex": insertion_index, "slideLayoutReference": {"predefinedLayout": "BLANK"}}}]

    # White background
    reqs.append({"updatePageProperties": {"objectId": slide_id, "pageProperties": {"pageBackgroundFill": {"solidFill": {"color": {"rgbColor": _rgb("#ffffff")}}}}, "fields": "pageBackgroundFill.solidFill.color"}})

    # Banner
    _add_banner(reqs, slide_id, slide_w_px, font_family, "INTERNAL USE ONLY", "#FF5F46", 9)

    # Title
    t_id = f"title_{uuid.uuid4().hex[:8]}"
    reqs.append({
        "createShape": {
            "objectId": t_id, "shapeType": "TEXT_BOX",
            "elementProperties": {
                "pageObjectId": slide_id,
                "size": {"width": {"magnitude": px(slide_w_px - 60), "unit": "EMU"}, "height": {"magnitude": px(40), "unit": "EMU"}},
                "transform": {"scaleX": 1, "scaleY": 1, "translateX": px(30), "translateY": px(22), "unit": "EMU"},
            },
        }
    })
    reqs.append({"insertText": {"objectId": t_id, "insertionIndex": 0, "text": content["title"]}})
    reqs.append({"updateTextStyle": {"objectId": t_id, "textRange": {"type": "ALL"}, "style": {"fontFamily": font_family, "fontSize": {"magnitude": title_font_pt, "unit": "PT"}, "foregroundColor": {"opaqueColor": {"rgbColor": _rgb("#000000")}}}, "fields": "fontFamily,fontSize,foregroundColor"}})
    reqs.append({"updateShapeProperties": {"objectId": t_id, "shapeProperties": {"outline": {"propertyState": "NOT_RENDERED"}, "contentAlignment": "MIDDLE"}, "fields": "outline.propertyState,contentAlignment"}})

    # 3 cards
    content_width = slide_w_px - 2 * side_margin_px
    card_width_px = (content_width - 2 * card_gap_px) // 3

    for i, col in enumerate(content["columns"][:3]):
        card_id = f"card_{uuid.uuid4().hex[:8]}"
        x_px = side_margin_px + i * (card_width_px + card_gap_px)

        reqs.append({
            "createShape": {
                "objectId": card_id, "shapeType": "RECTANGLE",
                "elementProperties": {
                    "pageObjectId": slide_id,
                    "size": {"width": {"magnitude": px(card_width_px), "unit": "EMU"}, "height": {"magnitude": px(card_height_px), "unit": "EMU"}},
                    "transform": {"scaleX": 1, "scaleY": 1, "translateX": px(x_px), "translateY": px(top_px), "unit": "EMU"},
                },
            }
        })
        reqs.append({
            "updateShapeProperties": {
                "objectId": card_id,
                "shapeProperties": {
                    "shapeBackgroundFill": {"solidFill": {"color": {"rgbColor": _rgb("#ffffff")}}},
                    "outline": {"outlineFill": {"solidFill": {"color": {"rgbColor": _rgb("#FF5F46")}}}, "weight": {"magnitude": 2, "unit": "PT"}, "dashStyle": "DASH"},
                    "contentAlignment": "TOP",
                },
                "fields": "shapeBackgroundFill.solidFill.color,outline.outlineFill.solidFill.color,outline.weight,outline.dashStyle,contentAlignment",
            }
        })

        header = col["header"]
        body_text = col["body"]
        plain_body, body_spans = parse_markup(body_text)
        full_text = header + "\n\n" + plain_body
        header_end = len(header)
        body_start = header_end + 2

        reqs.append({"insertText": {"objectId": card_id, "insertionIndex": 0, "text": full_text}})
        reqs.append({"updateParagraphStyle": {"objectId": card_id, "textRange": {"type": "ALL"}, "style": {"alignment": "START"}, "fields": "alignment"}})
        reqs.append({"updateTextStyle": {"objectId": card_id, "textRange": {"type": "FIXED_RANGE", "startIndex": 0, "endIndex": header_end}, "style": {"fontFamily": font_family, "bold": True, "fontSize": {"magnitude": header_font_pt, "unit": "PT"}, "foregroundColor": {"opaqueColor": {"rgbColor": _rgb("#FF5F46")}}}, "fields": "fontFamily,bold,fontSize,foregroundColor"}})
        if plain_body:
            reqs.append({"updateTextStyle": {"objectId": card_id, "textRange": {"type": "FIXED_RANGE", "startIndex": body_start, "endIndex": body_start + len(plain_body)}, "style": {"fontFamily": font_family, "fontSize": {"magnitude": body_font_pt, "unit": "PT"}, "foregroundColor": {"opaqueColor": {"rgbColor": _rgb("#000000")}}}, "fields": "fontFamily,fontSize,foregroundColor"}})
            _apply_markup_spans(reqs, card_id, body_start, body_spans)

    slides_svc.presentations().batchUpdate(presentationId=presentation_id, body={"requests": reqs}).execute()
    return slide_id


# ─── Main ─────────────────────────────────────────────────────────────────


def build_summary_deck(
    *,
    battlecard_id: str,
    output_dir: Path,
    presentation_title: str | None = None,
    use_llm: bool = True,
    model: str = "databricks-claude-sonnet-4",
) -> dict[str, Any]:
    print(f"Loading battlecard data for {battlecard_id}...")
    slides, gen_info = load_battlecard_slides(battlecard_id)
    if not slides:
        raise RuntimeError(f"No slides found for battlecard_id={battlecard_id}")

    competitor = (slides[0].get("competitor") or gen_info.get("competitor") or "Competitor").strip()
    product_area = (gen_info.get("product_area") or "Data Platform").strip()
    print(f"  Competitor: {competitor}")
    print(f"  Product area: {product_area}")
    print(f"  Differentiators: {len(slides)}")

    print("Building executive summary content...")
    exec_content = build_exec_summary_from_slides(slides, competitor, use_llm=use_llm, model=model)

    print("Building product portfolio content...")
    portfolio_cards = build_product_portfolio_from_slides(slides, competitor)

    print("Building technical summary content...")
    tech_content = build_tech_summary_from_slides(slides, competitor)

    title = presentation_title or f"{competitor} Summary Battlecard"
    print(f"Creating Google Slides presentation: {title}")
    slides_service, drive_service = get_google_services()
    presentation_id = create_presentation(slides_service, title)

    print("Drawing Executive Summary slide (1/3)...")
    draw_exec_summary_slide(slides_service, presentation_id, exec_content, insertion_index=0)

    print("Drawing Product Portfolio slide (2/3)...")
    draw_product_portfolio_slide(slides_service, presentation_id, portfolio_cards, competitor, insertion_index=1)

    print("Drawing Technical Summary slide (3/3)...")
    draw_tech_summary_slide(slides_service, presentation_id, tech_content, insertion_index=2)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"summary_{battlecard_id}_{timestamp}.pdf"
    print(f"Exporting PDF to {pdf_path}...")
    export_presentation_pdf(drive_service, presentation_id, pdf_path)

    result = {
        "battlecard_id": battlecard_id,
        "competitor": competitor,
        "product_area": product_area,
        "presentation_id": presentation_id,
        "presentation_url": f"https://docs.google.com/presentation/d/{presentation_id}/edit",
        "pdf_path": str(pdf_path),
        "slides_generated": ["executive_summary", "product_portfolio", "technical_summary"],
        "differentiators_used": len(slides),
        "llm_used": use_llm,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = output_dir / f"summary_{battlecard_id}_{timestamp}.json"
    metadata_path.write_text(json.dumps(result, indent=2))
    result["metadata_path"] = str(metadata_path)
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate Exec Summary + Product Portfolio + Tech Summary slides.")
    parser.add_argument("--battlecard-url", help="Battlecard URL")
    parser.add_argument("--battlecard-id", help="Battlecard UUID")
    parser.add_argument("--output-dir", default="/tmp/battlecard-slides/summaries", help="Output directory")
    parser.add_argument("--presentation-title", default=None, help="Override presentation title")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM synthesis, use naive data extraction")
    parser.add_argument("--model", default="databricks-claude-sonnet-4", help="LLM model for exec summary (default: databricks-claude-sonnet-4)")
    args = parser.parse_args()

    if not args.battlecard_url and not args.battlecard_id:
        parser.error("Provide either --battlecard-url or --battlecard-id")

    raw = args.battlecard_id or args.battlecard_url
    battlecard_id = extract_battlecard_id(raw)

    result = build_summary_deck(
        battlecard_id=battlecard_id,
        output_dir=Path(args.output_dir),
        presentation_title=args.presentation_title,
        use_llm=not args.no_llm,
        model=args.model,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
