#!/usr/bin/env python3
# /// script
# dependencies = [
#   "google-auth>=2.0.0",
#   "google-auth-oauthlib>=1.0.0",
#   "google-api-python-client>=2.0.0",
# ]
# ///
"""
Generate a Google Slides battlecard from an existing battlecard UUID, then export it as PDF.

This script reuses the app's existing Lakebase loader (`load_battlecard_slides`) so it stays
aligned with the review app's battlecard content model.

Auth model matches prior battlecard scripts:
  - google.auth.default() (ADC)
  - If needed: gcloud auth application-default login --scopes=https://www.googleapis.com/auth/presentations,https://www.googleapis.com/auth/drive

Example:
  uv run --with google-auth --with google-auth-oauthlib --with google-api-python-client \
    scripts/generate_battlecard_slides_local.py \
    --battlecard-url "https://.../battlecard/<uuid>" \
    --output-dir /tmp/battlecard-slides
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import textwrap
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
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


PRESENTATION_SCOPE = "https://www.googleapis.com/auth/presentations"
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
EMU_PER_PX = 12700


@dataclass
class SlideConfig:
    """All visual parameters for battlecard slide generation."""

    # --- Global ---
    font_family: str = "DM Sans"

    # --- Content controls ---
    rows_per_slide: int = 4
    max_details_per_side: int = 2
    line_wrap: int = 62
    max_desc_chars: int = 140
    strip_parentheticals: bool = True

    # --- Rating icons ---
    icon_advantage: str = "✓✓"
    icon_partial: str = "⚠"
    icon_disadvantage: str = "✗"
    color_icon_advantage: str = "#00D95F"
    color_icon_partial: str = "#FF9100"
    color_icon_disadvantage: str = "#FF1744"

    # --- Banner ---
    banner_height_px: int = 12
    banner_font_size_pt: float = 8
    banner_bg_color: str = "#FF5F46"
    banner_text_color: str = "#FFFFFF"
    banner_text: str = "INTERNAL USE ONLY"

    # --- Category slide title ---
    cat_title_font_size_pt: float = 19
    cat_title_color: str = "#1A3A3A"
    cat_title_height_px: int = 44
    cat_title_x_px: int = 10
    cat_title_y_px: int = 18

    # --- Table layout ---
    table_width_px: int = 690
    table_x_px: int = 15
    table_y_px: int = 62
    col_widths_px: list[int] = field(default_factory=lambda: [170, 260, 260])

    # --- Table header row ---
    header_font_size_pt: float = 10
    header_bg_color: str = "#1A3A3A"
    header_text_color: str = "#FFFFFF"

    # --- Column 0 (Key Differentiator) data cells ---
    col0_bg_color: str = "#F0F0F0"
    col0_title_font_size_pt: float = 9
    col0_title_color: str = "#000000"
    col0_subtitle_font_size_pt: float = 8
    col0_subtitle_color: str = "#777777"
    col0_base_font_size_pt: float = 8.5
    col0_base_color: str = "#666666"

    # --- Columns 1 & 2 (vendor) data cells ---
    vendor_body_font_size_pt: float = 8.5
    vendor_text_color: str = "#111827"

    # --- Title slide ---
    title_slide_title_font_size_pt: float = 32
    title_slide_title_color: str = "#1A3A3A"
    title_slide_subtitle_font_size_pt: float = 13
    title_slide_subtitle_color: str = "#4B5563"
    title_slide_y_px: int = 180

    @property
    def rating_icons(self) -> dict[str, str]:
        return {
            "advantage": self.icon_advantage,
            "partial": self.icon_partial,
            "disadvantage": self.icon_disadvantage,
        }

    @property
    def rating_colors(self) -> dict[str, str]:
        return {
            "advantage": self.color_icon_advantage,
            "partial": self.color_icon_partial,
            "disadvantage": self.color_icon_disadvantage,
        }


def _rgb(hex_color: str) -> dict[str, float]:
    value = hex_color.lstrip("#")
    return {
        "red": int(value[0:2], 16) / 255.0,
        "green": int(value[2:4], 16) / 255.0,
        "blue": int(value[4:6], 16) / 255.0,
    }


def strip_parenthetical(text: str) -> str:
    """Remove parenthetical suffixes: 'Foo (bar baz)' → 'Foo'."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()


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
    try:
        creds, _ = google.auth.default(scopes=[PRESENTATION_SCOPE, DRIVE_SCOPE])
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Google ADC auth failed. Run: "
            "gcloud auth application-default login "
            "--scopes=https://www.googleapis.com/auth/presentations,https://www.googleapis.com/auth/drive"
        ) from exc

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

    initial_slides = slides_service.presentations().get(
        presentationId=presentation_id
    ).execute().get("slides", [])
    if initial_slides:
        slides_service.presentations().batchUpdate(
            presentationId=presentation_id,
            body={"requests": [{"deleteObject": {"objectId": initial_slides[0]["objectId"]}}]},
        ).execute()
    return presentation_id


def normalize_competitor_name(raw: str) -> str:
    if not raw:
        return "Competitor"
    return raw.strip()


def format_key_diff_cell(slide: dict[str, Any], cfg: SlideConfig) -> dict[str, Any]:
    """Return structured cell data: {title, subtitle, full_text}."""
    title = (slide.get("key_differentiator") or "").strip()
    desc = (slide.get("description") or "").strip()
    if len(desc) > cfg.max_desc_chars:
        desc = desc[: cfg.max_desc_chars - 1].rstrip() + "..."
    full_text = f"{title}\n{desc}" if desc else title
    return {"title": title, "subtitle": desc, "full_text": full_text}


def _extract_detail_lines(
    detail_items: list[dict[str, Any]] | None,
    fallback_text: str,
    max_details_per_side: int,
) -> list[str]:
    lines: list[str] = []
    if detail_items:
        for item in detail_items:
            text_value = (item.get("text") or "").strip()
            if text_value:
                lines.append(text_value)
            if len(lines) >= max_details_per_side:
                break
    if not lines and fallback_text:
        parts = [p.strip(" -\t") for p in fallback_text.split("\n") if p.strip()]
        lines.extend(parts[:max_details_per_side])
    return lines[:max_details_per_side]


def format_vendor_cell(
    headline: str,
    detail_items: list[dict[str, Any]] | None,
    fallback_text: str,
    cfg: SlideConfig,
    rating: str = "",
) -> dict[str, Any]:
    """Return structured cell data: {icon, icon_color, headline, details_text, full_text}."""
    icon = cfg.rating_icons.get(rating, "")
    icon_color = cfg.rating_colors.get(rating, "")

    headline_text = (headline or "").strip()
    detail_lines: list[str] = []
    for line in _extract_detail_lines(detail_items, fallback_text, cfg.max_details_per_side):
        wrapped = textwrap.fill(line, width=cfg.line_wrap)
        detail_lines.append(f"- {wrapped}")
    details_text = "\n".join(detail_lines)

    # Build full text: "icon headline\ndetails"
    parts: list[str] = []
    prefix = f"{icon} " if icon else ""
    if headline_text:
        parts.append(f"{prefix}{headline_text}")
    if details_text:
        parts.append(details_text)
    full_text = "\n".join(parts).strip()

    return {
        "icon": icon,
        "icon_color": icon_color,
        "headline": headline_text,
        "details_text": details_text,
        "full_text": full_text,
        "prefix": prefix,
    }


def group_slides_by_category(slides: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for slide in slides:
        grouped[(slide.get("category") or "Uncategorized").strip()].append(slide)
    for category in grouped:
        grouped[category].sort(key=lambda s: int(s.get("rank") or 0))
    return dict(grouped)


def chunk_rows(rows: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    return [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]


def build_slide_with_table_requests(
    *,
    slide_id: str,
    title: str,
    competitor_name: str,
    rows: list[dict[str, Any]],
    cfg: SlideConfig,
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    font = cfg.font_family

    title_id = f"title_{uuid.uuid4().hex[:8]}"
    banner_id = f"banner_{uuid.uuid4().hex[:8]}"
    table_id = f"table_{uuid.uuid4().hex[:8]}"

    # Banner
    requests.append({
        "createShape": {
            "objectId": banner_id,
            "shapeType": "RECTANGLE",
            "elementProperties": {
                "pageObjectId": slide_id,
                "size": {
                    "width": {"magnitude": int(720 * EMU_PER_PX), "unit": "EMU"},
                    "height": {"magnitude": int(cfg.banner_height_px * EMU_PER_PX), "unit": "EMU"},
                },
                "transform": {
                    "scaleX": 1, "scaleY": 1,
                    "translateX": 0, "translateY": 0,
                    "unit": "EMU",
                },
            },
        }
    })
    requests.append({
        "updateShapeProperties": {
            "objectId": banner_id,
            "shapeProperties": {
                "shapeBackgroundFill": {"solidFill": {"color": {"rgbColor": _rgb(cfg.banner_bg_color)}}},
                "outline": {"propertyState": "NOT_RENDERED"},
            },
            "fields": "shapeBackgroundFill,outline",
        }
    })
    requests.append({"insertText": {"objectId": banner_id, "insertionIndex": 0, "text": cfg.banner_text}})
    requests.append({
        "updateParagraphStyle": {
            "objectId": banner_id,
            "textRange": {"type": "ALL"},
            "style": {"alignment": "CENTER"},
            "fields": "alignment",
        }
    })
    requests.append({
        "updateTextStyle": {
            "objectId": banner_id,
            "textRange": {"type": "ALL"},
            "style": {
                "bold": True,
                "fontFamily": font,
                "fontSize": {"magnitude": cfg.banner_font_size_pt, "unit": "PT"},
                "foregroundColor": {"opaqueColor": {"rgbColor": _rgb(cfg.banner_text_color)}},
            },
            "fields": "bold,fontFamily,fontSize,foregroundColor",
        }
    })

    # Category title
    requests.append({
        "createShape": {
            "objectId": title_id,
            "shapeType": "TEXT_BOX",
            "elementProperties": {
                "pageObjectId": slide_id,
                "size": {
                    "width": {"magnitude": int(700 * EMU_PER_PX), "unit": "EMU"},
                    "height": {"magnitude": int(cfg.cat_title_height_px * EMU_PER_PX), "unit": "EMU"},
                },
                "transform": {
                    "scaleX": 1, "scaleY": 1,
                    "translateX": int(cfg.cat_title_x_px * EMU_PER_PX),
                    "translateY": int(cfg.cat_title_y_px * EMU_PER_PX),
                    "unit": "EMU",
                },
            },
        }
    })
    requests.append({"insertText": {"objectId": title_id, "insertionIndex": 0, "text": title}})
    requests.append({
        "updateTextStyle": {
            "objectId": title_id,
            "textRange": {"type": "ALL"},
            "style": {
                "bold": True,
                "fontFamily": font,
                "fontSize": {"magnitude": cfg.cat_title_font_size_pt, "unit": "PT"},
                "foregroundColor": {"opaqueColor": {"rgbColor": _rgb(cfg.cat_title_color)}},
            },
            "fields": "bold,fontFamily,fontSize,foregroundColor",
        }
    })

    # Table container — height=1px so Google Slides auto-sizes rows to fit content
    row_count = len(rows) + 1
    requests.append({
        "createTable": {
            "objectId": table_id,
            "rows": row_count,
            "columns": 3,
            "elementProperties": {
                "pageObjectId": slide_id,
                "size": {
                    "width": {"magnitude": int(cfg.table_width_px * EMU_PER_PX), "unit": "EMU"},
                    "height": {"magnitude": int(1 * EMU_PER_PX), "unit": "EMU"},
                },
                "transform": {
                    "scaleX": 1, "scaleY": 1,
                    "translateX": int(cfg.table_x_px * EMU_PER_PX),
                    "translateY": int(cfg.table_y_px * EMU_PER_PX),
                    "unit": "EMU",
                },
            },
        }
    })

    # Column widths
    for col_idx, width_px in enumerate(cfg.col_widths_px):
        requests.append({
            "updateTableColumnProperties": {
                "objectId": table_id,
                "columnIndices": [col_idx],
                "tableColumnProperties": {
                    "columnWidth": {"magnitude": int(width_px * EMU_PER_PX), "unit": "EMU"}
                },
                "fields": "columnWidth",
            }
        })

    # Header row
    headers = ["Key Differentiator", "Databricks", competitor_name]
    for col_idx, value in enumerate(headers):
        requests.append({
            "insertText": {
                "objectId": table_id,
                "cellLocation": {"rowIndex": 0, "columnIndex": col_idx},
                "insertionIndex": 0,
                "text": value,
            }
        })
        requests.append({
            "updateTableCellProperties": {
                "objectId": table_id,
                "tableRange": {
                    "location": {"rowIndex": 0, "columnIndex": col_idx},
                    "rowSpan": 1, "columnSpan": 1,
                },
                "tableCellProperties": {
                    "tableCellBackgroundFill": {"solidFill": {"color": {"rgbColor": _rgb(cfg.header_bg_color)}}},
                    "contentAlignment": "TOP",
                },
                "fields": "tableCellBackgroundFill,contentAlignment",
            }
        })
        requests.append({
            "updateTextStyle": {
                "objectId": table_id,
                "cellLocation": {"rowIndex": 0, "columnIndex": col_idx},
                "textRange": {"type": "ALL"},
                "style": {
                    "bold": True,
                    "fontFamily": font,
                    "fontSize": {"magnitude": cfg.header_font_size_pt, "unit": "PT"},
                    "foregroundColor": {"opaqueColor": {"rgbColor": _rgb(cfg.header_text_color)}},
                },
                "fields": "bold,fontFamily,fontSize,foregroundColor",
            }
        })
        requests.append({
            "updateParagraphStyle": {
                "objectId": table_id,
                "cellLocation": {"rowIndex": 0, "columnIndex": col_idx},
                "textRange": {"type": "ALL"},
                "style": {
                    "spaceAbove": {"magnitude": 0, "unit": "PT"},
                    "spaceBelow": {"magnitude": 0, "unit": "PT"},
                },
                "fields": "spaceAbove,spaceBelow",
            }
        })

    # Data rows
    for i, slide in enumerate(rows, start=1):
        key_data = format_key_diff_cell(slide, cfg)
        db_data = format_vendor_cell(
            headline=(slide.get("databricks_headline") or ""),
            detail_items=slide.get("databricks_detail_items") or [],
            fallback_text=(slide.get("databricks_details") or ""),
            cfg=cfg,
            rating=(slide.get("databricks_rating") or ""),
        )
        comp_data = format_vendor_cell(
            headline=(slide.get("fabric_headline") or ""),
            detail_items=slide.get("fabric_detail_items") or [],
            fallback_text=(slide.get("fabric_details") or ""),
            cfg=cfg,
            rating=(slide.get("fabric_rating") or ""),
        )

        # --- Column 0: Key Differentiator (bold title + grey subtitle) ---
        col0_text = key_data["full_text"][:1200]
        requests.append({
            "insertText": {
                "objectId": table_id,
                "cellLocation": {"rowIndex": i, "columnIndex": 0},
                "insertionIndex": 0,
                "text": col0_text,
            }
        })
        requests.append({
            "updateTableCellProperties": {
                "objectId": table_id,
                "tableRange": {
                    "location": {"rowIndex": i, "columnIndex": 0},
                    "rowSpan": 1, "columnSpan": 1,
                },
                "tableCellProperties": {
                    "tableCellBackgroundFill": {"solidFill": {"color": {"rgbColor": _rgb(cfg.col0_bg_color)}}},
                    "contentAlignment": "TOP",
                },
                "fields": "tableCellBackgroundFill,contentAlignment",
            }
        })
        # Base style
        requests.append({
            "updateTextStyle": {
                "objectId": table_id,
                "cellLocation": {"rowIndex": i, "columnIndex": 0},
                "textRange": {"type": "ALL"},
                "style": {
                    "fontFamily": font,
                    "fontSize": {"magnitude": cfg.col0_base_font_size_pt, "unit": "PT"},
                    "foregroundColor": {"opaqueColor": {"rgbColor": _rgb(cfg.col0_base_color)}},
                },
                "fields": "fontFamily,fontSize,foregroundColor,bold",
            }
        })
        # Bold + dark for title portion
        title_len = len(key_data["title"])
        if title_len > 0:
            requests.append({
                "updateTextStyle": {
                    "objectId": table_id,
                    "cellLocation": {"rowIndex": i, "columnIndex": 0},
                    "textRange": {"type": "FIXED_RANGE", "startIndex": 0, "endIndex": title_len},
                    "style": {
                        "bold": True,
                        "fontSize": {"magnitude": cfg.col0_title_font_size_pt, "unit": "PT"},
                        "foregroundColor": {"opaqueColor": {"rgbColor": _rgb(cfg.col0_title_color)}},
                    },
                    "fields": "bold,fontSize,foregroundColor",
                }
            })
        # Subtitle
        if key_data["subtitle"]:
            sub_start = title_len + 1
            requests.append({
                "updateTextStyle": {
                    "objectId": table_id,
                    "cellLocation": {"rowIndex": i, "columnIndex": 0},
                    "textRange": {"type": "FIXED_RANGE", "startIndex": sub_start, "endIndex": sub_start + len(key_data["subtitle"])},
                    "style": {
                        "fontSize": {"magnitude": cfg.col0_subtitle_font_size_pt, "unit": "PT"},
                        "foregroundColor": {"opaqueColor": {"rgbColor": _rgb(cfg.col0_subtitle_color)}},
                    },
                    "fields": "fontSize,foregroundColor",
                }
            })
        requests.append({
            "updateParagraphStyle": {
                "objectId": table_id,
                "cellLocation": {"rowIndex": i, "columnIndex": 0},
                "textRange": {"type": "ALL"},
                "style": {
                    "spaceAbove": {"magnitude": 0, "unit": "PT"},
                    "spaceBelow": {"magnitude": 0, "unit": "PT"},
                },
                "fields": "spaceAbove,spaceBelow",
            }
        })

        # --- Columns 1 & 2: Vendor cells (icon + bold headline + details) ---
        for col_idx, vdata in [(1, db_data), (2, comp_data)]:
            cell_text = vdata["full_text"][:1200]
            if not cell_text:
                continue
            requests.append({
                "insertText": {
                    "objectId": table_id,
                    "cellLocation": {"rowIndex": i, "columnIndex": col_idx},
                    "insertionIndex": 0,
                    "text": cell_text,
                }
            })
            requests.append({
                "updateTableCellProperties": {
                    "objectId": table_id,
                    "tableRange": {
                        "location": {"rowIndex": i, "columnIndex": col_idx},
                        "rowSpan": 1, "columnSpan": 1,
                    },
                    "tableCellProperties": {"contentAlignment": "TOP"},
                    "fields": "contentAlignment",
                }
            })
            # Base style
            requests.append({
                "updateTextStyle": {
                    "objectId": table_id,
                    "cellLocation": {"rowIndex": i, "columnIndex": col_idx},
                    "textRange": {"type": "ALL"},
                    "style": {
                        "fontFamily": font,
                        "fontSize": {"magnitude": cfg.vendor_body_font_size_pt, "unit": "PT"},
                        "foregroundColor": {"opaqueColor": {"rgbColor": _rgb(cfg.vendor_text_color)}},
                    },
                    "fields": "fontFamily,fontSize,foregroundColor,bold",
                }
            })

            # Rating icon color
            icon = vdata["icon"]
            prefix = vdata["prefix"]
            headline = vdata["headline"]
            if icon and vdata["icon_color"]:
                icon_len = len(icon)
                requests.append({
                    "updateTextStyle": {
                        "objectId": table_id,
                        "cellLocation": {"rowIndex": i, "columnIndex": col_idx},
                        "textRange": {"type": "FIXED_RANGE", "startIndex": 0, "endIndex": icon_len},
                        "style": {
                            "bold": True,
                            "foregroundColor": {"opaqueColor": {"rgbColor": _rgb(vdata["icon_color"])}},
                        },
                        "fields": "bold,foregroundColor",
                    }
                })

            # Bold headline
            if headline:
                hl_start = len(prefix)
                hl_end = hl_start + len(headline)
                requests.append({
                    "updateTextStyle": {
                        "objectId": table_id,
                        "cellLocation": {"rowIndex": i, "columnIndex": col_idx},
                        "textRange": {"type": "FIXED_RANGE", "startIndex": hl_start, "endIndex": hl_end},
                        "style": {"bold": True},
                        "fields": "bold",
                    }
                })

            requests.append({
                "updateParagraphStyle": {
                    "objectId": table_id,
                    "cellLocation": {"rowIndex": i, "columnIndex": col_idx},
                    "textRange": {"type": "ALL"},
                    "style": {
                        "spaceAbove": {"magnitude": 0, "unit": "PT"},
                        "spaceBelow": {"magnitude": 0, "unit": "PT"},
                    },
                    "fields": "spaceAbove,spaceBelow",
                }
            })

    return requests


def add_title_slide(slides_service, presentation_id: str, title: str, subtitle: str, cfg: SlideConfig):
    font = cfg.font_family
    slide_id = f"slide_{uuid.uuid4().hex[:8]}"
    requests = [
        {
            "createSlide": {
                "objectId": slide_id,
                "insertionIndex": 0,
                "slideLayoutReference": {"predefinedLayout": "BLANK"},
            }
        },
        {
            "createShape": {
                "objectId": f"title_{uuid.uuid4().hex[:8]}",
                "shapeType": "TEXT_BOX",
                "elementProperties": {
                    "pageObjectId": slide_id,
                    "size": {
                        "width": {"magnitude": int(680 * EMU_PER_PX), "unit": "EMU"},
                        "height": {"magnitude": int(120 * EMU_PER_PX), "unit": "EMU"},
                    },
                    "transform": {
                        "scaleX": 1, "scaleY": 1,
                        "translateX": int(20 * EMU_PER_PX),
                        "translateY": int(cfg.title_slide_y_px * EMU_PER_PX),
                        "unit": "EMU",
                    },
                },
            }
        },
    ]
    title_shape_id = requests[1]["createShape"]["objectId"]
    requests.extend([
        {"insertText": {"objectId": title_shape_id, "insertionIndex": 0, "text": f"{title}\n{subtitle}"}},
        {
            "updateParagraphStyle": {
                "objectId": title_shape_id,
                "textRange": {"type": "ALL"},
                "style": {"alignment": "CENTER"},
                "fields": "alignment",
            }
        },
        {
            "updateTextStyle": {
                "objectId": title_shape_id,
                "textRange": {"type": "FIXED_RANGE", "startIndex": 0, "endIndex": len(title)},
                "style": {
                    "bold": True,
                    "fontFamily": font,
                    "fontSize": {"magnitude": cfg.title_slide_title_font_size_pt, "unit": "PT"},
                    "foregroundColor": {"opaqueColor": {"rgbColor": _rgb(cfg.title_slide_title_color)}},
                },
                "fields": "bold,fontFamily,fontSize,foregroundColor",
            }
        },
        {
            "updateTextStyle": {
                "objectId": title_shape_id,
                "textRange": {"type": "FIXED_RANGE", "startIndex": len(title), "endIndex": len(title) + 1 + len(subtitle)},
                "style": {
                    "fontFamily": font,
                    "fontSize": {"magnitude": cfg.title_slide_subtitle_font_size_pt, "unit": "PT"},
                    "foregroundColor": {"opaqueColor": {"rgbColor": _rgb(cfg.title_slide_subtitle_color)}},
                },
                "fields": "fontFamily,fontSize,foregroundColor",
            }
        },
    ])
    slides_service.presentations().batchUpdate(
        presentationId=presentation_id,
        body={"requests": requests},
    ).execute()


def add_category_slide(
    slides_service,
    presentation_id: str,
    category_title: str,
    competitor_name: str,
    rows: list[dict[str, Any]],
    insertion_index: int,
    cfg: SlideConfig,
):
    slide_id = f"slide_{uuid.uuid4().hex[:8]}"
    create_req = {
        "createSlide": {
            "objectId": slide_id,
            "insertionIndex": insertion_index,
            "slideLayoutReference": {"predefinedLayout": "BLANK"},
        }
    }
    requests = [create_req]
    requests.extend(
        build_slide_with_table_requests(
            slide_id=slide_id,
            title=category_title,
            competitor_name=competitor_name,
            rows=rows,
            cfg=cfg,
        )
    )
    slides_service.presentations().batchUpdate(
        presentationId=presentation_id,
        body={"requests": requests},
    ).execute()


def export_presentation_pdf(drive_service, presentation_id: str, out_pdf: Path):
    request = drive_service.files().export_media(
        fileId=presentation_id,
        mimeType="application/pdf",
    )
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fh = io.FileIO(out_pdf, "wb")
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.close()


def build_deck_from_battlecard(
    *,
    battlecard_id: str,
    output_dir: Path,
    cfg: SlideConfig,
    presentation_title: str | None = None,
) -> dict[str, Any]:
    slides, gen_info = load_battlecard_slides(battlecard_id)
    if not slides:
        raise RuntimeError(f"No slides found for battlecard_id={battlecard_id}")

    competitor_name = normalize_competitor_name(
        slides[0].get("competitor") or gen_info.get("competitor") or "Competitor"
    )
    product_area = (gen_info.get("product_area") or "Data Platform").strip()
    grouped = group_slides_by_category(slides)

    title = presentation_title or f"{competitor_name} Battlecard"
    subtitle = f"{product_area} \u2022 {len(slides)} differentiators \u2022 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"

    slides_service, drive_service = get_google_services()
    presentation_id = create_presentation(slides_service, title)
    add_title_slide(slides_service, presentation_id, title, subtitle, cfg)

    insertion_index = 1
    for category, category_rows in grouped.items():
        # Optionally strip parenthetical suffixes from category names
        display_category = strip_parenthetical(category) if cfg.strip_parentheticals else category

        chunks = chunk_rows(category_rows, cfg.rows_per_slide)
        for part_idx, chunk in enumerate(chunks, start=1):
            part_suffix = f" (Part {part_idx}/{len(chunks)})" if len(chunks) > 1 else ""
            add_category_slide(
                slides_service=slides_service,
                presentation_id=presentation_id,
                category_title=f"{display_category}{part_suffix}",
                competitor_name=competitor_name,
                rows=chunk,
                insertion_index=insertion_index,
                cfg=cfg,
            )
            insertion_index += 1

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"battlecard_{battlecard_id}_{timestamp}.pdf"
    export_presentation_pdf(drive_service, presentation_id, pdf_path)

    result = {
        "battlecard_id": battlecard_id,
        "competitor": competitor_name,
        "product_area": product_area,
        "presentation_id": presentation_id,
        "presentation_url": f"https://docs.google.com/presentation/d/{presentation_id}/edit",
        "pdf_path": str(pdf_path),
        "categories": len(grouped),
        "differentiators": len(slides),
        "rows_per_slide": cfg.rows_per_slide,
        "max_details_per_side": cfg.max_details_per_side,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    metadata_path = output_dir / f"battlecard_{battlecard_id}_{timestamp}.json"
    metadata_path.write_text(json.dumps(result, indent=2))
    result["metadata_path"] = str(metadata_path)
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Google Slides + PDF from a battlecard UUID in Lakebase.",
    )
    parser.add_argument("--battlecard-url", help="Battlecard URL (e.g. https://.../battlecard/<uuid>)")
    parser.add_argument("--battlecard-id", help="Battlecard UUID (alternative to --battlecard-url)")
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "runs" / "slides_local"), help="Directory for output PDF + metadata JSON")
    parser.add_argument("--presentation-title", default=None, help="Optional override for Google Slides deck title")

    # Content controls
    parser.add_argument("--rows-per-slide", type=int, default=4, help="Max differentiator rows per category slide (default: 4)")
    parser.add_argument("--max-details-per-side", type=int, default=2, help="Max detail bullets shown per vendor cell (default: 2)")
    parser.add_argument("--line-wrap", type=int, default=62, help="Character wrap width for bullet text (default: 62)")
    parser.add_argument("--max-desc-chars", type=int, default=140, help="Max chars for key diff description (default: 140)")
    parser.add_argument("--no-strip-parens", action="store_true", help="Keep parenthetical text in category titles")

    # Font
    parser.add_argument("--font", default="DM Sans", help="Font family (default: DM Sans)")

    # Font sizes (pt)
    parser.add_argument("--banner-font-size", type=float, default=8, help="Banner font size pt (default: 8)")
    parser.add_argument("--cat-title-font-size", type=float, default=19, help="Category title font size pt (default: 19)")
    parser.add_argument("--header-font-size", type=float, default=10, help="Table header font size pt (default: 10)")
    parser.add_argument("--col0-title-font-size", type=float, default=9, help="Col 0 title font size pt (default: 9)")
    parser.add_argument("--col0-subtitle-font-size", type=float, default=8, help="Col 0 subtitle font size pt (default: 8)")
    parser.add_argument("--vendor-font-size", type=float, default=8.5, help="Vendor cell body font size pt (default: 8.5)")
    parser.add_argument("--title-slide-title-font-size", type=float, default=32, help="Title slide title font size pt (default: 32)")
    parser.add_argument("--title-slide-subtitle-font-size", type=float, default=13, help="Title slide subtitle font size pt (default: 13)")

    # Colors
    parser.add_argument("--banner-bg-color", default="#FF5F46", help="Banner background color (default: #FF5F46)")
    parser.add_argument("--header-bg-color", default="#1A3A3A", help="Table header bg color (default: #1A3A3A)")
    parser.add_argument("--col0-bg-color", default="#F0F0F0", help="Col 0 background color (default: #F0F0F0)")
    parser.add_argument("--col0-title-color", default="#000000", help="Col 0 title text color (default: #000000)")
    parser.add_argument("--col0-subtitle-color", default="#777777", help="Col 0 subtitle text color (default: #777777)")
    parser.add_argument("--vendor-text-color", default="#111827", help="Vendor cell text color (default: #111827)")
    parser.add_argument("--cat-title-color", default="#1A3A3A", help="Category title color (default: #1A3A3A)")

    # Layout (px)
    parser.add_argument("--table-width", type=int, default=690, help="Table width px (default: 690)")
    parser.add_argument("--table-x", type=int, default=15, help="Table left offset px (default: 15)")
    parser.add_argument("--table-y", type=int, default=62, help="Table top offset px (default: 62)")
    parser.add_argument("--col-widths", default="170,260,260", help="Comma-separated column widths px (default: 170,260,260)")
    parser.add_argument("--cat-title-y", type=int, default=18, help="Category title Y offset px (default: 18)")
    parser.add_argument("--cat-title-height", type=int, default=44, help="Category title height px (default: 44)")

    args = parser.parse_args()

    if not args.battlecard_url and not args.battlecard_id:
        parser.error("Provide either --battlecard-url or --battlecard-id")
    return args


def config_from_args(args) -> SlideConfig:
    """Build SlideConfig from parsed CLI args."""
    col_widths = [int(x.strip()) for x in args.col_widths.split(",")]
    return SlideConfig(
        font_family=args.font,
        rows_per_slide=max(1, args.rows_per_slide),
        max_details_per_side=max(1, args.max_details_per_side),
        line_wrap=args.line_wrap,
        max_desc_chars=args.max_desc_chars,
        strip_parentheticals=not args.no_strip_parens,
        banner_font_size_pt=args.banner_font_size,
        banner_bg_color=args.banner_bg_color,
        cat_title_font_size_pt=args.cat_title_font_size,
        cat_title_color=args.cat_title_color,
        cat_title_y_px=args.cat_title_y,
        cat_title_height_px=args.cat_title_height,
        table_width_px=args.table_width,
        table_x_px=args.table_x,
        table_y_px=args.table_y,
        col_widths_px=col_widths,
        header_font_size_pt=args.header_font_size,
        header_bg_color=args.header_bg_color,
        col0_bg_color=args.col0_bg_color,
        col0_title_font_size_pt=args.col0_title_font_size,
        col0_title_color=args.col0_title_color,
        col0_subtitle_font_size_pt=args.col0_subtitle_font_size,
        col0_subtitle_color=args.col0_subtitle_color,
        vendor_body_font_size_pt=args.vendor_font_size,
        vendor_text_color=args.vendor_text_color,
        title_slide_title_font_size_pt=args.title_slide_title_font_size,
        title_slide_subtitle_font_size_pt=args.title_slide_subtitle_font_size,
    )


def main():
    args = parse_args()
    cfg = config_from_args(args)
    battlecard_raw = args.battlecard_id or args.battlecard_url
    battlecard_id = extract_battlecard_id(battlecard_raw)
    result = build_deck_from_battlecard(
        battlecard_id=battlecard_id,
        output_dir=Path(args.output_dir),
        cfg=cfg,
        presentation_title=args.presentation_title,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
