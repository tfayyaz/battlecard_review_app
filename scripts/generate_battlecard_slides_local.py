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


def _rgb(hex_color: str) -> dict[str, float]:
    value = hex_color.lstrip("#")
    return {
        "red": int(value[0:2], 16) / 255.0,
        "green": int(value[2:4], 16) / 255.0,
        "blue": int(value[4:6], 16) / 255.0,
    }


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


def format_key_diff_cell(slide: dict[str, Any], max_desc_chars: int = 140) -> str:
    title = (slide.get("key_differentiator") or "").strip()
    desc = (slide.get("description") or "").strip()
    if len(desc) > max_desc_chars:
        desc = desc[: max_desc_chars - 1].rstrip() + "..."
    return f"{title}\n{desc}" if desc else title


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
    max_details_per_side: int,
    line_wrap: int = 62,
) -> str:
    parts: list[str] = []
    if headline:
        parts.append(headline.strip())

    for line in _extract_detail_lines(detail_items, fallback_text, max_details_per_side):
        wrapped = textwrap.fill(line, width=line_wrap)
        parts.append(f"- {wrapped}")

    return "\n".join(parts).strip()


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
    max_details_per_side: int,
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []

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
                    "height": {"magnitude": int(12 * EMU_PER_PX), "unit": "EMU"},
                },
                "transform": {
                    "scaleX": 1,
                    "scaleY": 1,
                    "translateX": 0,
                    "translateY": 0,
                    "unit": "EMU",
                },
            },
        }
    })
    requests.append({
        "updateShapeProperties": {
            "objectId": banner_id,
            "shapeProperties": {
                "shapeBackgroundFill": {"solidFill": {"color": {"rgbColor": _rgb("#FF5F46")}}},
                "outline": {"propertyState": "NOT_RENDERED"},
            },
            "fields": "shapeBackgroundFill,outline",
        }
    })
    requests.append({"insertText": {"objectId": banner_id, "insertionIndex": 0, "text": "INTERNAL USE ONLY"}})
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
                "fontSize": {"magnitude": 8, "unit": "PT"},
                "foregroundColor": {"opaqueColor": {"rgbColor": _rgb("#FFFFFF")}},
            },
            "fields": "bold,fontSize,foregroundColor",
        }
    })

    # Title
    requests.append({
        "createShape": {
            "objectId": title_id,
            "shapeType": "TEXT_BOX",
            "elementProperties": {
                "pageObjectId": slide_id,
                "size": {
                    "width": {"magnitude": int(700 * EMU_PER_PX), "unit": "EMU"},
                    "height": {"magnitude": int(44 * EMU_PER_PX), "unit": "EMU"},
                },
                "transform": {
                    "scaleX": 1,
                    "scaleY": 1,
                    "translateX": int(10 * EMU_PER_PX),
                    "translateY": int(18 * EMU_PER_PX),
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
                "fontSize": {"magnitude": 19, "unit": "PT"},
                "foregroundColor": {"opaqueColor": {"rgbColor": _rgb("#1A3A3A")}},
            },
            "fields": "bold,fontSize,foregroundColor",
        }
    })

    # Table container
    row_count = len(rows) + 1
    requests.append({
        "createTable": {
            "objectId": table_id,
            "rows": row_count,
            "columns": 3,
            "elementProperties": {
                "pageObjectId": slide_id,
                "size": {
                    "width": {"magnitude": int(690 * EMU_PER_PX), "unit": "EMU"},
                    "height": {"magnitude": int(470 * EMU_PER_PX), "unit": "EMU"},
                },
                "transform": {
                    "scaleX": 1,
                    "scaleY": 1,
                    "translateX": int(15 * EMU_PER_PX),
                    "translateY": int(62 * EMU_PER_PX),
                    "unit": "EMU",
                },
            },
        }
    })

    # Column widths
    col_widths_px = [170, 260, 260]
    for col_idx, width_px in enumerate(col_widths_px):
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
                    "rowSpan": 1,
                    "columnSpan": 1,
                },
                "tableCellProperties": {
                    "tableCellBackgroundFill": {"solidFill": {"color": {"rgbColor": _rgb("#1A3A3A")}}}
                },
                "fields": "tableCellBackgroundFill",
            }
        })
        requests.append({
            "updateTextStyle": {
                "objectId": table_id,
                "cellLocation": {"rowIndex": 0, "columnIndex": col_idx},
                "textRange": {"type": "ALL"},
                "style": {
                    "bold": True,
                    "fontSize": {"magnitude": 10, "unit": "PT"},
                    "foregroundColor": {"opaqueColor": {"rgbColor": _rgb("#FFFFFF")}},
                },
                "fields": "bold,fontSize,foregroundColor",
            }
        })

    # Data rows
    for i, slide in enumerate(rows, start=1):
        key_text = format_key_diff_cell(slide)
        db_text = format_vendor_cell(
            headline=(slide.get("databricks_headline") or ""),
            detail_items=slide.get("databricks_detail_items") or [],
            fallback_text=(slide.get("databricks_details") or ""),
            max_details_per_side=max_details_per_side,
        )
        comp_text = format_vendor_cell(
            headline=(slide.get("fabric_headline") or ""),
            detail_items=slide.get("fabric_detail_items") or [],
            fallback_text=(slide.get("fabric_details") or ""),
            max_details_per_side=max_details_per_side,
        )
        for col_idx, cell_text in enumerate([key_text, db_text, comp_text]):
            text_value = (cell_text or "").strip()
            # Cap text size for predictable fit.
            text_value = text_value[:1200]
            requests.append({
                "insertText": {
                    "objectId": table_id,
                    "cellLocation": {"rowIndex": i, "columnIndex": col_idx},
                    "insertionIndex": 0,
                    "text": text_value,
                }
            })

            # First column styling
            if col_idx == 0:
                requests.append({
                    "updateTableCellProperties": {
                        "objectId": table_id,
                        "tableRange": {
                            "location": {"rowIndex": i, "columnIndex": 0},
                            "rowSpan": 1,
                            "columnSpan": 1,
                        },
                        "tableCellProperties": {
                            "tableCellBackgroundFill": {"solidFill": {"color": {"rgbColor": _rgb("#E8E8E8")}}}
                        },
                        "fields": "tableCellBackgroundFill",
                    }
                })

            requests.append({
                "updateTextStyle": {
                    "objectId": table_id,
                    "cellLocation": {"rowIndex": i, "columnIndex": col_idx},
                    "textRange": {"type": "ALL"},
                    "style": {
                        "fontSize": {"magnitude": 8.5, "unit": "PT"},
                        "foregroundColor": {
                            "opaqueColor": {
                                "rgbColor": _rgb("#000000" if col_idx == 0 else "#111827")
                            }
                        },
                    },
                    "fields": "fontSize,foregroundColor",
                }
            })

    return requests


def add_title_slide(slides_service, presentation_id: str, title: str, subtitle: str):
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
                        "scaleX": 1,
                        "scaleY": 1,
                        "translateX": int(20 * EMU_PER_PX),
                        "translateY": int(180 * EMU_PER_PX),
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
                    "fontSize": {"magnitude": 32, "unit": "PT"},
                    "foregroundColor": {"opaqueColor": {"rgbColor": _rgb("#1A3A3A")}},
                },
                "fields": "bold,fontSize,foregroundColor",
            }
        },
        {
            "updateTextStyle": {
                "objectId": title_shape_id,
                "textRange": {"type": "FIXED_RANGE", "startIndex": len(title), "endIndex": len(title) + 1 + len(subtitle)},
                "style": {
                    "fontSize": {"magnitude": 13, "unit": "PT"},
                    "foregroundColor": {"opaqueColor": {"rgbColor": _rgb("#4B5563")}},
                },
                "fields": "fontSize,foregroundColor",
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
    max_details_per_side: int,
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
            max_details_per_side=max_details_per_side,
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
    rows_per_slide: int,
    max_details_per_side: int,
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
    subtitle = f"{product_area} • {len(slides)} differentiators • {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"

    slides_service, drive_service = get_google_services()
    presentation_id = create_presentation(slides_service, title)
    add_title_slide(slides_service, presentation_id, title, subtitle)

    insertion_index = 1
    for category, category_rows in grouped.items():
        chunks = chunk_rows(category_rows, rows_per_slide)
        for part_idx, chunk in enumerate(chunks, start=1):
            part_suffix = f" (Part {part_idx}/{len(chunks)})" if len(chunks) > 1 else ""
            add_category_slide(
                slides_service=slides_service,
                presentation_id=presentation_id,
                category_title=f"{category}{part_suffix}",
                competitor_name=competitor_name,
                rows=chunk,
                insertion_index=insertion_index,
                max_details_per_side=max_details_per_side,
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
        "rows_per_slide": rows_per_slide,
        "max_details_per_side": max_details_per_side,
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
    parser.add_argument(
        "--battlecard-url",
        help="Battlecard URL (e.g. https://.../battlecard/<uuid>)",
    )
    parser.add_argument(
        "--battlecard-id",
        help="Battlecard UUID (alternative to --battlecard-url)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "runs" / "slides_local"),
        help="Directory for output PDF + metadata JSON",
    )
    parser.add_argument(
        "--rows-per-slide",
        type=int,
        default=4,
        help="Max differentiator rows per category slide (default: 4)",
    )
    parser.add_argument(
        "--max-details-per-side",
        type=int,
        default=2,
        help="Max detail bullets shown per vendor cell (default: 2)",
    )
    parser.add_argument(
        "--presentation-title",
        default=None,
        help="Optional override for Google Slides deck title",
    )
    args = parser.parse_args()

    if not args.battlecard_url and not args.battlecard_id:
        parser.error("Provide either --battlecard-url or --battlecard-id")
    return args


def main():
    args = parse_args()
    battlecard_raw = args.battlecard_id or args.battlecard_url
    battlecard_id = extract_battlecard_id(battlecard_raw)
    result = build_deck_from_battlecard(
        battlecard_id=battlecard_id,
        output_dir=Path(args.output_dir),
        rows_per_slide=max(1, int(args.rows_per_slide)),
        max_details_per_side=max(1, int(args.max_details_per_side)),
        presentation_title=args.presentation_title,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
