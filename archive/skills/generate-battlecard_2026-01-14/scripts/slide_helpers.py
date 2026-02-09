"""
Shared slide generation helpers.

Provides draw_table_from_rows function for creating formatted tables in Google Slides.
Supports inline markup: **bold**, {color|text}, {size=NN|text}
Supports per-cell backgrounds: <bg=color>
"""

import uuid
import re
from collections import OrderedDict

# --- Constants ---
PX_TO_EMU = 12700

def px(v):
    return int(v * PX_TO_EMU)


# --- Color handling ---
_COLOR_NAMES = {
    "red": (1, 0, 0),
    "orange": (1, 0.55, 0),
    "green": (0, 0.6, 0.2),
    "blue": (0.12, 0.47, 0.95),
    "gray": (0.38, 0.38, 0.38),
    "grey": (0.38, 0.38, 0.38),
    "black": (0, 0, 0),
    "white": (1, 1, 1),
    "teal": (0.1, 0.23, 0.23),
}


def _hex_to_rgb(s: str):
    s = s.lstrip('#')
    return (int(s[0:2], 16) / 255, int(s[2:4], 16) / 255, int(s[4:6], 16) / 255)


def _parse_color(token: str):
    token = token.strip()
    if token.startswith("#"):
        return _hex_to_rgb(token)
    return _COLOR_NAMES.get(token.lower())


def _ensure_rgb(color_like):
    """Accepts 'red', '#cccccc', (0.8,0.8,0.8), or (204,204,204). Returns floats 0..1."""
    if color_like is None:
        return None
    if isinstance(color_like, str):
        rgb = _parse_color(color_like)
        if rgb is None:
            raise ValueError(f"Unknown color: {color_like}")
        return rgb
    if isinstance(color_like, (list, tuple)) and len(color_like) == 3:
        r, g, b = color_like
        return (r / 255.0, g / 255.0, b / 255.0) if max(r, g, b) > 1 else (float(r), float(g), float(b))
    raise ValueError(f"Unsupported color value: {color_like}")


def _merge_rows_by_first_column(rows_data):
    """
    Merge rows that have the same value in the first column.
    Content in other columns is joined with newlines.
    First row (header) is never merged.
    """
    if len(rows_data) <= 1:
        return rows_data

    result = [rows_data[0]]
    merged = OrderedDict()

    for row in rows_data[1:]:
        key = str(row[0]) if row else ""
        clean_key = re.sub(r"<bg=[^>]+>", "", key, flags=re.IGNORECASE)
        clean_key = re.sub(r"\{[^|{}]+\|([^{}]*)\}", r"\1", clean_key)
        clean_key = re.sub(r"\*\*([^*]+)\*\*", r"\1", clean_key)
        clean_key = clean_key.strip()

        if clean_key not in merged:
            merged[clean_key] = {"original_key": row[0], "cols": [list(row[1:])]}
        else:
            merged[clean_key]["cols"].append(list(row[1:]))

    for clean_key, data in merged.items():
        original_key = data["original_key"]
        all_cols = data["cols"]

        if len(all_cols) == 1:
            result.append([original_key] + all_cols[0])
        else:
            num_cols = max(len(c) for c in all_cols)
            merged_row = [original_key]
            for col_idx in range(num_cols):
                col_values = []
                for row_cols in all_cols:
                    if col_idx < len(row_cols):
                        col_values.append(str(row_cols[col_idx]))
                merged_row.append("\n".join(col_values))
            result.append(merged_row)

    return result


def _normalize_color_size(s: str) -> str:
    """Normalize {color|text|size} -> {color|{size=NN|text}}"""
    pat = re.compile(r"\{([^|{}#]+|#[0-9A-Fa-f]{6})\|([^{}]*?)\|(\d{1,3})\}")
    return pat.sub(lambda m: "{%s|{size=%s|%s}}" % (m.group(1), m.group(3), m.group(2)), s)


def parse_markup(text: str):
    """Parse inline markup: **bold**, {color|text}, {size=NN|text}"""
    text = _normalize_color_size(text)
    i, n = 0, len(text)
    out, spans, stack = [], [], []

    def current_style():
        bold = any(s.get("bold") for s in stack) or None
        color, size = None, None
        for s in stack:
            if s.get("color") is not None:
                color = s["color"]
            if s.get("size") is not None:
                size = s["size"]
        return bold, color, size

    while i < n:
        if text.startswith("**", i):
            if stack and stack[-1].get("_m") == "b":
                stack.pop()
            else:
                stack.append({"bold": True, "_m": "b"})
            i += 2
            continue

        if text[i] == "{":
            m = re.match(r"\{([^|{}#]+|#[0-9A-Fa-f]{6}|size\s*=\s*\d+)\|", text[i:])
            if m:
                tok = m.group(1).strip()
                entry = {"_m": "span"}
                if tok.startswith("#"):
                    entry["color"] = _hex_to_rgb(tok)
                elif tok.lower().startswith("size"):
                    entry["size"] = int(re.search(r"\d+", tok).group())
                else:
                    entry["color"] = _parse_color(tok)
                stack.append(entry)
                i += m.end()
                continue

        if text[i] == "}":
            for k in range(len(stack) - 1, -1, -1):
                if stack[k].get("_m") == "span":
                    stack.pop(k)
                    break
            i += 1
            continue

        start = len(out)
        out.append(text[i])
        end = start + 1
        b, c, sz = current_style()
        spans.append({"start": start, "end": end, "bold": b, "color": c, "size": sz})
        i += 1

    merged = []
    for s in spans:
        if not merged:
            merged.append(s)
            continue
        last = merged[-1]
        if (last["end"] == s["start"] and last["bold"] == s["bold"] and
            last.get("color") == s.get("color") and last.get("size") == s.get("size")):
            last["end"] = s["end"]
        else:
            merged.append(s)
    return "".join(out), merged


def count_lines_in_cell(text: str, chars_per_line: int = 50) -> int:
    """
    Estimate number of lines a cell's text will take.
    Counts explicit newlines plus wrapping based on chars_per_line.
    """
    # Strip markup for counting
    plain = re.sub(r"\{[^|{}]+\|([^{}]*)\}", r"\1", text)
    plain = re.sub(r"\*\*([^*]+)\*\*", r"\1", plain)
    plain = re.sub(r"<bg=[^>]+>", "", plain, flags=re.IGNORECASE)

    total_lines = 0
    for line in plain.split("\n"):
        if not line:
            total_lines += 1
        else:
            total_lines += max(1, (len(line) + chars_per_line - 1) // chars_per_line)
    return total_lines


def estimate_row_lines(row: list, col_chars: list[int] = None) -> int:
    """
    Estimate max lines needed for a row based on all columns.
    col_chars specifies chars per line for each column.
    """
    if col_chars is None:
        col_chars = [25, 60, 60]  # Default: narrow first column, wider others

    max_lines = 1
    for i, cell in enumerate(row):
        chars = col_chars[i] if i < len(col_chars) else 50
        lines = count_lines_in_cell(str(cell), chars)
        max_lines = max(max_lines, lines)
    return max_lines


def draw_table_from_rows(
    slides_svc, presentation_id, rows_data,
    *, slide_index=None,
    insert_at_index=None,
    merge_duplicate_rows=True,
    side_margin_px=10, top_px=50, bottom_margin_px=4,
    slide_font_family="DM Sans",
    font_pt_header=10, font_pt_col1=10, font_pt_body=9,
    title_text=None, title_side_margin_px=10, title_top_px=12,
    title_font_pt=24, title_height_px=30, title_alignment="START",
    title_color="#1a3a3a",
    column_widths_pct=None,
    auto_row_heights=True,
    slide_bg_color="#ffffff",
    table_bg_color="#ffffff",
    table_border_pt=1,
    table_border_color="#e0e0e0",
    add_banner=True,
    banner_text="INTERNAL USE ONLY",
    banner_bg_color="#FF5F46",
    banner_text_color="white",
    banner_font_family="DM Sans",
    banner_font_pt=8,
    banner_height_px=9,
    banner_margin_px=0,
    table_object_id=None,
    header_bg_color=None,
    first_col_bg_color=None,
    header_colors=None,
    first_col_bold=False,
    expand_to_bottom=False,
):
    """
    Create a slide with a table from row data.

    Supports inline markup in cell text:
    - **bold text**
    - {red|red text}
    - {green|green text}
    - {#FF5F46|hex color text}
    - <bg=color> for cell background

    Args:
        slide_index: If specified, edit existing slide at this index. If None, create new slide.
        insert_at_index: When creating new slide, insert at this position.
        merge_duplicate_rows: If True, merge rows that have the same first column value.
        header_bg_color: Background color for header row (row 0) - used if header_colors not specified
        first_col_bg_color: Background color for first column (all rows except header)
        header_colors: List of dicts with per-column header styling: [{"bg": "#hex", "text": "#hex"}, ...]
        first_col_bold: If True, make all first column text bold (independent of background color)

    Returns:
        dict with page_id, slide_index, table_id, title_id, banner_id
    """
    if not rows_data:
        raise ValueError("rows_data must be non-empty")

    if merge_duplicate_rows:
        rows_data = _merge_rows_by_first_column(rows_data)

    cols = max(len(r) for r in rows_data)
    rows = len(rows_data)

    # Detect per-cell <bg=...>
    cell_bg = {}
    norm = []
    tag_re = re.compile(r"<bg=([^>]+)>", flags=re.IGNORECASE)
    for r_i, row in enumerate(rows_data):
        row = list(map(str, row))
        if len(row) < cols:
            row += [""] * (cols - len(row))
        cleaned = []
        for c_i, cell in enumerate(row):
            m = tag_re.search(cell)
            if m:
                rgb = _ensure_rgb(m.group(1))
                if rgb:
                    cell_bg[(r_i, c_i)] = rgb
                cell = tag_re.sub("", cell, count=1)
            cleaned.append(cell)
        norm.append(cleaned)

    pres = slides_svc.presentations().get(presentationId=presentation_id).execute()
    slides = pres.get("slides", [])
    w_emu = pres["pageSize"]["width"]["magnitude"]
    h_emu = pres["pageSize"]["height"]["magnitude"]
    slide_w_px = int(w_emu / PX_TO_EMU)
    slide_h_px = int(h_emu / PX_TO_EMU)

    # Create or select slide
    if slide_index is None:
        create_slide_request = {"createSlide": {}}
        if insert_at_index is not None:
            create_slide_request["createSlide"]["insertionIndex"] = insert_at_index

        slides_svc.presentations().batchUpdate(
            presentationId=presentation_id,
            body={"requests": [create_slide_request]}
        ).execute()

        pres = slides_svc.presentations().get(presentationId=presentation_id).execute()
        if insert_at_index is not None:
            page_id = pres["slides"][insert_at_index]["objectId"]
            result_slide_index = insert_at_index
        else:
            page_id = pres["slides"][-1]["objectId"]
            result_slide_index = len(pres["slides"]) - 1
    else:
        if not (0 <= slide_index < len(slides)):
            raise IndexError("slide_index out of range")
        page_id = slides[slide_index]["objectId"]
        result_slide_index = slide_index

    reqs = []

    # Slide background
    if slide_bg_color:
        rr, gg, bb = _ensure_rgb(slide_bg_color)
        reqs.append({
            "updatePageProperties": {
                "objectId": page_id,
                "pageProperties": {
                    "pageBackgroundFill": {
                        "solidFill": {"color": {"rgbColor": {"red": rr, "green": gg, "blue": bb}}}
                    }
                },
                "fields": "pageBackgroundFill.solidFill.color"
            }
        })

    # Banner
    banner_id = None
    if add_banner:
        banner_id = f"banner_{uuid.uuid4().hex[:8]}"
        bw_px = max(1, slide_w_px - 2 * banner_margin_px)
        bx_px = banner_margin_px
        by_px = 0

        reqs.append({
            "createShape": {
                "objectId": banner_id,
                "shapeType": "RECTANGLE",
                "elementProperties": {
                    "pageObjectId": page_id,
                    "size": {"width": {"magnitude": px(bw_px), "unit": "EMU"},
                             "height": {"magnitude": px(banner_height_px), "unit": "EMU"}},
                    "transform": {"scaleX": 1, "scaleY": 1,
                                  "translateX": px(bx_px), "translateY": px(by_px),
                                  "unit": "EMU"}
                }
            }
        })

        brr, bgg, bbb = _ensure_rgb(banner_bg_color)
        reqs.append({
            "updateShapeProperties": {
                "objectId": banner_id,
                "shapeProperties": {
                    "shapeBackgroundFill": {
                        "solidFill": {"color": {"rgbColor": {"red": brr, "green": bgg, "blue": bbb}}}
                    },
                    "outline": {
                        "outlineFill": {
                            "solidFill": {"color": {"rgbColor": {"red": brr, "green": bgg, "blue": bbb}}}
                        },
                        "weight": {"magnitude": 0.5, "unit": "PT"},
                        "dashStyle": "SOLID"
                    },
                    "contentAlignment": "MIDDLE"
                },
                "fields": (
                    "shapeBackgroundFill.solidFill.color,"
                    "outline.outlineFill.solidFill.color,"
                    "outline.weight,outline.dashStyle,"
                    "contentAlignment"
                )
            }
        })

        reqs.append({"insertText": {"objectId": banner_id, "insertionIndex": 0, "text": banner_text}})
        reqs.append({"updateParagraphStyle": {
            "objectId": banner_id, "textRange": {"type": "ALL"},
            "style": {"alignment": "CENTER"}, "fields": "alignment"}})
        tr, tg, tb = _ensure_rgb(banner_text_color)
        reqs.append({"updateTextStyle": {
            "objectId": banner_id, "textRange": {"type": "ALL"},
            "style": {
                "fontFamily": banner_font_family,
                "bold": True,
                "fontSize": {"magnitude": banner_font_pt, "unit": "PT"},
                "foregroundColor": {"opaqueColor": {"rgbColor": {"red": tr, "green": tg, "blue": tb}}}
            },
            "fields": "fontFamily,bold,fontSize,foregroundColor"}})

        top_px = max(top_px, banner_height_px + 8)

    # Layout for title/table
    x_px = side_margin_px
    w_px = max(1, slide_w_px - 2 * side_margin_px)
    y_px = top_px
    tiny_h_px = 1 if auto_row_heights else max(1, slide_h_px - top_px - bottom_margin_px)

    # Title
    title_id = None
    if title_text:
        title_id = f"title_{uuid.uuid4().hex[:10]}"
        reqs += [
            {
                "createShape": {
                    "objectId": title_id, "shapeType": "TEXT_BOX",
                    "elementProperties": {
                        "pageObjectId": page_id,
                        "size": {
                            "width": {"magnitude": px(max(1, slide_w_px - 2 * title_side_margin_px)), "unit": "EMU"},
                            "height": {"magnitude": px(title_height_px), "unit": "EMU"}
                        },
                        "transform": {
                            "scaleX": 1, "scaleY": 1,
                            "translateX": px(title_side_margin_px),
                            "translateY": px(title_top_px),
                            "unit": "EMU"
                        }
                    }
                }
            },
            {"insertText": {"objectId": title_id, "insertionIndex": 0, "text": title_text}},
            {"updateParagraphStyle": {
                "objectId": title_id, "textRange": {"type": "ALL"},
                "style": {"alignment": title_alignment},
                "fields": "alignment"
            }},
            {"updateTextStyle": {
                "objectId": title_id, "textRange": {"type": "ALL"},
                "style": {
                    "fontFamily": slide_font_family, "bold": True,
                    "fontSize": {"magnitude": title_font_pt, "unit": "PT"},
                    "foregroundColor": {"opaqueColor": {"rgbColor": {
                        "red": _ensure_rgb(title_color)[0],
                        "green": _ensure_rgb(title_color)[1],
                        "blue": _ensure_rgb(title_color)[2]
                    }}}
                },
                "fields": "fontFamily,bold,fontSize,foregroundColor"
            }},
            {"updateShapeProperties": {
                "objectId": title_id,
                "shapeProperties": {
                    "outline": {"propertyState": "NOT_RENDERED"},
                    "contentAlignment": "MIDDLE"
                },
                "fields": "outline.propertyState,contentAlignment"
            }}
        ]

    # Table shell
    table_id = table_object_id or f"tbl_{uuid.uuid4().hex[:10]}"
    reqs.append({"createTable": {
        "objectId": table_id,
        "elementProperties": {
            "pageObjectId": page_id,
            "size": {"width": {"magnitude": px(w_px), "unit": "EMU"},
                    "height": {"magnitude": px(tiny_h_px), "unit": "EMU"}},
            "transform": {"scaleX": 1, "scaleY": 1, "translateX": px(x_px), "translateY": px(y_px), "unit": "EMU"}},
        "rows": rows, "columns": cols
    }})

    # Table default background
    if table_bg_color:
        rr, gg, bb = _ensure_rgb(table_bg_color)
        reqs.append({
            "updateTableCellProperties": {
                "objectId": table_id,
                "tableRange": {"location": {"rowIndex": 0, "columnIndex": 0}, "rowSpan": rows, "columnSpan": cols},
                "tableCellProperties": {
                    "tableCellBackgroundFill": {"solidFill": {"color": {"rgbColor": {"red": rr, "green": gg, "blue": bb}}}}
                },
                "fields": "tableCellBackgroundFill.solidFill.color"
            }
        })

    # Header row background - per-column or uniform
    if header_colors:
        # Per-column header colors
        for c_idx, hc in enumerate(header_colors):
            if c_idx >= cols:
                break
            if hc and hc.get("bg"):
                hr, hg, hb = _ensure_rgb(hc["bg"])
                reqs.append({
                    "updateTableCellProperties": {
                        "objectId": table_id,
                        "tableRange": {"location": {"rowIndex": 0, "columnIndex": c_idx}, "rowSpan": 1, "columnSpan": 1},
                        "tableCellProperties": {
                            "tableCellBackgroundFill": {"solidFill": {"color": {"rgbColor": {"red": hr, "green": hg, "blue": hb}}}}
                        },
                        "fields": "tableCellBackgroundFill.solidFill.color"
                    }
                })
    elif header_bg_color:
        hr, hg, hb = _ensure_rgb(header_bg_color)
        reqs.append({
            "updateTableCellProperties": {
                "objectId": table_id,
                "tableRange": {"location": {"rowIndex": 0, "columnIndex": 0}, "rowSpan": 1, "columnSpan": cols},
                "tableCellProperties": {
                    "tableCellBackgroundFill": {"solidFill": {"color": {"rgbColor": {"red": hr, "green": hg, "blue": hb}}}}
                },
                "fields": "tableCellBackgroundFill.solidFill.color"
            }
        })

    # First column background (except header)
    if first_col_bg_color and rows > 1:
        fr, fg, fb = _ensure_rgb(first_col_bg_color)
        reqs.append({
            "updateTableCellProperties": {
                "objectId": table_id,
                "tableRange": {"location": {"rowIndex": 1, "columnIndex": 0}, "rowSpan": rows - 1, "columnSpan": 1},
                "tableCellProperties": {
                    "tableCellBackgroundFill": {"solidFill": {"color": {"rgbColor": {"red": fr, "green": fg, "blue": fb}}}}
                },
                "fields": "tableCellBackgroundFill.solidFill.color"
            }
        })

    # Fill text + styles + per-cell bg overrides
    for r in range(rows):
        for c in range(cols):
            plain, spans = parse_markup(norm[r][c])

            reqs.append({"insertText": {
                "objectId": table_id, "cellLocation": {"rowIndex": r, "columnIndex": c},
                "insertionIndex": 0, "text": plain}})

            reqs.append({"updateParagraphStyle": {
                "objectId": table_id, "cellLocation": {"rowIndex": r, "columnIndex": c},
                "textRange": {"type": "ALL"}, "style": {"alignment": "START"}, "fields": "alignment"}})

            base_sz = font_pt_header if r == 0 else (font_pt_col1 if c == 0 else font_pt_body)
            reqs.append({"updateTextStyle": {
                "objectId": table_id, "cellLocation": {"rowIndex": r, "columnIndex": c},
                "textRange": {"type": "ALL"},
                "style": {"fontFamily": slide_font_family, "fontSize": {"magnitude": base_sz, "unit": "PT"}},
                "fields": "fontFamily,fontSize"}})

            # Header row bold and text color (per-column or uniform white)
            if r == 0:
                if header_colors and c < len(header_colors) and header_colors[c]:
                    # Per-column header text color
                    text_color = header_colors[c].get("text", "#ffffff")
                    tr, tg, tb = _ensure_rgb(text_color)
                    reqs.append({"updateTextStyle": {
                        "objectId": table_id, "cellLocation": {"rowIndex": r, "columnIndex": c},
                        "textRange": {"type": "ALL"},
                        "style": {
                            "bold": True,
                            "foregroundColor": {"opaqueColor": {"rgbColor": {"red": tr, "green": tg, "blue": tb}}}
                        },
                        "fields": "bold,foregroundColor"}})
                elif header_bg_color:
                    reqs.append({"updateTextStyle": {
                        "objectId": table_id, "cellLocation": {"rowIndex": r, "columnIndex": c},
                        "textRange": {"type": "ALL"},
                        "style": {
                            "bold": True,
                            "foregroundColor": {"opaqueColor": {"rgbColor": {"red": 1, "green": 1, "blue": 1}}}
                        },
                        "fields": "bold,foregroundColor"}})

            # First column styling (bold and/or white text based on parameters)
            if c == 0 and r > 0:
                if first_col_bg_color:
                    # Has background color - make bold and white
                    reqs.append({"updateTextStyle": {
                        "objectId": table_id, "cellLocation": {"rowIndex": r, "columnIndex": c},
                        "textRange": {"type": "ALL"},
                        "style": {
                            "bold": True,
                            "foregroundColor": {"opaqueColor": {"rgbColor": {"red": 1, "green": 1, "blue": 1}}}
                        },
                        "fields": "bold,foregroundColor"}})
                elif first_col_bold:
                    # No background but want bold - just bold, black text
                    reqs.append({"updateTextStyle": {
                        "objectId": table_id, "cellLocation": {"rowIndex": r, "columnIndex": c},
                        "textRange": {"type": "ALL"},
                        "style": {"bold": True},
                        "fields": "bold"}})

            for sp in spans:
                style, fields = {}, []
                if sp.get("bold"):
                    style["bold"] = True
                    fields.append("bold")
                if sp.get("color") is not None:
                    rr, gg, bb = sp["color"]
                    style["foregroundColor"] = {"opaqueColor": {"rgbColor": {"red": rr, "green": gg, "blue": bb}}}
                    fields.append("foregroundColor")
                if sp.get("size") is not None:
                    style["fontSize"] = {"magnitude": int(sp["size"]), "unit": "PT"}
                    fields.append("fontSize")
                if fields:
                    reqs.append({"updateTextStyle": {
                        "objectId": table_id, "cellLocation": {"rowIndex": r, "columnIndex": c},
                        "textRange": {"type": "FIXED_RANGE", "startIndex": sp["start"], "endIndex": sp["end"]},
                        "style": style, "fields": ",".join(fields)}})

            if (r, c) in cell_bg:
                rr, gg, bb = cell_bg[(r, c)]
                reqs.append({
                    "updateTableCellProperties": {
                        "objectId": table_id,
                        "tableRange": {"location": {"rowIndex": r, "columnIndex": c}, "rowSpan": 1, "columnSpan": 1},
                        "tableCellProperties": {
                            "tableCellBackgroundFill": {"solidFill": {"color": {"rgbColor": {"red": rr, "green": gg, "blue": bb}}}}
                        },
                        "fields": "tableCellBackgroundFill.solidFill.color"
                    }
                })
            else:
                reqs.append({"updateTableCellProperties": {
                    "objectId": table_id,
                    "tableRange": {"location": {"rowIndex": r, "columnIndex": c}, "rowSpan": 1, "columnSpan": 1},
                    "tableCellProperties": {"contentAlignment": "TOP"},
                    "fields": "contentAlignment"}})

    # Column widths (%)
    total_w_emu = px(w_px)
    if column_widths_pct is None:
        base = total_w_emu // cols
        col_w = [base] * cols
        col_w[-1] = total_w_emu - base * (cols - 1)
    else:
        if len(column_widths_pct) != cols:
            raise ValueError(f"column_widths_pct must have {cols} values")
        vals = [max(0.0, float(p)) for p in column_widths_pct]
        s = sum(vals)
        if s <= 0:
            raise ValueError("column_widths_pct must sum to > 0")
        pct = [p * 100.0 / s for p in vals] if abs(s - 100) > 1e-6 else vals
        col_w = []
        acc = 0
        for i, p in enumerate(pct):
            if i < cols - 1:
                w = int(round(total_w_emu * (p / 100.0)))
                col_w.append(w)
                acc += w
            else:
                col_w.append(total_w_emu - acc)
    for c, w in enumerate(col_w):
        reqs.append({"updateTableColumnProperties": {
            "objectId": table_id, "columnIndices": [c],
            "tableColumnProperties": {"columnWidth": {"magnitude": int(w), "unit": "EMU"}},
            "fields": "columnWidth"}})

    # Row heights - expand table to fill available space
    if expand_to_bottom and rows > 0:
        # Calculate available height from table start to bottom margin
        available_h_px = slide_h_px - y_px - bottom_margin_px
        if available_h_px > 0:
            row_h_px = available_h_px / rows
            for r in range(rows):
                reqs.append({"updateTableRowProperties": {
                    "objectId": table_id,
                    "rowIndices": [r],
                    "tableRowProperties": {
                        "minRowHeight": {"magnitude": px(row_h_px), "unit": "EMU"}
                    },
                    "fields": "minRowHeight"
                }})

    # Borders
    br_r, br_g, br_b = _ensure_rgb(table_border_color)
    border_fill = {"solidFill": {"color": {"rgbColor": {"red": br_r, "green": br_g, "blue": br_b}}}}
    weight = {"magnitude": table_border_pt, "unit": "PT"}

    def b(pos, r0, c0, rs, cs):
        return {"updateTableBorderProperties": {
            "objectId": table_id,
            "tableRange": {"location": {"rowIndex": r0, "columnIndex": c0}, "rowSpan": rs, "columnSpan": cs},
            "borderPosition": pos,
            "tableBorderProperties": {"tableBorderFill": border_fill, "dashStyle": "SOLID", "weight": weight},
            "fields": "tableBorderFill.solidFill.color,dashStyle,weight"}}

    reqs += [b("TOP", 0, 0, 1, cols), b("BOTTOM", rows - 1, 0, 1, cols),
             b("LEFT", 0, 0, rows, 1), b("RIGHT", 0, cols - 1, rows, 1),
             b("INNER_HORIZONTAL", 0, 0, rows, cols), b("INNER_VERTICAL", 0, 0, rows, cols)]

    # Execute
    slides_svc.presentations().batchUpdate(
        presentationId=presentation_id, body={"requests": reqs}
    ).execute()

    return {
        "page_id": page_id,
        "slide_index": result_slide_index,
        "table_id": table_id,
        "title_id": title_id,
        "banner_id": banner_id
    }


def draw_title_slide(
    slides_svc, presentation_id,
    *,
    slide_index=None,
    insert_at_index=None,
    # Content
    subtitle=None,           # Smaller text above title (e.g. "Dataproc")
    title=None,              # Main title text (e.g. "Executive Summary")
    # Styling
    slide_bg_color="#FF5F46",
    text_color="#ffffff",
    slide_font_family="DM Sans",
    subtitle_font_pt=36,
    subtitle_bold=False,
    title_font_pt=60,
    title_bold=True,
    line_spacing_px=20,      # Space between subtitle and title
    # Footer
    add_footer=True,
    footer_text="©2024 Databricks Inc. — All rights reserved",
    footer_font_pt=10,
    footer_color=None,       # Defaults to text_color
    footer_bottom_px=20,
    footer_side_margin_px=30,
):
    """
    Create a simple title slide with centered text.

    Args:
        subtitle: Smaller text above the main title
        title: Main title text (larger, typically bold)
        slide_bg_color: Background color
        text_color: Color for title/subtitle text

    Returns:
        dict with page_id, slide_index, subtitle_id, title_id, footer_id
    """

    # Get presentation dimensions
    pres = slides_svc.presentations().get(presentationId=presentation_id).execute()
    slides = pres.get("slides", [])
    w_emu = pres["pageSize"]["width"]["magnitude"]
    h_emu = pres["pageSize"]["height"]["magnitude"]
    slide_w_px = int(w_emu / PX_TO_EMU)
    slide_h_px = int(h_emu / PX_TO_EMU)

    # Create or select slide
    if slide_index is None:
        create_slide_request = {"createSlide": {}}
        if insert_at_index is not None:
            create_slide_request["createSlide"]["insertionIndex"] = insert_at_index

        slides_svc.presentations().batchUpdate(
            presentationId=presentation_id,
            body={"requests": [create_slide_request]}
        ).execute()

        pres = slides_svc.presentations().get(presentationId=presentation_id).execute()
        if insert_at_index is not None:
            page_id = pres["slides"][insert_at_index]["objectId"]
            result_slide_index = insert_at_index
        else:
            page_id = pres["slides"][-1]["objectId"]
            result_slide_index = len(pres["slides"]) - 1
    else:
        if not (0 <= slide_index < len(slides)):
            raise IndexError("slide_index out of range")
        page_id = slides[slide_index]["objectId"]
        result_slide_index = slide_index

    reqs = []

    # Slide background
    if slide_bg_color:
        rr, gg, bb = _ensure_rgb(slide_bg_color)
        reqs.append({
            "updatePageProperties": {
                "objectId": page_id,
                "pageProperties": {
                    "pageBackgroundFill": {
                        "solidFill": {"color": {"rgbColor": {"red": rr, "green": gg, "blue": bb}}}
                    }
                },
                "fields": "pageBackgroundFill.solidFill.color"
            }
        })

    # Calculate vertical positioning for centered text
    subtitle_height_px = subtitle_font_pt * 1.5 if subtitle else 0
    title_height_px = title_font_pt * 1.5 if title else 0
    total_height = subtitle_height_px + line_spacing_px + title_height_px
    start_y = (slide_h_px - total_height) // 2

    text_margin_px = 50
    text_width_px = slide_w_px - 2 * text_margin_px

    subtitle_id = None
    title_id = None

    # --- Subtitle (smaller text above title) ---
    if subtitle:
        subtitle_id = f"subtitle_{uuid.uuid4().hex[:8]}"
        subtitle_y = start_y

        reqs.append({
            "createShape": {
                "objectId": subtitle_id,
                "shapeType": "TEXT_BOX",
                "elementProperties": {
                    "pageObjectId": page_id,
                    "size": {
                        "width": {"magnitude": px(text_width_px), "unit": "EMU"},
                        "height": {"magnitude": px(subtitle_height_px), "unit": "EMU"}
                    },
                    "transform": {
                        "scaleX": 1, "scaleY": 1,
                        "translateX": px(text_margin_px),
                        "translateY": px(subtitle_y),
                        "unit": "EMU"
                    }
                }
            }
        })

        reqs.append({"insertText": {"objectId": subtitle_id, "insertionIndex": 0, "text": subtitle}})

        reqs.append({
            "updateParagraphStyle": {
                "objectId": subtitle_id,
                "textRange": {"type": "ALL"},
                "style": {"alignment": "CENTER"},
                "fields": "alignment"
            }
        })

        tr, tg, tb = _ensure_rgb(text_color)
        reqs.append({
            "updateTextStyle": {
                "objectId": subtitle_id,
                "textRange": {"type": "ALL"},
                "style": {
                    "fontFamily": slide_font_family,
                    "bold": subtitle_bold,
                    "fontSize": {"magnitude": subtitle_font_pt, "unit": "PT"},
                    "foregroundColor": {"opaqueColor": {"rgbColor": {"red": tr, "green": tg, "blue": tb}}}
                },
                "fields": "fontFamily,bold,fontSize,foregroundColor"
            }
        })

        reqs.append({
            "updateShapeProperties": {
                "objectId": subtitle_id,
                "shapeProperties": {"outline": {"propertyState": "NOT_RENDERED"}},
                "fields": "outline.propertyState"
            }
        })

    # --- Title (main large text) ---
    if title:
        title_id = f"title_{uuid.uuid4().hex[:8]}"
        title_y = start_y + subtitle_height_px + line_spacing_px if subtitle else start_y

        reqs.append({
            "createShape": {
                "objectId": title_id,
                "shapeType": "TEXT_BOX",
                "elementProperties": {
                    "pageObjectId": page_id,
                    "size": {
                        "width": {"magnitude": px(text_width_px), "unit": "EMU"},
                        "height": {"magnitude": px(title_height_px), "unit": "EMU"}
                    },
                    "transform": {
                        "scaleX": 1, "scaleY": 1,
                        "translateX": px(text_margin_px),
                        "translateY": px(title_y),
                        "unit": "EMU"
                    }
                }
            }
        })

        reqs.append({"insertText": {"objectId": title_id, "insertionIndex": 0, "text": title}})

        reqs.append({
            "updateParagraphStyle": {
                "objectId": title_id,
                "textRange": {"type": "ALL"},
                "style": {"alignment": "CENTER"},
                "fields": "alignment"
            }
        })

        tr, tg, tb = _ensure_rgb(text_color)
        reqs.append({
            "updateTextStyle": {
                "objectId": title_id,
                "textRange": {"type": "ALL"},
                "style": {
                    "fontFamily": slide_font_family,
                    "bold": title_bold,
                    "fontSize": {"magnitude": title_font_pt, "unit": "PT"},
                    "foregroundColor": {"opaqueColor": {"rgbColor": {"red": tr, "green": tg, "blue": tb}}}
                },
                "fields": "fontFamily,bold,fontSize,foregroundColor"
            }
        })

        reqs.append({
            "updateShapeProperties": {
                "objectId": title_id,
                "shapeProperties": {"outline": {"propertyState": "NOT_RENDERED"}},
                "fields": "outline.propertyState"
            }
        })

    # --- Footer ---
    footer_id = None
    if add_footer:
        footer_id = f"footer_{uuid.uuid4().hex[:8]}"
        footer_height_px = footer_font_pt * 2

        reqs.append({
            "createShape": {
                "objectId": footer_id,
                "shapeType": "TEXT_BOX",
                "elementProperties": {
                    "pageObjectId": page_id,
                    "size": {
                        "width": {"magnitude": px(slide_w_px - 2 * footer_side_margin_px), "unit": "EMU"},
                        "height": {"magnitude": px(footer_height_px), "unit": "EMU"}
                    },
                    "transform": {
                        "scaleX": 1, "scaleY": 1,
                        "translateX": px(footer_side_margin_px),
                        "translateY": px(slide_h_px - footer_bottom_px - footer_height_px),
                        "unit": "EMU"
                    }
                }
            }
        })

        reqs.append({"insertText": {"objectId": footer_id, "insertionIndex": 0, "text": footer_text}})

        reqs.append({
            "updateParagraphStyle": {
                "objectId": footer_id,
                "textRange": {"type": "ALL"},
                "style": {"alignment": "START"},
                "fields": "alignment"
            }
        })

        fc = footer_color or text_color
        fr, fg, fb = _ensure_rgb(fc)
        reqs.append({
            "updateTextStyle": {
                "objectId": footer_id,
                "textRange": {"type": "ALL"},
                "style": {
                    "fontFamily": slide_font_family,
                    "fontSize": {"magnitude": footer_font_pt, "unit": "PT"},
                    "foregroundColor": {"opaqueColor": {"rgbColor": {"red": fr, "green": fg, "blue": fb}}}
                },
                "fields": "fontFamily,fontSize,foregroundColor"
            }
        })

        reqs.append({
            "updateShapeProperties": {
                "objectId": footer_id,
                "shapeProperties": {"outline": {"propertyState": "NOT_RENDERED"}},
                "fields": "outline.propertyState"
            }
        })

    # Execute
    slides_svc.presentations().batchUpdate(
        presentationId=presentation_id,
        body={"requests": reqs}
    ).execute()

    return {
        "page_id": page_id,
        "slide_index": result_slide_index,
        "subtitle_id": subtitle_id,
        "title_id": title_id,
        "footer_id": footer_id
    }
