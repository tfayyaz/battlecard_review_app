#!/usr/bin/env python3
# /// script
# dependencies = [
#   "google-auth>=2.0.0",
#   "google-api-python-client>=2.0.0",
# ]
# ///
"""
Generate battlecard slides from JSONL file with vibrant colors.

Styling:
- First column: Light grey background (#e8e8e8) with dark text
- First column header: Red background (#FF5F46)
- Subtitles: Not bold
- Checkmarks colored: ✓✓ (bright green), ✓ (medium green), ✗ (red), ⚠ (orange)

Usage:
    uv run generate_battlecard_from_jsonl.py --jsonl battlecard_data.jsonl --go-link "go/fabric/battle"
"""

import argparse
import sys
import re
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build

sys.path.append('/Users/tahir.fayyaz/databricks-dev/compete_automation/auto-battle/.claude/skills/generate-battlecard-google-slides/scripts')
from slide_helpers import draw_table_from_rows, draw_title_slide
from slide_drawing import draw_executive_summary, draw_product_portfolio_grid


# COLOR PALETTE
COLORS = {
    'double_check': '#00D95F',  # Bright bold green for ✓✓
    'single_check': '#4CAF50',  # Medium green for ✓
    'cross': '#FF1744',         # Bright red for ✗
    'warning': '#FF9100'        # Bright orange for ⚠
}


def get_slides_service():
    """Get authenticated Google Slides service."""
    try:
        from google.auth import default
        credentials, project = default(
            scopes=['https://www.googleapis.com/auth/presentations',
                   'https://www.googleapis.com/auth/drive']
        )
        return build('slides', 'v1', credentials=credentials)
    except Exception as e:
        print(f"Error authenticating: {e}")
        print("Run: gcloud auth application-default login --scopes=https://www.googleapis.com/auth/presentations,https://www.googleapis.com/auth/drive")
        sys.exit(1)


def create_presentation(slides_svc, title):
    """Create a new Google Slides presentation."""
    presentation = {
        'title': title,
        'pageSize': {
            'width': {'magnitude': 9144000, 'unit': 'EMU'},
            'height': {'magnitude': 6858000, 'unit': 'EMU'}
        }
    }
    presentation = slides_svc.presentations().create(body=presentation).execute()
    presentation_id = presentation.get('presentationId')

    slides = slides_svc.presentations().get(presentationId=presentation_id).execute().get('slides', [])
    if slides:
        requests = [{'deleteObject': {'objectId': slides[0]['objectId']}}]
        slides_svc.presentations().batchUpdate(
            presentationId=presentation_id,
            body={'requests': requests}
        ).execute()

    return presentation_id


def hex_to_rgb(hex_color):
    """Convert hex color to RGB dict for Google Slides API."""
    hex_color = hex_color.lstrip('#')
    return {
        'red': int(hex_color[0:2], 16) / 255.0,
        'green': int(hex_color[2:4], 16) / 255.0,
        'blue': int(hex_color[4:6], 16) / 255.0
    }


def apply_first_column_styling(slides_svc, presentation_id, slide_index):
    """Apply light grey background and dark text to first column (except header)."""
    pres = slides_svc.presentations().get(presentationId=presentation_id).execute()
    slides = pres.get('slides', [])

    if slide_index >= len(slides):
        return

    slide = slides[slide_index]
    requests = []

    for element in slide.get('pageElements', []):
        if 'table' in element:
            table = element['table']
            table_id = element['objectId']
            num_rows = len(table.get('tableRows', []))

            # Apply to all first column cells except header (row 0)
            for row_idx in range(1, num_rows):
                # Light grey background
                requests.append({
                    'updateTableCellProperties': {
                        'objectId': table_id,
                        'tableRange': {
                            'location': {'rowIndex': row_idx, 'columnIndex': 0},
                            'rowSpan': 1,
                            'columnSpan': 1
                        },
                        'tableCellProperties': {
                            'tableCellBackgroundFill': {
                                'solidFill': {'color': {'rgbColor': hex_to_rgb('#e8e8e8')}}
                            }
                        },
                        'fields': 'tableCellBackgroundFill'
                    }
                })

                # Get cell text to find where subtitle starts
                cell = table['tableRows'][row_idx]['tableCells'][0]
                if 'text' in cell and 'textElements' in cell['text']:
                    full_text = ''
                    for text_elem in cell['text']['textElements']:
                        if 'textRun' in text_elem and 'content' in text_elem['textRun']:
                            full_text += text_elem['textRun']['content']

                    # Main text (before newline) - make it black and bold
                    newline_pos = full_text.find('\n')
                    if newline_pos > 0:
                        # Main text bold and black
                        requests.append({
                            'updateTextStyle': {
                                'objectId': table_id,
                                'cellLocation': {'rowIndex': row_idx, 'columnIndex': 0},
                                'textRange': {'type': 'FIXED_RANGE', 'startIndex': 0, 'endIndex': newline_pos},
                                'style': {
                                    'bold': True,
                                    'foregroundColor': {'opaqueColor': {'rgbColor': {'red': 0, 'green': 0, 'blue': 0}}}
                                },
                                'fields': 'bold,foregroundColor'
                            }
                        })

                        # Subtitle - NOT bold, dark grey, smaller
                        requests.append({
                            'updateTextStyle': {
                                'objectId': table_id,
                                'cellLocation': {'rowIndex': row_idx, 'columnIndex': 0},
                                'textRange': {'type': 'FIXED_RANGE', 'startIndex': newline_pos, 'endIndex': len(full_text)},
                                'style': {
                                    'bold': False,
                                    'fontSize': {'magnitude': 8, 'unit': 'PT'},
                                    'foregroundColor': {'opaqueColor': {'rgbColor': {'red': 0.3, 'green': 0.3, 'blue': 0.3}}}
                                },
                                'fields': 'bold,fontSize,foregroundColor'
                            }
                        })
                    else:
                        # No subtitle, just make text black and bold
                        requests.append({
                            'updateTextStyle': {
                                'objectId': table_id,
                                'cellLocation': {'rowIndex': row_idx, 'columnIndex': 0},
                                'textRange': {'type': 'ALL'},
                                'style': {
                                    'bold': True,
                                    'foregroundColor': {'opaqueColor': {'rgbColor': {'red': 0, 'green': 0, 'blue': 0}}}
                                },
                                'fields': 'bold,foregroundColor'
                            }
                        })

    if requests:
        batch_size = 50
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i+batch_size]
            slides_svc.presentations().batchUpdate(
                presentationId=presentation_id,
                body={'requests': batch}
            ).execute()
        print(f"  Applied first column styling")


def color_symbols_in_table(slides_svc, presentation_id, slide_index):
    """Add vibrant colors to checkmarks and symbols."""
    pres = slides_svc.presentations().get(presentationId=presentation_id).execute()
    slides = pres.get('slides', [])

    if slide_index >= len(slides):
        return

    slide = slides[slide_index]
    requests = []

    for element in slide.get('pageElements', []):
        if 'table' in element:
            table = element['table']
            table_id = element['objectId']

            for row_idx, row in enumerate(table.get('tableRows', [])):
                for col_idx, cell in enumerate(row.get('tableCells', [])):
                    if 'text' in cell and 'textElements' in cell['text']:
                        full_text = ''
                        for text_elem in cell['text']['textElements']:
                            if 'textRun' in text_elem and 'content' in text_elem['textRun']:
                                full_text += text_elem['textRun']['content']

                        patterns = [
                            (r'✓✓', COLORS['double_check']),
                            (r'(?<!✓)✓(?!✓)', COLORS['single_check']),
                            (r'✗', COLORS['cross']),
                            (r'⚠', COLORS['warning'])
                        ]

                        for pattern, color in patterns:
                            for match in re.finditer(pattern, full_text):
                                requests.append({
                                    'updateTextStyle': {
                                        'objectId': table_id,
                                        'cellLocation': {'rowIndex': row_idx, 'columnIndex': col_idx},
                                        'textRange': {
                                            'type': 'FIXED_RANGE',
                                            'startIndex': match.start(),
                                            'endIndex': match.end()
                                        },
                                        'style': {
                                            'foregroundColor': {'opaqueColor': {'rgbColor': hex_to_rgb(color)}},
                                            'bold': True
                                        },
                                        'fields': 'foregroundColor,bold'
                                    }
                                })

    if requests:
        batch_size = 50
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i+batch_size]
            slides_svc.presentations().batchUpdate(
                presentationId=presentation_id,
                body={'requests': batch}
            ).execute()
        print(f"  Applied colors to {len(requests)} symbols")


def add_speaker_notes(slides_svc, presentation_id, slide_index, notes_text):
    """Add speaker notes to a slide using the speakerNotesObjectId."""
    try:
        # Get presentation
        pres = slides_svc.presentations().get(presentationId=presentation_id).execute()
        slides = pres.get('slides', [])

        if slide_index >= len(slides):
            print(f"  Warning: slide_index {slide_index} out of range")
            return

        slide = slides[slide_index]

        # Get speaker notes object ID from slide's notes page properties
        slide_properties = slide.get('slideProperties', {})
        notes_page = slide_properties.get('notesPage', {})
        notes_properties = notes_page.get('notesProperties', {})
        speaker_notes_object_id = notes_properties.get('speakerNotesObjectId')

        if not speaker_notes_object_id:
            print(f"  Warning: No speakerNotesObjectId found for slide {slide_index}")
            return

        # Insert text using the speaker notes object ID
        # According to official docs: https://developers.google.com/workspace/slides/api/guides/notes
        # The API auto-creates the shape if it doesn't exist
        # Just insert text - no need to delete first since shape is empty
        requests = [
            {'insertText': {'objectId': speaker_notes_object_id, 'insertionIndex': 0, 'text': notes_text}}
        ]

        slides_svc.presentations().batchUpdate(
            presentationId=presentation_id,
            body={'requests': requests}
        ).execute()

        print(f"  Added speaker notes to slide {slide_index}")

    except Exception as e:
        print(f"  Error adding speaker notes to slide {slide_index}: {str(e)}")


def draw_product_portfolio_wrapper(slides_svc, presentation_id, slide_data, insert_at_index):
    """Wrapper to convert JSONL data to format expected by draw_product_portfolio_grid."""
    # Convert JSONL format to cards_data format expected by the skill function
    # JSONL has categories with competitor products only

    # Detect competitor key dynamically (fabric, snowflake, etc.)
    competitor_key = None
    if slide_data['categories']:
        for key in slide_data['categories'][0].keys():
            if key not in ['name']:
                competitor_key = key
                break

    cards_data = []
    for category in slide_data['categories']:
        header = category['name']
        # Build body with competitor products
        body_lines = []
        for prod in category.get(competitor_key, []):
            body_lines.append(f"**{prod['product']}**: {prod['description']}")
        body = "\n".join(body_lines)
        cards_data.append({"header": header, "body": body})

    # Call the skill function
    draw_product_portfolio_grid(
        slides_svc,
        presentation_id,
        cards_data,
        insert_at_index=insert_at_index,
        title_text=slide_data['title'],
        num_rows=2,
        num_cols=4,
        # Layout - match L200 category slides
        # NOTE: side_margin_px=20 for cards, title uses default 10px - intentional alignment
        side_margin_px=20,          # Cards left margin
        title_top_px=12,            # Standard title vertical position
        top_px=54,                  # Cards moved up 16pt (was 70)
        card_height_px=165,         # Shorter boxes (was 190)
        card_border_pt=0.5,         # Thin borders like L200 tables
        # Font settings - leave slide_font_family blank to use default "DM Sans"
        title_font_pt=20,           # Match L200 title font
        header_font_pt=12,
        body_font_pt=9,
        card_bg_color="#ffffff",
        card_border_color="#FF5F46",
        header_color="#FF5F46",
        body_color="#000000"
    )

    print(f"  Created product portfolio with {len(cards_data)} categories")


def draw_technical_summary_wrapper(slides_svc, presentation_id, slide_data, insert_at_index):
    """Wrapper to convert L100 technical summary JSONL to card grid format (3x1)."""
    # Convert JSONL format to cards_data format
    # JSONL has categories with weaknesses
    cards_data = []
    for category in slide_data['categories']:
        header = category['name']
        # Build body with weakness points
        body_lines = []
        for weakness in category['weaknesses']:
            heading = weakness['heading']
            text = weakness['text']
            # Format: "Heading. Text"
            body_lines.append(f"**{heading}** {text}")
        body = "\n\n".join(body_lines)
        cards_data.append({"header": header, "body": body})

    # Call the skill function with 3x1 grid
    draw_product_portfolio_grid(
        slides_svc,
        presentation_id,
        cards_data,
        insert_at_index=insert_at_index,
        title_text=slide_data['title'],
        num_rows=1,                 # Single row
        num_cols=3,                 # Three columns
        # Layout - match product portfolio style
        side_margin_px=20,
        title_top_px=12,
        top_px=54,                  # Cards start position
        card_height_px=420,         # Taller boxes for more content
        card_border_pt=0.5,
        title_font_pt=20,
        header_font_pt=12,
        body_font_pt=9,
        card_bg_color="#ffffff",
        card_border_color="#FF5F46",
        header_color="#FF5F46",
        body_color="#000000"
    )

    print(f"  Created technical summary with {len(cards_data)} categories")


def calculate_exec_summary_dimensions(talking_points, slide_h_px=540):
    """
    Calculate individual row heights based on content length.

    Returns dict with 'row_heights' (list) and 'row_gap_px'.
    """
    num_boxes = len(talking_points)

    # Constants - optimized for 3 boxes
    banner_h_px = 12
    title_area_px = 100  # Title + subtitle + intro + spacing
    bottom_margin_px = 12  # Reduced from 20
    font_pt = 10  # Reduced from 11
    line_height_multiplier = 1.4  # Standard line height
    internal_padding_px = 20  # Reduced from 30
    row_gap_px = 8  # Reduced from 12

    # Calculate available vertical space
    available_space = slide_h_px - banner_h_px - title_area_px - bottom_margin_px

    # Calculate height needed for each box individually
    box_heights = []
    for point in talking_points:
        # Quote (heading) - estimate lines based on character count
        # Typical heading is 15-25 chars, at ~50 chars per line
        heading = point['heading']
        quote_lines = max(1, len(heading) // 50 + (1 if len(heading) % 50 > 0 else 0))
        quote_height_px = quote_lines * font_pt * line_height_multiplier

        # Body - handle both bullets (old) and text (new paragraph format)
        # Assume ~90 chars per line for wider boxes (moved left)
        if 'bullets' in point:
            # Old bullet format
            total_body_lines = len(point['bullets'])  # Start with number of bullets
            for bullet in point['bullets']:
                wrapped_lines = max(0, (len(bullet) // 90))
                total_body_lines += wrapped_lines
        else:
            # New paragraph format
            text = point.get('text', '')
            total_body_lines = len(text) // 90 + (1 if len(text) % 90 > 0 else 0)

        body_height_px = total_body_lines * font_pt * line_height_multiplier

        # Total box height
        box_height = quote_height_px + body_height_px + internal_padding_px
        box_heights.append(box_height)

    # Reduce box heights by 30%
    box_heights = [h * 0.7 for h in box_heights]

    # Check if all boxes fit with fixed gaps
    total_height_needed = sum(box_heights) + (num_boxes - 1) * row_gap_px

    # If boxes don't fit, scale them down proportionally
    if total_height_needed > available_space:
        scale_factor = available_space / total_height_needed
        box_heights = [int(h * scale_factor) for h in box_heights]
        print(f"  Scaling boxes down by {scale_factor:.2f} to fit page")

    return {
        'row_heights': [int(h) for h in box_heights],
        'row_gap_px': row_gap_px
    }


def draw_executive_summary_custom(slides_svc, presentation_id, slide_data, insert_at_index):
    """Draw executive summary with variable-height rounded rectangles."""
    import uuid

    # Calculate optimal dimensions based on content
    dimensions = calculate_exec_summary_dimensions(slide_data['talking_points'])
    row_heights = dimensions['row_heights']
    row_gap_px = dimensions['row_gap_px']

    print(f"  Calculated dimensions: row_heights={row_heights}px, row_gap={row_gap_px}px")

    PX_TO_EMU = 12700
    slide_w_px = 720
    slide_h_px = 540

    # Create slide
    slide_id = f"slide_{insert_at_index}"
    requests = [{
        'createSlide': {
            'objectId': slide_id,
            'insertionIndex': insert_at_index,
            'slideLayoutReference': {'predefinedLayout': 'BLANK'}
        }
    }]

    slides_svc.presentations().batchUpdate(
        presentationId=presentation_id,
        body={'requests': requests}
    ).execute()

    # Set dark background
    requests = []
    bg_rgb = hex_to_rgb("#1b3239")
    requests.append({
        'updatePageProperties': {
            'objectId': slide_id,
            'pageProperties': {
                'pageBackgroundFill': {
                    'solidFill': {'color': {'rgbColor': bg_rgb}}
                }
            },
            'fields': 'pageBackgroundFill.solidFill.color'
        }
    })

    # Banner
    banner_id = f"banner_{uuid.uuid4().hex[:8]}"
    banner_h = 12
    requests.append({
        'createShape': {
            'objectId': banner_id,
            'shapeType': 'RECTANGLE',
            'elementProperties': {
                'pageObjectId': slide_id,
                'size': {
                    'width': {'magnitude': slide_w_px * PX_TO_EMU, 'unit': 'EMU'},
                    'height': {'magnitude': banner_h * PX_TO_EMU, 'unit': 'EMU'}
                },
                'transform': {
                    'scaleX': 1, 'scaleY': 1,
                    'translateX': 0, 'translateY': 0,
                    'unit': 'EMU'
                }
            }
        }
    })
    requests.append({
        'updateShapeProperties': {
            'objectId': banner_id,
            'shapeProperties': {
                'shapeBackgroundFill': {
                    'solidFill': {'color': {'rgbColor': hex_to_rgb('#FF5F46')}}
                },
                'outline': {'propertyState': 'NOT_RENDERED'}
            },
            'fields': 'shapeBackgroundFill,outline'
        }
    })
    requests.append({'insertText': {'objectId': banner_id, 'insertionIndex': 0, 'text': 'INTERNAL USE ONLY'}})
    requests.append({
        'updateParagraphStyle': {
            'objectId': banner_id,
            'textRange': {'type': 'ALL'},
            'style': {'alignment': 'CENTER'},
            'fields': 'alignment'
        }
    })
    requests.append({
        'updateTextStyle': {
            'objectId': banner_id,
            'textRange': {'type': 'ALL'},
            'style': {
                'fontFamily': 'Arial',
                'bold': True,
                'fontSize': {'magnitude': 8, 'unit': 'PT'},
                'foregroundColor': {'opaqueColor': {'rgbColor': {'red': 1, 'green': 1, 'blue': 1}}}
            },
            'fields': 'fontFamily,bold,fontSize,foregroundColor'
        }
    })

    # Title - use DM Sans, reduced size, tighter spacing
    title_id = f"title_{uuid.uuid4().hex[:8]}"
    title_top = banner_h + 10
    requests.append({
        'createShape': {
            'objectId': title_id,
            'shapeType': 'TEXT_BOX',
            'elementProperties': {
                'pageObjectId': slide_id,
                'size': {
                    'width': {'magnitude': (slide_w_px - 20) * PX_TO_EMU, 'unit': 'EMU'},
                    'height': {'magnitude': 32 * PX_TO_EMU, 'unit': 'EMU'}
                },
                'transform': {
                    'scaleX': 1, 'scaleY': 1,
                    'translateX': 10 * PX_TO_EMU,
                    'translateY': title_top * PX_TO_EMU,
                    'unit': 'EMU'
                }
            }
        }
    })
    requests.append({'insertText': {'objectId': title_id, 'insertionIndex': 0, 'text': 'EXECUTIVE SUMMARY'}})
    requests.append({
        'updateTextStyle': {
            'objectId': title_id,
            'textRange': {'type': 'ALL'},
            'style': {
                'fontFamily': 'DM Sans',
                'bold': True,
                'fontSize': {'magnitude': 24, 'unit': 'PT'},
                'foregroundColor': {'opaqueColor': {'rgbColor': {'red': 1, 'green': 1, 'blue': 1}}}
            },
            'fields': 'fontFamily,bold,fontSize,foregroundColor'
        }
    })

    # Subtitle: "Main talking points" - tighter spacing
    subtitle_id = f"subtitle_{uuid.uuid4().hex[:8]}"
    subtitle_top = title_top + 28
    requests.append({
        'createShape': {
            'objectId': subtitle_id,
            'shapeType': 'TEXT_BOX',
            'elementProperties': {
                'pageObjectId': slide_id,
                'size': {
                    'width': {'magnitude': (slide_w_px - 20) * PX_TO_EMU, 'unit': 'EMU'},
                    'height': {'magnitude': 24 * PX_TO_EMU, 'unit': 'EMU'}
                },
                'transform': {
                    'scaleX': 1, 'scaleY': 1,
                    'translateX': 10 * PX_TO_EMU,
                    'translateY': subtitle_top * PX_TO_EMU,
                    'unit': 'EMU'
                }
            }
        }
    })
    requests.append({'insertText': {'objectId': subtitle_id, 'insertionIndex': 0, 'text': 'Main talking points'}})
    requests.append({
        'updateTextStyle': {
            'objectId': subtitle_id,
            'textRange': {'type': 'ALL'},
            'style': {
                'fontFamily': 'DM Sans',
                'bold': True,
                'fontSize': {'magnitude': 16, 'unit': 'PT'},
                'foregroundColor': {'opaqueColor': {'rgbColor': hex_to_rgb('#FF5F46')}}
            },
            'fields': 'fontFamily,bold,fontSize,foregroundColor'
        }
    })

    # Intro line - tighter spacing
    intro_id = f"intro_{uuid.uuid4().hex[:8]}"
    intro_top = subtitle_top + 20
    requests.append({
        'createShape': {
            'objectId': intro_id,
            'shapeType': 'TEXT_BOX',
            'elementProperties': {
                'pageObjectId': slide_id,
                'size': {
                    'width': {'magnitude': (slide_w_px - 20) * PX_TO_EMU, 'unit': 'EMU'},
                    'height': {'magnitude': 24 * PX_TO_EMU, 'unit': 'EMU'}
                },
                'transform': {
                    'scaleX': 1, 'scaleY': 1,
                    'translateX': 10 * PX_TO_EMU,
                    'translateY': intro_top * PX_TO_EMU,
                    'unit': 'EMU'
                }
            }
        }
    })
    requests.append({'insertText': {'objectId': intro_id, 'insertionIndex': 0, 'text': "Should Azure's Lakehouse (Fabric, Purview, Azure AI Studio, Azure ML) come up as a competitor, reinforce these 3 things:"}})
    requests.append({
        'updateTextStyle': {
            'objectId': intro_id,
            'textRange': {'type': 'ALL'},
            'style': {
                'fontFamily': 'DM Sans',
                'bold': False,
                'fontSize': {'magnitude': 11, 'unit': 'PT'},
                'foregroundColor': {'opaqueColor': {'rgbColor': {'red': 1, 'green': 1, 'blue': 1}}}
            },
            'fields': 'fontFamily,bold,fontSize,foregroundColor'
        }
    })

    slides_svc.presentations().batchUpdate(
        presentationId=presentation_id,
        body={'requests': requests}
    ).execute()

    # Create numbered cards with variable heights - moved left 20px
    side_margin = 0  # Boxes moved to left edge
    card_width = slide_w_px - 2 * side_margin - 50  # Leave room for number circle
    circle_size = 40
    circle_margin = 12
    top_start = intro_top + 28  # Start boxes after intro line (tighter spacing)

    current_y = top_start

    for idx, (point, height) in enumerate(zip(slide_data['talking_points'], row_heights)):
        num = idx + 1

        # Create numbered circle
        circle_id = f"circle_{insert_at_index}_{idx}"
        requests = []
        requests.append({
            'createShape': {
                'objectId': circle_id,
                'shapeType': 'ELLIPSE',
                'elementProperties': {
                    'pageObjectId': slide_id,
                    'size': {
                        'width': {'magnitude': circle_size * PX_TO_EMU, 'unit': 'EMU'},
                        'height': {'magnitude': circle_size * PX_TO_EMU, 'unit': 'EMU'}
                    },
                    'transform': {
                        'scaleX': 1, 'scaleY': 1,
                        'translateX': (side_margin - 10) * PX_TO_EMU,
                        'translateY': (current_y + 12) * PX_TO_EMU,
                        'unit': 'EMU'
                    }
                }
            }
        })
        requests.append({
            'updateShapeProperties': {
                'objectId': circle_id,
                'shapeProperties': {
                    'shapeBackgroundFill': {
                        'solidFill': {'color': {'rgbColor': hex_to_rgb('#FF5F46')}}
                    },
                    'outline': {'propertyState': 'NOT_RENDERED'},
                    'contentAlignment': 'MIDDLE'
                },
                'fields': 'shapeBackgroundFill,outline,contentAlignment'
            }
        })
        requests.append({'insertText': {'objectId': circle_id, 'insertionIndex': 0, 'text': str(num)}})
        requests.append({
            'updateParagraphStyle': {
                'objectId': circle_id,
                'textRange': {'type': 'ALL'},
                'style': {'alignment': 'CENTER'},
                'fields': 'alignment'
            }
        })
        requests.append({
            'updateTextStyle': {
                'objectId': circle_id,
                'textRange': {'type': 'ALL'},
                'style': {
                    'fontFamily': 'Arial',
                    'bold': True,
                    'fontSize': {'magnitude': 18, 'unit': 'PT'},
                    'foregroundColor': {'opaqueColor': {'rgbColor': {'red': 1, 'green': 1, 'blue': 1}}}
                },
                'fields': 'fontFamily,bold,fontSize,foregroundColor'
            }
        })

        # Create rounded card
        card_id = f"card_{insert_at_index}_{idx}"
        card_left = side_margin + circle_size + circle_margin
        requests.append({
            'createShape': {
                'objectId': card_id,
                'shapeType': 'ROUND_RECTANGLE',
                'elementProperties': {
                    'pageObjectId': slide_id,
                    'size': {
                        'width': {'magnitude': card_width * PX_TO_EMU, 'unit': 'EMU'},
                        'height': {'magnitude': height * PX_TO_EMU, 'unit': 'EMU'}
                    },
                    'transform': {
                        'scaleX': 1, 'scaleY': 1,
                        'translateX': card_left * PX_TO_EMU,
                        'translateY': current_y * PX_TO_EMU,
                        'unit': 'EMU'
                    }
                }
            }
        })
        requests.append({
            'updateShapeProperties': {
                'objectId': card_id,
                'shapeProperties': {
                    'shapeBackgroundFill': {
                        'solidFill': {'color': {'rgbColor': hex_to_rgb('#2a4249')}}
                    },
                    'outline': {
                        'outlineFill': {'solidFill': {'color': {'rgbColor': hex_to_rgb('#FF5F46')}}},
                        'weight': {'magnitude': 1 * PX_TO_EMU, 'unit': 'EMU'},
                        'dashStyle': 'DASH'
                    }
                },
                'fields': 'shapeBackgroundFill,outline'
            }
        })

        # Add text - handle both bullets and paragraph format
        heading = point['heading']

        # Check if using new paragraph format or old bullet format
        if 'text' in point:
            # New paragraph format
            body_text = point['text']
            full_text = f"{heading} • {body_text}"
        else:
            # Old bullet format (fallback)
            bullets_text = "\n".join([f"• {bullet}" for bullet in point['bullets']])
            full_text = f"{heading} • {bullets_text}"

        requests.append({'insertText': {'objectId': card_id, 'insertionIndex': 0, 'text': full_text}})

        # Left-align all text in the box
        requests.append({
            'updateParagraphStyle': {
                'objectId': card_id,
                'textRange': {'type': 'ALL'},
                'style': {'alignment': 'START'},  # LEFT alignment
                'fields': 'alignment'
            }
        })

        # Style heading (first line) - orange italic, DM Sans
        heading_len = len(heading)
        requests.append({
            'updateTextStyle': {
                'objectId': card_id,
                'textRange': {'type': 'FIXED_RANGE', 'startIndex': 0, 'endIndex': heading_len},
                'style': {
                    'fontFamily': 'DM Sans',
                    'fontSize': {'magnitude': 10, 'unit': 'PT'},
                    'foregroundColor': {'opaqueColor': {'rgbColor': hex_to_rgb('#FF5F46')}},
                    'italic': True
                },
                'fields': 'fontFamily,fontSize,foregroundColor,italic'
            }
        })

        # Style body - white, DM Sans
        requests.append({
            'updateTextStyle': {
                'objectId': card_id,
                'textRange': {'type': 'FIXED_RANGE', 'startIndex': heading_len, 'endIndex': len(full_text)},
                'style': {
                    'fontFamily': 'DM Sans',
                    'fontSize': {'magnitude': 10, 'unit': 'PT'},
                    'foregroundColor': {'opaqueColor': {'rgbColor': {'red': 1, 'green': 1, 'blue': 1}}}
                },
                'fields': 'fontFamily,fontSize,foregroundColor'
            }
        })

        slides_svc.presentations().batchUpdate(
            presentationId=presentation_id,
            body={'requests': requests}
        ).execute()

        # Move to next row position
        current_y += height + row_gap_px

    print(f"  Created executive summary")


def draw_l100_platform(slides_svc, presentation_id, slide_data, insert_at_index):
    """Draw L100 platform comparison slide."""
    # Detect competitor key dynamically (fabric, snowflake, etc.)
    competitor_key = None
    competitor_name = "Competitor"
    if slide_data['dimensions']:
        for key in slide_data['dimensions'][0].keys():
            if key not in ['name', 'databricks']:
                competitor_key = key
                break

    # Map competitor key to display name
    competitor_display_names = {
        'fabric': 'Microsoft Fabric',
        'snowflake': 'Snowflake Data & AI Cloud'
    }
    competitor_name = competitor_display_names.get(competitor_key, competitor_key.title() if competitor_key else "Competitor")

    rows = [["Dimension", "Databricks", competitor_name]]

    for dim in slide_data['dimensions']:
        name = dim['name']
        databricks_text = f"{dim['databricks']['rating']} {dim['databricks']['text']}"
        competitor_text = f"{dim[competitor_key]['rating']} {dim[competitor_key]['text']}"
        rows.append([f"**{name}**", databricks_text, competitor_text])

    # NOTE: title_side_margin_px uses default 10px, side_margin_px=20 for table
    # This creates intentional 10px title / 20px content alignment
    draw_table_from_rows(
        slides_svc,
        presentation_id,
        rows,
        insert_at_index=insert_at_index,
        title_text=slide_data['title'],
        title_top_px=12,          # Standard title vertical position
        title_font_pt=20,         # Match L200 title font
        column_widths_pct=[25, 37, 38],
        header_bg_color="#1a3a3a",
        first_col_bg_color="#FF5F46",
        merge_duplicate_rows=False,
        table_border_color="#cccccc",
        table_border_pt=0.5,
        font_pt_header=10,
        font_pt_col1=10,
        font_pt_body=9,
        side_margin_px=20,        # Table left margin
        top_px=60,                # Table start position
        bottom_margin_px=10,
        add_banner=True,
        banner_text="INTERNAL USE ONLY",
        expand_to_bottom=False
    )

    # Override first column styling (grey background, dark text)
    apply_first_column_styling(slides_svc, presentation_id, insert_at_index)

    # Apply symbol colors
    color_symbols_in_table(slides_svc, presentation_id, insert_at_index)

    print(f"  Created L100 platform comparison")


def generate_battlecard_from_jsonl(jsonl_file, go_link=None):
    """Generate battlecard from JSONL file."""
    slides_svc = get_slides_service()

    # Read JSONL
    slides_data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            slides_data.append(json.loads(line))

    print(f"\n=== Generating Battlecard from JSONL ===")
    print(f"Input: {jsonl_file}")
    print(f"Slides to generate: {len(slides_data)}\n")

    # Get title from first slide or use default
    title = "Databricks Battlecard"
    if slides_data and slides_data[0].get('slide_type') == 'title':
        title = slides_data[0].get('title', title)

    # Create presentation
    presentation_id = create_presentation(slides_svc, title)
    print(f"Created presentation: {title}")
    print(f"URL: https://docs.google.com/presentation/d/{presentation_id}/edit\n")

    slide_index = 0
    for slide_data in slides_data:
        slide_type = slide_data['slide_type']

        if slide_type == 'title':
            print(f"Creating title slide...")
            draw_title_slide(
                slides_svc,
                presentation_id,
                insert_at_index=slide_index,
                title=slide_data['title'],
                subtitle=f"{slide_data['go_link']} | {slide_data.get('alt_go_link', '')}",
                slide_bg_color="#1a3a3a",
                footer_text=f"Last Updated: {slide_data['last_updated']}"
            )
            slide_index += 1

        elif slide_type == 'executive_summary':
            print(f"Creating executive summary...")
            draw_executive_summary_custom(slides_svc, presentation_id, slide_data, slide_index)

            # Add speaker notes if present
            if 'notes' in slide_data:
                add_speaker_notes(slides_svc, presentation_id, slide_index, slide_data['notes'])
                print(f"  Added speaker notes to executive summary")

            slide_index += 1

        elif slide_type == 'product_portfolio':
            print(f"Creating product portfolio...")
            draw_product_portfolio_wrapper(slides_svc, presentation_id, slide_data, slide_index)
            slide_index += 1

        elif slide_type == 'l100_technical_summary':
            print(f"Creating L100 technical summary...")
            draw_technical_summary_wrapper(slides_svc, presentation_id, slide_data, slide_index)

            # Add speaker notes if present
            if 'notes' in slide_data:
                add_speaker_notes(slides_svc, presentation_id, slide_index, slide_data['notes'])
                print(f"  Added speaker notes to technical summary")

            slide_index += 1

        elif slide_type == 'l100_platform':
            print(f"Creating L100 platform comparison...")
            draw_l100_platform(slides_svc, presentation_id, slide_data, slide_index)
            slide_index += 1

        elif slide_type == 'l200_category':
            print(f"Creating {slide_data['category']} slide...")

            # Detect competitor key dynamically (fabric, snowflake, etc.)
            competitor_key = None
            competitor_name = "Competitor"
            if slide_data['differentiators']:
                for key in slide_data['differentiators'][0].keys():
                    if key not in ['name', 'subtitle', 'databricks', 'sources']:
                        competitor_key = key
                        break

            # Map competitor key to display name
            competitor_display_names = {
                'fabric': 'Microsoft Fabric',
                'snowflake': 'Snowflake Data & AI Cloud'
            }
            competitor_name = competitor_display_names.get(competitor_key, competitor_key.title() if competitor_key else "Competitor")

            # Check if any differentiator has sources (for 4-column layout)
            has_sources = any('sources' in diff for diff in slide_data['differentiators'])

            # Build rows from JSONL data
            if has_sources:
                # 4-column layout with citations
                rows = [["Key Differentiators", "Databricks", competitor_name, "Fact-Check Sources"]]
            else:
                # 3-column layout (original)
                rows = [["Key Differentiators", "Databricks", competitor_name]]

            for diff in slide_data['differentiators']:
                name = diff['name']
                subtitle = diff.get('subtitle', '')
                if subtitle:
                    row_name = f"{name}\n{{size=8|{subtitle}}}"
                else:
                    row_name = name

                databricks_text = f"{diff['databricks']['rating']} {diff['databricks']['text']}"
                competitor_text = f"{diff[competitor_key]['rating']} {diff[competitor_key]['text']}"

                if has_sources:
                    # Format sources for 4th column
                    sources = diff.get('sources', ['Unknown'])
                    if isinstance(sources, list):
                        sources_text = '\n\n'.join(sources)
                    else:
                        sources_text = str(sources)

                    rows.append([row_name, databricks_text, competitor_text, sources_text])
                else:
                    rows.append([row_name, databricks_text, competitor_text])

            # NOTE: title_side_margin_px uses default 10px
            # For citations: Use negative side_margin to extend table beyond slide width
            # This makes the 4th column visible only by scrolling right
            draw_table_from_rows(
                slides_svc,
                presentation_id,
                rows,
                insert_at_index=slide_index,
                title_text=f"{slide_data['category']}: Key Differentiators",
                title_top_px=12,          # Standard title vertical position
                title_font_pt=20,
                column_widths_pct=[20, 32, 32, 16] if has_sources else [25, 37.5, 37.5],  # 4th column for citations
                header_bg_color="#1a3a3a",
                first_col_bg_color="#FF5F46",  # This will be overridden to grey
                merge_duplicate_rows=False,
                table_border_color="#cccccc",
                table_border_pt=0.5,  # CRITICAL: Thin borders
                font_pt_header=9,
                font_pt_col1=9,
                font_pt_body=8.5,  # Smaller comparison font
                side_margin_px=-80 if has_sources else 10,  # Negative margin extends table off-screen for citations
                top_px=46,  # Moved up 14pts from original (was 60, then 40, now 46)
                bottom_margin_px=10,
                add_banner=True,
                banner_text="INTERNAL USE ONLY",
                expand_to_bottom=False  # CRITICAL: Keep rows compact
            )

            # Override first column styling (light grey with dark text)
            apply_first_column_styling(slides_svc, presentation_id, slide_index)

            # Apply symbol colors
            color_symbols_in_table(slides_svc, presentation_id, slide_index)

            # Add speaker notes
            if 'notes' in slide_data:
                add_speaker_notes(slides_svc, presentation_id, slide_index, slide_data['notes'])

            slide_index += 1

    print(f"\n✓ Successfully generated battlecard!")
    print(f"  URL: https://docs.google.com/presentation/d/{presentation_id}/edit")
    print(f"  Slides: {slide_index}")

    return presentation_id


def main():
    parser = argparse.ArgumentParser(
        description="Generate battlecard from JSONL file",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--jsonl", type=str, required=True, help="Path to JSONL file")
    parser.add_argument("--go-link", type=str, default=None, help="Go-link for footer")

    args = parser.parse_args()

    try:
        presentation_id = generate_battlecard_from_jsonl(args.jsonl, args.go_link)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
