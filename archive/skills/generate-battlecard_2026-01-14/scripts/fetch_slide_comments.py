# /// script
# dependencies = [
#   "google-auth>=2.0.0",
#   "google-auth-oauthlib>=1.0.0",
#   "google-api-python-client>=2.0.0",
# ]
# ///
"""
fetch_slide_comments.py - Fetch comments from Google Slides presentations

Uses the Google Drive API to retrieve comments and their anchored text
from Google Slides presentations.

Usage:
    uv run python fetch_slide_comments.py --presentation-id <ID>
    uv run python fetch_slide_comments.py --url <SLIDES_URL>
"""

import argparse
import json
import re
import google.auth
from googleapiclient.discovery import build


def extract_presentation_id(url_or_id: str) -> str:
    """Extract presentation ID from URL or return as-is if already an ID."""
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url_or_id)
    if match:
        return match.group(1)
    return url_or_id


def fetch_comments(presentation_id: str) -> list[dict]:
    """Fetch all comments from a Google Slides presentation using Drive API."""
    creds, project = google.auth.default(
        scopes=[
            'https://www.googleapis.com/auth/drive.readonly',
            'https://www.googleapis.com/auth/presentations.readonly',
        ]
    )

    drive_service = build('drive', 'v3', credentials=creds)

    response = drive_service.comments().list(
        fileId=presentation_id,
        fields='comments(id,content,anchor,quotedFileContent,author,createdTime,modifiedTime,resolved,replies)'
    ).execute()

    return response.get('comments', [])


def extract_text_from_element(element: dict) -> str:
    """Extract all text from a page element."""
    text_parts = []

    if 'shape' in element:
        text_content = element['shape'].get('text', {})
        for text_element in text_content.get('textElements', []):
            if 'textRun' in text_element:
                text_parts.append(text_element['textRun'].get('content', ''))

    if 'table' in element:
        table = element['table']
        for row in table.get('tableRows', []):
            for cell in row.get('tableCells', []):
                cell_text = cell.get('text', {})
                for text_element in cell_text.get('textElements', []):
                    if 'textRun' in text_element:
                        text_parts.append(text_element['textRun'].get('content', ''))

    return ''.join(text_parts)


def extract_table_structure(element: dict) -> dict:
    """Extract table structure with rows and columns."""
    if 'table' not in element:
        return None

    table = element['table']
    rows = []

    for row_idx, row in enumerate(table.get('tableRows', [])):
        row_data = []
        for cell in row.get('tableCells', []):
            cell_text = []
            for text_element in cell.get('text', {}).get('textElements', []):
                if 'textRun' in text_element:
                    cell_text.append(text_element['textRun'].get('content', ''))
            row_data.append(''.join(cell_text).strip())
        rows.append(row_data)

    return {
        'rows': len(rows),
        'cols': len(rows[0]) if rows else 0,
        'data': rows
    }


def fetch_detailed_slide_info(presentation_id: str) -> dict:
    """Fetch detailed slide information including all elements and their IDs."""
    creds, project = google.auth.default(
        scopes=[
            'https://www.googleapis.com/auth/presentations.readonly',
        ]
    )

    slides_service = build('slides', 'v1', credentials=creds)
    presentation = slides_service.presentations().get(
        presentationId=presentation_id
    ).execute()

    slide_info = {}
    element_to_slide = {}

    for i, slide in enumerate(presentation.get('slides', []), 1):
        slide_id = slide.get('objectId')

        # Extract slide title and use case from elements
        slide_title = None
        use_case = None
        critical_diff = None
        slide_type = None
        tables = []

        for element in slide.get('pageElements', []):
            element_id = element.get('objectId', '')
            element_to_slide[element_id] = slide_id

            # Check for title shapes
            if 'shape' in element:
                shape = element['shape']
                placeholder = shape.get('placeholder', {})
                placeholder_type = placeholder.get('type', '')

                text = extract_text_from_element(element).strip()

                # Title detection
                if placeholder_type in ['TITLE', 'CENTERED_TITLE'] or 'title' in element_id.lower():
                    slide_title = text

                    # Parse slide title for use case and type
                    # Format: "L200 | Data Engineering vs Microsoft Fabric"
                    # Format: "L300 | Data Engineering (1/4)"
                    if text:
                        if 'L200' in text:
                            slide_type = 'L200'
                            match = re.search(r'L200\s*\|\s*(.+?)\s+vs\s+', text)
                            if match:
                                use_case = match.group(1).strip()
                        elif 'L300' in text:
                            slide_type = 'L300'
                            match = re.search(r'L300\s*\|\s*(.+?)\s*\(', text)
                            if match:
                                use_case = match.group(1).strip()
                        elif 'Executive Summary' in text:
                            slide_type = 'Executive Summary'
                        elif 'Technical Summary' in text:
                            slide_type = 'Technical Summary'
                        elif 'Product Portfolio' in text:
                            slide_type = 'Product Portfolio'

            # Extract table info
            if 'table' in element:
                table_info = extract_table_structure(element)
                if table_info:
                    table_info['element_id'] = element_id

                    # Try to get critical differentiator from first column of first data row
                    if table_info['data'] and len(table_info['data']) > 1:
                        # First row is usually header
                        first_data_row = table_info['data'][1] if len(table_info['data']) > 1 else []
                        if first_data_row:
                            critical_diff = first_data_row[0] if first_data_row[0] else None

                    tables.append(table_info)

        slide_info[slide_id] = {
            'number': i,
            'id': slide_id,
            'title': slide_title,
            'slide_type': slide_type,
            'use_case': use_case,
            'tables': tables,
            'element_ids': [e.get('objectId') for e in slide.get('pageElements', [])]
        }

    return slide_info, element_to_slide


def parse_anchor(anchor_str: str) -> dict:
    """Parse the anchor JSON string to extract location info."""
    if not anchor_str:
        return {}
    try:
        return json.loads(anchor_str)
    except json.JSONDecodeError:
        return {'raw': anchor_str}


def find_row_in_table(table_data: list, quoted_text: str) -> dict:
    """Find which row in a table contains the quoted text."""
    if not table_data or not quoted_text:
        return None

    quoted_clean = quoted_text.strip().lower()

    for row_idx, row in enumerate(table_data):
        for col_idx, cell in enumerate(row):
            if quoted_clean in cell.lower():
                # Get the critical differentiator (usually first column)
                critical_diff = row[0] if row else None
                return {
                    'row_index': row_idx,
                    'col_index': col_idx,
                    'critical_differentiator': critical_diff,
                    'full_row': row
                }

    return None


def display_detailed_comments(comments: list[dict], slide_info: dict, element_to_slide: dict):
    """Display comments with full context including use case and critical differentiator."""
    if not comments:
        print("\nNo comments found in this presentation.")
        return

    print(f"\n{'='*80}")
    print(f"COMMENTS REPORT - {len(comments)} comment(s)")
    print(f"{'='*80}\n")

    # Group comments by slide
    comments_by_slide = {}

    for comment in comments:
        anchor = parse_anchor(comment.get('anchor', ''))
        page_id = anchor.get('page', 'unknown')

        if page_id not in comments_by_slide:
            comments_by_slide[page_id] = []
        comments_by_slide[page_id].append(comment)

    # Display grouped by slide
    for page_id, page_comments in sorted(comments_by_slide.items(),
                                          key=lambda x: slide_info.get(x[0], {}).get('number', 999)):
        slide = slide_info.get(page_id, {})
        slide_num = slide.get('number', '?')
        slide_title = slide.get('title', 'Unknown')
        slide_type = slide.get('slide_type', 'Unknown')
        use_case = slide.get('use_case', 'Unknown')

        print(f"\n{'─'*80}")
        print(f"SLIDE {slide_num}: {slide_title}")
        print(f"{'─'*80}")
        print(f"  Type: {slide_type}")
        print(f"  Use Case: {use_case}")
        print(f"  Comments: {len(page_comments)}")
        print()

        for i, comment in enumerate(page_comments, 1):
            anchor = parse_anchor(comment.get('anchor', ''))
            quoted = comment.get('quotedFileContent', {}).get('value', '')

            # Try to find the row/critical differentiator
            critical_diff = None
            row_info = None

            for table in slide.get('tables', []):
                row_info = find_row_in_table(table.get('data', []), quoted)
                if row_info:
                    critical_diff = row_info.get('critical_differentiator')
                    break

            print(f"  ┌─ Comment #{i} ─────────────────────────────────────────────")
            print(f"  │ Author:     {comment.get('author', {}).get('displayName', 'Unknown')}")
            print(f"  │ Created:    {comment.get('createdTime', 'Unknown')}")
            print(f"  │ Resolved:   {comment.get('resolved', False)}")
            print(f"  │")
            print(f"  │ QUOTED TEXT:")
            print(f"  │   \"{quoted[:100]}{'...' if len(quoted) > 100 else ''}\"")
            print(f"  │")
            if critical_diff:
                print(f"  │ CRITICAL DIFFERENTIATOR:")
                print(f"  │   {critical_diff}")
                print(f"  │")
            if row_info:
                print(f"  │ TABLE ROW: {row_info.get('row_index', 'N/A')}")
                print(f"  │")
            print(f"  │ COMMENT:")
            # Wrap long comments
            comment_text = comment.get('content', '')
            for line in comment_text.split('\n'):
                print(f"  │   {line}")
            print(f"  │")

            # Show replies
            replies = comment.get('replies', [])
            if replies:
                print(f"  │ REPLIES ({len(replies)}):")
                for reply in replies:
                    reply_author = reply.get('author', {}).get('displayName', 'Unknown')
                    reply_content = reply.get('content', '')
                    print(f"  │   → {reply_author}: {reply_content}")
                print(f"  │")

            # Show anchor details
            print(f"  │ ANCHOR:")
            print(f"  │   Type: {anchor.get('type', 'N/A')}")
            print(f"  │   Subtype: {anchor.get('subtype', 'N/A')}")
            print(f"  │   Target: {anchor.get('targets', ['N/A'])}")
            print(f"  └{'─'*60}")
            print()


def output_json(comments: list[dict], slide_info: dict, presentation_id: str):
    """Output comments as structured JSON with full context."""
    enriched_comments = []

    for comment in comments:
        anchor = parse_anchor(comment.get('anchor', ''))
        page_id = anchor.get('page', '')
        slide = slide_info.get(page_id, {})
        quoted = comment.get('quotedFileContent', {}).get('value', '')

        # Find critical differentiator
        critical_diff = None
        row_info = None
        for table in slide.get('tables', []):
            row_info = find_row_in_table(table.get('data', []), quoted)
            if row_info:
                critical_diff = row_info.get('critical_differentiator')
                break

        enriched = {
            'comment_id': comment.get('id'),
            'author': comment.get('author', {}).get('displayName'),
            'author_email': comment.get('author', {}).get('emailAddress'),
            'created': comment.get('createdTime'),
            'modified': comment.get('modifiedTime'),
            'resolved': comment.get('resolved', False),
            'comment_text': comment.get('content'),
            'quoted_text': quoted,
            'slide': {
                'number': slide.get('number'),
                'id': page_id,
                'title': slide.get('title'),
                'type': slide.get('slide_type'),
                'use_case': slide.get('use_case'),
            },
            'table_context': {
                'critical_differentiator': critical_diff,
                'row_index': row_info.get('row_index') if row_info else None,
                'full_row': row_info.get('full_row') if row_info else None,
            },
            'anchor': anchor,
            'replies': [
                {
                    'author': r.get('author', {}).get('displayName'),
                    'content': r.get('content'),
                    'created': r.get('createdTime'),
                }
                for r in comment.get('replies', [])
            ]
        }
        enriched_comments.append(enriched)

    output = {
        'presentation_id': presentation_id,
        'presentation_url': f'https://docs.google.com/presentation/d/{presentation_id}/edit',
        'total_comments': len(comments),
        'unresolved_comments': sum(1 for c in comments if not c.get('resolved', False)),
        'comments': enriched_comments,
    }

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Fetch comments from Google Slides presentations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch comments with full context
  uv run python fetch_slide_comments.py --url "https://docs.google.com/presentation/d/<ID>/edit"

  # Output as JSON
  uv run python fetch_slide_comments.py --url "<URL>" --output json

  # Output JSON to file
  uv run python fetch_slide_comments.py --url "<URL>" --output json > comments.json
        """
    )

    parser.add_argument(
        "--presentation-id",
        type=str,
        help="Google Slides presentation ID"
    )
    parser.add_argument(
        "--url",
        type=str,
        help="Google Slides URL (ID will be extracted)"
    )
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )

    args = parser.parse_args()

    if args.url:
        presentation_id = extract_presentation_id(args.url)
    elif args.presentation_id:
        presentation_id = args.presentation_id
    else:
        parser.print_help()
        print("\n\nError: Either --presentation-id or --url is required")
        return

    print(f"Fetching comments for presentation: {presentation_id}", file=__import__('sys').stderr)
    print("Fetching slide information...", file=__import__('sys').stderr)

    # Fetch data
    comments = fetch_comments(presentation_id)
    slide_info, element_to_slide = fetch_detailed_slide_info(presentation_id)

    # Output
    if args.output == "json":
        output = output_json(comments, slide_info, presentation_id)
        print(json.dumps(output, indent=2))
    else:
        display_detailed_comments(comments, slide_info, element_to_slide)


if __name__ == "__main__":
    main()
