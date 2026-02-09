#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "google-auth",
#     "google-auth-oauthlib",
#     "google-auth-httplib2",
#     "google-api-python-client",
# ]
# ///

"""Test script for speaker notes functionality."""

import os
import sys
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Add parent directory to path to import from slide_helpers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SCOPES = ['https://www.googleapis.com/auth/presentations']


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


def test_speaker_notes():
    """Test creating a slide and adding speaker notes."""
    slides_svc = get_slides_service()

    # Create a new presentation
    print("\n=== Creating Test Presentation ===")
    presentation = slides_svc.presentations().create(body={
        'title': 'Test Speaker Notes'
    }).execute()

    presentation_id = presentation['presentationId']
    print(f"Created presentation: {presentation_id}")
    print(f"URL: https://docs.google.com/presentation/d/{presentation_id}/edit")

    # Get presentation details
    pres = slides_svc.presentations().get(presentationId=presentation_id).execute()
    slides = pres.get('slides', [])

    print(f"\nPresentation has {len(slides)} slides")

    # Examine the first slide
    if slides:
        slide = slides[0]
        slide_id = slide['objectId']
        print(f"\n=== Examining Slide 0 ===")
        print(f"Slide ID: {slide_id}")

        # Check for notes page
        slide_properties = slide.get('slideProperties', {})
        notes_page = slide_properties.get('notesPage', {})
        notes_page_id = notes_page.get('objectId')

        print(f"Notes Page ID: {notes_page_id}")

        if notes_page_id:
            # Notes pages are NOT in the slides array!
            # We need to query with specific fields to get notes page data
            print("\n=== Getting Full Presentation with Notes Pages ===")

            # Get presentation with all fields including layouts, masters which might contain notes
            pres_full = slides_svc.presentations().get(
                presentationId=presentation_id
            ).execute()

            print(f"Full presentation keys: {list(pres_full.keys())}")

            # Check if notes pages are in layouts or masters
            if 'layouts' in pres_full:
                print(f"Found {len(pres_full['layouts'])} layouts")

            if 'masters' in pres_full:
                print(f"Found {len(pres_full['masters'])} masters")

            if 'notesMaster' in pres_full:
                print("Found notesMaster!")
                notes_master = pres_full['notesMaster']
                print(f"  notesMaster ID: {notes_master.get('objectId')}")
                print(f"  notesMaster elements: {len(notes_master.get('pageElements', []))}")

            # Try searching in ALL returned slides (including notes pages)
            all_pages = pres_full.get('slides', [])
            print(f"\nSearching {len(all_pages)} slides/pages for notes page ID: {notes_page_id}")

            found_notes_page = False
            for i, page in enumerate(all_pages):
                page_id = page.get('objectId')
                print(f"  Page {i}: {page_id}")

                if page_id == notes_page_id:
                    print(f"\n✓ Found notes page at index {i}")
                    found_notes_page = True

                    # Print all page elements
                    page_elements = page.get('pageElements', [])
                    print(f"Notes page has {len(page_elements)} elements:")

                    for j, elem in enumerate(page_elements):
                        elem_id = elem.get('objectId')
                        elem_type = elem.get('shape', {}).get('shapeType', 'N/A')
                        placeholder = elem.get('shape', {}).get('placeholder', {})
                        placeholder_type = placeholder.get('type', 'N/A')

                        print(f"  Element {j}:")
                        print(f"    ID: {elem_id}")
                        print(f"    Type: {elem_type}")
                        print(f"    Placeholder: {placeholder_type}")

                        # If it has text, show it
                        text_content = elem.get('shape', {}).get('text', {})
                        if text_content:
                            text_elements = text_content.get('textElements', [])
                            text = ''.join([te.get('textRun', {}).get('content', '') for te in text_elements])
                            print(f"    Text: {text[:50]}...")

                    break

            if not found_notes_page:
                print(f"\n✗ ERROR: Notes page {notes_page_id} not found in API response!")

        # Try to add speaker notes using a different approach
        print("\n=== Attempting to Add Speaker Notes (Direct Approach) ===")
        test_notes = "This is a test note.\n\nSecond paragraph of the test note."

        # Approach 1: Try creating a text box on the notes page directly
        print("\nApproach 1: Create text box on notes page")
        try:
            text_box_id = f"{notes_page_id}_textbox"
            requests = [
                {
                    'createShape': {
                        'objectId': text_box_id,
                        'shapeType': 'TEXT_BOX',
                        'elementProperties': {
                            'pageObjectId': notes_page_id,
                            'size': {
                                'width': {'magnitude': 6000000, 'unit': 'EMU'},
                                'height': {'magnitude': 4000000, 'unit': 'EMU'}
                            },
                            'transform': {
                                'scaleX': 1,
                                'scaleY': 1,
                                'translateX': 1000000,
                                'translateY': 1000000,
                                'unit': 'EMU'
                            }
                        }
                    }
                },
                {
                    'insertText': {
                        'objectId': text_box_id,
                        'insertionIndex': 0,
                        'text': test_notes
                    }
                }
            ]

            result = slides_svc.presentations().batchUpdate(
                presentationId=presentation_id,
                body={'requests': requests}
            ).execute()

            print(f"✓ Successfully created text box on notes page!")
            print(f"  Result: {result}")

        except Exception as e:
            print(f"✗ Approach 1 failed: {e}")

        # Approach 2: Use speakerNotesObjectId from NotesProperties (CORRECT METHOD per official docs)
        print("\nApproach 2: Use speakerNotesObjectId from NotesProperties")
        try:
            # Get the notes page properties - this should contain speakerNotesObjectId
            notes_page_data = slide_properties.get('notesPage', {})
            notes_properties = notes_page_data.get('notesProperties', {})
            speaker_notes_object_id = notes_properties.get('speakerNotesObjectId')

            print(f"NotesProperties: {notes_properties}")
            print(f"speakerNotesObjectId: {speaker_notes_object_id}")

            if speaker_notes_object_id:
                print(f"\nTrying to insert text using speakerNotesObjectId: {speaker_notes_object_id}")

                requests = [
                    {'insertText': {'objectId': speaker_notes_object_id, 'insertionIndex': 0, 'text': test_notes}}
                ]

                result = slides_svc.presentations().batchUpdate(
                    presentationId=presentation_id,
                    body={'requests': requests}
                ).execute()

                print(f"✓ SUCCESS! Added speaker notes using official API method")
                print(f"\nOpen presentation and verify notes:")
                print(f"https://docs.google.com/presentation/d/{presentation_id}/edit")

            else:
                print("⚠ No speakerNotesObjectId found - shape doesn't exist yet")
                print("According to docs, API should auto-create it when we insert text...")

        except Exception as e:
            print(f"✗ Approach 2 failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n=== Test Complete ===")
    print(f"Presentation URL: https://docs.google.com/presentation/d/{presentation_id}/edit")


if __name__ == '__main__':
    test_speaker_notes()
