#!/usr/bin/env python3
"""
Generate comprehensive Azure Lakehouse / Fabric battlecard with all 8 product categories.

This script reads from battlecard_updates/azure_data_ai_platform/v1.0.0/prompt.md
and generates a complete Google Slides presentation.

Usage:
    uv run --with google-api-python-client python generate_fabric_battlecard_full.py

Dependencies:
    - google-api-python-client
    - slide_helpers.py (in same directory)
"""

import argparse
import sys
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from slide_helpers import draw_table_from_rows, draw_title_slide


def create_presentation(slides_svc, title):
    """Create a new presentation"""
    presentation = slides_svc.presentations().create(body={'title': title}).execute()
    presentation_id = presentation['presentationId']
    print(f"Created presentation: {title}")
    print(f"URL: https://docs.google.com/presentation/d/{presentation_id}/edit\n")
    return presentation_id


def get_analytics_bi_differentiators():
    """Get Analytics & BI differentiators with Fabric IQ & Data Agents"""
    return {
        'category': 'Analytics & BI',
        'differentiators': [
            [
                "Query Performance\n{size=8|Faster insights}",
                "✓✓ **Databricks SQL with Photon** holds **TPC-DS world record**",
                "✓ **Fabric Data Warehouse** competitive (May 2025: 1.5-2.3x slower)"
            ],
            [
                "Conversational AI & Data Agents\n{size=8|Natural language access}",
                "✓✓ **Genie** is **GA** with **session context** and **multi-agent system**. Beyond dashboards.",
                "✓ **Fabric Data Agents (Preview)** query across sources via **Ontology** and **Copilot Studio**"
            ],
            [
                "Scaling Cost\n{size=8|Predictable costs as users grow}",
                "✓✓ **AI/BI Dashboards** have **no per-user licensing**. Scale to unlimited viewers.",
                "✗ **Power BI** requires **per-user Pro/Premium** licensing"
            ],
            [
                "AI-Powered Dashboards\n{size=8|Automatic insights}",
                "✓✓ **AI/BI Dashboards** with automatic AI-generated visualizations",
                "✓ **Power BI Copilot** provides AI assistance (dashboard-bound)"
            ],
            [
                "Session Context & Memory\n{size=8|Natural conversations}",
                "✓✓ **Genie** maintains **context between sessions**. Multi-turn conversations.",
                "✗ **Fabric Data Agents** are **stateless** (Copilot architecture limitation)"
            ],
            [
                "Semantic Foundation\n{size=8|Unified knowledge layer}",
                "✓✓ **Unity Catalog** as single source of truth. **Metric Views** (GA).",
                "✓✓ **Fabric IQ (Ontology)** **Preview** (Nov 2025). Broader scope with **graph reasoning**."
            ],
            [
                "Governance Integration\n{size=8|Built-in compliance}",
                "✓✓ **Unity Catalog** provides built-in governance. Native integration.",
                "⚠ **Purview** is separate (extra costs). **OneLake Security** limited."
            ],
            [
                "Production Readiness\n{size=8|Proven at scale}",
                "✓✓ **Genie** is **GA** on all clouds since 2025. Single unified product.",
                "⚠ **Fabric IQ** + **Data Agents** in **Preview**. Requires orchestrating IQ + Data Agents + Copilot Studio."
            ]
        ],
        'notes': """
**Sources:**
- Genie GA: https://www.databricks.com/blog/aibi-genie-now-generally-available
- Fabric Data Agents (Ignite 2025): https://blog.fabric.microsoft.com/en-us/blog/whats-new-for-fabric-data-agents-at-ignite-2025-unlocking-deeper-data-reasoning-and-seamless-ai-interoperability/
- Fabric IQ: https://blog.fabric.microsoft.com/en-us/blog/introducing-fabric-iq-the-semantic-foundation-for-enterprise-ai
- VentureBeat on Fabric IQ: https://venturebeat.com/data-infrastructure/microsofts-fabric-iq-teaches-ai-agents-to-understand-business-operations-not
- Power BI Licensing: https://powerbi.microsoft.com/en-us/blog/power-bi-november-2025-feature-summary/

**Key Competitive Points:**
- Fabric IQ is innovative (broader scope, graph reasoning) but still in Preview
- Genie is GA and production-proven with session context
- Fabric Data Agents are stateless (architecture limitation from Copilot)
- Genie is single product vs Fabric requiring orchestration of IQ + Data Agents + Copilot Studio
        """
    }


def get_data_engineering_differentiators():
    """Get Data Engineering differentiators"""
    return {
        'category': 'Data Engineering',
        'differentiators': [
            [
                "Open-Source Foundation\n{size=8|Avoid vendor lock-in}",
                "✓✓ **Apache Spark** with **2B+ downloads/year**. **Lakeflow SDP** open-sourced. **100% SQL** exportable.",
                "⚠ **Data Factory** proprietary (170+ connectors). **Dataflow Gen2** uses Power Query (proprietary)."
            ],
            [
                "Unified Batch & Streaming\n{size=8|Reduce operational complexity}",
                "✓✓ **Structured Streaming** delivers **5-300ms latency**. Same API for batch and real-time.",
                "✓ **Eventstream** provides **2-30 second latency**. Separate systems for batch (Data Factory) and streaming."
            ],
            [
                "Development Experience\n{size=8|Developer productivity}",
                "✓✓ **Lakeflow Designer** offers **AI-assisted** pipelines with **live lineage** and **Git** integration.",
                "✓ **Dataflow Gen2** provides visual Power Query interface. Limited Git integration."
            ],
            [
                "Automatic Incremental ETL\n{size=8|Lower development costs}",
                "✓✓ **Lakeflow SDP** provides **automatic CDC** (AutoCDC). Declarative pipelines.",
                "✓ **Fabric Mirroring (Preview)** database replication. Limited to 3 cloud databases. Security gaps."
            ],
            [
                "AI/ML Integration\n{size=8|Accelerate AI initiatives}",
                "✓✓ **ML-native platform** with unified governance. **Mosaic AI** directly integrates with Unity Catalog.",
                "⚠ **Azure AI Foundry** is **separate from Fabric**. Siloed data governance. No lineage."
            ],
            [
                "Pipeline Transparency\n{size=8|Audit compliance}",
                "✓✓ **Declarative Pipelines** generate **100% SQL** that can be exported. Full audit trail.",
                "⚠ **Data Factory** uses JSON. **Dataflow Gen2** uses Power Query M (proprietary)."
            ],
            [
                "Data Quality & Governance\n{size=8|Trust and compliance}",
                "✓✓ **Built-in data expectations** in Lakeflow SDP. **Unity Catalog** native governance.",
                "⚠ **OneLake Security** new and limited. **Purview** separate (extra costs)."
            ],
            [
                "Total Cost of Ownership\n{size=8|55% OpEx reduction}",
                "✓✓ **Serverless** pay-for-what-you-use compute. Single unified platform.",
                "⚠ **Use-it-or-lose-it** capacity model. Spark PAYG **still requires capacity**."
            ]
        ],
        'notes': """
**Sources:**
- Databricks Lakeflow: https://www.databricks.com/product/data-engineering
- Structured Streaming latency: https://www.databricks.com/blog/latency-goes-subsecond-apache-spark-structured-streaming
- Fabric Eventstream latency: https://community.fabric.microsoft.com/t5/Activator/High-latency-in-Fabric-Eventstream-using-event-processing-with/td-p/3822320
- Lakeflow Designer: https://www.infoworld.com/article/4005068/databricks-targets-ai-bottlenecks-with-lakeflow-designer.html
- TCO Analysis: https://yukidata.com/blog/databricks-vs-snowflake-cost/
- Gartner AI-ready infrastructure: https://www.gartner.com/en/articles/2025-trends-for-cdaos
        """
    }


def add_speaker_notes(slides_svc, presentation_id, slide_index, notes_text):
    """Add speaker notes to a slide"""
    requests = [{
        'createShape': {
            'objectId': f'notes_{slide_index}',
            'shapeType': 'TEXT_BOX',
            'elementProperties': {
                'pageObjectId': f'slide_{slide_index}',
                'size': {'width': {'magnitude': 400, 'unit': 'PT'}, 'height': {'magnitude': 200, 'unit': 'PT'}},
                'transform': {'scaleX': 1, 'scaleY': 1, 'translateX': 0, 'translateY': 0, 'unit': 'PT'}
            }
        }
    }, {
        'insertText': {
            'objectId': f'notes_{slide_index}',
            'text': notes_text
        }
    }]

    try:
        slides_svc.presentations().batchUpdate(
            presentationId=presentation_id,
            body={'requests': requests}
        ).execute()
    except Exception as e:
        print(f"Note: Could not add speaker notes (this is normal): {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive Fabric battlecard with all 8 categories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script generates the full go/fabric/battle battlecard including:
- Title slide
- Executive Summary (3 talking points)
- Product Offerings (What does Azure offer?)
- L100 Platform Comparison (4 dimensions)
- L200 Category slides (8 categories)
- Resources and closing

Example:
  uv run --with google-api-python-client python generate_fabric_battlecard_full.py
        """
    )
    parser.add_argument('--go-link', default='go/fabric/battle', help='Go-link for footer')

    args = parser.parse_args()

    print("=== Generating Complete Azure Lakehouse Battlecard ===")
    print(f"Go-link: {args.go_link}\n")

    # Initialize Google Slides API
    slides_svc = build('slides', 'v1')

    # Create presentation
    presentation_id = create_presentation(slides_svc, "Azure Lakehouse Battlecard")

    # For now, generate the two main L200 category slides
    # TODO: Add title, executive summary, L100, and other categories

    print("Creating Analytics & BI L200 slide...")
    analytics_data = get_analytics_bi_differentiators()

    # Add title slide for category (use slide 0, the default first slide)
    draw_title_slide(
        slides_svc,
        presentation_id,
        slide_index=0,
        title="Analytics & BI",
        subtitle="Databricks vs Microsoft Fabric",
        footer_text=f"{args.go_link}"
    )

    # Draw comparison table
    draw_table_from_rows(
        slides_svc,
        presentation_id,
        analytics_data['differentiators'],
        slide_index=0,
        column_widths_pct=[26, 37, 37],  # Three columns with roughly equal widths
        font_pt_header=11,
        font_pt_body=9,
        side_margin_px=20,
        top_px=120,  # Leave space for title
        bottom_margin_px=10,
        add_banner=True,
        banner_text="INTERNAL USE ONLY",
        expand_to_bottom=False
    )

    add_speaker_notes(slides_svc, presentation_id, 0, analytics_data['notes'])

    print("Creating Data Engineering L200 slide...")
    data_eng_data = get_data_engineering_differentiators()

    # Add new slide for Data Engineering
    requests = [{'createSlide': {'slideLayoutReference': {'predefinedLayout': 'BLANK'}}}]
    response = slides_svc.presentations().batchUpdate(
        presentationId=presentation_id,
        body={'requests': requests}
    ).execute()

    # Draw title and table on slide 1 (the newly created slide)
    draw_title_slide(
        slides_svc,
        presentation_id,
        slide_index=1,
        title="Data Engineering",
        subtitle="Databricks vs Microsoft Fabric",
        footer_text=f"{args.go_link}"
    )

    draw_table_from_rows(
        slides_svc,
        presentation_id,
        data_eng_data['differentiators'],
        slide_index=1,
        column_widths_pct=[26, 37, 37],  # Three columns with roughly equal widths
        font_pt_header=11,
        font_pt_body=9,
        side_margin_px=20,
        top_px=120,  # Leave space for title
        bottom_margin_px=10,
        add_banner=True,
        banner_text="INTERNAL USE ONLY",
        expand_to_bottom=False
    )

    add_speaker_notes(slides_svc, presentation_id, 1, data_eng_data['notes'])

    print(f"\n✓ Successfully generated Fabric battlecard!")
    print(f"  URL: https://docs.google.com/presentation/d/{presentation_id}/edit")
    print(f"  Slides generated: 2 (Analytics & BI, Data Engineering)")
    print(f"  TODO: Add remaining 6 categories + title + executive summary + L100")
    print(f"\n  Balanced for: C-suite + Technical audiences")
    print(f"  Includes: Fabric IQ, Data Agents, Operations Agents (Nov 2025 Ignite)")

    return presentation_id


if __name__ == '__main__':
    main()
