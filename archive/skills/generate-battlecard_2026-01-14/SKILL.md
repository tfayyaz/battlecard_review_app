---
name: generate-battlecard
description: Generic battlecard generator that works with versioned prompt.md files to generate Google Slides presentations
---

# Generate Battlecard (Generic)

This skill generates competitive battlecard presentations using a prompt-based approach. Each battlecard version has its own `prompt.md` file that defines the content, and generates a `battlecard_data.jsonl` file that feeds into the slide generation.

**Flow:**
1. Read `battlecard_updates/<name>/<version>/prompt.md`
2. Generate `battlecard_updates/<name>/<version>/battlecard_data.jsonl`
3. Generate Google Slides from the JSONL file

---

## Directory Structure

```
battlecard_updates/
└── <battlecard_name>/
    ├── CHANGELOG.md
    ├── v<version>/
    │   ├── prompt.md                 # Prompt and instructions for this version
    │   ├── context/                  # Source materials (snapshots)
    │   │   ├── PRODUCTS.md
    │   │   ├── COMPETITOR_PRODUCTS.md
    │   │   └── *.md (analysis files)
    │   ├── battlecard_data.jsonl     # Generated structured data
    │   └── screenshots/              # Exported slide images
    └── v<next_version>/
        ├── prompt.md
        ├── context/
        ├── battlecard_data.jsonl
        └── screenshots/
```

**Example:**
```
battlecard_updates/
└── azure_data_ai_platform/
    ├── CHANGELOG.md
    ├── v0.0.0/
    │   └── [Platform Battlecard] Azure Lakehouse _251217/
    │       └── images/
    └── v1.0.0/
        ├── prompt.md
        ├── context/
        │   ├── PRODUCTS.md
        │   ├── COMPETITOR_PRODUCTS.md
        │   ├── fabric_iq_analysis.md
        │   ├── analytics_bi_with_fabric_iq.md
        │   ├── balanced_audience_analysis.md
        │   └── balanced_battlecards_summary.md
        ├── battlecard_data.jsonl
        └── screenshots/
```

**Why context/ folder?**
- Makes each version completely self-contained
- Captures exact state of source materials at generation time
- Enables full reproducibility
- Creates audit trail for compliance

---

## Step 1: Create prompt.md

The `prompt.md` file contains:
- Battlecard metadata (title, go-link, competitor, categories)
- Content sources (PRODUCTS.md, COMPETITOR_PRODUCTS.md, analysis files)
- Slide structure definition
- Differentiators per category
- Citations and sources

### Example prompt.md Structure

```markdown
# Battlecard Generation Prompt - v0.1.0

## Metadata
- **Title:** Azure Lakehouse Battlecard
- **Go-Link:** go/fabric/battle
- **Competitor:** Microsoft (Fabric, Purview, Azure AI Foundry)
- **Categories:** All 8 from PRODUCTS.md
- **Date:** 2025-01-13

## Sources
- PRODUCTS.md (Databricks products)
- COMPETITOR_PRODUCTS.md (Microsoft products with Fabric IQ/Data Agents)
- battlecard_updates/analytics_bi_with_fabric_iq.md

## Slide Structure
1. Title slide
2. Executive Summary (3 talking points)
3. Product Offerings (What does Azure offer?)
4. L100 Platform Comparison (4 dimensions)
5. L200+ Category Slides (8 categories)

## Differentiators by Category

### Data Engineering (8 differentiators)
1. Open-Source Foundation
2. Unified Batch & Streaming
3. Development Experience
4. Automatic Incremental ETL
5. AI/ML Integration
6. Pipeline Transparency
7. Data Quality & Governance
8. Total Cost of Ownership

### Analytics & BI (8 differentiators)
1. Query Performance
2. Conversational AI & Data Agents
3. Scaling Cost
4. AI-Powered Dashboards
5. Session Context & Memory
6. Semantic Foundation
7. Governance Integration
8. Production Readiness

... (continue for all 8 categories)
```

---

## Step 2: Generate battlecard_data.jsonl

Run the generator script to create the JSONL file from prompt.md:

```bash
cd /Users/tahir.fayyaz/databricks-dev/compete_automation/battlecards-app/1st_principles_go_products/.claude/skills/generate-battlecard

# Generate JSONL from prompt.md
uv run python generate_battlecard_data.py \
  --prompt-file "../../battlecard_updates/azure_data_ai_platform/v0.1.0/prompt.md" \
  --output "../../battlecard_updates/azure_data_ai_platform/v0.1.0/battlecard_data.jsonl"
```

### JSONL Format

Each line in the JSONL file represents one slide:

```json
{"slide_type": "title", "title": "Azure Lakehouse Battlecard", "go_link": "go/fabric/battle", "last_updated": "FY26Q3"}
{"slide_type": "executive_summary", "talking_points": [...]}
{"slide_type": "product_offerings", "sections": [...]}
{"slide_type": "l100_platform", "dimensions": [...]}
{"slide_type": "l200_category", "category": "Data Engineering", "differentiators": [...]}
{"slide_type": "l200_category", "category": "Analytics & BI", "differentiators": [...]}
```

---

## Step 3: Generate Google Slides

Once the JSONL file exists, generate the slides:

```bash
cd /Users/tahir.fayyaz/databricks-dev/compete_automation/battlecards-app/1st_principles_go_products/.claude/skills/generate-battlecard

# Generate slides from JSONL
uv run python generate_slides_from_jsonl.py \
  --jsonl-file "../../battlecard_updates/azure_data_ai_platform/v0.1.0/battlecard_data.jsonl" \
  --template fy27
```

---

## Google Cloud CLI Setup (Run First!)

**Before generating slides**, you MUST check and set up Google Cloud CLI authentication.

### Step 1: Check if gcloud CLI is installed

```bash
which gcloud && gcloud --version
```

If `gcloud` is not found, install it:

```bash
brew install --cask google-cloud-sdk
```

### Step 2: Set up Application Default Credentials

```bash
gcloud auth application-default login \
  --scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/presentations,https://www.googleapis.com/auth/drive
```

### Step 3: Set the quota project

```bash
gcloud auth application-default set-quota-project pm-team-test
```

### Step 4: Verify authentication

```bash
cat ~/.config/gcloud/application_default_credentials.json | grep quota_project_id
```

You should see: `"quota_project_id": "pm-team-test"`

---

## Usage Workflow

### Complete Workflow Example

```bash
# 1. Create prompt.md (manually or with assistance)
# battlecard_updates/azure_data_ai_platform/v0.1.0/prompt.md

# 2. Generate JSONL from prompt
cd /Users/tahir.fayyaz/databricks-dev/compete_automation/battlecards-app/1st_principles_go_products/.claude/skills/generate-battlecard

uv run python generate_battlecard_data.py \
  --prompt-file ../../battlecard_updates/azure_data_ai_platform/v0.1.0/prompt.md \
  --output ../../battlecard_updates/azure_data_ai_platform/v0.1.0/battlecard_data.jsonl

# 3. Generate Google Slides from JSONL
uv run python generate_slides_from_jsonl.py \
  --jsonl-file ../../battlecard_updates/azure_data_ai_platform/v0.1.0/battlecard_data.jsonl \
  --template fy27

# 4. Export screenshots
cd ../../scripts/export_battlecards
uv run --script export_slides_to_images.py \
  --id {PRESENTATION_ID} \
  --output ../../battlecard_updates/azure_data_ai_platform/v0.1.0/screenshots

# 5. Update CHANGELOG.md
```

---

## CLI Arguments

### generate_battlecard_data.py

```bash
Arguments:
  --prompt-file     Path to prompt.md file (required)
  --output          Path to output JSONL file (required)
  --sources-dir     Directory containing PRODUCTS.md, COMPETITOR_PRODUCTS.md (default: ../..)
```

### generate_slides_from_jsonl.py

```bash
Arguments:
  --jsonl-file      Path to battlecard_data.jsonl file (required)
  --template        Template style: 'fy26' or 'fy27' (default: fy27)
  --presentation-id Optional existing presentation ID to update
```

---

## Slide Types in JSONL

### 1. Title Slide

```json
{
  "slide_type": "title",
  "title": "Azure Lakehouse Battlecard",
  "go_link": "go/fabric/battle",
  "alt_go_link": "go/azurelakehouse/battle",
  "last_updated": "FY26Q3"
}
```

### 2. Executive Summary

```json
{
  "slide_type": "executive_summary",
  "talking_points": [
    {
      "number": 1,
      "heading": "Fabric is a rebundling",
      "bullets": [
        "1.5 years post GA, compute remains fragmented",
        "Storage remains locked, governance immature"
      ]
    },
    {
      "number": 2,
      "heading": "Lock-in",
      "bullets": ["OneLake is vendor-locked", "3X cost"]
    },
    {
      "number": 3,
      "heading": "Expensive",
      "bullets": ["Use-it-or-lose-it", "Throttling issues"]
    }
  ]
}
```

### 3. Product Offerings

```json
{
  "slide_type": "product_offerings",
  "title": "What does Azure offer for Lakehouse?",
  "sections": [
    {
      "name": "Microsoft Fabric",
      "description": "Immature rebundling of existing services",
      "bullets": [
        "Broken, fragmented governance",
        "OneLake is vendor-lock",
        "Expensive use-it-or-lose-it pricing"
      ]
    },
    {
      "name": "Microsoft Purview",
      "description": "Business catalog only",
      "bullets": ["Cannot enforce security", "Extra costs"]
    },
    {
      "name": "Azure AI Foundry",
      "description": "Fragmented user experience",
      "bullets": ["Separate from Fabric", "Platform lock-in"]
    }
  ]
}
```

### 4. L100 Platform Comparison

```json
{
  "slide_type": "l100_platform",
  "title": "Platform Comparison",
  "dimensions": [
    {
      "name": "Unified Governance",
      "fabric": {
        "rating": "⚠",
        "text": "No unified catalog. OneLake Security limited."
      },
      "databricks": {
        "rating": "✓✓",
        "text": "Unity Catalog for all Data & AI assets"
      }
    },
    {
      "name": "Open Access",
      "fabric": {
        "rating": "⚠",
        "text": "OneLake vendor-locked, 3X tax"
      },
      "databricks": {
        "rating": "✓✓",
        "text": "Open catalog, runs anywhere"
      }
    }
  ]
}
```

### 5. L200 Category Slides

```json
{
  "slide_type": "l200_category",
  "category": "Data Engineering",
  "differentiators": [
    {
      "name": "Open-Source Foundation",
      "subtitle": "Avoid vendor lock-in",
      "databricks": {
        "rating": "✓✓",
        "text": "Apache Spark with 2B+ downloads/year. 100% portable.",
        "products": ["Lakeflow SDP", "Spark"]
      },
      "fabric": {
        "rating": "⚠",
        "text": "Data Factory proprietary. Limited portability.",
        "products": ["Data Factory", "Dataflow Gen2"]
      },
      "sources": [
        {
          "title": "Databricks open-sourced declarative pipelines",
          "url": "https://..."
        }
      ]
    }
  ]
}
```

---

## Speaker Notes

### Adding Speaker Notes to Slides

Speaker notes are automatically added to slides that include a `"notes"` field in the JSONL data. The notes are added using the Google Slides API's `speakerNotesObjectId`.

### How It Works

The implementation uses the official Google Slides API method for working with speaker notes:

1. **Get the speakerNotesObjectId**: Each slide has a `speakerNotesObjectId` in its notes page properties
2. **Insert text directly**: Use the `insertText` API request with the `speakerNotesObjectId`
3. **Auto-creation**: The API automatically creates the notes shape if it doesn't exist

### Technical Implementation

```python
# Get the speaker notes object ID from slide properties
slide_properties = slide.get('slideProperties', {})
notes_page = slide_properties.get('notesPage', {})
notes_properties = notes_page.get('notesProperties', {})
speaker_notes_object_id = notes_properties.get('speakerNotesObjectId')

# Insert text using the speaker notes object ID
requests = [
    {'insertText': {'objectId': speaker_notes_object_id, 'insertionIndex': 0, 'text': notes_text}}
]
```

### JSONL Format for Notes

Add a `"notes"` field to any slide type:

```json
{
  "slide_type": "l200_category",
  "category": "Data Engineering",
  "differentiators": [...],
  "notes": "Sources:\n- Databricks Lakeflow: https://www.databricks.com/product/data-engineering\n- Structured Streaming latency: https://www.databricks.com/blog/...\n\nKey Competitive Points:\n- Fabric Eventstream has 2-30 second latency vs Databricks 5-300ms\n- Data Factory uses proprietary JSON vs Databricks 100% exportable SQL"
}
```

### Best Practices

1. **Use plain text only**: DO NOT use markdown formatting like `**bold**` in notes - it will not render in Google Slides speaker notes
2. **Use line breaks for structure**: Use `\n` for line breaks to organize content into sections
3. **Include sources**: Add links to documentation, benchmarks, and competitive research
4. **Add competitive talking points**: Highlight key differentiators and objection handlers
5. **Keep notes concise**: Focus on what speakers need during presentations

### Important: Slide Text vs Speaker Notes Formatting

- **Slide text** (in differentiators, talking points, etc.): USE `**bold**` markdown - it will be rendered as bold in slides
- **Speaker notes** (in "notes" field): DO NOT use `**bold**` markdown - use plain text only

**Example:**
```json
{
  "differentiators": [
    {
      "databricks": {"text": "**Apache Spark** with 2B+ downloads"},  // ✓ Bold renders in slides
      "fabric": {"text": "**Data Factory** proprietary"}
    }
  ],
  "notes": "Sources:\n- Apache Spark downloads: https://..."  // ✓ Plain text for notes
}
```

### Supported Slide Types

Speaker notes work on all slide types, but are most commonly used for:
- **L200 Category slides**: Sources and competitive talking points
- **L100 Platform slides**: High-level positioning and objection handling
- **Executive Summary**: Strategic messaging and elevator pitch

### Official Documentation

- [Google Slides API: Work with speaker notes](https://developers.google.com/workspace/slides/api/guides/notes)
- [Google Slides API: Introduction](https://developers.google.com/slides/api/guides/overview)

---

## Benefits of This Approach

### 1. Version Control ✅
- Each version has its own `prompt.md` and `battlecard_data.jsonl`
- Easy to see what changed between versions
- Can regenerate any version from its JSONL

### 2. Auditable ✅
- Clear record of what prompt/context generated each version
- Citations preserved in JSONL
- Reproducible results

### 3. Generic ✅
- Works for any battlecard (Fabric, Snowflake, AWS, etc.)
- Same scripts for all battlecards
- Just change the prompt.md

### 4. Flexible ✅
- Can edit JSONL directly if needed
- Can regenerate slides from same JSONL with different templates
- Can update slides without regenerating JSONL

### 5. Testable ✅
- Can validate JSONL format
- Can preview JSONL content before generating slides
- Can test slide generation independently

---

## Changelog Management

Each battlecard has its own `CHANGELOG.md` file that tracks all versions, changes, and presentation links. Additionally, a master `CHANGELOG.md` exists at the top level of `battlecard_updates/` directory.

### Master Changelog Structure

**Location**: `battlecard_updates/CHANGELOG.md`

This file provides:
- **All Battlecards Table** - Quick reference to all battlecards with latest versions and links
- **Battlecard Summaries** - Key positioning for each battlecard
- **Version History Table** - Timeline of all releases
- **Quick Links** - Documentation and common tasks

### Individual Battlecard CHANGELOG

**Location**: `battlecard_updates/<battlecard_name>/CHANGELOG.md`

Each battlecard CHANGELOG must include:

#### 1. All Presentations Table (Top of File)

```markdown
## All Presentations

| Version | Date | Presentation Link | Status |
|---------|------|-------------------|--------|
| v1.0.1 | 2026-01-14 | [Battlecard v1.0.1](https://docs.google.com/presentation/d/...) | ✅ Current |
| v1.0.0 | 2026-01-14 | [Battlecard v1.0.0](https://docs.google.com/presentation/d/...) | ⚠️ Deprecated |
```

**Requirements**:
- Most recent version at top
- Include status indicator (✅ Current / ⚠️ Deprecated / 🚧 Draft)
- Include reason for deprecation if applicable

#### 2. Version Sections

Each version section must include:

```markdown
## v1.0.1 (2026-01-14)

**📊 Presentation**: [Battlecard v1.0.1](https://docs.google.com/presentation/d/...)

### <Release Type>

<Description of changes>

### <Subsections as needed>

### Sources
- Source 1: URL
- Source 2: URL
```

**Section Structure**:
- **Version header** with date
- **📊 Presentation link** immediately after header
- **Release type** (e.g., "Fact Corrections", "Major Restructuring", "Initial Release")
- **Change description** with before/after comparisons for corrections
- **Sources** section with URLs to documentation

#### 3. Version Numbering

Follow semantic versioning:
- **Major (x.0.0)**: Complete restructure, category changes, new battlecard
- **Minor (1.x.0)**: New differentiators, slide additions, significant content updates
- **Patch (1.0.x)**: Fact corrections, clarifications, minor wording changes

### When to Create a New Version

Create a new version when:
1. **Factual corrections needed** (patch version)
   - Pricing changes
   - Product status updates (GA, Preview, Deprecated)
   - Feature availability corrections
   - Incorrect claims identified in comments

2. **Content updates** (minor version)
   - New product announcements
   - New differentiators added
   - Slide reordering or restructuring
   - New competitive positioning

3. **Major restructures** (major version)
   - Battlecard focus change (e.g., Azure Lakehouse → Fabric Data Platform)
   - Category additions/removals
   - Complete content refresh
   - New competitor added

### Changelog Entry Template

```markdown
## v1.0.1 (YYYY-MM-DD)

**📊 Presentation**: [<Battlecard Name> v1.0.1](https://docs.google.com/presentation/d/...)

### <Release Type>

<Brief description of release purpose>

### Changes Made

1. **<Change Category>** (<Affected Slide>)
   - **OLD**: "<Previous claim>"
   - **NEW**: "<Corrected claim>"
   - **Reason**: <Why the change was made>
   - **Source**: <Documentation URL>
   - **Comment**: <Reviewer comment if applicable>

### Sources
- Source 1: https://...
- Source 2: https://...

### Comment Resolution Status (if applicable)
- ✅ Issue 1 resolved
- ✅ Issue 2 resolved
- ⏳ Issue 3 pending
```

### Fact-Checking Requirements

Before each release:
1. **Verify all claims** against official documentation
2. **Check pricing** against current pricing pages
3. **Verify product status** (GA, Preview, Deprecated)
4. **Validate benchmarks** with recent data
5. **Document sources** in CHANGELOG and speaker notes

### Comment Resolution Tracking

When addressing presentation comments:
1. **Fetch comments** using `fetch_slide_comments.py`
2. **Document each fix** in CHANGELOG with:
   - Comment author and text
   - Old vs new claim
   - Source verification
3. **Update status table** showing resolved/pending items
4. **Link to comment thread** if available

### Master Changelog Updates

After releasing a new battlecard version:
1. Update `battlecard_updates/CHANGELOG.md`
2. Update the "All Battlecards" table with latest version
3. Add entry to "Version History Summary" table
4. Ensure presentation link is correct and accessible

### Example Changelog Workflow

```bash
# 1. Fetch comments from presentation
cd .claude/skills/generate-battlecard/scripts
uv run python fetch_slide_comments.py --url "<PRESENTATION_URL>" --output json > comments.json

# 2. Create new version directory
mkdir -p battlecard_updates/<name>/v1.0.1

# 3. Update JSONL with corrections
# (Edit battlecard_complete.jsonl)

# 4. Generate new slides
uv run python generate_battlecard_from_jsonl.py --jsonl battlecard_updates/<name>/v1.0.1/battlecard_complete.jsonl

# 5. Update CHANGELOG.md
# - Add v1.0.1 section with presentation link
# - Document all changes with before/after
# - Add sources
# - Update "All Presentations" table

# 6. Update master CHANGELOG
# Edit battlecard_updates/CHANGELOG.md
```

### Best Practices

1. **Link early and often** - Add presentation links as soon as slides are generated
2. **Document sources** - Every claim should have a verifiable source
3. **Be specific** - "Limited support" → "Only supports Spark, SQL Endpoint, Power BI"
4. **Track deprecation** - Mark deprecated versions clearly with reasons
5. **Preserve history** - Never delete old changelog entries, only mark as deprecated
6. **Cross-reference** - Link to related changes in other battlecards

---

## Example: Azure Data & AI Platform Battlecard

### Directory Structure

```
battlecard_updates/azure_data_ai_platform/
├── CHANGELOG.md
├── v0.0.0/
│   └── [Platform Battlecard] Azure Lakehouse _251217/
│       └── images/
│           ├── slide_001.png
│           └── ...
└── v0.1.0/
    ├── prompt.md
    ├── battlecard_data.jsonl
    └── screenshots/
```

### Generate v0.1.0

```bash
# 1. Create prompt.md (manually)
# 2. Generate JSONL
cd .claude/skills/generate-battlecard
uv run python generate_battlecard_data.py \
  --prompt-file ../../battlecard_updates/azure_data_ai_platform/v0.1.0/prompt.md \
  --output ../../battlecard_updates/azure_data_ai_platform/v0.1.0/battlecard_data.jsonl

# 3. Generate slides
uv run python generate_slides_from_jsonl.py \
  --jsonl-file ../../battlecard_updates/azure_data_ai_platform/v0.1.0/battlecard_data.jsonl
```

---

## Troubleshooting

### Google Slides API errors

```bash
gcloud auth application-default login \
  --scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/presentations,https://www.googleapis.com/auth/drive
gcloud auth application-default set-quota-project pm-team-test
```

### Missing dependencies

```bash
cd .claude/skills/generate-battlecard
uv sync
```

### Invalid JSONL format

Validate the JSONL file:
```bash
uv run python validate_jsonl.py \
  --jsonl-file ../../battlecard_updates/azure_data_ai_platform/v0.1.0/battlecard_data.jsonl
```

---

## Future Enhancements

- [ ] Add JSONL validation schema
- [ ] Add preview mode (generate slides locally without API)
- [ ] Add diff tool to compare JSONL files between versions
- [ ] Add template customization options
- [ ] Add batch generation for multiple battlecards
