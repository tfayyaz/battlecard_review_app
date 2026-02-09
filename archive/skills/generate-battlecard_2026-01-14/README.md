# Generate Battlecard Skill - Quick Start

## Overview

This is a **generic, prompt-based battlecard generator** that works with versioned `prompt.md` files to create Google Slides presentations.

**Key Concept:** Each battlecard version is completely self-contained with its own prompt, context files, and generated data.

---

## Directory Structure

```
battlecard_updates/azure_data_ai_platform/
├── CHANGELOG.md
├── v0.0.0/                      # Baseline version (existing slides)
│   └── screenshots/
└── v1.0.0/                      # First structured version
    ├── prompt.md                # Instructions for this version
    ├── context/                 # Source materials (snapshots)
    │   ├── PRODUCTS.md
    │   ├── COMPETITOR_PRODUCTS.md
    │   ├── fabric_iq_analysis.md
    │   ├── analytics_bi_with_fabric_iq.md
    │   ├── balanced_audience_analysis.md
    │   └── balanced_battlecards_summary.md
    ├── battlecard_data.jsonl    # Generated (next step)
    └── screenshots/             # Generated (after slides)
```

---

## Workflow

### 1. Create prompt.md (Done! ✅)

Located at: `battlecard_updates/azure_data_ai_platform/v1.0.0/prompt.md`

**Contains:**
- Metadata (title, go-link, competitor, categories)
- Context files used
- Executive summary content
- L100 platform comparison
- L200 differentiators for all 8 categories
- Competitive positioning guidance
- Rating distribution

### 2. Create context/ folder (Done! ✅)

All source materials are copied to: `battlecard_updates/azure_data_ai_platform/v1.0.0/context/`

**Files included:**
- PRODUCTS.md (9.1K) - Databricks products
- COMPETITOR_PRODUCTS.md (18K) - Microsoft products with Fabric IQ
- fabric_iq_analysis.md (10K) - Fabric IQ vs Genie analysis
- analytics_bi_with_fabric_iq.md (9.1K) - Updated battlecard
- balanced_audience_analysis.md (13K) - Dual-audience framework
- balanced_battlecards_summary.md (8.2K) - Summary

### 3. Generate battlecard_data.jsonl (Next Step)

```bash
cd .claude/skills/generate-battlecard

# Generate JSONL from prompt.md
uv run python generate_battlecard_data.py \
  --prompt-file ../../battlecard_updates/azure_data_ai_platform/v1.0.0/prompt.md \
  --output ../../battlecard_updates/azure_data_ai_platform/v1.0.0/battlecard_data.jsonl
```

**Output:** JSONL file with one JSON object per slide

### 4. Generate Google Slides (After JSONL)

```bash
# Generate slides from JSONL
uv run python generate_slides_from_jsonl.py \
  --jsonl-file ../../battlecard_updates/azure_data_ai_platform/v1.0.0/battlecard_data.jsonl \
  --template fy27
```

**Output:** Google Slides presentation URL

### 5. Export Screenshots (After Slides)

```bash
cd ../../scripts/export_battlecards

uv run --script export_slides_to_images.py \
  --id {PRESENTATION_ID} \
  --output ../../battlecard_updates/azure_data_ai_platform/v1.0.0/screenshots
```

### 6. Update CHANGELOG.md

Document the changes in `battlecard_updates/azure_data_ai_platform/CHANGELOG.md`

---

## What Makes v1.0.0 Special

### Major Changes from v0.0.0

1. **Fabric IQ Integration** ✅
   - November 2025 Ignite announcement
   - Ontology with graph reasoning
   - Operations Agents
   - Honest assessment of innovation

2. **Fabric Data Agents** ✅
   - Conversational AI (Public Preview)
   - Direct Genie competitor
   - Stateless limitation documented

3. **Balanced Dual-Audience Approach** ✅
   - Appeals to both C-suite and technical practitioners
   - Business outcomes + technical depth
   - No over-simplification

4. **Updated Differentiators** ✅
   - "Semantic Layer" → "Semantic Foundation" (TIED ✓✓)
   - New: "Session Context & Memory" (Genie advantage)
   - New: "Production Readiness" (Genie GA vs Fabric Preview)

5. **Natural Product Name Integration** ✅
   - Product names in sentences, not prefixes
   - Better readability

---

## Benefits of This Approach

### ✅ Version Control
- Each version has its own prompt and context
- Easy to see what changed between versions
- Can regenerate any version from its JSONL

### ✅ Auditable
- Clear record of source materials
- Citations preserved in JSONL
- Reproducible results

### ✅ Generic
- Works for any battlecard (Fabric, Snowflake, AWS)
- Same scripts for all battlecards
- Just change the prompt.md

### ✅ Self-Contained
- Each version includes all its dependencies
- No external file references that might change
- Complete audit trail

---

## Next Steps

1. **Implement generate_battlecard_data.py**
   - Read prompt.md
   - Read context files
   - Generate JSONL with slide definitions

2. **Implement generate_slides_from_jsonl.py**
   - Read JSONL file
   - Call Google Slides API
   - Generate presentation

3. **Test with v1.0.0**
   - Generate JSONL
   - Generate slides
   - Export screenshots
   - Update CHANGELOG

4. **Create Additional Battlecards**
   - Snowflake
   - AWS
   - Other competitors using same pattern

---

## File Locations

**Skill:** `.claude/skills/generate-battlecard/`
- `skill.md` - Documentation
- `README.md` - This file
- `generate_battlecard_data.py` - (To be created)
- `generate_slides_from_jsonl.py` - (To be created)

**Battlecard v1.0.0:** `battlecard_updates/azure_data_ai_platform/v1.0.0/`
- `prompt.md` - ✅ Created (comprehensive instructions)
- `context/` - ✅ Created (6 source files)
- `battlecard_data.jsonl` - ⏳ To be generated
- `screenshots/` - ⏳ To be generated after slides

---

## Questions?

See `skill.md` for detailed documentation or ask for help!
