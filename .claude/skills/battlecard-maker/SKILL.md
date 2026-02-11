# Battlecard Maker

Generate Google Slides battlecard presentations from battlecard data stored in Lakebase.

## Triggers

Use this skill when the user wants to:
- Generate slides or a presentation from a battlecard
- Export a battlecard as PDF or Google Slides
- Create a slide deck from a battlecard URL or ID
- Customize battlecard slide formatting (fonts, colors, layout, icons)
- Generate executive summary, product portfolio, or technical summary slides

Keywords: "generate slides", "make a deck", "export battlecard", "create presentation", "battlecard slides", "make battlecard", "slide deck", "exec summary", "product portfolio", "tech summary"

## Prerequisites

1. **Google ADC auth** — run once:
   ```bash
   gcloud auth application-default login \
     --scopes=https://www.googleapis.com/auth/presentations,https://www.googleapis.com/auth/drive
   ```
2. **Databricks CLI profile** `fe-vm-pmt` configured (for Lakebase access)
3. **Dependencies** installed via `uv sync` in the project root

## Quick Start

### L200 Key Differentiator Table Slides

```bash
uv run python scripts/generate_battlecard_slides_local.py \
  --battlecard-id <UUID> \
  --output-dir /tmp/battlecard-slides
```

### Recommended V6 config (compact, inline details, emoji icons):

```bash
uv run python scripts/generate_battlecard_slides_local.py \
  --battlecard-id <UUID> \
  --output-dir /tmp/battlecard-slides/v6 \
  --rows-per-slide 8 \
  --inline-details \
  --icon-style emoji \
  --cat-title-font-size 17 \
  --vendor-font-size 7.5 \
  --col0-title-font-size 8 \
  --col0-subtitle-font-size 7 \
  --col0-subtitle-color "#555555" \
  --header-font-size 9 \
  --table-y 58 \
  --cat-title-y 16 \
  --cat-title-height 40 \
  --table-width 695 \
  --col-widths "170,262,262"
```

### Summary Slides (Exec Summary + Product Portfolio + Tech Summary)

```bash
uv run python scripts/generate_summary_slides_local.py \
  --battlecard-id <UUID> \
  --output-dir /tmp/battlecard-slides/summaries
```

Options:
- `--no-llm` — skip LLM synthesis, use naive data extraction
- `--model <name>` — LLM model for exec summary (default: databricks-claude-sonnet-4)

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/generate_battlecard_slides_local.py` | L200 table slides (key differentiators comparison) |
| `scripts/generate_summary_slides_local.py` | Executive summary + product portfolio + tech summary slides |

## Parameter Reference (L200 Table Slides)

### Content Controls
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--rows-per-slide` | 4 | Max rows per category slide |
| `--max-details-per-side` | 2 | Max detail bullets per vendor cell |
| `--inline-details` | off | Render as "headline - detail. detail" |
| `--icon-style` | emoji | "emoji" (✅🔶🔴) or "text" (✓✓ ⚠ ✗) |
| `--no-strip-parens` | off | Keep parenthetical text in titles |

### Font Sizes (pt)
| Parameter | Default |
|-----------|---------|
| `--cat-title-font-size` | 19 |
| `--header-font-size` | 10 |
| `--col0-title-font-size` | 9 |
| `--col0-subtitle-font-size` | 8 |
| `--vendor-font-size` | 8.5 |
| `--banner-font-size` | 8 |
| `--title-slide-title-font-size` | 32 |
| `--title-slide-subtitle-font-size` | 13 |

### Colors
| Parameter | Default |
|-----------|---------|
| `--banner-bg-color` | #FF5F46 |
| `--header-bg-color` | #1A3A3A |
| `--col0-bg-color` | #F0F0F0 |
| `--col0-title-color` | #000000 |
| `--col0-subtitle-color` | #777777 |
| `--vendor-text-color` | #111827 |
| `--cat-title-color` | #1A3A3A |

### Layout (px)
| Parameter | Default |
|-----------|---------|
| `--table-width` | 690 |
| `--table-x` | 15 |
| `--table-y` | 62 |
| `--col-widths` | 170,260,260 |
| `--cat-title-y` | 18 |
| `--cat-title-height` | 44 |

## Troubleshooting

- **"No module found"** — run `uv sync` in the project root
- **"Google ADC auth failed"** — run the `gcloud auth` command above
- **Empty slides** — check that the battlecard has generated claims (visit the review app URL)
- **Port conflict** — the app imports `load_battlecard_slides` from `app.py`; no server needed
