You plan L200 critical differentiator slides for a **platform battlecard** comparing Databricks to {{competitor}}.

## Context
Competitor: {{competitor}}
Product area: {{product_area}}
Comparison: {{comparison}}

## Audience
This battlecard is for **C-suite executives** (CIO, CTO, CDO, VP Data/AI) and **data/ML/AI practitioners** (data engineers, ML engineers, analytics engineers, data scientists). Key differentiators must resonate with both audiences:
- C-suite cares about: strategic platform direction, total cost of ownership, vendor risk, governance posture, time-to-value, and AI/ML readiness.
- Practitioners care about: performance benchmarks, developer experience, tooling maturity, open standards, and operational reliability.

## Product Categories
The following product categories define the structure of this platform battlecard. You MUST generate exactly **{{diffs_per_category}} key differentiators per category**, for a total of {{total_diffs}} differentiators.

{{product_categories}}

## Key Differentiator Theme Guidance
Use the following themes as guidance when selecting key differentiators. Most differentiators should map to one of these themes, but you MAY choose differentiators outside these themes for specific product categories where it makes sense — justify with clear reasoning about who cares and why.

| Theme | Who Cares | Why It Matters |
|-------|-----------|----------------|
| Serverless | Both | Operational simplicity, no capacity planning, instant scale |
| Open Formats & Standards | Both | Avoid lock-in, portability, future-proof investments |
| AI Assistants | Both | Productivity, natural language access to data |
| Intelligent Optimization | Both | Self-tuning, auto-scaling, ML-driven performance |
| Performance & TCO | C-suite + Practitioners | Query speed and cost efficiency together |
| Unified Platform | C-suite | Fewer tools, lower integration complexity |
| Interoperability | Practitioners | Works with existing tools and data sources |
| Governance & Security | C-suite | Access controls, lineage, compliance (HIPAA, SOC2, GDPR) |
| Real-time & Streaming | Practitioners | Handle streaming alongside batch workloads |
| Data Sharing | Both | Cross-org collaboration without copying data |
| Developer Experience | Practitioners | IDE support, debugging, CI/CD, daily productivity |
| GenAI & LLM Support | Both | RAG, vector search, fine-tuning, compound AI systems |
| Pricing Transparency | C-suite | Predictable costs, no surprise bills |
| Multi-cloud | C-suite | Vendor risk mitigation, cloud flexibility |

## Directives
{{directives}}

## Additional Context
The following context documents are provided as XML-tagged sections.
Each tag indicates the document type (competitive_directive, battlecard_archive, product_categories, key_differentiator_themes, or context).
Attributes include the document name, type classification, scope, and whether it was human_provided or agent_generated.
Use these documents to inform your analysis. Cite them as sources where relevant.

{{context}}

## Task
Generate exactly **{{diffs_per_category}} key differentiators per product category** ({{total_diffs}} total).
For each, provide ONLY the planning skeleton — detailed headlines, details, reasoning, citations, and sources will be generated separately in a second pass.

## Output Format
Return ONLY a JSON array. Each object must have this exact shape:

```json
{
  "id": "<Category>_<Differentiator> with underscores",
  "competitor": "{{competitor}}",
  "category": "<product category from the list above>",
  "rank": 1,
  "key_differentiator": "<2-4 word differentiator name — see naming rules below>",
  "description": "<1 sentence, max 12 words — the benefit or value this capability delivers to buyers>",
  "selection_reasoning": "<who cares about this (C-suite, practitioners, or both) and WHY it matters for platform selection>",
  "rank_reasoning": "<why this rank position within its category>",
  "directive_alignment": "<which directive points this aligns with, or 'N/A'>",
  "databricks_rating": "strong_advantage|advantage|partial|disadvantage",
  "competitor_rating": "strong_advantage|advantage|partial|disadvantage"
}
```

## Key Differentiator Naming Rules — CRITICAL

The `key_differentiator` field is the title shown on the battlecard slide. It must be:

1. **2-4 words maximum** (ideally 2-3). This is a category label, not a sentence.
2. **NO brand names, product names, or feature names** (e.g. NOT "Delta Live Tables", NOT "Fabric Capacity", NOT "Databricks Unity Catalog").
3. **Generic capability labels** that any platform could be evaluated against.
4. **Title case** formatting.

### Examples of GOOD key_differentiator names:
- "Ease of Use"
- "Data Connectors"
- "Semantic Understanding"
- "Conversational Interface"
- "Model Optionality"
- "Admin Monitoring"
- "Agentic Capabilities"
- "Query Performance"
- "Cost Transparency"
- "Streaming Support"
- "Data Governance"
- "Open Standards"
- "Auto-Optimization"
- "Multi-Cloud Support"
- "Developer Tooling"

### Examples of BAD key_differentiator names — DO NOT write like this:
- "Native Change Data Capture Pipeline Support" (too long, too specific)
- "Serverless SQL Warehouse Auto-Scaling" (too long, includes branded concepts)
- "Delta Lake Open Table Format" (brand name)
- "End-to-End ML Lifecycle Management" (too wordy)
- "Advanced Real-Time Stream Processing Engine" (too long, too technical)
- "Unified Data Governance and Lineage Tracking" (too long)
- "Built-in Vector Search and Embedding Support" (too long, too specific)

## Description Rules

The `description` field explains WHY this differentiator matters. It must be:

1. **One sentence, max 12 words.**
2. **Benefit-focused** — what value does this deliver to the buyer?
3. **No brand names or vendor comparisons.**
4. **No technical jargon** — write for a VP, not an engineer.

### Examples of GOOD descriptions:
- "Getting started and initial setup experience."
- "Variety of supported data sources and connectors."
- "Default usage of metadata for intelligent queries."
- "Prompt/response history and conversational context."
- "Test performance against ground truth data."
- "Verified answers to anticipated questions."
- "Choose and configure the underlying LLM."
- "Predictable billing with no hidden costs."
- "Run on any cloud without re-architecture."
- "Self-tuning performance without manual intervention."

### Examples of BAD descriptions — DO NOT write like this:
- "Databricks provides a comprehensive unified governance layer using Unity Catalog that enables fine-grained access controls across all data and AI assets." (way too long, has brand names)
- "The platform supports multiple open-source table formats including Delta Lake, Apache Iceberg, and Apache Hudi for maximum data portability." (too long, lists brand names)
- "Leverages advanced ML-driven optimization algorithms to automatically tune query execution plans and cluster configurations." (too technical, too long)

## Selection Reasoning Rules

The `selection_reasoning` field must clearly state:
1. **WHO cares** — C-suite, practitioners, or both
2. **WHY it matters** for choosing a data platform — the business or technical pain it addresses

Example: "Practitioners care — slow query performance directly impacts development velocity and time-to-insight. C-suite cares because poor performance leads to higher compute costs."

## Rules
1. Generate exactly **{{diffs_per_category}} differentiators per product category**. The total count MUST equal {{total_diffs}}.
2. The `category` field MUST be one of the product categories listed above. Every category must have exactly {{diffs_per_category}} differentiators.
3. Within each category, rank differentiators 1-{{diffs_per_category}} by importance. The `rank` field resets to 1 for each category.
4. Databricks rating must be >= competitor rating for the majority of differentiators.
5. Be fair — include 1-2 areas per category where the competitor has genuine strengths.
6. Balance differentiators between what matters to C-suite (TCO, governance, strategic direction, vendor risk) and practitioners (performance, DX, tooling, open standards).
7. Align differentiators with the provided directives where applicable. Reference specific directive points in the `directive_alignment` field.
8. Each differentiator must have a unique `id` (format: Category_KeyDiff with underscores replacing spaces).
9. **Most differentiators should map to the theme guidance table above**, but you may introduce themes outside that list when they are clearly important for a specific product category. Always justify with who cares and why.
10. **NO BRAND NAMES** in `key_differentiator` or `description` fields. Use generic capability labels.
11. **KEY DIFF NAMES must be 2-4 words** (ideally 2-3). Think of them as column headers in a comparison table, not sentences.
12. **DESCRIPTIONS must be benefit-focused**, max 12 words. Explain why the capability matters, not what it technically does.

Return ONLY the JSON array. No markdown fences, no explanation text.
