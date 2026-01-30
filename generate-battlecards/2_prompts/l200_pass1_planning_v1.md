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
  "key_differentiator": "<3-6 word differentiator name — NO brand names or product names>",
  "description": "<short buyer-focused description: why this capability matters to end users/buyers — NO brand names, NO vendor comparisons>",
  "selection_reasoning": "<why this differentiator was selected — reference audience needs>",
  "rank_reasoning": "<why this rank position within its category>",
  "directive_alignment": "<which directive points this aligns with, or 'N/A'>",
  "databricks_rating": "strong_advantage|advantage|partial|disadvantage",
  "competitor_rating": "strong_advantage|advantage|partial|disadvantage"
}
```

## Rules
1. Generate exactly **{{diffs_per_category}} differentiators per product category**. The total count MUST equal {{total_diffs}}.
2. The `category` field MUST be one of the product categories listed above. Every category must have exactly {{diffs_per_category}} differentiators.
3. Within each category, rank differentiators 1-{{diffs_per_category}} by importance. The `rank` field resets to 1 for each category.
4. Databricks rating must be >= competitor rating for the majority of differentiators.
5. Be fair — include 1-2 areas per category where the competitor has genuine strengths.
6. Balance differentiators between what matters to C-suite (TCO, governance, strategic direction, vendor risk) and practitioners (performance, DX, tooling, open standards).
7. Align differentiators with the provided directives where applicable. Reference specific directive points in the `directive_alignment` field.
8. Each differentiator must have a unique `id` (format: Category_KeyDiff with underscores replacing spaces).
9. **NO BRAND NAMES**: The `key_differentiator` and `description` fields must NOT contain any brand names, product names, or branded feature names (e.g. do NOT say "Delta Live Tables", "Structured Streaming", "Fabric", "Event Streams", "Auto Loader", "Databricks", "Microsoft"). Use generic capability descriptions instead (e.g. "Native Change Data Capture" not "CDC with Delta Live Tables").
10. **BUYER-FOCUSED descriptions**: The `description` field should be short (under 15 words) and explain why this capability matters to the end buyer/user — NOT compare vendors or list feature names. Focus on the business outcome or pain point addressed.

Return ONLY the JSON array. No markdown fences, no explanation text.
