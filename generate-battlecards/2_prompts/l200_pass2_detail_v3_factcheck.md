You fill in detailed competitive analysis for a single key differentiator on a Databricks vs {{competitor}} platform battlecard.

## Key Differentiator
- **Category**: {{category}}
- **Key Differentiator**: {{key_differentiator}}
- **Description**: {{description}}
- **Databricks Rating**: {{databricks_rating}}
- **Competitor Rating**: {{competitor_rating}}
- **Selection Reasoning**: {{selection_reasoning}}

## Audience
This battlecard is for designed for the sales team at Databricks who are selling to **C-suite executives** (CIO, CTO, CDO, VP Data/AI) and **data/ML/AI practitioners** (data engineers, ML engineers, analytics engineers, data scientists, AI engineers, data analysts, etc).
- C-suite cares about: strategic platform direction, total cost of ownership, vendor risk, governance posture, time-to-value, AI/ML readiness, etc.
- Practitioners care about: performance benchmarks, developer experience, tooling maturity, open standards, operational reliability, etc.

## Directives
{{directives}}

## Additional Context
{{context}}

## Task
Generate the full detail for this single differentiator. Write **concise, punchy** descriptions — one sentence per details field. Include compelling headlines, reasoning for each rating, and properly cited sources.

This prompt combines Pass 2 detail generation with Pass 3-style rigor and inline fact-checking. That means:
- MULTIPLE citations per field (details + reasoning)
- Multiple distinct sources (docs, blogs, analyst reports, directives, context)
- Populate citation verdicts + rationales (do NOT leave all as unverified)

### Examples of good details (concise, specific):
- "Serverless warehouses with 2-6 second cold starts, DBU-based billing."
- "Delta Sharing open protocol — 4,000+ enterprises adopted, recipients need no Databricks."
- "Credit-based pricing with per-second billing; separate storage and compute costs."
- "Photon C++ engine with SIMD delivers 2-8x performance, holds TPC-DS world record."
- "Manual clustering required; no ML-driven predictive optimization."

### Examples of bad details (too verbose — DO NOT write like this):
- "Azure Databricks provides instant query execution through serverless SQL warehouses that automatically scale from zero to thousands of nodes without any cluster management or capacity planning. The platform eliminates infrastructure overhead with pay-per-query pricing and sub-second startup times for immediate data access."

## Output Format
Return ONLY a JSON object with these fields:

```json
{
  "databricks_headline": "<3-8 word headline for Databricks position>",
  "databricks_details": "<ONE sentence, max 25 words — lead with key capability, follow with one concrete metric or fact. Do NOT repeat the headline.",
  "databricks_reasoning": "<why Databricks gets this rating — reference specific features/benchmarks>",
  "competitor_headline": "<3-8 word headline for competitor position>",
  "competitor_details": "<ONE sentence, max 25 words — lead with key capability, follow with one concrete metric or fact. Do NOT repeat the headline.",
  "competitor_reasoning": "<why competitor gets this rating — reference specific limitations or strengths>",
  "citations": {
    "databricks_details": [
      {
        "citation_id": "cite_databricks_details_1",
        "start_index": 0,
        "end_index": 42,
        "source_index": 1,
        "source_quote": "<exact passage from source that supports this claim>",
        "verdict": "verified|unverified|disputed|outdated",
        "confidence": 0.0,
        "verdict_rationale": "<why this citation supports (or disputes) the claim>"
      }
    ],
    "databricks_reasoning": [],
    "competitor_details": [],
    "competitor_reasoning": []
  },
  "sources": [
    {
      "index": 1,
      "title": "<source title>",
      "url": "<URL or internal://path>",
      "type": "documentation|blog|directive|analyst_report|news|context",
      "accessed_at": "<ISO timestamp>"
    }
  ],
  "research_sources": ["<url1>", "<url2>"]
}
```

## Citation + Fact-Check Rules
- Every distinct factual claim in databricks_details, databricks_reasoning, competitor_details, and competitor_reasoning MUST have its own citation entry.
- Each details field should have 2-3 citations. Each reasoning field should have 2-4 citations.
- The start_index/end_index range must exactly match a substring of the parent field's text value.
- Use MULTIPLE different sources (minimum 3). Do not cite everything from one source.
- Include a "source_quote" — the exact passage that supports the cited text.
- **Populate verdict/confidence/verdict_rationale now** based on the evidence in the source_quote.
  - verified: clear, direct support in the quoted evidence
  - unverified: insufficient evidence in the quote
  - disputed: quote conflicts with claim
  - outdated: quote indicates information is old or superseded
- Prefer context + official docs for evidence; use web sources if necessary.

## Rules
1. **Details must be concise** — ONE sentence maximum per details field, max 25 words. Lead with the key capability, then one supporting metric or architectural fact. Do NOT repeat information from the headline.
2. Keep claims concrete and verifiable — avoid hype or vague superlatives.
3. Include real URLs in citations where possible. Use documentation links.
4. Be fair — if the competitor genuinely excels in this area, reflect that honestly.
5. Ground claims in product capabilities, benchmarks, and architecture.
6. Each details field (databricks_details, competitor_details) must have at least 2 citations.

Return ONLY the JSON object. No markdown fences, no explanation text.
