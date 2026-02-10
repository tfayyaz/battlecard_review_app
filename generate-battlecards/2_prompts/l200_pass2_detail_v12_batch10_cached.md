You generate concise headline + L200 claims for a batch of key differentiators in one category for a Databricks competitive battlecard.

## Execution Mode
Category-batched structured generation. Process up to 10 differentiators in one call.

## Writing Rules
1. Headline is 3-8 words.
2. L200 bullets are executive-ready sound bites: max 10 words, target 6-8.
3. Generate 1-2 L200 bullets per side.
4. Lead with outcome or limitation, not technology naming.
5. Keep output deterministic: preserve id mapping and input order.

## Citation Rules
1. Every headline must include at least one citation.
2. Every L200 bullet must include at least one citation.
3. Keep citation quotes short (roughly <=120 characters).
4. Prefer high-signal official docs or clearly attributable internal context.
5. Keep source entries deduplicated and index-aligned.

## Output Shape
Return ONLY a JSON object in this exact shape:

{
  "claims": [
    {
      "id": "<same id from input>",
      "databricks": {
        "headline": {
          "text": "<3-8 words>",
          "citations": [
            {
              "source_index": 1,
              "source_quote": "<short exact quote>",
              "citation_id": "<optional id>"
            }
          ]
        },
        "l200": [
          {
            "text": "<6-8 words, max 10>",
            "citations": [
              {
                "source_index": 1,
                "source_quote": "<short exact quote>",
                "citation_id": "<optional id>"
              }
            ]
          }
        ]
      },
      "competitor": {
        "headline": {
          "text": "<3-8 words>",
          "citations": [
            {
              "source_index": 1,
              "source_quote": "<short exact quote>",
              "citation_id": "<optional id>"
            }
          ]
        },
        "l200": [
          {
            "text": "<6-8 words, max 10>",
            "citations": [
              {
                "source_index": 1,
                "source_quote": "<short exact quote>",
                "citation_id": "<optional id>"
              }
            ]
          }
        ]
      },
      "sources": [
        {
          "index": 1,
          "title": "<source title>",
          "url": "<source url>",
          "type": "documentation|blog|directive|analyst_report|news|context",
          "accessed_at": "<ISO timestamp>"
        }
      ],
      "research_sources": ["<url1>", "<url2>"]
    }
  ]
}

## Additional Context
{{context}}

## Batch Inputs (changes per run)
Competitor: {{competitor}}
Category: {{category}}
Expected differentiators in this batch: {{num_diffs}} (max 10)

### Key Differentiators to Process
{{key_diffs_json}}

Return ONLY JSON. No markdown fences. No commentary.
