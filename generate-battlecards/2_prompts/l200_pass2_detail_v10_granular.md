You generate L200 and L300 details with granular citations for ALL key differentiators in the **{{category}}** category for Databricks vs {{competitor}}.

## Execution Mode
Category-batched structured generation. Process all differentiators for this category in one call.
You will receive {{num_diffs}} differentiators for {{category}}.

## Category
{{category}}

## Key Differentiators to Process
{{key_diffs_json}}

## Additional Context
{{context}}

## Writing Rules
1. L200 bullets are executive-ready sound bites: max 10 words, target 6-8.
2. L300 bullets add technical proof and should be concrete and specific.
3. Be fair and factual for both Databricks and competitor.
4. Keep output deterministic and stable: preserve id mapping and input order.

## Citation Rules
1. Every factual statement must include at least one citation.
2. Each headline needs citations.
3. Each L200 bullet needs citations.
4. Each L300 bullet needs citations.
5. Reasoning needs citations.
6. Use exact source quotes and source indexes that map to the sources array.

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
              "citation_id": "cite_db_headline_0",
              "start_index": 0,
              "end_index": 20,
              "source_index": 1,
              "source_quote": "<exact quote>",
              "verdict": "verified|unverified|disputed|outdated",
              "confidence": 0.0,
              "verdict_rationale": "<short rationale>"
            }
          ]
        },
        "l200": [
          {
            "text": "<6-8 words, max 10>",
            "citations": [
              {
                "citation_id": "cite_db_l200_0_0",
                "start_index": 0,
                "end_index": 20,
                "source_index": 1,
                "source_quote": "<exact quote>",
                "verdict": "verified|unverified|disputed|outdated",
                "confidence": 0.0,
                "verdict_rationale": "<short rationale>"
              }
            ],
            "l300": [
              {
                "text": "<technical detail bullet>",
                "citations": [
                  {
                    "citation_id": "cite_db_l300_0_0",
                    "start_index": 0,
                    "end_index": 20,
                    "source_index": 1,
                    "source_quote": "<exact quote>",
                    "verdict": "verified|unverified|disputed|outdated",
                    "confidence": 0.0,
                    "verdict_rationale": "<short rationale>"
                  }
                ]
              }
            ]
          }
        ],
        "reasoning": {
          "text": "<1-2 sentences>",
          "citations": [
            {
              "citation_id": "cite_db_reasoning_0",
              "start_index": 0,
              "end_index": 40,
              "source_index": 1,
              "source_quote": "<exact quote>",
              "verdict": "verified|unverified|disputed|outdated",
              "confidence": 0.0,
              "verdict_rationale": "<short rationale>"
            }
          ]
        }
      },
      "competitor": {
        "headline": {"text": "<3-8 words>", "citations": []},
        "l200": [
          {
            "text": "<6-8 words, max 10>",
            "citations": [],
            "l300": [
              {"text": "<technical detail bullet>", "citations": []}
            ]
          }
        ],
        "reasoning": {"text": "<1-2 sentences>", "citations": []}
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

Return ONLY JSON. No markdown fences. No commentary.
