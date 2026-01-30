# Internal Battlecard: Competing Against Microsoft Fabric
**Classification:** Confidential - Databricks Internal Only
**Audience:** Account Executives (AEs) and Solutions Architects (SAs)
**Last Updated:** January 2025

---

## Strategic Positioning

1. **Adopt "Beat OneLake" not "Block OneLake"** - Say "OneLake is great… for some things" to stay in the conversation and play to our strengths. Blocking removes us from customer discussions.

2. **Partner in Public, Compete in Private** - Use joint/integration slides with Microsoft present. Use compete deck only when customer-only meetings and Microsoft is non-cooperative or intentionally excluding Databricks.

3. **Lead with Azure Databricks, Extend with Microsoft** - Position as: Azure Databricks for data platform + Power BI + AI Foundry + Fabric Data Factory for complementary workloads.

4. **Emphasize First-Party Status** - Azure Databricks is 1P on Azure with zero-config identity, networking, compute, storage. MSFT sellers receive equal compensation.

---

## Core Competitive Themes

5. **Data Management is Paramount** - Every business is disrupted by AI. Customers need an open, proven, reliable platform fit for production today AND future-proofing tomorrow. Databricks delivers this; Fabric is unproven and immature (2 years old).

6. **Fabric Marketing is 3-5 Years Ahead of Product Truth** - Acknowledge their marketing resonates (they're copying the best), but customers consistently report shortcomings and reliability issues. Don't bash; stick to facts.

7. **OneLake is Just Storage** - For Databricks, whether ADLS or OneLake, it's just storage. ADLS is proven, reliable, fast - continue to recommend ADLS by default.

---

## Technical Differentiators

8. **Unified Governance Wins** - Unity Catalog is the ONLY place to govern files, tables, ML models, and AI in a single tool. Set access once, respected across all workloads. Fabric requires setting access separately in each incompatible engine.

9. **Cost Model Advantage** - Azure Databricks: pay only for what you use. Fabric: use-it-or-lose-it capacity model bills for idle time, bursting penalties, and throttling issues. Emphasize $9K+/month savings potential.

10. **2-3X Performance with Best-in-Class TCO** - Photon delivers superior performance. One unified compute layer vs. Fabric's multiple inconsistent engines (Spark, KQL, SQL, Analysis Services).

11. **Table & Column-Level Lineage** - Deep visibility into data lifecycle captured live. Fabric only offers object-level lineage, unable to trace data properly.

12. **Data Discovery Across All Workloads** - Single source of truth vs. Fabric's fragmented data assets across multiple tools leading to duplication and lack of trust.

---

## Integration Story

13. **Bidirectional Data Flow is Key** - All Databricks data available in OneLake (via UC Mirroring - GA now). All OneLake/Fabric data accessible in Databricks (via OneLake Catalog Federation - Private Preview Jan 2026).

14. **No Pipeline Changes Required** - "None of your Databricks pipelines need to be changed for the data to be in OneLake." Use mirroring, not data migration.

15. **AI Better Together** - Azure AI Foundry (app & agent factory) + Azure Databricks (AI agent systems on your data). Integration via Genie, Agent Bricks, MCP, and OAuth OBO. OpenAI models GA, Genie as Copilot tool in November.

---

## Messaging: SHALL SAY ✓

16. **"All data in Azure should be open and accessible to all engines"** - Position as customer choice and flexibility.

17. **"Data managed in Databricks is available in OneLake via mirroring (GA)"** - Don't use term "write"; say "store" when discussing OneLake support.

18. **"All your Databricks data is available to Microsoft AI and Power BI"** - Emphasize existing seamless integration.

19. **"These are not one-way doors"** - Customers can choose the system and engine that suits their needs. Fabric is built on Delta Lake, so future flexibility exists.

---

## Messaging: SHALL NOT SAY ❌

20. **Never bash products** - Don't say "OneLake/Fabric is not secure" or similar negative statements about Microsoft products.

21. **Don't create false requirements** - Don't say "You need to land data in ADLS to use Databricks." OneLake support is coming.

22. **Don't claim policy sync is solved** - Be honest: "Many customers push policies to Unity Catalog with tools like Immuta or Purview. We're working on industry standards; until then external tools are needed."

---

## Objection Handling

23. **"When will Databricks write to OneLake?"** → "It is already available in OneLake via mirroring, just like M365 data."

24. **"Do I need OneLake for AI/Power BI?"** → "No. All Azure Databricks data integrates with Microsoft AI and Power BI today through existing first-party connectors."

25. **Customer wants OneLake to simplify operational overhead** → "We're adding support for OneLake storage (Jan 2026). From your experience perspective, nothing changes. ADLS or OneLake - for Databricks it's just storage, and we'll continue to recommend proven, reliable ADLS by default."

---

## Call to Action

- **Use the Simple OneLake Narrative** in every Microsoft, customer, and partner conversation
- **Keep tone positive, factual, confident** - Never defensive
- **Leverage resources**: go/redfish site, Azure Databricks Story slides, #azuredatabricks, #azure-compete Slack channels
- **NDA Required**: Do not leave compete slides behind or share via email
