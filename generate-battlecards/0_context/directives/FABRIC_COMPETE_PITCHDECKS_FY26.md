# Fabric Compete Documentation

## Document 1: Fabric Compete (2025-12-03)
**Author:** Robert Saxby - Dir. Product
**Classification:** Internal use only

### Summary - Competing against Fabric

#### Three Key Themes:

**1. Data management is paramount**
- Every business is getting disrupted by AI
- Data is the fuel
- Customers need an open, proven, reliable platform
- Databricks is that platform
- Fabric is unproven and immature
- Customers can't afford to take the risk

**2. Fabric Marketing is 3-5 years ahead of product truth**
- Fabric is trying to imitate Databricks
- Their marketing resonates (copying from the best)
- Fabric is 2 years old, marketing is 3-5 years ahead
- Customers consistently report shortcomings and reliability issues

**3. OneLake is just storage**
- All Databricks is accessible in Fabric
- Now, any OneLake data is easily accessible in Databricks

### Azure Databricks - A few things to remember

- Azure Databricks is 1P (first-party) on Azure
- 1P integrations including:
  - Zero-config Identity, Networking, Compute, Storage
  - Confidential Compute
  - Unified customer billing
- MSFT sellers are incentivized and receive equal compensation
- Customers have choice; partnership approach to ensure success

### Customers have choice - We are there to advise them

**Key Points:**
- Every business is getting disrupted by Gen AI and Agents
- Customer's data is the fuel
- Data management is paramount
- Customers need an open, proven, reliable platform
- Needed for production today AND future-proofing for tomorrow

**Competitive Positioning:**
- Fabric is trying to imitate Databricks
- Marketing resonates (copying from the best)
- Fabric is 2 years old, marketing is 3-5 years ahead of product truth
- Customers consistently report on Fabric's shortcomings and reliability issues
- Databricks is customer obsessed and also 1P on MSFT
- Only Azure Databricks is:
  - Fit for production today
  - Future-proofing for tomorrow

### OneLake != Fabric - It's just storage

**Background:**
- Fabric launched as a SaaS platform
- Trying to build from Power BI user base
- ADLS comes with operational overhead
- OneLake was required to simplify this for Fabric
- Databricks already does this (and more) through UC
- MSFT makes it easier for MSFT data to land in OneLake
- Primary intention: make it easily consumable by Fabric
- It's now just as easy to consume in Databricks

**Interoperability:**
- Already, all Databricks is accessible in Fabric
  - As a secure, open platform, Databricks enables this
- Now, any OneLake data is accessible in Databricks
  - OneLake tables can read like any other external table
  - If customers need all benefits of Databricks tables → upgrade to UC Managed Tables

**About Writes to OneLake:**
- **Don't use the term "write"**; instead say **"store"**
- Working to add support for OneLake storage
- From customer experience perspective, nothing changes
- How you access Databricks data in Fabric will not change

**Customer Perspective:**
- A few customers want OneLake instead of ADLS to simplify operational overhead (storage policies, lifecycle management)
- This messaging comes directly from MSFT
- MSFT trying to make it easier for customers and provide additional value to make workloads more sticky to Azure
- MSFT has a lot of work to do - ADLS is massively adopted
- For Databricks: ADLS or OneLake - **it is just storage**
- **ADLS is proven, reliable, fast** - will continue to default and recommend ADLS

### Shall Say / Shall Not Say

#### Shall Say ✓
- All data in Azure should be open and accessible to all engines
- Lead with Azure Databricks, extend with OneLake, AI Foundry, Power BI, etc
- Data managed in Databricks is available in OneLake (UC mirroring, GA)
- Data stored in OneLake will be available in Databricks (Private Preview Jan 2026)
- These are not one-way doors, choose the system and engine that suits business needs
- "Synching" policies between the two systems requires external tools today (Purview etc.)

#### Shall Not Say ❌

**Databricks won't say:**
- No product bashing - "OneLake is not secure", "Fabric is not secure", etc.
- "You need to land data in ADLS to use with Databricks"

**Microsoft won't say:**
- "OneLake is required to do AI over Databricks data"
- "ADB only connects to AI and PowerBI through OneLake"
- "You need to change all your data pipelines to store/migrate your data in OneLake"

### Why should customers choose Databricks?

**Four Key Reasons:**

**1. Govern all your enterprise data with Federation**
- Access data in OneLake, Glue, Horizon and Iceberg REST Catalogs from Unity Catalog
- NOTE: MSFT has poisoned the word "federation"
  - Catalog Federation = Zero Copy Mirroring
  - SDP = Replication Mirroring

**2. Govern Data & AI in one place**
- UC is the ONLY place to govern files, tables, ML models, and more in a single tool

**3. Interoperate with all of your tools with Open APIs**
- Access any table from any client with Unity and Iceberg REST APIs
- Regardless of file format
- Including Microsoft Fabric

**4. Automated data classification and governance**
- Automatically detect PII and other sensitive fields
- Leverage ABAC to restrict access based on attributes like PII

### Architecture Integration

**Microsoft Fabric components:**
- Synapse Data Warehousing, Engineering, Factory, Science
- Synapse Real Time Analytics, Power BI
- Serverless compute (T-SQL, Spark, KQL, Analysis Services)
- OneSecurity
- Warehouse, Lakehouse, Kusto DB, Dataset
- OneLake (foundation)

**Azure Databricks components:**
- Agent Bricks (Agentic AI & Machine Learning)
- AI/BI (Agentic Business Intelligence)
- Apps (Secure data & AI apps)
- Lakeflow (Ingest, ETL, streaming)
- DB SQL (Data warehousing)
- Lakebase (Transactional database)
- Unity Catalog (foundation)

**Key integration:**
- Mirroring between OneLake and Unity Catalog
- All Data in OneLake & Unity Catalog available to engine of choice
- Available across Microsoft Business, BI and AI surfaces (Teams, PowerBI, PowerApps, Power Automate, Copilot, Foundry)
- Both systems sit on ADLS (Azure Data Lake Storage)

---

## Document 2: INVEST - Microsoft GTM Update (October 2025)
**Presenter:** David Meyer - SVP Product, Databricks
**Classification:** Confidential - Databricks internal only - Do not share

### Strategic shift in OneLake Positioning

#### Block vs Beat Strategy

**BLOCK ONELAKE ❌**
- MSFT Messaging = "OneLake Central"
- Blocking OneLake removes us from the conversation

**BEAT ONELAKE ✓**
- "OneLake is great… for some things" = We're back in conversation and can play to our strengths

### Simple OneLake Narrative
**Use the right tool for the job**

**Integration Model:**
- All Azure Databricks data available in OneLake (via Mirroring - current)
- All Fabric/OneLake data available in UC (via "Mirroring"* - Jan 2026)
  - *via OneLake Cat Federation

**Foundation:** ADLS

### MSFT Unify Data GTM for Redfish Accounts

**For Redfish Accounts and Accounts who choose Databricks as Data Platform**

*Note: Accounts = Tenants/Departments not TPIDs*

**Three-Step Approach:**

**1. Grow ADB ACR + Complement with PowerBI, AI Foundry, and Azure Databases**
- Azure Databricks + Complement with:
  - Power BI
  - Azure AI Foundry
  - Fabric Data Factory
  - Fabric RTI
  - Azure Databases

**2. Unify your Data Estate in Microsoft OneLake**
- Unified, Governed, and Secure Data Foundation
- Unity Catalog ↔ Mirroring ↔ OneLake
- ADLS
- Hybrid and Multi-Cloud Data Estates
- Shortcut | Mirror | Native Read and Write

**3. Educate on Fabric when customer initiates interest**

### Don't be defensive, stick to the facts

**Key Messages:**

- "OneLake is great… for some things"
- "None of your Databricks pipelines need to be changed for the data to be in OneLake"
- "All of your Databricks data is available to MSFT AI and Power BI"

**Q: When will Databricks write to OneLake?**
- "It is already available in OneLake, just like M365, via mirroring"

**Q: How do I sync access policies between OneLake and Unity Catalog?**
- "Many of our customers push policies to Unity Catalog with tools like Immuta or Purview. We are working on industry standards for policy sync, until those come you will need external tools"

### New ERA of MSFT AI GTM

#### Get the best AI of both worlds

**Azure AI Foundry** (The AI app & agent factory) ↔ **Azure Databricks** (AI agent systems based on your data)

**Integration Points:**

1. **Foundry Agent Service + Genie space**
   - GenAI models & classic ML (frontier and OSS)

2. **Foundry models + Mosaic AI agent tools** (coming soon)
   - Agents connected to data and tools (RAG + MCP)

3. **Foundry tools + Mosaic AI agent**
   - Evaluation, accuracy, product readiness

4. **Foundry Observability + ADB Observability & Eval**
   - Orchestration governance & management (MLOps)

**Azure AI Foundry Capabilities:**
- Premier & open models
- Knowledge with agentic RAG
- Comprehensive agent toolchain
- Secure, trustworthy AI

#### AI Better Together Timeline

**OpenAI:**
- Models: **GA end October**

**Copilot:**
- Agent: **Genie as Tool: November**

**Foundry:**
- Agent: **Genie | Agent as Agent: Fast Follow**

**Key enabler:** MCP and OAuth OBO allows custom configurations

**Integration with Databricks:**
- Genie
- Agent Bricks
- Custom Agent

### Call to Action - Top 3 next steps

**1. Adopt the new message**
- Shift from "blocking OneLake" to "beating OneLake through partnership"
- Reinforce that Databricks and OneLake are complementary, delivering customer success and choice

**2. Use the Simple OneLake Narrative and Shall Say Guide**
- Keep tone positive, factual and confident
- Use in every conversation with Microsoft, customers and partners

**3. Leverage the latest resources and experts**
- Use the new Redfish site (go/redfish) and Azure Databricks Story slides
- Get answers on #azuredatabricks, #azure-compete

---

## Document 3: [EXTERNAL] Azure Databricks Comparison with Microsoft Fabric

**Date:** 2024
**Classification:** NDA Required - Do Not Distribute

### How to Use This Deck

**Important Guidelines:**
- **MANDATORY:** Do not leave these slides behind
- Do not share the slides directly (e.g., over email)
- Appendix are optional topic-specific slides
- NDA must be signed with Databricks
- **Intended Audience:** Customers

**Usage Decision Tree:**
"Partner in Public, Compete in Private"

1. Meet with customer to understand their pain points
2. Determine who you're talking to:
   - **Customer + Microsoft or SI Partner:**
     - Use Joint Slides & Integration Slides

   - **Customer only:**
     - Is Microsoft or SI Partner Cooperative?
       - **Yes:** Use FY25 Compete Deck with CUSTOMER
       - **No, mutually agreed to compete:** Use Gloves Off Deck with CUSTOMER (this deck)
       - **No, Microsoft intentionally excluding Databricks:** Use Gloves Off Deck with CUSTOMER (this deck)
       - **No, SI Partner-led and intentionally excluding Databricks:** Use Gloves Off Deck with CUSTOMER (this deck)

### Azure Databricks is an integral first-party part of the Microsoft ecosystem

**Unifying Data & AI with the Databricks Data Intelligence Platform on Azure**

**Integrations with:**
- Copilot Studio
- Event Hubs
- AI Foundry
- Azure Data Lake Storage
- Power Platform
- Power BI
- Data Factory
- Azure OpenAI
- Excel
- Dataverse

### Databricks Data Intelligence Platform on Azure

**Platform Components:**

**Data Workloads:**
- Agent Bricks (Artificial intelligence)
- DB SQL (Data warehousing)
- Lakebase (Transactional database)
- AI/BI (Unified intelligence)
- Lakeflow (Ingest, ETL, streaming)
- Apps (Secure data & AI apps)
- Marketplace (Data & AI marketplace)

**Foundation:**
- Unity Catalog
- Delta Lake & Iceberg

**Why Customers Choose:**

**Customers:**
- Choose Azure Databricks for the unified platform, strong governance, and simplicity
- Stay for its unparalleled scale and lowest TCO

**Developers:**
- Choose Azure Databricks for choice of languages, open-source core, and latest AI models
- Stay for its rapid pace of innovation

### Adopted by thousands of global enterprises

**Customer Logos Include:**
- ABN AMRO, Adobe, Ahold Delhaize, AT&T
- Barilla, Bayer, bp, Dell
- Estée Lauder, ExxonMobil, General Motors, GSK
- Hershey, Johnson & Johnson, MARS, Mercedes-Benz
- Michelin, Reckitt, Shell, Swiss Re, Walgreens

### Azure Databricks is Enterprise-Ready Today

**Comparison Table:**

| Category | Azure Databricks | Microsoft Fabric |
|----------|------------------|------------------|
| **Security** | | |
| Unified Governance - Set once, respected across all workloads | ✓ | ✗ |
| Data Lineage - End-to-end data lifecycle visibility | ✓ | ~ Object-level, not table or column |
| Data Discovery - Single source of truth | ✓ | ✗ Fragmented data assets |
| **Manageability** | | |
| Unified Platform for Data & AI - One platform for all workloads | ✓ | ~ Multiple tools for many ML & Gen AI workloads |
| Centralized Access Control - Manage access to data & AI in one place | ✓ | ✗ Manage access separately in each engine |
| Unified Storage Layer - All data stored in an open file format | ✓ | ✓ |
| **Cost Savings** | | |
| Pay for What You Use - Only pay for compute you use | ✓ | ✗ Use-it-or-lose it capacity model |
| Best-in-class TCO - Reduce your costs with better performance | ✓ | ✗ |

### Security You Can Trust

**Three Key Advantages:**

**1. Set access once, respected everywhere → data you can trust**
- Fabric: Set access in each, incompatible engine → exposure to data privacy risk

**2. Table & column-level lineage, captured live → deep visibility into your data**
- Fabric: Object-level lineage only → unable to trace data lifecycle

**3. Discover data assets across all workloads → faster pace of innovation**
- Fabric: Data assets fragmented across Fabric objects → duplication of data, lack of trust

### More Than Meets the (U)I
**Don't waste time maintaining multiple tools, engines, governance layers**

**Azure Databricks: One Product, Right Tool for the Job**
- Data Engineering & Real-Time Analytics: **Lakeflow**
- Data Warehousing: **Databricks SQL**
- Data Science & AI: **Mosaic AI**
- **Unified Data & AI Governance:** Unity Catalog
- **Unified, Performant Compute Layer:** Photon
- **Open Storage Layer:** ADLS Gen 2

**Microsoft Fabric: Multiple Products & Tools Per Workload**
- Data Engineering: Data Engineering & Data Flows
- Real-Time Analytics: Event Streams & KQL Queryset
- Data Warehousing: Data Warehouse & Lakehouse
- Data Science & AI: Data Science, Azure ML, Azure AI Foundry
- **Engine-Specific Governance:**
  - OneLake Data Access Roles
  - KQL Database Security Roles
  - SQL Granular Permissions
  - Power BI RLS/CLS
- **Multiple Compute Engines, Inconsistent Performance:** Spark, KQL, SQL, Analysis Services
- **Proprietary, Coupled Storage Layer:** OneLake

### On-Prem Economics: Tightly Coupled Compute & Storage

**Fabric Capacity Model:**
- Idle Time Billed
- Bursting Above Capacity
- Compute Used
- **Issues:**
  - Use-it-or-lose-it capacity license model means you get billed for idle time
  - Not fast enough? Fabric will eat tomorrow's lunch, today with bursting. And bill you for it next week
  - Shared compute + throttling. Isn't that just the same as good ole fashion resource contention?
  - OneLake tightly coupled compute & storage means if you turn it off, your data goes with it

**Azure Databricks:**
- Compute Used (2-3X faster performance)
- Pay for only what you use

**1 Month Total Cost Example:**

**Fabric:**
- $22K utilized
- $32K billed
- BI workloads throttled

**Databricks:**
- $23K utilized
- $23K billed
- No throttling

**$9K/mo saved in idle compute when using Azure Databricks**

### Here's the good news

**Two Key Points:**

✓ **Fabric (Power BI, Data Factory) already seamlessly integrate with Azure Databricks**

✓ **Fabric is built on Delta Lake**, so if they address these (3) issues in the future, you can leverage these investments with Fabric later

**Architecture Diagram:**
- Shows Delta Lake as foundation
- Open and Governed Data Lakehouse
- Integration between Data Integration, Data Engineering, Data Science, Data Warehouse, Real-time & streaming analytics, and Business Intelligence
- Hybrid and Multi-Cloud Data Sources (Appliances, Cloud DW, Databases, Hadoop)

---

## Key Takeaways Across All Documents

### Strategic Positioning
1. **Partnership over Competition:** Shift from blocking OneLake to beating it through complementary positioning
2. **Product Maturity Gap:** Fabric's marketing is 3-5 years ahead of product capabilities
3. **Data Management is Paramount:** Emphasize governance, security, and reliability

### Technical Advantages
1. **Unified Governance:** Unity Catalog provides single-point governance across all workloads
2. **Cost Model:** Pay-for-what-you-use vs. capacity-based pricing
3. **Performance:** 2-3X faster performance with Photon
4. **Open Architecture:** ADLS + Unity Catalog vs. closed OneLake ecosystem

### Integration Strategy
1. **Bidirectional Data Flow:** Databricks data available in OneLake (GA), OneLake data accessible in Databricks (Jan 2026)
2. **AI Integration:** Azure AI Foundry + Azure Databricks for complete AI lifecycle
3. **Microsoft Ecosystem:** First-party integrations with Power BI, AI Foundry, Event Hubs, etc.

### Messaging Guidelines
**Do Say:**
- "OneLake is great… for some things"
- All data should be open and accessible
- Lead with Azure Databricks, extend with Microsoft services
- These are not one-way doors

**Don't Say:**
- Product bashing about security
- OneLake is required for AI/PowerBI
- Need to migrate data to OneLake
