<xml slide="01" title="LAKEFLOW_DECLARATIVE_PIPELINES_BATTLECARD">
DATABRICKS
LAKEFLOW DECLARATIVE PIPELINES BATTLECARD
GO/PIPELINES/BATTLE
LAST UPDATES: FY26Q2
©2024 DATABRICKS INC. — ALL RIGHTS RESERVED
</xml>

<xml slide="02" title="LAKEFLOW_DECLARATIVE_PIPELINES_L200">
LAKEFLOW DECLARATIVE PIPELINES
L200
©2024 DATABRICKS INC. — ALL RIGHTS RESERVED
</xml>

<xml slide="03" title="HOW_TO_WIN_WHAT_CUSTOMERS_CARE_ABOUT_THE_MOST">
HOW TO WIN – WHAT CUSTOMERS CARE ABOUT THE MOST
Key Declarative Pipelines aspects to highlight to decision makers and end users

COLUMNS:
- LAKEFLOW DECLARATIVE PIPELINES
- FABRIC MATERIALIZED LAKE VIEWS
- AWS EMR / GLUE ETL / REDSHIFT

ROW: AUTO-SCALING SERVERLESS (Reduces compute used and overall TCO)
- Lakeflow: Automatic cluster size selection that dynamically auto-scales during job execution
- Fabric: Need to manually define Spark cluster size. It can dynamically auto-scale during job execution.
- AWS: AWS currently does not have it’s own Declarative Pipelines Framework. Customer can use dbt with Redshift. OSS Spark Declarative Pipelines on EMR could also be supported in the future but has not been confirmed by AWS.

ROW: AUTOMATIC INCREMENTALIZATION (Simplifies dev and reduces TCO)
- Lakeflow: Streaming Tables incrementally ingest files and messages. Materialized views incrementally refresh and update tables
- Fabric: No incremental file ingestion. Only incremental materialization. Currently in Private Preview so perf unknown

ROW: BATCH OR STREAMING (Support any latency use cases without switching tools)
- Lakeflow: Schedule, trigger on new file or table updates or run in continuous mode.
- Fabric: Schedule mode only. No file/table triggers. No continuous or streaming mode.

ROW: MODERN DEV EXPERIENCE (Fast development and reliable pipelines)
- Lakeflow: Code or no‑code development in a modern IDE with AI Assistant, live lineage view, git integration & easy auto‑deploy.
- Fabric: Code only. IDE with AI assistant and Git integration. No live lineage, manual deployment scripts.

ROW: OPEN‑SOURCE FRAMEWORK (Avoid vendor lock‑in)
- Lakeflow: Open‑Source framework with Spark Declarative Pipelines that can run on any Spark platform
- Fabric: Lake view framework is proprietary to Fabric. Supports writing to OneLake managed tables.
</xml>

<xml slide="04" title="HOW_TO_WIN_KEY_PRODUCT_AREAS_AND_FEATURES_TO_HIGHLIGHT_LAKEFLOW_VS_FABRIC_RTI">
HOW TO WIN – KEY PRODUCT AREAS & FEATURES TO HIGHLIGHT
COLUMNS: LAKEFLOW DECLARATIVE PIPELINES | FABRIC RTI EVENTSTREAMS

HIGH THROUGHPUT & LOW LATENCY
- Lakeflow: Easily manages 400+MB/s or 1M rows for 200–300 ms latency.
- Fabric: Can go upto 200MB/s for stateless streaming jobs for 100 of Milliseconds latency.

INGESTION CONNECTORS THROUGHPUT
- Lakeflow: Streaming connectors can scale and manage throughput according to message brokers throughput and partitions. Lakeflow Connect is not streaming CDC yet.
- Fabric: 1 v‑core connectors for most of the streaming sources & Databases CDC except Azure Event Hub or IoT Hub. It is impossible to scale for out beyond 1‑v‑core at the moment.

UNIFIED BATCH AND STREAMING
- Lakeflow: Pipelines can be scheduled, trigger on new file or table updates or run in continuous mode.
- Fabric: EventStreams is only Streaming Pipelines and batch pipelines are separate in Fabric outside of RTI.

DEVELOPMENT EXPERIENCE
- Lakeflow: Code or no‑code development (Coming Soon) in a modern IDE with AI Assistant, live lineage view, git integration, CI/CD & easy auto‑deploy.
- Fabric: No‑Code only. No IDE with Git integration. No AI assistant is available for EventStreams, and limited SQL Operator for stream processing that you can’t combine with other Operators.

OPEN‑SOURCE & MULTI CLOUD
- Lakeflow: Based on the Open‑Source Spark Declarative Pipelines framework that can run on any Spark platform. Supports Delta Lake tables.
- Fabric: EventStreams is proprietary to Azure/Fabric.

TYPES OF STREAM PROCESSING
- Lakeflow: Manages complex stateful stream processing transformations with possibility for arbitrary actions.
- Fabric: Ideal for Stateless and Limited stateful stream processing with no possibility of complex stream and arbitrary actions.

SERVERLESS WITH AUTO‑SIZING AND AUTO‑SCALING
- Lakeflow: Serverless automatically selects the optimal cluster size for each job and dynamically auto‑scales during job execution
- Fabric: Autoscaling in eventstream and in general in Fabric is not possible except Spark. Fixed Compute Resources without options to scale down.

More compete content: go/fabric-rti/glean
</xml>

<xml slide="05" title="LAKEFLOW_DECLARATIVE_PIPELINES_L300">
LAKEFLOW DECLARATIVE PIPELINES
L300
©2024 DATABRICKS INC. — ALL RIGHTS RESERVED
</xml>

<xml slide="06" title="FABRIC_MLV_VS_LAKEFLOW_DECLARATIVE_PIPELINES">
FABRIC MLV–VS–LAKEFLOW DECLARATIVE PIPELINES
(INTERNAL USE ONLY)

| PRODUCT AREA | FABRIC MATERIALIZED LAKE VIEWS | LAKEFLOW DECLARATIVE PIPELINES |
| --- | --- | --- |
| AUTO‑SCALING SERVERLESS (Compute elasticity and cost optimisation) | (+) Serverless autoscaling via Autoscale Billing for Spark – Automatically scales Spark compute on demand without pre‑sizing clusters | (+) Automatic cluster size selection and scaling – Lakeflow pipelines scale horizontally and vertically during execution to optimise cost and performance. |
| AUTOMATIC INCREMENTALIZATION (Simplify development and cut cost.) | (‑) No incremental file ingestion – Incremental refresh is still preview and performance is unproven. | (+) Streaming tables ingest files and messages incrementally – Materialized views refresh incrementally with built‑in change tracking. |
| BATCH OR STREAMING (Meet any latency goal.) | (‑) Schedule mode only – No file or table triggers and no continuous streaming mode. | (+) Schedule, trigger, or continuous – Pipelines run on a schedule, on data arrival, or in always‑on streaming mode. |
| MODERN DEV EXPERIENCE (Fast development and reliable pipelines.) | (‑) Code‑only IDE – No live lineage view and deployments rely on manual scripts despite AI assist and Git integration. | (+) Code or no‑code with AI assistant – Visual designer, live lineage, Git integration, and auto‑deploy through Lakeflow Designer. |
| OPEN‑SOURCE FRAMEWORK (Reduce lock‑in risk.) | (‑) Proprietary to Fabric – Lake view framework writes only to OneLake managed tables. | (+) Open‑source Spark Declarative Pipelines – Donated to Apache Spark and portable across any Spark platform. |
</xml>

<xml slide="07" title="MICROSOFT_FABRIC_ETL_DATA_ENGINEERING_BATTLECARD">
DATABRICKS
MICROSOFT FABRIC ETL & DATA ENGINEERING BATTLECARD
GO/FABRIC-ETL/BATTLE
LAST UPDATED: JULY 2024
©2024 DATABRICKS INC. — ALL RIGHTS RESERVED
</xml>

<xml slide="08" title="COMPETING_WITH_MICROSOFT_FABRIC_ETL_WHAT_YOU_NEED_TO_KNOW">
COMPETING WITH MICROSOFT FABRIC ETL – WHAT YOU NEED TO KNOW
(INTERNAL USE ONLY)

WHAT IS ETL AND DATA ENGINEERING ON MICROSOFT FABRIC
Microsoft Fabric offers multiple products for ETL and data engineering use cases.
“The ETL and Data Engineering user experience on Fabric is fragmented, lacks enterprise level governance, does not scale for large workloads and has a high TCO for many of the services”
Products/use cases:
- Fabric Data Factory: Data integration and orchestration
- Fabric Dataflow Gen2: No‑code/Low‑code data ingestion and transformation
- Fabric Spark (Notebooks and Jobs): Develop and deploy serverless Spark workloads
- Fabric Real‑Time Intelligence: Ingest, transform and visualise real‑time events

FABRIC ETL STRENGTHS
- Many data connectors
- Serverless compute
- No‑code/Low‑code UIs
- Copilot assistant

FABRIC ETL WEAKNESSES
- Lacks unified enterprise data governance
- Higher TCO and poor performance
- Scalability issues with No‑code/Low‑code
- Fragmented programming languages and UIs

FABRIC FOR DATA INGESTION (“High cost and slow ingestion performance”)
- Incremental ETL not supported for all sources
  - No support for incremental ETL in Data Factory, Spark or Dataflow Gen2 resulting in complex workarounds. Mirroring CDC only supports SCD Type 1
- High cost for ingestion from Azure Data Lake Storage
  - High TCO for ingestion from cloud storage into Delta Lake tables compared to Auto Loader and Photon

FABRIC FOR ORCHESTRATION (“Not designed to orchestrate all of the Lakehouse”)
- Does not support selecting Fabric spark compute
  - Data Factory does not allow you to select Spark environments for Notebooks (uses workspace default) resulting in scaling issues
- Missing Fabric and Databricks Lakehouse integrations
  - Difficult to orchestrate everything in your lakehouse as it cannot orchestrate SQL queries, refreshing reports or Databricks jobs

FABRIC FOR PROCESSING & TRANSFORMATION (“Processing workloads does not scale”)
- Poor support for Streaming workloads
  - Fabric Spark streaming is difficult to develop, deploy and monitor. Real‑time analytics only has basic streaming features
- Spark compute for ETL is far less performant
  - Lags behind on Spark version. Even with Native Execution Engine (Velox/Gluten) performance is far worse than Photon
</xml>

<xml slide="09" title="COMPETING_WITH_FABRIC_ON_SERVERLESS_SPARK">
COMPETING WITH FABRIC ON SERVERLESS SPARK
(INTERNAL USE ONLY)
More info: Win with Databricks Serverless Compute

| PRODUCT AREA AND CUSTOMER VALUE | WHAT FABRIC OFFERS | HOW DATABRICKS COMPARES (WIP) | 
| --- | --- | --- |
| SERVERLESS WITH AUTOSCALING. Fast startup times. Customers do not need to configure VMs or manually scale compute based on data volumes | + Fabric Serverless Spark offer fast startup times only for starter pools. – Capacity pools take up to 3 mins. Customers often still need to choose cpu and memory size. – Node‑level autoscaling is not smart. Result in over‑provisioned clusters increasing TCO further. | + Serverless compute in Workflows and Notebooks offers fast startup times. + Removes the need to pick a size for the serverless compute. + Node‑level autoscaling means compute is always right‑sized for the job further reducing TCO. |
| FAST PERFORMANCE WITH LOW TCO. Drive value from data as fast as possible without high costs. | + Fabric Serverless Spark offers fast start‑up. – Perf is worse and TCO is often higher. +/‑ Fabric Native Engine (Velox) improves performance but is not production ready. | + Databricks Photon and Serverless compute offer industry leading performance with lower TCO than Snowflake. |
| VERSIONLESS. Automatically get the latest features (DBR, Photon, Apache Spark etc.) | – Fabric Serverless Spark has multiple versions requiring manual upgrades. | + Databricks Serverless always runs the latest version of Apache Spark optimized with Photon. + Spark API upgrades automatically managed. |
| CLOUD DEVELOPER EXPERIENCE. Fully managed Cloud Notebooks or IDE with serverless compute. | + Serverless notebooks can connect to Fabric Serverless Spark compute. | + Easy to use notebooks runs on serverless compute. |
| SECURE SHARED SPARK COMPUTE. The first and only serverless Spark service for multi‑user with full code isolation using Unity Catalog LakeGuard. | – Fabric Spark is not compatible with OneLake table ACLs defined in Fabric warehouses. Disjointed experience. | + Share serverless compute between users with full Table ACLs. +/‑ Not all Spark features are compatible. |
</xml>

<xml slide="10" title="COMPETING_WITH_FABRIC_FOR_DATA_INGESTION">
COMPETING WITH FABRIC FOR DATA INGESTION
(INTERNAL USE ONLY)

| PRODUCT AREA AND CUSTOMER VALUE | WHAT FABRIC OFFERS | HOW DATABRICKS COMPARES (WIP) |
| --- | --- | --- |
| SIMPLE TO SET‑UP AND MAINTAIN. Allow any team to easily set‑up ingestion and monitor everything in the one place. | + Mirroring, Data Factory and Data Flow gen2 have simple wizard based UIs for set‑up. – Multiple UIs means users need to understand limitations and decide which one to use. – No unified monitoring for ingestion across all 3. | + LakeFlow offers a simple wizard based UI. + LakeFlow offers only one ingestion UI and therefore also unified monitoring. |
| LOW COST. Make it cost effective to ingest large volumes of data. | + Mirroring is listed as free and offers 1TB storage per capacity F unit. Eg. 64TB free for F64. + Data Factory / Data Flow gen2 have a low cost. – You must have your Fabric capacity running which reduces the benefits of the low cost. | +/‑ LakeFlow connect pricing will be higher than Fabric but feature wise will offer more including incremental / CDC ingestion, easier to use interface and unified monitoring. |
| LARGE NO OF SUPPORTED DATA SOURCES. Cover all use cases for your company. | – Mirroring supports only 3 data sources. + Data Factory and Data Flow gen2 support over 100+ sources (1,2). | +/‑ LakeFlow connect will offer many of the key connectors e.g. SQL Server, Salesforce and Workday. + Multiple partners inc. Fivetran for all sources. |
| INCREMENTAL INGESTION. Reduce costs and improve performance by only ingesting and processing new and updated data. | + Mirroring supports incremental CDC ingestion. Currently only SCD Type 1. – Data Factory and Data Flow gen2 do not support incremental ingestion. | + Auto Loader and LakeFlow connect support incremental ingestion reducing costs. |
| ENTERPRISE GOVERNANCE AND SECURITY. Ensure connections to sources are secure and data can be governed once ingested. | – Mirroring has many Enterprise security limitations. – Once data is ingested OneLake governance does not allow unified permissions on data. | + LakeFlow connect will cover all enterprise security requirements. + LakeFlow connect integrates with Unity Catalog giving all the same benefits over Fabric OneLake. |
</xml>

<xml slide="11" title="COMPETING_WITH_MICROSOFT_FABRIC_TALKING_TO_CUSTOMERS_AND_PROSPECTS">
COMPETING WITH MICROSOFT FABRIC – TALKING TO CUSTOMERS AND PROSPECTS
(INTERNAL USE ONLY)

ELEVATOR PITCH VS FABRIC
The lakehouse is the future of data platforms, Fabric strengthens this message. Fabric is built on Delta, tell customers to double down on Databricks and get the benefits of Delta now – they can always look to adopt Fabric when (if) overcomes current limitations and meets their requirements.
Lakehouse platforms need strong enterprise governance – Unity Catalog is the only lakehouse governance solution that currently fulfills all requirements with a single solution.

TABLE (EXCERPT)
Microsoft Fabric | Databricks
- Maturity: Fabric is GAed in Nov. 2023. Feature parity lagging behind Databricks. | Databricks Lakehouse is production ready
- Governance: No centralized data governance. Each warehouse and lakehouse manages its own. | Centralized catalog layer that all compute offering can access. Centralized data governance
- Unification: Fabric is unified on the surface (UI) but not underlying. Fragmented object permissions. Divided compute options. | No separate Lakehouse/Data warehouse concept. Truly unified platform to serve all workloads

DATABRICKS LAKEHOUSE WINS
Coming Soon...
Share any wins with: competitive-intelligence@databricks.com

WHAT’S MICROSOFT STRATEGY WITH FABRIC FOR ETL AND DATA ENGINEERING?
Push customers to migrate from Azure Data Factory and/or Synapse Spark to Fabric. Bundle all features together with a complicated pricing concept.

COMMON SCENARIOS
Even though Fabric just went into general availability, we expect interest in Fabric to come from:
- Large Power BI user bases looking to access more data
- Customers exploring a CSP first‑party lakehouse

In all cases, enterprise data governance is prerequisite for a lakehouse. Fabric doesn’t, and won’t for the foreseeable future be able to offer enterprise data governance. Databricks does with Unity Catalog.
Power BI already connects with Databricks – if a deeper Power BI integration is being requested then reach out to competitive-intelligence@databricks.com

ADDITIONAL RESOURCES
- go/compete
- go/azurelakehouse
- Fabric ETL – Common Use Case Support Matrix
- Fabric POC Guidance
- Fabric Performance & Pricing: Field FAQ
- Fabric Security & Governance: Field FAQ

LEVERAGE THE EXPERTS
- Give customer feedback on Fabric: go/fabric/feedback
- Slack channels: #azure-compete, #competition
- Request an expert via go/ecl-charter and go/findmyce!
</xml>

<xml slide="12" title="LAKEFLOW_JOBS_ORCHESTRATION_L100">
INTERNAL
LAKEFLOW JOBS – ORCHESTRATION
L100
©2024 DATABRICKS INC. — ALL RIGHTS RESERVED
</xml>

<xml slide="13" title="ORCHESTRATION_PRODUCTS_L100_COMPARISON">
ORCHESTRATION PRODUCTS L100 COMPARISON
(INTERNAL ONLY)

| PRODUCT AREA | LAKEFLOW JOBS | ADF/FDF | AIRFLOW | SNOWFLAKE TASKS |
| --- | --- | --- | --- | --- |
| UNIFIED PLATFORM | ✓ Integrated catalog & observability | ~ | ~ | ✓ Integrated catalog & observability |
| DEVELOPER EXPERIENCE | ✓ UI‑based, Python SDK, DABs or CLI | ~ Primarily UI‑based. | ~ Only code‑based | ~ Only code‑based |
| ADVANCED CONTROL FLOW | ✓ | ✓ | ✓ | ✗ |
| SCHEDULES AND TRIGGERS | ✓ | ~ No continuous orchestration | ~ No continuous orchestration | ~ No continuous orchestration |
| INTERNAL AND EXTERNAL TASKS | ~ Limited external tasks | ~ Limited external tasks. | ✓ Many external connectors | ✗ No external connectors |
| FULLY MANAGED SERVICE | ✓ | ✓ | ~ Requires cluster management | ✓ |
| OBSERVABILITY AND ALERTS | ✓ | ~ Disjointed from Spark/DWH | ~ Disjointed from Spark/DWH and data | ✓ |
| UNIFIED GOVERNANCE | ✓ | ~ Disjointed permissions & lineage | ~ Disjointed permissions & lineage | ✓ |
</xml>

<xml slide="14" title="LAKEFLOW_JOBS_ORCHESTRATION_L300">
INTERNAL
LAKEFLOW JOBS – ORCHESTRATION
L300
©2024 DATABRICKS INC. — ALL RIGHTS RESERVED
</xml>

<xml slide="15" title="COMPETING_WITH_FABRIC_AZURE_DATA_FACTORY_ORCHESTRATION_1">
COMPETING WITH FABRIC/AZURE DATA FACTORY – ORCHESTRATION
(INTERNAL USE ONLY)

| PRODUCT AREA AND CUSTOMER VALUE | WHAT FABRIC/AZURE DATA FACTORY OFFERS | HOW DATABRICKS COMPARES |
| --- | --- | --- |
| UNIFIED PLATFORM. Integrations with authoring UIs, governance, monitoring and more | + Schedule button in Notebooks. – Observability disconnected from external Spark or Warehouse observability. – Different permission models in ADF and UC. – No lineage between jobs and Delta Lake tables. | + Schedule button in Notebooks, DBSQL and AI/BI. + Integrated observability. Jobs and DBSQL/Spark monitoring all in one UI. + Shared permissions through Unity Catalog. + Automated lineage across jobs and UC objects. |
| DEVELOPMENT EXPERIENCE. Easily author and deploy workflows using GUIs or code | + Easy to use drag‑and‑drop orchestration GUI. – No authoring SDK. | + Easy to use orchestration GUI. + Develop locally + CI/CD using DABs & Python SDK. |
| ADVANCED CONTROL FLOW. Support many types of workflows with flexibility | + Supports advanced control flow options inc. For Each, If/Else, Execute Pipeline, Set Variable. | + Supports advanced control flow options inc. For Each, If/Else, Execute Pipeline, Set Variable. |
| SCHEDULES AND TRIGGERS. Run pipelines at any frequency including real‑time or data aware | + File/event triggers (Not in FDF) & schedules. – No continuous streaming mode. + Supports Backfills. | + File trigger, table trigger, schedules. + Continuous streaming mode. * Roadmap backfills coming soon. |
| INTERNAL AND EXTERNAL TASKS. Connect to a large variety of Databricks and external systems | * Only supports limited Databricks task types. + Connect to many third‑party services using native operators and sensors. | + Connect to all Databricks products. * Limited external task types (PowerBI, dbt Cloud) – Can use python SDKs or APIs to connect to any external systems. |
</xml>

<xml slide="16" title="COMPETING_WITH_FABRIC_AZURE_DATA_FACTORY_ORCHESTRATION_2">
COMPETING WITH FABRIC/AZURE DATA FACTORY – ORCHESTRATION
(INTERNAL USE ONLY)

| PRODUCT AREA AND CUSTOMER VALUE | WHAT FABRIC/AZURE DATA FACTORY OFFERS | HOW DATABRICKS COMPARES |
| --- | --- | --- |
| FULLY MANAGED SERVICE. Low maintenance and high reliability | + ADF is a fully managed service. – Users pay for various orchestration fees. | + Lakeflow Jobs is a fully managed service. + No additional orchestration fees. |
| OBSERVABILITY AND ALERTS. Easily monitor and fix any workflow issues | – ADF observability disconnected from Spark/DWH observability. + Email and message service alerts. – No system tables to build custom reports. | + Integrated workflows and DBSQL/Spark observability. + Email and message service alerts. + System tables to build custom reports. |
| CROSS‑WORKSPACE ORCHESTRATION. Orchestrate workloads across different teams | + Connect to multiple Databricks Workspaces | – Can only orchestrate within the same workspace |
</xml>

<xml slide="17" title="DATA_WAREHOUSE_BATTLECARD">
DATABRICKS
DATA WAREHOUSE BATTLECARD
GO/DWH/BATTLE
LAST UPDATED: FY26Q3
©2024 DATABRICKS INC. — ALL RIGHTS RESERVED
</xml>

<xml slide="18" title="EXECUTIVE_SUMMARY_COMPETING_AGAINST_FABRIC_DATA_WAREHOUSE">
EXECUTIVE SUMMARY – COMPETING AGAINST FABRIC DATA WAREHOUSE

1) “Fabric (moreover OneLake) locks in customers”: Access to tables created in Fabric (both lakehouse and warehouse tables) always require a running capacity – you always require active Fabric compute even when you don’t need it.
Moreover writing to data warehouse tables from any other engine (including Fabric Spark) cannot be done without the DWH engine running, which results in double compute costs even within Fabric.

2) “Fabric data warehouse is not production ready”: Customers struggled with Synapse, often having to resort to expensive consultants to tune queries and optimise data… Fabric is a repackaging of Synapse.
After almost 2 years in GA, Fabric data warehouse is in fact worse than Synapse:
- Security and governance is still influx. OneSecurity came and went – the rebranded OneLake Security is still in preview, and still disjointed across warehouse and lakehouse..
- Automation options (APIs, GIT), required for production, are an afterthought – with basic capabilities only just in preview
No‑one, other than sponsored customers, has workloads (other than Power BI) in production. If Synapse didn’t work for you, why would you even consider migrating to Fabric?

3) “Fabric is slow and expensive”: Fabric is a new platform, yet we already have customers migrating to Databricks/DBSQL due to high costs (Rodonaves) or/and poor performance (RNP).
Supporters of Fabric and Power BI warn and criticize that Fabric services are very expensive (even for basic tasks like copying files to tables linkedin post – Marco Russo). These unexpectedly expensive, simple tasks, combined with the capacity model (shared subscription), can throttle and even break production workloads running on Fabric data warehouse.
Our benchmarks show that Databricks outperforms Fabric Data Warehouse in both performance and TCO by upto 2x.
</xml>

<xml slide="19" title="DATA_WAREHOUSE_L100">
DATA WAREHOUSE
L100
©2024 DATABRICKS INC. — ALL RIGHTS RESERVED
</xml>

<xml slide="20" title="COMPETING_AGAINST_FABRIC_DATA_WAREHOUSE">
COMPETING AGAINST FABRIC DATA WAREHOUSE
INTERNAL USE ONLY
DATABRICKS CONFIDENTIAL – NDA REQUIRED – DO NOT DISTRIBUTE

NO DATA GOVERNANCE
- OneLake data catalog is not a true catalog – it can list DWH objects and show basic lineage but lacks essential features like security enforcements or column level lineage.
- The entire security model is highly fragmented. Customers need to learn how to secure and manage access to their data across various locations: workspace, item (UI only), and SQL engine. There is no unified approach to data security like in Unity Catalog.
- Advanced governance features require Purview (for an additional fee), which offers a data discoverability option but lacks actionable features like access policy pushdown.

EXPENSIVE AND SLOW
- Fabric’s use‑it‑or‑lose‑it capacity model, combined with slow performance, makes Fabric DWH very expensive. For Data Warehouse workloads DBSQL is 2x faster and 2x cheaper.
- Customers need to overprovision the Fabric Capacity to run mixed workloads and to avoid performance bottlenecks.
- Both Fabric DWH and DBSQL support the Delta. However, in Fabric data is only accessible when Fabric compute is active (which costs money), and even other Fabric engines like Spark cannot write to the DWH tables without paying double tax.

NOT PRODUCTION READY
- Customers continue to be very vocal on linkedin and reddit about Fabric outages and losing data access e.g: Sept 3rd 2025
- Customers adopt Fabric for Power BI, not for data warehousing.
- After more than 2 years, the Fabric Data Warehouse still has limited T‑SQL parity, which forces customers to implement inefficient SQL statements. Key features like source control (GIT integration) are still in preview.
- The “UI first” approach prevents automating tasks like granting DWH, Lakehouse access, or creating OneLake security roles due to lack of APIs.
</xml>

<xml slide="21" title="DATA_WAREHOUSE_L200_PLUS">
DATA WAREHOUSE
L200+
©2024 DATABRICKS INC. — ALL RIGHTS RESERVED
</xml>

<xml slide="22" title="HOW_TO_WIN_KEY_PRODUCT_AREAS_FEATURES_DBSLQ_VS_FABRIC_DWH">
HOW TO WIN – KEY PRODUCT AREAS & FEATURES TO HIGHLIGHT
DBSQL vs FABRIC DATA WAREHOUSE

| AREA | DBSQL | FABRIC DATA WAREHOUSE |
| --- | --- | --- |
| OPEN STORAGE LAYER (Removes risks of vendor lock‑in) | Data is stored in open format: Delta or Iceberg. Managed tables can be served in either format to any Delta or Iceberg client. | Data is stored in Delta format but can only be used with OneLake as a storage layer. OneLake is tightly coupled with Fabric Capacity. |
| OPEN CATALOG (Read/Write from any engine to avoid vendor lock‑in) | Data can be written to or read from any external engine without requiring Databricks compute. Easily integrates with external tools and enables seamless open data sharing across any preferred platform. | Tightly coupled storage (OneLake) with Fabric compute. External data access requires Fabric compute to run. External engines incur ~3x the read costs compared to native Fabric engines. |
| UNIFIED WITH CATALOG (End‑to‑end governance from source to consumer) | Unified governance for DWH and BI. Supports both Delta and Iceberg tables. Catalog and query federation allows seamless integration with a wide range of data sources. | Not a real catalog item level lineage only. Limited to Fabric users. Fabric requires a Purview – which is expensive – to do proper data discovery. |
| LOWEST TCO (Benchmarked against our competitors) | Best or matching performance with the lowest TCO. Consistent performance enhancements over the years and no price increase. | Slower performance (~2x) and higher TCO (~2x). Optimized for the basic TPC‑H benchmark. |
| UNIFIED WITH ETL (Built‑in best in class ETL capabilities) | Fully integrated with modern ETL – Lakeflow with features like native incremental and CDC ingestion. Best in class Spark and SQL Scripting / Stored Procedures (Preview) available. Full support for both batch and streaming workloads, enhanced by advanced orchestration features. | Multiple ETL tools, each with limitations: Mirroring – supports only limited sources but provides CDC; Data Factory – offers restricted capabilities; Data Flow Gen2 – has low performance; EventStream – limited to streaming. Governance and monitoring of ETL tools is challenging. |
| UNIFIED WITH AI (Use any LLM via SQL, use DBSQL in external agents) | Fully unified with Mosaic AI and featuring a broad set of built‑in AI SQL functions, compatible with both off the shelf and custom models. Best in class performance and cost. | Missing AI SQL functions (PrPr for basic AI functions). Not possible to infer pre‑trained models from Fabric MLflow. |
| INTEGRATED BI (AI/BI) | Solid BI capabilities at no additional cost. Out‑of‑the‑box full governance, including lineage, access, and audit features. | Strong BI capabilities but limited support for data lineage and governance. |
</xml>

<xml slide="23" title="DATA_GOVERNANCE_BATTLECARD">
DATABRICKS
DATA GOVERNANCE BATTLECARD
GO/DATAGOV/BATTLE
LAST UPDATED: OCTOBER 2025
©2024 DATABRICKS INC. — ALL RIGHTS RESERVED
</xml>

<xml slide="24" title="COMPETING_AGAINST_MICROSOFT_FABRIC_ONELAKE_PURVIEW">
COMPETING AGAINST MICROSOFT FABRIC / ONELAKE / PURVIEW

| CATEGORY | FABRIC / ONELAKE / PURVIEW | UNITY CATALOG |
| --- | --- | --- |
| UNIFIED: PURPOSE‑BUILT LAKEHOUSE CATALOG FOR DATA + AI | No catalog – File system != a catalog. Only supports tables and files (no Views) | Supports tables, views, MVs, STs, ML models, files, etc. |
| UNIFIED: SECURITY & GOVERNANCE | OneLake Security is not supported by every engine, RLS/CLS do not work consistently | Unified catalog that secures all Data + AI assets |
| UNIFIED: SEARCH & DISCOVERY | OneLake Catalog charges capacity for exploration | Built‑in and free |
| UNIFIED: LINEAGE | Item‑level lineage only (no table or column level) | Supports all Data + AI assets |
| OPEN: ANY ENGINE, TOOL | Supports most engines, but no support for FGAC | Supports most engines |
| OPEN: COLLABORATION & SHARING | Only for other Fabric users | Supports Databricks + Non‑Databricks users |
| OPEN: TABLE FORMATS | Delta Lake + Iceberg (via XTable) | Delta Lake + Iceberg + Hudi |
| INTELLIGENT: DOMAIN INTELLIGENCE VIA GENAI, LINEAGE, CONSUMPTION | Copilot is experience‑specific due to lack of centralized catalog | Intelligence talks to UC, which has full context of Data + AI assets |
</xml>

<xml slide="25" title="AI_BI_BATTLECARD">
DATABRICKS
AI/BI BATTLECARD
GO/AIBI/BATTLE
LAST UPDATES: FY26Q3
INTERNAL
©2024 DATABRICKS INC. — ALL RIGHTS RESERVED
</xml>

<xml slide="26" title="AI_BI_L200_PLUS">
AI/BI
L200+
©2024 DATABRICKS INC. — ALL RIGHTS RESERVED
</xml>

<xml slide="27" title="AI_BI_GENIE">
AI/BI GENIE
©2024 DATABRICKS INC. — ALL RIGHTS RESERVED
</xml>

<xml slide="28" title="FABRIC_DATA_AGENT_VS_AI_BI_GENIE">
FABRIC DATA AGENT–VS–AI/BI GENIE

| PRODUCT AREA | FABRIC DATA AGENT | DATABRICKS |
| --- | --- | --- |
| TARGET USE CASES OR AUDIENCE | For Business users that want to ask questions on data in Fabric. For those working on big capacities (F64+). Requires a very difficult setup, which is not justified as quality of answers is low and hard to iteratively improve. | For Business users within Databricks’ ecosystem. Easy set up experience, and connection to data in Unity Catalog. Genie is free and users only pay for cost‑optimal compute (DBSQL) with no idle time. |
| GETTING STARTED (Enabling the feature & initial use) | (‑) Not simply serverless – Starting a capacity takes multiple switches between Azure & Fabric; confusing for first time users (link, link). (+) Easy to add data to Fabric, for small datasets (link). (‑) Difficult to enable AI Skills: SKU and tenant switches hard to get right. | (+) Very easy to start with: Genie is enabled by default in Databricks workspaces. Very easy to setup a Genie space (simple and clear UX). (+) Serverless: No need to setup underlying compute. |
| LICENSING/COST | (‑) Requires an expensive capacity. AI Skills only work on F64 capacity, which is ~5.6k/month (link). | (+) There is no extra cost for Genie. There is no per‑user licensing; costs are tied to compute usage for execution queries. |
| CONFIGURATION OPTIONS | (‑) Users must add extensive model notes and instructions – the system does not make use of a semantic model, and thus require model notes and instructions to work with good quality. | (+) Can add instructions and example queries. (link): Those are easy to set up and optional (as system can work with semantic model). (+) Ability to upvote/downvote responses and also the ability to leverage metadata and AI generated column details from Unity Catalog. |
| QUALITY OF RESPONSES | (‑) No semantic understanding – leads to lower quality; you only can adjust instructions. (‑) No eval and monitoring – no metrics of performance or concept of measuring quality; cannot improve it if you cannot measure it. | (+) Eval & monitoring – Able to compare responses against ground truth to quantify quality of responses. Ability to leverage trusted assets, example queries etc. to improve quality based on benchmarking. |
| GOVERNANCE | (‑) No monitoring of responses or ability to track quality. (‑) Not connected to the catalog – does not utilise established data management permissions and semantic model. | (+) Monitoring – Can use logs to setup benchmarks for monitoring performance. (+) Integrated with the catalog (UC). |
| OTHERS | (‑) Low reliability – Complex queries that require many joins or sophisticated logic tend to have lower reliability. (‑) Limited integration – can’t connect the AI skill to Fabric copilots, Microsoft Teams, or other experiences outside of Fabric. (‑) Doesn’t support a conversational interface; does not remember history of conversation. | (+) Good integrations – Genie APIs support leveraging in an agent system or integration into any other application. (+) Supports conversational interface (has conversation history and uses it in answers – better user experience). |
</xml>

<xml slide="29" title="AI_BI_DASHBOARDS">
AI/BI DASHBOARDS
©2024 DATABRICKS INC. — ALL RIGHTS RESERVED
</xml>

<xml slide="30" title="POWER_BI_VS_AI_BI_DASHBOARDS">
POWER BI–VS–AI/BI DASHBOARDS

POWER BI ELEVATOR PITCH
- Supports a variety of reporting and analytics – ad‑hoc, self serve to enterprise reporting for a large audience.
- Feature rich for advanced visualization and analytics capabilities
- Very comprehensive data connectivity
- Deep integration into Microsoft ecosystem (Excel, Power Point, Teams)
- Combines data ingestion, transformation(!!), modelling and visualization
- Developer skill set relatively easy to find in the market
- Caters to analysts, Excel users and to the low/no code audience

AI/BI DASHBOARDS ELEVATOR PITCH
- You already have it if using Databricks. No extra license, no data movement or integrations, no context switch
- Uses all Unity Catalog governance out of the box (access control, lineage, audit)
- Less feature rich than Power BI, but good enough for most scenarios
- Build / re‑use logic using SQL
- Deep integration with Genie and UC semantics (constraints, metadata, metrics)

| AREA | POWER BI | AI/BI DASHBOARDS |
| --- | --- | --- |
| TARGET USE CASES OR AUDIENCE | Almost every enterprise user. Great for large scale enterprise reporting with filtering, drill down/drill through and complex calculations. Also for business user data ingestion/transformation. | For usage in the Databricks ecosystem. Easy dashboarding with filtering and scheduled refreshes. It provides integration, so it’s a good choice for reporting that combines multiple tooling (e.g. Notebooks). Natural language driven dashboarding. |
| LICENSING/COST | (‑) Pro or PPU (per user) licenses for authoring. Consumption via either Pro or PPU licenses, or Premium, a capacity model, which supports unlimited consumers. Premium (now bundled into Fabric F64 upwards) is the most common in enterprises. (‑) Complex bundling – capabilities like Copilot require capacity minimums. | (+) No additional cost for dashboards. Cost incurred is for DBSQL usage for query execution. No per‑seat cost model encourages data democratisation. |
| SUPPORT FOR DATA SOURCES | (+) More than 160+ external data connectors available, with easy UI to ingest data. | Delta lake + sources supported by lakehouse federation. (‑) No connectors for other external sources. |
| DATABRICKS DELTA LAKE INTEGRATION | Can connect to the delta lake via Databricks connector (via DBSQL preferably). Fabric has Delta Lake too. | (+) Deep integration with the Databricks Delta Lake via DBSQL, works on a single copy of data; native integration with UC. |
| SECURITY AND GOVERNANCE | Supports native RLS & OLS (object level security for tables, views etc) DLP support for PII data; EntraID for access control. (+) Can use UC security in DirectQuery. | (+) Automatically respects UC policies and guardrails. Also uses UC context for semantics and business understanding. (+) Out of the box integration with UC features (lineage, auditing, semantics). |
| VISUALISATION CAPABILITY | (+) Over 20 out of the box visualisations support. Custom visualisations available through Microsoft App source. Advanced cross filtering functionality. | (~) Relatively smaller support for visualisation types but it can do commonly used visual types; no custom visuals. |
| SEMANTIC MODELLING | (+) Mature semantic model capabilities with advanced capabilities for creating derived metrics and dimensions and hierarchies. (‑) A proprietary model based on legacy OLAP technology. | Early support with calculated columns & UC metrics. Not enterprise ready, but great for customers to begin adopting in smaller scopes. Strategic capability for UC & Genie semantics – missing BI integration at the moment. |
| EXCEL COMPATIBILITY | (+) Mature Excel connectivity using 3 different ways to analysis datasets. | Can download visualisation or underlying query data in CSV or excel. Larger download limits compared to PBI (?). |
| AI ASSISTANT/EQUIVALENT | (+) Co‑Pilot to create reports and ask natural language questions. | (+) Great(?) support for Genie AI assistant enabled dashboard building with natural language instructions. Leverages UC semantics and business context. |
| DEVELOPER SKILL SET | No‑Code, DAX, SQL and M language (less often) | SQL |
</xml>

<xml slide="31" title="POWER_BI_VS_AI_BI_DASHBOARDS_BI_CAPABILITIES">
POWER BI–VS–AI/BI DASHBOARDS (BI CAPABILITIES)

| CAPABILITY | POWER BI | AI/BI DASHBOARDS |
| --- | --- | --- |
| FILTERING, DRILLDOWN/DRILLTHROUGH, INTERACTIVITY | Advanced support for associative filtering, cross visual filtering, drill down and drill through, bookmarks, action buttons | Filtering, visual interactions like zoom in/zoom out etc. |
| CALCULATIONS | Support for complex DAX based calculations – including the ability to set calculator context, use hierarchies, dynamic rollups etc | Support calculations in SQL and metrics. UC metrics in preview |
| ADVANCED VISUALISATIONS | Support for a wider variety of advanced visuals. Also supports authoring custom visuals | Little support |
| ALERTING & SCHEDULING | Possible | Possible |
| NATURAL LANGUAGE | With co‑pilot | Inbuilt support for natural language queries augmented with business semantics |
| UI CUSTOMISATION, BRANDING | A variety of options | Little support |
| WRITE BACK | With PowerApps | Possible when integrated with lakehouse apps |
| HIERARCHIES, AGGREGATIONS | Yes | Limited |
| EXPORT DATA | Yes (Up to x records) | For the visual or the query powering a visual |
</xml>
