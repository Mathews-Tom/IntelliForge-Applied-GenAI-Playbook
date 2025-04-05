# Application Documentation: CompetitiveAnalysisAgent - Web Competitor Intelligence

- **Version:** 1.0
- **Parent Project:** [IntelliForge: Applied GenAI Playbook](../overview.md)
- **Application Folder:** [`apps/competitive_analysis_agent/`](../../apps/competitive_analysis_agent/)
- **App README:** [apps/competitive_analysis_agent/README.md](../../apps/competitive_analysis_agent/README.md)

---

## 1. Introduction

CompetitiveAnalysisAgent is an application within the IntelliForge suite designed to automate aspects of **competitor intelligence gathering and analysis** using web data. It leverages the **`crawl4ai`** library to perform targeted crawls of specified competitor websites, extracting relevant information (e.g., product features, pricing, news mentions, company info). This extracted data forms a knowledge base that users can then query using natural language, enabling comparisons and insights powered by Google's **Gemini 2.5 Pro**.

The goal is to demonstrate how combining automated web crawling, RAG, and LLM synthesis can create a powerful tool for market research and competitive analysis, reducing manual effort and providing structured insights.

## 2. Core AI Concepts Demonstrated

- **Targeted Web Crawling:** Using `crawl4ai` with specific configurations (URLs, potentially path inclusions/exclusions) to focus data gathering on relevant sections of competitor websites.
- **Information Extraction (Implicit/Explicit):** While `crawl4ai` provides RAG-optimized Markdown, further processing (potentially using Gemini 2.5 Pro) might be needed to extract specific structured data like prices or feature lists from the crawled content.
- **Multi-Source Knowledge Base:** Building a vector index containing information explicitly tagged by competitor/source.
- **Comparative RAG:** Performing retrieval that specifically pulls information related to multiple competitors to answer comparative queries.
- **LLM for Analysis & Synthesis:** Using Gemini 2.5 Pro not just to answer factual questions but to perform comparative analysis, summarize findings, and potentially identify strategic positioning based on the crawled data.
- **Agentic Workflow:** The system acts as an agent that takes high-level goals (analyze competitors X and Y) and breaks them down into steps (crawl sites, index data, synthesize comparison).

## 3. Architecture & Workflow

The agent operates by first building a competitor knowledge base, then facilitating queries against it.

1. **Competitor Specification:** User provides competitor identifiers (e.g., company names) and their primary website URLs via the Streamlit UI (`src/app.py`). They might also specify areas of interest.
2. **Targeted Crawling:** The backend configures and initiates `crawl4ai` (via `core/utils/web_crawler.py` or app logic) for each competitor URL, focusing the crawl.
3. **Content Extraction & Indexing:** `crawl4ai` returns RAG-optimized Markdown. Optional LLM step can extract structured data. Content/chunks are embedded and indexed into a vector store (`core/utils/retrieval_utils.py`), tagged with competitor/source.
4. **User Query:** User asks comparative or specific questions about competitors.
5. **Comparative Retrieval:** Backend queries the vector store, filtering by competitor tags based on the query, to retrieve relevant information chunks (`core/utils/retrieval_utils.py`).
6. **LLM Analysis & Synthesis:** Query and multi-source context sent to Gemini 2.5 Pro (`core/llm/gemini_utils.py`) with prompts for comparative analysis.
7. **Display Insights:** Generated analysis, comparisons, or summaries displayed in the Streamlit UI.

### Architecture Diagram

#### Competitive Analysis Agent Workflow Architecture

```mermaid
%%{init: {'theme': 'base'}}%%
graph TD
    subgraph "Competitive Analysis Agent Workflow"
        A[User Input <br> (Competitor URLs/Names, Focus Areas)] --> B(Streamlit UI);
        B -- Analyze Request --> C{CompAnalysisAgent Backend};

        subgraph Data_Gathering_Phase ["Data Gathering Phase"]
            direction LR
            C -- Initiate Crawls --> D["Targeted Web Crawler <br> (core/utils/web_crawler.py w/ crawl4ai)"];
            D -- Crawl Competitor A URL --> DA(Website A);
            D -- Crawl Competitor B URL --> DB(Website B);
            D -- Crawled Content + Source Tags --> E{Content Processor / Structurer (Optional LLM)};
            E -- Chunks + Comp. Tags --> F{Embedding & Indexing <br> (core/utils/retrieval_utils)};
            F -- Build/Update --> G[(Vector Store <br> Competitor Knowledge Base)];
        end

        subgraph Query_Phase ["Query Phase"]
            direction LR
            B -- Comparative Query --> C;
            C -- Query --> H{Comparative RAG Retrieval};
            H -- Search (Filter by Competitor?) --> G;
            G -- Retrieved Comp. A & Comp. B Chunks --> H;
            H -- Retrieved Context --> I{Analysis & Synthesis Module};
            I -- Query + Multi-Competitor Context --> J["LLM (Gemini 2.5 Pro) <br> Prompted for Comparison"];
            J -- Generated Analysis / Comparison --> I;
            I -- Final Insights --> C;
            C -- Display Insights --> B;
        end
    end
```

## 4. Key Features

- **Targeted Competitor Crawling:** Uses `crawl4ai` to focus on relevant competitor web pages.
- **Multi-Competitor Analysis:** Builds a unified KB allowing direct comparison across competitors.
- **Feature & Pricing Extraction (Potential):** Aims to pull out key structured data points.
- **Comparative Question Answering:** Leverages LLM to generate analytical comparisons.
- **Automated Intelligence Gathering:** Reduces manual effort in basic competitor web research.

## 5. Technology Stack

- **Core LLM:** Google Gemini 2.5 Pro
- **Language:** Python 3.8+
- **Web Framework:** Streamlit
- **Web Crawling:** **`crawl4ai`**
- **Retrieval:** Vector DB (e.g., ChromaDB supporting metadata), Embedding Models via `core/utils/retrieval_utils.py`.
- **Core Utilities:** `google-generativeai`, `python-dotenv`, `pandas`.

## 6. Setup and Usage

*(Assumes the main project setup, including cloning and `.env` file creation, is complete as described in the main project [README](../../README.md) or [Overview](../overview.md).)*

1. **Navigate to App Directory:**

    ```bash
    cd path/to/IntelliForge-Applied-GenAI-Playbook/apps/competitive_analysis_agent
    ```

2. **Create & Activate Virtual Environment (Recommended).**

3. **Install Requirements:**
    - Create/update `apps/competitive_analysis_agent/requirements.txt`, including `streamlit`, `google-generativeai`, `python-dotenv`, `crawl4ai`, `chromadb-client`, etc.
    - Install: `pip install -r requirements.txt`

4. **Run the Application:**

    ```bash
    streamlit run src/app.py
    ```

    *(Assuming the main application file is `src/app.py`)*

5. **Interact:**
    - Open the local URL.
    - Enter competitor website URLs. Optionally specify focus areas/depth.
    - Initiate crawling and indexing. Wait for completion.
    - Ask comparative questions (e.g., "Compare features of Competitor A vs Competitor B", "Summarize Competitor A's pricing").
    - Review the synthesized insights.

## 7. Potential Future Enhancements

- More robust extraction of structured data (prices, feature tables).
- Integration with news APIs or social media monitoring for broader intelligence.
- Sentiment analysis on crawled content.
- Historical tracking: Running crawls periodically to track competitor website changes.
- Generating SWOT analyses based on crawled data.
- Visualization of competitor positioning or feature comparisons.
