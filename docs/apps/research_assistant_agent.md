# Application Documentation: ResearchAssistantAgent - Web Research Helper

- **Version:** 1.0
- **Parent Project:** [IntelliForge: Applied GenAI Playbook](../overview.md)
- **Application Folder:** [`apps/research_assistant_agent/`](../../apps/research_assistant_agent/)
- **App README:** [apps/research_assistant_agent/README.md](../../apps/research_assistant_agent/README.md)

---

## 1. Introduction

ResearchAssistantAgent is an application within the IntelliForge suite designed to act as an AI-powered helper for **gathering, organizing, and querying research information** from the web. Users can specify a research topic or provide seed URLs, and the agent uses **`crawl4ai`** to perform a focused crawl of relevant web sources (academic sites, reputable news, blogs, etc.). The extracted information forms a topic-specific knowledge base, which can then be queried using natural language via Google's **Gemini 2.5 Pro**, with potential support for source tracking and citation management.

The goal is to demonstrate how automated web crawling and RAG can significantly accelerate the initial literature review and information gathering phases of research, providing a centralized place to query and synthesize findings from multiple web sources.

## 2. Core AI Concepts Demonstrated

- **Topic-Focused Web Crawling:** Using `crawl4ai`, potentially configured with keywords or seed URLs, to gather web content relevant to a specific research area.
- **Information Extraction & Organization:** Processing the RAG-optimized Markdown from `crawl4ai` and indexing it in a way that preserves source information (URLs, potentially publication dates if extractable).
- **RAG for Research QA:** Applying RAG techniques to answer specific research questions based *only* on the content gathered during the focused web crawl.
- **LLM for Synthesis & Summarization:** Using Gemini 2.5 Pro to generate summaries, synthesize findings from multiple sources, and answer complex queries based on the crawled research data.
- **Citation Management (Potential):** Tracking source URLs and possibly formatting them into basic citations.

## 3. Architecture & Workflow

Similar to WebQuestRAG and CompetitiveAnalysisAgent, this involves a crawl/index phase followed by a query phase, tailored for research topics.

1. **Topic/Source Specification:** User provides a research topic/keywords or seed URLs via Streamlit UI (`src/app.py`). May include source type preferences (e.g., prioritize `.edu`, `.gov`).
2. **Focused Crawling:** Backend uses `crawl4ai` (via `core/utils/web_crawler.py`) to crawl relevant pages, potentially guided by keywords or source types.
3. **Content Processing & Indexing:** `crawl4ai` returns RAG-optimized Markdown. Content is chunked, embedded, and indexed into a vector store (`core/utils/retrieval_utils.py`) with source metadata.
4. **Research Query:** User asks specific research questions about the topic.
5. **RAG Retrieval:** Backend retrieves relevant research content chunks from the vector store.
6. **LLM Synthesis & Answering:** Query and retrieved context sent to Gemini 2.5 Pro (`core/llm/gemini_utils.py`) with prompts suitable for research summarization/Q&A.
7. **Display Results:** Answers, summaries, and potentially source URLs/citations displayed in UI.

### Architecture Diagram

#### Research Assistant Agent Workflow Architecture

```mermaid
%%{init: {'theme': 'base'}}%%
graph TD
    subgraph "Research Assistant Agent Workflow"
        A[User Input <br> (Research Topic/Keywords, Seed URLs)] --> B(Streamlit UI);
        B -- Start Research Request --> C{ResearchAssistantAgent Backend};

        subgraph Information_Gathering ["Information Gathering Phase"]
            direction LR
            C -- Initiate Focused Crawl --> D["Focused Web Crawler <br> (core/utils/web_crawler.py w/ crawl4ai)"];
            D -- Crawl Relevant Sources --> E[Web Sources (Academic, News, etc.)];
            D -- Crawled Content (RAG Markdown) --> F{Content Processor & Chunker};
            F -- Chunks + Source Metadata --> G{Embedding & Indexing <br> (core/utils/retrieval_utils)};
            G -- Build/Update --> H[(Vector Store <br> Research Topic Knowledge Base)];
        end

        subgraph Query_Synthesis ["Query & Synthesis Phase"]
            direction LR
            B -- Research Question --> C;
            C -- Query --> I{Research RAG Retrieval};
            I -- Search --> H;
            H -- Retrieved Research Chunks + Sources --> I;
            I -- Retrieved Context --> J{Synthesis & Answering Module};
            J -- Query + Context --> K["LLM (Gemini 2.5 Pro) <br> Prompted for Research Q&A/Summary"];
            K -- Generated Answer / Summary --> J;
            J -- Final Answer + Citations --> C;
            C -- Display Results --> B;
        end
    end

```

## 4. Key Features

- **Topic-Based Crawling:** Gathers web information related to a user's research area.
- **Focused Information Gathering:** Uses `crawl4ai` to target relevant sources.
- **Research Question Answering:** Answers specific questions based on the crawled data.
- **Summarization:** Generates summaries of findings from multiple sources.
- **Citation Tracking (Basic):** Retains source URLs.

## 5. Technology Stack

- **Core LLM:** Google Gemini 2.5 Pro
- **Language:** Python 3.8+
- **Web Framework:** Streamlit
- **Web Crawling:** **`crawl4ai`**
- **Retrieval:** Vector DB (e.g., ChromaDB), Embedding Models via `core/utils/retrieval_utils.py`.
- **Core Utilities:** `google-generativeai`, `python-dotenv`, `pandas`.

## 6. Setup and Usage

*(Assumes the main project setup, including cloning and `.env` file creation, is complete as described in the main project [README](../../README.md) or [Overview](../overview.md).)*

1. **Navigate to App Directory:**

    ```bash
    cd path/to/IntelliForge-Applied-GenAI-Playbook/apps/research_assistant_agent
    ```

2. **Create & Activate Virtual Environment (Recommended).**

3. **Install Requirements:**
    - Create/update `apps/research_assistant_agent/requirements.txt` including `streamlit`, `google-generativeai`, `python-dotenv`, `crawl4ai`, `chromadb-client`, etc.
    - Install: `pip install -r requirements.txt`

4. **Run the Application:**

    ```bash
    streamlit run src/app.py
    ```

    *(Assuming the main application file is `src/app.py`)*

5. **Interact:**
    - Open the local URL.
    - Enter your research topic or keywords/seed URLs. Configure crawl options if available.
    - Start the research gathering (crawl + index). Wait for completion.
    - Ask specific questions about your topic (e.g., "Summarize recent findings on X", "What are the main arguments presented about Y?").
    - Review the answers and source information.

## 7. Potential Future Enhancements

- More robust citation formatting (BibTeX, APA, etc.).
- Integration with academic search APIs (PubMed, arXiv) alongside web crawling.
- Source credibility scoring/filtering.
- Identification of key concepts, authors, or papers within the crawled data.
- Generating literature review outlines or drafts.
- Integration with reference management tools (Zotero, Mendeley).
- Visualizing connections between research papers/concepts (Knowledge Graph extension).
