# Application Documentation: MultiPerspectiveSynth - Synthesizing Diverse Sources

**Version:** 1.0
**Parent Project:** [IntelliForge: Applied GenAI Playbook](../overview.md)
**Application Folder:** [`apps/multi_perspective_synth/`](../../apps/multi_perspective_synth/)
**App README:** [apps/multi_perspective_synth/README.md](../../apps/multi_perspective_synth/README.md)

---

## 1. Introduction

MultiPerspectiveSynth is a demonstration application within the IntelliForge suite focusing on an advanced **Retrieval-Augmented Generation (RAG)** challenge: synthesizing information from multiple documents that may present **diverse or even conflicting viewpoints** on a single topic. Standard RAG often implicitly assumes a homogeneous knowledge base, but real-world scenarios frequently involve analyzing information from various sources (e.g., different news articles, competing product reviews, varied scientific opinions).

This application allows users to upload multiple documents, potentially representing different perspectives. When queried, it retrieves relevant information from across these sources and uses Google's **Gemini 2.5 Pro**, with specialized prompting, to generate a nuanced answer that explicitly **acknowledges and synthesizes** these differing viewpoints, highlighting areas of consensus and contention.

The goal is to showcase how RAG systems can be designed to handle information diversity and provide more comprehensive, balanced summaries rather than simply echoing the viewpoint of the most retrieved document.

## 2. Core AI Concept: Multi-Source Synthesis & Perspective Handling

MultiPerspectiveSynth demonstrates:

* **Multi-Source RAG:** Performing retrieval across a corpus explicitly known to contain documents from potentially different sources or viewpoints. This requires mechanisms to track document provenance.
* **Perspective Identification (Implicit/Explicit):** While retrieving, or in a post-retrieval analysis step, identifying that the context chunks come from sources likely to have different perspectives. This might involve using document metadata or even using the LLM to analyze the stance of retrieved chunks.
* **LLM as Synthesizer:** The core capability lies in prompting Gemini 2.5 Pro not just to answer the query based on the context, but to specifically *analyze* the combined context for variations in perspective, agreement, and disagreement, and to reflect this analysis in the final generated output.
* **Balanced Representation:** Aiming to generate answers that fairly represent the key viewpoints found in the retrieved context, rather than overfitting to one particular source.
* **Source Attribution:** Clearly indicating which source supports which part of the synthesized answer can be a crucial part of this pattern (thoughอาจ implementation might vary).

## 3. Architecture & Workflow

This application modifies the standard RAG flow to handle multiple sources and explicitly perform synthesis.

### 3.1. Pre-computation/Indexing

Indexing needs to potentially store source information alongside the chunks:

1. **Document Loading (Multi-Source):** Loading documents, possibly associating each with a source identifier (e.g., filename, user-provided label) (`core/utils/data_helpers.py`).
2. **Text Chunking:** Standard chunking (`core/utils/data_helpers.py`).
3. **Metadata Association:** Ensuring chunk metadata includes the source identifier.
4. **Embedding Generation & Vector Store Indexing:** Storing chunks and their metadata (including source ID) in the vector store (`core/utils/retrieval_utils.py`).
5. **Keyword Indexing (Optional):** Building keyword index, potentially incorporating source metadata if searchable.

### 3.2. Online Query Flow

1. **User Query:** User submits a query via the Streamlit UI (`src/app.py`).
2. **Multi-Source Retrieval:** The backend performs retrieval (e.g., vector search) across the entire indexed corpus. The retrieved chunks will likely come from various original documents (`core/utils/retrieval_utils.py`). Crucially, the source identifier for each chunk is retrieved along with the text.
3. **Context Formulation:** The retrieved text chunks *and their associated source identifiers* are collected and formatted for the LLM.
4. **Synthesis Prompting:** A specialized prompt is constructed. It includes the user query and the multi-source context, but explicitly instructs Gemini 2.5 Pro to:
    * Answer the query based *only* on the provided context.
    * Identify the main perspectives or viewpoints presented across the different sources in the context.
    * Highlight areas of agreement and disagreement.
    * Synthesize these viewpoints into a coherent answer.
5. **LLM Synthesis Call:** The prompt is sent to Gemini 2.5 Pro (via `core/llm/gemini_utils.py`).
6. **Result Generation:** The LLM generates the synthesized answer, potentially including explicit mentions of different perspectives or contradictions found.
7. **Display Results:** The nuanced, synthesized answer is displayed in the Streamlit UI. Optionally, a separate section might summarize the detected perspectives or conflicting points, potentially with source attribution.

### 3.3. Architecture Diagram (Mermaid)

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'titleColor': '#333', 'titleFontSize': '20px'}}}%%
graph TD
    %% Title at the top
    classDef titleClass fill:none,stroke:none,color:#333,font-size:18px,font-weight:bold;
    title["MultiPerspectiveSynth: Synthesizing Diverse Sources Architecture"]:::titleClass;

    subgraph Indexing Phase (Handles Multiple Sources)
        direction LR
        PrepIn1[Doc 1 (Source A)] --> PrepProc{Doc Processor / Chunker};
        PrepIn2[Doc 2 (Source B)] --> PrepProc;
        PrepInN[Doc N (Source X)] --> PrepProc;
        PrepProc -- Chunks + Source Metadata --> PrepEmbed[Embedding Model];
        PrepEmbed --> PrepStore[(Vector Store <br> w/ Source Metadata)];
    end

   subgraph Online Query Phase
       direction LR
        A[User] --> B(Streamlit UI);
        B -- Query --> C{MultiPerspectiveSynth Backend};
        C -- Query --> D[Multi-Source Retrieval];
        D -- Retrieve From --> PrepStore;
        D -- Retrieved Chunks + Source IDs --> C;
        C -- Formatted Context + Synthesis Prompt --> E["core/llm/gemini_utils.py <br> (Gemini 2.5 Pro for Synthesis)"];
        E -- Synthesized Answer <br> (Highlighting Perspectives) --> C;
        C -- Display Answer + Perspective Summary --> B;
   end

    %% Position title implicitly
```

## 4. Key Features

* **Handles Multiple Documents:** Accepts and processes input from several source documents simultaneously.
* **Perspective Synthesis:** Explicitly designed to identify and synthesize differing viewpoints found in the source material.
* **Highlights Agreement/Disagreement:** The generated output aims to point out consensus and conflict across sources.
* **Balanced Answers:** Strives to provide a neutral, comprehensive summary of the diverse information retrieved.
* **Source Awareness:** The process is aware of the origin of different pieces of information during synthesis (metadata tracking is key).

## 5. Technology Stack

* **Core LLM:** Google Gemini 2.5 Pro
* **Language:** Python 3.8+
* **Web Framework:** Streamlit
* **Retrieval:** Vector DB (e.g., ChromaDB supporting metadata), Embedding Models via `core/utils/retrieval_utils.py`.
* **Core Utilities:** `google-generativeai`, `python-dotenv`, `pandas`.

## 6. Setup and Usage

*(Assumes the main project setup is complete.)*

1. **Navigate to App Directory:**

    ```bash
    cd path/to/IntelliForge-Applied-GenAI-Playbook/apps/multi_perspective_synth
    ```

2. **Create & Activate Virtual Environment (Recommended).**

3. **Install Requirements:**
    * Create/update `apps/multi_perspective_synth/requirements.txt` (e.g., `streamlit`, `google-generativeai`, `python-dotenv`, `chromadb-client`, etc.).
    * Install: `pip install -r requirements.txt`

4. **Prepare Data & Indexes:**
    * Place the multiple source documents you want to compare/synthesize in `apps/multi_perspective_synth/data/`. Ensure filenames or some structure helps distinguish sources if needed by the loading script.
    * Run the necessary indexing process (using shared utils), ensuring source metadata is associated with chunks in the vector store.

5. **Run the Application:**

    ```bash
    streamlit run src/app.py
    ```

6. **Interact:**
    * Open the local URL provided by Streamlit.
    * Ensure the multi-source documents have been indexed.
    * Enter a query that would likely elicit different responses based on the different source documents (e.g., "What are the main arguments regarding [topic X]?").
    * Observe the generated answer. Note how it attempts to present different viewpoints or explicitly mentions agreement/disagreement found in the context. Compare this to a standard RAG answer which might only reflect one dominant source.

## 7. Potential Future Enhancements

* Implement more explicit perspective/stance detection as a separate step before synthesis.
* Allow users to tag documents with source labels or viewpoints during upload.
* Add UI elements to visualize the different perspectives identified.
* Implement more granular source attribution within the generated text.
* Allow users to filter retrieval based on specific sources.
* Compare different synthesis prompting strategies for effectiveness.
