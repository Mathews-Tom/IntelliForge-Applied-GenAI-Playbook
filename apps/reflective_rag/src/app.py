"""
ReflectiveRAG: Self-Correcting Retrieval
A Streamlit application for enhanced RAG with self-correction using Gemini 2.5 Pro.
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# Add the project root to the Python path to import core modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from core.llm.gemini_utils import GeminiModelType, generate_content
from core.utils.data_helpers import load_documents_from_csv
from core.utils.evaluation import (
    evaluate_answer_faithfulness,
    evaluate_retrieval_relevance,
)
from core.utils.file_io import save_uploaded_file
from core.utils.retrieval_utils import embedding_retrieval, rerank_documents
from core.utils.ui_helpers import (
    create_footer,
    create_header,
    create_query_input,
    create_sidebar,
    display_documents,
    display_processing_steps,
    file_uploader,
    timed_operation,
)


# Function to load documents from a CSV file
@st.cache_data
def load_documents(file_path):
    """Load documents from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        if "text" not in df.columns:
            st.error("CSV file must contain a 'text' column.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading documents: {e}")
        return None


# Function to evaluate retrieval quality
def evaluate_retrieval(query, retrieved_docs):
    """
    Evaluate the quality of retrieved documents and identify issues.

    Args:
        query: The user query
        retrieved_docs: List of (document, score) tuples

    Returns:
        Dictionary with evaluation results and issues
    """
    # Extract text from documents if they are dictionaries
    doc_texts = []
    for doc, _ in retrieved_docs:
        if isinstance(doc, dict):
            doc_texts.append(doc.get("text", ""))
        else:
            doc_texts.append(doc)

    # Create prompt for Gemini
    docs_text = "\n\n".join(
        [f"Document {i + 1}:\n{doc}" for i, doc in enumerate(doc_texts)]
    )

    prompt = f"""
    Evaluate the relevance of the following documents to the query.

    Query: {query}

    {docs_text}

    For each document:
    1. Provide a relevance score from 0 to 10
    2. Identify any issues (irrelevance, contradictions, etc.)
    
    Then, provide an overall assessment:
    1. Is the retrieved information sufficient to answer the query? (Yes/No)
    2. What specific information is missing, if any?
    3. How could the query be reformulated to get better results?
    
    Return your evaluation as a JSON object with the following structure:
    {{
        "document_scores": [
            {{"doc_id": 1, "score": 8, "issues": ["issue1", "issue2"]}},
            ...
        ],
        "overall": {{
            "sufficient": true/false,
            "missing_information": "description of missing info",
            "query_reformulation": "suggested reformulation"
        }}
    }}
    """

    try:
        response = generate_content(GeminiModelType.GEMINI_2_5_PRO, prompt)

        if response:
            # Try to parse the response as JSON
            try:
                import json

                result = json.loads(response.strip())
                return result
            except:
                # If parsing fails, return a simple result
                return {
                    "document_scores": [
                        {
                            "doc_id": i + 1,
                            "score": 5,
                            "issues": ["Failed to parse evaluation"],
                        }
                        for i in range(len(retrieved_docs))
                    ],
                    "overall": {
                        "sufficient": False,
                        "missing_information": "Failed to parse evaluation result",
                        "query_reformulation": query,
                    },
                }
        else:
            return {
                "document_scores": [
                    {"doc_id": i + 1, "score": 5, "issues": ["Failed to evaluate"]}
                    for i in range(len(retrieved_docs))
                ],
                "overall": {
                    "sufficient": False,
                    "missing_information": "Failed to evaluate retrieval",
                    "query_reformulation": query,
                },
            }
    except Exception as e:
        return {
            "document_scores": [
                {"doc_id": i + 1, "score": 5, "issues": [str(e)]}
                for i in range(len(retrieved_docs))
            ],
            "overall": {
                "sufficient": False,
                "missing_information": f"Error in evaluation: {e}",
                "query_reformulation": query,
            },
        }


# Function to reformulate query
def reformulate_query(query, evaluation_result):
    """
    Reformulate the query based on evaluation results.

    Args:
        query: Original query
        evaluation_result: Evaluation results from evaluate_retrieval

    Returns:
        Reformulated query
    """
    if not evaluation_result or "overall" not in evaluation_result:
        return query

    # If the evaluation suggests the information is sufficient, no need to reformulate
    if evaluation_result["overall"].get("sufficient", True):
        return query

    # If there's a suggested reformulation, use it
    if "query_reformulation" in evaluation_result["overall"]:
        reformulation = evaluation_result["overall"]["query_reformulation"]
        if reformulation and reformulation != query:
            return reformulation

    # Otherwise, create a prompt to reformulate the query
    missing_info = evaluation_result["overall"].get("missing_information", "")

    prompt = f"""
    I need to reformulate this query to get better search results.
    
    Original query: {query}
    
    Issues with current results:
    - {missing_info}
    
    Please reformulate the query to address these issues and get more relevant information.
    Return ONLY the reformulated query, nothing else.
    """

    try:
        response = generate_content(GeminiModelType.GEMINI_2_5_PRO, prompt)

        if response:
            # Clean up the response
            reformulated = response.strip().strip("\"'")
            if reformulated:
                return reformulated

    except Exception as e:
        st.error(f"Error reformulating query: {e}")

    # If all else fails, return the original query
    return query


# Function to generate an answer
def generate_answer(query, context):
    """
    Generate an answer based on the query and context.

    Args:
        query: User query
        context: Retrieved context

    Returns:
        Generated answer
    """
    prompt = f"""
    Answer the following question based on the provided context.
    If the context doesn't contain enough information to answer the question,
    say so clearly and provide what information you can.

    Question: {query}

    Context:
    {context}
    """

    try:
        response = generate_content(GeminiModelType.GEMINI_2_5_PRO, prompt)
        return response if response else "Failed to generate an answer."
    except Exception as e:
        return f"Error generating answer: {e}"


# Function to refine an answer
def refine_answer(query, context, draft_answer, faithfulness_result):
    """
    Refine an answer based on faithfulness evaluation.

    Args:
        query: User query
        context: Retrieved context
        draft_answer: Initial answer
        faithfulness_result: Faithfulness evaluation results

    Returns:
        Refined answer
    """
    if not faithfulness_result or faithfulness_result.get("score", 0) >= 7:
        # If the answer is already faithful, return it as is
        return draft_answer

    issues = faithfulness_result.get("issues", [])
    issues_text = "\n".join([f"- {issue}" for issue in issues])

    prompt = f"""
    Refine the following answer to address faithfulness issues.
    
    Question: {query}
    
    Context:
    {context}
    
    Draft answer:
    {draft_answer}
    
    Faithfulness issues:
    {issues_text}
    
    Please provide a refined answer that:
    1. Only contains information from the context or common knowledge
    2. Does not contradict the context
    3. Avoids speculation or unsupported details
    4. Clearly indicates when information is not available in the context
    
    Refined answer:
    """

    try:
        response = generate_content(GeminiModelType.GEMINI_2_5_PRO, prompt)
        return response if response else draft_answer
    except Exception as e:
        st.error(f"Error refining answer: {e}")
        return draft_answer


# Main Streamlit app
def main():
    # Create header
    create_header(
        "ReflectiveRAG: Self-Correcting Retrieval",
        "An enhanced RAG system with self-correction and reflection capabilities, powered by Gemini 2.5 Pro.",
        icon="🔄",
    )

    # Create sidebar
    create_sidebar(
        "ReflectiveRAG Settings",
        "Configure how the self-correcting retrieval system works.",
    )

    # Sidebar settings
    with st.sidebar:
        st.header("Retrieval Settings")

        top_k = st.slider(
            "Number of documents to retrieve:", min_value=1, max_value=10, value=5
        )

        reflection_threshold = st.slider(
            "Reflection threshold (0-10):",
            min_value=0,
            max_value=10,
            value=6,
            help="Minimum average relevance score needed to skip re-retrieval",
        )

        faithfulness_threshold = st.slider(
            "Faithfulness threshold (0-10):",
            min_value=0,
            max_value=10,
            value=7,
            help="Minimum faithfulness score needed to skip answer refinement",
        )

        show_process = st.checkbox("Show detailed process", value=True)

    # Main content
    if "GOOGLE_API_KEY" in os.environ:
        # File upload
        st.subheader("1. Upload Documents")

        uploaded_file = file_uploader(
            "Upload a CSV file with documents (must contain a 'text' column)",
            types=["csv"],
            key="document_upload",
        )

        use_sample_data = st.checkbox("Use sample data instead", value=True)

        # Load documents
        documents_df = None

        if uploaded_file:
            # Save the uploaded file
            file_path = save_uploaded_file(uploaded_file)
            documents_df = load_documents(file_path)
            use_sample_data = False
        elif use_sample_data:
            sample_path = Path(__file__).parent.parent / "data" / "sample_docs.csv"
            documents_df = load_documents(sample_path)

        if documents_df is not None:
            # Extract documents
            documents = documents_df["text"].tolist()

            # Display the number of documents
            st.info(f"Loaded {len(documents)} documents.")

            # Allow users to view the documents
            with st.expander("View Documents"):
                for i, doc in enumerate(documents):
                    title = (
                        documents_df["title"][i]
                        if "title" in documents_df.columns
                        else f"Document {i + 1}"
                    )
                    source = (
                        documents_df["source"][i]
                        if "source" in documents_df.columns
                        else "Unknown"
                    )
                    st.markdown(f"**{title}** (Source: {source})")
                    st.markdown(doc)
                    st.markdown("---")

            # Query input
            st.subheader("2. Ask a Question")
            query, search_clicked = create_query_input(
                "Enter your question:", button_text="Search with Reflection"
            )

            # Process query
            if search_clicked and query:
                # Initialize processing steps
                steps = []

                # Step 1: Initial retrieval
                with st.spinner("Performing initial retrieval..."):
                    start_time = time.time()

                    # Retrieve documents
                    retrieved_docs = embedding_retrieval(query, documents, top_k=top_k)

                    retrieval_time = time.time() - start_time

                    # Add to steps
                    steps.append(
                        {
                            "title": "Initial Retrieval",
                            "content": f"Retrieved {len(retrieved_docs)} documents in {retrieval_time:.2f} seconds.",
                            "time": retrieval_time,
                        }
                    )

                # Step 2: Evaluate retrieval
                with st.spinner("Evaluating retrieval quality..."):
                    start_time = time.time()

                    # Evaluate retrieval
                    evaluation_result = evaluate_retrieval(query, retrieved_docs)

                    evaluation_time = time.time() - start_time

                    # Calculate average relevance score
                    avg_score = 0
                    if evaluation_result and "document_scores" in evaluation_result:
                        scores = [
                            item.get("score", 0)
                            for item in evaluation_result["document_scores"]
                        ]
                        avg_score = sum(scores) / len(scores) if scores else 0

                    # Add to steps
                    steps.append(
                        {
                            "title": "Retrieval Evaluation",
                            "content": {
                                "Average Relevance": f"{avg_score:.1f}/10",
                                "Sufficient": "Yes"
                                if evaluation_result.get("overall", {}).get(
                                    "sufficient", False
                                )
                                else "No",
                                "Missing Information": evaluation_result.get(
                                    "overall", {}
                                ).get("missing_information", "None"),
                                "Suggested Reformulation": evaluation_result.get(
                                    "overall", {}
                                ).get("query_reformulation", "None"),
                            },
                            "time": evaluation_time,
                        }
                    )

                # Step 3: Re-retrieval if needed
                final_docs = retrieved_docs
                if avg_score < reflection_threshold or not evaluation_result.get(
                    "overall", {}
                ).get("sufficient", False):
                    with st.spinner(
                        "Performing re-retrieval with reformulated query..."
                    ):
                        start_time = time.time()

                        # Reformulate query
                        reformulated_query = reformulate_query(query, evaluation_result)

                        # Only proceed with re-retrieval if the query was actually reformulated
                        if reformulated_query != query:
                            # Retrieve documents with reformulated query
                            re_retrieved_docs = embedding_retrieval(
                                reformulated_query, documents, top_k=top_k
                            )

                            # Rerank all documents
                            all_docs = list(
                                set(
                                    [doc for doc, _ in retrieved_docs]
                                    + [doc for doc, _ in re_retrieved_docs]
                                )
                            )
                            final_docs = rerank_documents(query, all_docs, top_k=top_k)

                            re_retrieval_time = time.time() - start_time

                            # Add to steps
                            steps.append(
                                {
                                    "title": "Re-Retrieval",
                                    "content": {
                                        "Reformulated Query": reformulated_query,
                                        "Documents Retrieved": len(re_retrieved_docs),
                                        "Final Documents After Reranking": len(
                                            final_docs
                                        ),
                                    },
                                    "time": re_retrieval_time,
                                }
                            )

                # Step 4: Generate draft answer
                with st.spinner("Generating draft answer..."):
                    start_time = time.time()

                    # Prepare context
                    context_docs = [doc for doc, _ in final_docs]
                    if isinstance(context_docs[0], dict):
                        context = "\n\n".join(
                            [doc.get("text", "") for doc in context_docs]
                        )
                    else:
                        context = "\n\n".join(context_docs)

                    # Generate answer
                    draft_answer = generate_answer(query, context)

                    generation_time = time.time() - start_time

                    # Add to steps
                    steps.append(
                        {
                            "title": "Draft Answer Generation",
                            "content": draft_answer,
                            "time": generation_time,
                        }
                    )

                # Step 5: Evaluate answer faithfulness
                with st.spinner("Evaluating answer faithfulness..."):
                    start_time = time.time()

                    # Evaluate faithfulness
                    faithfulness_result = evaluate_answer_faithfulness(
                        draft_answer, context, query
                    )

                    faithfulness_time = time.time() - start_time

                    # Add to steps
                    steps.append(
                        {
                            "title": "Faithfulness Evaluation",
                            "content": {
                                "Faithfulness Score": f"{faithfulness_result.get('score', 0)}/10",
                                "Explanation": faithfulness_result.get(
                                    "explanation", "No explanation provided"
                                ),
                                "Issues": faithfulness_result.get(
                                    "issues", ["None identified"]
                                ),
                            },
                            "time": faithfulness_time,
                        }
                    )

                # Step 6: Refine answer if needed
                final_answer = draft_answer
                if faithfulness_result.get("score", 10) < faithfulness_threshold:
                    with st.spinner("Refining answer..."):
                        start_time = time.time()

                        # Refine answer
                        final_answer = refine_answer(
                            query, context, draft_answer, faithfulness_result
                        )

                        refinement_time = time.time() - start_time

                        # Add to steps
                        steps.append(
                            {
                                "title": "Answer Refinement",
                                "content": final_answer,
                                "time": refinement_time,
                            }
                        )

                # Display results
                st.subheader("Final Answer")
                st.markdown(final_answer)

                # Display retrieved documents
                st.subheader("Retrieved Documents")

                # Convert to format expected by display_documents
                docs_with_scores = []
                for doc, score in final_docs:
                    if isinstance(doc, dict):
                        doc_with_score = doc.copy()
                        doc_with_score["score"] = score
                        docs_with_scores.append(doc_with_score)
                    else:
                        docs_with_scores.append({"text": doc, "score": score})

                scores = [score for _, score in final_docs]
                display_documents(docs_with_scores, scores=scores)

                # Display processing steps if enabled
                if show_process:
                    display_processing_steps(steps)

            elif search_clicked:
                st.warning("Please enter a question.")
        else:
            st.warning(
                "Please upload a CSV file with documents or use the sample data."
            )

    # Add footer
    create_footer()


if __name__ == "__main__":
    main()
