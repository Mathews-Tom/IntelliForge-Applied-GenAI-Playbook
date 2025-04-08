"""
Evaluation utility functions for IntelliForge applications.

This module contains functions for evaluating retrieval quality,
answer faithfulness, and other metrics used across multiple applications.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import streamlit as st  # Added for caching

# Add the project root to the Python path to import core modules
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from core.llm.gemini_utils import GeminiModelType, generate_content


@st.cache_data(show_spinner="Evaluating retrieval relevance...")
async def evaluate_retrieval_relevance(
    query: str, retrieved_docs: List[Tuple[Union[str, Dict[str, Any]], float]]
) -> str:
    """
    Evaluate the relevance of retrieved documents to a query using Gemini (async, cached).

    Args:
        query: Query string.
        retrieved_docs: List of (document, score) tuples.

    Returns:
        Evaluation report as a string.
    """
    print(
        f"Cache miss: Evaluating relevance for query '{query[:50]}...'"
    )  # Log cache misses

    if not retrieved_docs:
        return "No documents retrieved to evaluate."

    # Extract text from documents if they are dictionaries
    doc_texts = []
    for doc, _ in retrieved_docs:
        if isinstance(doc, dict):
            # Ensure text is string
            doc_texts.append(str(doc.get("text", "")))
        else:
            # Ensure text is string
            doc_texts.append(str(doc) if doc is not None else "")

    # Create prompt for Gemini
    docs_text = "\n\n".join(
        [f"Document {i + 1}:\n{doc}" for i, doc in enumerate(doc_texts)]
    )

    prompt = f"""Evaluate the relevance of the following documents to the query.

Query: {query}

Documents:
{docs_text}

For each document, provide a relevance score from 0 to 10 (0=Not relevant, 10=Highly relevant) and a brief explanation.
Format your response as a Markdown table with columns: "Document Number", "Relevance Score", "Explanation"."""

    try:
        # Use await for the async generate_content
        response = await generate_content(
            GeminiModelType.GEMINI_2_5_PRO, prompt
        )  # Consider faster/cheaper model?

        if response:
            return response
        else:
            st.warning("Failed to get relevance evaluation from LLM.")
            return "Evaluation failed: No response from LLM."
    except Exception as e:
        st.error(f"Error during relevance evaluation: {e}")
        return f"Evaluation failed: Error - {e}"


def calculate_precision_recall(
    relevant_docs: List[int], retrieved_docs: List[int]
) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score.

    Args:
        relevant_docs: List of indices of relevant documents
        retrieved_docs: List of indices of retrieved documents

    Returns:
        Dictionary with precision, recall, and F1 score
    """
    relevant_set = set(relevant_docs)
    retrieved_set = set(retrieved_docs)

    # Calculate true positives (relevant documents that were retrieved)
    true_positives = len(relevant_set.intersection(retrieved_set))

    # Calculate precision (fraction of retrieved documents that are relevant)
    precision = true_positives / len(retrieved_set) if retrieved_set else 0

    # Calculate recall (fraction of relevant documents that were retrieved)
    recall = true_positives / len(relevant_set) if relevant_set else 0

    # Calculate F1 score (harmonic mean of precision and recall)
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return {"precision": precision, "recall": recall, "f1": f1}


@st.cache_data(show_spinner="Evaluating answer faithfulness...")
async def evaluate_answer_faithfulness(
    answer: str, context: str, query: str
) -> Dict[str, Any]:
    """
    Evaluate the faithfulness of an answer to the provided context (async, cached).

    Args:
        answer: Generated answer.
        context: Context used to generate the answer.
        query: Original query.

    Returns:
        Dictionary with faithfulness score, explanation, and issues.
    """
    print(
        f"Cache miss: Evaluating faithfulness for query '{query[:50]}...'"
    )  # Log cache misses

    # Basic input validation
    if not answer or not context:
        return {
            "score": 0,
            "explanation": "Input answer or context is empty.",
            "issues": ["Missing input"],
        }

    prompt = f"""Evaluate the faithfulness of the following answer ONLY based on the provided context.

Query: {query}

Context:
---
{context}
---

Answer:
---
{answer}
---

A faithful answer must ONLY contain information directly supported by the context. It should not contradict the context or add external information (unless it's extremely common knowledge implicitly needed to understand the context).

Rate the faithfulness on a scale of 0 to 10 (0=Not faithful, 10=Highly faithful).
Provide a brief explanation for your rating.
List specific phrases or claims in the answer that are NOT supported by the context.

Return your evaluation ONLY as a JSON object with keys "score" (int), "explanation" (str), and "issues" (list of strings). Example:
{{
  "score": 8,
  "explanation": "The answer is mostly faithful but includes one minor detail not explicitly mentioned.",
  "issues": ["Claim 'XYZ' is not supported by the context."]
}}"""

    # Default result in case of failure
    default_fail_result = {
        "score": -1,  # Use -1 to indicate failure
        "explanation": "Evaluation failed.",
        "issues": ["Evaluation could not be completed."],
    }

    try:
        # Use await for the async generate_content
        response = await generate_content(
            GeminiModelType.GEMINI_2_5_PRO, prompt
        )  # Consider faster/cheaper model?

        if response:
            # Try to parse the response as JSON
            try:
                import json

                # Clean potential markdown code fences
                cleaned_response = (
                    response.strip().strip("```json").strip("```").strip()
                )
                result = json.loads(cleaned_response)
                # Basic validation of the parsed structure
                if (
                    isinstance(result, dict)
                    and "score" in result
                    and "explanation" in result
                    and "issues" in result
                ):
                    # Ensure score is an int
                    result["score"] = int(result.get("score", -1))
                    return result
                else:
                    print(
                        f"Warning: Faithfulness evaluation response JSON structure invalid: {response}"
                    )
                    default_fail_result["explanation"] = (
                        "Failed to parse evaluation result: Invalid JSON structure."
                    )
                    default_fail_result["issues"] = [
                        f"Invalid JSON structure in response: {response}"
                    ]
                    return default_fail_result

            except json.JSONDecodeError as json_err:
                print(
                    f"Warning: Faithfulness evaluation response JSON parsing failed: {json_err}. Response: {response}"
                )
                default_fail_result["explanation"] = (
                    f"Failed to parse evaluation result: {json_err}"
                )
                default_fail_result["issues"] = [
                    f"JSON parsing error in response: {response}"
                ]
                return default_fail_result
            except Exception as parse_err:
                print(
                    f"Warning: Faithfulness evaluation response parsing failed unexpectedly: {parse_err}. Response: {response}"
                )
                default_fail_result["explanation"] = (
                    f"Failed to parse evaluation result: {parse_err}"
                )
                default_fail_result["issues"] = [
                    f"Unexpected parsing error in response: {response}"
                ]
                return default_fail_result
        else:
            st.warning("Failed to get faithfulness evaluation from LLM.")
            default_fail_result["explanation"] = (
                "Failed to evaluate faithfulness: No response from LLM."
            )
            default_fail_result["issues"] = ["No response from LLM"]
            return default_fail_result
    except Exception as e:
        st.error(f"Error during faithfulness evaluation: {e}")
        default_fail_result["explanation"] = f"Error evaluating faithfulness: {e}"
        default_fail_result["issues"] = [str(e)]
        return default_fail_result


@st.cache_data(show_spinner="Evaluating answer quality...")
async def evaluate_answer_quality(answer: str, query: str) -> Dict[str, Any]:
    """
    Evaluate the overall quality of an answer (async, cached).

    Args:
        answer: Generated answer.
        query: Original query.

    Returns:
        Dictionary with quality metrics (relevance, completeness, clarity, conciseness, overall, explanation).
    """
    print(
        f"Cache miss: Evaluating quality for query '{query[:50]}...'"
    )  # Log cache misses

    if not answer:
        return {
            "relevance": 0,
            "completeness": 0,
            "clarity": 0,
            "conciseness": 0,
            "overall": 0,
            "explanation": "Input answer is empty.",
        }

    prompt = f"""Evaluate the quality of the following answer provided for the query.

Query: {query}

Answer:
---
{answer}
---

Evaluate the answer on the following dimensions, each on a scale of 0 to 10:
1. Relevance: How well the answer directly addresses the specific query asked.
2. Completeness: How thoroughly the answer covers the main points needed to address the query.
3. Clarity: How clear, understandable, and well-structured the answer is.
4. Conciseness: How efficiently the answer conveys information without unnecessary verbosity.

Return your evaluation ONLY as a JSON object with keys "relevance" (int), "completeness" (int), "clarity" (int), "conciseness" (int), and "explanation" (str, a brief overall justification). Example:
{{
  "relevance": 9,
  "completeness": 7,
  "clarity": 8,
  "conciseness": 6,
  "explanation": "The answer is relevant and clear, but could be more complete and less verbose."
}}"""

    # Default result in case of failure
    default_fail_result = {
        "relevance": -1,
        "completeness": -1,
        "clarity": -1,
        "conciseness": -1,
        "overall": -1,
        "explanation": "Evaluation failed.",
    }

    try:
        # Use await for the async generate_content
        response = await generate_content(
            GeminiModelType.GEMINI_2_5_PRO, prompt
        )  # Consider faster/cheaper model?

        if response:
            # Try to parse the response as JSON
            try:
                import json

                # Clean potential markdown code fences
                cleaned_response = (
                    response.strip().strip("```json").strip("```").strip()
                )
                result = json.loads(cleaned_response)

                # Basic validation and calculation of overall score
                required_keys = [
                    "relevance",
                    "completeness",
                    "clarity",
                    "conciseness",
                    "explanation",
                ]
                if isinstance(result, dict) and all(
                    key in result for key in required_keys
                ):
                    scores = []
                    for key in ["relevance", "completeness", "clarity", "conciseness"]:
                        # Ensure scores are ints
                        score_val = int(result.get(key, -1))
                        result[key] = score_val
                        if score_val != -1:
                            scores.append(score_val)

                    result["overall"] = sum(scores) / len(scores) if scores else -1
                    return result
                else:
                    print(
                        f"Warning: Quality evaluation response JSON structure invalid: {response}"
                    )
                    default_fail_result["explanation"] = (
                        "Failed to parse evaluation result: Invalid JSON structure."
                    )
                    return default_fail_result

            except json.JSONDecodeError as json_err:
                print(
                    f"Warning: Quality evaluation response JSON parsing failed: {json_err}. Response: {response}"
                )
                default_fail_result["explanation"] = (
                    f"Failed to parse evaluation result: {json_err}"
                )
                return default_fail_result
            except Exception as parse_err:
                print(
                    f"Warning: Quality evaluation response parsing failed unexpectedly: {parse_err}. Response: {response}"
                )
                default_fail_result["explanation"] = (
                    f"Failed to parse evaluation result: {parse_err}"
                )
                return default_fail_result
        else:
            st.warning("Failed to get quality evaluation from LLM.")
            default_fail_result["explanation"] = (
                "Failed to evaluate answer quality: No response from LLM."
            )
            return default_fail_result
    except Exception as e:
        st.error(f"Error during quality evaluation: {e}")
        default_fail_result["explanation"] = f"Error evaluating answer quality: {e}"
        return default_fail_result
        return default_fail_result
