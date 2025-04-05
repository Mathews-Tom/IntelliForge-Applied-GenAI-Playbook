"""
Evaluation utility functions for IntelliForge applications.

This module contains functions for evaluating retrieval quality,
answer faithfulness, and other metrics used across multiple applications.
"""

from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
import sys
from pathlib import Path

# Add the project root to the Python path to import core modules
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from core.llm.gemini_utils import GeminiModelType, generate_content

def evaluate_retrieval_relevance(query: str, 
                               retrieved_docs: List[Tuple[Union[str, Dict[str, Any]], float]]) -> str:
    """
    Evaluate the relevance of retrieved documents to a query using Gemini.
    
    Args:
        query: Query string
        retrieved_docs: List of (document, score) tuples
        
    Returns:
        Evaluation report as a string
    """
    # Extract text from documents if they are dictionaries
    doc_texts = []
    for doc, _ in retrieved_docs:
        if isinstance(doc, dict):
            doc_texts.append(doc.get('text', ''))
        else:
            doc_texts.append(doc)
    
    # Create prompt for Gemini
    docs_text = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(doc_texts)])
    
    prompt = f"""
    Evaluate the relevance of the following documents to the query.

    Query: {query}

    {docs_text}

    For each document, provide a relevance score from 0 to 10, where:
    - 0-3: Not relevant
    - 4-6: Somewhat relevant
    - 7-10: Highly relevant

    Also provide a brief explanation for each score.

    Format your response as a table with columns: Document Number, Relevance Score, Explanation
    """

    try:
        response = generate_content(GeminiModelType.GEMINI_2_5_PRO, prompt)

        if response:
            return response
        else:
            return "Failed to evaluate relevance."
    except Exception as e:
        return f"Error evaluating relevance: {e}"

def calculate_precision_recall(relevant_docs: List[int], 
                             retrieved_docs: List[int]) -> Dict[str, float]:
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
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def evaluate_answer_faithfulness(answer: str, 
                               context: str,
                               query: str) -> Dict[str, Any]:
    """
    Evaluate the faithfulness of an answer to the provided context.
    
    Args:
        answer: Generated answer
        context: Context used to generate the answer
        query: Original query
        
    Returns:
        Dictionary with faithfulness score and explanation
    """
    prompt = f"""
    Evaluate the faithfulness of the following answer to the provided context.
    
    Query: {query}
    
    Context:
    {context}
    
    Answer:
    {answer}
    
    A faithful answer should:
    1. Only contain information that is present in the context or is common knowledge
    2. Not contradict the context
    3. Not add speculative or unsupported details
    
    Rate the faithfulness on a scale of 0 to 10, where:
    - 0-3: Not faithful (contains significant information not in the context)
    - 4-6: Somewhat faithful (contains some information not in the context)
    - 7-10: Highly faithful (contains only information from the context)
    
    Return your evaluation as a JSON object with the following fields:
    - score: The faithfulness score (0-10)
    - explanation: A brief explanation of your rating
    - issues: A list of specific issues found (if any)
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
                    "score": 5,
                    "explanation": "Failed to parse evaluation result.",
                    "issues": ["Error parsing Gemini response as JSON"]
                }
        else:
            return {
                "score": 5,
                "explanation": "Failed to evaluate faithfulness.",
                "issues": ["No response from Gemini"]
            }
    except Exception as e:
        return {
            "score": 5,
            "explanation": f"Error evaluating faithfulness: {e}",
            "issues": [str(e)]
        }

def evaluate_answer_quality(answer: str, 
                          query: str) -> Dict[str, Any]:
    """
    Evaluate the overall quality of an answer.
    
    Args:
        answer: Generated answer
        query: Original query
        
    Returns:
        Dictionary with quality metrics
    """
    prompt = f"""
    Evaluate the quality of the following answer to the query.
    
    Query: {query}
    
    Answer:
    {answer}
    
    Evaluate the answer on the following dimensions:
    1. Relevance: How well the answer addresses the query (0-10)
    2. Completeness: How thoroughly the answer covers the topic (0-10)
    3. Clarity: How clear and understandable the answer is (0-10)
    4. Conciseness: How concise and to-the-point the answer is (0-10)
    
    Return your evaluation as a JSON object with the following fields:
    - relevance: Score for relevance (0-10)
    - completeness: Score for completeness (0-10)
    - clarity: Score for clarity (0-10)
    - conciseness: Score for conciseness (0-10)
    - overall: Average of the above scores
    - explanation: A brief explanation of your rating
    """
    
    try:
        response = generate_content(GeminiModelType.GEMINI_2_5_PRO, prompt)
        
        if response:
            # Try to parse the response as JSON
            try:
                import json
                result = json.loads(response.strip())
                
                # Calculate overall score if not provided
                if "overall" not in result:
                    scores = [
                        result.get("relevance", 5),
                        result.get("completeness", 5),
                        result.get("clarity", 5),
                        result.get("conciseness", 5)
                    ]
                    result["overall"] = sum(scores) / len(scores)
                    
                return result
            except:
                # If parsing fails, return a simple result
                return {
                    "relevance": 5,
                    "completeness": 5,
                    "clarity": 5,
                    "conciseness": 5,
                    "overall": 5,
                    "explanation": "Failed to parse evaluation result."
                }
        else:
            return {
                "relevance": 5,
                "completeness": 5,
                "clarity": 5,
                "conciseness": 5,
                "overall": 5,
                "explanation": "Failed to evaluate answer quality."
            }
    except Exception as e:
        return {
            "relevance": 5,
            "completeness": 5,
            "clarity": 5,
            "conciseness": 5,
            "overall": 5,
            "explanation": f"Error evaluating answer quality: {e}"
        }
