"""
Evaluation metrics for RAG system performance assessment.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    query: str
    expected_answer: str
    actual_answer: str
    retrieved_docs: List[str]
    relevance_scores: List[float]
    response_time: float
    precision: float
    recall: float
    accuracy: float
    f1_score: float
    semantic_similarity: float
    answer_quality_score: float


class EvaluationMetrics:
    """Comprehensive evaluation metrics for RAG systems."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    
    def calculate_precision(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """Calculate precision@k for retrieved documents."""
        if not retrieved_docs:
            return 0.0
        
        relevant_retrieved = set(retrieved_docs) & set(relevant_docs)
        return len(relevant_retrieved) / len(retrieved_docs)
    
    def calculate_recall(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """Calculate recall@k for retrieved documents."""
        if not relevant_docs:
            return 1.0 if not retrieved_docs else 0.0
        
        relevant_retrieved = set(retrieved_docs) & set(relevant_docs)
        return len(relevant_retrieved) / len(relevant_docs)
    
    def calculate_accuracy(self, predicted_labels: List[bool], true_labels: List[bool]) -> float:
        """Calculate accuracy for binary classification."""
        if len(predicted_labels) != len(true_labels):
            raise ValueError("Predicted and true labels must have the same length")
        
        correct = sum(1 for p, t in zip(predicted_labels, true_labels) if p == t)
        return correct / len(predicted_labels)
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using TF-IDF and cosine similarity."""
        try:
            # Clean and prepare texts
            texts = [self._clean_text(text1), self._clean_text(text2)]
            
            # Create TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception:
            return 0.0
    
    def calculate_answer_quality_score(self, answer: str, expected_answer: str) -> float:
        """Calculate overall answer quality score based on multiple factors."""
        if not answer or not expected_answer:
            return 0.0
        
        # Semantic similarity (40% weight)
        semantic_sim = self.calculate_semantic_similarity(answer, expected_answer)
        
        # Length appropriateness (20% weight)
        length_score = self._calculate_length_score(answer, expected_answer)
        
        # Completeness (20% weight)
        completeness_score = self._calculate_completeness_score(answer, expected_answer)
        
        # Clarity (20% weight)
        clarity_score = self._calculate_clarity_score(answer)
        
        # Weighted combination
        quality_score = (
            0.4 * semantic_sim +
            0.2 * length_score +
            0.2 * completeness_score +
            0.2 * clarity_score
        )
        
        return min(1.0, max(0.0, quality_score))
    
    def evaluate_rag_response(
        self,
        query: str,
        expected_answer: str,
        actual_answer: str,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        response_time: float
    ) -> EvaluationResult:
        """Comprehensive evaluation of a RAG response."""
        
        # Calculate relevance scores for retrieved documents
        relevance_scores = []
        for doc in retrieved_docs:
            if doc in relevant_docs:
                relevance_scores.append(1.0)
            else:
                # Calculate semantic similarity to query as proxy for relevance
                relevance_scores.append(self.calculate_semantic_similarity(query, doc))
        
        # Calculate metrics
        precision = self.calculate_precision(retrieved_docs, relevant_docs)
        recall = self.calculate_recall(retrieved_docs, relevant_docs)
        
        # For accuracy, we need binary labels
        predicted_labels = [1.0 if score > 0.5 else 0.0 for score in relevance_scores]
        true_labels = [1.0 if doc in relevant_docs else 0.0 for doc in retrieved_docs]
        accuracy = self.calculate_accuracy(predicted_labels, true_labels)
        
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Semantic similarity between expected and actual answers
        semantic_similarity = self.calculate_semantic_similarity(expected_answer, actual_answer)
        
        # Overall answer quality
        answer_quality = self.calculate_answer_quality_score(actual_answer, expected_answer)
        
        return EvaluationResult(
            query=query,
            expected_answer=expected_answer,
            actual_answer=actual_answer,
            retrieved_docs=retrieved_docs,
            relevance_scores=relevance_scores,
            response_time=response_time,
            precision=precision,
            recall=recall,
            accuracy=accuracy,
            f1_score=f1,
            semantic_similarity=semantic_similarity,
            answer_quality_score=answer_quality
        )
    
    def batch_evaluate(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Evaluate multiple test cases and return results as DataFrame."""
        results = []
        
        for test_case in test_cases:
            try:
                result = self.evaluate_rag_response(
                    query=test_case["query"],
                    expected_answer=test_case["expected_answer"],
                    actual_answer=test_case["actual_answer"],
                    retrieved_docs=test_case.get("retrieved_docs", []),
                    relevant_docs=test_case.get("relevant_docs", []),
                    response_time=test_case.get("response_time", 0.0)
                )
                
                results.append({
                    "query": result.query,
                    "expected_answer": result.expected_answer,
                    "actual_answer": result.actual_answer,
                    "response_time": result.response_time,
                    "precision": result.precision,
                    "recall": result.recall,
                    "accuracy": result.accuracy,
                    "f1_score": result.f1_score,
                    "semantic_similarity": result.semantic_similarity,
                    "answer_quality_score": result.answer_quality_score,
                    "num_retrieved_docs": len(result.retrieved_docs),
                    "num_relevant_docs": len(result.relevance_scores),
                    "avg_relevance_score": np.mean(result.relevance_scores) if result.relevance_scores else 0.0
                })
                
            except Exception as e:
                print(f"Error evaluating test case: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def calculate_summary_metrics(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate summary metrics from evaluation results."""
        if results_df.empty:
            return {}
        
        return {
            "avg_precision": results_df["precision"].mean(),
            "avg_recall": results_df["recall"].mean(),
            "avg_accuracy": results_df["accuracy"].mean(),
            "avg_f1_score": results_df["f1_score"].mean(),
            "avg_semantic_similarity": results_df["semantic_similarity"].mean(),
            "avg_answer_quality": results_df["answer_quality_score"].mean(),
            "avg_response_time": results_df["response_time"].mean(),
            "total_queries": len(results_df),
            "successful_queries": len(results_df[results_df["answer_quality_score"] > 0.5])
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean text for processing."""
        if not text:
            return ""
        
        # Remove special characters and normalize whitespace
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _calculate_length_score(self, answer: str, expected: str) -> float:
        """Calculate length appropriateness score."""
        if not answer or not expected:
            return 0.0
        
        length_ratio = len(answer) / len(expected) if len(expected) > 0 else 0.0
        
        # Optimal ratio is around 0.8-1.2
        if 0.8 <= length_ratio <= 1.2:
            return 1.0
        elif 0.5 <= length_ratio <= 1.5:
            return 0.8
        elif 0.3 <= length_ratio <= 2.0:
            return 0.6
        else:
            return 0.3
    
    def _calculate_completeness_score(self, answer: str, expected: str) -> float:
        """Calculate completeness score based on keyword coverage."""
        if not answer or not expected:
            return 0.0
        
        # Extract key terms from expected answer
        expected_terms = set(self._clean_text(expected).split())
        answer_terms = set(self._clean_text(answer).split())
        
        if not expected_terms:
            return 1.0
        
        # Calculate coverage
        coverage = len(expected_terms & answer_terms) / len(expected_terms)
        return min(1.0, coverage)
    
    def _calculate_clarity_score(self, answer: str) -> float:
        """Calculate clarity score based on text structure and readability."""
        if not answer:
            return 0.0
        
        # Simple heuristics for clarity
        sentences = re.split(r'[.!?]+', answer)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Optimal sentence length is 15-20 words
        if 15 <= avg_sentence_length <= 20:
            length_score = 1.0
        elif 10 <= avg_sentence_length <= 25:
            length_score = 0.8
        else:
            length_score = 0.6
        
        # Check for proper structure (has sentences, not just fragments)
        structure_score = 1.0 if len(sentences) > 1 else 0.7
        
        return (length_score + structure_score) / 2


# Convenience functions
def calculate_precision(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
    """Calculate precision@k for retrieved documents."""
    metrics = EvaluationMetrics()
    return metrics.calculate_precision(retrieved_docs, relevant_docs)


def calculate_recall(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
    """Calculate recall@k for retrieved documents."""
    metrics = EvaluationMetrics()
    return metrics.calculate_recall(retrieved_docs, relevant_docs)


def calculate_accuracy(predicted_labels: List[bool], true_labels: List[bool]) -> float:
    """Calculate accuracy for binary classification."""
    metrics = EvaluationMetrics()
    return metrics.calculate_accuracy(predicted_labels, true_labels)
