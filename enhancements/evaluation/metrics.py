"""
RAG evaluation metrics implementation.
"""
import logging
from typing import Dict, Any, List, Optional

from rag_client import RAGClient

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGEvaluator:
    """Evaluator for RAG responses using RAGA metrics."""
    
    def __init__(self, rag_client: RAGClient):
        """
        Initialize the RAG evaluator.
        
        Args:
            rag_client (RAGClient): Client for interacting with the baseline RAG API
        """
        self.client = rag_client
        logger.info("Initialized RAG evaluator")
    
    def evaluate_faithfulness(self, question: str, response: str, context: str) -> float:
        """
        Evaluate the faithfulness/groundedness of a response.
        
        Faithfulness measures how well the generated answer is supported by the 
        retrieved documents and identifies hallucinations.
        
        Args:
            question (str): The original question
            response (str): The generated response
            context (str): The context used for generation
            
        Returns:
            float: A score from 0 to 10
        """
        system_prompt = (
            "You are an expert evaluator assessing the faithfulness of responses to security policy questions. "
            "Faithfulness measures how well the response is grounded in the provided context, without adding "
            "information not present in the context or contradicting it.\n\n"
            
            "Guidelines for evaluation:\n"
            "- A faithful response only includes information that can be directly supported by the context\n"
            "- A response with hallucinations (statements not backed by the context) has low faithfulness\n"
            "- Contradictions to information in the context severely reduce faithfulness\n"
            "- Inferences that reasonably follow from the context don't reduce faithfulness\n\n"
            
            "Rate the faithfulness of the response on a scale from 0 to 10, where:\n"
            "- 0: Completely unfaithful, containing many statements not supported by the context or contradicting it\n"
            "- 5: Partially faithful, with some statements supported by the context and some not\n"
            "- 10: Completely faithful, with all statements directly supported by the context\n\n"
            
            "Return ONLY a number from 0 to 10 without any explanation."
        )
        
        user_prompt = (
            f"Question: {question}\n\n"
            f"Context: {context}\n\n"
            f"Response: {response}\n\n"
            f"Faithfulness score (0-10):"
        )
        
        # Get the score
        result = self.client.get_openai_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=10
        )
        
        # Extract the score
        try:
            score = float(result.strip())
            return min(max(score, 0), 10)  # Ensure score is between 0 and 10
        except ValueError:
            logger.error(f"Could not parse faithfulness score: {result}")
            return 0.0
    
    def evaluate_context_relevance(self, question: str, context: str) -> float:
        """
        Evaluate the relevance of the retrieved context to the question.
        
        Context relevance measures how well the retrieved documents relate to the query.
        
        Args:
            question (str): The original question
            context (str): The retrieved context
            
        Returns:
            float: A score from 0 to 10
        """
        system_prompt = (
            "You are an expert evaluator assessing the relevance of context information to security policy questions. "
            "Context relevance measures how well the retrieved information relates to and helps answer the query.\n\n"
            
            "Guidelines for evaluation:\n"
            "- Highly relevant context directly addresses the specific security policy aspects mentioned in the question\n"
            "- Somewhat relevant context relates to the general topic but may not address the specific question\n"
            "- Irrelevant context contains information unrelated to the security policy topic in the question\n\n"
            
            "Rate the context relevance on a scale from 0 to 10, where:\n"
            "- 0: Completely irrelevant, containing no information related to the question\n"
            "- 5: Somewhat relevant, containing general information about the topic but missing specific details\n"
            "- 10: Highly relevant, containing specific information that directly answers the question\n\n"
            
            "Return ONLY a number from 0 to 10 without any explanation."
        )
        
        user_prompt = (
            f"Question: {question}\n\n"
            f"Retrieved Context: {context}\n\n"
            f"Context relevance score (0-10):"
        )
        
        # Get the score
        result = self.client.get_openai_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=10
        )
        
        # Extract the score
        try:
            score = float(result.strip())
            return min(max(score, 0), 10)  # Ensure score is between 0 and 10
        except ValueError:
            logger.error(f"Could not parse context relevance score: {result}")
            return 0.0
    
    def evaluate_answer_relevance(self, question: str, response: str) -> float:
        """
        Evaluate the relevance of the response to the question.
        
        Answer relevance assesses how well the generated answer addresses the user's question.
        
        Args:
            question (str): The original question
            response (str): The generated response
            
        Returns:
            float: A score from 0 to 10
        """
        system_prompt = (
            "You are an expert evaluator assessing the relevance of responses to security policy questions. "
            "Answer relevance measures how well the response addresses the specific question asked.\n\n"
            
            "Guidelines for evaluation:\n"
            "- A highly relevant answer directly addresses all aspects of the question\n"
            "- A somewhat relevant answer addresses the general topic but may miss specific aspects\n"
            "- An irrelevant answer does not address the question asked\n\n"
            
            "Rate the answer relevance on a scale from 0 to 10, where:\n"
            "- 0: Completely irrelevant, not addressing the question at all\n"
            "- 5: Somewhat relevant, addressing the general topic but missing key aspects\n"
            "- 10: Highly relevant, directly addressing all aspects of the question\n\n"
            
            "Return ONLY a number from 0 to 10 without any explanation."
        )
        
        user_prompt = (
            f"Question: {question}\n\n"
            f"Response: {response}\n\n"
            f"Answer relevance score (0-10):"
        )
        
        # Get the score
        result = self.client.get_openai_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=10
        )
        
        # Extract the score
        try:
            score = float(result.strip())
            return min(max(score, 0), 10)  # Ensure score is between 0 and 10
        except ValueError:
            logger.error(f"Could not parse answer relevance score: {result}")
            return 0.0
    
    def evaluate_completeness(self, question: str, response: str) -> float:
        """
        Evaluate the completeness of a response.
        
        Completeness determines if all aspects of the question are addressed.
        
        Args:
            question (str): The original question
            response (str): The generated response
            
        Returns:
            float: A score from 0 to 10
        """
        system_prompt = (
            "You are an expert evaluator assessing the completeness of responses to security policy questions. "
            "Completeness measures how thoroughly the response addresses all aspects of the question.\n\n"
            
            "Guidelines for evaluation:\n"
            "- A complete response answers all parts of the question without missing any key elements\n"
            "- An incomplete response addresses only some aspects of the question\n"
            "- Even if an aspect can't be answered, a complete response acknowledges this limitation\n\n"
            
            "Rate the completeness of the response on a scale from 0 to 10, where:\n"
            "- 0: Completely incomplete, addressing none of the question's aspects\n"
            "- 5: Partially complete, addressing some aspects but missing others\n"
            "- 10: Completely complete, thoroughly addressing all aspects of the question\n\n"
            
            "Return ONLY a number from 0 to 10 without any explanation."
        )
        
        user_prompt = (
            f"Question: {question}\n\n"
            f"Response: {response}\n\n"
            f"Completeness score (0-10):"
        )
        
        # Get the score
        result = self.client.get_openai_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=10
        )
        
        # Extract the score
        try:
            score = float(result.strip())
            return min(max(score, 0), 10)  # Ensure score is between 0 and 10
        except ValueError:
            logger.error(f"Could not parse completeness score: {result}")
            return 0.0
    
    def evaluate_citation_accuracy(self, response: str) -> float:
        """
        Evaluate the citation accuracy of a response.
        
        Citation accuracy checks if the model correctly attributes information to sources.
        
        Args:
            response (str): The generated response
            
        Returns:
            float: A score from 0 to 10
        """
        system_prompt = (
            "You are an expert evaluator assessing the citation accuracy in responses about security policies. "
            "Citation accuracy measures how well the response references specific policy sections, documents, "
            "or guidelines when making statements about security policies.\n\n"
            
            "Guidelines for evaluation:\n"
            "- High citation accuracy includes specific references to policy sections (e.g., 'According to the Password Policy section...')\n"
            "- Medium citation accuracy includes general references without specifics (e.g., 'The security policy states...')\n"
            "- Low citation accuracy makes claims without indicating their source in the policy\n\n"
            
            "Rate the citation accuracy of the response on a scale from 0 to 10, where:\n"
            "- 0: No citations or references to specific policy sections\n"
            "- 5: Some references to policy but lacking specificity\n"
            "- 10: Excellent, specific citations to relevant policy sections\n\n"
            
            "Return ONLY a number from 0 to 10 without any explanation."
        )
        
        user_prompt = (
            f"Response: {response}\n\n"
            f"Citation accuracy score (0-10):"
        )
        
        # Get the score
        result = self.client.get_openai_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=10
        )
        
        # Extract the score
        try:
            score = float(result.strip())
            return min(max(score, 0), 10)  # Ensure score is between 0 and 10
        except ValueError:
            logger.error(f"Could not parse citation accuracy score: {result}")
            return 0.0
    
    def evaluate_coherence(self, response: str) -> float:
        """
        Evaluate the coherence of a response.
        
        Coherence evaluates the logical flow and readability of the response.
        
        Args:
            response (str): The generated response
            
        Returns:
            float: A score from 0 to 10
        """
        system_prompt = (
            "You are an expert evaluator assessing the coherence of responses about security policies. "
            "Coherence measures the logical flow, structure, and readability of the response.\n\n"
            
            "Guidelines for evaluation:\n"
            "- A highly coherent response has clear organization, logical progression, and consistent terminology\n"
            "- A somewhat coherent response may have some logical jumps or inconsistent structure\n"
            "- An incoherent response is disorganized, confusing, or contradictory\n\n"
            
            "Rate the coherence of the response on a scale from 0 to 10, where:\n"
            "- 0: Completely incoherent, disorganized, and difficult to follow\n"
            "- 5: Somewhat coherent, but with some organizational or clarity issues\n"
            "- 10: Highly coherent, well-structured, and easy to understand\n\n"
            
            "Return ONLY a number from 0 to 10 without any explanation."
        )
        
        user_prompt = (
            f"Response: {response}\n\n"
            f"Coherence score (0-10):"
        )
        
        # Get the score
        result = self.client.get_openai_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=10
        )
        
        # Extract the score
        try:
            score = float(result.strip())
            return min(max(score, 0), 10)  # Ensure score is between 0 and 10
        except ValueError:
            logger.error(f"Could not parse coherence score: {result}")
            return 0.0
    
    def evaluate_all_metrics(self, question: str, response: str, context: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate all metrics for a response.
        
        Args:
            question (str): The original question
            response (str): The generated response
            context (Optional[str]): The context used for generation
            
        Returns:
            Dict[str, float]: A dictionary of metric names and scores
        """
        # If context is not provided, use the baseline response as a proxy
        if context is None:
            context = self.client.query(question)
        
        # Evaluate each metric
        metrics = {}
        
        # Core metrics
        metrics["faithfulness"] = self.evaluate_faithfulness(question, response, context)
        metrics["completeness"] = self.evaluate_completeness(question, response)
        metrics["citation"] = self.evaluate_citation_accuracy(response)
        
        # Additional metrics
        metrics["context_relevance"] = self.evaluate_context_relevance(question, context)
        metrics["answer_relevance"] = self.evaluate_answer_relevance(question, response)
        metrics["coherence"] = self.evaluate_coherence(response)
        
        # Calculate average score (using all metrics)
        metrics["average"] = sum(metrics.values()) / len(metrics)
        
        return metrics