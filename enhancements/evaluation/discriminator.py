"""
LLM-based discriminator for evaluating RAG responses.
"""
import logging
import re
from typing import Dict, Any, List, Optional

from rag_client import RAGClient

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMDiscriminator:
    """LLM-based discriminator for evaluating RAG responses."""
    
    def __init__(self, rag_client: RAGClient):
        """
        Initialize the LLM discriminator.
        
        Args:
            rag_client (RAGClient): Client for interacting with the baseline RAG API
        """
        self.client = rag_client
        logger.info("Initialized LLM discriminator")
    
    def evaluate(self, question: str, response: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a response using the LLM as a discriminator.
        
        Args:
            question (str): The original question
            response (str): The generated response
            context (Optional[str]): The context used for generation
            
        Returns:
            Dict[str, Any]: A dictionary with evaluation results
        """
        # If context is not provided, use the baseline response as a proxy
        if context is None:
            context = self.client.query(question)
        
        system_prompt = (
            "You are an expert security policy evaluator tasked with assessing the quality of responses "
            "to security policy questions. Your evaluation should be detailed, specific, and focused on "
            "how well the response serves an employee trying to understand company security policies.\n\n"
            
            "Evaluate the response on these five criteria:\n"
            "1. Policy Accuracy (0-10): How accurately does the response reflect security policy information in the context?\n"
            "2. Completeness (0-10): How thoroughly does the response address all aspects of the question?\n"
            "3. Policy Relevance (0-10): How relevant is the response to the specific security policy question asked?\n"
            "4. Clarity & Structure (0-10): How well-organized and easy to understand is the response?\n"
            "5. Actionability (0-10): How useful would this response be for an employee needing to follow security procedures?\n\n"
            
            "For each criterion:\n"
            "- Provide a score from 0 to 10\n"
            "- Give 2-3 sentences explaining the score\n"
            "- Highlight specific examples from the response\n\n"
            
            "End with an overall assessment (2-3 paragraphs) and overall score (0-10)."
        )
        
        user_prompt = (
            f"Security Policy Question: {question}\n\n"
            f"Context Information: {context}\n\n"
            f"Response to Evaluate: {response}\n\n"
            f"Please provide your detailed evaluation of this response."
        )
        
        # Get the evaluation
        evaluation = self.client.get_openai_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=1000
        )
        
        # Extract scores and calculate the overall score
        raw_metrics = self._extract_scores(evaluation)
        
        # Format the result
        result = {
            "detailed_evaluation": evaluation,
            "raw_metrics": raw_metrics,
            "overall_score": raw_metrics.get("overall", 0.0)
        }
        
        return result
    
    def _extract_scores(self, evaluation: str) -> Dict[str, float]:
        """
        Extract numerical scores from the evaluation text.
        
        Args:
            evaluation (str): The evaluation text
            
        Returns:
            Dict[str, float]: A dictionary of metric names and scores
        """
        scores = {}
        
        # Define patterns to look for specific scores
        patterns = {
            "policy_accuracy": r"policy accuracy:?\s*(\d+(?:\.\d+)?)",
            "completeness": r"completeness:?\s*(\d+(?:\.\d+)?)",
            "policy_relevance": r"policy relevance:?\s*(\d+(?:\.\d+)?)",
            "clarity_structure": r"clarity.{0,10}structure:?\s*(\d+(?:\.\d+)?)",
            "actionability": r"actionability:?\s*(\d+(?:\.\d+)?)",
            "overall": r"overall.{0,15}score:?\s*(\d+(?:\.\d+)?)"
        }
        
        # Search for each pattern in the evaluation text
        for metric, pattern in patterns.items():
            matches = re.finditer(pattern, evaluation.lower())
            for match in matches:
                try:
                    score = float(match.group(1))
                    scores[metric] = score
                    break  # Take the first match for each pattern
                except (ValueError, IndexError):
                    continue
        
        # If overall score wasn't found but we have other scores, calculate average
        if "overall" not in scores and len(scores) > 0:
            scores["overall"] = sum(scores.values()) / len(scores)
        
        # Ensure all scores are within 0-10 range
        for metric in scores:
            scores[metric] = min(max(scores[metric], 0), 10)
        
        return scores
    
    def get_comparison_ranking(self, question: str, responses: Dict[str, str], context: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare and rank multiple RAG responses.
        
        Args:
            question (str): The original question
            responses (Dict[str, str]): Dictionary mapping variant names to responses
            context (Optional[str]): The context used for generation
            
        Returns:
            Dict[str, Any]: Comparison results with rankings
        """
        # If context is not provided, use the baseline response as a proxy
        if context is None and "Baseline RAG" in responses:
            context = responses["Baseline RAG"]
        elif context is None:
            # Get context from the API if no baseline is available
            context = self.client.query(question)
        
        # Format the responses for comparison
        formatted_responses = ""
        for i, (variant, response) in enumerate(responses.items(), 1):
            formatted_responses += f"Response {i} ({variant}):\n{response}\n\n"
        
        # Create a prompt for comparative evaluation
        system_prompt = (
            "You are an expert evaluator of security policy information systems. "
            "Your task is to compare multiple responses to the same security policy question "
            "and rank them from best to worst.\n\n"
            
            "Compare the responses based on these criteria:\n"
            "1. Accuracy: How well the response reflects the information in the context\n"
            "2. Completeness: How thoroughly the response addresses all aspects of the question\n"
            "3. Clarity: How clear and well-structured the response is\n"
            "4. Policy Citations: How well the response references specific policy details\n"
            "5. Usefulness: How helpful the response would be to an employee\n\n"
            
            "Provide your comparison in this format:\n"
            "- Brief analysis of each response (2-3 sentences per response)\n"
            "- Comparative strengths and weaknesses\n"
            "- Final ranking from best to worst with brief justification\n\n"
            
            "End with a clear numerical ranking in the format: 'FINAL RANKING: Response X (#1), Response Y (#2), ...'"
        )
        
        user_prompt = (
            f"Security Policy Question: {question}\n\n"
            f"Context Information: {context}\n\n"
            f"{formatted_responses}\n"
            f"Please compare and rank these responses."
        )
        
        # Get the comparative evaluation
        comparison = self.client.get_openai_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=1000
        )
        
        # Extract the ranking
        ranking = self._extract_ranking(comparison, list(responses.keys()))
        
        return {
            "detailed_comparison": comparison,
            "ranking": ranking
        }
    
    def _extract_ranking(self, comparison: str, variant_names: List[str]) -> List[str]:
        """
        Extract the ranking of variants from the comparison text.
        
        Args:
            comparison (str): The comparison text
            variant_names (List[str]): List of variant names to look for
            
        Returns:
            List[str]: Ordered list of variant names from best to worst
        """
        # Look for the final ranking section
        ranking_section = ""
        if "FINAL RANKING:" in comparison:
            ranking_section = comparison.split("FINAL RANKING:")[1].strip()
        elif "final ranking:" in comparison.lower():
            ranking_section = comparison.split("final ranking:")[1].strip()
        elif "Ranking:" in comparison:
            ranking_section = comparison.split("Ranking:")[1].strip()
        
        # If we found a ranking section, try to extract the order
        if ranking_section:
            # Create a list to hold the ordered variants
            ordered_variants = []
            
            # Try to match variants based on their names or response numbers
            for variant in variant_names:
                # Check if the variant name is mentioned directly
                if variant in ranking_section:
                    ordered_variants.append(variant)
                    continue
                
                # Check for response numbers (Response 1, Response 2, etc.)
                variant_index = variant_names.index(variant) + 1
                response_patterns = [
                    f"Response {variant_index}",
                    f"response {variant_index}",
                    f"#{variant_index}",
                    f"({variant_index})",
                    f"[{variant_index}]"
                ]
                
                for pattern in response_patterns:
                    if pattern in ranking_section:
                        ordered_variants.append(variant)
                        break
            
            # If we found all variants, return them
            if len(ordered_variants) == len(variant_names):
                return ordered_variants
        
        # If we couldn't extract a clean ranking, return the original list
        # This is better than returning nothing
        logger.warning("Could not extract clean ranking from comparison, returning original order")
        return variant_names