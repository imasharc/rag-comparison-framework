"""
Chain-of-Thought prompting technique for RAG.
"""
import logging

from rag_client import RAGClient
from rag_variants.base_variant import RAGVariant

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChainOfThoughtRAG(RAGVariant):
    """RAG with Chain-of-Thought prompting."""
    
    def __init__(self, rag_client: RAGClient):
        """
        Initialize the Chain-of-Thought RAG variant.
        
        Args:
            rag_client (RAGClient): Client for interacting with the baseline RAG API
        """
        super().__init__("Chain-of-Thought RAG", rag_client)
    
    def query(self, question: str) -> str:
        """
        Process a query using Chain-of-Thought prompting.
        
        Args:
            question (str): The question to answer
            
        Returns:
            str: The enhanced response
        """
        try:
            logger.info(f"Processing query with {self.name}: {question}")
            
            # First, get the baseline response to extract context information
            baseline_response = self.client.query(question)
            
            # Now create a Chain-of-Thought prompt that enhances the reasoning
            system_prompt = (
                "You are a security policy expert answering questions about company security policies. "
                "Use a structured chain-of-thought approach to analyze and answer this security policy question:\n\n"
                "Step 1: Identify the specific security policy domain(s) this question relates to.\n"
                "Step 2: Recall the relevant policy details, requirements, and guidelines.\n"
                "Step 3: Consider any exceptions, special cases, or related policies that might apply.\n"
                "Step 4: Analyze how these policies would apply in the context of the question.\n"
                "Step 5: Formulate a comprehensive answer that directly addresses the question.\n\n"
                "For each step, think through your reasoning explicitly before moving to the next step. "
                "Your final answer should cite specific sections of the security policy where relevant and "
                "make it clear which parts are direct policy requirements versus general best practices."
            )
            
            # Include the baseline response as context
            user_prompt = (
                f"Question about security policy: {question}\n\n"
                f"Relevant policy information: {baseline_response}\n\n"
                f"Please answer this question using the step-by-step approach described above. "
                f"For each step, start with 'Step X:' and explain your reasoning."
            )
            
            # Get the enhanced chain-of-thought response
            cot_response = self.client.get_openai_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=800
            )
            
            # Now extract just the final answer for a cleaner response
            final_answer_prompt = (
                "Given your detailed chain-of-thought analysis below, extract just the final answer "
                "to the original question. The answer should be comprehensive and well-structured, "
                "but without the explicit step-by-step reasoning. Keep all policy citations and important details."
            )
            
            final_answer = self.client.get_openai_completion(
                system_prompt=final_answer_prompt,
                user_prompt=cot_response,
                temperature=0.3,
                max_tokens=500
            )
            
            logger.info(f"Generated enhanced response using {self.name}")
            return final_answer
            
        except Exception as e:
            logger.error(f"Error processing query with {self.name}: {str(e)}")
            # Fall back to baseline if there's an error
            return self.client.query(question)