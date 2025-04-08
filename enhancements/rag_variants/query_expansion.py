"""
Query expansion and reranking RAG implementation.
"""
import logging
from typing import List

from rag_client import RAGClient
from rag_variants.base_variant import RAGVariant

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryExpansionRAG(RAGVariant):
    """RAG with query expansion and reranking."""
    
    def __init__(self, rag_client: RAGClient):
        """
        Initialize the query expansion RAG variant.
        
        Args:
            rag_client (RAGClient): Client for interacting with the baseline RAG API
        """
        super().__init__("Query Expansion + Reranking", rag_client)
    
    def expand_query(self, question: str) -> List[str]:
        """
        Expand a query into multiple related queries.
        
        Args:
            question (str): The original question
            
        Returns:
            List[str]: A list of expanded queries
        """
        system_prompt = (
            "Generate 3 different versions of the following query about security policies. "
            "Each version should rephrase the question to improve document retrieval by using "
            "different terminology, specific security policy terms, or focusing on different aspects "
            "of the question.\n\n"
            "Return ONLY the rewritten queries, one per line, without numbering or explanation."
        )
        
        user_prompt = f"Original query: {question}"
        
        # Get expanded queries from the LLM
        response = self.client.get_openai_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=150
        )
        
        # Split into individual queries and add the original
        expanded = [q.strip() for q in response.strip().split("\n") if q.strip()]
        expanded.append(question)  # Include the original query
        
        logger.info(f"Expanded '{question}' into {len(expanded)} queries")
        return expanded
    
    def rerank_results(self, question: str, results: List[str]) -> str:
        """
        Rerank and synthesize results from multiple queries.
        
        Args:
            question (str): The original question
            results (List[str]): The results from each expanded query
            
        Returns:
            str: A synthesized response
        """
        # Create a prompt that combines all results and asks for a synthesized answer
        system_prompt = (
            "You are a security policy expert answering questions about company security policies. "
            "Below are several answers to variations of the same question about security policies. "
            "\n\n"
            "Your task is to:\n"
            "1. Analyze all answers to identify the most accurate and relevant information\n"
            "2. Rerank the information from most to least relevant to the original question\n"
            "3. Synthesize a single comprehensive response that addresses all aspects of the question\n"
            "4. Ensure you cite specific security policy details\n"
            "5. Structure your answer clearly with the most important information first\n"
            "\n"
            "Your synthesized answer should be more thorough and accurate than any individual answer."
        )
        
        # Format the results into the user prompt
        user_prompt = f"Original question: {question}\n\n"
        for i, result in enumerate(results):
            user_prompt += f"Answer variation {i+1}:\n{result}\n\n"
        user_prompt += "Please synthesize these into a single comprehensive answer to the original question."
        
        # Get the synthesized response
        synthesized = self.client.get_openai_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.5,
            max_tokens=800
        )
        
        return synthesized
    
    def query(self, question: str) -> str:
        """
        Process a query using query expansion and reranking.
        
        Args:
            question (str): The question to answer
            
        Returns:
            str: The enhanced response
        """
        try:
            logger.info(f"Processing query with {self.name}: {question}")
            
            # Expand the query into multiple variations
            expanded_queries = self.expand_query(question)
            
            # Get results for each expanded query
            results = []
            for query in expanded_queries:
                result = self.client.query(query)
                results.append(result)
            
            # Rerank and synthesize the results
            final_response = self.rerank_results(question, results)
            
            logger.info(f"Generated enhanced response using {self.name}")
            return final_response
            
        except Exception as e:
            logger.error(f"Error processing query with {self.name}: {str(e)}")
            # Fall back to baseline if there's an error
            return self.client.query(question)