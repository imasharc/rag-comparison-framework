"""
Hybrid search and contextual compression RAG implementation.
"""
import logging
from typing import List, Dict, Any

from rag_client import RAGClient
from rag_variants.base_variant import RAGVariant

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridSearchRAG(RAGVariant):
    """RAG with hybrid search (semantic + keyword) and contextual compression."""
    
    def __init__(self, rag_client: RAGClient):
        """
        Initialize the hybrid search RAG variant.
        
        Args:
            rag_client (RAGClient): Client for interacting with the baseline RAG API
        """
        super().__init__("Hybrid Search + Contextual Compression", rag_client)
    
    def extract_keywords(self, question: str) -> List[str]:
        """
        Extract important keywords from the question for keyword search.
        
        Args:
            question (str): The original question
            
        Returns:
            List[str]: A list of extracted keywords
        """
        system_prompt = (
            "Extract the 3-5 most important keywords or phrases from this security policy question. "
            "Focus on technical terms, specific security concepts, policy elements, or named entities. "
            "Return ONLY the keywords, one per line, without numbering or explanation."
        )
        
        user_prompt = f"Question: {question}"
        
        # Get keywords from the LLM
        response = self.client.get_openai_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=100
        )
        
        # Split into individual keywords
        keywords = [k.strip() for k in response.strip().split("\n") if k.strip()]
        
        logger.info(f"Extracted keywords from question: {keywords}")
        return keywords
    
    def compress_context(self, question: str, original_response: str) -> str:
        """
        Compress and focus the context to be more relevant to the question.
        
        Args:
            question (str): The original question
            original_response (str): The response from the baseline RAG system
            
        Returns:
            str: The compressed context
        """
        system_prompt = (
            "You are an expert at contextual compression for security policy information. "
            "Given a question and a response that may contain some irrelevant information, "
            "your task is to:\n\n"
            "1. Identify the most relevant parts of the response that directly answer the question\n"
            "2. Remove any irrelevant sections, tangential information, or redundancies\n"
            "3. Preserve all factual details, specific policy references, and security requirements\n"
            "4. Maintain the accuracy of the information\n\n"
            "Return a compressed version that contains only the information needed to "
            "comprehensively answer the question."
        )
        
        user_prompt = (
            f"Question: {question}\n\n"
            f"Original Response: {original_response}\n\n"
            f"Compressed Context:"
        )
        
        # Get compressed context from the LLM
        compressed = self.client.get_openai_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=600
        )
        
        return compressed
    
    def enhance_with_keywords(self, question: str, compressed_context: str, keywords: List[str]) -> str:
        """
        Enhance the response with keyword search results.
        
        Args:
            question (str): The original question
            compressed_context (str): The compressed context from semantic search
            keywords (List[str]): The extracted keywords
            
        Returns:
            str: The enhanced response
        """
        # Create a system prompt that emphasizes using keywords to enhance the answer
        system_prompt = (
            "You are a security policy expert answering questions about security policies. "
            "Given a question, some context information, and important keywords, "
            "your task is to create a comprehensive answer that:\n\n"
            "1. Addresses the original question completely\n"
            "2. Uses the provided context information as the primary source\n"
            "3. Emphasizes information related to the identified keywords\n"
            "4. Organizes the response in a clear, structured format\n"
            "5. Cites specific security policy details and sections when relevant\n\n"
            "Focus particularly on the key security concepts identified in the keywords list, "
            "and ensure these aspects are thoroughly explained in your answer."
        )
        
        # Format the user prompt with question, context, and keywords
        user_prompt = (
            f"Question: {question}\n\n"
            f"Context Information: {compressed_context}\n\n"
            f"Important Keywords: {', '.join(keywords)}\n\n"
            f"Comprehensive Answer:"
        )
        
        # Get the enhanced response
        enhanced = self.client.get_openai_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.5,
            max_tokens=800
        )
        
        return enhanced
    
    def query(self, question: str) -> str:
        """
        Process a query using hybrid search and contextual compression.
        
        Args:
            question (str): The question to answer
            
        Returns:
            str: The enhanced response
        """
        try:
            logger.info(f"Processing query with {self.name}: {question}")
            
            # First, get a baseline response using semantic search
            baseline_response = self.client.query(question)
            
            # Extract important keywords for keyword-based search
            keywords = self.extract_keywords(question)
            
            # Compress the context to focus on the most relevant parts
            compressed_context = self.compress_context(question, baseline_response)
            
            # Enhance the response using the compressed context and keywords
            final_response = self.enhance_with_keywords(question, compressed_context, keywords)
            
            logger.info(f"Generated enhanced response using {self.name}")
            return final_response
            
        except Exception as e:
            logger.error(f"Error processing query with {self.name}: {str(e)}")
            # Fall back to baseline if there's an error
            return self.client.query(question)