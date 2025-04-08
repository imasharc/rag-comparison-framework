"""
Adaptive chunking and self-query RAG implementation.
"""
import logging
from typing import List, Dict, Any, Optional

from rag_client import RAGClient
from rag_variants.base_variant import RAGVariant

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdaptiveChunkingRAG(RAGVariant):
    """RAG with adaptive chunking and self-query refinement."""
    
    def __init__(self, rag_client: RAGClient):
        """
        Initialize the adaptive chunking RAG variant.
        
        Args:
            rag_client (RAGClient): Client for interacting with the baseline RAG API
        """
        super().__init__("Adaptive Chunking + Self-Query", rag_client)
    
    def identify_topic_structure(self, question: str) -> Dict[str, Any]:
        """
        Identify the topic structure to guide adaptive chunking.
        
        Args:
            question (str): The original question
            
        Returns:
            Dict[str, Any]: A dictionary with topic structure information
        """
        system_prompt = (
            "Analyze this question about security policies to determine the optimal information structure for answering it. "
            "Identify the main topic, any subtopics, and the appropriate level of detail needed.\n\n"
            "Return your analysis in the following format:\n"
            "Main Topic: [The primary security policy area being asked about]\n"
            "Subtopics: [List of 2-3 related subtopics that should be addressed]\n"
            "Detail Level: [Brief/Moderate/Comprehensive]\n"
            "Document Sections: [Likely sections of a security policy document that would contain relevant information]\n"
            "Key Terms: [Important technical or policy terms related to this question]"
        )
        
        user_prompt = f"Security Policy Question: {question}"
        
        # Get structure analysis from the LLM
        response = self.client.get_openai_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=250
        )
        
        # Parse the response into a structured format
        analysis = {}
        
        for line in response.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                analysis[key.strip()] = value.strip()
        
        logger.info(f"Identified topic structure for question")
        return analysis
    
    def generate_self_queries(self, question: str, topic_structure: Dict[str, Any]) -> List[str]:
        """
        Generate refined self-queries based on topic structure.
        
        Args:
            question (str): The original question
            topic_structure (Dict[str, Any]): The topic structure analysis
            
        Returns:
            List[str]: A list of refined self-queries
        """
        # Create a prompt that uses the topic structure to generate better queries
        system_prompt = (
            "You are an expert at breaking down complex security policy questions into more specific queries. "
            "Based on the original question and topic analysis, generate 3 more specific queries that will "
            "help retrieve the most relevant information from a security policy document.\n\n"
            "The queries should:\n"
            "1. Target specific aspects of the main topic\n" 
            "2. Use precise security policy terminology\n"
            "3. Focus on retrieving concrete policy details, not general information\n"
            "4. Be directly answerable from a security policy document\n\n"
            "Return ONLY the 3 refined queries, one per line, without numbering or explanation."
        )
        
        # Format the topic structure for the prompt
        structure_text = "\n".join([f"{k}: {v}" for k, v in topic_structure.items()])
        
        user_prompt = (
            f"Original Question: {question}\n\n"
            f"Topic Analysis:\n{structure_text}\n\n"
            f"Generate 3 specific queries to find the most relevant information:"
        )
        
        # Get self-queries from the LLM
        response = self.client.get_openai_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=200
        )
        
        # Split into individual queries and include the original
        refined_queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
        refined_queries.append(question)  # Include the original query
        
        logger.info(f"Generated {len(refined_queries)} self-queries")
        return refined_queries
    
    def adaptive_chunk_synthesis(self, question: str, query_responses: List[str], topic_structure: Dict[str, Any]) -> str:
        """
        Synthesize information with adaptive chunking based on topic structure.
        
        Args:
            question (str): The original question
            query_responses (List[str]): Responses from self-queries
            topic_structure (Dict[str, Any]): The topic structure analysis
            
        Returns:
            str: The synthesized response
        """
        # Determine the appropriate synthesis approach based on the detail level
        detail_level = topic_structure.get("Detail Level", "Moderate").strip().lower()
        
        if "brief" in detail_level:
            synthesis_style = "Provide a concise summary focusing only on the most essential points."
        elif "comprehensive" in detail_level:
            synthesis_style = "Provide a detailed, comprehensive answer covering all aspects in depth."
        else:  # Moderate is the default
            synthesis_style = "Provide a balanced response with moderate detail on all relevant aspects."
        
        # Create a system prompt for adaptive synthesis
        system_prompt = (
            "You are a security policy expert tasked with synthesizing information from multiple sources "
            "to answer a question about security policies.\n\n"
            f"Based on the topic analysis, you should: {synthesis_style}\n\n"
            "Your synthesis should:\n"
            "1. Organize information according to the identified topic structure\n"
            "2. Emphasize the main topic while covering all relevant subtopics\n"
            "3. Include specific security policy details, requirements, and procedures\n"
            "4. Cite relevant sections of the security policy where appropriate\n"
            "5. Use consistent terminology from the security domain\n\n"
            "The final answer should be well-structured, authoritative, and directly address the original question."
        )
        
        # Format the query responses and topic structure for the prompt
        responses_text = ""
        for i, response in enumerate(query_responses):
            responses_text += f"Information Source {i+1}:\n{response}\n\n"
        
        structure_text = "\n".join([f"{k}: {v}" for k, v in topic_structure.items()])
        
        user_prompt = (
            f"Original Question: {question}\n\n"
            f"Topic Analysis:\n{structure_text}\n\n"
            f"{responses_text}\n"
            f"Please synthesize this information into a {detail_level.lower()} response to the original question."
        )
        
        # Get the synthesized response
        synthesized = self.client.get_openai_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.4,
            max_tokens=800
        )
        
        return synthesized
    
    def query(self, question: str) -> str:
        """
        Process a query using adaptive chunking and self-query refinement.
        
        Args:
            question (str): The question to answer
            
        Returns:
            str: The enhanced response
        """
        try:
            logger.info(f"Processing query with {self.name}: {question}")
            
            # Step 1: Identify the topic structure to guide chunking
            topic_structure = self.identify_topic_structure(question)
            
            # Step 2: Generate refined self-queries based on the topic structure
            refined_queries = self.generate_self_queries(question, topic_structure)
            
            # Step 3: Get responses for each refined query
            query_responses = []
            for query in refined_queries:
                response = self.client.query(query)
                query_responses.append(response)
            
            # Step 4: Synthesize with adaptive chunking based on topic structure
            final_response = self.adaptive_chunk_synthesis(question, query_responses, topic_structure)
            
            logger.info(f"Generated enhanced response using {self.name}")
            return final_response
            
        except Exception as e:
            logger.error(f"Error processing query with {self.name}: {str(e)}")
            # Fall back to baseline if there's an error
            return self.client.query(question)