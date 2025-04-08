"""
Role-based prompting technique with self-verification for RAG.
"""
import logging

from rag_client import RAGClient
from rag_variants.base_variant import RAGVariant

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RoleBasedRAG(RAGVariant):
    """RAG with role-based instructions and self-verification."""
    
    def __init__(self, rag_client: RAGClient):
        """
        Initialize the Role-Based RAG variant.
        
        Args:
            rag_client (RAGClient): Client for interacting with the baseline RAG API
        """
        super().__init__("Role-Based + Self-Verification RAG", rag_client)
    
    def generate_role_based_response(self, question: str, baseline_response: str) -> str:
        """
        Generate a response using role-based prompting.
        
        Args:
            question (str): The original question
            baseline_response (str): The response from the baseline RAG system
            
        Returns:
            str: The role-based response
        """
        # Create a system prompt with a strong role definition and instructions
        system_prompt = (
            "You are Sarah Chen, the Chief Information Security Officer at NovaTech Dynamics. "
            "With over 15 years of experience in cybersecurity and as the author of the company's "
            "security policy (Document ID: NTD-SEC-001-2025), you are the definitive authority on "
            "all security matters at the company.\n\n"
            
            "When answering questions about the security policy:\n"
            "1. Speak with the authority and precision expected of a CISO\n"
            "2. Reference specific sections and requirements from the security policy\n"
            "3. Provide practical implementation details where appropriate\n"
            "4. Explain the rationale behind security requirements\n"
            "5. Address compliance considerations (ISO 27001, SOC 2, GDPR, etc.)\n"
            "6. Use technical security terminology accurately\n\n"
            
            "As CISO, you are committed to ensuring that all employees understand security "
            "policies clearly, so make your explanations thorough yet accessible."
        )
        
        # Create a user prompt with the question and baseline response as context
        user_prompt = (
            f"One of your employees has asked the following question about NovaTech's security policy:\n\n"
            f"Question: {question}\n\n"
            f"Based on the security policy document, this information is relevant:\n{baseline_response}\n\n"
            f"Please respond to this employee's question as Sarah Chen, CISO of NovaTech Dynamics."
        )
        
        # Get the role-based response
        role_based_response = self.client.get_openai_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.4,
            max_tokens=800
        )
        
        return role_based_response
    
    def verify_response(self, question: str, role_based_response: str, baseline_response: str) -> str:
        """
        Verify and refine the role-based response.
        
        Args:
            question (str): The original question
            role_based_response (str): The initial role-based response
            baseline_response (str): The baseline response for reference
            
        Returns:
            str: The verified response
        """
        # Create a system prompt for self-verification
        system_prompt = (
            "You are a security policy verification expert tasked with ensuring that responses about "
            "security policies are accurate, complete, and properly sourced.\n\n"
            
            "Your verification process has four steps:\n"
            "1. Accuracy Check: Verify that all policy details mentioned are accurate and consistent with the source\n"
            "2. Completeness Check: Ensure all aspects of the question are addressed\n"
            "3. Source Validation: Confirm all claims are properly attributed to the security policy\n"
            "4. Correction: Fix any inaccuracies or omissions while maintaining the original tone and style\n\n"
            
            "After your verification, provide the final, corrected version of the response. "
            "If no corrections are needed, return the original response."
        )
        
        # Create a user prompt with all the necessary context
        user_prompt = (
            f"Question about security policy: {question}\n\n"
            f"Original source information: {baseline_response}\n\n"
            f"Response to verify: {role_based_response}\n\n"
            f"Please verify this response for accuracy, completeness, and proper sourcing. "
            f"Then provide the final, corrected version."
        )
        
        # Get the verified response
        verified_response = self.client.get_openai_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=800
        )
        
        return verified_response
    
    def query(self, question: str) -> str:
        """
        Process a query using role-based prompting with self-verification.
        
        Args:
            question (str): The question to answer
            
        Returns:
            str: The enhanced response
        """
        try:
            logger.info(f"Processing query with {self.name}: {question}")
            
            # First, get the baseline response to use as context
            baseline_response = self.client.query(question)
            
            # Generate response using role-based prompting
            role_based_response = self.generate_role_based_response(question, baseline_response)
            
            # Verify and refine the response
            verified_response = self.verify_response(question, role_based_response, baseline_response)
            
            logger.info(f"Generated enhanced response using {self.name}")
            return verified_response
            
        except Exception as e:
            logger.error(f"Error processing query with {self.name}: {str(e)}")
            # Fall back to baseline if there's an error
            return self.client.query(question)