"""
Few-Shot prompting technique for RAG.
"""
import logging

from rag_client import RAGClient
from rag_variants.base_variant import RAGVariant

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FewShotRAG(RAGVariant):
    """RAG with Few-Shot prompting examples."""
    
    def __init__(self, rag_client: RAGClient):
        """
        Initialize the Few-Shot RAG variant.
        
        Args:
            rag_client (RAGClient): Client for interacting with the baseline RAG API
        """
        super().__init__("Few-Shot RAG", rag_client)
        
        # Define high-quality few-shot examples
        self.few_shot_examples = [
            {
                "question": "What is NovaTech's password policy?",
                "answer": """
According to NovaTech's Security Policy (Section: Password Policy), passwords must meet the following requirements:

1. Passwords must be changed every 90 days through the automated password management system
2. Password history of 12 previous passwords is maintained to prevent reuse
3. All passwords must use industry-standard hashing algorithms (SHA-256 minimum)
4. Minimum requirements include:
   - 12 characters minimum length
   - At least one uppercase letter
   - At least one lowercase letter
   - At least one number
   - At least one special character
   - No dictionary words or common patterns

Additionally, all system administrators and employees with access to Level 1 data must use the company-provided LastPass Enterprise password manager.

These requirements are documented in the Security Policy document NTD-SEC-001-2025 (version 3.2), approved by CEO Michael Rodriguez and effective as of February 1, 2025.
                """
            },
            {
                "question": "What is the data classification system at NovaTech?",
                "answer": """
According to NovaTech's Security Policy, the company employs a three-tier data classification system designed to protect information based on sensitivity levels:

1. Level 1 - Restricted: Highest sensitivity data including proprietary algorithms and financial forecasts
2. Level 2 - Confidential: Medium sensitivity data including customer data and employee records
3. Level 3 - Internal: Low sensitivity data including general communications and non-sensitive documentation

The policy explicitly identifies the following as data subject to special protection (Level 1 & 2):
- Information about contracts with key clients (Westfield Healthcare, GlobalTech, Federal agencies)
- Financial information (quarterly figures, profit margins, investment strategies)
- Organizational information (restructuring plans, M&A strategies)
- Access data to cloud environments, database systems, and internal networks
- Personal data of employees, contractors, and clients (subject to GDPR and CCPA compliance)
- R&D information constituting competitive advantage (AI model parameters, proprietary algorithms)
- Any information explicitly marked as "confidential information" or "restricted data"

This classification system is documented in the Security Policy document NTD-SEC-001-2025 (version 3.2), which was last updated January 15, 2025.
                """
            },
            {
                "question": "What happens during the employee termination process?",
                "answer": """
According to NovaTech's Security Policy (Section: Employee Termination Process), when employment is terminated, the following security procedures are implemented:

1. HR initiates an offboarding workflow in ServiceNow
2. All access is revoked within 4 hours of termination notification
3. Equipment is collected and sanitized according to NIST standards
4. Digital access cards are deactivated
5. Password resets are forced on shared systems

This procedure ensures the protection of company data and systems by promptly removing access privileges for terminated employees. The policy applies to all termination scenarios and is part of NovaTech's comprehensive security framework.

These requirements are documented in the Security Policy document NTD-SEC-001-2025 (version 3.2), approved by CEO Michael Rodriguez and effective as of February 1, 2025.
                """
            }
        ]
    
    def generate_few_shot_prompt(self, question: str, baseline_response: str) -> str:
        """
        Generate a few-shot prompt with examples.
        
        Args:
            question (str): The original question
            baseline_response (str): The response from the baseline RAG system
            
        Returns:
            str: The enhanced response using few-shot prompting
        """
        # Create a system prompt with few-shot examples
        system_prompt = (
            "You are a security policy expert answering questions about company security policies. "
            "Below are some examples of high-quality answers to previous questions about security policies. "
            "Use these examples as a guide for how to structure and present your answer to the new question.\n\n"
        )
        
        # Add the few-shot examples
        for i, example in enumerate(self.few_shot_examples):
            system_prompt += f"Example Question {i+1}: {example['question']}\n\n"
            system_prompt += f"Example Answer {i+1}: {example['answer']}\n\n"
        
        system_prompt += (
            "Your answer should follow a similar format to these examples. Specifically:\n"
            "1. Begin by referencing the relevant section of the security policy\n"
            "2. Provide a comprehensive answer with specific details, not generalities\n"
            "3. List requirements, procedures, or policies in a structured format\n"
            "4. Include specific values (e.g., password length, time periods) where relevant\n"
            "5. Cite the policy document details (document ID, version, approval) at the end\n\n"
            "Use the provided context information to create a response that follows this structure."
        )
        
        # Create a user prompt with the question and baseline response as context
        user_prompt = (
            f"New Question: {question}\n\n"
            f"Context Information: {baseline_response}\n\n"
            f"Please provide a detailed answer following the structured format shown in the examples."
        )
        
        # Get the enhanced response using few-shot learning
        enhanced_response = self.client.get_openai_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.4,
            max_tokens=800
        )
        
        return enhanced_response
    
    def query(self, question: str) -> str:
        """
        Process a query using Few-Shot prompting.
        
        Args:
            question (str): The question to answer
            
        Returns:
            str: The enhanced response
        """
        try:
            logger.info(f"Processing query with {self.name}: {question}")
            
            # First, get the baseline response to use as context
            baseline_response = self.client.query(question)
            
            # Generate enhanced response using few-shot examples
            enhanced_response = self.generate_few_shot_prompt(question, baseline_response)
            
            logger.info(f"Generated enhanced response using {self.name}")
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Error processing query with {self.name}: {str(e)}")
            # Fall back to baseline if there's an error
            return self.client.query(question)