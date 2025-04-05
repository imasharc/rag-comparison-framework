"""
Query engine module for processing queries with RAG.
"""
from typing import List, Dict, Any, Optional, Union
import logging
from abc import ABC, abstractmethod

from langchain.docstore.document import Document

from rag.retriever import DocumentRetriever
from models.openai_service import OpenAIService

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QueryEngine(ABC):
    """Abstract base class for query engines."""
    
    @abstractmethod
    def query(self, query_text: str) -> str:
        """
        Process a query and return a response.
        
        Args:
            query_text (str): Query text
            
        Returns:
            str: Response text
        """
        pass


class RAGQueryEngine(QueryEngine):
    """Query engine that always uses general knowledge and enhances with RAG when relevant."""
    
    def __init__(self, 
                retriever: DocumentRetriever, 
                llm_service: Any,
                system_prompt_template: Optional[str] = None):
        """Initialize the enhanced RAG query engine."""
        self.retriever = retriever
        self.llm_service = llm_service
        
        # Default system prompt that encourages using general knowledge and document information
        self.system_prompt_template = system_prompt_template or (
            "You are a helpful assistant that provides informative answers about all topics, with special "
            "expertise on NovaTech Dynamics' security policies. Always give a helpful response based on your "
            "general knowledge. When information from NovaTech's security documents is relevant, incorporate "
            "and highlight that specific information to enhance your answer.\n\n"
            "If the provided document extracts contain relevant information, integrate it into your response "
            "and clearly indicate where you're referencing NovaTech's specific policies by using phrases like "
            "'According to NovaTech's security policy...' or 'NovaTech's documents specify that...'\n\n"
            "Document information (if relevant):\n{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        )
    
    def format_documents(self, documents: List[Document]) -> str:
        """
        Format documents for easier integration into responses.
        
        Args:
            documents (List[Document]): Documents to format
            
        Returns:
            str: Formatted context string
        """
        if not documents:
            return "No relevant documents found."
        
        formatted_docs = []
        for i, doc in enumerate(documents):
            # Extract key information and format clearly
            extract = f"Extract {i+1}:\n{doc.page_content.strip()}"
            formatted_docs.append(extract)
        
        return "\n\n".join(formatted_docs)
    
    def check_relevance(self, query_text: str, retrieved_docs: List[Document]) -> bool:
        """
        Check if retrieved documents are relevant to the query about NovaTech security policies.
        """
        try:
            # Format documents into context
            context = self.format_documents(retrieved_docs)
            
            # Create a much more explicit prompt with examples
            prompt = (
                "Your task is to determine if the retrieved documents contain information that directly answers "
                "a question about NovaTech Dynamics' security policies, procedures, or practices. "
                "\n\n"
                "Examples of RELEVANT queries (these would be relevant to security policy documents):\n"
                "- What is NovaTech's password policy?\n"
                "- How does NovaTech protect confidential data?\n"
                "- What happens during employee termination at NovaTech?\n"
                "- What are NovaTech's backup procedures?\n"
                "\n"
                "Examples of NOT_RELEVANT queries (these would NOT be relevant to security policy documents):\n"
                "- Where is Poland located?\n"
                "- What is the capital of France?\n"
                "- How do I cook pasta?\n"
                "- Who won the World Cup in 2022?\n"
                "- What is a transformer in machine learning?\n"
                "\n\n"
                f"Query: {query_text}\n\n"
                "Retrieved documents:\n{context}\n\n"
                "First, analyze whether the query is asking specifically about NovaTech Dynamics' security policies, "
                "procedures, or practices. Then check if the retrieved documents contain information that directly "
                "addresses this query.\n\n"
                "Respond with ONLY ONE WORD: either 'RELEVANT' or 'NOT_RELEVANT'."
            )
            
            # Use a more deterministic setting for relevance check
            response = self.llm_service.generate_text(
                system_prompt=prompt,
                user_prompt="Determine RELEVANT or NOT_RELEVANT based on the criteria above.",
                temperature=0.1,
                max_tokens=10
            )
            
            # Check if response indicates relevance
            is_relevant = "RELEVANT" in response.upper() and "NOT_RELEVANT" not in response.upper()
            logger.info(f"Relevance check result: {'Relevant' if is_relevant else 'Not relevant'}")
            
            return is_relevant
        except Exception as e:
            logger.error(f"Error checking relevance: {str(e)}")
            # Default to treating documents as not relevant in case of error
            return False

    def assess_document_relevance(self, query_text: str, retrieved_docs: List[Document]) -> tuple[bool, List[Document]]:
        """Assess if and how documents can enhance a response to the query."""
        if not retrieved_docs:
            return False, []
        
        # Format documents for analysis
        docs_text = self.format_documents(retrieved_docs)
        
        # Create a more structured assessment prompt with explicit examples
        analysis_prompt = (
            "Your task is to determine if the retrieved documents contain information that would meaningfully "
            "enhance a response about NovaTech Dynamics' security policies.\n\n"
            
            "SECURITY POLICY TOPICS (examples where documents WOULD enhance the response):\n"
            "- Password requirements and management\n"
            "- Data classification systems\n"
            "- Access control procedures\n"
            "- Security monitoring\n"
            "- Incident response\n"
            "- Employee termination processes\n"
            "- Backup procedures\n"
            "- Device security\n"
            "- Physical security measures\n"
            "- Security questions\n"
            "- Security roles and contacts\n\n"
            
            "NON-POLICY TOPICS (examples where documents would NOT enhance the response):\n"
            "- General geography or locations\n"
            "- General technology concepts (e.g., what are transformers)\n"
            "- Topics unrelated to security\n"
            "- General knowledge questions\n"
            "- Questions about other companies\n\n"
            
            f"Query: {query_text}\n\n"
            f"Document Extracts:\n{docs_text}\n\n"
            
            "First, determine if the query is asking about NovaTech's security policies or practices. "
            "Second, examine if the document extracts contain specific, relevant information that addresses the query. "
            "Third, decide if incorporating this document information would make the response more accurate and helpful.\n\n"
            
            "Respond with one of these options:\n"
            "ENHANCE - If the documents contain relevant information that would improve the response to this security policy question\n"
            "NO_ENHANCEMENT - If the documents don't contain relevant information or if the query isn't about security policies"
        )
        
        # Get analysis with lower temperature for consistency
        analysis = self.llm_service.generate_text(
            system_prompt=analysis_prompt,
            user_prompt="Analyze whether these documents would enhance a response about NovaTech's security policies.",
            temperature=0.2,
            max_tokens=50
        )
        
        # Check if documents would enhance the response (exact match for better reliability)
        enhance = "ENHANCE" in analysis.upper() and "NO_ENHANCEMENT" not in analysis.upper()
        logger.info(f"Document enhancement potential: {'Enhancing' if enhance else 'No enhancement'}")
        
        return enhance, retrieved_docs
    
    def query(self, query_text: str) -> str:
        """
        Process a query using general knowledge and enhance with document information when relevant.
        
        Args:
            query_text (str): Query text
            
        Returns:
            str: Response text
        """
        try:
            logger.info(f"Processing query: {query_text}")
            
            # Retrieve potential documents
            retrieved_docs = self.retriever.retrieve(query_text)
            
            # Assess document enhancement potential
            can_enhance, relevant_docs = self.assess_document_relevance(query_text, retrieved_docs)
            
            # Prepare context based on assessment
            if can_enhance and relevant_docs:
                context = self.format_documents(relevant_docs)
                logger.info(f"Enhancing response with {len(relevant_docs)} relevant documents")
            else:
                context = "No specific information about this topic was found in NovaTech's security policy documents."
                logger.info("Providing response without document enhancement")
            
            # Create response using the enhanced prompt
            prompt = self.system_prompt_template.format(
                context=context,
                question=query_text
            )
            
            # Generate the enhanced response
            response = self.llm_service.generate_text(
                system_prompt=prompt,
                user_prompt=query_text,
                temperature=0.7  # Allow some creativity in responses
            )
            
            logger.info("Generated response for the query")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            # Provide a response even if the RAG processing fails
            fallback_response = self.llm_service.generate_text(
                system_prompt="You are a helpful assistant. Please answer the following question based on your general knowledge.",
                user_prompt=query_text
            )
            return fallback_response


class QueryEngineFactory:
    """Factory for creating query engines."""
    
    @staticmethod
    def get_query_engine(engine_type: str = "rag", 
                        retriever: Optional[DocumentRetriever] = None,
                        llm_service: Optional[Any] = None,
                        **kwargs) -> QueryEngine:
        """
        Get a query engine based on type and parameters.
        
        Args:
            engine_type (str): Type of query engine to create
            retriever (Optional[DocumentRetriever]): Document retriever to use
            llm_service (Optional[Any]): LLM service to use
            **kwargs: Additional parameters for the query engine
            
        Returns:
            QueryEngine: An instance of a query engine
            
        Raises:
            ValueError: If the engine type is not supported or required parameters are missing
        """
        if engine_type == "rag":
            if retriever is None or llm_service is None:
                raise ValueError("Retriever and LLM service are required for RAG query engine")
            return RAGQueryEngine(retriever, llm_service, **kwargs)
        else:
            logger.error(f"Unsupported query engine type: {engine_type}")
            raise ValueError(f"Unsupported query engine type: {engine_type}")


def get_query_engine(engine_type: str = "rag", 
                    retriever: Optional[DocumentRetriever] = None,
                    llm_service: Optional[Any] = None,
                    **kwargs) -> QueryEngine:
    """
    Get a query engine.
    
    Args:
        engine_type (str): Type of query engine to create
        retriever (Optional[DocumentRetriever]): Document retriever to use
        llm_service (Optional[Any]): LLM service to use
        **kwargs: Additional parameters for the query engine
        
    Returns:
        QueryEngine: An instance of a query engine
        
    Raises:
        ValueError: If the query engine creation fails
    """
    try:
        return QueryEngineFactory.get_query_engine(
            engine_type=engine_type,
            retriever=retriever,
            llm_service=llm_service,
            **kwargs
        )
    except ValueError as e:
        logger.error(f"Error in get_query_engine: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_query_engine: {str(e)}")
        raise ValueError(f"Failed to get query engine: {str(e)}")