"""
Engine for comparing different RAG approaches.
"""
import logging
import pandas as pd
import os
import time
from typing import List, Dict, Any, Optional, Tuple, Callable
from datetime import datetime

from rag_client import RAGClient
from rag_variants.base_variant import BaselineRAG
from rag_variants.query_expansion import QueryExpansionRAG
from rag_variants.hybrid_search import HybridSearchRAG
from rag_variants.adaptive_chunking import AdaptiveChunkingRAG
from rag_variants.prompting.chain_of_thought import ChainOfThoughtRAG
from rag_variants.prompting.few_shot import FewShotRAG
from rag_variants.prompting.role_based import RoleBasedRAG
from evaluation.metrics import RAGEvaluator
from evaluation.discriminator import LLMDiscriminator

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGComparisonEngine:
    """Engine for comparing different RAG approaches."""
    
    # Descriptions of each RAG variant
    VARIANT_DESCRIPTIONS = {
        "Baseline RAG": "The standard RAG implementation that retrieves relevant document chunks and generates responses using the retrieved context.",
        
        "Query Expansion + Reranking": "Expands the original query into multiple related queries to improve retrieval coverage, then reranks the combined results based on relevance to the original question.",
        
        "Hybrid Search + Contextual Compression": "Combines semantic (vector-based) search with keyword search to find more relevant documents, then compresses the retrieved context to focus on the most important information.",
        
        "Adaptive Chunking + Self-Query": "Dynamically adjusts document chunking based on content semantics and uses the model to generate refined versions of the query that target specific aspects of the question.",
        
        "Chain-of-Thought RAG": "Guides the language model through an explicit step-by-step reasoning process, encouraging it to carefully consider each aspect of the security policy before generating a final answer.",
        
        "Few-Shot RAG": "Provides the model with carefully selected examples of high-quality question-answer pairs about security policies, helping it learn the expected format and level of detail.",
        
        "Role-Based + Self-Verification RAG": "Assigns the model a specific expert role (security officer) and includes a verification step where the model checks its own response for accuracy and completeness."
    }
    
    def __init__(self, base_url: str = "http://127.0.0.1:5000/api"):
        """
        Initialize the comparison engine.
        
        Args:
            base_url (str): Base URL of the RAG API
        """
        # Create the RAG client
        self.client = RAGClient(base_url)
        
        # Initialize the baseline RAG
        self.baseline = BaselineRAG(self.client)
        
        # Initialize all RAG variants
        self.variants = [
            self.baseline,                   # Include the baseline for comparison
            QueryExpansionRAG(self.client),  # Module Pair 1
            HybridSearchRAG(self.client),    # Module Pair 2
            AdaptiveChunkingRAG(self.client),# Module Pair 3
            ChainOfThoughtRAG(self.client),  # Prompt Technique 1
            FewShotRAG(self.client),         # Prompt Technique 2
            RoleBasedRAG(self.client)        # Prompt Technique 3
        ]
        
        # Initialize the evaluator and discriminator
        self.evaluator = RAGEvaluator(self.client)
        self.discriminator = LLMDiscriminator(self.client)
        
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        logger.info(f"Initialized RAG comparison engine with {len(self.variants)} variants")
    
    def get_variant_names(self) -> List[str]:
        """
        Get the names of all variants.
        
        Returns:
            List[str]: List of variant names
        """
        return [variant.get_name() for variant in self.variants]
    
    def get_variant_description(self, variant_name: str) -> str:
        """
        Get the description of a specific RAG variant.
        
        Args:
            variant_name (str): The name of the variant
            
        Returns:
            str: Description of the variant
        """
        return self.VARIANT_DESCRIPTIONS.get(variant_name, "No description available.")
    
    def get_all_variant_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions for all RAG variants.
        
        Returns:
            Dict[str, str]: Dictionary mapping variant names to descriptions
        """
        return {variant.get_name(): self.get_variant_description(variant.get_name()) 
                for variant in self.variants}
    
    def query_with_variant(self, question: str, variant_name: str) -> str:
        """
        Query with a specific variant.
        
        Args:
            question (str): The question to answer
            variant_name (str): The name of the variant to use
            
        Returns:
            str: The response from the variant
        """
        # Find the variant by name
        for variant in self.variants:
            if variant.get_name() == variant_name:
                return variant.query(question)
        
        # Fallback to baseline if variant not found
        logger.warning(f"Variant '{variant_name}' not found, using baseline")
        return self.baseline.query(question)
    
    def query_all_variants(self, question: str, progress_callback: Optional[Callable[[str, float], None]] = None) -> Dict[str, str]:
        """
        Query with all variants.
        
        Args:
            question (str): The question to answer
            progress_callback: Optional callback function to report progress
            
        Returns:
            Dict[str, str]: A dictionary mapping variant names to responses
        """
        results = {}
        total_variants = len(self.variants)
        
        for i, variant in enumerate(self.variants):
            try:
                name = variant.get_name()
                if progress_callback:
                    progress_callback(f"Generating response using {name}", i / total_variants)
                
                logger.info(f"Generating response from {name}")
                response = variant.query(question)
                results[name] = response
                logger.info(f"Generated response from {name}")
                
                # Add a small delay to avoid rate limiting
                time.sleep(1)
                
                if progress_callback:
                    progress_callback(f"Completed {name}", (i + 1) / total_variants)
                
            except Exception as e:
                logger.error(f"Error with variant {variant.get_name()}: {str(e)}")
                results[variant.get_name()] = f"Error: {str(e)}"
        
        if progress_callback:
            progress_callback("All variants completed", 1.0)
            
        return results
    
    def evaluate_response(self, question: str, response: str, context: Optional[str] = None,
                         progress_callback: Optional[Callable[[str, float], None]] = None) -> Dict[str, Any]:
        """
        Evaluate a response.
        
        Args:
            question (str): The original question
            response (str): The generated response
            context (Optional[str]): The context used for generation
            progress_callback: Optional callback function to report progress
            
        Returns:
            Dict[str, Any]: A dictionary with evaluation results
        """
        if progress_callback:
            progress_callback("Starting evaluation metrics", 0.0)
            
        # Get evaluation metrics
        metrics = self.evaluator.evaluate_all_metrics(question, response, context)
        
        if progress_callback:
            progress_callback("Computing discriminator evaluation", 0.5)
            
        # Get discriminator evaluation
        discriminator = self.discriminator.evaluate(question, response, context)
        
        if progress_callback:
            progress_callback("Evaluation complete", 1.0)
            
        # Combine results
        return {
            "metrics": metrics,
            "discriminator": discriminator
        }
    
    def evaluate_all_variants(self, question: str, 
                             progress_callback: Optional[Callable[[str, float], None]] = None) -> pd.DataFrame:
        """
        Evaluate all variants for a question.
        
        Args:
            question (str): The question to answer
            progress_callback: Optional callback function to report progress
            
        Returns:
            pd.DataFrame: A dataframe with evaluation results
        """
        results = []
        
        # Report initial progress
        if progress_callback:
            progress_callback("Starting variant query process", 0.0)
        
        # Get responses from all variants
        responses = self.query_all_variants(question, 
                                          lambda msg, prog: progress_callback(msg, prog * 0.4) if progress_callback else None)
        
        # Get baseline response for context
        baseline_response = responses.get(self.baseline.get_name(), "")
        
        # Evaluate each response
        total_variants = len(responses)
        for i, (variant_name, response) in enumerate(responses.items()):
            if progress_callback:
                progress_callback(f"Evaluating {variant_name} ({i+1}/{total_variants})", 
                                0.4 + (i / total_variants) * 0.6)
            
            logger.info(f"Evaluating response from {variant_name}")
            evaluation = self.evaluate_response(question, response, baseline_response)
            
            # Create a row with variant name, response, and metrics
            row = {
                "question": question,
                "variant": variant_name,
                "response": response
            }
            
            # Add metrics
            for metric, score in evaluation["metrics"].items():
                row[f"metric_{metric}"] = score
            
            # Add discriminator scores
            if "raw_metrics" in evaluation["discriminator"]:
                for metric, score in evaluation["discriminator"]["raw_metrics"].items():
                    row[f"discriminator_{metric}"] = score
            
            # Add overall discriminator score
            row["discriminator_overall"] = evaluation["discriminator"].get("overall_score", 0.0)
            
            # Add detailed evaluation text
            row["evaluation_details"] = evaluation["discriminator"].get("detailed_evaluation", "")
            
            results.append(row)
            
            # Add a small delay to avoid rate limiting
            time.sleep(1)
        
        # Create a dataframe
        df = pd.DataFrame(results)
        
        if progress_callback:
            progress_callback("Evaluation complete", 1.0)
            
        return df
    
    def run_benchmark(self, questions: List[str], 
                     progress_callback: Optional[Callable[[str, float, Dict[str, Any]], None]] = None) -> pd.DataFrame:
        """
        Run a benchmark on a list of questions.
        
        Args:
            questions (List[str]): The questions to benchmark
            progress_callback: Optional callback function to report progress and additional data
            
        Returns:
            pd.DataFrame: A dataframe with benchmark results
        """
        all_results = []
        total_questions = len(questions)
        
        for i, question in enumerate(questions, 1):
            question_progress = (i - 1) / total_questions
            logger.info(f"Processing question {i}/{total_questions}: {question}")
            
            if progress_callback:
                progress_callback(
                    f"Processing question {i}/{total_questions}",
                    question_progress,
                    {"current_question": question, "question_num": i, "total_questions": total_questions}
                )
            
            # Evaluate all variants for this question
            results = self.evaluate_all_variants(
                question,
                lambda msg, prog: progress_callback(
                    msg, 
                    question_progress + prog / total_questions,
                    {"current_question": question, "question_num": i, "total_questions": total_questions}
                ) if progress_callback else None
            )
            
            all_results.append(results)
            
            # Save intermediate results
            intermediate_df = pd.concat(all_results, ignore_index=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            intermediate_df.to_csv(f"results/benchmark_intermediate_{timestamp}.csv", index=False)
            
            logger.info(f"Completed question {i}/{total_questions}")
            
            if progress_callback:
                progress_callback(
                    f"Completed question {i}/{total_questions}",
                    i / total_questions,
                    {"current_question": question, "question_num": i, "total_questions": total_questions, 
                     "completed": True}
                )
        
        # Combine all results
        benchmark_df = pd.concat(all_results, ignore_index=True)
        
        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        benchmark_df.to_csv(f"results/benchmark_final_{timestamp}.csv", index=False)
        
        if progress_callback:
            progress_callback("Benchmark complete", 1.0, {"completed": True})
            
        return benchmark_df
    
    def run_discriminator_comparison(self, question: str, responses: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Run a discriminator comparison of all variants for a question.
        
        Args:
            question (str): The question to answer
            responses (Optional[Dict[str, str]]): Pre-generated responses (or None to generate new ones)
            
        Returns:
            Dict[str, Any]: Comparison results
        """
        # Generate responses if not provided
        if responses is None:
            responses = self.query_all_variants(question)
        
        # Get baseline response for context
        baseline_response = responses.get(self.baseline.get_name(), "")
        
        # Run the discriminator comparison
        comparison = self.discriminator.get_comparison_ranking(question, responses, baseline_response)
        
        return comparison
    
    def generate_comparison_report(self, question: str, responses: Optional[Dict[str, str]] = None) -> str:
        """
        Generate a detailed comparison report for a question.
        
        Args:
            question (str): The question to answer
            responses (Optional[Dict[str, str]]): Pre-generated responses (or None to generate new ones)
            
        Returns:
            str: A formatted comparison report
        """
        # Generate responses if not provided
        if responses is None:
            responses = self.query_all_variants(question)
        
        # Create a dataframe for evaluation metrics
        results_df = pd.DataFrame()
        
        # Get baseline response for context
        baseline_response = responses.get(self.baseline.get_name(), "")
        
        # Evaluate each response
        for variant_name, response in responses.items():
            evaluation = self.evaluate_response(question, response, baseline_response)
            
            # Create a row with variant name and metrics
            row = {
                "Variant": variant_name,
            }
            
            # Add metrics
            for metric, score in evaluation["metrics"].items():
                row[metric.capitalize()] = score
            
            # Add to dataframe
            results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
        
        # Run discriminator comparison
        comparison = self.discriminator.get_comparison_ranking(question, responses, baseline_response)
        
        # Format the report
        report = (
            f"# Comparison Report for Question: '{question}'\n\n"
            f"## Responses\n\n"
        )
        
        # Add each response
        for variant_name, response in responses.items():
            # Include the variant description
            description = self.get_variant_description(variant_name)
            report += f"### {variant_name}\n\n**Description**: {description}\n\n{response}\n\n"
        
        # Add metrics table
        report += "## Evaluation Metrics\n\n"
        report += results_df.to_markdown(index=False) + "\n\n"
        
        # Add discriminator comparison
        report += "## Expert Comparison\n\n"
        report += comparison["detailed_comparison"] + "\n\n"
        
        # Add ranking
        report += "## Ranking\n\n"
        for i, variant in enumerate(comparison["ranking"], 1):
            report += f"{i}. {variant}\n"
        
        return report
    
    def save_comparison_report(self, question: str, responses: Optional[Dict[str, str]] = None) -> str:
        """
        Generate and save a comparison report for a question.
        
        Args:
            question (str): The question to answer
            responses (Optional[Dict[str, str]]): Pre-generated responses (or None to generate new ones)
            
        Returns:
            str: The path to the saved report
        """
        # Generate the report
        report = self.generate_comparison_report(question, responses)
        
        # Save the report
        os.makedirs("results/reports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"results/reports/comparison_{timestamp}.md"
        
        with open(report_path, "w") as f:
            f.write(report)
        
        return report_path