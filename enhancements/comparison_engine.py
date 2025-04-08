"""
Engine for comparing different RAG approaches.
"""
import logging
import pandas as pd
import os
import time
from typing import List, Dict, Any, Optional, Tuple
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
    
    def query_all_variants(self, question: str) -> Dict[str, str]:
        """
        Query with all variants.
        
        Args:
            question (str): The question to answer
            
        Returns:
            Dict[str, str]: A dictionary mapping variant names to responses
        """
        results = {}
        
        for variant in self.variants:
            try:
                name = variant.get_name()
                logger.info(f"Generating response from {name}")
                response = variant.query(question)
                results[name] = response
                logger.info(f"Generated response from {name}")
                
                # Add a small delay to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error with variant {variant.get_name()}: {str(e)}")
                results[variant.get_name()] = f"Error: {str(e)}"
        
        return results
    
    def evaluate_response(self, question: str, response: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a response.
        
        Args:
            question (str): The original question
            response (str): The generated response
            context (Optional[str]): The context used for generation
            
        Returns:
            Dict[str, Any]: A dictionary with evaluation results
        """
        # Get evaluation metrics
        metrics = self.evaluator.evaluate_all_metrics(question, response, context)
        
        # Get discriminator evaluation
        discriminator = self.discriminator.evaluate(question, response, context)
        
        # Combine results
        return {
            "metrics": metrics,
            "discriminator": discriminator
        }
    
    def evaluate_all_variants(self, question: str) -> pd.DataFrame:
        """
        Evaluate all variants for a question.
        
        Args:
            question (str): The question to answer
            
        Returns:
            pd.DataFrame: A dataframe with evaluation results
        """
        results = []
        
        # Get responses from all variants
        responses = self.query_all_variants(question)
        
        # Get baseline response for context
        baseline_response = responses.get(self.baseline.get_name(), "")
        
        # Evaluate each response
        for variant_name, response in responses.items():
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
        return df
    
    def run_benchmark(self, questions: List[str]) -> pd.DataFrame:
        """
        Run a benchmark on a list of questions.
        
        Args:
            questions (List[str]): The questions to benchmark
            
        Returns:
            pd.DataFrame: A dataframe with benchmark results
        """
        all_results = []
        total_questions = len(questions)
        
        for i, question in enumerate(questions, 1):
            logger.info(f"Processing question {i}/{total_questions}: {question}")
            
            # Evaluate all variants for this question
            results = self.evaluate_all_variants(question)
            all_results.append(results)
            
            # Save intermediate results
            intermediate_df = pd.concat(all_results, ignore_index=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            intermediate_df.to_csv(f"results/benchmark_intermediate_{timestamp}.csv", index=False)
            
            logger.info(f"Completed question {i}/{total_questions}")
        
        # Combine all results
        benchmark_df = pd.concat(all_results, ignore_index=True)
        
        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        benchmark_df.to_csv(f"results/benchmark_final_{timestamp}.csv", index=False)
        
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
            report += f"### {variant_name}\n\n{response}\n\n"
        
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