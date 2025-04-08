"""
Streamlit application for comparing different RAG approaches.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys
import json
import time

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from comparison_engine import RAGComparisonEngine

# Configuration
API_URL = "http://127.0.0.1:5000/api"  # URL of the baseline RAG API

# Page configuration
st.set_page_config(
    page_title="RAG Comparison Tool",
    page_icon="🔍",
    layout="wide"
)

# Initialize comparison engine
@st.cache_resource
def get_comparison_engine():
    return RAGComparisonEngine(API_URL)

# Custom CSS for better styling
st.markdown("""
<style>
    .variant-header {
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .response-container {
        background-color: #ffffff;
        border-radius: 0.5rem;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .benchmark-header {
        font-size: 1.5rem;
        margin-bottom: 1rem;
        font-weight: bold;
        color: #0e1117;
    }
    .instructions {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize the comparison engine
try:
    engine = get_comparison_engine()
    variant_names = engine.get_variant_names()
except Exception as e:
    st.error(f"Error initializing comparison engine: {str(e)}")
    st.error("Make sure the baseline RAG API is running at " + API_URL)
    st.stop()

# Initialize session state
if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = None

if "last_query" not in st.session_state:
    st.session_state.last_query = None

if "last_results" not in st.session_state:
    st.session_state.last_results = None

if "comparison_report" not in st.session_state:
    st.session_state.comparison_report = None

# Page header
st.title("Enhanced RAG Comparison Tool")
st.caption("Compare different RAG approaches for security policy questions")

# Sidebar with information
with st.sidebar:
    st.header("About This Tool")
    st.markdown("""
    This tool compares different Retrieval-Augmented Generation (RAG) approaches for answering 
    questions about NovaTech's security policies.
    
    **Available RAG Variants:**
    """)
    
    # List all variants
    for variant in variant_names:
        st.markdown(f"- **{variant}**")
    
    st.markdown("""
    **Evaluation Metrics:**
    - **Faithfulness**: How well the answer aligns with the context
    - **Completeness**: Whether all aspects of the question are addressed
    - **Citation**: How well sources are referenced
    - **Coherence**: The logical flow and readability of the response
    - **Context Relevance**: How relevant the retrieved documents are to the query
    - **Answer Relevance**: How well the answer addresses the question
    
    The tool also uses an LLM discriminator to provide additional qualitative evaluation.
    """)
    
    # Benchmark section
    st.header("Benchmark")
    with st.form("benchmark_form"):
        default_questions = (
            "What is NovaTech's password policy?\n"
            "How does NovaTech protect confidential data?\n"
            "What happens during employee termination at NovaTech?"
        )
        
        benchmark_questions = st.text_area(
            "Enter benchmark questions (one per line):",
            value=default_questions,
            height=150,
            help="Add questions to run a full benchmark across all RAG variants"
        )
        
        run_full_benchmark = st.form_submit_button("Run Full Benchmark")
    
    if run_full_benchmark and benchmark_questions:
        questions = [q.strip() for q in benchmark_questions.split("\n") if q.strip()]
        if questions:
            # Show progress
            progress_container = st.empty()
            progress_bar = progress_container.progress(0)
            status_text = st.empty()
            
            status_text.text("Initializing benchmark...")
            
            try:
                # Run benchmark for all questions
                benchmark_df = engine.run_benchmark(questions)
                
                # Store in session state
                st.session_state.benchmark_results = benchmark_df
                
                # Complete progress
                progress_bar.progress(1.0)
                status_text.success("Benchmark completed!")
                
                # Provide download link
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv = benchmark_df.to_csv(index=False)
                st.download_button(
                    label="Download Results CSV",
                    data=csv,
                    file_name=f"rag_benchmark_{timestamp}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                status_text.error(f"Error running benchmark: {str(e)}")
                progress_container.empty()

# Main content area
tabs = st.tabs(["Interactive Comparison", "Benchmark Results"])

# Interactive Comparison Tab
with tabs[0]:
    st.header("Compare RAG Approaches")
    
    # Instructions
    st.markdown("""
    <div class="instructions">
    Enter a question about NovaTech's security policy to see how different RAG approaches handle it.
    For best results, try questions related to:
    <ul>
        <li>Password requirements</li>
        <li>Data classification</li>
        <li>Access control</li>
        <li>Employee termination</li>
        <li>Security monitoring</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Query input
    with st.form("query_form"):
        query = st.text_input("Enter your question about NovaTech's security policy:")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            submitted = st.form_submit_button("Compare Approaches")
        with col2:
            st.markdown("<div style='padding-top: 10px;'><small>This will process your query with all 7 RAG variants</small></div>", unsafe_allow_html=True)
    
    if submitted and query:
        # Store query in session state
        st.session_state.last_query = query
        
        # Show progress
        with st.spinner("Generating and evaluating responses..."):
            # Get responses from all variants
            responses = engine.query_all_variants(query)
            
            # Store results in session state
            st.session_state.last_results = responses
            
            # Generate comparison report
            st.session_state.comparison_report = engine.generate_comparison_report(query, responses)
    
    # Display results if available
    if st.session_state.last_query and st.session_state.last_results is not None:
        st.subheader(f"Results for: {st.session_state.last_query}")
        
        responses = st.session_state.last_results
        
        # Create tabs for comparing approaches
        variant_tabs = st.tabs(list(responses.keys()))
        
        # Set up radar chart data
        metrics_data = {
            "variant": [],
            "metric": [],
            "value": []
        }
        
        for i, (variant_name, tab) in enumerate(zip(responses.keys(), variant_tabs)):
            with tab:
                response = responses[variant_name]
                
                # Evaluate this response
                baseline_response = responses.get("Baseline RAG", "")
                evaluation = engine.evaluate_response(st.session_state.last_query, response, baseline_response)
                
                # Display metrics
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                
                # Get metrics for display and radar chart
                metric_cols = st.columns(3)
                
                # Core metrics
                with metric_cols[0]:
                    faithfulness = evaluation["metrics"].get("faithfulness", 0)
                    st.metric("Faithfulness", f"{faithfulness:.1f}/10")
                    metrics_data["variant"].append(variant_name)
                    metrics_data["metric"].append("Faithfulness")
                    metrics_data["value"].append(faithfulness)
                
                with metric_cols[1]:
                    completeness = evaluation["metrics"].get("completeness", 0)
                    st.metric("Completeness", f"{completeness:.1f}/10")
                    metrics_data["variant"].append(variant_name)
                    metrics_data["metric"].append("Completeness")
                    metrics_data["value"].append(completeness)
                
                with metric_cols[2]:
                    citation = evaluation["metrics"].get("citation", 0)
                    st.metric("Citation", f"{citation:.1f}/10")
                    metrics_data["variant"].append(variant_name)
                    metrics_data["metric"].append("Citation")
                    metrics_data["value"].append(citation)
                
                # Additional metrics
                metric_cols2 = st.columns(3)
                
                with metric_cols2[0]:
                    coherence = evaluation["metrics"].get("coherence", 0)
                    st.metric("Coherence", f"{coherence:.1f}/10")
                    metrics_data["variant"].append(variant_name)
                    metrics_data["metric"].append("Coherence")
                    metrics_data["value"].append(coherence)
                
                with metric_cols2[1]:
                    context_relevance = evaluation["metrics"].get("context_relevance", 0)
                    st.metric("Context Relevance", f"{context_relevance:.1f}/10")
                    metrics_data["variant"].append(variant_name)
                    metrics_data["metric"].append("Context Relevance")
                    metrics_data["value"].append(context_relevance)
                
                with metric_cols2[2]:
                    avg_score = evaluation["metrics"].get("average", 0)
                    st.metric("Average Score", f"{avg_score:.1f}/10")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display response
                st.markdown('<div class="response-container">', unsafe_allow_html=True)
                st.markdown("### Response:")
                st.markdown(response)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Add a divider
                st.divider()
                
                # Expand for detailed discriminator evaluation
                with st.expander("View detailed evaluation"):
                    # Get discriminator evaluation
                    if "discriminator" in evaluation and "detailed_evaluation" in evaluation["discriminator"]:
                        st.markdown("### Discriminator Evaluation:")
                        st.markdown(evaluation["discriminator"]["detailed_evaluation"])
        
        # Create metrics dataframe for visualization
        metrics_df = pd.DataFrame(metrics_data)
        
        # Add comparison charts
        st.subheader("Metrics Comparison")
        
        # Create a radar chart using plotly express
        radar_fig = px.line_polar(
            metrics_df, 
            r="value", 
            theta="metric", 
            color="variant", 
            line_close=True,
            range_r=[0, 10],
            labels={"value": "Score", "metric": "Metric", "variant": "Variant"}
        )
        
        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )
            ),
            showlegend=True
        )
        
        st.plotly_chart(radar_fig, use_container_width=True)
        
        # Add option to save comparison report
        if st.session_state.comparison_report:
            st.subheader("Comparison Report")
            st.markdown("The comparison report includes all responses, metrics, and an expert evaluation.")
            
            # Download button for the report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="Download Comparison Report",
                data=st.session_state.comparison_report,
                file_name=f"rag_comparison_{timestamp}.md",
                mime="text/markdown"
            )

# Benchmark Results Tab
with tabs[1]:
    st.header("Benchmark Results")
    
    if st.session_state.benchmark_results is not None:
        benchmark_df = st.session_state.benchmark_results
        
        # Show summary statistics
        st.subheader("Summary Statistics")
        
        # Group by variant and calculate mean scores for core metrics
        core_metrics = ['metric_faithfulness', 'metric_completeness', 'metric_citation', 'metric_average']
        summary = benchmark_df.groupby("variant")[core_metrics].mean().reset_index()
        
        # Rename columns for display
        summary.columns = [col.replace('metric_', '') for col in summary.columns]
        
        # Create a bar chart of average scores by variant
        fig = px.bar(
            summary, 
            x="variant", 
            y="average",
            title="Average Performance by Variant",
            labels={"variant": "Variant", "average": "Average Score"},
            color="average",
            color_continuous_scale="Blues",
            text_auto='.2f'
        )
        
        fig.update_layout(
            xaxis_title="RAG Variant",
            yaxis_title="Average Score (0-10)",
            yaxis_range=[0, 10]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create a heatmap of core metrics by variant
        heatmap_data = summary.set_index('variant')[['faithfulness', 'completeness', 'citation']]
        
        fig = px.imshow(
            heatmap_data.values,
            labels=dict(x="Metric", y="Variant", color="Score"),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            color_continuous_scale="Blues",
            zmin=0,
            zmax=10,
            text_auto='.2f'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed results in a table
        st.subheader("Detailed Results")
        
        # Create tabs for each question
        questions = benchmark_df["question"].unique()
        question_tabs = st.tabs([f"Q{i+1}: {q[:50] + '...' if len(q) > 50 else q}" for i, q in enumerate(questions)])
        
        for question, tab in zip(questions, question_tabs):
            with tab:
                # Filter for this question
                question_df = benchmark_df[benchmark_df["question"] == question]
                
                # Display the question
                st.markdown(f"**Question:** {question}")
                
                # Display metrics for this question
                display_cols = ["variant", "metric_faithfulness", "metric_completeness", "metric_citation", "metric_average"]
                
                # Rename columns for display
                display_df = question_df[display_cols].copy()
                display_df.columns = [col.replace('metric_', '') for col in display_cols]
                
                # Sort by average score
                display_df = display_df.sort_values('average', ascending=False)
                
                # Style the dataframe
                styled_df = display_df.style.format({
                    'faithfulness': '{:.2f}',
                    'completeness': '{:.2f}',
                    'citation': '{:.2f}',
                    'average': '{:.2f}'
                }).background_gradient(cmap='Blues', subset=['faithfulness', 'completeness', 'citation', 'average'])
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Create expander for each variant's response
                for _, row in display_df.iterrows():
                    variant = row['variant']
                    variant_data = question_df[question_df['variant'] == variant].iloc[0]
                    
                    with st.expander(f"{variant} (Avg Score: {row['average']:.2f})"):
                        st.markdown(variant_data['response'])
        
        # Add download button for full results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv = benchmark_df.to_csv(index=False)
        st.download_button(
            label="Download Full Results CSV",
            data=csv,
            file_name=f"rag_benchmark_{timestamp}.csv",
            mime="text/csv"
        )
    else:
        st.info("Run a benchmark from the sidebar to see results here.")
        st.markdown("""
        The benchmark will:
        1. Process each question with all 7 RAG variants
        2. Evaluate responses using 6 different metrics
        3. Generate a detailed comparison report
        4. Visualize the results for easy analysis
        """)