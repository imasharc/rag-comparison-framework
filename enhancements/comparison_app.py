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
    .variant-description {
        background-color: #eaf4ff;
        border-radius: 0.5rem;
        padding: 0.8rem;
        margin-bottom: 1rem;
        border-left: 4px solid #0077ff;
        font-style: italic;
    }
    .progress-detail {
        font-size: 0.9rem;
        margin-top: 0.2rem;
        color: #4a4a4a;
    }
    .stage-tracker {
        margin-top: 1rem;
        padding: 0.5rem;
        background-color: #f9f9f9;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
    }
    .info-box {
        background-color: #e8f4f9;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1e88e5;
    }
    /* Additional CSS for the animation */
    .thinking-animation {
        width: 100%;
        height: 160px;
        background-color: #f0f7ff;
        border-radius: 10px;
        padding: 20px;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
    }
    
    .brain {
        position: relative;
        width: 80px;
        height: 80px;
        background-color: #6c8ebf;
        border-radius: 50%;
        animation: pulse 2s infinite;
        box-shadow: 0 0 20px rgba(108, 142, 191, 0.7);
    }
    
    .connection {
        position: absolute;
        width: 200px;
        height: 2px;
        background: linear-gradient(90deg, #6c8ebf, transparent);
        animation: flow 1.5s infinite;
    }
    
    .connection-1 {
        top: 50px;
        transform: rotate(30deg);
    }
    
    .connection-2 {
        top: 80px;
        transform: rotate(-10deg);
    }
    
    .connection-3 {
        top: 110px;
        transform: rotate(10deg);
    }
    
    .document {
        position: absolute;
        width: 40px;
        height: 50px;
        background-color: white;
        border-radius: 3px;
        right: 30px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .document-1 {
        top: 30px;
        animation: float 3s infinite;
    }
    
    .document-2 {
        top: 90px;
        animation: float 3s infinite 0.5s;
    }
    
    .document-line {
        position: absolute;
        width: 30px;
        height: 2px;
        background-color: #ccc;
        left: 5px;
    }
    
    .line-1 { top: 10px; }
    .line-2 { top: 15px; }
    .line-3 { top: 20px; }
    .line-4 { top: 25px; }
    
    .thinking-text {
        position: absolute;
        bottom: 10px;
        color: #555;
        font-weight: bold;
        font-size: 14px;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    
    @keyframes flow {
        0% { opacity: 0; width: 0; }
        50% { opacity: 1; width: 200px; }
        100% { opacity: 0; width: 0; }
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
        100% { transform: translateY(0px); }
    }
    
    .thinking-dots:after {
        content: ' .';
        animation: dots 1.5s steps(5, end) infinite;
    }
    
    @keyframes dots {
        0%, 20% { color: rgba(0,0,0,0); text-shadow: 0.3em 0 0 rgba(0,0,0,0), 0.6em 0 0 rgba(0,0,0,0); }
        40% { color: #555; text-shadow: 0.3em 0 0 rgba(0,0,0,0), 0.6em 0 0 rgba(0,0,0,0); }
        60% { text-shadow: 0.3em 0 0 #555, 0.6em 0 0 rgba(0,0,0,0); }
        80%, 100% { text-shadow: 0.3em 0 0 #555, 0.6em 0 0 #555; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize the comparison engine
try:
    engine = get_comparison_engine()
    variant_names = engine.get_variant_names()
    variant_descriptions = engine.get_all_variant_descriptions()
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

if "progress_status" not in st.session_state:
    st.session_state.progress_status = {"message": "", "progress": 0, "active": False, "data": {}}

if "detailed_stages" not in st.session_state:
    st.session_state.detailed_stages = []

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
    
    # List all variants with descriptions
    for variant in variant_names:
        with st.expander(f"**{variant}**"):
            st.markdown(variant_descriptions[variant])
    
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

    # Create placeholder for progress information
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    animation_placeholder = st.empty()
    detail_placeholder = st.empty()

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

    # Store benchmark state in session
    if "benchmark_state" not in st.session_state:
        st.session_state.benchmark_state = {
            "running": False,
            "questions": [],
            "current_question_index": 0,
            "current_variant_index": 0,
            "results": [],
            "status": "",
            "progress": 0.0
        }

    # Process benchmark
    if run_full_benchmark and benchmark_questions:
        # Parse questions
        questions = [q.strip() for q in benchmark_questions.split("\n") if q.strip()]
        
        if questions:
            # Initialize benchmark state
            st.session_state.benchmark_state = {
                "running": True,
                "questions": questions,
                "current_question_index": 0,
                "current_variant_index": 0,
                "results": [],
                "status": "Starting benchmark...",
                "progress": 0.0
            }

    # Continue processing if benchmark is running
    if st.session_state.benchmark_state["running"]:
        state = st.session_state.benchmark_state
        
        # Display current progress
        progress_placeholder.progress(state["progress"])
        status_placeholder.info(state["status"])
        
        # Show a brain animation with the SVG
        animation_placeholder.markdown("""
        <div style="width:100%; height:160px; background-color:#f0f7ff; border-radius:10px; padding:20px; position:relative; overflow:hidden; margin-bottom:20px; text-align:center;">
            <div style="width:80px; height:80px; margin:0 auto; animation:pulse 2s infinite;">
                <svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg" style="width:100%; height:100%;">
                    <g>
                        <path fill="#6c8ebf" d="M45.6,18.7,41,14.9V7.5a1,1,0,0,0-.6-.9L30.5,2.1h-.4l-.6.2L24,5.9,18.5,2.2,17.9,2h-.4L7.6,6.6a1,1,0,0,0-.6.9v7.4L2.4,18.7a.8.8,0,0,0-.4.8v9H2a.8.8,0,0,0,.4.8L7,33.1v7.4a1,1,0,0,0,.6.9l9.9,4.5h.4l.6-.2L24,42.1l5.5,3.7.6.2h.4l9.9-4.5a1,1,0,0,0,.6-.9V33.1l4.6-3.8a.8.8,0,0,0,.4-.7V19.4h0A.8.8,0,0,0,45.6,18.7Zm-5.1,6.8H42v1.6l-3.5,2.8-.4.3-.4-.2a1.4,1.4,0,0,0-2,.7,1.5,1.5,0,0,0,.6,2l.7.3h0v5.4l-6.6,3.1-4.2-2.8-.7-.5V25.5H27a1.5,1.5,0,0,0,0-3H25.5V9.7l.7-.5,4.2-2.8L37,9.5v5.4h0l-.7.3a1.5,1.5,0,0,0-.6,2,1.4,1.4,0,0,0,1.3.9l.7-.2.4-.2.4.3L42,20.9v1.6H40.5a1.5,1.5,0,0,0,0,3ZM21,25.5h1.5V38.3l-.7.5-4.2,2.8L11,38.5V33.1h0l.7-.3a1.5,1.5,0,0,0,.6-2,1.4,1.4,0,0,0-2-.7l-.4.2-.4-.3L6,27.1V25.5H7.5a1.5,1.5,0,0,0,0-3H6V20.9l3.5-2.8.4-.3.4.2.7.2a1.4,1.4,0,0,0,1.3-.9,1.5,1.5,0,0,0-.6-2L11,15h0V9.5l6.6-3.1,4.2,2.8.7.5V22.5H21a1.5,1.5,0,0,0,0,3Z"/>
                        <path fill="#6c8ebf" d="M13.9,9.9a1.8,1.8,0,0,0,0,2.2l2.6,2.5v2.8l-4,4v5.2l4,4v2.8l-2.6,2.5a1.8,1.8,0,0,0,0,2.2,1.5,1.5,0,0,0,1.1.4,1.5,1.5,0,0,0,1.1-.4l3.4-3.5V29.4l-4-4V22.6l4-4V13.4L16.1,9.9A1.8,1.8,0,0,0,13.9,9.9Z"/>
                        <path fill="#6c8ebf" d="M31.5,14.6l2.6-2.5a1.8,1.8,0,0,0,0-2.2,1.8,1.8,0,0,0-2.2,0l-3.4,3.5v5.2l4,4v2.8l-4,4v5.2l3.4,3.5a1.7,1.7,0,0,0,2.2,0,1.8,1.8,0,0,0,0-2.2l-2.6-2.5V30.6l4-4V21.4l-4-4Z"/>
                    </g>
                </svg>
            </div>
            <div style="position:absolute; bottom:15px; width:100%; text-align:center; left:0; color:#555; font-weight:bold; font-size:14px;">
                RAG processing<span style="overflow:hidden; display:inline-block; animation:dots 1.5s steps(5,end) infinite;">...</span>
            </div>
        </div>

        <style>
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }

        @keyframes dots {
            0% { width: 0px; }
            33% { width: 10px; }
            66% { width: 20px; }
            100% { width: 30px; }
        }
        </style>
        """, unsafe_allow_html=True)
        
        if state["current_question_index"] < len(state["questions"]):
            # Get current question
            question = state["questions"][state["current_question_index"]]
            
            # Show current question information
            detail_placeholder.markdown(f"""
            <div class="info-box">
                <b>Processing question {state['current_question_index']+1}/{len(state['questions'])}:</b> {question}
            </div>
            """, unsafe_allow_html=True)
            
            # Process single question
            try:
                # Get all responses for this question
                if state["current_variant_index"] == 0:
                    # Start processing this question
                    status_placeholder.info(f"Processing question {state['current_question_index']+1}/{len(state['questions'])}: Getting responses")
                    
                    # Get responses for all variants
                    responses = engine.query_all_variants(question)
                    
                    # Store responses temporarily in session state
                    st.session_state.temp_responses = responses
                    
                    # Update state
                    state["current_variant_index"] = 1
                    state["status"] = f"Processing question {state['current_question_index']+1}/{len(state['questions'])}: Evaluating responses"
                    state["progress"] = (state["current_question_index"] + 0.5) / len(state["questions"])
                    
                    # Force rerun to update UI
                    st.rerun()
                else:
                    # Evaluate responses
                    responses = st.session_state.temp_responses
                    
                    # Create evaluation dataframe for current question
                    baseline_response = responses.get("Baseline RAG", "")
                    question_results = []
                    
                    for variant_name, response in responses.items():
                        # Show current variant being evaluated
                        status_placeholder.info(f"Evaluating {variant_name} for question {state['current_question_index']+1}/{len(state['questions'])}")
                        
                        # Evaluate this response
                        evaluation = engine.evaluate_response(question, response, baseline_response)
                        
                        # Create a row with results
                        row = {
                            "question": question,
                            "variant": variant_name,
                            "response": response
                        }
                        
                        # Add metrics
                        for metric, score in evaluation["metrics"].items():
                            row[f"metric_{metric}"] = score
                        
                        # Add to results
                        question_results.append(row)
                    
                    # Add results from this question to overall results
                    state["results"].extend(question_results)
                    
                    # Move to next question
                    state["current_question_index"] += 1
                    state["current_variant_index"] = 0
                    state["progress"] = state["current_question_index"] / len(state["questions"])
                    state["status"] = f"Completed question {state['current_question_index']}/{len(state['questions'])}"
                    
                    # Remove temporary data
                    if "temp_responses" in st.session_state:
                        del st.session_state.temp_responses
                    
                    # Force rerun to process next question
                    st.rerun()
            except Exception as e:
                # Display error
                st.error(f"Error processing benchmark: {str(e)}")
                state["running"] = False
        else:
            # Benchmark complete
            status_placeholder.success("Benchmark completed!")
            progress_placeholder.progress(1.0)
            animation_placeholder.empty()  # Remove animation when complete
            detail_placeholder.empty()
            
            # Create final dataframe
            if state["results"]:
                benchmark_df = pd.DataFrame(state["results"])
                
                # Store in session state
                st.session_state.benchmark_results = benchmark_df
                
                # Save results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs("results", exist_ok=True)  # Ensure results directory exists
                benchmark_df.to_csv(f"results/benchmark_final_{timestamp}.csv", index=False)
                
                # Provide download link
                csv = benchmark_df.to_csv(index=False)
                st.download_button(
                    label="Download Results CSV",
                    data=csv,
                    file_name=f"rag_benchmark_{timestamp}.csv",
                    mime="text/csv"
                )
            
            # Reset benchmark state
            state["running"] = False

# Display progress status if active
if st.session_state.progress_status["active"]:
    st.markdown("### Current Progress")
    
    # Main progress bar
    st.progress(st.session_state.progress_status["progress"])
    
    # Current status message
    st.info(st.session_state.progress_status["message"])
    
    # Show detailed progress information when available
    if st.session_state.progress_status["data"]:
        data = st.session_state.progress_status["data"]
        
        if "current_question" in data:
            q_num = data.get("question_num", 0)
            q_total = data.get("total_questions", 0)
            
            st.markdown(f"""
            <div class="info-box">
                <b>Current Question ({q_num}/{q_total}):</b> {data['current_question']}
            </div>
            """, unsafe_allow_html=True)
    
    # Show detailed stage tracker
    if st.session_state.detailed_stages:
        with st.expander("View detailed progress", expanded=True):
            st.markdown('<div class="stage-tracker">', unsafe_allow_html=True)
            
            for i, stage in enumerate(st.session_state.detailed_stages):
                status = "✅" if stage["completed"] else "⏳"
                st.markdown(f"**{status} {stage['message']}** <span class='progress-detail'>({stage['timestamp']})</span>", unsafe_allow_html=True)
                
                if i < len(st.session_state.detailed_stages) - 1:
                    st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

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
        progress_container = st.empty()
        status_text = st.empty()
        progress_detail = st.empty()
        
        with st.spinner("Generating and evaluating responses..."):
            # Initialize progress
            progress_bar = progress_container.progress(0)
            status_text.info("Initializing comparison...")
            
            # Define progress callback
            def update_interactive_progress(message, progress, data=None):
                progress_bar.progress(progress)
                status_text.info(message)
                if data and "current_variant" in data:
                    progress_detail.markdown(f"Processing variant: **{data['current_variant']}**")
            
            # Get responses from all variants with progress updates
            def update_variant_progress(message, progress):
                update_interactive_progress(
                    message, progress, 
                    {"current_variant": message.replace("Generating response using ", "").replace("Completed ", "")}
                )
                
            responses = engine.query_all_variants(query, update_variant_progress)
            
            # Update progress
            progress_bar.progress(0.5)
            status_text.info("Evaluating responses...")
            
            # Store results in session state
            st.session_state.last_results = responses
            
            # Generate comparison report
            progress_bar.progress(0.8)
            status_text.info("Generating comparison report...")
            st.session_state.comparison_report = engine.generate_comparison_report(query, responses)
            
            # Complete progress
            progress_bar.progress(1.0)
            status_text.success("Comparison complete!")
            progress_detail.empty()
    
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
                
                # Display variant description
                st.markdown(f'<div class="variant-description">{variant_descriptions[variant_name]}</div>', unsafe_allow_html=True)
                
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
                        # Display the variant description
                        st.markdown(f"**About this approach:** {variant_descriptions[variant]}")
                        st.markdown("---")
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
        st.info("Run a benchmark from the sidebar to see results here. The prcess may take even 15 minutes to complete.")
        st.markdown("""
        The benchmark will:
        1. Process each question with all 7 RAG variants
        2. Evaluate responses using 6 different metrics
        3. Generate a detailed comparison report
        4. Visualize the results for easy analysis
        """)