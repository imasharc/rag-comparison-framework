# Enhanced RAG Comparison Framework

This project implements a comprehensive framework for comparing and evaluating different Retrieval-Augmented Generation (RAG) approaches for question answering on security policies. The system includes a baseline RAG implementation and six enhanced variants using advanced techniques in prompt engineering and retrieval methods.

## Project Overview

This framework was created to demonstrate how different RAG techniques can improve the performance of a security policy question-answering system. The system uses a sample security policy for the fictional NovaTech Dynamics company and evaluates how different approaches impact the quality of answers to user questions.

The enhanced RAG system includes:
- **3 Module Pairs**: Query Expansion + Reranking, Hybrid Search + Contextual Compression, Adaptive Chunking + Self-Query
- **3 Prompt Engineering Techniques**: Chain-of-Thought, Few-Shot Examples, Role-Based + Self-Verification

These enhancements are evaluated using multiple RAGAs metrics and an LLM discriminator, with results visualized through an interactive Streamlit interface.

## Installation Instructions

### 1. Copy the Project and Open in Editor

```bash
# Clone the repository
git clone https://github.com/imasharc/rag-comparison-framework.git

# Navigate to the project directory
cd rag-comparison-framework
```

Open the project folder in your preferred code editor (VSCodium, VS Code, PyCharm, etc.).

### 2. Create a Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install base dependencies
pip install -r requirements.txt

# Install enhancement dependencies
pip install -r enhancements/requirements.txt
```

### 4. Set Up OpenAI API Key

The system requires an OpenAI API key for embeddings and text generation:

```bash
# On Windows (Command Prompt)
set OPENAI_API_KEY=your_api_key_here

# On Windows (PowerShell)
$env:OPENAI_API_KEY="your_api_key_here"

# On macOS/Linux
export OPENAI_API_KEY=your_api_key_here
```

Alternatively, create a `.env` file in the project root:

```
OPENAI_API_KEY=your_api_key_here
```

## Running the Application

The application consists of two components that need to be run separately:

### 1. Start the Baseline RAG API

In a terminal window:

```bash
# Make sure your virtual environment is activated
python app.py
```

This will start the Flask API server on http://127.0.0.1:5000. You should see:
```
Starting API server at http://127.0.0.1:5000
```

### 2. Start the Comparison UI

In a new terminal window:

```bash
# Make sure your virtual environment is activated
streamlit run enhancements/comparison_app.py
```

This will start the Streamlit application and automatically open your browser to http://localhost:8501

If the browser doesn't open automatically, manually navigate to the URL shown in the terminal output.

## Conducting the Evaluation (Assignment Goal)

The primary goal of this project is to evaluate different RAG approaches on security policy questions. Here's how to run the evaluation:

### Running the Benchmark on 3 Questions

1. In the Streamlit application, locate the "Benchmark" section in the left sidebar
2. You'll see a text area with 3 default security policy questions:
   - What is NovaTech's password policy?
   - How does NovaTech protect confidential data?
   - What happens during employee termination at NovaTech?
3. Click the "Run Full Benchmark" button to start the evaluation
4. The system will:
   - Process each question with all 7 RAG variants (baseline + 6 enhancements)
   - Evaluate each response using 6 different metrics
   - Use an LLM discriminator to provide expert assessment
   - Generate visualizations and detailed comparisons
5. This process takes about 10-15 minutes to complete (the progress will be displayed)

### Understanding the Evaluation Results

Once the benchmark completes, navigate to the "Benchmark Results" tab to view the comprehensive evaluation:

1. **Summary Statistics**:
   - The bar chart shows average performance of each RAG variant across all questions
   - The heatmap displays detailed performance on three core metrics (faithfulness, completeness, citation)
   - Higher scores (0-10 scale) indicate better performance

2. **Question-Specific Results**:
   - Each question has its own tab showing detailed metrics for all variants
   - Variants are ranked by average score
   - You can expand each variant to see the full response

3. **What the Metrics Mean**:
   - **Faithfulness (0-10)**: How well the response is grounded in facts from the security policy without hallucination
   - **Completeness (0-10)**: Whether all aspects of the question are thoroughly addressed
   - **Citation (0-10)**: How well the response references specific sections of the security policy
   - **Context Relevance (0-10)**: How relevant the retrieved documents are to the query
   - **Answer Relevance (0-10)**: How well the answer addresses the specific question
   - **Coherence (0-10)**: The logical flow and readability of the response

4. **Downloading Results**:
   - Click "Download Full Results CSV" to save the complete evaluation data
   - This CSV contains all 21 rows (3 questions × 7 variants) with all metrics
   - The CSV can be imported into Excel, Python, or other tools for further analysis

### Key Findings to Look For

When analyzing the results, pay attention to:

1. **Which RAG variants perform best overall**: Look at the average scores across all questions
2. **Where each enhancement shines**: Some variants may excel at specific metrics (e.g., better citation accuracy)
3. **Question-specific performance**: Different approaches may work better for different types of questions
4. **Trade-offs between approaches**: Some methods may improve one metric at the expense of another

## Interactive Testing

You can also test additional questions:

1. Navigate to the "Interactive Comparison" tab
2. Enter any question about NovaTech's security policy
3. Click "Compare Approaches"
4. View the detailed responses and metrics from all 7 RAG variants
5. The radar chart visually compares performance across multiple metrics
6. Download a detailed comparison report by clicking "Download Comparison Report"

## Troubleshooting Common Issues

1. **API Connection Error**:
   - Ensure the Flask API (`app.py`) is running before starting the Streamlit app
   - Check that the API is running on port 5000 (http://127.0.0.1:5000)

2. **OpenAI API Errors**:
   - Verify your API key is correctly set
   - Check for rate limiting if running many evaluations

3. **Missing Dependencies**:
   - Make sure you've installed dependencies from both requirements.txt files

4. **Slow Performance**:
   - The benchmark process is comprehensive and requires multiple API calls
   - Results are saved periodically, so no progress is lost even if interrupted

If you encounter any issues, check the error messages in the terminal windows running the Flask API and Streamlit application for detailed information.

## Project Structure Overview

The project consists of:

1. **Baseline RAG System**:
   - Pre-existing RAG implementation for security policy QA
   - Uses OpenAI for embeddings and text generation
   - Processes PDF documents and retrieves relevant information

2. **Enhancement Framework**:
   - Six different RAG enhancements using advanced techniques
   - Comprehensive evaluation system with multiple metrics
   - Interactive visualization and comparison UI
   - Result export capabilities for further analysis

The enhancements demonstrate various approaches to improving RAG performance through better retrieval methods and prompt engineering techniques.