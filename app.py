import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os

from utils import GroqClient, calculate_stats
from dataset_gen import DatasetGenerator
from eval_engine import EvaluationEngine
from graders import CodeBasedGraders

# Page configuration
st.set_page_config(
    page_title="Prompt Performance Analyzer",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0e1117;
        color: #fafafa;
        text-align: center;
        padding: 15px 0;
        font-size: 14px;
        border-top: 1px solid #262730;
        z-index: 999;
    }
    .footer a {
        color: #1f77b4;
        text-decoration: none;
        margin: 0 10px;
    }
    .footer a:hover {
        color: #4a9eda;
        text-decoration: underline;
    }
    .footer-icons {
        margin-top: 5px;
    }
    .footer-icons a {
        margin: 0 8px;
        font-size: 20px;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'groq_client' not in st.session_state:
        st.session_state.groq_client = None
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None
    if 'test_cases' not in st.session_state:
        st.session_state.test_cases = None
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = None
    if 'use_own_api' not in st.session_state:
        st.session_state.use_own_api = False
    
    # Try to initialize with hosted API key from .env
    if st.session_state.groq_client is None:
        try:
            # Load API key from .env (hosted key)
            hosted_key = os.getenv("GROQ_API_KEY")
            if hosted_key and not st.session_state.use_own_api:
                st.session_state.groq_client = GroqClient(hosted_key)
        except:
            pass


def create_score_distribution_chart(results):
    """Create score distribution chart"""
    scores = [r["model_grade"]["score"] for r in results 
             if "model_grade" in r and not r["model_grade"].get("is_technical_error", False)]
    
    if not scores:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=10,
        marker_color='#1f77b4',
        name='Score Distribution'
    ))
    
    fig.update_layout(
        title="Score Distribution",
        xaxis_title="Score",
        yaxis_title="Count",
        showlegend=False,
        height=400
    )
    
    return fig


def create_comparison_chart(comparison_results):
    """Create prompt comparison chart"""
    summary = comparison_results["comparison"]["summary"]
    
    prompt_names = list(summary.keys())
    avg_scores = [summary[name]["average"] for name in prompt_names]
    
    fig = go.Figure(data=[
        go.Bar(
            x=prompt_names,
            y=avg_scores,
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(prompt_names)],
            text=avg_scores,
            texttemplate='%{text:.2f}',
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Prompt Comparison - Average Scores",
        xaxis_title="Prompt Version",
        yaxis_title="Average Score",
        height=400,
        yaxis_range=[0, 10]
    )
    
    return fig


def main():
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🔍 Prompt Performance Analyzer</div>', unsafe_allow_html=True)
    if st.session_state.use_own_api:
        st.markdown("**Powered by Groq API (Llama 3.1 70B) - Using Your API Key**")
    else:
        st.markdown("**Powered by Groq API (Llama 3.1 70B) - Hosted Service**")
    
    # Sidebar for API configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Show service status
        if st.session_state.groq_client and not st.session_state.use_own_api:
            st.success("✅ Service Ready (Hosted API)")
            st.caption("You're using the hosted API - no setup needed!")
        
        # Option to use own API
        use_own = st.checkbox(
            "Use your own API key",
            value=st.session_state.use_own_api,
            help="Switch to your own Groq API key instead of the hosted service"
        )
        
        if use_own != st.session_state.use_own_api:
            st.session_state.use_own_api = use_own
            if not use_own:
                # Switch back to hosted API
                try:
                    hosted_key = os.getenv("GROQ_API_KEY")
                    if hosted_key:
                        st.session_state.groq_client = GroqClient(hosted_key)
                        st.rerun()
                except:
                    pass
        
        # Show API key input only if user wants to use their own
        if st.session_state.use_own_api:
            st.markdown("---")
            api_key = st.text_input(
                "Your Groq API Key",
                type="password",
                help="Get your free API key at https://console.groq.com"
            )
            
            if api_key:
                try:
                    st.session_state.groq_client = GroqClient(api_key)
                    st.success("✅ Your API Key Connected")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("Please enter your API key above")
        
        st.markdown("---")
        st.markdown("### 📊 API Info")
        if st.session_state.use_own_api:
            st.info("Free Tier: 14,400 requests/day")
        else:
            st.info("Using shared hosted API")
        
        st.markdown("---")
        st.markdown("### 🔗 Quick Links")
        if st.session_state.use_own_api:
            st.markdown("- [Groq Console](https://console.groq.com)")
            st.markdown("- [Documentation](https://console.groq.com/docs)")
        else:
            st.markdown("- [About This Tool](https://github.com)")
            st.markdown("- [Report Issues](https://github.com)")
    
    # Main content
    if not st.session_state.groq_client:
        st.error("⚠️ Service temporarily unavailable")
        st.info("""
        **Option 1:** Wait a moment and refresh the page
        
        **Option 2:** Use your own API key:
        1. Check "Use your own API key" in the sidebar
        2. Sign up for free at [console.groq.com](https://console.groq.com)
        3. Generate an API key
        4. Enter it in the sidebar
        """)
        return
    
    # Tabs for different features
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Single Prompt Evaluation",
        "⚖️ Prompt Comparison",
        "📈 Results Dashboard",
        "🛠️ Custom Graders"
    ])
    
    # ===== TAB 1: Single Prompt Evaluation =====
    with tab1:
        st.header("Single Prompt Evaluation")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            prompt = st.text_area(
                "Your Prompt Template",
                height=200,
                placeholder="Enter your prompt here...\nExample: You are a professional email classifier. Classify the following email as urgent, normal, or low priority.",
                help="This is the prompt you want to evaluate"
            )
        
        with col2:
            use_case = st.text_input(
                "Use Case Description",
                placeholder="e.g., email classifier",
                help="Describe what your prompt does"
            )
            
            num_cases = st.slider(
                "Number of Test Cases",
                min_value=5,
                max_value=20,
                value=10,
                help="More test cases = more thorough evaluation"
            )
            
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Lower = more consistent, Higher = more creative"
            )
        
        # Code-based graders selection
        st.markdown("### Code-Based Validators (Optional)")
        col_g1, col_g2, col_g3 = st.columns(3)
        
        with col_g1:
            check_length = st.checkbox("Length Validator")
        with col_g2:
            check_json = st.checkbox("JSON Format Validator")
        with col_g3:
            check_sentiment = st.checkbox("Sentiment Analyzer")
        
        # Run evaluation button
        if st.button("🚀 Generate & Run Evaluation", type="primary", use_container_width=True):
            if not prompt or not use_case:
                st.error("Please provide both prompt and use case description")
            else:
                engine = EvaluationEngine(st.session_state.groq_client)
                
                # Generate test cases
                with st.spinner(f"🔄 Generating {num_cases} test cases..."):
                    test_cases = engine.dataset_gen.generate_test_cases(prompt, use_case, num_cases)
                    st.session_state.test_cases = test_cases
                
                st.success(f"✅ Generated {len(test_cases)} test cases")
                
                # Prepare code graders
                code_graders = []
                if check_length:
                    code_graders.append("length_validator")
                if check_json:
                    code_graders.append("json_validator")
                if check_sentiment:
                    code_graders.append("sentiment_analyzer")
                
                # Run evaluation
                with st.spinner("🔄 Running evaluation and grading..."):
                    results = engine.run_evaluation(
                        prompt,
                        test_cases,
                        use_model_grading=True,
                        code_graders=code_graders if code_graders else None,
                        temperature=temperature
                    )
                    st.session_state.evaluation_results = results
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Evaluation Results")
                
                # Check if all evaluations failed
                stats = results["stats"]
                if "error" in stats:
                    st.error("⚠️ " + stats["error"])
                    st.info("""
                    **Troubleshooting Steps:**
                    1. Verify your Groq API key is correct in the `.env` file
                    2. Check your internet connection
                    3. Ensure you haven't exceeded API rate limits
                    4. Try using your own API key (check the sidebar option)
                    """)
                    return
                
                # Show warning if some failed
                if stats.get("failed_evaluations", 0) > 0:
                    st.warning(f"⚠️ {stats['failed_evaluations']} evaluation(s) failed due to API issues")
                
                # Metrics
                stats = results["stats"]
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Average Score", f"{stats['average']}/10")
                with col2:
                    successful = stats['count']
                    total = results["metadata"]["total_cases"]
                    st.metric("Success Rate", f"{successful}/{total}")
                with col3:
                    st.metric("Best Score", f"{stats['max']}/10")
                with col4:
                    st.metric("Worst Score", f"{stats['min']}/10")
                
                # Score distribution chart
                fig = create_score_distribution_chart(results["results"])
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="score_dist_tab1")
                
                # Improvement suggestions
                st.markdown("### 💡 Improvement Suggestions")
                suggestions = engine.suggest_improvements(results)
                for suggestion in suggestions:
                    st.info(suggestion)
                
                # Detailed results table
                st.markdown("### 📋 Detailed Results")
                
                table_data = []
                for i, result in enumerate(results["results"], 1):
                    grade = result.get("model_grade", {})
                    is_error = grade.get("is_technical_error", False)
                    
                    row = {
                        "#": i,
                        "Input": result["test_case"]["input"][:100] + "...",
                        "Response": result["response"][:100] + "...",
                        "Score": "❌ Error" if is_error else (grade.get("score", "N/A")),
                        "Reason": grade.get("reason", "N/A")[:100] + ("..." if len(grade.get("reason", "")) > 100 else "")
                    }
                    table_data.append(row)
                
                df = pd.DataFrame(table_data)
                st.dataframe(df, use_container_width=True)
                
                # Download results
                st.download_button(
                    label="📥 Download Full Results (JSON)",
                    data=json.dumps(results, indent=2),
                    file_name=f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    # ===== TAB 2: Prompt Comparison =====
    with tab2:
        st.header("Prompt Comparison")
        st.markdown("Compare 2-3 prompt versions side-by-side on the same test cases")
        
        num_prompts = st.radio("Number of prompts to compare:", [2, 3], horizontal=True)
        
        prompts_to_compare = []
        
        for i in range(num_prompts):
            with st.expander(f"Prompt Version {i+1}", expanded=(i==0)):
                name = st.text_input(f"Name for version {i+1}", value=f"Version {i+1}", key=f"name_{i}")
                prompt = st.text_area(f"Prompt {i+1}", height=150, key=f"prompt_{i}")
                
                if name and prompt:
                    prompts_to_compare.append({"name": name, "prompt": prompt})
        
        use_case_comp = st.text_input(
            "Use Case Description",
            placeholder="e.g., email classifier",
            key="use_case_comp"
        )
        
        num_cases_comp = st.slider(
            "Number of Test Cases",
            min_value=5,
            max_value=15,
            value=10,
            key="num_cases_comp"
        )
        
        if st.button("⚖️ Run Comparison", type="primary", use_container_width=True):
            if len(prompts_to_compare) < 2:
                st.error("Please provide at least 2 complete prompts")
            elif not use_case_comp:
                st.error("Please provide a use case description")
            else:
                engine = EvaluationEngine(st.session_state.groq_client)
                
                # Generate test cases
                with st.spinner("🔄 Generating test cases..."):
                    test_cases = engine.dataset_gen.generate_comparison_cases(use_case_comp, num_cases_comp)
                
                # Run comparison
                with st.spinner("🔄 Running comparison evaluation..."):
                    comparison = engine.compare_prompts(prompts_to_compare, test_cases)
                    st.session_state.comparison_results = comparison
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Comparison Results")
                
                # Winner announcement
                winner = comparison["comparison"]["winner"]
                st.markdown(f'<div class="success-box"><h3>🏆 Winner: {winner}</h3></div>', unsafe_allow_html=True)
                
                # Comparison chart
                fig = create_comparison_chart(comparison)
                st.plotly_chart(fig, use_container_width=True, key="comparison_chart_tab2")
                
                # Summary stats table
                st.markdown("### 📈 Summary Statistics")
                summary_data = []
                for name, stats in comparison["comparison"]["summary"].items():
                    summary_data.append({
                        "Prompt": name,
                        "Avg Score": stats["average"],
                        "Min": stats["min"],
                        "Max": stats["max"],
                        "Pass Rate": f"{stats['pass_rate']}%"
                    })
                
                df_summary = pd.DataFrame(summary_data)
                st.dataframe(df_summary, use_container_width=True)
                
                # Improvements/Regressions
                if "improvements" in comparison["comparison"]:
                    imp = comparison["comparison"]["improvements"]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### ✅ Top Improvements")
                        for item in imp["improvements"][:3]:
                            st.success(f"**{item['score_change']}** - {item['input'][:80]}...")
                    
                    with col2:
                        st.markdown("### ⚠️ Regressions")
                        for item in imp["regressions"][:3]:
                            st.warning(f"**{item['score_change']}** - {item['input'][:80]}...")
                    
                    st.metric("Net Score Change", f"{imp['net_change']:+.2f}")
    
    # ===== TAB 3: Results Dashboard =====
    with tab3:
        st.header("Results Dashboard")
        
        if st.session_state.evaluation_results:
            results = st.session_state.evaluation_results
            
            # Check if there are valid results
            if "error" in results.get("stats", {}):
                st.error("The previous evaluation encountered errors. Please check your API configuration.")
                st.info("Run a new evaluation in the 'Single Prompt Evaluation' tab with a valid API key.")
                return
            
            st.subheader("Current Evaluation Summary")
            
            # Key metrics
            stats = results["stats"]
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Average Score", f"{stats['average']}/10")
            with col2:
                st.metric("Total Tests", stats['count'])
            with col3:
                st.metric("Pass Rate", f"{stats['pass_rate']}%")
            with col4:
                st.metric("Score Range", f"{stats['min']}-{stats['max']}")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Score over test cases
                scores = [r["model_grade"]["score"] for r in results["results"] 
                         if "model_grade" in r and not r["model_grade"].get("is_technical_error", False)]
                
                if scores:
                    fig = px.line(
                        x=list(range(1, len(scores)+1)),
                        y=scores,
                        labels={"x": "Test Case", "y": "Score"},
                        title="Scores Across Test Cases"
                    )
                    fig.add_hline(y=7, line_dash="dash", line_color="green", annotation_text="Pass Threshold")
                    st.plotly_chart(fig, use_container_width=True, key="scores_line_tab3")
                else:
                    st.warning("No valid scores to display")
            
            with col2:
                # Score distribution
                fig = create_score_distribution_chart(results["results"])
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="score_dist_tab3")
                else:
                    st.warning("No valid scores to display")
            
            # Detailed view
            st.markdown("### 🔍 Detailed Test Cases")
            
            for i, result in enumerate(results["results"], 1):
                grade = result.get("model_grade", {})
                is_error = grade.get("is_technical_error", False)
                score_display = "❌ Error" if is_error else f"{grade.get('score', 'N/A')}/10"
                
                with st.expander(f"Test Case #{i} - Score: {score_display}"):
                    st.markdown("**Input:**")
                    st.info(result["test_case"]["input"])
                    
                    st.markdown("**Expected Criteria:**")
                    st.text(result["test_case"]["expected_criteria"])
                    
                    st.markdown("**Response:**")
                    st.success(result["response"])
                    
                    if "model_grade" in result:
                        grade = result["model_grade"]
                        st.markdown(f"**Score:** {grade['score']}/10")
                        st.markdown(f"**Reason:** {grade['reason']}")
                        
                        if "strengths" in grade and grade["strengths"]:
                            st.markdown("**Strengths:**")
                            for s in grade["strengths"]:
                                st.write(f"- {s}")
                        
                        if "weaknesses" in grade and grade["weaknesses"]:
                            st.markdown("**Weaknesses:**")
                            for w in grade["weaknesses"]:
                                st.write(f"- {w}")
        
        else:
            st.info("Run an evaluation in the 'Single Prompt Evaluation' tab to see results here")
    
    # ===== TAB 4: Custom Graders =====
    with tab4:
        st.header("Custom Graders")
        st.markdown("Pre-built validation templates for common use cases")
        
        grader_type = st.selectbox(
            "Select Grader Template",
            [
                "Length Validator",
                "JSON Format Validator",
                "Keyword Checker",
                "Regex Pattern Matcher",
                "Sentiment Analyzer",
                "Format Validator (Email, URL, Phone)",
                "Word Count Validator"
            ]
        )
        
        st.markdown("### Test Your Grader")
        
        test_response = st.text_area("Test Response", height=150, placeholder="Enter a test response...")
        
        graders = CodeBasedGraders()
        
        if st.button("🧪 Test Grader"):
            if not test_response:
                st.warning("Please enter a test response")
            else:
                result = None
                
                if grader_type == "Length Validator":
                    min_len = st.number_input("Min Length", value=10)
                    max_len = st.number_input("Max Length", value=1000)
                    result = graders.length_validator(test_response, min_len, max_len)
                
                elif grader_type == "JSON Format Validator":
                    result = graders.json_validator(test_response)
                
                elif grader_type == "Keyword Checker":
                    keywords = st.text_input("Keywords (comma-separated)", "hello,world").split(",")
                    keywords = [k.strip() for k in keywords]
                    result = graders.keyword_checker(test_response, keywords)
                
                elif grader_type == "Regex Pattern Matcher":
                    pattern = st.text_input("Regex Pattern", r"\d{3}-\d{3}-\d{4}")
                    result = graders.regex_matcher(test_response, pattern)
                
                elif grader_type == "Sentiment Analyzer":
                    expected = st.selectbox("Expected Sentiment", ["positive", "negative", "neutral"])
                    result = graders.sentiment_analyzer(test_response, expected)
                
                elif grader_type == "Format Validator (Email, URL, Phone)":
                    format_type = st.selectbox("Format Type", ["email", "url", "phone", "date", "number"])
                    result = graders.format_validator(test_response, format_type)
                
                elif grader_type == "Word Count Validator":
                    from graders import create_word_count_validator
                    min_words = st.number_input("Min Words", value=10)
                    max_words = st.number_input("Max Words", value=100)
                    validator = create_word_count_validator(min_words, max_words)
                    result = validator(test_response)
                
                if result:
                    st.markdown("---")
                    st.subheader("Grading Result")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Score", f"{result['score']}/10")
                    with col2:
                        status = "✅ Passed" if result.get('passed', result['score'] >= 7) else "❌ Failed"
                        st.metric("Status", status)
                    
                    st.info(f"**Reason:** {result['reason']}")
                    
                    if "details" in result:
                        st.json(result["details"])
    
    # Footer
    st.markdown("""
    <div class="footer">
        <div>
            Developed by <a href="https://redwan-rahman.netlify.app/" target="_blank">Redwan Rahman</a>
        </div>
        <div class="footer-icons">
            <a href="https://github.com/Red1-Rahman" target="_blank" title="GitHub">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                </svg>
            </a>
            <a href="https://www.linkedin.com/in/redwan-rahman-13098a34b" target="_blank" title="LinkedIn">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/>
                </svg>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

