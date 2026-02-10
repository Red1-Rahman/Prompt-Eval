# 🔍 Prompt Performance Analyzer

A powerful, **100% free** prompt evaluation tool powered by Groq API and Llama 3.1 70B. Evaluate, compare, and optimize your LLM prompts with automated test case generation, intelligent grading, and actionable insights.

## ✨ Features

### Core Capabilities
- 🤖 **Automated Test Case Generation** - Generate diverse, challenging test cases using Groq API
- 📊 **Multi-Level Grading System**
  - Code-based validators (length, JSON, keywords, regex, sentiment)
  - Model-based intelligent grading with detailed feedback
  - Custom grader templates
- ⚖️ **Prompt Comparison** - Side-by-side evaluation of multiple prompt versions
- 💡 **Auto-Improvement Suggestions** - Get actionable recommendations to enhance prompts
- 📈 **Interactive Dashboard** - Beautiful Streamlit UI with charts and detailed results
- 💾 **Export Results** - Download evaluation reports as JSON

### Tech Stack (100% Free)
- **LLM**: Groq API with Llama 3.1 70B (14,400 requests/day free)
- **Frontend**: Streamlit (free Community Cloud hosting)
- **Grading**: TextBlob for sentiment, custom validators
- **Charts**: Plotly for interactive visualizations
- **Storage**: Local JSON files

## 🚀 Quick Start

### Ready to Use - No Setup Required!

The app uses a **hosted API** by default - just run and start evaluating prompts!

```bash
# Clone the repository
git clone <your-repo-url>
cd Prompt-Eval

# Install dependencies
pip install -r requirements.txt

# Download TextBlob corpora
python -m textblob.download_corpora

# Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501` - **ready to use immediately!**

### Optional: Use Your Own API Key

Want to use your own Groq API for unlimited usage?

1. Check **"Use your own API key"** in the sidebar
2. Sign up at [console.groq.com](https://console.groq.com)
3. Generate a free API key (14,400 requests/day)
4. Enter it in the sidebar

### For Hosting Your Own Instance

If you're hosting this tool, create a `.env` file:

```bash
GROQ_API_KEY=your_api_key_here
```

This key will be used as the default hosted API.

## 📖 Usage Guide

### Single Prompt Evaluation

1. **Enter Your Prompt**: Write the prompt template you want to evaluate
2. **Describe Use Case**: Explain what your prompt does (e.g., "email classifier")
3. **Configure Settings**:
   - Number of test cases (5-20)
   - Temperature (0.0-1.0)
   - Optional code-based validators
4. **Run Evaluation**: Click "Generate & Run Evaluation"
5. **Review Results**: 
   - View average score and pass rate
   - See score distribution chart
   - Get improvement suggestions
   - Analyze detailed results per test case

### Prompt Comparison

1. **Select Number of Prompts**: Choose 2 or 3 variants to compare
2. **Enter Each Prompt Version**: Name and write each prompt
3. **Set Use Case**: Describe the common use case
4. **Run Comparison**: Click "Run Comparison"
5. **Analyze Results**:
   - See winner announcement
   - Compare average scores
   - Review improvements/regressions
   - Identify which test cases improved

### Custom Graders

Test individual grading functions:

- **Length Validator**: Check min/max character limits
- **JSON Validator**: Ensure valid JSON output
- **Keyword Checker**: Verify required keywords present
- **Regex Matcher**: Match specific patterns
- **Sentiment Analyzer**: Detect positive/negative/neutral tone
- **Format Validator**: Validate emails, URLs, phone numbers, dates

## 🏗️ Architecture

```
app.py              # Streamlit web interface
├── utils.py        # Groq API client, helper functions
├── dataset_gen.py  # Test case generation
├── graders.py      # Code-based and model-based graders
└── eval_engine.py  # Orchestration and evaluation logic
```

### Component Overview

#### `utils.py` - Core Utilities
- `GroqClient`: Wrapper for Groq API calls
- `save_results()`: Save evaluation reports
- `calculate_stats()`: Compute score statistics

#### `dataset_gen.py` - Test Generation
- `DatasetGenerator`: Generate diverse test cases
  - `generate_test_cases()`: Create evaluation test cases
  - `generate_comparison_cases()`: Create consistent cases for A/B testing

#### `graders.py` - Grading System
- `CodeBasedGraders`: Fast, free validation functions
  - Length, JSON, keywords, regex, sentiment, format validators
- `ModelBasedGrader`: Intelligent LLM-based grading
  - Scores responses 1-10 with detailed feedback
- `CustomGrader`: Register custom validation functions

#### `eval_engine.py` - Orchestration
- `EvaluationEngine`: Main coordinator
  - `run_evaluation()`: Complete prompt evaluation
  - `compare_prompts()`: Side-by-side comparison
  - `suggest_improvements()`: Generate recommendations

## 💡 Best Practices

### Prompt Evaluation Tips

1. **Start with 10-15 test cases** for comprehensive evaluation
2. **Use lower temperature (0.3)** if consistency is important
3. **Enable relevant validators** (e.g., JSON validator for structured outputs)
4. **Review low-scoring cases** to identify specific issues
5. **Iterate based on suggestions** to improve performance

### Test Case Coverage

Good test datasets should include:
- ✅ Happy path scenarios
- ✅ Edge cases (empty input, very long input)
- ✅ Ambiguous cases
- ✅ Different difficulty levels
- ✅ Various categories of your use case

### Grading Strategy

- **Code-based graders**: Fast, deterministic, good for format/structure
- **Model-based grading**: Intelligent, contextual, good for quality/accuracy
- **Combined approach**: Use both for comprehensive evaluation

## 📊 Example Use Cases

### 1. Email Classifier
```python
Prompt: "Classify the following email as urgent, normal, or low priority. 
Return only the classification."

Use Case: "Email priority classification"
Test Cases: 15
Temperature: 0.3
Validators: [keyword_checker]
```

### 2. JSON Data Extractor
```python
Prompt: "Extract key information from the text and return as JSON with 
fields: name, email, phone, company."

Use Case: "Contact information extraction"
Test Cases: 12
Temperature: 0.5
Validators: [json_validator, keyword_checker]
```

### 3. Content Summarizer
```python
Prompt: "Summarize the following text in 2-3 sentences. Be concise and objective."

Use Case: "Content summarization"
Test Cases: 10
Temperature: 0.7
Validators: [length_validator, sentiment_analyzer]
```

## 🎯 Improvement Suggestions

The tool automatically suggests improvements based on:

- **Low Average Score** → Add examples (few-shot prompting)
- **High Score Variance** → Lower temperature for consistency
- **Format Violations** → Use XML tags or structured output
- **Common Weaknesses** → Address frequently mentioned issues

## 🔧 Advanced Usage

### Programmatic Evaluation

```python
from utils import GroqClient
from eval_engine import EvaluationEngine

# Initialize
client = GroqClient(api_key="your_key")
engine = EvaluationEngine(client)

# Generate test cases
test_cases = engine.dataset_gen.generate_test_cases(
    prompt="Your prompt here",
    use_case_description="Your use case",
    num_cases=15
)

# Run evaluation
results = engine.run_evaluation(
    prompt="Your prompt here",
    test_cases=test_cases,
    use_model_grading=True,
    temperature=0.7
)

# Get suggestions
suggestions = engine.suggest_improvements(results)

# Save report
filepath = engine.generate_report(results)
```

### Custom Validator Example

```python
from graders import CustomGrader

def my_validator(response: str) -> dict:
    word_count = len(response.split())
    
    if 50 <= word_count <= 100:
        return {
            "score": 10,
            "reason": f"Perfect length ({word_count} words)",
            "passed": True
        }
    else:
        return {
            "score": 5,
            "reason": f"Length out of range ({word_count} words)",
            "passed": False
        }

# Register and use
grader = CustomGrader()
grader.register_validator("my_validator", my_validator)
result = grader.run_validator("my_validator", "Your response text here")
```

## 🌐 Deployment

### Streamlit Community Cloud (Free)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Add `GROQ_API_KEY` to Secrets
5. Deploy!

### Local Deployment

```bash
# Production mode
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## 📝 API Rate Limits

Groq Free Tier:
- **14,400 requests/day**
- **30 requests/minute**

For a typical evaluation:
- 15 test cases = ~45 API calls (generation + execution + grading)
- Can run ~320 evaluations per day

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional grader templates
- More visualization options
- Export formats (CSV, HTML reports)
- Advanced analytics
- Batch evaluation mode

## 📄 License

MIT License - Feel free to use for personal or commercial projects!

## 🐛 Troubleshooting

### "Service temporarily unavailable"
- If using hosted service, wait a moment and refresh
- Alternatively, check "Use your own API key" and enter your Groq API key

### "Rate limit exceeded"
- If using your own API: Free tier has limits - wait a minute and try again
- Reduce number of test cases if hitting limits
- Or switch back to hosted service (uncheck "Use your own API key")

### "JSON parsing error"
- Groq might return non-JSON occasionally
- The tool has fallback mechanisms
- Try re-running the evaluation

### TextBlob import error
```bash
python -m textblob.download_corpora
```

## 🎓 Learn More

- [Groq Documentation](https://console.groq.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Prompt Engineering Guide](https://www.promptingguide.ai)

## 📧 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review example use cases

---

**Built with ❤️ using Groq, Streamlit, and Llama 3.1**
