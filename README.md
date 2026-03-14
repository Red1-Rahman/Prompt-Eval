# 🔍 Prompt Performance Analyzer

A powerful prompt evaluation tool powered by Groq API. Evaluate, compare, and optimize your LLM prompts with automated test case generation, intelligent grading, and actionable insights.

## ✨ Features

### Core Capabilities
- 🤖 **Automated Test Case Generation** - Generate diverse, challenging test cases using Groq API
- 📊 **Multi-Level Grading System**
  - Code-based validators (length, JSON, keywords, regex, sentiment) - run first
  - Model-based intelligent grading with detailed feedback
  - Few-shot example support for better model-based grading
  - Custom grader templates
- ⚡ **Token Optimization** - Code validators run first; model grading skipped on critical format failures (saves tokens!)
- ⚖️ **Prompt Comparison** - Side-by-side evaluation of multiple prompt versions
- 💡 **Auto-Improvement Suggestions** - Get actionable recommendations to enhance prompts
- 📈 **Interactive Dashboard** - Beautiful Streamlit UI with charts and detailed results
- 💾 **Export Results** - Download evaluation reports as JSON

### Tech Stack (100% Free)
- **LLM**: Groq API with llama-3.1-8b-instant
- **Frontend**: Streamlit
- **Grading**: TextBlob for sentiment, custom validators
- **Charts**: Plotly for interactive visualizations
- **Storage**: Local JSON files

## 📋 Test Case Format

```json
{
  "input": "The task/prompt to evaluate",
  "expected_criteria": "What the response should accomplish",
  "format": "Optional - expected output format (json, python, regex, etc.)"
}
```

## 🚀 Token Optimization

Code-based validators run first and instantly catch format errors. If critical issues are found (invalid JSON, missing keywords), expensive model-based grading is **skipped automatically** to save tokens. Perfect for high-volume evaluations!


### Optional: Use Your Own API Key

Want to use your own Groq API for unlimited usage?

1. Check **"Use your own API key"** in the sidebar
2. Sign up at [console.groq.com](https://console.groq.com)
3. Generate a free API key (14,400 requests/day)
4. Enter it in the sidebar

### Single Prompt Evaluation   

<img width="1920" height="2625" alt="screencapture-localhost-8501-2026-02-10-13_10_45" src="https://github.com/user-attachments/assets/4d5fb82b-223f-4393-afda-1483b54f7e6d" />


### Prompt Comparison   

<img width="1920" height="2768" alt="screencapture-localhost-8501-2026-02-10-13_19_34" src="https://github.com/user-attachments/assets/61c9a3c4-daf6-4ec1-99ef-5b55434aec0b" />


### Custom Graders    

<img width="1365" height="925" alt="localhost_8501_ (1)" src="https://github.com/user-attachments/assets/388689af-7174-4b40-9872-7c93faed4220" />


## 🎓 Learn More

- [Groq Documentation](https://console.groq.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Prompt Engineering Guide](https://www.promptingguide.ai)

## 📧 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review example use cases
- emai: redwanrrahman2002@outlook.com
---

**Built with ❤️ using Groq, Streamlit and llma**
