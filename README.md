# Green Agent - LLM-Based A/B Testing Evaluator

A sophisticated evaluation system that uses Google Gemini LLM to assess A/B testing analysis quality across three dimensions: Code Quality, Analytical Soundness, and Report Clarity.

## 🚀 Features

- **LLM-Powered Evaluation**: Uses Google Gemini 2.0 Flash for intelligent assessment
- **Multi-Dimensional Analysis**: Evaluates code quality, analytical soundness, and report clarity
- **Detailed Feedback**: Provides specific issues and actionable recommendations
- **Comprehensive Scoring**: 0-100 scoring system with detailed reasoning
- **JSON Output**: Structured results for easy integration
- **Robust Testing**: Extensive test suite with mock and integration tests

## 📋 Requirements

- Python 3.7+
- Google Gemini API key
- Required packages (see `requirements.txt`)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd cs294-agents
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add your Google API key
   ```

4. **Get your Google Gemini API key**:
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the key to your `.env` file

## 🔧 Configuration

Create a `.env` file with your API credentials:

```bash
# Google Gemini API Configuration
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_MODEL=gemini-2.0-flash

# Logging Configuration
LOG_LEVEL=INFO
```

## 📖 Usage

### Basic Usage

```python
from green_agent import GreenAgent, WhiteAgentOutput

# Initialize the Green Agent
agent = GreenAgent()

# Create sample White Agent output
white_output = WhiteAgentOutput(
    plan="A/B test for button color change",
    do="Chi-square test results: p=0.03",
    act="Confidence interval: [0.02, 0.08], Effect size: 0.15"
)

# Evaluate the output
results = agent.evaluate_white_agent(white_output)

# View results
print(f"Overall Score: {results['overall_score']}/100")
print(f"Assessment: {results['assessment']}")
```

### Running Examples

```bash
# Run the example
python3 example_llm_usage.py

# View results
python3 view_results.py

# Run tests
python3 test_llm_mocks.py
python3 test_green_agent_enhanced.py
```

## 📊 Output Format

The Green Agent provides detailed evaluation results:

```json
{
  "overall_score": 83.0,
  "evaluation_method": "LLM",
  "dimension_scores": {
    "code_quality": 85.0,
    "analytical_soundness": 85.0,
    "report_clarity": 75.0
  },
  "evaluations": {
    "code_quality": {
      "score": 85.0,
      "issues": ["Issue 1", "Issue 2"],
      "recommendations": ["Recommendation 1", "Recommendation 2"]
    }
  }
}
```

## 🧪 Testing

The project includes comprehensive tests:

- **Unit Tests**: `test_green_agent_enhanced.py`
- **LLM Mock Tests**: `test_llm_mocks.py`
- **Integration Tests**: `test_llm_integration.py`

Run all tests:
```bash
python3 run_tests.py
```

## 📁 Project Structure

```
cs294-agents/
├── green_agent.py              # Main Green Agent implementation
├── example_llm_usage.py       # Usage example
├── view_results.py            # Results viewer
├── test_*.py                  # Test suites
├── requirements.txt           # Dependencies
├── .env.example              # Environment template
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## 🔒 Security

- **Never commit your `.env` file** - it contains your API key
- Use `.env.example` as a template
- The `.gitignore` file protects sensitive files

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is part of CS294 coursework.

## 🆘 Troubleshooting

### Common Issues

1. **API Key Error**: Make sure your `GOOGLE_API_KEY` is set in `.env`
2. **Model Not Found**: Ensure you're using `gemini-2.0-flash` or another available model
3. **Import Errors**: Run `pip install -r requirements.txt`

### Getting Help

- Check the test files for usage examples
- Review the `example_llm_usage.py` for basic usage
- Run `python3 view_results.py` to see output format