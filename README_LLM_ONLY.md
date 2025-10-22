# Green Agent - LLM-only A/B Testing Analysis Evaluator

## Overview

The Green Agent is now an **LLM-only** implementation that uses OpenAI ChatGPT to provide sophisticated, nuanced evaluation of A/B testing analyses. It evaluates across three dimensions:

1. **Code Quality & Statistical Accuracy**
2. **Analytical Soundness**  
3. **Report Clarity & Insightfulness**

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Your OpenAI API Key

**Option A: Using the setup script (recommended)**
```bash
python setup.py
```

**Option B: Manual setup**
```bash
# Get your API key from: https://platform.openai.com/api-keys
export OPENAI_API_KEY="your_api_key_here"
```

**Option C: Using .env file**
Create a `.env` file:
```
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4
```

### 3. Run Example
```bash
python example_llm_usage.py
```

## 📋 Requirements

- Python 3.7+
- OpenAI API key with sufficient credits
- Internet connection for API calls

## 🔧 Usage

### Basic Usage

```python
from green_agent import GreenAgent, WhiteAgentOutput

# Initialize agent (uses API key from environment)
agent = GreenAgent()

# Or specify API key directly
agent = GreenAgent(api_key="your_api_key_here")

# Or use different model
agent = GreenAgent(model="gpt-3.5-turbo")
```

### Complete Example

```python
from green_agent import GreenAgent, WhiteAgentOutput

# Initialize agent
agent = GreenAgent()

# Create white agent output
white_output = WhiteAgentOutput(
    think_phase={
        'null_hypothesis': 'No difference in conversion rates',
        'alternative_hypothesis': 'Difference in conversion rates',
        'significance_level': 0.05,
        'data_type': 'categorical',
        'test_type': 'two-tailed'
    },
    plan_phase={
        'statistical_test': 'chi-square test',
        'sample_size': 1000,
        'minimum_detectable_effect': 0.05
    },
    act_phase={
        'p_value': 0.03,
        'confidence_interval': [0.02, 0.08],
        'effect_size': 0.15
    },
    measure_phase={
        'conclusion': 'We reject the null hypothesis'
    },
    code_snippets=[
        'from scipy.stats import chi2_contingency\nimport pandas as pd\n\n# Perform chi-square test\nchi2, p_value, dof, expected = chi2_contingency(contingency_table)'
    ],
    final_report="""Executive Summary: We conducted an A/B test to evaluate the impact of a new button design on conversion rates.
    
    Methodology: We used a chi-square test to compare conversion rates between the control and treatment groups.
    
    Results: The test revealed a statistically significant difference (p < 0.05) with a 15% improvement in conversion rates.
    
    Conclusion: We recommend implementing the new button design as it shows a significant positive impact on conversions."""
)

# Run evaluation
results = agent.evaluate_white_agent(white_output)

# Access results
print(f"Overall Score: {results['overall_score']}/100")
print(f"Method: {results['evaluation_method']}")  # Always 'LLM'

# Access dimension scores
for dimension, score in results['dimension_scores'].items():
    print(f"{dimension}: {score}/100")

# Access LLM reasoning
for dimension, eval_data in results['evaluations'].items():
    print(f"\n{dimension} Reasoning:")
    print(eval_data['reasoning'])

# Generate full report
report = agent.generate_evaluation_report(results)
print(report)
```

## 🧪 Testing

### Run Tests
```bash
# Basic tests
python test_green_agent.py

# Enhanced tests
python test_green_agent_enhanced.py

# Mock tests (no API calls)
python test_llm_mocks.py

# All tests
python run_tests.py --suite all
```

### Test Without API Calls
The mock tests allow you to test the LLM integration logic without making actual API calls:
```bash
python test_llm_mocks.py
```

## ⚙️ Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_MODEL`: Model to use (default: gpt-4)
- `LOG_LEVEL`: Logging level (default: INFO)

### Model Options

- `gpt-4`: Most capable, recommended for production
- `gpt-3.5-turbo`: Faster and cheaper, good for development
- `gpt-4-turbo`: Latest GPT-4 variant

## 📊 Evaluation Dimensions

### 1. Code Quality & Statistical Accuracy
The LLM evaluates:
- Code syntax and correctness
- Appropriate use of statistical functions
- Import statements and dependencies
- Code organization and readability
- Statistical calculation accuracy
- Error handling and robustness

### 2. Analytical Soundness
The LLM evaluates:
- Hypothesis formulation consistency
- Statistical test appropriateness for data type
- Sample size adequacy
- Result interpretation accuracy
- Logical consistency between phases
- Assumptions and limitations

### 3. Report Clarity & Insightfulness
The LLM evaluates:
- Report structure and organization
- Clarity of statistical concepts explanation
- Actionable insights and business impact
- Communication effectiveness
- Completeness of analysis
- Visualizations and supporting materials

## 🔍 Output Format

The evaluation returns a comprehensive dictionary:

```python
{
    'overall_score': 85.2,
    'evaluation_method': 'LLM',
    'dimension_scores': {
        'code_quality': 80.0,
        'analytical_soundness': 90.0,
        'report_clarity': 85.0
    },
    'evaluations': {
        'code_quality': {
            'score': 80.0,
            'issues': ['Missing error handling'],
            'recommendations': ['Add try-catch blocks'],
            'reasoning': 'The code is syntactically correct but lacks error handling...',
            'details': {...}
        },
        # ... other dimensions
    },
    'summary': {
        'total_issues': 3,
        'total_recommendations': 5,
        'strongest_dimension': 'analytical_soundness',
        'weakest_dimension': 'code_quality',
        'overall_assessment': 'Good'
    }
}
```

## 🚨 Error Handling

The agent includes robust error handling:

- **Missing API Key**: Clear error message with setup instructions
- **API Errors**: Graceful handling with detailed error information
- **JSON Parsing Errors**: Handles malformed LLM responses
- **Network Issues**: Proper error reporting for connectivity problems

## 💰 Cost Considerations

- **GPT-4**: ~$0.03-0.06 per evaluation (3 API calls)
- **GPT-3.5-turbo**: ~$0.001-0.003 per evaluation (3 API calls)
- **Token Usage**: ~2000-4000 tokens per evaluation

## 🔧 Troubleshooting

### Common Issues

1. **"OpenAI API key is required"**
   - Run: `python setup.py`
   - Or set: `export OPENAI_API_KEY="your_key"`

2. **"Rate limit exceeded"**
   - Wait a few minutes and retry
   - Consider using GPT-3.5-turbo for development

3. **"Invalid API key"**
   - Verify your key at: https://platform.openai.com/api-keys
   - Ensure you have sufficient credits

4. **Import errors**
   - Install dependencies: `pip install -r requirements.txt`

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📁 File Structure

```
cs294-agents/
├── green_agent.py              # Main LLM-only implementation
├── green_agent_llm.py          # Standalone LLM version
├── requirements.txt            # Dependencies
├── setup.py                    # Setup script
├── example_llm_usage.py        # Usage example
├── test_green_agent.py         # Basic tests
├── test_green_agent_enhanced.py # Enhanced tests
├── test_llm_mocks.py          # Mock tests
├── test_llm_integration.py    # Integration tests
├── run_tests.py               # Test runner
├── env_template.txt           # Environment template
└── README_LLM_ONLY.md         # This file
```

## 🤝 Contributing

1. Add new test cases to appropriate test files
2. Update documentation for new features
3. Ensure all tests pass: `python run_tests.py --suite all`
4. Follow existing code style and patterns

## 📄 License

This project maintains the same license as the original Green Agent implementation.

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section above
2. Run the setup script: `python setup.py`
3. Test with mock tests: `python test_llm_mocks.py`
4. Check your API key and credits at: https://platform.openai.com/api-keys
