# Enhanced Green Agent - LLM-based A/B Testing Analysis Evaluator

## Overview

The Green Agent has been enhanced to use Large Language Models (LLMs) for more sophisticated and nuanced evaluation of A/B testing analyses. The agent now supports both LLM-based evaluation (default) and rule-based evaluation (fallback), providing comprehensive assessment across three dimensions:

1. **Code Quality & Statistical Accuracy**
2. **Analytical Soundness**  
3. **Report Clarity & Insightfulness**

## Key Features

### LLM-based Evaluation
- Uses OpenAI ChatGPT (GPT-4 or GPT-3.5-turbo) for intelligent evaluation
- Provides detailed reasoning for each evaluation decision
- Handles complex statistical concepts and nuanced analysis
- Graceful fallback to rule-based evaluation when LLM is unavailable

### Enhanced Test Suite
- Comprehensive test coverage including edge cases
- Mock tests to avoid API costs during development
- Integration tests for real LLM evaluation
- Performance and stress tests

### Backward Compatibility
- Maintains compatibility with existing rule-based evaluation
- Seamless fallback when LLM services are unavailable
- Same API interface for both evaluation methods

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your_api_key_here"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4
```

## Usage

### Basic Usage

```python
from green_agent import GreenAgent, create_sample_white_output

# Initialize with LLM evaluation (default)
agent = GreenAgent(use_llm=True, api_key="your_api_key")

# Or initialize with rule-based evaluation
agent = GreenAgent(use_llm=False)

# Create sample data
sample_output = create_sample_white_output()

# Run evaluation
results = agent.evaluate_white_agent(sample_output)

# Generate report
report = agent.generate_evaluation_report(results)
print(report)
```

### Advanced Usage

```python
# Use specific OpenAI model
agent = GreenAgent(use_llm=True, model="gpt-3.5-turbo")

# Access individual evaluation results
results = agent.evaluate_white_agent(white_output)
code_quality_score = results['dimension_scores']['code_quality']
llm_reasoning = results['evaluations']['code_quality']['reasoning']

# Check evaluation method used
evaluation_method = results['evaluation_method']  # 'LLM' or 'Rule-based'
```

## Test Suites

### Running Tests

1. **All Tests** (recommended):
```bash
python run_tests.py --suite all
```

2. **Specific Test Suite**:
```bash
python run_tests.py --suite enhanced
python run_tests.py --suite llm-integration
python run_tests.py --suite llm-mocks
```

3. **Using pytest**:
```bash
python run_tests.py --suite pytest
```

4. **Check Dependencies**:
```bash
python run_tests.py --check-deps
```

### Test Suite Descriptions

- **Basic Tests** (`test_green_agent.py`): Original functionality tests
- **Enhanced Tests** (`test_green_agent_enhanced.py`): Comprehensive tests with edge cases
- **LLM Integration Tests** (`test_llm_integration.py`): Tests with real OpenAI API calls
- **LLM Mock Tests** (`test_llm_mocks.py`): Mock tests to avoid API costs
- **Test Runner** (`run_tests.py`): Unified test runner for all suites

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required for LLM evaluation)
- `OPENAI_MODEL`: OpenAI model to use (default: gpt-4)
- `LOG_LEVEL`: Logging level (default: INFO)

### Model Selection

The agent supports different OpenAI models:

- `gpt-4`: Most capable, recommended for production
- `gpt-3.5-turbo`: Faster and cheaper, good for development
- `gpt-4-turbo`: Latest GPT-4 variant with improved performance

## Evaluation Dimensions

### 1. Code Quality & Statistical Accuracy
- **LLM Focus**: Code syntax, statistical function usage, error handling, modularity
- **Rule-based Focus**: Syntax validation, import checking, statistical calculation accuracy

### 2. Analytical Soundness
- **LLM Focus**: Hypothesis consistency, test appropriateness, sample size adequacy, result interpretation
- **Rule-based Focus**: Statistical test selection, p-value validation, confidence interval checking

### 3. Report Clarity & Insightfulness
- **LLM Focus**: Report structure, statistical concept explanation, actionable insights, business impact
- **Rule-based Focus**: Section presence, word count, keyword detection

## Error Handling

The agent includes robust error handling:

- **API Errors**: Graceful fallback to rule-based evaluation
- **JSON Parsing Errors**: Handles malformed LLM responses
- **Missing Dependencies**: Automatic fallback when OpenAI library unavailable
- **Invalid Input**: Validates input data and provides meaningful error messages

## Performance Considerations

- **LLM Evaluation**: ~2-5 seconds per dimension (depends on API response time)
- **Rule-based Evaluation**: ~0.1-0.5 seconds per dimension
- **Caching**: Consider implementing response caching for repeated evaluations
- **Rate Limits**: Built-in handling of OpenAI rate limits

## Examples

### Example 1: Basic Evaluation

```python
from green_agent import GreenAgent, WhiteAgentOutput

agent = GreenAgent()  # Uses LLM by default if API key available

white_output = WhiteAgentOutput(
    think_phase={'significance_level': 0.05},
    plan_phase={'statistical_test': 't-test'},
    act_phase={'p_value': 0.03},
    measure_phase={'conclusion': 'Significant'},
    code_snippets=['import numpy as np\nfrom scipy import stats'],
    final_report="Our A/B test showed significant results..."
)

results = agent.evaluate_white_agent(white_output)
print(f"Overall Score: {results['overall_score']}")
print(f"Method: {results['evaluation_method']}")
```

### Example 2: Comparing Evaluation Methods

```python
# LLM evaluation
llm_agent = GreenAgent(use_llm=True)
llm_results = llm_agent.evaluate_white_agent(white_output)

# Rule-based evaluation
rule_agent = GreenAgent(use_llm=False)
rule_results = rule_agent.evaluate_white_agent(white_output)

# Compare results
print(f"LLM Score: {llm_results['overall_score']}")
print(f"Rule-based Score: {rule_results['overall_score']}")
```

## Troubleshooting

### Common Issues

1. **"OpenAI library not available"**
   - Install: `pip install openai python-dotenv`

2. **"OpenAI API key is required"**
   - Set environment variable: `export OPENAI_API_KEY="your_key"`
   - Or create `.env` file with your API key

3. **"Rate limit exceeded"**
   - The agent handles this automatically with fallback
   - Consider implementing exponential backoff for production use

4. **Tests failing**
   - Check dependencies: `python run_tests.py --check-deps`
   - Run specific test suite: `python run_tests.py --suite llm-mocks`

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Add new test cases to appropriate test files
2. Update documentation for new features
3. Ensure all tests pass: `python run_tests.py --suite all`
4. Follow existing code style and patterns

## License

This project maintains the same license as the original Green Agent implementation.
