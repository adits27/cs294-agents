# Green Agent - A/B Testing Analysis Evaluator

The Green Agent is a comprehensive evaluation system designed to benchmark and assess White Agents that perform A/B testing analysis. It evaluates White Agent outputs across three critical dimensions to ensure high-quality statistical analysis and reporting.

## Overview

The Green Agent evaluates White Agent performance across three key dimensions:

1. **Code Quality & Statistical Accuracy** (40% weight)
   - Syntax validation and code structure
   - Statistical function usage and imports
   - Accuracy of p-values, confidence intervals, and sample size calculations
   - Validation against standard statistical libraries

2. **Analytical Soundness** (40% weight)
   - Hypothesis consistency and logical flow
   - Appropriateness of statistical test selection
   - Correct interpretation of results
   - Alignment between conclusions and statistical evidence

3. **Report Clarity & Insightfulness** (20% weight)
   - Report structure and organization
   - Statistical concept communication
   - Business impact and actionable insights
   - Segmentation analysis and recommendations

## Features

- **Comprehensive Evaluation**: Multi-dimensional assessment covering technical accuracy, analytical rigor, and communication quality
- **Automated Scoring**: Weighted scoring system with detailed issue identification and recommendations
- **Benchmarking Framework**: Support for evaluating multiple White Agent runs and generating comparative reports
- **Detailed Reporting**: Human-readable evaluation reports with specific feedback and improvement suggestions
- **Statistical Validation**: Cross-validation of statistical calculations using scipy.stats and statsmodels
- **Extensible Design**: Modular architecture allowing easy addition of new evaluation criteria

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cs294-agents
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from green_agent import GreenAgent, WhiteAgentOutput

# Initialize the Green Agent
green_agent = GreenAgent()

# Create White Agent output (your A/B testing analysis)
white_output = WhiteAgentOutput(
    think_phase={
        'null_hypothesis': 'No difference in conversion rates',
        'alternative_hypothesis': 'Variant B has higher conversion rate',
        'significance_level': 0.05,
        'data_type': 'continuous'
    },
    plan_phase={
        'statistical_test': 't-test',
        'sample_size': 1000,
        'minimum_detectable_effect': 0.02
    },
    act_phase={
        'p_value': 0.03,
        'confidence_interval': [0.01, 0.05],
        'effect_size': 0.15
    },
    measure_phase={
        'conclusion': 'We reject the null hypothesis'
    },
    code_snippets=[
        'from scipy.stats import ttest_ind\n# ... your statistical code'
    ],
    final_report="Your comprehensive A/B test report..."
)

# Evaluate the White Agent output
results = green_agent.evaluate_white_agent(white_output)

# Generate human-readable report
report = green_agent.generate_evaluation_report(results)
print(report)
```

### Running Example Scenarios

```bash
python example_usage.py
```

This will run comprehensive evaluations on multiple test scenarios and demonstrate the benchmarking framework.

## Architecture

### Core Components

1. **GreenAgent**: Main orchestrator class that coordinates all evaluations
2. **CodeQualityEvaluator**: Evaluates code syntax, statistical accuracy, and function usage
3. **AnalyticalSoundnessEvaluator**: Assesses hypothesis consistency and test appropriateness
4. **ReportClarityEvaluator**: Evaluates report structure and communication quality

### Data Structures

- **WhiteAgentOutput**: Container for White Agent's four-phase output
- **EvaluationResult**: Structured results for each evaluation dimension
- **BenchmarkingFramework**: System for comparing multiple White Agent runs

## Evaluation Criteria

### Code Quality & Statistical Accuracy
- ✅ Syntax validation and error detection
- ✅ Proper import statements for statistical libraries
- ✅ Correct usage of statistical functions (scipy.stats, statsmodels)
- ✅ Validation of p-values (0 ≤ p ≤ 1)
- ✅ Confidence interval validation (lower < upper bound)
- ✅ Sample size and effect size validation
- ✅ Code complexity assessment

### Analytical Soundness
- ✅ Null and alternative hypothesis consistency
- ✅ Statistical test appropriateness for data type
- ✅ Sample size adequacy for chosen test
- ✅ Correct interpretation of p-values vs significance level
- ✅ Logical consistency between results and conclusions
- ✅ Test direction alignment (one-tailed vs two-tailed)

### Report Clarity & Insightfulness
- ✅ Report structure completeness (executive summary, methodology, results, conclusions)
- ✅ Statistical concept explanation in plain language
- ✅ Business impact and KPI connections
- ✅ Segmentation insights and recommendations
- ✅ Acknowledgment of limitations and caveats
- ✅ Actionable next steps and recommendations

## Scoring System

The Green Agent uses a weighted scoring system:

- **Overall Score**: Weighted average of all three dimensions
- **Code Quality**: 40% weight
- **Analytical Soundness**: 40% weight  
- **Report Clarity**: 20% weight

Scores range from 0-100 with the following assessments:
- 90-100: Excellent
- 80-89: Good
- 70-79: Satisfactory
- 60-69: Needs Improvement
- 0-59: Poor

## Benchmarking Framework

The Green Agent includes a comprehensive benchmarking system for evaluating multiple White Agent runs:

```python
from example_usage import create_benchmarking_framework

# Create benchmarking framework
benchmark = create_benchmarking_framework()

# Add multiple White Agent runs
benchmark.add_white_agent_run("run_1", white_output_1, metadata={"test_type": "button_test"})
benchmark.add_white_agent_run("run_2", white_output_2, metadata={"test_type": "banner_test"})

# Generate comprehensive benchmark report
report = benchmark.generate_benchmark_report()
print(report)

# Save results
benchmark.save_benchmark_results("my_benchmark.json")
```

## Example Scenarios

The system includes three example scenarios:

1. **Good Output**: Well-structured A/B test with proper statistical analysis
2. **Poor Output**: Contains multiple issues (wrong test, syntax errors, logical inconsistencies)
3. **Banner Test**: Real-world banner A/B testing scenario

## Output Files

The Green Agent generates several output files:

- `evaluation_results.json`: Detailed evaluation results in JSON format
- `evaluation_report.txt`: Human-readable evaluation report
- `benchmark_results.json`: Comprehensive benchmarking data
- Individual scenario files: `evaluation_good_output.json`, etc.

## Customization

### Adding New Evaluation Criteria

To add new evaluation criteria, extend the appropriate evaluator class:

```python
class CustomEvaluator:
    def evaluate(self, white_output: WhiteAgentOutput) -> EvaluationResult:
        # Your custom evaluation logic
        pass
```

### Modifying Scoring Weights

Adjust the weights in the `GreenAgent._calculate_overall_score()` method:

```python
weights = {
    EvaluationDimension.CODE_QUALITY: 0.4,      # Adjust as needed
    EvaluationDimension.ANALYTICAL_SOUNDNESS: 0.4,  # Adjust as needed
    EvaluationDimension.REPORT_CLARITY: 0.2     # Adjust as needed
}
```

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, issues, or contributions, please open an issue on the GitHub repository.

---

**Note**: This Green Agent is specifically designed for evaluating A/B testing analysis agents. It focuses on UI/UX testing scenarios including button changes, banner modifications, and other interface variations commonly tested in A/B experiments.