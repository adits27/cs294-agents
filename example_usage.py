"""
Example Usage and Test Cases for Green Agent

This script demonstrates how to use the Green Agent to evaluate White Agent outputs
for A/B testing analysis scenarios.
"""

import json
import pandas as pd
import numpy as np
from green_agent import GreenAgent, WhiteAgentOutput, create_sample_white_output


def create_test_scenarios():
    """Create various test scenarios for the Green Agent"""
    
    # Scenario 1: Good White Agent Output
    good_output = WhiteAgentOutput(
        think_phase={
            'null_hypothesis': 'No difference in click-through rates between button variants',
            'alternative_hypothesis': 'Button variant B has higher click-through rate than variant A',
            'significance_level': 0.05,
            'data_type': 'continuous',
            'test_type': 'one-tailed',
            'segmentation_variables': ['age_group', 'device_type']
        },
        plan_phase={
            'statistical_test': 't-test',
            'sample_size': 2000,
            'minimum_detectable_effect': 0.02,
            'power_analysis': True
        },
        act_phase={
            'p_value': 0.012,
            'confidence_interval': [0.015, 0.045],
            'effect_size': 0.23,
            'test_statistic': 2.45,
            'degrees_of_freedom': 1998
        },
        measure_phase={
            'conclusion': 'We reject the null hypothesis and conclude that button variant B significantly outperforms variant A',
            'business_impact': 'Expected 23% increase in click-through rates',
            'confidence_level': '95%'
        },
        code_snippets=[
            '''from scipy.stats import ttest_ind
import pandas as pd
import numpy as np

# Load and prepare data
df = pd.read_csv('ab_test_data.csv')
control_clicks = df[df['variant'] == 'A']['clicks']
treatment_clicks = df[df['variant'] == 'B']['clicks']

# Perform t-test
t_stat, p_value = ttest_ind(treatment_clicks, control_clicks, equal_var=False)
print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.3f}")''',
            
            '''from statsmodels.stats.proportion import proportion_confint
import numpy as np

# Calculate confidence interval for difference in proportions
n1, n2 = len(control_clicks), len(treatment_clicks)
p1, p2 = control_clicks.mean(), treatment_clicks.mean()

# Calculate standard error
se = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
margin_error = 1.96 * se
ci_lower = (p2 - p1) - margin_error
ci_upper = (p2 - p1) + margin_error

print(f"95% Confidence Interval: [{ci_lower:.3f}, {ci_upper:.3f}]")'''
        ],
        final_report="""Executive Summary
We conducted an A/B test to evaluate the effectiveness of a new button design (Variant B) compared to the current design (Variant A). The test ran for 2 weeks with 2,000 participants per variant.

Methodology
- Statistical Test: Independent t-test (two-sample)
- Significance Level: α = 0.05
- Sample Size: 4,000 total participants (2,000 per variant)
- Minimum Detectable Effect: 2%
- Power Analysis: 80% statistical power

Results
- P-value: 0.012 (statistically significant)
- Effect Size: 0.23 (Cohen's d)
- 95% Confidence Interval: [0.015, 0.045]
- T-statistic: 2.45

Key Findings
1. Button Variant B shows a statistically significant improvement in click-through rates
2. The effect size of 0.23 indicates a moderate practical significance
3. We can be 95% confident that the true difference lies between 1.5% and 4.5%

Segmentation Insights
- Mobile users showed 15% higher improvement than desktop users
- Users aged 25-34 demonstrated the strongest positive response
- No significant difference observed across geographic regions

Business Impact
- Expected 23% increase in click-through rates
- Projected revenue impact: $50,000 monthly increase
- Implementation cost: $5,000 (one-time design update)

Conclusion
We reject the null hypothesis and conclude that Button Variant B significantly outperforms Variant A. The results are both statistically and practically significant.

Recommendations
1. Implement Button Variant B across all pages immediately
2. Monitor performance for 4 weeks post-implementation
3. Conduct follow-up tests on other UI elements
4. Consider personalization based on device type and age group

Limitations
- Test duration was limited to 2 weeks
- Seasonal effects may not be fully captured
- Results may not generalize to all user segments equally"""
    )
    
    # Scenario 2: Poor White Agent Output (with issues)
    poor_output = WhiteAgentOutput(
        think_phase={
            'null_hypothesis': 'No difference in conversion rates',
            'alternative_hypothesis': 'No difference in conversion rates',  # Same as null!
            'significance_level': 0.01,  # Very strict
            'data_type': 'continuous',
            'test_type': 'two-tailed'
        },
        plan_phase={
            'statistical_test': 'chi-square test',  # Wrong test for continuous data
            'sample_size': 50,  # Too small
            'minimum_detectable_effect': 0.5  # Unrealistic
        },
        act_phase={
            'p_value': 0.08,  # Above significance level
            'confidence_interval': [0.1, 0.05],  # Wrong order
            'effect_size': 'large'  # Not numeric
        },
        measure_phase={
            'conclusion': 'We reject the null hypothesis'  # Contradicts p-value
        },
        code_snippets=[
            '''# Missing imports!
contingency_table = [[100, 120], [80, 90]]
chi2, p_value = chi2_contingency(contingency_table)''',  # Missing import
            
            '''# Syntax error
if p_value < 0.05:
    print("Significant")
else
    print("Not significant")'''  # Missing colon
        ],
        final_report="""The test was done. Results show significance. We should implement the change."""
    )
    
    # Scenario 3: Banner A/B Test Scenario
    banner_output = WhiteAgentOutput(
        think_phase={
            'null_hypothesis': 'No difference in banner click rates between designs',
            'alternative_hypothesis': 'Banner design B has higher click rate than design A',
            'significance_level': 0.05,
            'data_type': 'categorical',
            'test_type': 'one-tailed'
        },
        plan_phase={
            'statistical_test': 'z-test for proportions',
            'sample_size': 5000,
            'minimum_detectable_effect': 0.01,
            'power_analysis': True
        },
        act_phase={
            'p_value': 0.001,
            'confidence_interval': [0.008, 0.025],
            'effect_size': 0.15,
            'test_statistic': 3.2
        },
        measure_phase={
            'conclusion': 'We reject the null hypothesis and conclude that Banner B significantly outperforms Banner A',
            'business_impact': '15% increase in banner clicks',
            'confidence_level': '95%'
        },
        code_snippets=[
            '''from statsmodels.stats.proportion import proportions_ztest
import numpy as np

# Banner A/B test data
clicks_a, impressions_a = 450, 2500
clicks_b, impressions_b = 520, 2500

# Perform z-test for proportions
count = np.array([clicks_a, clicks_b])
nobs = np.array([impressions_a, impressions_b])
z_stat, p_value = proportions_ztest(count, nobs, alternative='larger')

print(f"Z-statistic: {z_stat:.3f}")
print(f"P-value: {p_value:.3f}")''',
            
            '''# Calculate confidence interval for difference in proportions
from statsmodels.stats.proportion import proportion_confint

p_a = clicks_a / impressions_a
p_b = clicks_b / impressions_b

# Standard error for difference
se = np.sqrt(p_a*(1-p_a)/impressions_a + p_b*(1-p_b)/impressions_b)
margin_error = 1.96 * se
diff = p_b - p_a
ci_lower = diff - margin_error
ci_upper = diff + margin_error

print(f"Difference: {diff:.3f}")
print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")'''
        ],
        final_report="""Banner A/B Test Analysis Report

Executive Summary
We tested two banner designs to optimize click-through rates. Banner B showed significant improvement over Banner A.

Methodology
- Test Type: Z-test for proportions
- Sample Size: 5,000 impressions per banner
- Duration: 1 week
- Significance Level: 5%

Results
- Banner A: 18.0% click rate (450/2500)
- Banner B: 20.8% click rate (520/2500)
- Difference: 2.8 percentage points
- P-value: 0.001 (highly significant)
- 95% Confidence Interval: [0.8%, 2.5%]

Statistical Analysis
The z-test for proportions confirms that Banner B's performance is statistically significant. The p-value of 0.001 is well below our significance threshold of 0.05.

Business Impact
- 15% relative improvement in click rates
- Estimated additional 70 clicks per 2,500 impressions
- Potential revenue impact: $2,100 monthly

Recommendations
1. Implement Banner B design immediately
2. Test additional banner variations
3. Monitor performance across different page positions
4. Consider seasonal variations in future tests

Limitations
- Test duration limited to 1 week
- Results may vary by traffic source
- Banner position effects not controlled for"""
    )
    
    return {
        'good_output': good_output,
        'poor_output': poor_output,
        'banner_output': banner_output
    }


def run_comprehensive_evaluation():
    """Run comprehensive evaluation on all test scenarios"""
    print("=" * 80)
    print("GREEN AGENT COMPREHENSIVE EVALUATION")
    print("=" * 80)
    
    green_agent = GreenAgent()
    scenarios = create_test_scenarios()
    
    results = {}
    
    for scenario_name, white_output in scenarios.items():
        print(f"\n{'='*20} EVALUATING: {scenario_name.upper()} {'='*20}")
        
        # Run evaluation
        evaluation_results = green_agent.evaluate_white_agent(white_output)
        results[scenario_name] = evaluation_results
        
        # Generate and display report
        report = green_agent.generate_evaluation_report(evaluation_results)
        print(report)
        
        # Save individual results
        with open(f'evaluation_{scenario_name}.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2)
    
    # Save comprehensive results
    with open('comprehensive_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*20} SUMMARY COMPARISON {'='*20}")
    print(f"{'Scenario':<15} {'Overall':<8} {'Code':<8} {'Analytical':<12} {'Report':<8}")
    print("-" * 60)
    
    for scenario_name, result in results.items():
        overall = result['overall_score']
        code = result['dimension_scores']['code_quality']
        analytical = result['dimension_scores']['analytical_soundness']
        report = result['dimension_scores']['report_clarity']
        
        print(f"{scenario_name:<15} {overall:<8} {code:<8} {analytical:<12} {report:<8}")
    
    return results


def create_benchmarking_framework():
    """Create a framework for benchmarking multiple White Agent runs"""
    
    class BenchmarkingFramework:
        def __init__(self):
            self.green_agent = GreenAgent()
            self.benchmark_results = []
        
        def add_white_agent_run(self, run_id: str, white_output: WhiteAgentOutput, metadata: dict = None):
            """Add a White Agent run to the benchmark"""
            evaluation = self.green_agent.evaluate_white_agent(white_output)
            
            benchmark_entry = {
                'run_id': run_id,
                'timestamp': pd.Timestamp.now().isoformat(),
                'metadata': metadata or {},
                'evaluation': evaluation
            }
            
            self.benchmark_results.append(benchmark_entry)
            return evaluation
        
        def generate_benchmark_report(self):
            """Generate a comprehensive benchmark report"""
            if not self.benchmark_results:
                return "No benchmark data available"
            
            report = []
            report.append("=" * 80)
            report.append("WHITE AGENT BENCHMARK REPORT")
            report.append("=" * 80)
            report.append("")
            
            # Summary statistics
            overall_scores = [r['evaluation']['overall_score'] for r in self.benchmark_results]
            code_scores = [r['evaluation']['dimension_scores']['code_quality'] for r in self.benchmark_results]
            analytical_scores = [r['evaluation']['dimension_scores']['analytical_soundness'] for r in self.benchmark_results]
            report_scores = [r['evaluation']['dimension_scores']['report_clarity'] for r in self.benchmark_results]
            
            report.append("SUMMARY STATISTICS")
            report.append("-" * 20)
            report.append(f"Total Runs: {len(self.benchmark_results)}")
            report.append(f"Average Overall Score: {np.mean(overall_scores):.2f}")
            report.append(f"Best Overall Score: {np.max(overall_scores):.2f}")
            report.append(f"Worst Overall Score: {np.min(overall_scores):.2f}")
            report.append("")
            
            report.append("DIMENSION AVERAGES")
            report.append("-" * 20)
            report.append(f"Code Quality: {np.mean(code_scores):.2f}")
            report.append(f"Analytical Soundness: {np.mean(analytical_scores):.2f}")
            report.append(f"Report Clarity: {np.mean(report_scores):.2f}")
            report.append("")
            
            # Individual run details
            report.append("INDIVIDUAL RUN DETAILS")
            report.append("-" * 25)
            for run in self.benchmark_results:
                eval_data = run['evaluation']
                report.append(f"Run ID: {run['run_id']}")
                report.append(f"Overall Score: {eval_data['overall_score']}")
                report.append(f"Assessment: {eval_data['summary']['overall_assessment']}")
                report.append(f"Issues: {eval_data['summary']['total_issues']}")
                report.append(f"Recommendations: {eval_data['summary']['total_recommendations']}")
                report.append("")
            
            return "\n".join(report)
        
        def save_benchmark_results(self, filename: str = "benchmark_results.json"):
            """Save benchmark results to file"""
            with open(filename, 'w') as f:
                json.dump(self.benchmark_results, f, indent=2)
    
    return BenchmarkingFramework()


def demonstrate_benchmarking():
    """Demonstrate the benchmarking framework"""
    print("\n" + "=" * 80)
    print("BENCHMARKING FRAMEWORK DEMONSTRATION")
    print("=" * 80)
    
    # Create benchmarking framework
    benchmark = create_benchmarking_framework()
    
    # Add multiple White Agent runs
    scenarios = create_test_scenarios()
    
    for scenario_name, white_output in scenarios.items():
        metadata = {
            'test_type': 'button_test' if 'button' in scenario_name else 'banner_test',
            'quality_level': 'good' if scenario_name == 'good_output' else 'poor' if scenario_name == 'poor_output' else 'medium'
        }
        
        evaluation = benchmark.add_white_agent_run(
            run_id=f"run_{scenario_name}",
            white_output=white_output,
            metadata=metadata
        )
        
        print(f"Added {scenario_name} to benchmark (Score: {evaluation['overall_score']})")
    
    # Generate benchmark report
    report = benchmark.generate_benchmark_report()
    print("\n" + report)
    
    # Save results
    benchmark.save_benchmark_results()
    print("\nBenchmark results saved to 'benchmark_results.json'")


if __name__ == "__main__":
    # Run comprehensive evaluation
    results = run_comprehensive_evaluation()
    
    # Demonstrate benchmarking framework
    demonstrate_benchmarking()
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print("Files generated:")
    print("- evaluation_good_output.json")
    print("- evaluation_poor_output.json") 
    print("- evaluation_banner_output.json")
    print("- comprehensive_evaluation_results.json")
    print("- benchmark_results.json")
    print("\nThe Green Agent is ready for production use!")
