"""
Green Agent - LLM-based A/B Testing Analysis Evaluator

This agent uses OpenAI ChatGPT to evaluate the White Agent's A/B testing analysis across three dimensions:
1. Code Quality & Statistical Accuracy
2. Analytical Soundness  
3. Report Clarity & Insightfulness
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationDimension(Enum):
    CODE_QUALITY = "code_quality"
    ANALYTICAL_SOUNDNESS = "analytical_soundness"
    REPORT_CLARITY = "report_clarity"


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    dimension: EvaluationDimension
    score: float  # 0-100
    details: Dict[str, Any]
    issues: List[str]
    recommendations: List[str]
    reasoning: str  # LLM's reasoning for the evaluation


@dataclass
class WhiteAgentOutput:
    """Container for White Agent's output"""
    think_phase: Dict[str, Any]
    plan_phase: Dict[str, Any]
    act_phase: Dict[str, Any]
    measure_phase: Dict[str, Any]
    code_snippets: List[str]
    final_report: str
    raw_data: Optional[pd.DataFrame] = None


class LLMEvaluator:
    """LLM-based evaluator using OpenAI ChatGPT"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize the LLM evaluator
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment variable OPENAI_API_KEY
            model: OpenAI model to use (default: gpt-4)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        logger.info(f"LLM Evaluator initialized with model: {model}")
    
    def evaluate_dimension(self, white_output: WhiteAgentOutput, dimension: EvaluationDimension) -> EvaluationResult:
        """
        Evaluate a specific dimension using LLM
        
        Args:
            white_output: WhiteAgentOutput containing all phases and results
            dimension: The evaluation dimension to assess
            
        Returns:
            EvaluationResult with score, issues, recommendations, and reasoning
        """
        prompt = self._create_evaluation_prompt(white_output, dimension)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt(dimension)},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent evaluations
                max_tokens=2000
            )
            
            evaluation_text = response.choices[0].message.content
            return self._parse_evaluation_response(evaluation_text, dimension)
            
        except Exception as e:
            logger.error(f"Error in LLM evaluation for {dimension.value}: {str(e)}")
            return self._create_fallback_result(dimension, str(e))
    
    def _get_system_prompt(self, dimension: EvaluationDimension) -> str:
        """Get system prompt for specific evaluation dimension"""
        base_prompt = """You are an expert data scientist and A/B testing analyst evaluating the quality of A/B testing analysis. 
        You must provide a comprehensive evaluation in JSON format with the following structure:
        {
            "score": <number between 0-100>,
            "issues": [<list of specific issues found>],
            "recommendations": [<list of actionable recommendations>],
            "reasoning": "<detailed explanation of your evaluation>",
            "details": {
                <additional analysis details>
            }
        }
        
        Be thorough, objective, and constructive in your evaluation."""
        
        dimension_specific = {
            EvaluationDimension.CODE_QUALITY: """
            Focus on:
            - Code syntax and correctness
            - Appropriate use of statistical functions
            - Import statements and dependencies
            - Code organization and readability
            - Statistical calculation accuracy
            """,
            
            EvaluationDimension.ANALYTICAL_SOUNDNESS: """
            Focus on:
            - Hypothesis formulation consistency
            - Statistical test appropriateness for the data type
            - Sample size adequacy
            - Result interpretation accuracy
            - Logical consistency between phases
            """,
            
            EvaluationDimension.REPORT_CLARITY: """
            Focus on:
            - Report structure and organization
            - Clarity of statistical concepts explanation
            - Actionable insights and business impact
            - Communication effectiveness
            - Completeness of analysis
            """
        }
        
        return base_prompt + dimension_specific[dimension]
    
    def _create_evaluation_prompt(self, white_output: WhiteAgentOutput, dimension: EvaluationDimension) -> str:
        """Create evaluation prompt for the LLM"""
        prompt_parts = [
            f"Please evaluate the following A/B testing analysis for {dimension.value.replace('_', ' ')}:",
            "",
            "=== WHITE AGENT OUTPUT ===",
            "",
            "THINK PHASE:",
            json.dumps(white_output.think_phase, indent=2),
            "",
            "PLAN PHASE:",
            json.dumps(white_output.plan_phase, indent=2),
            "",
            "ACT PHASE:",
            json.dumps(white_output.act_phase, indent=2),
            "",
            "MEASURE PHASE:",
            json.dumps(white_output.measure_phase, indent=2),
            "",
            "CODE SNIPPETS:",
        ]
        
        for i, snippet in enumerate(white_output.code_snippets, 1):
            prompt_parts.append(f"Snippet {i}:")
            prompt_parts.append(snippet)
            prompt_parts.append("")
        
        prompt_parts.extend([
            "FINAL REPORT:",
            white_output.final_report,
            "",
            "Please provide your evaluation in the specified JSON format."
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_evaluation_response(self, response_text: str, dimension: EvaluationDimension) -> EvaluationResult:
        """Parse LLM response into EvaluationResult"""
        try:
            # Try to extract JSON from the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                evaluation_data = json.loads(json_str)
                
                return EvaluationResult(
                    dimension=dimension,
                    score=float(evaluation_data.get('score', 0)),
                    details=evaluation_data.get('details', {}),
                    issues=evaluation_data.get('issues', []),
                    recommendations=evaluation_data.get('recommendations', []),
                    reasoning=evaluation_data.get('reasoning', response_text)
                )
            else:
                # Fallback if JSON parsing fails
                return self._create_fallback_result(dimension, "Failed to parse JSON response")
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {str(e)}")
            return self._create_fallback_result(dimension, f"JSON parsing error: {str(e)}")
        except Exception as e:
            logger.error(f"Error parsing evaluation response: {str(e)}")
            return self._create_fallback_result(dimension, f"Parsing error: {str(e)}")
    
    def _create_fallback_result(self, dimension: EvaluationDimension, error_message: str) -> EvaluationResult:
        """Create a fallback result when LLM evaluation fails"""
        return EvaluationResult(
            dimension=dimension,
            score=50.0,  # Neutral score
            details={'error': error_message},
            issues=[f"Evaluation failed: {error_message}"],
            recommendations=["Review the analysis manually due to evaluation error"],
            reasoning=f"Fallback evaluation due to error: {error_message}"
        )


class GreenAgentLLM:
    """Main Green Agent class that uses LLM for evaluation"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize the LLM-based Green Agent
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment variable OPENAI_API_KEY
            model: OpenAI model to use (default: gpt-4)
        """
        self.llm_evaluator = LLMEvaluator(api_key=api_key, model=model)
        logger.info("Green Agent LLM initialized successfully")
    
    def evaluate_white_agent(self, white_output: WhiteAgentOutput) -> Dict[str, Any]:
        """
        Main evaluation method that evaluates the White Agent's output using LLM
        
        Args:
            white_output: WhiteAgentOutput containing all phases and results
            
        Returns:
            Dictionary containing comprehensive evaluation results
        """
        logger.info("Starting LLM-based evaluation of White Agent output")
        
        # Run evaluations for all three dimensions
        code_quality_result = self.llm_evaluator.evaluate_dimension(white_output, EvaluationDimension.CODE_QUALITY)
        analytical_soundness_result = self.llm_evaluator.evaluate_dimension(white_output, EvaluationDimension.ANALYTICAL_SOUNDNESS)
        report_clarity_result = self.llm_evaluator.evaluate_dimension(white_output, EvaluationDimension.REPORT_CLARITY)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score([
            code_quality_result,
            analytical_soundness_result,
            report_clarity_result
        ])
        
        # Compile comprehensive results
        evaluation_results = {
            'overall_score': overall_score,
            'dimension_scores': {
                'code_quality': code_quality_result.score,
                'analytical_soundness': analytical_soundness_result.score,
                'report_clarity': report_clarity_result.score
            },
            'evaluations': {
                'code_quality': {
                    'score': code_quality_result.score,
                    'issues': code_quality_result.issues,
                    'recommendations': code_quality_result.recommendations,
                    'details': code_quality_result.details,
                    'reasoning': code_quality_result.reasoning
                },
                'analytical_soundness': {
                    'score': analytical_soundness_result.score,
                    'issues': analytical_soundness_result.issues,
                    'recommendations': analytical_soundness_result.recommendations,
                    'details': analytical_soundness_result.details,
                    'reasoning': analytical_soundness_result.reasoning
                },
                'report_clarity': {
                    'score': report_clarity_result.score,
                    'issues': report_clarity_result.issues,
                    'recommendations': report_clarity_result.recommendations,
                    'details': report_clarity_result.details,
                    'reasoning': report_clarity_result.reasoning
                }
            },
            'summary': self._generate_evaluation_summary([
                code_quality_result,
                analytical_soundness_result,
                report_clarity_result
            ])
        }
        
        logger.info(f"LLM evaluation completed. Overall score: {overall_score}")
        return evaluation_results
    
    def _calculate_overall_score(self, results: List[EvaluationResult]) -> float:
        """Calculate weighted overall score"""
        weights = {
            EvaluationDimension.CODE_QUALITY: 0.4,
            EvaluationDimension.ANALYTICAL_SOUNDNESS: 0.4,
            EvaluationDimension.REPORT_CLARITY: 0.2
        }
        
        weighted_sum = sum(result.score * weights[result.dimension] for result in results)
        return round(weighted_sum, 2)
    
    def _generate_evaluation_summary(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Generate a summary of the evaluation"""
        total_issues = sum(len(result.issues) for result in results)
        total_recommendations = sum(len(result.recommendations) for result in results)
        
        # Identify strongest and weakest dimensions
        scores = [(result.dimension.value, result.score) for result in results]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        strongest_dimension = scores[0][0]
        weakest_dimension = scores[-1][0]
        
        return {
            'total_issues': total_issues,
            'total_recommendations': total_recommendations,
            'strongest_dimension': strongest_dimension,
            'weakest_dimension': weakest_dimension,
            'overall_assessment': self._get_overall_assessment(sum(r.score for r in results) / len(results))
        }
    
    def _get_overall_assessment(self, average_score: float) -> str:
        """Get overall assessment based on average score"""
        if average_score >= 90:
            return "Excellent"
        elif average_score >= 80:
            return "Good"
        elif average_score >= 70:
            return "Satisfactory"
        elif average_score >= 60:
            return "Needs Improvement"
        else:
            return "Poor"
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate a human-readable evaluation report"""
        report = []
        report.append("=" * 60)
        report.append("GREEN AGENT LLM EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall summary
        report.append("OVERALL SUMMARY")
        report.append("-" * 20)
        report.append(f"Overall Score: {evaluation_results['overall_score']}/100")
        report.append(f"Assessment: {evaluation_results['summary']['overall_assessment']}")
        report.append(f"Total Issues Found: {evaluation_results['summary']['total_issues']}")
        report.append(f"Total Recommendations: {evaluation_results['summary']['total_recommendations']}")
        report.append("")
        
        # Dimension scores
        report.append("DIMENSION SCORES")
        report.append("-" * 20)
        for dimension, score in evaluation_results['dimension_scores'].items():
            report.append(f"{dimension.replace('_', ' ').title()}: {score}/100")
        report.append("")
        
        # Detailed evaluations
        for dimension, eval_data in evaluation_results['evaluations'].items():
            report.append(f"{dimension.replace('_', ' ').upper()} EVALUATION")
            report.append("-" * 30)
            report.append(f"Score: {eval_data['score']}/100")
            report.append("")
            
            if eval_data['reasoning']:
                report.append("LLM Reasoning:")
                report.append(eval_data['reasoning'])
                report.append("")
            
            if eval_data['issues']:
                report.append("Issues Found:")
                for i, issue in enumerate(eval_data['issues'], 1):
                    report.append(f"  {i}. {issue}")
                report.append("")
            
            if eval_data['recommendations']:
                report.append("Recommendations:")
                for i, rec in enumerate(eval_data['recommendations'], 1):
                    report.append(f"  {i}. {rec}")
                report.append("")
        
        report.append("=" * 60)
        return "\n".join(report)


# Example usage and testing functions
def create_sample_white_output() -> WhiteAgentOutput:
    """Create a sample White Agent output for testing"""
    return WhiteAgentOutput(
        think_phase={
            'null_hypothesis': 'There is no difference in conversion rates between variants',
            'alternative_hypothesis': 'There is a difference in conversion rates between variants',
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
            'conclusion': 'We reject the null hypothesis and conclude there is a significant difference'
        },
        code_snippets=[
            'from scipy.stats import chi2_contingency\nimport pandas as pd\n\n# Perform chi-square test\nchi2, p_value, dof, expected = chi2_contingency(contingency_table)',
            'import numpy as np\nfrom statsmodels.stats.proportion import proportion_confint\n\n# Calculate confidence interval\nci = proportion_confint(count, nobs, alpha=0.05)'
        ],
        final_report="""Executive Summary: We conducted an A/B test to evaluate the impact of a new button design on conversion rates. 
        
        Methodology: We used a chi-square test to compare conversion rates between the control and treatment groups.
        
        Results: The test revealed a statistically significant difference (p < 0.05) with a 15% improvement in conversion rates.
        
        Conclusion: We recommend implementing the new button design as it shows a significant positive impact on conversions.
        
        Recommendations: 
        1. Implement the new design across all pages
        2. Monitor performance for 2 weeks post-implementation
        3. Consider testing additional design variations"""
    )


if __name__ == "__main__":
    # Example usage
    try:
        green_agent = GreenAgentLLM()
        sample_output = create_sample_white_output()
        
        # Run evaluation
        results = green_agent.evaluate_white_agent(sample_output)
        
        # Generate and print report
        report = green_agent.generate_evaluation_report(results)
        print(report)
        
        # Save results to JSON
        with open('evaluation_results_llm.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\nEvaluation results saved to 'evaluation_results_llm.json'")
        
    except Exception as e:
        print(f"Error running Green Agent LLM: {str(e)}")
        print("Make sure to set your OPENAI_API_KEY environment variable")
