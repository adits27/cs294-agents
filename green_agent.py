"""
Green Agent - A/B Testing Analysis Evaluator

This agent evaluates the White Agent's A/B testing analysis across three dimensions:
1. Code Quality & Statistical Accuracy
2. Analytical Soundness  
3. Report Clarity & Insightfulness
"""

import ast
import re
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

# Statistical libraries for validation
from scipy import stats
from statsmodels.stats.power import ttest_power
from statsmodels.stats.proportion import proportion_confint
import warnings
warnings.filterwarnings('ignore')

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


class CodeQualityEvaluator:
    """Evaluates code quality and statistical accuracy"""
    
    def __init__(self):
        self.statistical_functions = {
            'ttest_1samp', 'ttest_ind', 'ttest_rel', 'chi2_contingency',
            'mannwhitneyu', 'wilcoxon', 'kruskal', 'f_oneway',
            'proportion_confint', 'ttest_power', 'normaltest'
        }
    
    def evaluate(self, white_output: WhiteAgentOutput) -> EvaluationResult:
        """Main evaluation method for code quality"""
        issues = []
        recommendations = []
        details = {}
        
        # Evaluate code snippets
        code_issues = self._evaluate_code_snippets(white_output.code_snippets)
        issues.extend(code_issues['issues'])
        recommendations.extend(code_issues['recommendations'])
        details['code_analysis'] = code_issues['details']
        
        # Evaluate statistical calculations
        stat_issues = self._evaluate_statistical_calculations(white_output)
        issues.extend(stat_issues['issues'])
        recommendations.extend(stat_issues['recommendations'])
        details['statistical_analysis'] = stat_issues['details']
        
        # Calculate overall score
        score = self._calculate_code_quality_score(issues, details)
        
        return EvaluationResult(
            dimension=EvaluationDimension.CODE_QUALITY,
            score=score,
            details=details,
            issues=issues,
            recommendations=recommendations
        )
    
    def _evaluate_code_snippets(self, code_snippets: List[str]) -> Dict[str, Any]:
        """Evaluate the quality of code snippets"""
        issues = []
        recommendations = []
        details = {
            'total_snippets': len(code_snippets),
            'syntax_errors': 0,
            'statistical_functions_used': [],
            'missing_imports': [],
            'code_complexity': []
        }
        
        for i, snippet in enumerate(code_snippets):
            try:
                # Check syntax
                ast.parse(snippet)
                
                # Check for statistical functions
                used_functions = self._extract_statistical_functions(snippet)
                details['statistical_functions_used'].extend(used_functions)
                
                # Check for missing imports
                missing_imports = self._check_missing_imports(snippet)
                if missing_imports:
                    details['missing_imports'].extend(missing_imports)
                    issues.append(f"Snippet {i+1}: Missing imports for {missing_imports}")
                
                # Check code complexity
                complexity = self._calculate_complexity(snippet)
                details['code_complexity'].append(complexity)
                
            except SyntaxError as e:
                details['syntax_errors'] += 1
                issues.append(f"Snippet {i+1}: Syntax error - {str(e)}")
        
        # Generate recommendations
        if details['syntax_errors'] > 0:
            recommendations.append("Fix syntax errors in code snippets")
        
        if details['missing_imports']:
            recommendations.append("Add proper import statements for statistical functions")
        
        if not details['statistical_functions_used']:
            recommendations.append("Use appropriate statistical functions from scipy.stats or statsmodels")
        
        return {
            'issues': issues,
            'recommendations': recommendations,
            'details': details
        }
    
    def _extract_statistical_functions(self, code: str) -> List[str]:
        """Extract statistical function calls from code"""
        functions = []
        for func in self.statistical_functions:
            if func in code:
                functions.append(func)
        return functions
    
    def _check_missing_imports(self, code: str) -> List[str]:
        """Check for missing imports in code"""
        missing = []
        if 'stats.' in code and 'from scipy import stats' not in code and 'import scipy.stats' not in code:
            missing.append('scipy.stats')
        if 'proportion_confint' in code and 'from statsmodels.stats.proportion' not in code:
            missing.append('statsmodels.stats.proportion')
        return missing
    
    def _calculate_complexity(self, code: str) -> int:
        """Calculate simple code complexity metric"""
        lines = code.split('\n')
        complexity = 0
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                complexity += 1
        return complexity
    
    def _evaluate_statistical_calculations(self, white_output: WhiteAgentOutput) -> Dict[str, Any]:
        """Evaluate statistical calculations for accuracy"""
        issues = []
        recommendations = []
        details = {
            'p_value_validation': {},
            'confidence_interval_validation': {},
            'sample_size_validation': {},
            'effect_size_validation': {}
        }
        
        # Extract statistical results from phases
        act_phase = white_output.act_phase
        plan_phase = white_output.plan_phase
        
        # Validate p-values
        if 'p_value' in act_phase:
            p_val = act_phase['p_value']
            if not isinstance(p_val, (int, float)) or p_val < 0 or p_val > 1:
                issues.append(f"Invalid p-value: {p_val}. Must be between 0 and 1")
            else:
                details['p_value_validation']['valid'] = True
                details['p_value_validation']['value'] = p_val
        
        # Validate confidence intervals
        if 'confidence_interval' in act_phase:
            ci = act_phase['confidence_interval']
            if not isinstance(ci, (list, tuple)) or len(ci) != 2:
                issues.append("Confidence interval must be a list/tuple of length 2")
            elif ci[0] >= ci[1]:
                issues.append("Confidence interval lower bound must be less than upper bound")
            else:
                details['confidence_interval_validation']['valid'] = True
                details['confidence_interval_validation']['value'] = ci
        
        # Validate sample size calculations
        if 'sample_size' in plan_phase:
            sample_size = plan_phase['sample_size']
            if not isinstance(sample_size, (int, float)) or sample_size <= 0:
                issues.append(f"Invalid sample size: {sample_size}. Must be positive")
            else:
                details['sample_size_validation']['valid'] = True
                details['sample_size_validation']['value'] = sample_size
        
        # Validate effect size
        if 'effect_size' in act_phase:
            effect_size = act_phase['effect_size']
            if not isinstance(effect_size, (int, float)):
                issues.append("Effect size must be numeric")
            else:
                details['effect_size_validation']['valid'] = True
                details['effect_size_validation']['value'] = effect_size
        
        # Generate recommendations
        if issues:
            recommendations.append("Review statistical calculations for accuracy")
        
        return {
            'issues': issues,
            'recommendations': recommendations,
            'details': details
        }
    
    def _calculate_code_quality_score(self, issues: List[str], details: Dict[str, Any]) -> float:
        """Calculate overall code quality score"""
        base_score = 100.0
        
        # Deduct points for issues
        issue_penalty = min(len(issues) * 10, 50)  # Max 50 points deduction
        
        # Deduct points for syntax errors
        syntax_penalty = details.get('code_analysis', {}).get('syntax_errors', 0) * 15
        
        # Deduct points for missing imports
        import_penalty = len(details.get('code_analysis', {}).get('missing_imports', [])) * 5
        
        score = max(0, base_score - issue_penalty - syntax_penalty - import_penalty)
        return round(score, 2)


class AnalyticalSoundnessEvaluator:
    """Evaluates analytical soundness and logical consistency"""
    
    def evaluate(self, white_output: WhiteAgentOutput) -> EvaluationResult:
        """Main evaluation method for analytical soundness"""
        issues = []
        recommendations = []
        details = {}
        
        # Evaluate hypothesis consistency
        hypothesis_issues = self._evaluate_hypothesis_consistency(white_output)
        issues.extend(hypothesis_issues['issues'])
        recommendations.extend(hypothesis_issues['recommendations'])
        details['hypothesis_analysis'] = hypothesis_issues['details']
        
        # Evaluate statistical test appropriateness
        test_issues = self._evaluate_test_appropriateness(white_output)
        issues.extend(test_issues['issues'])
        recommendations.extend(test_issues['recommendations'])
        details['test_analysis'] = test_issues['details']
        
        # Evaluate result interpretation
        interpretation_issues = self._evaluate_result_interpretation(white_output)
        issues.extend(interpretation_issues['issues'])
        recommendations.extend(interpretation_issues['recommendations'])
        details['interpretation_analysis'] = interpretation_issues['details']
        
        # Calculate overall score
        score = self._calculate_analytical_soundness_score(issues, details)
        
        return EvaluationResult(
            dimension=EvaluationDimension.ANALYTICAL_SOUNDNESS,
            score=score,
            details=details,
            issues=issues,
            recommendations=recommendations
        )
    
    def _evaluate_hypothesis_consistency(self, white_output: WhiteAgentOutput) -> Dict[str, Any]:
        """Evaluate consistency between hypotheses and results"""
        issues = []
        recommendations = []
        details = {
            'null_hypothesis': None,
            'alternative_hypothesis': None,
            'test_direction': None,
            'conclusion': None
        }
        
        think_phase = white_output.think_phase
        measure_phase = white_output.measure_phase
        
        # Extract hypotheses
        if 'null_hypothesis' in think_phase:
            details['null_hypothesis'] = think_phase['null_hypothesis']
        if 'alternative_hypothesis' in think_phase:
            details['alternative_hypothesis'] = think_phase['alternative_hypothesis']
        
        # Extract test direction
        if 'test_type' in think_phase:
            test_type = think_phase['test_type']
            if 'one-tailed' in test_type.lower():
                details['test_direction'] = 'one-tailed'
            elif 'two-tailed' in test_type.lower():
                details['test_direction'] = 'two-tailed'
        
        # Extract conclusion
        if 'conclusion' in measure_phase:
            details['conclusion'] = measure_phase['conclusion']
        
        # Check for consistency issues
        if details['null_hypothesis'] and details['alternative_hypothesis']:
            if details['null_hypothesis'] == details['alternative_hypothesis']:
                issues.append("Null and alternative hypotheses are identical")
        
        if details['conclusion'] and details['test_direction']:
            # Check if conclusion matches test direction
            if 'reject' in details['conclusion'].lower() and 'fail to reject' in details['conclusion'].lower():
                issues.append("Conclusion contains contradictory statements")
        
        return {
            'issues': issues,
            'recommendations': recommendations,
            'details': details
        }
    
    def _evaluate_test_appropriateness(self, white_output: WhiteAgentOutput) -> Dict[str, Any]:
        """Evaluate if the chosen statistical test is appropriate"""
        issues = []
        recommendations = []
        details = {
            'chosen_test': None,
            'data_type': None,
            'sample_size': None,
            'test_appropriateness': 'unknown'
        }
        
        plan_phase = white_output.plan_phase
        think_phase = white_output.think_phase
        
        # Extract test information
        if 'statistical_test' in plan_phase:
            details['chosen_test'] = plan_phase['statistical_test']
        
        if 'data_type' in think_phase:
            details['data_type'] = think_phase['data_type']
        
        if 'sample_size' in plan_phase:
            details['sample_size'] = plan_phase['sample_size']
        
        # Evaluate test appropriateness
        if details['chosen_test'] and details['data_type']:
            test = details['chosen_test'].lower()
            data_type = details['data_type'].lower()
            
            if 'continuous' in data_type and 'chi-square' in test:
                issues.append("Chi-square test inappropriate for continuous data")
                details['test_appropriateness'] = 'inappropriate'
            elif 'categorical' in data_type and 't-test' in test:
                issues.append("T-test inappropriate for categorical data")
                details['test_appropriateness'] = 'inappropriate'
            else:
                details['test_appropriateness'] = 'appropriate'
        
        # Check sample size requirements
        if details['sample_size'] and details['chosen_test']:
            sample_size = details['sample_size']
            test = details['chosen_test'].lower()
            
            if sample_size < 30 and 't-test' in test:
                issues.append("Sample size may be too small for t-test (consider non-parametric alternatives)")
            elif sample_size < 5 and 'chi-square' in test:
                issues.append("Sample size too small for chi-square test")
        
        return {
            'issues': issues,
            'recommendations': recommendations,
            'details': details
        }
    
    def _evaluate_result_interpretation(self, white_output: WhiteAgentOutput) -> Dict[str, Any]:
        """Evaluate how results are interpreted"""
        issues = []
        recommendations = []
        details = {
            'p_value': None,
            'significance_level': None,
            'conclusion': None,
            'interpretation_consistency': 'unknown'
        }
        
        act_phase = white_output.act_phase
        think_phase = white_output.think_phase
        measure_phase = white_output.measure_phase
        
        # Extract key values
        if 'p_value' in act_phase:
            details['p_value'] = act_phase['p_value']
        
        if 'significance_level' in think_phase:
            details['significance_level'] = think_phase['significance_level']
        
        if 'conclusion' in measure_phase:
            details['conclusion'] = measure_phase['conclusion']
        
        # Check interpretation consistency
        if details['p_value'] and details['significance_level'] and details['conclusion']:
            p_val = details['p_value']
            alpha = details['significance_level']
            conclusion = details['conclusion'].lower()
            
            if p_val < alpha and 'fail to reject' in conclusion:
                issues.append("P-value < significance level but conclusion says 'fail to reject'")
                details['interpretation_consistency'] = 'inconsistent'
            elif p_val >= alpha and 'reject' in conclusion:
                issues.append("P-value >= significance level but conclusion says 'reject'")
                details['interpretation_consistency'] = 'inconsistent'
            else:
                details['interpretation_consistency'] = 'consistent'
        
        return {
            'issues': issues,
            'recommendations': recommendations,
            'details': details
        }
    
    def _calculate_analytical_soundness_score(self, issues: List[str], details: Dict[str, Any]) -> float:
        """Calculate overall analytical soundness score"""
        base_score = 100.0
        
        # Deduct points for issues
        issue_penalty = min(len(issues) * 15, 60)  # Max 60 points deduction
        
        # Check test appropriateness
        test_appropriateness = details.get('test_analysis', {}).get('details', {}).get('test_appropriateness', 'unknown')
        if test_appropriateness == 'inappropriate':
            issue_penalty += 20
        
        # Check interpretation consistency
        interpretation_consistency = details.get('interpretation_analysis', {}).get('details', {}).get('interpretation_consistency', 'unknown')
        if interpretation_consistency == 'inconsistent':
            issue_penalty += 25
        
        score = max(0, base_score - issue_penalty)
        return round(score, 2)


class ReportClarityEvaluator:
    """Evaluates report clarity and insightfulness"""
    
    def evaluate(self, white_output: WhiteAgentOutput) -> EvaluationResult:
        """Main evaluation method for report clarity"""
        issues = []
        recommendations = []
        details = {}
        
        # Evaluate report structure
        structure_issues = self._evaluate_report_structure(white_output.final_report)
        issues.extend(structure_issues['issues'])
        recommendations.extend(structure_issues['recommendations'])
        details['structure_analysis'] = structure_issues['details']
        
        # Evaluate statistical communication
        communication_issues = self._evaluate_statistical_communication(white_output)
        issues.extend(communication_issues['issues'])
        recommendations.extend(communication_issues['recommendations'])
        details['communication_analysis'] = communication_issues['details']
        
        # Evaluate actionable insights
        insights_issues = self._evaluate_actionable_insights(white_output)
        issues.extend(insights_issues['issues'])
        recommendations.extend(insights_issues['recommendations'])
        details['insights_analysis'] = insights_issues['details']
        
        # Calculate overall score
        score = self._calculate_report_clarity_score(issues, details)
        
        return EvaluationResult(
            dimension=EvaluationDimension.REPORT_CLARITY,
            score=score,
            details=details,
            issues=issues,
            recommendations=recommendations
        )
    
    def _evaluate_report_structure(self, report: str) -> Dict[str, Any]:
        """Evaluate the structure and organization of the report"""
        issues = []
        recommendations = []
        details = {
            'word_count': len(report.split()),
            'has_executive_summary': False,
            'has_methodology': False,
            'has_results': False,
            'has_conclusion': False,
            'has_recommendations': False,
            'section_count': 0
        }
        
        report_lower = report.lower()
        
        # Check for key sections
        if 'executive summary' in report_lower or 'summary' in report_lower:
            details['has_executive_summary'] = True
        
        if 'methodology' in report_lower or 'method' in report_lower:
            details['has_methodology'] = True
        
        if 'results' in report_lower or 'findings' in report_lower:
            details['has_results'] = True
        
        if 'conclusion' in report_lower or 'conclusions' in report_lower:
            details['has_conclusion'] = True
        
        if 'recommendation' in report_lower or 'next steps' in report_lower:
            details['has_recommendations'] = True
        
        # Count sections
        details['section_count'] = sum([
            details['has_executive_summary'],
            details['has_methodology'],
            details['has_results'],
            details['has_conclusion'],
            details['has_recommendations']
        ])
        
        # Generate issues and recommendations
        if details['word_count'] < 100:
            issues.append("Report is too short for comprehensive analysis")
        
        if not details['has_executive_summary']:
            recommendations.append("Add an executive summary section")
        
        if not details['has_methodology']:
            recommendations.append("Include methodology section explaining statistical approach")
        
        if not details['has_results']:
            recommendations.append("Add clear results section with key findings")
        
        if not details['has_conclusion']:
            recommendations.append("Include conclusion section with final decision")
        
        if not details['has_recommendations']:
            recommendations.append("Provide actionable recommendations based on results")
        
        return {
            'issues': issues,
            'recommendations': recommendations,
            'details': details
        }
    
    def _evaluate_statistical_communication(self, white_output: WhiteAgentOutput) -> Dict[str, Any]:
        """Evaluate how statistical concepts are communicated"""
        issues = []
        recommendations = []
        details = {
            'explains_p_value': False,
            'explains_confidence_interval': False,
            'explains_significance_level': False,
            'uses_plain_language': False,
            'includes_visualizations': False
        }
        
        report = white_output.final_report.lower()
        
        # Check for statistical concept explanations
        if 'p-value' in report and ('probability' in report or 'chance' in report):
            details['explains_p_value'] = True
        
        if 'confidence interval' in report and ('range' in report or 'between' in report):
            details['explains_confidence_interval'] = True
        
        if 'significance level' in report and ('alpha' in report or '0.05' in report):
            details['explains_significance_level'] = True
        
        # Check for plain language usage
        plain_language_indicators = ['we found', 'the results show', 'this means', 'in other words']
        if any(indicator in report for indicator in plain_language_indicators):
            details['uses_plain_language'] = True
        
        # Check for visualization mentions
        visualization_indicators = ['chart', 'graph', 'plot', 'figure', 'table']
        if any(indicator in report for indicator in visualization_indicators):
            details['includes_visualizations'] = True
        
        # Generate issues and recommendations
        if not details['explains_p_value']:
            recommendations.append("Explain what p-value means in plain language")
        
        if not details['explains_confidence_interval']:
            recommendations.append("Explain confidence intervals in accessible terms")
        
        if not details['uses_plain_language']:
            recommendations.append("Use more plain language to explain statistical concepts")
        
        if not details['includes_visualizations']:
            recommendations.append("Consider including charts or tables to illustrate results")
        
        return {
            'issues': issues,
            'recommendations': recommendations,
            'details': details
        }
    
    def _evaluate_actionable_insights(self, white_output: WhiteAgentOutput) -> Dict[str, Any]:
        """Evaluate the quality of actionable insights"""
        issues = []
        recommendations = []
        details = {
            'has_business_impact': False,
            'has_segmentation_insights': False,
            'has_next_steps': False,
            'has_limitations': False,
            'insight_count': 0
        }
        
        report = white_output.final_report.lower()
        
        # Check for business impact
        business_indicators = ['revenue', 'conversion', 'engagement', 'retention', 'satisfaction']
        if any(indicator in report for indicator in business_indicators):
            details['has_business_impact'] = True
        
        # Check for segmentation insights
        segmentation_indicators = ['segment', 'subgroup', 'demographic', 'cohort']
        if any(indicator in report for indicator in segmentation_indicators):
            details['has_segmentation_insights'] = True
        
        # Check for next steps
        next_steps_indicators = ['next', 'future', 'recommend', 'suggest', 'should']
        if any(indicator in report for indicator in next_steps_indicators):
            details['has_next_steps'] = True
        
        # Check for limitations
        limitation_indicators = ['limitation', 'caveat', 'note that', 'however', 'but']
        if any(indicator in report for indicator in limitation_indicators):
            details['has_limitations'] = True
        
        # Count insights
        details['insight_count'] = sum([
            details['has_business_impact'],
            details['has_segmentation_insights'],
            details['has_next_steps'],
            details['has_limitations']
        ])
        
        # Generate issues and recommendations
        if not details['has_business_impact']:
            recommendations.append("Connect results to business impact and KPIs")
        
        if not details['has_segmentation_insights']:
            recommendations.append("Provide insights about different user segments")
        
        if not details['has_next_steps']:
            recommendations.append("Include clear next steps and recommendations")
        
        if not details['has_limitations']:
            recommendations.append("Acknowledge limitations and caveats of the analysis")
        
        return {
            'issues': issues,
            'recommendations': recommendations,
            'details': details
        }
    
    def _calculate_report_clarity_score(self, issues: List[str], details: Dict[str, Any]) -> float:
        """Calculate overall report clarity score"""
        base_score = 100.0
        
        # Deduct points for issues
        issue_penalty = min(len(issues) * 8, 40)  # Max 40 points deduction
        
        # Check structure completeness
        structure_details = details.get('structure_analysis', {}).get('details', {})
        if structure_details.get('section_count', 0) < 3:
            issue_penalty += 15
        
        # Check word count
        if structure_details.get('word_count', 0) < 100:
            issue_penalty += 20
        
        # Check insights quality
        insights_details = details.get('insights_analysis', {}).get('details', {})
        insight_count = insights_details.get('insight_count', 0)
        if insight_count < 2:
            issue_penalty += 10
        
        score = max(0, base_score - issue_penalty)
        return round(score, 2)


class GreenAgent:
    """Main Green Agent class that orchestrates the evaluation process"""
    
    def __init__(self):
        self.code_quality_evaluator = CodeQualityEvaluator()
        self.analytical_soundness_evaluator = AnalyticalSoundnessEvaluator()
        self.report_clarity_evaluator = ReportClarityEvaluator()
        logger.info("Green Agent initialized successfully")
    
    def evaluate_white_agent(self, white_output: WhiteAgentOutput) -> Dict[str, Any]:
        """
        Main evaluation method that evaluates the White Agent's output
        
        Args:
            white_output: WhiteAgentOutput containing all phases and results
            
        Returns:
            Dictionary containing comprehensive evaluation results
        """
        logger.info("Starting evaluation of White Agent output")
        
        # Run evaluations for all three dimensions
        code_quality_result = self.code_quality_evaluator.evaluate(white_output)
        analytical_soundness_result = self.analytical_soundness_evaluator.evaluate(white_output)
        report_clarity_result = self.report_clarity_evaluator.evaluate(white_output)
        
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
                    'details': code_quality_result.details
                },
                'analytical_soundness': {
                    'score': analytical_soundness_result.score,
                    'issues': analytical_soundness_result.issues,
                    'recommendations': analytical_soundness_result.recommendations,
                    'details': analytical_soundness_result.details
                },
                'report_clarity': {
                    'score': report_clarity_result.score,
                    'issues': report_clarity_result.issues,
                    'recommendations': report_clarity_result.recommendations,
                    'details': report_clarity_result.details
                }
            },
            'summary': self._generate_evaluation_summary([
                code_quality_result,
                analytical_soundness_result,
                report_clarity_result
            ])
        }
        
        logger.info(f"Evaluation completed. Overall score: {overall_score}")
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
        report.append("GREEN AGENT EVALUATION REPORT")
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
    green_agent = GreenAgent()
    sample_output = create_sample_white_output()
    
    # Run evaluation
    results = green_agent.evaluate_white_agent(sample_output)
    
    # Generate and print report
    report = green_agent.generate_evaluation_report(results)
    print(report)
    
    # Save results to JSON
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nEvaluation results saved to 'evaluation_results.json'")
