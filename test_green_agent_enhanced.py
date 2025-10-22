"""
Enhanced Test Suite for Green Agent functionality

This test suite includes comprehensive tests for both LLM-based and rule-based evaluation,
including edge cases, error handling, and integration tests.
"""

import sys
import traceback
import json
from unittest.mock import Mock, patch, MagicMock
import os
from green_agent import GreenAgent, create_sample_white_output, WhiteAgentOutput, EvaluationDimension

# Optional pytest import
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False


class TestGreenAgentBasic:
    """Test basic Green Agent functionality"""
    
    def test_initialization_with_llm(self):
        """Test Green Agent initialization with LLM"""
        with patch('green_agent.LLM_AVAILABLE', True):
            with patch('green_agent.LLMEvaluator') as mock_llm:
                mock_llm.return_value = Mock()
                agent = GreenAgent(use_llm=True, api_key="test_key")
                assert agent.use_llm is True
                mock_llm.assert_called_once()
    
    def test_initialization_without_llm(self):
        """Test Green Agent initialization without LLM"""
        with patch('green_agent.LLM_AVAILABLE', False):
            agent = GreenAgent(use_llm=False)
            assert agent.use_llm is False
            assert hasattr(agent, 'code_quality_evaluator')
            assert hasattr(agent, 'analytical_soundness_evaluator')
            assert hasattr(agent, 'report_clarity_evaluator')
    
    def test_llm_fallback_to_rule_based(self):
        """Test fallback from LLM to rule-based when LLM fails"""
        with patch('green_agent.LLM_AVAILABLE', True):
            with patch('green_agent.LLMEvaluator') as mock_llm:
                mock_llm.side_effect = Exception("API Error")
                agent = GreenAgent(use_llm=True)
                assert agent.use_llm is False
                assert hasattr(agent, 'code_quality_evaluator')


class TestEvaluationResults:
    """Test evaluation result structure and content"""
    
    def test_evaluation_result_structure(self):
        """Test that evaluation results have correct structure"""
        agent = GreenAgent(use_llm=False)
        sample_output = create_sample_white_output()
        results = agent.evaluate_white_agent(sample_output)
        
        # Check top-level structure
        assert 'overall_score' in results
        assert 'evaluation_method' in results
        assert 'dimension_scores' in results
        assert 'evaluations' in results
        assert 'summary' in results
        
        # Check dimension scores
        dimensions = ['code_quality', 'analytical_soundness', 'report_clarity']
        for dim in dimensions:
            assert dim in results['dimension_scores']
            assert 0 <= results['dimension_scores'][dim] <= 100
        
        # Check evaluations structure
        for dim in dimensions:
            assert dim in results['evaluations']
            eval_data = results['evaluations'][dim]
            assert 'score' in eval_data
            assert 'issues' in eval_data
            assert 'recommendations' in eval_data
            assert 'details' in eval_data
            assert 'reasoning' in eval_data
    
    def test_score_ranges(self):
        """Test that all scores are within valid ranges"""
        agent = GreenAgent(use_llm=False)
        sample_output = create_sample_white_output()
        results = agent.evaluate_white_agent(sample_output)
        
        # Check overall score
        assert 0 <= results['overall_score'] <= 100
        
        # Check dimension scores
        for score in results['dimension_scores'].values():
            assert 0 <= score <= 100


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_white_output(self):
        """Test evaluation with minimal/empty white output"""
        agent = GreenAgent(use_llm=False)
        empty_output = WhiteAgentOutput(
            think_phase={},
            plan_phase={},
            act_phase={},
            measure_phase={},
            code_snippets=[],
            final_report=""
        )
        
        results = agent.evaluate_white_agent(empty_output)
        assert 'overall_score' in results
        assert isinstance(results['overall_score'], (int, float))
    
    def test_invalid_code_snippets(self):
        """Test evaluation with invalid code snippets"""
        agent = GreenAgent(use_llm=False)
        invalid_output = WhiteAgentOutput(
            think_phase={'significance_level': 0.05},
            plan_phase={'statistical_test': 't-test'},
            act_phase={'p_value': 0.03},
            measure_phase={'conclusion': 'Significant'},
            code_snippets=['invalid syntax here !!!', 'another invalid line'],
            final_report="Short report"
        )
        
        results = agent.evaluate_white_agent(invalid_output)
        assert results['evaluations']['code_quality']['issues']  # Should have issues
    
    def test_invalid_statistical_values(self):
        """Test evaluation with invalid statistical values"""
        agent = GreenAgent(use_llm=False)
        invalid_output = WhiteAgentOutput(
            think_phase={'significance_level': 0.05},
            plan_phase={'statistical_test': 't-test', 'sample_size': -100},
            act_phase={'p_value': 1.5, 'confidence_interval': [0.8, 0.2]},  # Invalid values
            measure_phase={'conclusion': 'Significant'},
            code_snippets=['import numpy as np'],
            final_report="Report with invalid stats"
        )
        
        results = agent.evaluate_white_agent(invalid_output)
        # Should detect statistical issues
        assert len(results['evaluations']['code_quality']['issues']) > 0
    
    def test_very_long_report(self):
        """Test evaluation with very long report"""
        agent = GreenAgent(use_llm=False)
        long_report = "This is a test report. " * 1000  # Very long report
        
        long_output = WhiteAgentOutput(
            think_phase={'significance_level': 0.05},
            plan_phase={'statistical_test': 't-test'},
            act_phase={'p_value': 0.03},
            measure_phase={'conclusion': 'Significant'},
            code_snippets=['import numpy as np'],
            final_report=long_report
        )
        
        results = agent.evaluate_white_agent(long_output)
        assert 'overall_score' in results
    
    def test_missing_phases(self):
        """Test evaluation with missing phase data"""
        agent = GreenAgent(use_llm=False)
        incomplete_output = WhiteAgentOutput(
            think_phase={'significance_level': 0.05},
            plan_phase={},  # Empty plan phase
            act_phase={'p_value': 0.03},
            measure_phase={},  # Empty measure phase
            code_snippets=[],
            final_report="Minimal report"
        )
        
        results = agent.evaluate_white_agent(incomplete_output)
        assert 'overall_score' in results


class TestLLMEvaluation:
    """Test LLM-based evaluation functionality"""
    
    @patch('green_agent.LLM_AVAILABLE', True)
    @patch('green_agent.OpenAI')
    def test_llm_evaluation_success(self, mock_openai):
        """Test successful LLM evaluation"""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "score": 85,
            "issues": ["Minor issue found"],
            "recommendations": ["Good recommendation"],
            "reasoning": "The analysis is generally good with minor issues",
            "details": {"test_quality": "good"}
        })
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        agent = GreenAgent(use_llm=True, api_key="test_key")
        sample_output = create_sample_white_output()
        results = agent.evaluate_white_agent(sample_output)
        
        assert results['evaluation_method'] == 'LLM'
        assert results['overall_score'] > 0
    
    @patch('green_agent.LLM_AVAILABLE', True)
    @patch('green_agent.OpenAI')
    def test_llm_evaluation_failure(self, mock_openai):
        """Test LLM evaluation failure handling"""
        # Mock OpenAI to raise exception
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        agent = GreenAgent(use_llm=True, api_key="test_key")
        sample_output = create_sample_white_output()
        results = agent.evaluate_white_agent(sample_output)
        
        # Should still return results with fallback scores
        assert 'overall_score' in results
        assert results['evaluation_method'] == 'LLM'
    
    @patch('green_agent.LLM_AVAILABLE', True)
    @patch('green_agent.OpenAI')
    def test_llm_json_parsing_error(self, mock_openai):
        """Test handling of malformed JSON response from LLM"""
        # Mock OpenAI response with invalid JSON
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is not valid JSON"
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        agent = GreenAgent(use_llm=True, api_key="test_key")
        sample_output = create_sample_white_output()
        results = agent.evaluate_white_agent(sample_output)
        
        # Should handle gracefully with fallback
        assert 'overall_score' in results


class TestReportGeneration:
    """Test report generation functionality"""
    
    def test_report_generation(self):
        """Test that evaluation report is generated correctly"""
        agent = GreenAgent(use_llm=False)
        sample_output = create_sample_white_output()
        results = agent.evaluate_white_agent(sample_output)
        
        report = agent.generate_evaluation_report(results)
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "GREEN AGENT EVALUATION REPORT" in report
        assert "OVERALL SUMMARY" in report
        assert "DIMENSION SCORES" in report
    
    def test_report_includes_evaluation_method(self):
        """Test that report includes evaluation method"""
        agent = GreenAgent(use_llm=False)
        sample_output = create_sample_white_output()
        results = agent.evaluate_white_agent(sample_output)
        
        report = agent.generate_evaluation_report(results)
        
        assert "Evaluation Method:" in report
        assert "Rule-based" in report or "LLM" in report


class TestStatisticalValidation:
    """Test statistical validation functionality"""
    
    def test_p_value_validation(self):
        """Test p-value validation"""
        agent = GreenAgent(use_llm=False)
        
        # Test valid p-value
        valid_output = WhiteAgentOutput(
            think_phase={'significance_level': 0.05},
            plan_phase={'statistical_test': 't-test'},
            act_phase={'p_value': 0.03},
            measure_phase={'conclusion': 'Significant'},
            code_snippets=['import numpy as np'],
            final_report="Valid p-value test"
        )
        
        results = agent.evaluate_white_agent(valid_output)
        assert results['overall_score'] > 0
        
        # Test invalid p-value
        invalid_output = WhiteAgentOutput(
            think_phase={'significance_level': 0.05},
            plan_phase={'statistical_test': 't-test'},
            act_phase={'p_value': 1.5},  # Invalid p-value
            measure_phase={'conclusion': 'Significant'},
            code_snippets=['import numpy as np'],
            final_report="Invalid p-value test"
        )
        
        results = agent.evaluate_white_agent(invalid_output)
        # Should have issues with invalid p-value
        assert len(results['evaluations']['code_quality']['issues']) > 0
    
    def test_confidence_interval_validation(self):
        """Test confidence interval validation"""
        agent = GreenAgent(use_llm=False)
        
        # Test valid confidence interval
        valid_output = WhiteAgentOutput(
            think_phase={'significance_level': 0.05},
            plan_phase={'statistical_test': 't-test'},
            act_phase={'confidence_interval': [0.02, 0.08]},
            measure_phase={'conclusion': 'Significant'},
            code_snippets=['import numpy as np'],
            final_report="Valid CI test"
        )
        
        results = agent.evaluate_white_agent(valid_output)
        assert results['overall_score'] > 0
        
        # Test invalid confidence interval
        invalid_output = WhiteAgentOutput(
            think_phase={'significance_level': 0.05},
            plan_phase={'statistical_test': 't-test'},
            act_phase={'confidence_interval': [0.8, 0.2]},  # Invalid CI
            measure_phase={'conclusion': 'Significant'},
            code_snippets=['import numpy as np'],
            final_report="Invalid CI test"
        )
        
        results = agent.evaluate_white_agent(invalid_output)
        # Should have issues with invalid CI
        assert len(results['evaluations']['code_quality']['issues']) > 0


class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_evaluation(self):
        """Test complete end-to-end evaluation process"""
        agent = GreenAgent(use_llm=False)
        sample_output = create_sample_white_output()
        
        # Run evaluation
        results = agent.evaluate_white_agent(sample_output)
        
        # Generate report
        report = agent.generate_evaluation_report(results)
        
        # Verify complete workflow
        assert isinstance(results, dict)
        assert isinstance(report, str)
        assert results['overall_score'] > 0
        assert len(report) > 100
    
    def test_json_serialization(self):
        """Test that results can be serialized to JSON"""
        agent = GreenAgent(use_llm=False)
        sample_output = create_sample_white_output()
        results = agent.evaluate_white_agent(sample_output)
        
        # Should be able to serialize to JSON
        json_str = json.dumps(results)
        assert isinstance(json_str, str)
        
        # Should be able to deserialize
        deserialized = json.loads(json_str)
        assert deserialized['overall_score'] == results['overall_score']


class TestPerformance:
    """Performance and stress tests"""
    
    def test_large_code_snippets(self):
        """Test evaluation with large code snippets"""
        agent = GreenAgent(use_llm=False)
        
        # Create large code snippet
        large_snippet = """
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

# Large statistical analysis
def perform_analysis(data):
    results = {}
    for i in range(100):
        subset = data[i:i+10]
        mean_val = np.mean(subset)
        std_val = np.std(subset)
        results[f'iteration_{i}'] = {'mean': mean_val, 'std': std_val}
    
    # Perform t-test
    t_stat, p_val = stats.ttest_1samp(data, 0)
    results['t_test'] = {'statistic': t_stat, 'p_value': p_val}
    
    # Calculate confidence interval
    ci = proportion_confint(len(data[data > 0]), len(data), alpha=0.05)
    results['confidence_interval'] = ci
    
    return results

# Execute analysis
data = np.random.normal(0, 1, 1000)
analysis_results = perform_analysis(data)
print(f"Analysis complete: {len(analysis_results)} results generated")
""" * 5  # Repeat 5 times to make it large
        
        large_output = WhiteAgentOutput(
            think_phase={'significance_level': 0.05},
            plan_phase={'statistical_test': 't-test'},
            act_phase={'p_value': 0.03},
            measure_phase={'conclusion': 'Significant'},
            code_snippets=[large_snippet],
            final_report="Report with large code snippet"
        )
        
        results = agent.evaluate_white_agent(large_output)
        assert 'overall_score' in results
    
    def test_multiple_evaluations(self):
        """Test multiple evaluations in sequence"""
        agent = GreenAgent(use_llm=False)
        
        # Run multiple evaluations
        for i in range(5):
            sample_output = create_sample_white_output()
            results = agent.evaluate_white_agent(sample_output)
            assert 'overall_score' in results
            assert 0 <= results['overall_score'] <= 100


def run_all_tests():
    """Run all tests and return results"""
    print("=" * 60)
    print("ENHANCED GREEN AGENT TEST SUITE")
    print("=" * 60)
    
    test_classes = [
        TestGreenAgentBasic,
        TestEvaluationResults,
        TestEdgeCases,
        TestLLMEvaluation,
        TestReportGeneration,
        TestStatisticalValidation,
        TestIntegration,
        TestPerformance
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        test_instance = test_class()
        
        # Get all test methods
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                getattr(test_instance, test_method)()
                print(f"  ✅ {test_method}")
                passed_tests += 1
            except Exception as e:
                print(f"  ❌ {test_method}: {str(e)}")
                traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed_tests}/{total_tests} tests passed")
    print("=" * 60)
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Enhanced Green Agent is ready for use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
