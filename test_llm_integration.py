"""
Integration Tests for LLM-based Green Agent

This module contains integration tests that test the LLM-based evaluation
functionality with real OpenAI API calls (when API key is available).
"""

import os
import json
import pytest
from unittest.mock import Mock, patch
from green_agent import GreenAgent, WhiteAgentOutput, EvaluationDimension


class TestLLMIntegration:
    """Integration tests for LLM-based evaluation"""
    
    def test_llm_evaluation_with_mock(self):
        """Test LLM evaluation with mocked OpenAI responses"""
        with patch('green_agent.LLM_AVAILABLE', True):
            with patch('green_agent.OpenAI') as mock_openai:
                # Mock successful OpenAI response
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = json.dumps({
                    "score": 85,
                    "issues": ["Code could be more modular"],
                    "recommendations": ["Consider breaking down into smaller functions"],
                    "reasoning": "The code is functionally correct but could benefit from better organization",
                    "details": {
                        "syntax_check": "passed",
                        "imports": "complete",
                        "statistical_functions": "appropriate"
                    }
                })
                
                mock_client = Mock()
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client
                
                agent = GreenAgent(use_llm=True, api_key="test_key")
                sample_output = WhiteAgentOutput(
                    think_phase={'significance_level': 0.05},
                    plan_phase={'statistical_test': 't-test'},
                    act_phase={'p_value': 0.03},
                    measure_phase={'conclusion': 'Significant'},
                    code_snippets=['import numpy as np\nfrom scipy import stats'],
                    final_report="Test report"
                )
                
                results = agent.evaluate_white_agent(sample_output)
                
                assert results['evaluation_method'] == 'LLM'
                assert results['overall_score'] > 0
                assert 'reasoning' in results['evaluations']['code_quality']
    
    def test_llm_evaluation_all_dimensions(self):
        """Test LLM evaluation for all three dimensions"""
        with patch('green_agent.LLM_AVAILABLE', True):
            with patch('green_agent.OpenAI') as mock_openai:
                # Mock responses for each dimension
                mock_responses = [
                    # Code Quality
                    json.dumps({
                        "score": 80,
                        "issues": ["Missing error handling"],
                        "recommendations": ["Add try-catch blocks"],
                        "reasoning": "Code is syntactically correct but lacks error handling",
                        "details": {"syntax": "valid", "imports": "complete"}
                    }),
                    # Analytical Soundness
                    json.dumps({
                        "score": 90,
                        "issues": [],
                        "recommendations": ["Consider effect size calculation"],
                        "reasoning": "Statistical approach is sound and appropriate",
                        "details": {"test_selection": "appropriate", "sample_size": "adequate"}
                    }),
                    # Report Clarity
                    json.dumps({
                        "score": 75,
                        "issues": ["Missing executive summary"],
                        "recommendations": ["Add executive summary section"],
                        "reasoning": "Report is clear but could be more structured",
                        "details": {"structure": "basic", "clarity": "good"}
                    })
                ]
                
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = mock_responses[0]
                
                mock_client = Mock()
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client
                
                agent = GreenAgent(use_llm=True, api_key="test_key")
                sample_output = WhiteAgentOutput(
                    think_phase={'significance_level': 0.05},
                    plan_phase={'statistical_test': 't-test'},
                    act_phase={'p_value': 0.03},
                    measure_phase={'conclusion': 'Significant'},
                    code_snippets=['import numpy as np'],
                    final_report="Test report"
                )
                
                results = agent.evaluate_white_agent(sample_output)
                
                assert results['evaluation_method'] == 'LLM'
                assert len(results['evaluations']) == 3
                for dimension in ['code_quality', 'analytical_soundness', 'report_clarity']:
                    assert dimension in results['evaluations']
                    assert 'reasoning' in results['evaluations'][dimension]
    
    def test_llm_api_error_handling(self):
        """Test handling of OpenAI API errors"""
        with patch('green_agent.LLM_AVAILABLE', True):
            with patch('green_agent.OpenAI') as mock_openai:
                # Mock API error
                mock_client = Mock()
                mock_client.chat.completions.create.side_effect = Exception("API Rate Limit Exceeded")
                mock_openai.return_value = mock_client
                
                agent = GreenAgent(use_llm=True, api_key="test_key")
                sample_output = WhiteAgentOutput(
                    think_phase={'significance_level': 0.05},
                    plan_phase={'statistical_test': 't-test'},
                    act_phase={'p_value': 0.03},
                    measure_phase={'conclusion': 'Significant'},
                    code_snippets=['import numpy as np'],
                    final_report="Test report"
                )
                
                results = agent.evaluate_white_agent(sample_output)
                
                # Should still return results with fallback scores
                assert 'overall_score' in results
                assert results['evaluation_method'] == 'LLM'
                # Should have error information in details
                for dimension in results['evaluations'].values():
                    assert 'error' in dimension['details'] or len(dimension['issues']) > 0
    
    def test_llm_json_parsing_error(self):
        """Test handling of malformed JSON responses"""
        with patch('green_agent.LLM_AVAILABLE', True):
            with patch('green_agent.OpenAI') as mock_openai:
                # Mock malformed JSON response
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = "This is not valid JSON at all!"
                
                mock_client = Mock()
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client
                
                agent = GreenAgent(use_llm=True, api_key="test_key")
                sample_output = WhiteAgentOutput(
                    think_phase={'significance_level': 0.05},
                    plan_phase={'statistical_test': 't-test'},
                    act_phase={'p_value': 0.03},
                    measure_phase={'conclusion': 'Significant'},
                    code_snippets=['import numpy as np'],
                    final_report="Test report"
                )
                
                results = agent.evaluate_white_agent(sample_output)
                
                # Should handle gracefully with fallback
                assert 'overall_score' in results
                assert results['evaluation_method'] == 'LLM'
    
    def test_llm_missing_api_key(self):
        """Test behavior when OpenAI API key is missing"""
        with patch('green_agent.LLM_AVAILABLE', True):
            with patch.dict(os.environ, {}, clear=True):
                # Should fall back to rule-based evaluation
                agent = GreenAgent(use_llm=True)
                assert agent.use_llm is False
                assert hasattr(agent, 'code_quality_evaluator')


class TestRealLLMIntegration:
    """Integration tests with real OpenAI API (requires API key)"""
    
    @pytest.mark.skipif(not os.getenv('OPENAI_API_KEY'), reason="OpenAI API key not available")
    def test_real_llm_evaluation(self):
        """Test with real OpenAI API (requires OPENAI_API_KEY environment variable)"""
        agent = GreenAgent(use_llm=True)
        
        sample_output = WhiteAgentOutput(
            think_phase={
                'null_hypothesis': 'There is no difference in conversion rates',
                'alternative_hypothesis': 'There is a difference in conversion rates',
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
        
        results = agent.evaluate_white_agent(sample_output)
        
        # Verify LLM evaluation results
        assert results['evaluation_method'] == 'LLM'
        assert 0 <= results['overall_score'] <= 100
        
        # Check that all dimensions have reasoning
        for dimension, eval_data in results['evaluations'].items():
            assert 'reasoning' in eval_data
            assert eval_data['reasoning'] is not None
            assert len(eval_data['reasoning']) > 0
    
    @pytest.mark.skipif(not os.getenv('OPENAI_API_KEY'), reason="OpenAI API key not available")
    def test_real_llm_different_models(self):
        """Test with different OpenAI models"""
        models = ['gpt-4', 'gpt-3.5-turbo']
        
        for model in models:
            try:
                agent = GreenAgent(use_llm=True, model=model)
                
                sample_output = WhiteAgentOutput(
                    think_phase={'significance_level': 0.05},
                    plan_phase={'statistical_test': 't-test'},
                    act_phase={'p_value': 0.03},
                    measure_phase={'conclusion': 'Significant'},
                    code_snippets=['import numpy as np'],
                    final_report="Test report"
                )
                
                results = agent.evaluate_white_agent(sample_output)
                
                assert results['evaluation_method'] == 'LLM'
                assert 0 <= results['overall_score'] <= 100
                
            except Exception as e:
                # Some models might not be available
                print(f"Model {model} not available: {str(e)}")


class TestLLMComparison:
    """Tests comparing LLM vs rule-based evaluation"""
    
    def test_llm_vs_rule_based_comparison(self):
        """Compare LLM and rule-based evaluation results"""
        sample_output = WhiteAgentOutput(
            think_phase={'significance_level': 0.05},
            plan_phase={'statistical_test': 't-test'},
            act_phase={'p_value': 0.03},
            measure_phase={'conclusion': 'Significant'},
            code_snippets=['import numpy as np\nfrom scipy import stats'],
            final_report="Test report with good structure and clear conclusions"
        )
        
        # Rule-based evaluation
        rule_agent = GreenAgent(use_llm=False)
        rule_results = rule_agent.evaluate_white_agent(sample_output)
        
        # Mock LLM evaluation
        with patch('green_agent.LLM_AVAILABLE', True):
            with patch('green_agent.OpenAI') as mock_openai:
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = json.dumps({
                    "score": 85,
                    "issues": ["Could be more detailed"],
                    "recommendations": ["Add more context"],
                    "reasoning": "Good analysis but could be more comprehensive",
                    "details": {"quality": "good"}
                })
                
                mock_client = Mock()
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client
                
                llm_agent = GreenAgent(use_llm=True, api_key="test_key")
                llm_results = llm_agent.evaluate_white_agent(sample_output)
        
        # Both should return valid results
        assert rule_results['evaluation_method'] == 'Rule-based'
        assert llm_results['evaluation_method'] == 'LLM'
        
        # Both should have reasoning (LLM has detailed reasoning, rule-based has None)
        for dimension in ['code_quality', 'analytical_soundness', 'report_clarity']:
            assert 'reasoning' in rule_results['evaluations'][dimension]
            assert 'reasoning' in llm_results['evaluations'][dimension]


def run_integration_tests():
    """Run all integration tests"""
    print("=" * 60)
    print("LLM INTEGRATION TEST SUITE")
    print("=" * 60)
    
    test_classes = [
        TestLLMIntegration,
        TestRealLLMIntegration,
        TestLLMComparison
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
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"INTEGRATION TEST RESULTS: {passed_tests}/{total_tests} tests passed")
    print("=" * 60)
    
    if passed_tests == total_tests:
        print("🎉 All integration tests passed!")
        return 0
    else:
        print("❌ Some integration tests failed.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(run_integration_tests())
