"""
Mock Tests for Green Agent LLM functionality

This module contains tests that mock OpenAI API calls to avoid API costs
during testing while still validating the LLM integration logic.
"""

import json
from unittest.mock import Mock, patch, MagicMock
from green_agent import GreenAgent, WhiteAgentOutput, EvaluationDimension, LLMEvaluator

# Optional pytest import
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False


class TestLLMMocks:
    """Test LLM functionality with mocked OpenAI responses"""
    
    @patch('green_agent.LLM_AVAILABLE', True)
    @patch('green_agent.OpenAI')
    def test_llm_evaluator_initialization(self, mock_openai):
        """Test LLM evaluator initialization"""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        evaluator = LLMEvaluator(api_key="test_key", model="gpt-4")
        
        assert evaluator.model == "gpt-4"
        assert evaluator.api_key == "test_key"
        mock_openai.assert_called_once_with(api_key="test_key")
    
    @patch('green_agent.LLM_AVAILABLE', True)
    @patch('green_agent.OpenAI')
    def test_llm_evaluation_code_quality(self, mock_openai):
        """Test LLM evaluation for code quality dimension"""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "score": 85,
            "issues": [
                "Missing error handling in statistical calculations",
                "Code could be more modular"
            ],
            "recommendations": [
                "Add try-catch blocks around statistical operations",
                "Break down large functions into smaller, focused functions"
            ],
            "reasoning": "The code is syntactically correct and uses appropriate statistical functions, but lacks error handling and could benefit from better organization.",
            "details": {
                "syntax_check": "passed",
                "imports": "complete",
                "statistical_functions": "appropriate",
                "error_handling": "missing",
                "modularity": "poor"
            }
        })
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        evaluator = LLMEvaluator(api_key="test_key")
        sample_output = WhiteAgentOutput(
            think_phase={'significance_level': 0.05},
            plan_phase={'statistical_test': 't-test'},
            act_phase={'p_value': 0.03},
            measure_phase={'conclusion': 'Significant'},
            code_snippets=[
                'import numpy as np\nfrom scipy import stats\n\n# Perform t-test\nt_stat, p_val = stats.ttest_1samp(data, 0)'
            ],
            final_report="Test report"
        )
        
        result = evaluator.evaluate_dimension(sample_output, EvaluationDimension.CODE_QUALITY)
        
        assert result.dimension == EvaluationDimension.CODE_QUALITY
        assert result.score == 85
        assert len(result.issues) == 2
        assert len(result.recommendations) == 2
        assert "error handling" in result.reasoning.lower()
        assert result.details['syntax_check'] == "passed"
    
    @patch('green_agent.LLM_AVAILABLE', True)
    @patch('green_agent.OpenAI')
    def test_llm_evaluation_analytical_soundness(self, mock_openai):
        """Test LLM evaluation for analytical soundness dimension"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "score": 92,
            "issues": [],
            "recommendations": [
                "Consider calculating effect size for practical significance",
                "Document assumptions about data distribution"
            ],
            "reasoning": "The statistical approach is sound and appropriate for the data type. The hypothesis formulation is clear and the test selection is correct.",
            "details": {
                "hypothesis_formulation": "excellent",
                "test_selection": "appropriate",
                "sample_size": "adequate",
                "result_interpretation": "correct"
            }
        })
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        evaluator = LLMEvaluator(api_key="test_key")
        sample_output = WhiteAgentOutput(
            think_phase={
                'null_hypothesis': 'No difference in means',
                'alternative_hypothesis': 'Difference in means',
                'significance_level': 0.05,
                'data_type': 'continuous'
            },
            plan_phase={'statistical_test': 't-test', 'sample_size': 100},
            act_phase={'p_value': 0.03},
            measure_phase={'conclusion': 'Reject null hypothesis'},
            code_snippets=['import numpy as np'],
            final_report="Test report"
        )
        
        result = evaluator.evaluate_dimension(sample_output, EvaluationDimension.ANALYTICAL_SOUNDNESS)
        
        assert result.dimension == EvaluationDimension.ANALYTICAL_SOUNDNESS
        assert result.score == 92
        assert len(result.issues) == 0
        assert len(result.recommendations) == 2
        assert "sound" in result.reasoning.lower()
    
    @patch('green_agent.LLM_AVAILABLE', True)
    @patch('green_agent.OpenAI')
    def test_llm_evaluation_report_clarity(self, mock_openai):
        """Test LLM evaluation for report clarity dimension"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "score": 78,
            "issues": [
                "Missing executive summary",
                "Statistical concepts not explained in plain language"
            ],
            "recommendations": [
                "Add an executive summary at the beginning",
                "Explain p-values and confidence intervals in accessible terms",
                "Include visualizations to illustrate results"
            ],
            "reasoning": "The report has good structure and includes actionable insights, but lacks an executive summary and could better explain statistical concepts to non-technical audiences.",
            "details": {
                "structure": "good",
                "executive_summary": "missing",
                "statistical_explanations": "technical",
                "actionable_insights": "present",
                "visualizations": "missing"
            }
        })
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        evaluator = LLMEvaluator(api_key="test_key")
        sample_output = WhiteAgentOutput(
            think_phase={'significance_level': 0.05},
            plan_phase={'statistical_test': 't-test'},
            act_phase={'p_value': 0.03},
            measure_phase={'conclusion': 'Significant'},
            code_snippets=['import numpy as np'],
            final_report="""Methodology: We used a t-test to compare means.
            Results: p-value was 0.03, indicating statistical significance.
            Conclusion: We recommend implementing the change."""
        )
        
        result = evaluator.evaluate_dimension(sample_output, EvaluationDimension.REPORT_CLARITY)
        
        assert result.dimension == EvaluationDimension.REPORT_CLARITY
        assert result.score == 78
        assert len(result.issues) == 2
        assert len(result.recommendations) == 3
        assert "executive summary" in result.reasoning.lower()
    
    @patch('green_agent.LLM_AVAILABLE', True)
    @patch('green_agent.OpenAI')
    def test_llm_api_error_handling(self, mock_openai):
        """Test handling of OpenAI API errors"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("Rate limit exceeded")
        mock_openai.return_value = mock_client
        
        evaluator = LLMEvaluator(api_key="test_key")
        sample_output = WhiteAgentOutput(
            think_phase={'significance_level': 0.05},
            plan_phase={'statistical_test': 't-test'},
            act_phase={'p_value': 0.03},
            measure_phase={'conclusion': 'Significant'},
            code_snippets=['import numpy as np'],
            final_report="Test report"
        )
        
        result = evaluator.evaluate_dimension(sample_output, EvaluationDimension.CODE_QUALITY)
        
        # Should return fallback result
        assert result.score == 50.0  # Neutral fallback score
        assert "Rate limit exceeded" in result.issues[0]
        assert "error" in result.details
        assert "Rate limit exceeded" in result.reasoning
    
    @patch('green_agent.LLM_AVAILABLE', True)
    @patch('green_agent.OpenAI')
    def test_llm_malformed_json_response(self, mock_openai):
        """Test handling of malformed JSON responses from LLM"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is not valid JSON at all!"
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        evaluator = LLMEvaluator(api_key="test_key")
        sample_output = WhiteAgentOutput(
            think_phase={'significance_level': 0.05},
            plan_phase={'statistical_test': 't-test'},
            act_phase={'p_value': 0.03},
            measure_phase={'conclusion': 'Significant'},
            code_snippets=['import numpy as np'],
            final_report="Test report"
        )
        
        result = evaluator.evaluate_dimension(sample_output, EvaluationDimension.CODE_QUALITY)
        
        # Should return fallback result
        assert result.score == 50.0
        assert any("Failed to parse JSON response" in issue for issue in result.issues)
        assert "error" in result.details
    
    @patch('green_agent.LLM_AVAILABLE', True)
    @patch('green_agent.OpenAI')
    def test_llm_partial_json_response(self, mock_openai):
        """Test handling of partial JSON responses"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"score": 85, "issues": ["test issue"]'  # Missing closing brace
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        evaluator = LLMEvaluator(api_key="test_key")
        sample_output = WhiteAgentOutput(
            think_phase={'significance_level': 0.05},
            plan_phase={'statistical_test': 't-test'},
            act_phase={'p_value': 0.03},
            measure_phase={'conclusion': 'Significant'},
            code_snippets=['import numpy as np'],
            final_report="Test report"
        )
        
        result = evaluator.evaluate_dimension(sample_output, EvaluationDimension.CODE_QUALITY)
        
        # Should return fallback result
        assert result.score == 50.0
        assert any("Failed to parse JSON response" in issue for issue in result.issues)


class TestGreenAgentLLMMocks:
    """Test Green Agent with mocked LLM functionality"""
    
    @patch('green_agent.LLM_AVAILABLE', True)
    @patch('green_agent.OpenAI')
    def test_green_agent_llm_evaluation(self, mock_openai):
        """Test Green Agent with LLM evaluation"""
        # Mock responses for all three dimensions
        mock_responses = [
            # Code Quality
            json.dumps({
                "score": 80,
                "issues": ["Missing error handling"],
                "recommendations": ["Add try-catch blocks"],
                "reasoning": "Code is correct but needs error handling",
                "details": {"syntax": "valid"}
            }),
            # Analytical Soundness
            json.dumps({
                "score": 90,
                "issues": [],
                "recommendations": ["Consider effect size"],
                "reasoning": "Statistical approach is sound",
                "details": {"test_selection": "appropriate"}
            }),
            # Report Clarity
            json.dumps({
                "score": 75,
                "issues": ["Missing summary"],
                "recommendations": ["Add executive summary"],
                "reasoning": "Report is clear but incomplete",
                "details": {"structure": "basic"}
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
        assert 0 <= results['overall_score'] <= 100
        assert len(results['evaluations']) == 3
        
        # Check that all dimensions have reasoning
        for dimension in results['evaluations'].values():
            assert 'reasoning' in dimension
            assert dimension['reasoning'] is not None
    
    @patch('green_agent.LLM_AVAILABLE', True)
    @patch('green_agent.OpenAI')
    def test_green_agent_llm_fallback(self, mock_openai):
        """Test Green Agent fallback when LLM fails"""
        # Mock API error
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
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
        
        # Should still return results
        assert 'overall_score' in results
        assert results['evaluation_method'] == 'LLM'
        
        # Should have error information
        for dimension in results['evaluations'].values():
            assert 'error' in dimension['details'] or len(dimension['issues']) > 0


class TestLLMPromptGeneration:
    """Test LLM prompt generation"""
    
    @patch('green_agent.LLM_AVAILABLE', True)
    @patch('green_agent.OpenAI')
    def test_prompt_creation(self, mock_openai):
        """Test that prompts are created correctly"""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        evaluator = LLMEvaluator(api_key="test_key")
        
        sample_output = WhiteAgentOutput(
            think_phase={'significance_level': 0.05, 'data_type': 'continuous'},
            plan_phase={'statistical_test': 't-test', 'sample_size': 100},
            act_phase={'p_value': 0.03, 'confidence_interval': [0.01, 0.05]},
            measure_phase={'conclusion': 'Reject null hypothesis'},
            code_snippets=['import numpy as np\nfrom scipy import stats'],
            final_report="This is a test report with results."
        )
        
        # Test prompt creation for code quality
        prompt = evaluator._create_evaluation_prompt(sample_output, EvaluationDimension.CODE_QUALITY)
        
        assert "code quality" in prompt.lower()
        assert "think phase" in prompt.lower()
        assert "plan phase" in prompt.lower()
        assert "act phase" in prompt.lower()
        assert "measure phase" in prompt.lower()
        assert "code snippets" in prompt.lower()
        assert "final report" in prompt.lower()
        assert "import numpy as np" in prompt
        assert "This is a test report" in prompt
    
    @patch('green_agent.LLM_AVAILABLE', True)
    @patch('green_agent.OpenAI')
    def test_system_prompt_generation(self, mock_openai):
        """Test system prompt generation for different dimensions"""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        evaluator = LLMEvaluator(api_key="test_key")
        
        # Test code quality system prompt
        code_prompt = evaluator._get_system_prompt(EvaluationDimension.CODE_QUALITY)
        assert "code syntax" in code_prompt.lower()
        assert "statistical functions" in code_prompt.lower()
        assert "import statements" in code_prompt.lower()
        
        # Test analytical soundness system prompt
        analytical_prompt = evaluator._get_system_prompt(EvaluationDimension.ANALYTICAL_SOUNDNESS)
        assert "hypothesis" in analytical_prompt.lower()
        assert "statistical test" in analytical_prompt.lower()
        assert "sample size" in analytical_prompt.lower()
        
        # Test report clarity system prompt
        report_prompt = evaluator._get_system_prompt(EvaluationDimension.REPORT_CLARITY)
        assert "report structure" in report_prompt.lower()
        assert "statistical concepts" in report_prompt.lower()
        assert "actionable insights" in report_prompt.lower()


def run_mock_tests():
    """Run all mock tests"""
    print("=" * 60)
    print("LLM MOCK TEST SUITE")
    print("=" * 60)
    
    test_classes = [
        TestLLMMocks,
        TestGreenAgentLLMMocks,
        TestLLMPromptGeneration
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
    print(f"MOCK TEST RESULTS: {passed_tests}/{total_tests} tests passed")
    print("=" * 60)
    
    if passed_tests == total_tests:
        print("🎉 All mock tests passed!")
        return 0
    else:
        print("❌ Some mock tests failed.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(run_mock_tests())
