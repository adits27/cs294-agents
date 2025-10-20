"""
Test script for Green Agent functionality
"""

import sys
import traceback
from green_agent import GreenAgent, create_sample_white_output


def test_basic_functionality():
    """Test basic Green Agent functionality"""
    print("Testing basic Green Agent functionality...")
    
    try:
        # Initialize Green Agent
        green_agent = GreenAgent()
        print("✅ Green Agent initialized successfully")
        
        # Create sample output
        sample_output = create_sample_white_output()
        print("✅ Sample White Agent output created")
        
        # Run evaluation
        results = green_agent.evaluate_white_agent(sample_output)
        print("✅ Evaluation completed successfully")
        
        # Check results structure
        assert 'overall_score' in results
        assert 'dimension_scores' in results
        assert 'evaluations' in results
        assert 'summary' in results
        print("✅ Results structure is correct")
        
        # Check dimension scores
        dimensions = ['code_quality', 'analytical_soundness', 'report_clarity']
        for dim in dimensions:
            assert dim in results['dimension_scores']
            assert 0 <= results['dimension_scores'][dim] <= 100
        print("✅ Dimension scores are valid")
        
        # Generate report
        report = green_agent.generate_evaluation_report(results)
        assert len(report) > 0
        print("✅ Evaluation report generated successfully")
        
        print(f"\nOverall Score: {results['overall_score']}")
        print(f"Assessment: {results['summary']['overall_assessment']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_evaluation_dimensions():
    """Test individual evaluation dimensions"""
    print("\nTesting individual evaluation dimensions...")
    
    try:
        green_agent = GreenAgent()
        sample_output = create_sample_white_output()
        
        # Test Code Quality Evaluator
        code_result = green_agent.code_quality_evaluator.evaluate(sample_output)
        assert hasattr(code_result, 'score')
        assert hasattr(code_result, 'issues')
        assert hasattr(code_result, 'recommendations')
        print("✅ Code Quality Evaluator working")
        
        # Test Analytical Soundness Evaluator
        analytical_result = green_agent.analytical_soundness_evaluator.evaluate(sample_output)
        assert hasattr(analytical_result, 'score')
        assert hasattr(analytical_result, 'issues')
        assert hasattr(analytical_result, 'recommendations')
        print("✅ Analytical Soundness Evaluator working")
        
        # Test Report Clarity Evaluator
        report_result = green_agent.report_clarity_evaluator.evaluate(sample_output)
        assert hasattr(report_result, 'score')
        assert hasattr(report_result, 'issues')
        assert hasattr(report_result, 'recommendations')
        print("✅ Report Clarity Evaluator working")
        
        return True
        
    except Exception as e:
        print(f"❌ Dimension test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_scoring_system():
    """Test the scoring system"""
    print("\nTesting scoring system...")
    
    try:
        green_agent = GreenAgent()
        sample_output = create_sample_white_output()
        
        results = green_agent.evaluate_white_agent(sample_output)
        
        # Check overall score calculation
        overall_score = results['overall_score']
        assert 0 <= overall_score <= 100
        print(f"✅ Overall score is valid: {overall_score}")
        
        # Check individual dimension scores
        for dim, score in results['dimension_scores'].items():
            assert 0 <= score <= 100
            print(f"✅ {dim} score is valid: {score}")
        
        # Check assessment
        assessment = results['summary']['overall_assessment']
        valid_assessments = ['Excellent', 'Good', 'Satisfactory', 'Needs Improvement', 'Poor']
        assert assessment in valid_assessments
        print(f"✅ Assessment is valid: {assessment}")
        
        return True
        
    except Exception as e:
        print(f"❌ Scoring test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling with invalid inputs"""
    print("\nTesting error handling...")
    
    try:
        green_agent = GreenAgent()
        
        # Test with minimal valid input
        from green_agent import WhiteAgentOutput
        minimal_output = WhiteAgentOutput(
            think_phase={},
            plan_phase={},
            act_phase={},
            measure_phase={},
            code_snippets=[],
            final_report=""
        )
        
        results = green_agent.evaluate_white_agent(minimal_output)
        assert 'overall_score' in results
        print("✅ Handles minimal input gracefully")
        
        # Test with invalid code snippets
        invalid_output = WhiteAgentOutput(
            think_phase={'significance_level': 0.05},
            plan_phase={'statistical_test': 't-test'},
            act_phase={'p_value': 0.03},
            measure_phase={'conclusion': 'Significant'},
            code_snippets=['invalid syntax here !!!'],
            final_report="Short report"
        )
        
        results = green_agent.evaluate_white_agent(invalid_output)
        assert results['evaluations']['code_quality']['issues']  # Should have issues
        print("✅ Detects code syntax errors")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("GREEN AGENT TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_basic_functionality,
        test_evaluation_dimensions,
        test_scoring_system,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! Green Agent is ready for use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
