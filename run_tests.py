"""
Comprehensive Test Runner for Green Agent

This module provides a unified test runner that can execute all test suites:
- Basic functionality tests
- Enhanced test suite with edge cases
- LLM integration tests
- Mock tests for LLM functionality
"""

import sys
import os
import argparse
from typing import List, Tuple


def run_basic_tests() -> Tuple[int, int]:
    """Run basic functionality tests"""
    print("=" * 60)
    print("RUNNING BASIC FUNCTIONALITY TESTS")
    print("=" * 60)
    
    try:
        from test_green_agent import main as basic_main
        result = basic_main()
        return (1, 0) if result == 0 else (0, 1)
    except Exception as e:
        print(f"Error running basic tests: {str(e)}")
        return (0, 1)


def run_enhanced_tests() -> Tuple[int, int]:
    """Run enhanced test suite"""
    print("=" * 60)
    print("RUNNING ENHANCED TEST SUITE")
    print("=" * 60)
    
    try:
        from test_green_agent_enhanced import run_all_tests
        result = run_all_tests()
        return (1, 0) if result == 0 else (0, 1)
    except Exception as e:
        print(f"Error running enhanced tests: {str(e)}")
        return (0, 1)


def run_llm_integration_tests() -> Tuple[int, int]:
    """Run LLM integration tests"""
    print("=" * 60)
    print("RUNNING LLM INTEGRATION TESTS")
    print("=" * 60)
    
    try:
        from test_llm_integration import run_integration_tests
        result = run_integration_tests()
        return (1, 0) if result == 0 else (0, 1)
    except Exception as e:
        print(f"Error running LLM integration tests: {str(e)}")
        return (0, 1)


def run_llm_mock_tests() -> Tuple[int, int]:
    """Run LLM mock tests"""
    print("=" * 60)
    print("RUNNING LLM MOCK TESTS")
    print("=" * 60)
    
    try:
        from test_llm_mocks import run_mock_tests
        result = run_mock_tests()
        return (1, 0) if result == 0 else (0, 1)
    except Exception as e:
        print(f"Error running LLM mock tests: {str(e)}")
        return (0, 1)


def run_pytest_tests() -> Tuple[int, int]:
    """Run tests using pytest framework"""
    print("=" * 60)
    print("RUNNING PYTEST TESTS")
    print("=" * 60)
    
    try:
        import pytest
        # Run pytest on all test files
        test_files = [
            'test_green_agent.py',
            'test_green_agent_enhanced.py',
            'test_llm_integration.py',
            'test_llm_mocks.py'
        ]
        
        # Filter to only existing files
        existing_files = [f for f in test_files if os.path.exists(f)]
        
        if not existing_files:
            print("No test files found for pytest")
            return (0, 1)
        
        # Run pytest
        result = pytest.main(['-v', '--tb=short'] + existing_files)
        return (1, 0) if result == 0 else (0, 1)
        
    except ImportError:
        print("pytest not available, skipping pytest tests")
        return (0, 0)
    except Exception as e:
        print(f"Error running pytest tests: {str(e)}")
        return (0, 1)


def check_dependencies() -> bool:
    """Check if all required dependencies are available"""
    print("Checking dependencies...")
    
    required_packages = [
        'numpy',
        'pandas', 
        'scipy',
        'statsmodels',
        'openai',
        'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("All dependencies available!")
    return True


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description='Comprehensive Green Agent Test Runner')
    parser.add_argument('--suite', choices=['basic', 'enhanced', 'llm-integration', 'llm-mocks', 'pytest', 'all'], 
                       default='all', help='Test suite to run')
    parser.add_argument('--check-deps', action='store_true', help='Check dependencies only')
    parser.add_argument('--skip-deps-check', action='store_true', help='Skip dependency check')
    
    args = parser.parse_args()
    
    # Check dependencies unless skipped
    if not args.skip_deps_check:
        if not check_dependencies():
            if not args.check_deps:
                print("\nDependencies check failed. Use --skip-deps-check to run tests anyway.")
                return 1
            else:
                return 1
    
    if args.check_deps:
        return 0
    
    # Run selected test suite
    test_suites = {
        'basic': run_basic_tests,
        'enhanced': run_enhanced_tests,
        'llm-integration': run_llm_integration_tests,
        'llm-mocks': run_llm_mock_tests,
        'pytest': run_pytest_tests
    }
    
    if args.suite == 'all':
        # Run all test suites
        total_passed = 0
        total_failed = 0
        
        for suite_name, suite_func in test_suites.items():
            print(f"\n{'='*20} {suite_name.upper()} {'='*20}")
            passed, failed = suite_func()
            total_passed += passed
            total_failed += failed
        
        print("\n" + "=" * 60)
        print("OVERALL TEST RESULTS")
        print("=" * 60)
        print(f"Test Suites Passed: {total_passed}")
        print(f"Test Suites Failed: {total_failed}")
        print("=" * 60)
        
        if total_failed == 0:
            print("🎉 All test suites passed!")
            return 0
        else:
            print("❌ Some test suites failed.")
            return 1
    else:
        # Run specific test suite
        if args.suite in test_suites:
            passed, failed = test_suites[args.suite]()
            return 0 if failed == 0 else 1
        else:
            print(f"Unknown test suite: {args.suite}")
            return 1


if __name__ == "__main__":
    sys.exit(main())
