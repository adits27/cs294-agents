#!/usr/bin/env python3
"""
Example usage of LLM-only Green Agent

This script demonstrates how to use the Green Agent with OpenAI ChatGPT.
Make sure to set your OPENAI_API_KEY environment variable before running.
"""

import os
from green_agent import GreenAgent, create_sample_white_output


def main():
    """Main example function"""
    print("=" * 60)
    print("GREEN AGENT LLM EXAMPLE")
    print("=" * 60)
    
    # Check if API key is available
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY environment variable not set")
        print("\nTo set your API key:")
        print("1. Get your API key from: https://makersuite.google.com/app/apikey")
        print("2. Set the environment variable:")
        print("   export GOOGLE_API_KEY='your_api_key_here'")
        print("\nOr create a .env file with:")
        print("   GOOGLE_API_KEY=your_api_key_here")
        return 1
    
    try:
        # Initialize Green Agent with LLM
        print("🚀 Initializing Green Agent with LLM...")
        agent = GreenAgent(api_key=api_key)  # Will use model from .env file
        
        # Create sample White Agent output
        print("📊 Creating sample White Agent output...")
        sample_output = create_sample_white_output()
        
        # Run evaluation
        print("🔍 Running LLM-based evaluation...")
        results = agent.evaluate_white_agent(sample_output)
        
        # Display results
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Overall Score: {results['overall_score']}/100")
        print(f"Evaluation Method: {results['evaluation_method']}")
        print(f"Assessment: {results['summary']['overall_assessment']}")
        
        print("\nDimension Scores:")
        for dimension, score in results['dimension_scores'].items():
            print(f"  {dimension.replace('_', ' ').title()}: {score}/100")
        
        print(f"\nTotal Issues Found: {results['summary']['total_issues']}")
        print(f"Total Recommendations: {results['summary']['total_recommendations']}")
        
        # Show LLM reasoning for each dimension
        print("\n" + "=" * 60)
        print("LLM REASONING")
        print("=" * 60)
        
        for dimension, eval_data in results['evaluations'].items():
            print(f"\n{dimension.replace('_', ' ').upper()}:")
            print(f"Score: {eval_data['score']}/100")
            if eval_data.get('reasoning'):
                print(f"Reasoning: {eval_data['reasoning']}")
            
            if eval_data['issues']:
                print("Issues:")
                for i, issue in enumerate(eval_data['issues'], 1):
                    print(f"  {i}. {issue}")
            
            if eval_data['recommendations']:
                print("Recommendations:")
                for i, rec in enumerate(eval_data['recommendations'], 1):
                    print(f"  {i}. {rec}")
        
        # Generate and save full report
        print("\n" + "=" * 60)
        print("GENERATING FULL REPORT")
        print("=" * 60)
        
        report = agent.generate_evaluation_report(results)
        print(report)
        
        # Save results to JSON
        import json
        with open('evaluation_results_llm.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to 'evaluation_results_llm.json'")
        print("🎉 Example completed successfully!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check that your Google API key is valid")
        print("2. Ensure you have sufficient API credits")
        print("3. Check your internet connection")
        print("4. Verify the API key at: https://makersuite.google.com/app/apikey")
        return 1


if __name__ == "__main__":
    exit(main())
