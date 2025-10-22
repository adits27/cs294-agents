#!/usr/bin/env python3
"""
Setup script for Green Agent LLM

This script helps you configure your OpenAI API key for the Green Agent.
"""

import os
import sys


def setup_api_key():
    """Interactive setup for OpenAI API key"""
    print("=" * 60)
    print("GREEN AGENT LLM SETUP")
    print("=" * 60)
    
    # Check if API key already exists
    existing_key = os.getenv("OPENAI_API_KEY")
    if existing_key:
        print(f"✅ OpenAI API key already set: {existing_key[:8]}...")
        choice = input("Do you want to update it? (y/n): ").lower().strip()
        if choice != 'y':
            print("Keeping existing API key.")
            return 0
    
    print("\nTo use the Green Agent with LLM, you need a Google API key.")
    print("Get your API key from: https://makersuite.google.com/app/apikey")
    
    # Get API key from user
    api_key = input("\nEnter your Google API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Setup cancelled.")
        return 1
    
    # Validate API key format (basic check)
    if not api_key.startswith('AI'):
        print("⚠️  Warning: API key doesn't start with 'AI'. Please verify it's correct.")
        confirm = input("Continue anyway? (y/n): ").lower().strip()
        if confirm != 'y':
            print("Setup cancelled.")
            return 1
    
    # Create .env file
    env_content = f"""# Google Gemini API Configuration
GOOGLE_API_KEY={api_key}
GOOGLE_MODEL=gemini-1.5-flash

# Logging Configuration
LOG_LEVEL=INFO
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ API key saved to .env file")
    except Exception as e:
        print(f"❌ Error saving .env file: {str(e)}")
        print("\nYou can manually set the environment variable:")
        print(f"export GOOGLE_API_KEY='{api_key}'")
        return 1
    
    # Test the setup
    print("\n🧪 Testing setup...")
    try:
        os.environ["GOOGLE_API_KEY"] = api_key
        from green_agent import GreenAgent
        
        # Try to initialize (this will test the API key)
        agent = GreenAgent(api_key=api_key)
        print("✅ Setup successful! Green Agent is ready to use.")
        
        print("\n📝 Next steps:")
        print("1. Run the example: python example_llm_usage.py")
        print("2. Or run tests: python test_green_agent.py")
        
        return 0
        
    except Exception as e:
        print(f"❌ Setup test failed: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Verify your API key is correct")
        print("2. Check that you have sufficient API credits")
        print("3. Ensure you have internet connectivity")
        print("4. Verify the API key at: https://makersuite.google.com/app/apikey")
        return 1


def show_usage():
    """Show usage instructions"""
    print("=" * 60)
    print("GREEN AGENT LLM USAGE")
    print("=" * 60)
    
    print("\n1. Setup (first time only):")
    print("   python setup.py")
    
    print("\n2. Run example:")
    print("   python example_llm_usage.py")
    
    print("\n3. Run tests:")
    print("   python test_green_agent.py")
    
    print("\n4. Use in your code:")
    print("""
from green_agent import GreenAgent, WhiteAgentOutput

# Initialize agent
agent = GreenAgent()  # Uses API key from .env file

# Create your white agent output
white_output = WhiteAgentOutput(
    think_phase={'significance_level': 0.05},
    plan_phase={'statistical_test': 't-test'},
    act_phase={'p_value': 0.03},
    measure_phase={'conclusion': 'Significant'},
    code_snippets=['import numpy as np'],
    final_report="Your analysis report here"
)

# Run evaluation
results = agent.evaluate_white_agent(white_output)
print(f"Score: {results['overall_score']}")
""")


def main():
    """Main setup function"""
    if len(sys.argv) > 1 and sys.argv[1] == "usage":
        show_usage()
        return 0
    
    return setup_api_key()


if __name__ == "__main__":
    sys.exit(main())
