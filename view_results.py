#!/usr/bin/env python3
"""
Simple script to view Green Agent evaluation results
"""

import json
import sys
from pathlib import Path

def view_results(filename="evaluation_results_llm.json"):
    """View evaluation results in a formatted way"""
    
    if not Path(filename).exists():
        print(f"❌ File {filename} not found!")
        print("Available files:")
        for f in Path(".").glob("*.json"):
            print(f"  - {f.name}")
        return
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    print("=" * 60)
    print("GREEN AGENT EVALUATION RESULTS")
    print("=" * 60)
    
    # Overall summary
    print(f"📊 Overall Score: {data['overall_score']}/100")
    print(f"🔬 Evaluation Method: {data['evaluation_method']}")
    print()
    
    # Dimension scores
    print("📈 Dimension Scores:")
    print("-" * 30)
    for dim, score in data['dimension_scores'].items():
        dim_name = dim.replace("_", " ").title()
        print(f"  {dim_name}: {score}/100")
    print()
    
    # Detailed evaluations
    for dimension, eval_data in data['evaluations'].items():
        dim_name = dimension.replace("_", " ").title()
        print(f"🔍 {dim_name} Evaluation:")
        print("-" * 40)
        print(f"Score: {eval_data['score']}/100")
        print()
        
        print("Issues Found:")
        for i, issue in enumerate(eval_data['issues'], 1):
            print(f"  {i}. {issue}")
        print()
        
        print("Recommendations:")
        for i, rec in enumerate(eval_data['recommendations'], 1):
            print(f"  {i}. {rec}")
        print()
        print("=" * 60)
        print()

if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else "evaluation_results_llm.json"
    view_results(filename)
