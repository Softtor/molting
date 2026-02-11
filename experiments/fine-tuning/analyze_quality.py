#!/usr/bin/env python3
"""
Dataset Quality Analysis and Filtering
Identifies quality issues and creates a filtered dataset for fine-tuning
"""

import json
from pathlib import Path
from typing import Dict, List, Any


def analyze_quality(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze dataset quality metrics"""
    issues = {
        'too_short': [],
        'empty_response': [],
        'tool_only': [],
        'good_quality': []
    }
    
    for idx, item in enumerate(dataset):
        conv = item['conversations']
        input_text = conv[0]['value']
        output_text = conv[1]['value']
        
        # Check for empty responses
        if not output_text.strip():
            issues['empty_response'].append(idx)
            continue
        
        # Check for tool-only responses (heuristic: very short with many newlines)
        if len(output_text) < 100 and output_text.count('\n') > 5:
            issues['tool_only'].append(idx)
            continue
        
        # Check for too-short responses (likely acknowledgments)
        if len(output_text) < 80:
            issues['too_short'].append(idx)
            continue
        
        issues['good_quality'].append(idx)
    
    return {
        'total': len(dataset),
        'empty_response': len(issues['empty_response']),
        'tool_only': len(issues['tool_only']),
        'too_short': len(issues['too_short']),
        'good_quality': len(issues['good_quality']),
        'quality_percentage': 100 * len(issues['good_quality']) / len(dataset),
        'good_indices': issues['good_quality']
    }


def create_filtered_dataset(
    dataset: List[Dict[str, Any]], 
    good_indices: List[int],
    output_path: Path
):
    """Create filtered dataset with only high-quality pairs"""
    filtered = [dataset[i] for i in good_indices]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Filtered dataset saved to {output_path}")
    print(f"   Original: {len(dataset)} pairs")
    print(f"   Filtered: {len(filtered)} pairs ({100*len(filtered)/len(dataset):.1f}%)")


def show_quality_samples(dataset: List[Dict[str, Any]], indices: List[int], label: str, count: int = 3):
    """Show sample pairs from a quality category"""
    print(f"\n=== {label.upper()} (showing {min(count, len(indices))} of {len(indices)}) ===")
    for i in indices[:count]:
        item = dataset[i]
        output = item['conversations'][1]['value']
        print(f"\n[{i}] Length: {len(output)} chars")
        print(f"Response: {output[:150]}{'...' if len(output) > 150 else ''}")


if __name__ == '__main__':
    base_dir = Path(__file__).parent
    
    print("🔍 Analyzing dataset quality...")
    
    # Load dataset
    with open(base_dir / 'dataset_sharegpt.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Analyze
    analysis = analyze_quality(dataset)
    
    # Show results
    print(f"\n📊 QUALITY ANALYSIS RESULTS")
    print(f"Total pairs: {analysis['total']}")
    print(f"Empty responses: {analysis['empty_response']}")
    print(f"Tool-only responses: {analysis['tool_only']}")
    print(f"Too short (<80 chars): {analysis['too_short']}")
    print(f"✅ Good quality: {analysis['good_quality']} ({analysis['quality_percentage']:.1f}%)")
    
    # Show samples of problematic cases
    show_quality_samples(dataset, 
                        [i for i in range(len(dataset)) if i not in analysis['good_indices']][:3],
                        "Problematic examples")
    
    # Show samples of good quality
    show_quality_samples(dataset, 
                        analysis['good_indices'][:3],
                        "Good quality examples")
    
    # Create filtered dataset
    filtered_path = base_dir / 'dataset_sharegpt_filtered.json'
    create_filtered_dataset(dataset, analysis['good_indices'], filtered_path)
    
    # Also create filtered Alpaca format
    with open(base_dir / 'dataset_alpaca.json', 'r', encoding='utf-8') as f:
        alpaca_dataset = json.load(f)
    
    filtered_alpaca = [alpaca_dataset[i] for i in analysis['good_indices']]
    alpaca_filtered_path = base_dir / 'dataset_alpaca_filtered.json'
    
    with open(alpaca_filtered_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_alpaca, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Filtered Alpaca dataset saved to {alpaca_filtered_path}")
    
    print("\n📋 RECOMMENDATION:")
    if analysis['quality_percentage'] >= 60:
        print(f"✅ Dataset quality is acceptable ({analysis['quality_percentage']:.1f}%). Use filtered dataset for training.")
    else:
        print(f"⚠️  Dataset quality is low ({analysis['quality_percentage']:.1f}%). Consider augmenting with synthetic data.")
