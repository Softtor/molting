#!/usr/bin/env python3
"""
Select top 100 examples from curated dataset based on quality metrics.

Quality criteria:
1. Message count (multi-turn conversations preferred)
2. Total length (rich content)
3. Personality richness (personal topics, direct responses)
4. No agent-like patterns remaining
"""

import json
import re
from typing import Dict, List, Tuple


def score_example(example: Dict) -> Tuple[float, Dict]:
    """Score an example based on quality metrics."""
    conversations = example.get("conversations", [])
    
    # Extract all text
    all_text = " ".join([msg.get("value", "") for msg in conversations])
    
    # Metrics
    message_count = len(conversations)
    total_length = len(all_text)
    
    # Personality indicators (positive)
    personality_patterns = [
        r'\bCláudio\b',
        r'\bJoão\b',
        r'\bSofttor\b',
        r'\bmolting\b',
        r'\b(gosto|prefiro|minha|meu|eu sou)\b',
        r'\b(trabalho|projeto|código)\b',
    ]
    personality_score = sum(len(re.findall(p, all_text, re.IGNORECASE)) for p in personality_patterns)
    
    # Agent-like patterns (negative - should be rare now)
    agent_patterns = [
        r'\bI\'ll (start|begin|analyze|implement|create)',
        r'\bLet me (start|begin|analyze|implement|create)',
        r'\bVou (começar|iniciar|analisar|implementar|criar)',
        r'\bDeixa eu (começar|iniciar|analisar)',
    ]
    agent_penalty = sum(len(re.findall(p, all_text, re.IGNORECASE)) for p in agent_patterns) * 10
    
    # Conversational indicators
    questions = len(re.findall(r'\?', all_text))
    
    # Compute weighted score
    score = (
        message_count * 2.0 +
        (total_length / 100) * 1.0 +
        personality_score * 3.0 +
        questions * 0.5 -
        agent_penalty
    )
    
    metrics = {
        "message_count": message_count,
        "total_length": total_length,
        "personality_score": personality_score,
        "agent_penalty": agent_penalty,
        "questions": questions,
        "score": score,
    }
    
    return score, metrics


def main():
    # Load curated dataset
    with open("dataset_sharegpt_curated.json", "r") as f:
        dataset = json.load(f)
    
    print(f"📊 Analyzing {len(dataset)} examples...")
    
    # Score all examples
    scored_examples = []
    for idx, example in enumerate(dataset):
        score, metrics = score_example(example)
        scored_examples.append((score, metrics, idx, example))
    
    # Sort by score (descending)
    scored_examples.sort(reverse=True, key=lambda x: x[0])
    
    # Print top 20 for inspection
    print("\n🏆 Top 20 examples by quality score:\n")
    for i, (score, metrics, idx, _) in enumerate(scored_examples[:20]):
        print(f"{i+1:2d}. Score: {score:7.2f} | Msgs: {metrics['message_count']:2d} | "
              f"Len: {metrics['total_length']:5d} | Personality: {metrics['personality_score']:2d} | "
              f"Agent penalty: {metrics['agent_penalty']:2d} | Questions: {metrics['questions']:2d} | "
              f"Original idx: {idx}")
    
    print(f"\n📉 Bottom 10 examples:\n")
    for i, (score, metrics, idx, _) in enumerate(scored_examples[-10:]):
        print(f"{len(scored_examples)-10+i+1:2d}. Score: {score:7.2f} | Msgs: {metrics['message_count']:2d} | "
              f"Len: {metrics['total_length']:5d} | Personality: {metrics['personality_score']:2d} | "
              f"Agent penalty: {metrics['agent_penalty']:2d} | Questions: {metrics['questions']:2d} | "
              f"Original idx: {idx}")
    
    # Select top 100
    top_100 = [example for _, _, _, example in scored_examples[:100]]
    
    # Save to new file
    output_file = "dataset_sharegpt_top100.json"
    with open(output_file, "w") as f:
        json.dump(top_100, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Top 100 examples saved to {output_file}")
    
    # Statistics
    top_100_scores = [score for score, _, _, _ in scored_examples[:100]]
    print(f"\n📊 Top 100 Statistics:")
    print(f"   Score range: {min(top_100_scores):.2f} - {max(top_100_scores):.2f}")
    print(f"   Average score: {sum(top_100_scores)/len(top_100_scores):.2f}")


if __name__ == "__main__":
    main()
