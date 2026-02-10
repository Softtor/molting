#!/usr/bin/env python3
"""
Extract fine-tuning dataset from Claude session logs.
Outputs ShareGPT/Alpaca format for personality fine-tuning.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from collections import Counter
import re

CLAUDE_PROJECTS = Path("/home/joao/.claude/projects")
OUTPUT_DIR = Path(__file__).parent
MIN_RESPONSE_LENGTH = 50  # chars
MIN_INSTRUCTION_LENGTH = 10  # chars


def find_session_files() -> List[Path]:
    """Find all JSONL session files."""
    files = []
    for jsonl_file in CLAUDE_PROJECTS.rglob("*.jsonl"):
        # Skip subagent files for now (focus on main sessions)
        if "subagents" not in str(jsonl_file):
            files.append(jsonl_file)
    return files


def extract_turns(jsonl_file: Path) -> List[Tuple[str, str, Dict]]:
    """
    Extract instruction-response pairs from a session.
    
    Returns:
        List of (instruction, response, metadata) tuples
    """
    pairs = []
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            events = [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"⚠️  Error reading {jsonl_file.name}: {e}")
        return []
    
    # Separate user and assistant messages
    user_msgs = [e for e in events if e.get('type') == 'user']
    assistant_msgs = [e for e in events if e.get('type') == 'assistant']
    
    # Match user → assistant pairs by parentUuid
    for user_msg in user_msgs:
        user_uuid = user_msg.get('uuid')
        user_content = user_msg.get('message', {}).get('content', '')
        
        if not user_content or len(user_content) < MIN_INSTRUCTION_LENGTH:
            continue
        
        # Find matching assistant response
        matching_assistant = None
        for assistant_msg in assistant_msgs:
            if assistant_msg.get('parentUuid') == user_uuid:
                matching_assistant = assistant_msg
                break
        
        if not matching_assistant:
            continue
        
        # Extract assistant content
        assistant_content_blocks = matching_assistant.get('message', {}).get('content', [])
        if isinstance(assistant_content_blocks, list):
            # Concatenate text blocks
            text_blocks = [
                block.get('text', '') 
                for block in assistant_content_blocks 
                if block.get('type') == 'text'
            ]
            assistant_content = '\n'.join(text_blocks).strip()
        else:
            assistant_content = str(assistant_content_blocks)
        
        if not assistant_content or len(assistant_content) < MIN_RESPONSE_LENGTH:
            continue
        
        # Check for tool-only responses (heuristic: very short + no natural language)
        if len(assistant_content) < 100 and assistant_content.count('\n') > 5:
            continue
        
        # Extract metadata
        metadata = {
            'session_id': user_msg.get('sessionId', 'unknown'),
            'timestamp': user_msg.get('timestamp', ''),
            'model': matching_assistant.get('message', {}).get('model', 'unknown'),
            'file': jsonl_file.name
        }
        
        pairs.append((user_content, assistant_content, metadata))
    
    return pairs


def classify_topic(instruction: str, response: str) -> str:
    """
    Classify the topic/domain of a conversation turn.
    """
    text = (instruction + " " + response).lower()
    
    # Define topic keywords
    topics = {
        'coding': ['code', 'function', 'debug', 'git', 'commit', 'python', 'javascript', 'api'],
        'architecture': ['architecture', 'design', 'system', 'database', 'microservice', 'pattern'],
        'project': ['project', 'molting', 'softtor', 'crm', 'task', 'plan', 'feature'],
        'research': ['research', 'paper', 'study', 'analysis', 'experiment', 'benchmark'],
        'personal': ['joão', 'joao', 'você', 'your', 'personal', 'prefer', 'opinion'],
        'tools': ['file', 'read', 'write', 'exec', 'command', 'shell', 'terminal'],
        'other': []
    }
    
    # Count keyword matches
    scores = {}
    for topic, keywords in topics.items():
        if topic == 'other':
            continue
        scores[topic] = sum(1 for kw in keywords if kw in text)
    
    # Return topic with highest score
    if scores:
        max_topic = max(scores, key=scores.get)
        if scores[max_topic] > 0:
            return max_topic
    
    return 'other'


def format_sharegpt(pairs: List[Tuple[str, str, Dict]]) -> List[Dict]:
    """Format as ShareGPT (multi-turn conversations)."""
    conversations = []
    
    for instruction, response, metadata in pairs:
        conversations.append({
            "conversations": [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": response}
            ],
            "source": metadata['file'],
            "session_id": metadata['session_id'],
            "timestamp": metadata['timestamp']
        })
    
    return conversations


def format_alpaca(pairs: List[Tuple[str, str, Dict]]) -> List[Dict]:
    """Format as Alpaca (instruction-output)."""
    examples = []
    
    for instruction, response, metadata in pairs:
        examples.append({
            "instruction": instruction,
            "input": "",
            "output": response,
            "metadata": metadata
        })
    
    return examples


def analyze_dataset(pairs: List[Tuple[str, str, Dict]]) -> Dict:
    """Generate dataset statistics."""
    if not pairs:
        return {}
    
    instruction_lengths = [len(p[0]) for p in pairs]
    response_lengths = [len(p[1]) for p in pairs]
    
    topics = [classify_topic(p[0], p[1]) for p in pairs]
    topic_counts = Counter(topics)
    
    models = [p[2]['model'] for p in pairs]
    model_counts = Counter(models)
    
    return {
        "total_pairs": len(pairs),
        "avg_instruction_length": round(sum(instruction_lengths) / len(instruction_lengths), 1),
        "avg_response_length": round(sum(response_lengths) / len(response_lengths), 1),
        "median_instruction_length": sorted(instruction_lengths)[len(instruction_lengths)//2],
        "median_response_length": sorted(response_lengths)[len(response_lengths)//2],
        "topic_distribution": dict(topic_counts.most_common()),
        "model_distribution": dict(model_counts.most_common()),
        "unique_sessions": len(set(p[2]['session_id'] for p in pairs))
    }


def main():
    print("=" * 80)
    print("FINE-TUNING DATASET PREPARATION")
    print("=" * 80)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Source: {CLAUDE_PROJECTS}")
    print()
    
    # Find session files
    print("Finding session files...")
    session_files = find_session_files()
    print(f"✓ Found {len(session_files)} session files")
    print()
    
    # Extract all pairs
    print("Extracting instruction-response pairs...")
    all_pairs = []
    
    for i, jsonl_file in enumerate(session_files, 1):
        if i % 10 == 0:
            print(f"  Processed {i}/{len(session_files)} files, {len(all_pairs)} pairs so far...")
        
        pairs = extract_turns(jsonl_file)
        all_pairs.extend(pairs)
    
    print(f"✓ Extracted {len(all_pairs)} pairs from {len(session_files)} files")
    print()
    
    if not all_pairs:
        print("⚠️  No valid pairs found!")
        return
    
    # Analyze dataset
    print("Analyzing dataset...")
    stats = analyze_dataset(all_pairs)
    print(f"✓ Analysis complete")
    print()
    
    # Format outputs
    print("Formatting datasets...")
    sharegpt_data = format_sharegpt(all_pairs)
    alpaca_data = format_alpaca(all_pairs)
    
    # Save datasets
    sharegpt_file = OUTPUT_DIR / "dataset_sharegpt.json"
    alpaca_file = OUTPUT_DIR / "dataset_alpaca.json"
    stats_file = OUTPUT_DIR / "dataset_stats.json"
    
    with open(sharegpt_file, 'w', encoding='utf-8') as f:
        json.dump(sharegpt_data, f, indent=2, ensure_ascii=False)
    
    with open(alpaca_file, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, indent=2, ensure_ascii=False)
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Saved ShareGPT format: {sharegpt_file}")
    print(f"✓ Saved Alpaca format: {alpaca_file}")
    print(f"✓ Saved statistics: {stats_file}")
    print()
    
    # Print summary
    print("=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"Total pairs: {stats['total_pairs']}")
    print(f"Unique sessions: {stats['unique_sessions']}")
    print(f"Avg instruction length: {stats['avg_instruction_length']} chars")
    print(f"Avg response length: {stats['avg_response_length']} chars")
    print()
    print("Topic distribution:")
    for topic, count in sorted(stats['topic_distribution'].items(), key=lambda x: x[1], reverse=True):
        pct = (count / stats['total_pairs']) * 100
        print(f"  {topic:15s}: {count:4d} ({pct:5.1f}%)")
    print()
    print("Model distribution:")
    for model, count in sorted(stats['model_distribution'].items(), key=lambda x: x[1], reverse=True):
        pct = (count / stats['total_pairs']) * 100
        print(f"  {model:20s}: {count:4d} ({pct:5.1f}%)")
    print()
    print("✅ COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
