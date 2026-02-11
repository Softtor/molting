#!/usr/bin/env python3
"""
Dataset Curation Script - Phase 5
Removes "agent-like" patterns from training data to preserve personality.

Author: Cláudio (subagent molting-phase5-feb11)
Date: 2026-02-11
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

# Agent-like patterns to detect and remove
AGENT_PATTERNS = [
    # Meta planning language (English)
    r"I'?ll\s+(start|begin|first|now|continue)\s+(by|with|the)",
    r"Let me\s+(start|begin|first|read|check|analyze|examine|look|see|understand|pick up)",
    r"First,?\s+I'?ll",
    r"I need to\s+(first|start|begin|read|check|analyze)",
    
    # Meta planning language (Portuguese)
    r"Vou\s+(implementar|analisar|começar|primeiro|ler|verificar|examinar)",
    r"Deixe-me\s+(ler|analisar|começar|verificar|primeiro)",
    r"Primeiro,?\s+(vou|deixe-me)",
    r"Preciso\s+(primeiro|começar|ler|analisar)",
    
    # Over-explained actions
    r"I'?ll\s+(read|check|examine|analyze|look at|review|continue|pick up)\s+",
    r"Let's\s+(start|begin)\s+(by|with)",
    
    # Planning metacommentary
    r"Here'?s\s+(what|how)\s+I'?ll",
    r"My\s+plan\s+is\s+to",
    r"I'?m\s+going\s+to\s+(start|begin|first)",
    
    # Continuation phrases
    r"I'?ll\s+continue\s+(with|the|where)",
    r"picking up where",
    
    # Overly formal acknowledgments
    r"^Excellent!?\s*$",
    r"^Perfect!?\s*$",
    r"^Great!?\s*$",
    
    # Tool/skill invocation metacommentary
    r"by\s+invoking\s+the\s+relevant\s+skills?",
    r"I'?ll\s+use\s+the\s+\w+\s+skill",
    r"vou\s+usar\s+o\s+skill",
]

# Compile patterns for efficiency
COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in AGENT_PATTERNS]


def has_agent_pattern(text: str) -> Tuple[bool, List[str]]:
    """
    Check if text contains agent-like patterns.
    Returns (has_pattern, list_of_matched_patterns)
    """
    matches = []
    for pattern in COMPILED_PATTERNS:
        if pattern.search(text):
            matches.append(pattern.pattern)
    return len(matches) > 0, matches


def is_too_short(text: str, min_chars: int = 50) -> bool:
    """Filter out responses that are too short to be meaningful."""
    return len(text.strip()) < min_chars


def is_tool_only(text: str) -> bool:
    """
    Heuristic: responses with very few words but many newlines
    are likely tool-only invocations without personality.
    """
    word_count = len(text.split())
    newline_count = text.count('\n')
    return word_count < 20 and newline_count > 5


def curate_conversation(item: Dict) -> Tuple[bool, str]:
    """
    Analyze a conversation item and decide if it should be kept.
    
    Returns:
        (keep: bool, reason: str)
    """
    conversations = item.get('conversations', [])
    
    if not conversations:
        return False, "empty_conversation"
    
    # Check only assistant responses (GPT role)
    for msg in conversations:
        role = msg.get('from', msg.get('role', ''))
        content = msg.get('value', msg.get('content', ''))
        
        if role.lower() in ['gpt', 'assistant']:
            # Check for agent patterns
            has_pattern, patterns = has_agent_pattern(content)
            if has_pattern:
                return False, f"agent_pattern: {patterns[0][:50]}"
            
            # Check if too short
            if is_too_short(content):
                return False, "too_short"
            
            # Check if tool-only
            if is_tool_only(content):
                return False, "tool_only"
    
    return True, "ok"


def curate_dataset(
    input_path: str,
    output_path: str,
    report_path: str,
    dry_run: bool = False
) -> Dict:
    """
    Main curation function.
    
    Args:
        input_path: Path to input JSON dataset
        output_path: Path to write curated dataset
        report_path: Path to write curation report
        dry_run: If True, don't write output, just analyze
    
    Returns:
        Statistics dictionary
    """
    print(f"Loading dataset from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total = len(data)
    print(f"Total examples: {total}")
    
    # Stats tracking
    stats = {
        'total': total,
        'kept': 0,
        'removed': 0,
        'removal_reasons': {},
    }
    
    curated_data = []
    
    for i, item in enumerate(data, 1):
        if i % 50 == 0:
            print(f"Processing {i}/{total}...")
        
        keep, reason = curate_conversation(item)
        
        if keep:
            curated_data.append(item)
            stats['kept'] += 1
        else:
            stats['removed'] += 1
            stats['removal_reasons'][reason] = stats['removal_reasons'].get(reason, 0) + 1
    
    # Write curated dataset
    if not dry_run:
        print(f"\nWriting curated dataset to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(curated_data, f, indent=2, ensure_ascii=False)
    
    # Generate report
    report = generate_report(stats, input_path, output_path, report_path)
    
    if not dry_run:
        print(f"Writing report to {report_path}...")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
    
    print("\n" + "="*80)
    print(report)
    print("="*80)
    
    return stats


def generate_report(stats: Dict, input_path: str, output_path: str, report_path: str) -> str:
    """Generate markdown report of curation process."""
    
    removal_pct = (stats['removed'] / stats['total']) * 100 if stats['total'] > 0 else 0
    kept_pct = (stats['kept'] / stats['total']) * 100 if stats['total'] > 0 else 0
    
    report = f"""# Dataset Curation Report - Phase 5

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Agent:** Cláudio (subagent molting-phase5-feb11)

## Goal

Remove "agent-like" patterns from training data to improve personality preservation:
- Patterns like "I'll start by...", "Let me first...", "I need to..."
- Tool-only responses without personality
- Overly short acknowledgments

## Input/Output

- **Input:** `{Path(input_path).name}`
- **Output:** `{Path(output_path).name}`

## Results

| Metric | Value |
|--------|-------|
| **Total examples** | {stats['total']:,} |
| **Kept** | {stats['kept']:,} ({kept_pct:.1f}%) |
| **Removed** | {stats['removed']:,} ({removal_pct:.1f}%) |

## Removal Breakdown

"""
    
    # Sort removal reasons by count
    sorted_reasons = sorted(
        stats['removal_reasons'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for reason, count in sorted_reasons:
        pct = (count / stats['removed']) * 100 if stats['removed'] > 0 else 0
        report += f"- **{reason}:** {count:,} ({pct:.1f}% of removed)\n"
    
    report += f"""

## Patterns Detected

The following regex patterns were used to identify agent-like responses:

"""
    
    for pattern in AGENT_PATTERNS:
        report += f"- `{pattern}`\n"
    
    report += f"""

## Quality Criteria

**Kept examples must:**
- ✅ Have responses ≥50 characters
- ✅ NOT contain agent-like planning language
- ✅ NOT be tool-only invocations (heuristic: <20 words + >5 newlines)

## Next Steps

1. **Review curated dataset** - Spot check samples to verify quality
2. **Retrain model** - Use {Path(output_path).name} for QLoRA fine-tuning
3. **Evaluate personality** - Compare base vs curated-trained model
4. **Iterate if needed** - Adjust patterns or thresholds based on results

## Files Generated

- `{Path(output_path).name}` - Curated dataset ({stats['kept']:,} examples)
- `{Path(report_path).name}` - This report

---

**Status:** ✅ Curation complete. Dataset ready for retraining.
"""
    
    return report


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Curate fine-tuning dataset')
    parser.add_argument(
        '--input',
        default='dataset_sharegpt_filtered.json',
        help='Input dataset path'
    )
    parser.add_argument(
        '--output',
        default='dataset_sharegpt_curated.json',
        help='Output curated dataset path'
    )
    parser.add_argument(
        '--report',
        default='CURATION_REPORT.md',
        help='Curation report output path'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Analyze only, do not write output files'
    )
    
    args = parser.parse_args()
    
    curate_dataset(
        input_path=args.input,
        output_path=args.output,
        report_path=args.report,
        dry_run=args.dry_run
    )
