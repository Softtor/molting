#!/usr/bin/env python3
"""
Extract conversation turns from OpenClaw session logs.
Outputs structured JSON with user/assistant exchanges.
"""

import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Generator

SESSIONS_DIR = Path.home() / ".openclaw/agents/main/sessions"
OUTPUT_FILE = Path(__file__).parent / "conversations.json"

def parse_jsonl(filepath: Path) -> Generator[dict, None, None]:
    """Parse JSONL file, yield each record."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

def extract_text_from_content(content) -> str:
    """Extract text from content array or string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict):
                if part.get('type') == 'text':
                    texts.append(part.get('text', ''))
            elif isinstance(part, str):
                texts.append(part)
        return '\n'.join(texts)
    return ''

def extract_session(filepath: Path) -> dict:
    """Extract conversation turns from a session file."""
    session_id = filepath.stem
    turns = []
    session_start = None
    
    for record in parse_jsonl(filepath):
        record_type = record.get('type')
        timestamp = record.get('timestamp')
        
        # Track session start
        if record_type == 'session' and not session_start:
            session_start = timestamp
        
        # Extract messages
        if record_type == 'message':
            msg = record.get('message', {})
            role = msg.get('role')
            content = msg.get('content', [])
            
            # Only capture user and assistant text messages
            if role in ('user', 'assistant'):
                text = extract_text_from_content(content)
                
                # Skip very short or empty messages
                if text.strip() and len(text.strip()) > 5:
                    # Skip tool results and system messages
                    if role == 'assistant' and not text.strip():
                        continue
                    
                    turns.append({
                        'role': role,
                        'content': text.strip(),
                        'timestamp': timestamp,
                        'id': record.get('id')
                    })
    
    return {
        'session_id': session_id,
        'session_start': session_start,
        'turn_count': len(turns),
        'turns': turns
    }

def main():
    print(f"Scanning sessions in {SESSIONS_DIR}")
    
    sessions = []
    total_turns = 0
    
    for filepath in sorted(SESSIONS_DIR.glob("*.jsonl")):
        session = extract_session(filepath)
        if session['turn_count'] > 0:
            sessions.append(session)
            total_turns += session['turn_count']
    
    # Sort by session start time
    sessions.sort(key=lambda s: s.get('session_start') or '')
    
    output = {
        'extracted_at': datetime.now(timezone.utc).isoformat(),
        'total_sessions': len(sessions),
        'total_turns': total_turns,
        'sessions': sessions
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nExtracted {total_turns} turns from {len(sessions)} sessions")
    print(f"Output: {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == '__main__':
    main()
