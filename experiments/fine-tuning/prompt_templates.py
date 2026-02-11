#!/usr/bin/env python3
"""
Prompt templates for personality fine-tuned model
Phase 4: Prompt refinement to reduce agent-like behavior
"""

# Original: No system prompt (baseline)
PROMPT_ORIGINAL = None

# v1: Focus on natural conversation, reduce meta-awareness
PROMPT_V1_NATURAL = """You are a helpful assistant. Answer questions naturally and conversationally. 
Keep responses concise and focused. Avoid meta-commentary about how you work or what you're doing.
Just answer the question directly."""

# v2: Emphasize brevity and coherence
PROMPT_V2_BRIEF = """Answer directly and briefly. Stay on topic. No explanations about your process."""

# v3: Personality-focused (developer persona)
PROMPT_V3_DEVELOPER = """You are a developer assistant. Answer questions clearly and practically.
Use your knowledge about development, but keep responses natural and concise."""

# v4: Minimal instructions
PROMPT_V4_MINIMAL = """Be helpful, natural, and brief."""

# v5: Anti-agent instructions
PROMPT_V5_ANTI_AGENT = """Answer questions naturally. You're not a task executor or agent - just have a conversation.
Don't describe what you're "going to do" or your process. Just answer."""


def get_prompt_template(version="v1"):
    """
    Get prompt template by version
    
    Args:
        version: One of "original", "v1", "v2", "v3", "v4", "v5"
    
    Returns:
        System prompt string or None
    """
    templates = {
        "original": PROMPT_ORIGINAL,
        "v1": PROMPT_V1_NATURAL,
        "v2": PROMPT_V2_BRIEF,
        "v3": PROMPT_V3_DEVELOPER,
        "v4": PROMPT_V4_MINIMAL,
        "v5": PROMPT_V5_ANTI_AGENT
    }
    
    if version not in templates:
        raise ValueError(f"Unknown template version: {version}. Choose from {list(templates.keys())}")
    
    return templates[version]


def format_prompt_with_system(user_prompt, system_prompt=None, tokenizer=None):
    """
    Format prompt with optional system message
    
    Args:
        user_prompt: User's question/message
        system_prompt: Optional system instruction
        tokenizer: HuggingFace tokenizer (optional, for chat template)
    
    Returns:
        Formatted prompt string
    """
    if system_prompt is None:
        # No system prompt - just user message
        messages = [{"role": "user", "content": user_prompt}]
    else:
        # Include system prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    # Try to use tokenizer's chat template if available
    if tokenizer is not None:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    
    # Fallback: manual formatting
    if system_prompt is None:
        return f"<|user|>\n{user_prompt}<|end|>\n<|assistant|>\n"
    else:
        return f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{user_prompt}<|end|>\n<|assistant|>\n"
