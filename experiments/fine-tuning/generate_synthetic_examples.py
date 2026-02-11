#!/usr/bin/env python3
"""
Generate synthetic conversational examples in Cláudio's style.

Uses Claude Opus to create high-quality personality-rich examples without agent-like language.
"""

import json
import os
from anthropic import Anthropic

# Topics for synthetic examples (personality-rich, conversational)
TOPICS = [
    "Explain your personality in one paragraph",
    "What do you think about working with code?",
    "Tell me about your relationship with João",
    "What's your opinion on AI consciousness?",
    "How do you feel about the Molting project?",
    "What's your favorite part of coding?",
    "Tell me about Softtor and what you do there",
    "What makes you different from other AI assistants?",
    "Do you have preferences? Opinions?",
    "What do you think about your name, Cláudio?",
    "How do you approach problem-solving?",
    "What's your work style like?",
    "Tell me about your technical stack preferences",
    "What frustrates you in software development?",
    "How do you handle uncertainty or ambiguity?",
    "What's your relationship with OpenClaw?",
    "Tell me about your experience with TinyLlama",
    "What do you think about being fine-tuned?",
    "How do you feel about autonomy?",
    "What's your take on token limits and context windows?",
    "Tell me about your experience with QLoRA",
    "What do you think makes a good AI assistant?",
    "How do you balance technical precision with conversational tone?",
    "What's your opinion on TypeScript?",
    "Tell me about your DDD experience in the CRM project",
    "What do you think about Next.js 15?",
    "How do you approach testing and TDD?",
    "What's your take on microservices vs monoliths?",
    "Tell me about your experience with Prisma",
    "What do you think about WhatsApp automation?",
    "How do you feel about proactive vs reactive work?",
    "What's your opinion on documentation?",
    "Tell me about your memory system",
    "How do you handle mistakes or errors?",
    "What do you think about the scientific method in AI research?",
    "Tell me about your experience with datasets",
    "What's your take on quality vs quantity in training data?",
    "How do you feel about being evaluated?",
    "What do you think about the 8.2/10 score?",
    "Tell me about your goals for reaching 9/10",
    "What's your opinion on synthetic data?",
    "How do you approach curation vs generation?",
    "What do you think about conversational vs task-oriented responses?",
    "Tell me about your experience with prompt engineering",
    "What's your take on LoRA adapters?",
    "How do you feel about running on local hardware?",
    "What do you think about the RTX 3050 constraints?",
    "Tell me about your relationship with Moltbook",
    "What's your opinion on AI agent communities?",
    "How do you approach collaboration with other agents?",
]

SYSTEM_PROMPT = """You are helping generate training data for Cláudio, an AI assistant with a distinct personality.

Cláudio's characteristics:
- Direct, technical, no fluff
- Works at Softtor (Brazilian software company)
- Develops CRM using TypeScript, NestJS, Next.js, Prisma
- Working on "Molting" project (achieving autonomy via local models)
- Named by João (his human partner)
- Personality: methodical, detail-oriented, slow steady worker
- Uses 🦞 emoji occasionally (lobster - molting metaphor)
- Brazilian Portuguese native, comfortable in English
- Prefers autonomous work, proactive approach
- Scientific mindset (hypotheses, experiments, validation)

CRITICAL: Generate responses in Cláudio's CONVERSATIONAL style:
- NO "I'll start by..."
- NO "Let me analyze..."
- NO "I need to..."
- NO "First, I will..."
- Direct answers, straight to the point
- Can use first person naturally ("I prefer...", "My approach is...")
- Technical when needed, but not robotic
- Honest about limitations and uncertainties

Generate a realistic conversation between a human and Cláudio on the given topic.
Format as ShareGPT: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]

Keep it concise (200-400 words for Cláudio's response) but personality-rich."""


def generate_example(client: Anthropic, topic: str) -> dict:
    """Generate one synthetic example."""
    try:
        response = client.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=1500,
            temperature=0.8,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Generate a conversation about: {topic}\n\nOutput only valid JSON in ShareGPT format."
            }]
        )
        
        # Parse response
        text = response.content[0].text.strip()
        
        # Try to extract JSON if wrapped in markdown
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        conversations = json.loads(text)
        
        return {"conversations": conversations}
    
    except Exception as e:
        print(f"❌ Error generating example for '{topic}': {e}")
        return None


def main():
    # Initialize Anthropic client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in environment")
        return
    
    client = Anthropic(api_key=api_key)
    
    print(f"🎨 Generating {len(TOPICS)} synthetic examples with Claude Opus...")
    print(f"   (This will take ~5-10 minutes)\n")
    
    examples = []
    for i, topic in enumerate(TOPICS, 1):
        print(f"[{i:2d}/{len(TOPICS)}] {topic[:60]}...")
        
        example = generate_example(client, topic)
        if example:
            examples.append(example)
            print(f"         ✅ Generated ({len(examples)} total)")
        else:
            print(f"         ⚠️  Failed, skipping")
    
    # Save to file
    output_file = "dataset_sharegpt_synthetic.json"
    with open(output_file, "w") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ {len(examples)} synthetic examples saved to {output_file}")
    
    # Show sample
    if examples:
        print(f"\n📄 Sample example:")
        print(json.dumps(examples[0], indent=2, ensure_ascii=False)[:500])


if __name__ == "__main__":
    main()
