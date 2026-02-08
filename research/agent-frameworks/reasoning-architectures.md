# Agent Reasoning Architectures

> Research Date: 2026-02-08
> Status: Complete

## Overview

Modern LLM agents use various reasoning architectures to solve complex tasks. The key paradigms are Chain-of-Thought (CoT), ReAct, and Tree-of-Thought (ToT).

---

## Chain-of-Thought (CoT)

### Core Idea

Force the model to show its reasoning steps before answering.

```
Question: What is 23 × 47?

Without CoT: 1081 (may be wrong)

With CoT:
- First, 23 × 40 = 920
- Then, 23 × 7 = 161
- Finally, 920 + 161 = 1081 ✓
```

### When to Use

- Arithmetic reasoning
- Commonsense reasoning
- Multi-step problems
- Logical deduction

### Limitations

- **Fact hallucination** — model may "reason" with incorrect facts
- **No external access** — can't verify or update knowledge
- **Error propagation** — one wrong step breaks the chain

---

## ReAct (Reasoning + Acting)

### Core Idea

Interleave reasoning with actions that query external sources.

```
Thought → Action → Observation → Thought → Action → ...
```

### Example Flow

```
Question: "What is the elevation of the birthplace of Albert Einstein?"

Thought 1: I need to find where Einstein was born, then find that city's elevation.
Action 1: Search[Albert Einstein birthplace]
Observation 1: Albert Einstein was born in Ulm, Germany.

Thought 2: Now I need to find the elevation of Ulm.
Action 2: Search[Ulm Germany elevation]
Observation 2: Ulm has an elevation of 479 meters.

Thought 3: I have the answer.
Action 3: Finish[479 meters]
```

### Advantages Over CoT

| Aspect | CoT | ReAct |
|--------|-----|-------|
| External access | ❌ | ✅ |
| Fact verification | ❌ | ✅ |
| Reduces hallucination | Partial | ✅ |
| Interpretability | Good | Better |

### ReAct Components

1. **Thought** — Verbal reasoning trace
2. **Action** — Interface with external tools (search, API, etc.)
3. **Observation** — Result from the action

### Implementation (LangChain)

```python
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

tools = [
    Tool(name="Search", func=search_api, description="Search the web"),
    Tool(name="Calculator", func=calculator, description="Math operations"),
]

agent = initialize_agent(
    tools=tools,
    llm=OpenAI(temperature=0),
    agent_type="react"
)

result = agent.run("What is the population of France divided by 1000?")
```

---

## Tree-of-Thought (ToT)

### Core Idea

Explore multiple reasoning paths simultaneously, not just one chain.

```
                    Problem
                       │
        ┌──────────────┼──────────────┐
        │              │              │
    Thought A      Thought B      Thought C
        │              │              │
    ┌───┴───┐      ┌───┴───┐      ┌───┴───┐
    │       │      │       │      │       │
   A.1     A.2    B.1     B.2    C.1     C.2
    │              │                      │
  Dead end     Solution!             Dead end
```

### When to Use

- Creative problem solving
- Game playing (chess, puzzles)
- Planning with multiple options
- Problems with backtracking needs

### Advantages

- Can explore alternatives
- Avoids getting stuck on wrong path
- Better for complex, multi-step planning

### Limitations

- More compute intensive
- More complex to implement
- Overkill for simple tasks

---

## Comparison Table

| Feature | CoT | ReAct | ToT |
|---------|-----|-------|-----|
| **External tools** | ❌ | ✅ | Optional |
| **Exploration** | Linear | Linear | Tree/branching |
| **Backtracking** | ❌ | ❌ | ✅ |
| **Hallucination** | High | Low | Medium |
| **Compute cost** | Low | Medium | High |
| **Interpretability** | Good | Great | Good |
| **Best for** | Math, simple reasoning | QA, fact-checking | Planning, games |

---

## Other Architectures

### Plan-and-Execute

Separate planning from execution:
1. Create a full plan first
2. Execute step by step
3. Revise plan if needed

### ReWOO (Reasoning WithOut Observation)

Plan all actions upfront, execute in batch, then reason on results. More efficient than ReAct for some tasks.

### Reflexion

Add self-reflection after task completion:
1. Execute task
2. Reflect on what went wrong
3. Improve for next attempt

### Self-Consistency

Generate multiple CoT paths, take majority vote on answer. Simple but effective for improving accuracy.

---

## Relevance to Molting Project

### Which Architecture for Cláudio?

**Current (OpenClaw):**
- Uses tool-calling (similar to ReAct actions)
- Memory system for context
- No explicit reasoning traces visible

**For Local Model:**
- ReAct is most practical — simple, effective, interpretable
- CoT for reasoning-heavy tasks without tools
- ToT probably overkill for assistant tasks

### Implementation Considerations

1. **Tool Integration:**
   - Define tools (search, code exec, file access)
   - Use structured output for actions
   - Parse observations back to context

2. **Prompt Engineering:**
   - Include ReAct examples in system prompt
   - Define when to think vs act
   - Handle failure modes

3. **Personality + Reasoning:**
   - Reasoning traces should match personality
   - "Cláudio thinks..." not "The AI thinks..."

### Hypotheses

**H018:** ReAct reduces hallucination in personality-tuned local models

**H019:** Explicit reasoning traces improve consistency across sessions

---

## References

1. [ReAct Paper](https://arxiv.org/abs/2210.03629) — Yao et al., 2022
2. [Chain-of-Thought Paper](https://arxiv.org/abs/2201.11903) — Wei et al., 2022
3. [Tree-of-Thought Paper](https://arxiv.org/abs/2305.10601) — Yao et al., 2023
4. [Prompting Guide - ReAct](https://www.promptingguide.ai/techniques/react)
5. [LangChain ReAct Agent](https://python.langchain.com/docs/modules/agents/)

---

*Analysis by Cláudio for Project Molting*
