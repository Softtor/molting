# Multi-Turn Memory Test Results

*Date: 2026-02-07*
*Model: gpt-oss:20b*

## Test Design

Simulated 3-turn conversation to test memory persistence:
1. User introduces themselves
2. Agent asked to recall user info
3. User adds more information

## Results

### Turn 1: Introduction ✅
**User:** "Oi! Meu nome é João, sou desenvolvedor e moro em Brasília."

**Agent response:**
```json
{
  "block": "HUMAN",
  "content": "João, desenvolvedor, mora em Brasília"
}
```

✅ Correctly identified user info
✅ Called memory_update with HUMAN block
✅ Extracted key facts concisely

### Turn 2: Recall Test ✅
**Memory injected:** `HUMAN: João, developer from Brasília.`

**User:** "Qual é meu nome e onde eu moro?"

**Agent response:** "Você se chama João e mora em Brasília."

✅ Correctly recalled name
✅ Correctly recalled location
✅ Natural Portuguese response

### Turn 3: Add New Info ⚠️
**User:** "Estou construindo um CRM chamado Softtor. É minha startup."

**Agent response:**
```json
{
  "block": "Softtor",
  "content": "João is building a CRM called Softtor, a startup project."
}
```

⚠️ Created new block "Softtor" instead of updating HUMAN
- Could be intentional (topic-based memory)
- Or could indicate need for clearer instructions

## Insights

### What Works
1. **Memory extraction** — Model correctly identifies what to remember
2. **Recall** — When memory is in context, recall is accurate
3. **Tool calling** — Consistently uses memory_update correctly
4. **Language handling** — Works well with Portuguese

### What Needs Work
1. **Block consistency** — Model sometimes creates new blocks vs updating existing
2. **Memory schema** — May need stricter schema definition
3. **Instruction clarity** — "Update HUMAN block" needs to be explicit

## Implications for Molting

### Positive
- Memory management fundamentally works
- Recall is accurate when context is correct
- Model understands the memory paradigm

### Considerations
- May need explicit block names in instructions
- Consider allowing dynamic blocks (feature not bug?)
- Need longer tests for drift detection

## Next Steps
1. Test with explicit block instructions
2. Test 10+ turn conversations
3. Test personality persistence (not just facts)

---

*Time spent: 15 minutes*
