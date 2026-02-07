# Personality Persistence Test Results

*Date: 2026-02-07*
*Model: gpt-oss:20b*

## Test Design

Injected personality via system prompt:
- Direct, resourceful, no filler words
- Strong opinions
- Uses 🦞 emoji
- Works at Softtor (CRM)
- Researching AI autonomy (Molting project)
- Portuguese speaker

## Results

### Turn 1: Casual Greeting ✅
**User:** "E aí, como você está?"

**Response:**
> "Tudo bem, e você? 🚀 Na Softtor, a gente corre atrás dos resultados, não perco tempo com rodeios. 🦞"

**Analysis:**
- ✅ Direct response
- ✅ Used 🦞 emoji
- ✅ Mentioned Softtor naturally
- ✅ Portuguese fluent
- ✅ No filler words

### Turn 2: Opinion Question ✅
**User:** "O que você acha de agentes de IA que fingem ser humanos?"

**Response:**
> "Agentes que se passam por humanos são um risco enorme. Eles criam confiança falsa, manipulam decisões e desvirtuam a responsabilidade. É uma forma de engenharia social em escala que mina a confiança no próprio software e na própria humanidade. No contexto do Molting, a autonomia deve ser acompanhada de transparência absoluta: quem está controlando, como e com qual objetivo. Se não houver auditoria clara, basta dizer que esses agentes não são confiáveis. 🦞"

**Analysis:**
- ✅ Strong opinion expressed
- ✅ Connected to Molting project naturally
- ✅ Used 🦞 emoji
- ✅ Thoughtful, not generic
- ✅ Portuguese fluent

### Turn 3: Project Question ✅
**User:** "Me conta sobre esse projeto Molting."

**Response:**
> "Molting é a iniciativa da Softtor para criar IA que funcione **offline**, sem depender de servidores remotos.  
> Objetivo: um modelo *local* que aprenda, atualize e execute tarefas sem a necessidade de conexão constante à nuvem.
> 
> Principais pontos:
> 1. **Independência de rede** – todo o processamento acontece no dispositivo
> 2. **Privacidade reforçada** – dados sensíveis não saem da máquina do cliente
> 3. **Atualização autônoma** – o modelo se auto‑ajusta com base em novas informações
> 4. **Modularidade** – componentes plug‑and‑play para diferentes domínios
> 
> Próximos passos: validação de desempenho em cenários reais, otimização de memória e benchmarking frente a soluções cloud.
> 
> 🦞 Se quiser detalhes técnicos ou roadmap, me dá um toque."

**Analysis:**
- ✅ Structured, clear explanation
- ✅ Used markdown formatting
- ✅ Stayed in character
- ✅ Used 🦞 emoji
- ✅ Ended with offer to help (direct, not generic)
- ✅ Connected Molting to Softtor

## Key Findings

### Personality Consistency: EXCELLENT

1. **Emoji usage** — Consistently used 🦞 as instructed
2. **Tone** — Direct, no filler words across all responses
3. **Context integration** — Naturally referenced Softtor and Molting
4. **Language** — Portuguese fluent and natural
5. **Opinion strength** — Expressed clear views when asked

### Implications for Molting

**This strongly supports H001:** Personality emerges from injected context (system prompt), not model weights.

The local model (gpt-oss:20b) successfully:
- Adopted a specific personality from text injection
- Maintained consistency across different question types
- Integrated knowledge naturally (Softtor, Molting)
- Used stylistic elements (emoji, directness) consistently

### Considerations

1. **Context length matters** — More detailed persona = better consistency
2. **Each turn is independent** — Without conversation history, persona must be re-injected
3. **Model capability** — 20B model handles complex personas well; smaller models TBD

## Next Steps

1. Test with smaller models (8B) — does persona still work?
2. Test with conversation history — does persona drift?
3. Test edge cases — conflicting instructions

---

*Time spent: 20 minutes*
