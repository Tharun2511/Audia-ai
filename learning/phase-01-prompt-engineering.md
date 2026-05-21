# Phase 1 — Prompt Engineering

## Session 1.1 — Prompt engineering fundamentals

**Built in Audia:** rebuilt [summarizeTranscript](../src/lib/ai.ts) to use JSON output mode (`response_format: { type: "json_object" }`), few-shot examples in the system prompt (one normal case + one too-short case), and Zod schema validation (`SummaryResponseSchema` with `tooShort` + `bullets`). Function signature unchanged for backward compatibility; internals are now production-grade with typed runtime data.

### Concept summary

Prompts are specifications, not strings. The five-part audit applies to every prompt: **role** (who is the model?), **task** (what's the goal?), **format** (what shape is the output?), **constraints** (what NOT to do?), **examples** (show the pattern). Zero-shot is fine for tasks the model has seen abundantly; few-shot wins for custom formats, classification with non-obvious labels, and edge cases. Chain-of-thought (CoT) helps on multi-step reasoning but hurts on simple lookups — modern reasoning models do CoT internally. Structured output requires three layers of defense: provider JSON mode, provider schema mode (where supported), and client-side Zod validation — each catches different failure modes.

### 5 most-likely interview questions

1. **Q: How do you reliably get JSON out of an LLM?**
   A: Three layers. First, the prompt explicitly specifies the schema with at least one example of the exact shape. Second, the provider's structured output feature — `response_format: { type: "json_object" }` for OpenAI/Groq — constrains the decoder to valid JSON. Third, client-side schema validation with Zod or similar. JSON mode prevents syntax errors; example-driven prompting prevents shape errors; Zod catches edge cases providers miss and gives typed runtime data.

2. **Q: Walk me through Audia's summary prompt.**
   A: Five components. *Role*: "You are a meeting summarizer." *Task*: produce a JSON object with `tooShort` and `bullets`. *Format*: schema spec inline in the system prompt. *Constraints*: "no text outside JSON, no code fences" — negative constraints that block common failure modes. *Examples*: two few-shot examples, one normal case and one too-short case, locking in both the output pattern and the edge case. API call uses `response_format: { type: "json_object" }` for syntactic JSON guarantee, response is parsed through a Zod schema for shape validation.

3. **Q: When does few-shot beat zero-shot, and what's the cost?**
   A: Few-shot wins for (1) custom output formats — your specific shape, not generic JSON, (2) classification with non-obvious labels or domain categories, (3) edge cases the model would otherwise mishandle, (4) domain-specific phrasing. The cost is input tokens — a 5-shot prompt can be 10× larger than zero-shot. Rule of thumb: start zero-shot, add examples only when outputs are unreliable. More than 5 examples rarely helps — consider fine-tuning instead.

4. **Q: When should you NOT use chain-of-thought?**
   A: Simple lookups, one-step transforms, creative writing, and tasks where the model already "thinks" naturally like summarization. CoT spends output tokens and adds latency without quality gain on these. Also: modern reasoning models (Claude with extended thinking, o1, DeepSeek-R1) do CoT internally — explicitly asking for it on these wastes tokens and can confuse the model.

5. **Q: Why put instructions in `system` and data in `user`?**
   A: Three reasons. (1) Training — models are explicitly trained to weight system content as operator intent, so instructions in system are stickier. (2) Conversation persistence — system messages stay across multi-turn chats; user messages are turn-specific. (3) Security — user input is untrusted in production; if instructions live alongside user data in the user message, a malicious user payload can override them via prompt injection. Phase 1.2 covers the defenses.

### Gotchas

- **"Respond in JSON" prompt without `response_format` flag** — the model wraps output in markdown code fences (```json...```) and your `JSON.parse` throws. Always pair the prompt instruction with the provider's JSON mode flag.
- **Schema in prose, not examples** — descriptions are weaker than demonstrations. "Output `{"a": 1}`" beats "output an object with an `a` field set to a number."
- **JSON output without bumping `max_tokens`** — JSON has 20–40% overhead from quotes/brackets/keys. A 250-token cap that worked for bullets may truncate JSON mid-array.
- **Few-shot examples that don't match real inputs** — examples need to look like production data, not toy data. The model pattern-matches from the examples.
- **Adding CoT to short tasks** — wastes output tokens and latency. CoT shines on multi-step reasoning, not classification or summarization.
- **Trusting `system` as a hard contract** — a determined user message can still override. System weighting is statistical, not enforced. Phase 1.2 covers this.

### Go-deeper resources (optional)

- Anthropic prompt engineering overview — docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview
- OpenAI prompt engineering guide — platform.openai.com/docs/guides/prompt-engineering
- Lilian Weng, *"Prompt Engineering"* (lilianweng.github.io/posts/2023-03-15-prompt-engineering/) — academic survey
- Wei et al., *"Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"* — read the abstract for the famous "Let's think step by step" result
