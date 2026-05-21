# Phase 0 — LLM Foundations

## Session 0.1 — How an LLM decides the next token

**Built in Audia:** refactored [summarizeTranscript](../src/lib/ai.ts) — extracted system role, set `temperature: 0.2`, capped `max_tokens: 250`.

### Concept summary

LLMs are **autoregressive next-token predictors**. Given a token sequence, they output a probability distribution over the vocabulary for the next token, sample one, append it, and repeat. The pipeline: text → tokens (BPE sub-words) → embeddings (learned vector per token ID) → positional encoding (RoPE in Llama) → stacked transformer blocks (attention + MLP) → final projection to logits → sampling.

**Attention** is the core: each token produces Query, Key, Value vectors via learned projections. `softmax(QKᵀ / √d_k) · V` produces a weighted blend where each token attends to others by relevance. Multi-head runs this in parallel with different projections. The causal mask (−∞ on future positions) is what makes generation autoregressive. Sampling parameters (temperature, top-p, top-k) shape the output distribution after the model is done; they're inference-time knobs, not training-time.

### 5 most-likely interview questions

1. **Q: Walk me through how an LLM generates a response.**
   A: Tokenize input → embedding lookup → add positional info (RoPE in modern models) → pass through N transformer blocks (each = multi-head attention + MLP with residuals + layer norm) → final layer projects to vocab-size logits → apply sampling (temperature / top-p) → emit one token → append to context → repeat until stop token or max length. The model is *stateless between calls* — every turn re-processes the full conversation.

2. **Q: Why divide attention scores by √d_k?**
   A: Without it, for large d_k the dot products `Q·Kᵀ` have variance proportional to d_k, which pushes softmax into saturation (one value ≈ 1, rest ≈ 0). Gradients through softmax vanish in that regime. Dividing by √d_k keeps the variance ≈ 1 regardless of d_k, keeping the distribution differentiable.

3. **Q: What's the difference between top-k, top-p, and temperature?**
   A: Temperature scales logits *before* softmax — `T<1` peakier (deterministic), `T>1` flatter (creative). Top-k keeps only the k highest-probability tokens (hard cutoff). Top-p (nucleus) keeps the smallest set whose cumulative probability ≥ p (adaptive cutoff). In practice we combine temperature + top-p; top-k is older. For structured output use T=0.1–0.3; for creative writing T=0.8–1.0.

4. **Q: Why don't LLMs remember conversations between API calls?**
   A: The model is a stateless function. There's no hidden state between requests. "Conversation memory" is implemented by the *caller* — you send the entire chat history with every request, and the model re-processes it. This is why context window size matters and why long conversations get expensive linearly with history length. (Phase 5 will cover memory strategies — rolling buffer, summary buffer, vector memory.)

5. **Q: Audia's summarizer uses temperature 0.2 — why?**
   A: Summaries are structured factual output. We want consistency (same transcript → same summary) and adherence to the format spec (bullets starting with "• ", no preamble). High temperature would produce stylistic variation per call and increase the chance of format drift. T=0.2 makes softmax sharply prefer the top-1 token, giving near-deterministic output while keeping a small amount of variation in case the top token is a degenerate choice.

### Gotchas

- **Default temperature varies by provider.** OpenAI default is 1.0, Anthropic is 1.0, Groq follows OpenAI. Always set it explicitly for structured output — don't trust the default.
- **`max_tokens` counts *output* only.** It's not the context window. Context window = input + output. Output cost is typically 2–5× input cost per token.
- **System role isn't a hard contract.** A sufficiently adversarial user message can still override system instructions ("ignore previous instructions"). That's prompt injection — Phase 1.2.
- **Streaming ≠ faster generation.** It's the *same* serial token-by-token process; streaming just sends each token to the client as it's generated instead of buffering.

### Go-deeper resources (optional, for later)

- Karpathy, *"Let's build GPT: from scratch, in code, spelled out"* — youtube.com/watch?v=kCc8FmEb1nY
- *Attention Is All You Need* (Vaswani et al., 2017) — read the abstract + section 3.2 only; rest is overkill for our needs
- Lilian Weng, *"The Transformer Family Version 2.0"* — comprehensive reference (lilianweng.github.io)

---

## Session 0.2 — The model API surface: sampling, streaming, cost math

**Built in Audia:** added [src/lib/ai-usage.ts](../src/lib/ai-usage.ts) with pricing table + cost computation + structured logger. Wired it into [summarizeTranscript](../src/lib/ai.ts) and [chat route](../src/app/api/chat/route.ts). Every model call now logs `[ai-usage] label model=… in=… out=… latency=…ms cost=$…` to the server console. Added `stream_options: { include_usage: true }` to the chat stream so usage is captured during streaming.

### Concept summary

The model API surface has three families of concerns: **sampling parameters** (knobs that shape token selection from the model's probability distribution), **streaming** (how the autoregressive token-by-token generation gets delivered to the client), and **cost math** (input × input_rate + output × output_rate, with output 2–5× more expensive due to serial generation). Streaming is the most important — it's not a bolted-on feature but a direct expression of how LLMs work: tokens are generated one at a time anyway, so streaming just forwards each one to the client immediately instead of buffering. This is also the most visually satisfying way to internalize the autoregressive loop from §1.

### 5 most-likely interview questions

1. **Q: What sampling parameters do you set on every production LLM call and why?**
   A: Temperature (controls determinism — 0.1–0.3 for structured output, 0.7+ for creative), max_tokens (bounds cost ceiling), stop sequences (prevent the model from hallucinating fake user turns in chat), stream_options.include_usage (to capture token counts for cost logging), and seed only when reproducibility matters (evals). Top_p is usually left at the provider's default of 0.9.

2. **Q: How does streaming work technically? Does it make the model faster?**
   A: No, it doesn't make the model faster — the model is generating one token at a time regardless. Streaming forwards each token to the client as it's produced rather than buffering. Implementation: provider SDK exposes an async iterator (`for await of`), each yielded chunk has a delta of new text plus optional metadata, we enqueue each delta into a ReadableStream that the HTTP response is bound to. Protocols: SSE (OpenAI, Anthropic) or HTTP chunked (Audia uses this). The win is perceived latency — users tolerate a 20-second streamed response but not a 3-second blank screen.

3. **Q: Why does output cost 2–5× more than input?**
   A: Input is processed in one forward pass — all tokens in parallel, GPU throughput is high. Output is N forward passes — one per generated token, serial, GPU underutilized. Same compute per pass, but output forces serial work, so providers price the throughput hit into the output rate. This is also why generation feels slow while prompt processing feels instant.

4. **Q: How would you control AI spend in production?**
   A: Four levers, in increasing complexity: (1) always set `max_tokens` so any single call has a known ceiling; (2) instrument every call with a structured logger capturing input/output tokens and dollar cost (like Audia's [ai-usage logger](../src/lib/ai-usage.ts)) — you can't optimize what you can't measure; (3) use prompt caching for stable prompt prefixes — Anthropic and OpenAI cache the KV-state of repeated prefixes and charge ~10× less for the cached portion; (4) switch from "stuff everything into context" to RAG so input length stops growing linearly with corpus size.

5. **Q: For Audia's chat endpoint, how would you capture usage data when streaming?**
   A: Pass `stream_options: { include_usage: true }` to the SDK call. The final chunk in the stream will have `choices: []` and a `usage` object with `prompt_tokens` and `completion_tokens`. In the iteration loop, check `chunk.usage` on every iteration (because you don't know which is the last chunk) and save it. After the loop completes, log it with the wall-clock latency.

### Gotchas

- **`stream_options: { include_usage: true }` is opt-in.** Without it, usage data is silently omitted from streaming responses. You'll think the SDK is broken.
- **Seed is best-effort, not bit-identical.** GPU non-determinism (parallel reductions in CUDA aren't deterministic) means even with a fixed seed you get "very close" outputs, not byte-identical.
- **`max_tokens` set too low looks like a model bug.** A response that abruptly ends mid-sentence is almost always a `max_tokens` cap, not a model failure.
- **Counting words ≠ counting tokens for cost estimation.** Especially wrong for code, foreign names, and non-Latin scripts. Always log actual `usage` rather than estimating from word count.
- **Streaming doesn't reduce cost.** Same tokens, same price. It only reduces perceived latency.
