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
