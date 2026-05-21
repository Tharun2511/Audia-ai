# Audia AI learning log

One line per session. Date · phase · what we covered · what stuck · what's fuzzy.

---

**2026-05-17 · Phase 0.1 — How an LLM decides the next token**
- Covered: tokens (BPE), embeddings as learned vector lookup, RoPE positional encoding, attention mechanism intuition (Q/K/V — math marked optional in revision doc), multi-head, causal mask, sampling (temperature/top-p/top-k), autoregressive loop, statelessness between API calls.
- Built: refactored [summarizeTranscript](../src/lib/ai.ts) with system role + `temperature: 0.2` + `max_tokens: 250`.
- **Stuck (well-internalized):**
  - Temperature concept — strong, near-bulletproof answer on Q1
  - Map-Reduce / hierarchical summarization as a pattern — reached for it independently in Q3
  - Pipeline-level architecture of Audia (audio → Deepgram → text → Groq)
- **Fuzzy (needs reinforcement):**
  - The **autoregressive / one-token-at-a-time** nature of LLMs — didn't surface in Feynman, didn't know Q2 ("how do LLMs have conversations"). Re-anchor in Session 0.2 with streaming as a concrete demonstration.
  - **Statelessness** — same gap; conversations as "caller re-sends full history" not yet sticky.
  - **Tokenization + embeddings** — didn't surface in any answer, so depth is unknown. Surface in Session 0.2 via tokenizer demo + cost math exercise.
- **Calibrations applied:** attention/softmax math went over his head; revision doc §1 math sections marked `[OPTIONAL]`. Going forward, default to mechanism intuition + named concepts, not derivations. See [feedback_math_depth.md](../../../memory/feedback_math_depth.md).

---

**2026-05-21 · Phase 0.2 — The model API surface: sampling, streaming, cost math**
- Covered: full sampling parameter table (temperature, top_p, top_k, max_tokens, stop, frequency/presence penalties, seed), three streaming protocols (SSE, ReadableStream, WebSockets), why output costs 2–5× input, cost-per-call math at real Groq rates, prompt caching as a future optimization.
- Built: new [src/lib/ai-usage.ts](../src/lib/ai-usage.ts) (pricing table + `computeCost` + `logUsage`); wired into [summarizeTranscript](../src/lib/ai.ts) and [chat/route.ts](../src/app/api/chat/route.ts) with `stream_options: { include_usage: true }`. Server console now prints structured usage on every LLM call.
- **Stuck (well-internalized):**
  - `stream_options.include_usage` concept — got the core idea right (without it, usage data is omitted from streaming responses)
  - Streaming-faster trap — correctly identified it's *perceived* speed, not actual speed; just needs TTFT terminology to be interview-bulletproof
- **Fuzzy (needs reinforcement):**
  - **Cost math arithmetic** — got $0.000058 instead of $0.000132 on a straightforward computation. Drill the "cents per million" mental shortcut; redo cost estimates on 3 future model calls.
  - **`finish_reason` field + max_tokens diagnostic flow** — didn't know. Production-bug pattern worth memorizing. Will surface again naturally in Phase 6 (evals).
  - **Calibrated "when to stream" decision-making** — default instinct was "more streaming = better." Reinforce the heuristic: stream when total generation > ~2 seconds; skip for short atomic outputs.
  - **TTFT (time to first token) terminology** — knew the concept, not the name. Same for TPS (tokens per second).
