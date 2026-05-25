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

---

**2026-05-21 · Phase 1.1 — Prompt engineering fundamentals**
- Covered: five-part prompt structure (role/task/format/constraints/examples), three roles deep dive, zero-shot vs few-shot vs chain-of-thought (with when-to-use trade-offs), structured output triple defense (JSON mode + schema mode + Zod), the five-part prompt audit checklist.
- Built: rebuilt [summarizeTranscript](../src/lib/ai.ts) — `SummaryResponseSchema` Zod schema with `tooShort` + `bullets`, system prompt with explicit JSON format + two few-shot examples + negative constraints, `response_format: { type: "json_object" }` for provider JSON guarantee, `safeParse` for client-side validation, observability upgrade: separated JSON-parse failures from shape-mismatch failures with distinct warn logs (incl. `ZodError.issues`).
- **Stuck (well-internalized):**
  - **Provider JSON mode vs. Zod (syntactic vs structural validity).** Q2 strong on cold call — articulated the layering correctly.
  - **Soft vs hard constraint (prompt advice vs. `response_format` enforcement).** Q4 strong — got the core point that prompts are advice, response_format is enforced at the decoder.
  - **Spotting inconsistencies in teaching.** Caught that I taught `json_schema` as the strongest layer but implemented `json_object` — that's engineer-grade attention.
- **Fuzzy (needs reinforcement):**
  - **Five-part prompt structure (Role / Task / Format / Constraints / Examples).** Q1 — couldn't recall the framework on cold call. This is foundational; will reappear in 1.2 (injection defenses build on the same structure).
  - **When to NOT use few-shot (esp. reasoning models like o1/o3, Claude extended thinking).** Q3 — didn't attempt; surface again when we touch tool use / agents.
  - **Zod `.refine()` for cross-field invariants + the "silent failure" risk it prevents.** Q5 — didn't attempt; this pattern recurs every time we add structured output, will reinforce naturally.
  - **Feynman SHAPE (analogy → plain-language change → risk-removed close).** Content was there; pattern wasn't. The pattern was explicitly taught after 0.2's Feynman and not applied here. Practice the three-beat structure in Session 1.2.
  - **Calibrated hyperbole.** Claimed "most reliable of all industry standards" — those ARE the industry standards. Use "matches modern best practice" instead. Calibration tells > false confidence.

---

**2026-05-21 · Phase 1.2 — Prompt injection & output safety**
- Covered: the threat model (system role is statistical not enforced), four attack categories (direct, indirect, jailbreak, data exfiltration), defense-in-depth principle, 9-layer defense stack (1-5 free, 6-9 architectural/costly), the prompt-injection-resistant prompt template, output safety vs prompt-injection (complementary concerns).
- Built: hardened [summarizeTranscript](../src/lib/ai.ts) with `<transcript>` delimiters, CRITICAL SECURITY RULE in system prompt, sandwich line after few-shot examples, and a third few-shot example demonstrating injection-resistant output. Added [`CHAT_SYSTEM_PROMPT`](../src/app/api/chat/route.ts) to chat endpoint (it had no system prompt at all — naked user input went to the model) with security rules, delimiter discipline, and refusal-to-reveal-instructions clause.
- **Stuck (well-internalized):**
  - **Cost-asymmetry framing.** Q4 — got the "alternative to imperfect defense isn't perfect defense, it's no defense" instinct right.
  - **Direct vs indirect distinction.** Q1, Q2 — correctly identified indirect as Audia's worst risk and why (Deepgram → transcript pipeline).
  - **String-filtering is insufficient.** Q2 — got the paraphrasing/obfuscation counter quickly.
- **Fuzzy (needs reinforcement):**
  - **Full four-category taxonomy.** Q1 — recalled only "direct"; couldn't name jailbreak and data exfiltration. Drill the DIJD memory aid.
  - **Measurable signals for layer 6/7 escalation.** Q3 — answered "when it goes public" (correct but vague). Senior framing is *measurable triggers*: shape-mismatch rate, ticket volume, architectural change (adding tool-use), threat-profile shift (PII/regulated).
  - **Architectural least-privilege as primary defense.** Q4 — focused on prompt-level only; missed that the strongest defense is restricting what the model *can do* via architecture, not what we *tell it not to do* via prompts.
  - **Structural "zero vs one layer" framing.** Q5 — answered "user-facing" (directionally right but secondary). The senior insight: chat pre-1.2 had ZERO layers (no system prompt at all), summarizer had ONE — gap from 0→1 is structurally larger than 1→2.
  - **Inventing statistics.** Q4 — said "90-95% of attacks can be prevented" — fabricated number. Use qualitative framing in interviews: *"raises cost of attack significantly"* not made-up percentages.
  - **Feynman SHAPE — third session, still didn't apply organically.** Asked me to model it again. Pattern: analogy → what changed in plain language → risk removed. For Session 2, drill: brainstorm 3 candidate analogies first; pick the one where translation feels natural, not forced.

---

**2026-05-23 · Phase 2.1 — Streaming UX: cancellation, partial-save, lifecycle**
- Covered: full 7-step streaming lifecycle (request → connect → upstream call → TTFT → loop → close OR abort), protocol comparison (SSE / ReadableStream / WebSockets / Vercel AI SDK and when each), `AbortController` deep dive (signal propagation, AbortError differentiation, already-aborted behavior), the "abort as first-class control flow" pattern, partial-state persistence in `finally`, forwarding cancellation upstream to stop the token meter, the stop-button UX convention, micro-patterns (typing indicator, streaming cursor, unblocked input, optimistic empty message, auto-scroll).
- Built: server-side restructure of [chat/route.ts](../src/app/api/chat/route.ts) with try/catch/finally, `req.signal` forwarded to Groq SDK, `controller.enqueue` race-protected, AbortError filtering, `chat-aborted` log label for observability. Client-side [ChatPanel.tsx](../src/app/components/ChatPanel.tsx): `abortControllerRef`, `stop()` function, conditional Send/Stop button render with destructive red, AbortError differentiation (leave partial content visible), unblocked TextField during streaming.
- **Stuck (well-internalized):**
  - **TTFT terminology graduated to active recall** — used "time to first token" naturally in Q4 (was on fuzzy list from 0.2 and Q1 today). Compounds matter.
  - **"What we're NOT doing" framing** — Q3 used this structure unprompted; matches the senior pattern called out in the protocols deep-dive.
  - **Diagnostic methodology (check client first, then server)** — Q2 used systematic localization. Good debugging instinct.
- **Fuzzy (needs reinforcement):**
  - **Step 7 of the streaming lifecycle (clean close OR abort)** — Q1, omitted the load-bearing termination branch. This is the whole point of the session. Drill: the lifecycle has SEVEN steps; the 7th is the load-bearing one.
  - **Forwarding `signal: req.signal` as the specific line** — Q2 had right diagnosis but didn't name the specific line of code. Practice "narrow to the line, not just the layer."
  - **Three termination paths (clean close / error / abort) as distinct cases** — Q5 lumped error and abort together. They're different and `finally` exists because of the abort path specifically.
  - **Hedge phrasing in interviews** — Q4 said "a little complexity" which concedes the interviewer's premise. Senior counter: "actually less complex than buffering."
  - **Feynman attempt avoidance (fourth session running).** Asked me to model rather than attempt. The shape won't internalize without trying. Commit to attempting Session 3.1's Feynman *first*, even if bad — then I model after. This is the only fix.

---

**2026-05-23 · Phase 3.1 — Embeddings deep dive**
- Covered: what an embedding is (learned vector representation of text), the geometry of meaning (position = topic, direction = attribute, distance = dissimilarity), three similarity metrics with full operational math (cosine = angle, dot = same as cosine when normalized, L2 = geometric distance), embedding model selection across providers (OpenAI 3-small/large, Gemini, Cohere v3, sentence-transformers, BGE) with MTEB as the comparison axis, "embed once query many" economics, why switching models means re-embedding the corpus.
- Built: [src/lib/embeddings.ts](../src/lib/embeddings.ts) (initially text-embedding-004, patched to gemini-embedding-2 after deprecation 404, then patched again to move outputDimensionality to root after `config` body shape rejected), [src/lib/vector-math.ts](../src/lib/vector-math.ts) (dotProduct, norm, cosineSimilarity, l2Distance), [src/app/api/embed-demo/route.ts](../src/app/api/embed-demo/route.ts) — throwaway demo.
- **Stuck (well-internalized):**
  - **Cosine formula** — Q1 wrote it cold. Just missed the "why divide by norms" half.
  - **The math equivalence at norm=1** — Q2 best answer of the session (9/10). Connected "norm=1 in demo" to "dot=cosine in demo" as the same fact.
  - **Migration risk + cost intuition** — Q3 correctly raised cost/risk; correctly proposed sample-then-decide.
  - **Dim ≠ quality rebuttal** — Q4 correctly pushed back on "more dim = better."
  - **"Unrelated text scores ~0.3 not 0"** — Q5 correctly identified empirical range + correct practical threshold floor.
  - **Feynman attempt finally happened** — first attempt in 5 sessions. Big meta-win even though execution was weak.
- **Fuzzy (needs reinforcement):**
  - **"Why divide by norms" in cosine** — Q1 missed; the answer is "normalize for magnitude so we isolate the angle (direction in meaning-space) from length."
  - **MTEB by name** — Q4 didn't name it. The right reference for "model X vs Y" in interviews is MTEB.
  - **Dual-column / dual-write migration pattern** — Q3 proposed sequential migration. Senior pattern is parallel embedding columns + shadow retrieval + flip read traffic, not "backup, replace, hope."
  - **Cost math reflex** — Q3 said migrating 10k transcripts is "huge cost." Actual is ~$1.30 on OpenAI 3-large. Always calculate before claiming expense.
  - **"Probability distribution" terminology slip in Q5** — embeddings produce real-valued vectors and cosine is a real-valued metric in [−1, +1], NOT a probability. Watch terminology in interviews.
  - **Top-k + floor as production retrieval pattern** — Q5 mentioned threshold; senior pattern combines top-k with a similarity floor.
  - **Feynman SHAPE** — analogy was wrong (memory ≠ embeddings), jargon slipped ("our embeddings"), no risk-removed close. Practice: brainstorm 3 analogies and reject the bad ones before writing; ban specific words upfront; end every Feynman with "without this..."
- **Provider debugging real-world reps:** API hit two model changes in one session — text-embedding-004 deprecation (404 model not found) → gemini-embedding-2 (then `config` body shape rejected → outputDimensionality at root). Pattern reinforced: read the error body, patch one field, retry. Expected on first contact with any new provider API.

---

**2026-05-25 · Phase 3.2 — Chunking strategies**
- Covered: why chunking exists (context window limits + retrieval granularity + cost per query), four canonical strategies with definitions (fixed-window, sentence-based, recursive, semantic), the chunk size central trade-off (small = precise / no context; large = self-contained / wasteful), chunk overlap (10-20%) and why it prevents boundary-straddling failures, "lost in the middle" preview (Liu et al. 2023) and its chunking implication (favor smaller well-targeted chunks).
- Built: [src/lib/chunking.ts](../src/lib/chunking.ts) — segment-grouped fixed-window chunker respecting Deepgram's TranscriptSegment boundaries with configurable `targetChars` (default 1200 ≈ 300 tokens) and `overlapSegments` (default 1). Returns chunks with rich metadata (segmentIndices, speakers, startTime/endTime, charCount). [src/app/api/chunk-demo/route.ts](../src/app/api/chunk-demo/route.ts) — query-param-driven demo (`?target=400&overlap=1`) for live trade-off comparison on a 10-segment sample.
- **Stuck (well-internalized):**
  - **Four-strategy taxonomy by name** — Q1 recalled all four (fixed/sentence/recursive/semantic) cold. Cold-recall of taxonomies is rare and credible in interviews.
  - **Lost in the middle named unprompted** — Q3 used the term as the core pushback to "more context = better." Best argument of the quiz.
  - **Unified "Deepgram gave us structure" insight** — Q4 and Q5 both reasoned from the same underlying point (input is already segmented; recursive's boundary-finding job is redundant; character-overlap would re-break what segments fixed). He didn't explicitly state this unifier but the reasoning showed it.
  - **Overlap as the fix for boundary-straddling failures** — Q2 diagnosed correctly without prompting.
  - **Feynman attempted (second session running).** Pattern reinforcement.
- **Fuzzy (needs reinforcement):**
  - **Numeric calibration on "starting value" questions.** Q2 diagnosed overlap as missing but didn't propose a starting value (10-20% / `overlapSegments: 1`). Half-answers signal "concept known, ship not done." Drill: always pair diagnosis with calibration.
  - **Terminology precision.** Q4 said "user numbers" (meant speaker labels), Q5 said "semantic meaning bonded" (meant bounded). Voice-transcription noise but the underlying vocab needs to lock — say "speaker labels" and "semantically bounded" explicitly.
  - **Feynman SHAPE — banned-word discipline.** Used "chunk" 4 times despite explicit banned-word list. Analogy (bucket of water) mismatched the concept (chunking is about many small containers, not one with a size limit). "Three reasons" promise unfulfilled. Risk-removed close buried. Practice: write the banned-word list above the draft + test the analogy against the banned-word list before writing.
  - **Citation strength on lost-in-the-middle.** Q3 named the concept; can level up to "Liu et al. 2023" in the future for paper-citation credibility.
