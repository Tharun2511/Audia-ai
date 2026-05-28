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

---

**2026-05-25 · Phase 3.3 — pgvector setup on Neon**
- Covered: what pgvector is (Postgres extension; 4 distance operators; 2 index types), distance operators in depth (`<->` L2, `<#>` negative inner product, `<=>` cosine distance, `<+>` L1), index trade-off (IVFFlat needs data before build; HNSW incremental + better but heavier), storage math (768-dim ≈ 3 KB per vector), the TypeORM + pgvector friction and the ALTER-TABLE-after-sync workaround, KNN query template with ownership filter, "no premature indexing" principle.
- Built: [src/entity/TranscriptChunk.ts](../src/entity/TranscriptChunk.ts), `ensurePgvector()` in [src/db/data-source.ts](../src/db/data-source.ts), [src/lib/chunks.ts](../src/lib/chunks.ts) (saveChunkWithEmbedding + findSimilarChunks), chunking+embedding wired into [transcribe pipeline](../src/app/api/transcribe/route.ts), [search-demo route](../src/app/api/search-demo/route.ts), removed throwaway embed-demo from 3.1.
- **Phase 3 milestone:** End-to-end vector pipeline working. Record a session → chunks created → embeddings stored. Search via `/api/search-demo?q=` returns top-k chunks ranked by cosine. Foundation complete for Phase 4 (wire retrieval into chat).
- **Stuck (well-internalized):**
  - **Pgvector operators by name** — Q1 recalled all four (`<->`, `<#>`, `<=>`, `<+>`) cold.
  - **Premature indexing instinct** — Q3 best-of-session 8/10. Got both sides cleanly: HNSW's value AND why it's premature at <10k vectors (memory + insert speed + no perceptible query gain).
  - **Pinecone-vs-pgvector pushback** — Q4 8/10 reached the "hybrid lexical + semantic via SQL joins" argument unprompted; that's senior-level.
  - **Ownership-filter security/perf duality** — Q2 right placement (WHERE before ORDER BY), right concept.
  - **Feynman trajectory** — 3.1 (3/10) → 3.2 (4/10) → 3.3 (6.5/10). First banned-word-clean run; analogy actually fit the concept; all three beats present.
- **Fuzzy (needs reinforcement):**
  - **Distance vs similarity terminology.** Q1 called `<=>` "cosine similarity" — it's cosine *distance* (1 − cos θ), range [0, 2]. The operator returns distance because ORDER BY ascending wants smallest = most similar. Memorize: pgvector ops return distances.
  - **L2 range** — Q1 said -inf to inf; correct is [0, ∞). Distances are non-negative by definition.
  - **L1/Manhattan description** — Q1 named it but didn't say what it measures (sum of absolute differences). Round out the answer when listing all four.
  - **Concrete numbers when answering vs vibes.** Q4 said "external overhead of infrastructure" instead of "$70/month." Always quote dollar amounts when comparing managed vs self-hosted.
  - **Codebase-walkthrough questions.** Q5 4/10 — only walked the new Phase 3.3 portion; missed the existing 7-step pipeline (Deepgram, audio upload, summary, transcription save) and cited zero files. Drill: re-read `transcribe/route.ts` end-to-end once per phase.
  - **Measurable-trigger phrasing.** When pushing back on premature optimization or default-tool choices, name the *measurable* trigger ("p95 latency above X", "corpus past Y vectors") not vibes ("when we grow").
  - **Feynman: opener.** Don't lead with "we built an important feature." Lead with the analogy.
  - **Feynman: risk-removed sharpness.** "Might not get correct book" → name the actual consequence (slow, expensive, fails past 30min).

---

**2026-05-25 · Phase 4.1 — Retrieval: top-k, MMR, hybrid, lost-in-the-middle**
- Covered: top-k's three failure modes (redundancy / missed exact matches / no diversity), MMR algorithm with λ knob (formula `λ·rel − (1−λ)·max-sim-to-selected`; λ=0.7 production default), hybrid search via BM25 + dense + RRF (theory only; implementation deferred to Phase 8.2), lost-in-the-middle U-curve and 3 mitigations (fewer chunks / edge-reorder / smaller chunks), full production retrieval pipeline (fan-out-and-narrow: embed → coarse N → optional hybrid → re-rank → optional lost-in-middle reorder → top-k → LLM).
- Built: extended [chunks.ts](../src/lib/chunks.ts) with `findCandidateChunks()` returning embeddings via `embedding::text` parse; new [rerank.ts](../src/lib/rerank.ts) with generic `maximalMarginalRelevance<T>()` — O(N×k), iterative greedy; updated [search-demo](../src/app/api/search-demo/route.ts) to return `naiveTopK` + `mmrReranked` side-by-side with per-stage timing.
- Debugged in-session: NULL-embedding orphans surfaced as `parseVector(null).slice` crash. Fixed by adding `AND embedding IS NOT NULL` to both retrieval queries. Root cause is non-atomic two-step write in `saveChunkWithEmbedding` (TypeORM save then UPDATE) — orphans possible if embed() fails between rows. Atomic-INSERT refactor offered, deferred.
- **Stuck (well-internalized):**
  - **MMR formula cold** — Q1 wrote it correctly with the right λ knob mental model.
  - **N=k pushback** — Q3 8.5/10. The senior insight ("MMR without coarse retrieval is theatrical / reduces to top-k regardless of λ") landed.
  - **"Lower λ" diagnostic for redundancy symptoms** — Q2 clean diagnose + fix.
  - **Feynman banned-word discipline** — second clean run; pattern is solidifying.
  - **Library + cover-pages analogy** — extension of 3.3's library analogy that translates "near-duplicate chunks" cleanly.
- **Fuzzy (needs reinforcement):**
  - **Cross-encoders vs MMR — "different problems" framing** — Q4 3/10. Said only "MMR eliminates redundancy." Missed the senior framing: MMR optimizes diversity, cross-encoders optimize relevance precision; they're complementary, not alternatives. Production stack runs both. The "X and Y solve different problems" rebuttal pattern applies to most tool-A-vs-tool-B interview questions.
  - **TypeORM/pg/pgvector layering** — Q5 6/10. Got the concept; missed the structural framing "the driver doesn't know vector at any layer" and the two-options choice (register custom encoder via pgvector-node, vs text round-trip).
  - **Feynman closer.** Banned words clean BUT lost the "without this..." final beat that was strong in 3.3. Rule for next session: **write the "without this" sentence FIRST**, then the rest of the Feynman, so it can't be forgotten.
  - **Feynman opener filler.** "This session mainly focused on..." is a junior tell. Lead with the analogy directly.
  - **Orphan-prevention via atomic writes** — surfaced as a real bug; deferred to optional refactor. When we get there, frame as "the non-transactional two-step pattern is a recurring bug class — always prefer single atomic writes when both columns are determined at the same time."

---

**2026-05-26 · Phase 4.2 — Generation with retrieval: templates, citations, hallucination control**
- Covered: RAG generation as post-retrieval phase, the 3-layer prompt template (system + numbered context + user input), three citation patterns (inline markers / structured JSON / post-hoc), three hallucination-control techniques (grounding rule / refusal anchoring / quote-then-answer), edge-position reordering for lost-in-the-middle, context budget math at Audia's scale, two protocols for citation metadata delivery (response header vs NDJSON envelope), the `Access-Control-Expose-Headers` gotcha.
- Built: full RAG pipeline in [chat/route.ts](../src/app/api/chat/route.ts) — embed → coarse N=20 → MMR (k=5, λ=0.7) → edgeReorder → numbered context block → stream LLM with `X-Citations` header. New `CHAT_SYSTEM_PROMPT` with grounding rules + citation format + security rules. [ChatPanel.tsx](../src/app/components/ChatPanel.tsx) sends `transcriptionId` not segments; parses `X-Citations` header; regex-splits `[N]` markers in streamed text; renders chips with Tooltip + click-to-seek. [SessionView.tsx](../src/app/components/SessionView.tsx) wires `onCitationClick → seekTo(startTime)`.
- **Phase 4 milestone:** "Chat with your transcripts" is real end-to-end. Audia now answers questions grounded in retrieved chunks, with clickable citations that seek the audio. The architectural shift (client sends reference, server retrieves) is what makes it scale beyond short meetings.
- **Mid-session bug fixes worth keeping:**
  - **Orphan NULL-embedding chunks** — root cause was non-atomic two-step write in `saveChunkWithEmbedding`. Refactored to single atomic INSERT with client-side `randomUUID()` + `validateEmbedding()` guard. Pattern: when two pieces of state must agree, write them in one statement, not two.
  - **`c.speakers.join is not a function`** — raw SQL via `db.query` returns `simple-json` columns as raw strings; TypeORM's repo auto-parses them but raw queries don't. Added `parseJsonColumn<T>()` helper that handles both cases. Pattern: at any boundary where raw-SQL data enters typed app code, explicitly parse anything the driver doesn't natively handle.
  - **Small-model hallucination on empty retrieval** — Llama-3.1-8b confidently invented "Sarah/John/budget" when sent a question with 0 chunks. Fixed via empty-retrieval short-circuit: skip the LLM call entirely, stream a deterministic "no indexed content" message. Pattern: architectural guards beat prompt rules for failure modes you can detect upfront.
  - **Stale chunks on transcript edit** — PATCH now regenerates chunks (delete → re-chunk → re-embed → re-save) when segments change; DELETE cascades to chunks. Strategy A from the design space.
  - **Backfill route** for existing transcriptions that pre-date Phase 3.3's chunking pipeline. One-off; delete after running.
- **Stuck (well-internalized):**
  - **RAG layers (system / context / user)** — Q1 8/10, all 3 citation patterns recalled.
  - **Inline markers vs JSON streaming trade-off** — Q3 8/10, the "JSON can't be partially parsed by Zod, kills TTFT" insight is interview-grade.
  - **Hallucinated chunk number client fallback** — Q2 7/10, right concept (render plain text, don't hide).
  - **Banned-word discipline in Feynman** — 4 sessions running. Solid.
  - **Atomic-write discipline** — internalized through the orphan-chunk bug fix in real time.
- **Fuzzy (needs reinforcement):**
  - **"Why bother with citations if model has chunks?" rebuttal** — Q4 6/10. Only verification angle. Missed: hallucination detection, retrieval-quality debugging, trust, audit-trail. The rebuttal pattern is "reject the framing — the model isn't the only consumer; the user is too."
  - **Honoring question scope.** Q5 walked the entire pipeline when asked specifically about client-side from fetch-resolve onward. Same gap as Phase 3.3 Q5. Drill for Phase 5+: when a question says "from X to Y," constrain the answer to that range.
  - **Feynman: opener with "without this..." sentence FIRST** — rule from 4.1 wasn't applied; risk-removed closer missing again. Try the mechanical constraint in 5.1: write the without-this line first, then the rest.
  - **Feynman: USING the analogy vs NAMING it** — 4.2's Wikipedia analogy was tacked on as a label, not used to STRUCTURE the explanation. Sentence test: if you can replace the analogy noun with "the feature" and the sentence still works, you've decorated, not constructed. In 5.1: write sentence 2 entirely in the analogy's vocabulary (no "Audia," no "the system," no "the AI" — just the analogy nouns).
- **Loose ends to follow up in Session 5.1:**
  - Backfill route at `/api/backfill-chunks` exists but hasn't been triggered by Tharun yet. Once run, delete the route.
  - Old transcripts (pre-Phase 3.3) may still need backfill before chat works on them.
  - The "context tag echo in model output" issue from the JS-transcript hallucination is likely fixed by the empty-retrieval short-circuit, but worth verifying on a session that does have chunks.

---

**2026-05-26 · Phase 5.1 — Conversation memory: rolling buffer, summary buffer, vector memory**
- Covered: LLM statelessness as the foundation (every turn is independent, conversation lives in the client's `messages[]`), the three memory families with definitions/algorithms/trade-offs (rolling buffer, summary buffer, vector memory), the production hybrid pattern (rolling + vector layered), cost economics (O(N²) unbounded vs O(N×K) capped — with real $ numbers on llama-3.1-8b vs gpt-4o), memory vs RAG as orthogonal layers, the `[N]` citation-marker-across-turns trap (strip on load, restore from stored citations metadata for UI replay), the persist-twice trap (mirror of Phase 4.2's orphan bug), the sessionId protocol (server-minted UUID, returned in `X-Chat-Session` header, multi-tenant security via `WHERE userEmail = ?` on history loads).
- Built: [src/entity/ChatMessage.ts](../src/entity/ChatMessage.ts) with composite `(sessionId, createdAt)` index; [src/lib/chat-memory.ts](../src/lib/chat-memory.ts) (`loadRecentTurns` DESC+LIMIT+reverse with citation-marker stripping, `saveTurn`, `HISTORY_TURN_PAIRS=5`); rolling-buffer wiring in [chat/route.ts](../src/app/api/chat/route.ts) — accepts `sessionId | null`, mints on null, loads last-5 turn pairs, persists user turn before stream + assistant turn in `finally`, extended `CHAT_SYSTEM_PROMPT` with CONVERSATION RULES + cross-turn `[N]` caveat, returns `X-Chat-Session` header; [ChatPanel.tsx](../src/app/components/ChatPanel.tsx) — `sessionId` state, echoes on subsequent fetches, new Reset (↻) IconButton aborts in-flight stream + clears messages + clears sessionId; entity registered in [data-source.ts](../src/db/data-source.ts).
- **Phase 5 milestone (partial):** chat is no longer amnesiac. Follow-up questions like "and what about Bob?" work. Foundation laid for 5.2 (vector memory) and Phase 7 (agents with persistent context).
- **Stuck (well-internalized):**
  - **Consequence-reasoning on systems-design trade-offs** — Q3 (summary buffer for gist vs fact-precise chat) 7/10 and Q4 (localStorage durability + portability) 6.5/10 both reasoned cleanly from first principles to real consequences. This is his consistent strength.
  - **Empty-window behavior** — Q2 correctly predicted that an answer depending on turn 3 with a last-5 buffer produces hallucination/unrelated output. Got the mechanism (turn outside window → not in prompt → model can't see it).
  - **Feynman risk-removed CLOSE finally present** — absent in 4.1 AND 4.2; reappeared here. Amnesia-patient analogy was his best-FITTING pick yet (statelessness ↔ no recall between questions).
- **Fuzzy (needs reinforcement):**
  - **Naming the precise mechanism/term (recurring weakness).** Q1 (4/10) answered "why not store centrally" instead of naming **statelessness** — the model is a pure function of input tokens, nothing persists between calls. Q5 (4/10) explained `.reverse()` (presentation order) but missed why **DESC is mandatory**: `ASC LIMIT N` selects the OLDEST N rows, not the newest — DESC is row-selection, reverse is reading-order. Same shape as Phase 3.3/4.2 Q5. Fix reflex: when asked "name the property" / "what concrete bug," lead with the term, then reasoning.
  - **bounded vs unbounded confusion** — Q2 attributed "too many tokens / unnecessary details" (the unbounded symptom) to the last-5 case, where the problem is the OPPOSITE (too little context). Keep the two regimes distinct.
  - **Precise fix-naming** — Q2 said "chunking the chat history" for the turn-3-outside-window fix; the named answer is **vector memory** (embed past turns, retrieve relevant ones) or hybrid rolling+vector. Same retrieval machinery as Phase 4, pointed at turns.
  - **Summary drift + extra-LLM-call** — Q3 named detail-loss but not the compounding **summary drift** failure mode nor the per-turn second-LLM-call cost/latency.
  - **Tampering surface** — Q4 missed the strongest senior point: client-supplied history is forgeable (jailbreak/exfil surface), so the server must own it.
  - **Feynman: USING vs NAMING (the live bottleneck) + banned-word slip.** Used banned word "memory" 2-3× (breaking a 4-session clean streak) precisely BECAUSE he named the analogy ("implemented memory for him") instead of extending it into analogy-native nouns (notepad / jots / glances / scribbled-over). Close was circular (restated the feature) not consequence-named. Drill for 5.2: list the analogy's 3-4 nouns BEFORE writing; build the mechanism from only those nouns; if reaching for "memory"/"context," he's slipped to naming.
- **Loose ends to follow up in Session 5.2:**
  - Verify rolling-buffer behavior end-to-end in the browser: ask Q1, ask follow-up "and the other speaker?", confirm answer uses Q1 context.
  - The empty-retrieval short-circuit now also persists turns. Verify a transcript with zero chunks generates a coherent two-row session record (user + deterministic refusal).
  - Backfill route at `/api/backfill-chunks` still not triggered. Run + delete soon.
  - Consider promoting `ChatMessage` → `ChatSession + ChatMessage` (two tables) once we add session-listing UI in 5.2.

---

**2026-05-26 · Phase 6.1 — Eval theory: offline/online, metric families, golden sets, LLM-as-judge**
- Covered: evals as the unit-test of non-deterministic systems (graded measurement, not exact-match); offline vs online (golden set/CI vs production-traffic feedback, the flywheel); the four metric families (programmatic — cheap/exact/first; LLM-as-judge — modern default; BLEU/ROUGE — n-gram overlap, weak because words≠meaning; human — gold standard, validates the judge); golden-set construction (coverage>volume, hold out few-shots, version-control, criteria not exact strings); LLM-as-judge mechanics (pointwise vs pairwise, position/verbosity/self-enhancement bias + mitigations, specific rubric, low-cardinality verdicts, reason-before-score, Cohen's kappa human-validation); RAG-specific eval (RAGAS: context precision/recall for retrieval, faithfulness/answer-relevancy for generation — diagnose separately); pass rate as headline metric + the "100% = set too easy" insight.
- Built: refactored [ai.ts](../src/lib/ai.ts) into `summarizeTranscriptStructured()` (eval seam) + `summarizeTranscript()` (display wrapper, unchanged contract); [golden-summary.ts](../src/evals/golden-summary.ts) (15 held-out cases); [judge.ts](../src/evals/judge.ts) (faithfulness judge on llama-3.3-70b-versatile — different/stronger model to dodge self-enhancement bias, pass/fail, reason-first, temp 0); [summary.eval.ts](../src/evals/summary.eval.ts) (programmatic-first then judge, per-case table + pass rate, exit 1 below 90%); `npm run eval` wired with the server-only/tsx/env-file flag trio. **First run: 15/15 (100%).**
- Build chose: summarizer surface (programmatic-first teaching arc) + standalone `npm run eval` script (primitives-first, clean CI seam for 6.2) over RAG-chat / vitest.
- Debugged in-session: `import "server-only"` → `MODULE_NOT_FOUND` running outside Next (it's bundler-provided, not on disk). Fix: `npm i -D server-only` + `--conditions=react-server` resolves its no-op `empty.js`. Plus tsx for `@/*` paths + `--env-file=.env` for GROQ key. Good "re-create what the framework's bundler provided" talking point.
- **Phase 6 milestone (partial):** Audia has its first real eval harness — vibes-based dev is over for the summarizer. 6.2 = expand the set + RAG-chat evals + CI gate.
- **Stuck (well-internalized):**
  - **🎯 TERM-FIRST REFLEX LANDED — Q3 (8.5/10).** Said *"That is self-enhancement bias"* then explained. This is the single reflex we've been drilling since 3.3/4.2/5.1, and today was the cleanest hit. Acknowledge + lock — this is the most important pattern of the session.
  - **Signal-attribution reasoning** — Q5 (8.5/10) articulated *why* programmatic-before-judge matters not just for cost but for diagnostic chain integrity ("if both fail you can't tell which verdict matters; if programmatic fails but judge passes, the judge is wrong"). Senior framing without prompting.
  - **Set-too-easy instinct** — Q2 (8/10) flagged 15/15 as "abnormal for production-level model" — the right instinct ("100% on a fresh set means the set isn't adversarial yet, not that the system is correct"), even if the exact phrasing wasn't there.
  - **All four metric families named cold** — Q1 listed reference/LLM-as-judge/programmatic/human. Taxonomies are recall-strong.
- **Fuzzy (needs reinforcement):**
  - **Wrong about which family Audia leans on most** — Q1 (5/10) said "LLM-as-judge for transcription" but Audia leans on **programmatic** (5-8 checks per case: schema, tooShort, bullet count, mustMention/mustNotMention substring) with judge as a thin faithfulness layer on top. The whole point of choosing summarizer first was the programmatic-heavy teaching arc.
  - **Reference-metric description slid into programmatic territory** — Q1 said reference = "search for exact keywords." That's regex/programmatic. Reference = n-gram overlap with a *gold text*. Keep the families distinct.
  - **"Non-determinism" mis-named as BLEU/ROUGE's failure cause** — Q4 (6.5/10) attributed BLEU's weakness to LLMs being non-deterministic. Actual cause: **n-gram overlap measures surface words, not meaning** — true even at temp 0. Right consequence, wrong mechanism. Same shape as Phase 3.3/5.1 Q1: consequence lands, term doesn't always anchor.
  - **Judge un-validated** — Q2 missed the additional "we haven't spot-checked the judge against humans (Cohen's kappa) so its passes are themselves unverified." Worth surfacing in future eval-trust questions.
  - **Implicit choices** — Q3 explained why Option A is bad but didn't explicitly say "therefore Option B." Senior delivery: name the term, state the choice, then justify.
- **🧠 Feynman — 8/10, structural breakthrough.** Analogy: structured-interview-with-rubric vs vibes-based-interview. Trajectory: 3.1 (3) → 3.2 (4) → 3.3 (6.5) → 4.1 (5.5) → 4.2 (4.5) → 5.1 (6) → **6.1 (8)** — three sessions of climb, this one a structural inflection.
  - **🎯 Both long-standing mechanical constraints landed in the same attempt:**
    1. *"Without this..." line FIRST* — opened with the risk-removed sentence. The rule from 4.1/5.1 finally applied without reminder.
    2. *USE not NAME* — every mechanism sentence built from analogy-native nouns (guidelines, rules, questions, answers, "words that should be mentioned"). Sentence test passes: swapping "interviewer" for "the feature" breaks the sentences — the analogy is load-bearing, not decorative. *"Words that should be mentioned"* mapped directly to `mustMention` substrings — concrete implementation noun via analogy vocabulary.
  - **Banned-word discipline mostly clean.** One slip: "judge him whether he qualified" — "judge" was banned. Hard because it's the analogy's natural verb. Single slip, much cleaner than 5.1's three "memory" slips.
  - **Weakness: close trails off.** Ended on *"the set of words that should be mentioned in those answers"* — descriptive not punchy. Stronger close lands the value in analogy vocabulary (e.g. "...so now the interviewer can defend his call — 'he qualified because he hit these specific answers' — instead of 'I had a good feeling about him.'"). Drill for 6.2: write the close before the body too, mirroring the opener rule.
  - **Headroom unused.** Could've extended the analogy to cover the judge layer (a senior interviewer double-checking the junior's notes = different/stronger model dodging self-enhancement bias) or the "100% pass = set too easy" insight (all candidates acing = questions too easy, not great candidates). Optional but available.
  - **Big-picture pattern.** Same shape as Q3's term-first reflex earlier this session: corrections from prior sessions compounding into automatic behavior. The Feynman drill is starting to internalize as habit, not as a checklist.
- **Loose ends to follow up in Session 6.2:**
  - Harden the golden set: add cases the 8B is likely to FAIL (long multi-number meetings, subtle injections, ambiguous too-short). 15/15 means it's too easy right now.
  - Validate the judge against human spot-checks (~20) before trusting it as a gate — currently unvalidated.
  - judge.ts doesn't call `logUsage` — the 70B judge calls are uninstrumented. Wire cost logging if we care about eval-run cost.
  - Build the RAG-chat eval (RAGAS-style faithfulness + context metrics) + wire `npm run eval` into CI with the threshold gate.
  - Still-open from 5.2: browser verification of rolling buffer; backfill route run+delete.
