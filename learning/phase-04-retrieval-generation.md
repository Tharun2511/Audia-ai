# Phase 4 — RAG part 2: retrieval & generation

## Session 4.1 — Retrieval: top-k, MMR, hybrid, lost-in-the-middle

**Built in Audia:**
- Extended [src/lib/chunks.ts](../src/lib/chunks.ts) with `findCandidateChunks()` — wide top-N retrieval that includes the embedding column (parses pgvector's text output `[1,2,3]` back to `number[]` for in-memory use)
- New [src/lib/rerank.ts](../src/lib/rerank.ts) with `maximalMarginalRelevance<T>()` — generic MMR over any embedded objects; iterates picking the candidate maximizing `λ·rel − (1−λ)·max-sim-to-selected`. λ default 0.7
- Updated [search-demo route](../src/app/api/search-demo/route.ts) to return BOTH `naiveTopK` and `mmrReranked` side-by-side with timing per stage; accepts `?k=`, `?n=`, `?lambda=`, `?transcript=`

### Concept summary

Naive top-k retrieval (`ORDER BY distance LIMIT k`) has three known failure modes: redundancy (near-duplicate chunks), missed exact-keyword matches, no diversity. Production RAG layers re-ranking on top: MMR for diversity-aware selection, hybrid search (BM25 + dense via RRF) for keyword robustness, and edge-position reordering for lost-in-the-middle attention decay. The pattern is "fan out wide, narrow at each stage" — coarse vector retrieval gets top-N (3-5× the final k), re-ranking picks top-k by a richer criterion. Audia today does step 1 (coarse) + step 2 (MMR re-rank). Phase 4.2 adds the generation stage with citations and hallucination control; Phase 8.2 adds hybrid search.

### 5 most-likely interview questions

1. **Q: Walk me through your retrieval pipeline.**
   A: Fan-out-and-narrow pattern. Step 1: embed query with the same model used at ingest. Step 2: vector top-N from pgvector with ownership filter — `WHERE userEmail = $1 ORDER BY embedding <=> $2 LIMIT n`, N=20 (3-5× final k). Step 3 (deferred to Phase 8.2): hybrid via BM25 + RRF. Step 4: MMR re-rank with λ=0.7 to balance relevance and diversity, eliminates near-duplicates from naive top-k. Step 5 (next session, 4.2): edge-position reorder for lost-in-the-middle. Step 6: take top-k = 5. Step 7: pass to LLM with citation markers.

2. **Q: What's MMR and when do you use it?**
   A: Maximal Marginal Relevance — re-ranking algorithm picking k items balancing relevance to query against diversity from already-selected. Formula: `λ·sim(candidate, query) − (1−λ)·max-sim(candidate, selected)`. Pick highest scorer, add, repeat. λ knob: 1.0 = pure relevance (collapses to top-k), 0.7 = production default, 0.5 = balanced, 0.0 = pure diversity (usually wrong). Use when top-k may return near-duplicates — most RAG workloads. Cost O(N×k); negligible at typical N=20, k=5.

3. **Q: Why hybrid search? Isn't vector search enough?**
   A: Vector embeddings encode meaning but lose surface form. If a query has specific terms — names, codes, exact phrases — vector search may miss documents that contain them literally, because surrounding semantic context doesn't match. BM25 catches exact terms; vectors catch paraphrases. Combine via RRF (Reciprocal Rank Fusion): sum `1/(60 + rank)` across rankers per document. Robust to score-scale differences (BM25 raw scores vs cosine in [0,2]); pulls documents that rank well in either.

4. **Q: How do you handle lost-in-the-middle?**
   A: Three layers. Keep k small — pass 3-5 chunks, not 10. After re-ranking by relevance, reorder for placement: most-relevant at position 0 AND k-1 (the edges where attention is strongest), least-relevant in the middle. Smaller chunks at ingest so each one is fully visible. The Liu et al. 2023 paper named the U-curve failure mode and is the citation.

5. **Q: Why fetch top-N=20 candidates with embeddings before re-ranking, instead of fetching top-k=5 directly?**
   A: Re-rankers need a candidate pool. If I fetch only 5 and "re-rank" 5, the choices were already locked in by the vector search — the re-rank step is theatrical. Wide coarse retrieval (3-5× k) gives re-rankers room to find better picks: less-similar-to-query but more-diverse, more keyword-relevant, better-positioned for context. The wider N also helps with recall — if the right chunk ranks 8th by cosine but is uniquely relevant in some other dimension, naive top-5 missed it. The cost of going from N=5 to N=20 is tiny (pgvector scan is the same, just LIMIT is higher); the benefit to re-rank quality is substantial.

### Gotchas

- **Skipping the wide coarse retrieval.** MMR on 5 chunks "chosen from" 5 chunks is just top-k with extra steps. Pull 3-5× k.
- **Hardcoding λ=0.5.** Most production teams find λ=0.7 works better — relevance leads, diversity tie-breaks.
- **Running re-rankers without candidate embeddings.** MMR needs vectors. Don't try to re-rank `(text, distance)` tuples — fetch with the embedding column.
- **Pure-diversity λ=0.0.** Ignores the query. Almost never correct.
- **Assuming re-ranking compensates for bad coarse retrieval.** If the right chunk isn't in top-N, no re-ranker can surface it. Tune N up if recall is the limit.
- **No ownership filter at the SQL layer.** Re-ranking a leaked cross-tenant result is still a leak. WHERE clause comes first.

### Go-deeper resources

- Pinecone, [*"Re-ranking with Maximal Marginal Relevance"*](https://www.pinecone.io/learn/maximal-marginal-relevance/) — primary
- Carbonell & Goldstein 1998 — the original MMR paper, two pages
- Liu et al. 2023 [*"Lost in the Middle"*](https://arxiv.org/abs/2307.03172) — the U-curve paper
- Cormack, Clarke & Buettcher 2009 — original RRF paper, three pages
- Lewis et al. 2020 — original RAG paper

---

## Session 4.2 — Generation with retrieval: templates, citations, hallucination control

**Built in Audia:**
- [chat/route.ts](../src/app/api/chat/route.ts) became the full RAG pipeline: embed question → `findCandidateChunks(n=20)` → MMR re-rank (k=5, λ=0.7) → `edgeReorder()` → numbered context block → streaming LLM response with citation header. New `CHAT_SYSTEM_PROMPT` has grounding rules ("use ONLY context, say so if not present") + citation format spec (inline `[N]` markers) + security rules from Phase 1.2.
- [ChatPanel.tsx](../src/app/components/ChatPanel.tsx) now sends `{question, transcriptionId}` instead of the whole transcript. Reads `X-Citations` response header before streaming, parses `[N]` markers in streamed text via regex split, renders them as clickable MUI Chips with Tooltip metadata (speaker + timestamp + preview), routes click to `onCitationClick(startTime)`.
- [SessionView.tsx](../src/app/components/SessionView.tsx) wires `onCitationClick → seekTo(startTime)` so clicking a chip seeks the audio player to that exact moment.
- New `edgeReorder()` helper in chat route — 10-line loop that places most-relevant chunks at positions 0 and k-1 to mitigate lost-in-the-middle.

### Concept summary

RAG generation is the post-retrieval phase: take retrieved chunks, insert into a structured prompt with grounding rules ("use ONLY context, refuse if not present") and citation format ("cite with `[N]` markers"), stream the LLM response, parse citations on the client. The architectural shift in Audia is the client now sends a reference (transcriptionId) rather than the data itself — server does retrieval, only top-k chunks reach the LLM. Three citation patterns: inline markers (streams, fuzzy), structured JSON (strict, doesn't stream), post-hoc attribution (most accurate, extra call). Three hallucination control techniques layered as needed: explicit grounding rule (always), refusal anchoring few-shot (sometimes), quote-then-answer (rarely, expensive). Custom response headers need `Access-Control-Expose-Headers` or fetch hides them silently.

### 5 most-likely interview questions

1. **Q: Walk me through your RAG chat end-to-end.**
   A: Client sends `{question, transcriptionId}` to chat route. Server embeds the question with Gemini text-embedding-2, calls `findCandidateChunks(n=20)` from pgvector with ownership filter, MMR re-ranks to k=5 with λ=0.7 for diversity, edge-position reorders so most-relevant chunks land at positions 0 and k-1, builds a numbered context block, wraps in `<context>` tags alongside the question in `<user_input>` tags. System prompt has grounding rules + citation format + security rules. Streams the LLM response from Groq llama-3.1-8b-instant. Citation metadata travels in `X-Citations` response header (with `Access-Control-Expose-Headers` so browser exposes it). Client parses `[N]` markers in streamed text, renders as clickable chips with tooltips; click seeks audio to chunk timestamp. Total prompt ~3500 of 8k tokens — comfortable.

2. **Q: How do you prevent hallucination?**
   A: Three layers. Explicit grounding rule in system prompt ("Use ONLY the context. Say so if not present.") — free, baseline. Optional refusal anchoring — few-shot example of refusal — adds tokens but reinforces pattern. Optional quote-then-answer — instruct model to quote supporting text before each claim — most expensive but tightest grounding. Layer count depends on stakes: Audia uses layer 1; Phase 6 evals would tell me whether to add 2 or 3. Architectural control matters more than prompts: bad retrieval breaks any grounding rule. Wide coarse retrieval + re-rank is the foundation.

3. **Q: Why send citations in a header instead of in the streamed body?**
   A: Simplicity. Body stays pure text tokens — existing reader loop unchanged. NDJSON envelope would force a client refactor and mix protocol with content. Citation payload is fully known up front — server has the chunks before any tokens stream — so sending once via header fits naturally. Trade-off: header size limit (~8KB total in most clients). At 5 chunks × ~200 bytes each = 1KB, comfortable. Critical detail: set `Access-Control-Expose-Headers: X-Citations` or the browser hides the custom header from fetch — silent failure.

4. **Q: What's the architectural shift in Phase 4.2?**
   A: Before: chat client sent the entire transcript with every question. Worked on 5-min meetings, broken on 2-hour ones — context window saturated, cost scaled with meeting length not relevance. After: client sends only a reference (transcriptionId) plus the question; server does retrieval scoped to that transcript and includes only the top-k relevant chunks in the prompt. Scales to any meeting length; per-request cost is bounded by k, not by total content. Client is also lighter — chat feature doesn't hold the whole transcript anymore.

5. **Q: Why use inline `[N]` markers instead of structured JSON citations?**
   A: Streaming. JSON output via `response_format: { type: "json_object" }` doesn't stream cleanly — you can't render a partial JSON object incrementally without complex partial-parse logic. Inline `[N]` markers stream as part of the text and can be parsed by a simple regex split on the client. Trade-off: model can hallucinate citation numbers (cite `[7]` when we only sent 5 chunks) — we handle this gracefully by falling back to plain text. Structured JSON wins when citation correctness matters more than streaming UX (legal, medical, audit-trail systems).

### Gotchas

- **`Access-Control-Expose-Headers` missing** → browser hides custom header from fetch; client never sees citations. Silent fail.
- **Model cites unknown chunk numbers** → client must render fallback. Don't assume every `[N]` has a matching citation in the map.
- **Header size limits** → 5-10 chunks of metadata fits; 50 would not. For larger citation payloads, switch to NDJSON.
- **Forgetting to send `transcriptionId`** in the request body → server retrieves across ALL user's transcripts. Sometimes wanted, sometimes not — be explicit.
- **Grounding rule too soft** ("try to use the context") → models will use it loosely. The rule must be "ONLY", with explicit "say so if not present" refusal directive.
- **Top-k too large for context window** → math the budget. At chunk size 300 tokens, k=20 fills the 8k window before the answer even starts.

### Go-deeper resources

- Anthropic, [*RAG cookbook*](https://docs.anthropic.com/en/docs/build-with-claude/contextual-retrieval) — best single explainer of the production pipeline
- OpenAI cookbook on grounded answers + citations
- Liu et al. 2023, *"Lost in the Middle"* — same paper as 4.1, now applied via edge-reorder
- Lewis et al. 2020, *"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"* — the original RAG paper
