# Phase 8 — Semantic search as UI

## Session 8.1 — Cross-encoder re-ranking & cross-transcript search

**Built in Audia:**
- New [src/lib/rerank-cross.ts](../src/lib/rerank-cross.ts) — `crossEncoderRerank(query, candidates, topK)`, Stage 2 of retrieve-and-rerank. Calls the **Jina Reranker API** (`jina-reranker-v2-base-multilingual`, a purpose-built cross-encoder, free tier). Returns candidates re-sorted by relevance with a `relevanceScore` in [0,1]. **Graceful degradation:** missing `JINA_API_KEY` or any API error → falls back to bi-encoder order, tagging score `NaN` so the UI can tell "not reranked" from a real 0.
- New [src/app/api/search/route.ts](../src/app/api/search/route.ts) — `GET /api/search?q=&k=&n=`. The full pipeline as a **user-facing surface**: `embed(q)` → `findCandidateChunks(n=30, all transcripts)` (Stage 1, bi-encoder recall) → `crossEncoderRerank(q, …, k=8)` (Stage 2, precision) → attach meeting titles (ownership-filtered `In(ids)` lookup). Returns `hits[]` with `relevanceScore` AND `biEncoderSimilarity` for transparency, plus per-stage timing.
- New [src/app/components/SearchResults.tsx](../src/app/components/SearchResults.tsx) — main-pane view (no LLM): ranked hits, each showing the cross-encoder relevance (purple % chip), meeting title, timestamp, speaker dots, 2-line snippet. Click → open that meeting.
- [src/app/components/SidebarSearch.tsx](../src/app/components/SidebarSearch.tsx) + [src/app/HomeClient.tsx](../src/app/HomeClient.tsx) — two behaviors on one box: **filter-as-you-type** (instant local substring filter, unchanged) + **search-on-Enter** (semantic content search across all meetings). New `mainView: "search"` state; `runSemanticSearch` / `clearSearch` / `openSearchHit` handlers.

### Concept summary

Everything in Audia's retrieval until now was a **bi-encoder**: query and documents embedded *separately* into 768-dim vectors, ranked by cosine (`pgvector <=>`). Fast (vectors precomputed, ANN over the whole corpus) but **lossy** — query and document never interact, so negation ("did NOT ship"), directionality ("Alice asked Bob"), and exact-phrase matches get flattened into a topic vector. That was tolerable when retrieval just fed an LLM (the model papered over a mediocre #1 chunk). Phase 8 exposes ranking **directly to the user** as a search results list — the moment the order is visible, bi-encoder precision isn't good enough.

The fix is a **cross-encoder**: feed the query and ONE document *together* through a transformer in a single pass, let their tokens attend to each other, emit one relevance score. Much more precise — but nothing can be precomputed (the score depends on the query), so it costs one forward pass *per candidate*. You can't run it over millions of chunks. Hence **retrieve-and-rerank**, the canonical two-stage RAG pipeline: Stage 1 bi-encoder narrows millions → coarse top-N (recall); Stage 2 cross-encoder re-scores those N → top-k (precision). The cross-encoder is affordable *only because* Stage 1 funneled the field first.

Cross-encoder ≠ MMR. **MMR optimizes diversity** (anti-redundancy); **a cross-encoder optimizes relevance precision**. They're complementary, not alternatives — a mature stack runs bi-encoder → cross-encoder → MMR. For a pure relevance-ranked search list we drop MMR (the user often *wants* the most-relevant moments in honest order). Scores are provider-dependent: a raw BGE cross-encoder emits **raw logits** (sigmoid them yourself; not comparable across queries); Jina's API returns a normalized [0,1] `relevance_score`. Either way you rank *within* one query, never threshold globally without calibration. And latency is now **user-facing** — the cross-encoder adds time proportional to N (an API round-trip here), which is why N stays small.

### 5 most-likely interview questions

1. **Q: Bi-encoder vs cross-encoder — what's the difference and when do you use each?**
   A: "A bi-encoder embeds the query and each document *separately* into vectors and compares by cosine — fast because document vectors are precomputed, so it scales to millions via ANN, but lossy because the two never interact. A cross-encoder feeds query and document *together* through a transformer and outputs one relevance score — far more precise because their tokens attend to each other, but it can't precompute anything, so it's one forward pass per candidate. You use them in sequence: bi-encoder for cheap recall over the whole corpus, cross-encoder to re-rank a small candidate set for precision. That's retrieve-and-rerank."

2. **Q: Why can't you just run the cross-encoder over your whole corpus?**
   A: "Because its score depends on the query, so nothing is precomputable — every query would need a full transformer forward pass for *every* document. At millions of chunks that's intractable per query and per keystroke. The bi-encoder's whole value is that document vectors are computed once and stored; the cross-encoder's whole cost is that they can't be. So you let the bi-encoder narrow millions to ~30, then spend the cross-encoder's budget only on those 30. The funnel is what makes the expensive stage affordable."

3. **Q: How is a cross-encoder different from MMR? You already had MMR.**
   A: "Different problems. MMR maximizes *diversity* — it penalizes a candidate for being too similar to what you've already picked, so you don't fill the context with five paraphrases of one moment. A cross-encoder maximizes *relevance precision* — is this document actually a good match for the query? They're complementary: production order is bi-encoder retrieve → cross-encoder rerank → MMR diversify. For a search results page I dropped MMR, because a user scanning results usually wants the most-relevant moments in honest order, duplicates and all — diversity matters more when you're packing a fixed LLM context budget."

4. **Q: The reranker returns a score — can you threshold on it, like score > 0.5?**
   A: "Depends on the provider, and you have to be careful. A raw BGE cross-encoder emits raw *logits* — unbounded, not probabilities, and not comparable across queries — so a global threshold is meaningless without a sigmoid plus calibration. Jina's API returns a normalized [0,1] relevance_score, which is friendlier, but I still treat it as a *within-query* ranking signal, not an absolute gate, unless I've calibrated it against labeled data. In Audia I rank by it and show it as a % chip; I don't drop results below a hardcoded cutoff."

5. **Q: Why does re-ranking show up specifically in the 'search as UI' phase and not earlier in RAG?**
   A: "Because earlier, retrieval was hidden plumbing feeding an LLM — the user saw the generated answer, which smoothed over a mediocre #1 chunk. The cost of imperfect ranking was low. A search results page exposes the ranking *directly*: the user reads result #1 first, there's no LLM in between. So bi-encoder precision that was 'good enough for RAG' becomes visibly wrong, and the cross-encoder's precision lift is suddenly worth its latency. Same retrieval machinery, but the quality bar moved because the consumer changed from a model to a human."

### Gotchas

- **Treating reranker scores as probabilities.** Raw BGE = logits; sigmoid first. Even normalized scores aren't cross-query comparable. Rank within a query.
- **No graceful degradation.** A reranker is an external dependency; if the API is down, search must still return bi-encoder results, not 500. We fall back to Stage-1 order with `NaN` score.
- **Running the cross-encoder on too-large N.** Latency scales with N. Keep N small (Stage 1's job); we use n=30, k=8.
- **Forgetting the cross-transcript ownership filter.** The chunk query is user-scoped; so is the *title* lookup (`In(ids)` + `userEmail`). Two queries, two ownership filters — defense in depth.
- **Mixing two search behaviors on one box without telling the user.** Filter-as-you-type vs search-on-Enter is discoverable only because of the helper hint under the input. Silent dual-mode confuses.
- **Importing a server-only route module into a client component.** Use `import type { SearchHit }` — type-only imports are erased at compile time, so no server code leaks into the client bundle.

### Operational notes

- **Requires `JINA_API_KEY`** in `.env` (free key at jina.ai). Without it, search degrades to bi-encoder order — which is also the cleanest **A/B eval**: toggle the key off → no % chips, pure cosine order; toggle on → reranked order + % chips. The reorder you see *is* the cross-encoder's precision lift.
- **Verify manually:** `npm run dev` → log in → press `/` → type a phrase about meeting *content* → Enter → ranked results with % chips; click → opens that meeting.
- **No automated eval yet** for search ranking — a Phase 8.2/6.3 follow-up would need labeled query→relevant-chunk pairs (RAGAS context precision/recall) to measure the lift numerically rather than by eye.
- **Cross-transcript retrieval was already a primitive** — `findCandidateChunks` with no `transcriptionId` searches all the user's meetings. 8.1 added the precision stage + the UI; the recall query existed since 3.3.
- **Deferred:** click-to-seek to the exact timestamp (currently opens the meeting; seeking needs lifting `seekTo` state into `SessionView` via a prop). MMR-then-rerank composition. Local-BGE option (no vendor).

---

## Session 8.2 — Hybrid search: BM25/lexical + dense, fused with RRF

**Built in Audia:**
- [src/lib/chunks.ts](../src/lib/chunks.ts) — new `findLexicalChunks(query, userEmail, {n})`: the **sparse/keyword arm** via Postgres full-text search (`websearch_to_tsquery` + `to_tsvector` + `ts_rank_cd`), ownership-filtered. Computed **inline** (no generated tsvector column / GIN index yet) — same "no premature indexing" stance as the deferred HNSW (3.3), and it dodges the TypeORM-synchronize-drops-orphan-column bug class. Documented the indexed-column upgrade path.
- [src/lib/rerank.ts](../src/lib/rerank.ts) — new generic `reciprocalRankFusion(lists, keyFn, k=60)`: fuses ranked lists on **rank**, returns items tagged with `rrfScore` + `sources` (which arms found them).
- [src/app/api/search/route.ts](../src/app/api/search/route.ts) — rewired to hybrid: `embed(q)` then **dense (`findSimilarChunks`) + lexical (`findLexicalChunks`) in parallel** → **RRF fuse** → cap to 30 → **cross-encoder rerank** (8.1) → titles. Response adds `sources` per hit + `arms` counts.
- [SearchResults.tsx](../src/app/components/SearchResults.tsx) — shows a per-hit source tag (`semantic` / `keyword` / **`both`** highlighted in primary) so the hybrid behavior is visible.
- `npx tsc --noEmit` clean. Query expansion taught as theory; implementation deferred.

### Concept summary

8.1 made ranking more **precise**; 8.2 makes retrieval **recall more**. The bridge insight: a cross-encoder can only re-rank what dense retrieval already surfaced — **you can't rerank a chunk that was never retrieved.** Dense (bi-encoder) search matches *meaning* but blurs rare exact tokens (IDs, codes, product names) into the topic vector; **lexical/BM25** search matches *exact terms* but is blind to paraphrase. Opposite blind spots → run both and fuse. BM25 scores by term frequency (with saturation, knob `k1`) × inverse document frequency × length-normalization (knob `b`); it's "sparse" because the representation is a mostly-zero vector over the vocabulary. The fusion problem: cosine (~0–1) and BM25 (unbounded) are incomparable scales, so you can't add them. **Reciprocal Rank Fusion** fuses on **rank** instead: `RRF(d) = Σ 1/(k + rank_list(d))`, `k=60` — scale-invariant, no tuning, and it rewards docs ranked high in *both* lists. Because RRF only uses ranks, the lexical arm doesn't need *true* BM25 — Postgres `ts_rank_cd` (not BM25) is fine; its only job is to order keyword hits, which is why losing ParadeDB's `pg_search` on Neon costs us nothing here. **Query expansion** (taught, not built) is the pre-retrieval recall lever: multi-query (LLM paraphrases → retrieve each → RRF-fuse), HyDE (embed a hypothetical answer, not the question), lexical synonym expansion — all trade an LLM call + drift risk for recall.

### 5 most-likely interview questions

1. **Q: You already had a cross-encoder reranker. Why add hybrid search?**
   A: "Because reranking and hybrid fix different stages. The cross-encoder improves *precision* on the candidate set dense retrieval handed it — but it can't surface a chunk that was never retrieved. If a rare exact term like an error code or product name got blurred by the bi-encoder and ranked #340, no reranker saves it; that's a *recall* failure. Hybrid adds a lexical arm that nails exact terms, fusing it with dense so the candidate set is better *before* reranking. Recall upstream, precision downstream — they stack."

2. **Q: Dense vs lexical — what does each miss?**
   A: "Dense/embedding search matches meaning, so it finds paraphrase and synonyms — 'how did we feel about the launch' → 'the release went great' — but it blurs rare exact tokens into a topic vector, so IDs, error codes, acronyms, and product names slip. Lexical/BM25 is the mirror: it nails those exact tokens via term frequency × IDF, but it's blind to synonyms and meaning. Opposite blind spots, which is exactly why you fuse them."

3. **Q: Why RRF instead of just adding or averaging the two scores?**
   A: "Because the scores live on incomparable scales — cosine is roughly 0–1, BM25 is unbounded and corpus-dependent. Adding them lets one dominate arbitrarily; normalizing (min-max, z-score) is fiddly and distribution-sensitive. RRF fuses on *rank* instead of score: each doc gets Σ 1/(k+rank) across the lists, k≈60. It's scale-invariant — only position matters — needs no tuning, and rewards docs that rank high in *both* lists. That last property is the quality signal: agreement across two independent methods."

4. **Q: You used Postgres full-text search, not real BM25. Doesn't that hurt quality?**
   A: "Barely, because of RRF. ts_rank_cd isn't true BM25 — it lacks BM25's exact IDF/saturation behavior — but RRF fuses on *rank*, so the lexical arm's raw scores are thrown away; all I need is for it to *order* keyword hits sensibly, which FTS does. The win from the lexical arm is recall (catching exact-term matches dense missed), not its score calibration. True BM25 via ParadeDB's pg_search would be marginally better ordering, but it's unavailable on new Neon projects as of March 2026, and the RRF design makes the gap immaterial."

5. **Q: What is query expansion and when would you reach for it?**
   A: "Rewriting or augmenting the query before retrieval to bridge vocabulary gaps and boost recall. Forms: multi-query (LLM generates N paraphrases, retrieve each, RRF-fuse the lists), HyDE (have the LLM write a hypothetical answer and embed *that* — a fake answer sits closer to real answers in embedding space than the question does), and lexical synonym expansion. You reach for it when hybrid alone still misses — but each form adds an LLM call (latency + cost) and risks drift, so it's a recall lever, not a default."

### Gotchas

- **Reranking can't fix recall.** If both arms miss a chunk, it's gone — no downstream stage recovers it. Hybrid is the recall fix; reranking is precision.
- **Never add raw dense + lexical scores.** Incomparable scales. Fuse on rank (RRF) or carefully normalize first.
- **RRF rank is 1-based.** The top item is rank 1, not 0 (1/(60+0) would over-weight it).
- **`websearch_to_tsquery`, not `to_tsquery`.** The former parses arbitrary user input safely (quotes, OR, negation) and never throws; `to_tsquery` throws on raw user text.
- **Don't add a tsvector column carelessly under `synchronize: true`.** TypeORM may drop an orphan column it doesn't know about (the pgvector bug). Inline FTS, or put the generated column on the entity.
- **Empty tsquery** (all stopwords) → zero lexical matches; the dense arm must still carry the result. Handled.

### Operational notes

- **Inline FTS, no index yet** — sequential ts match, fine at <10k chunks/user after the `userEmail` filter. Upgrade at scale: `tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED` column + GIN index.
- **Verify the hybrid lift:** search an **exact term** that appears verbatim in a meeting but is phrased oddly (an error code, a name). Dense-only (8.1) might miss it; hybrid should surface it with a `keyword` or `both` tag. Search a **conceptual** phrase with different words → `semantic`. A great hit shows **`both`**.
- **No automated eval yet** — needs labeled query→relevant-chunk pairs (RAGAS context recall) to measure the recall lift numerically.
- **Deferred:** query expansion implementation (multi-query / HyDE); hybrid in the *chat* RAG path (currently chat is dense+MMR; same `findLexicalChunks` + RRF would drop in); generated tsvector column + GIN index at scale.
