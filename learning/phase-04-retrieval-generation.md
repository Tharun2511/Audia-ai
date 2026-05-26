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
