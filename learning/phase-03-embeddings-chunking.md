# Phase 3 — RAG part 1: embeddings, chunking, pgvector

## Session 3.1 — Embeddings deep dive

**Built in Audia:**
- New [src/lib/embeddings.ts](../src/lib/embeddings.ts) — `embed(text)` function calling Gemini's `text-embedding-004` (free tier, 768-dim, pre-normalized output)
- New [src/lib/vector-math.ts](../src/lib/vector-math.ts) — `dotProduct`, `norm`, `cosineSimilarity`, `l2Distance`. Pure functions, easy to test, will resurface in pgvector SQL by 3.3
- New throwaway [src/app/api/embed-demo/route.ts](../src/app/api/embed-demo/route.ts) — embeds 5 sample sentences, computes pairwise cosine + dot product, sorts by similarity. Confirms (a) Gemini returns pre-normalized vectors (norm = 1.0), (b) dot product equals cosine when normalized, (c) semantically-related pairs rank highest. Will delete in Phase 3.3.

### Concept summary

An embedding is a learned vector representation of text — a neural net trained on billions of text pairs maps any input to a fixed-size float array (typically 768 or 1536 dims) such that **semantic closeness ≈ geometric closeness in the vector space**. RAG retrieval reduces to: embed the user's question, find the top-k nearest chunks in vector space, feed those to the LLM as context. Three similarity metrics: **cosine** (default for text, measures angle), **dot product** (cheaper, equals cosine for unit-normalized vectors), **L2** (geometric distance, for non-normalized). The economics are "embed once at ingest, query many" — corpus embedding amortizes across millions of searches, but switching models means re-embedding the whole corpus because vectors from different models live in incompatible spaces.

### 5 most-likely interview questions

1. **Q: What's an embedding and how does it enable RAG?**
   A: A learned vector representation of text — typically 768 or 1536 floats — with the property that semantically similar texts produce nearby vectors in the space. RAG exploits this: at ingest, chunk the corpus and embed each chunk, store the vectors. At query time, embed the question and use a similarity metric (cosine for text, dot product if vectors are unit-normalized) to find the top-k nearest chunks. Those become the context the LLM sees alongside the question. Embedding turns "search by exact keyword" into "search by meaning."

2. **Q: Cosine vs dot product vs L2 — when do you use which?**
   A: Cosine measures angle, ignoring magnitude — universal default for text because sentence length shouldn't bias similarity. If vectors are unit-normalized (modern APIs like Gemini's text-embedding-004 give you this by default), dot product equals cosine and is faster. L2 measures geometric distance; for unit-normalized vectors produces the same ranking as cosine. Rule: cosine for text, L2 for non-normalized (some image embeddings).

3. **Q: How would you pick an embedding model for production?**
   A: Three axes. Quality — check MTEB rankings on tasks relevant to the corpus, since dim count isn't a linear quality signal. Cost — corpus of 10M tokens is ~$200 one-time on OpenAI text-embedding-3-small, free on Gemini text-embedding-004 or open-weight BGE. Operational — latency, rate limits, provider stability. For English-only RAG I'd default to OpenAI text-embedding-3-small until cost forces a switch; for free/learning use Gemini; for multilingual Cohere embed-v3.

4. **Q: What happens if I want to change embedding models on an existing RAG system?**
   A: You re-embed the entire corpus, because vectors from different models live in incompatible spaces — cross-model cosine similarity is meaningless. Strategic decision, not casual. Production pattern: version the storage layer (column per model version or separate tables), migrate during low-traffic windows, have a rollback plan. For Audia today, since we don't have users yet, model changes are cheap; once we do, it becomes a real migration problem.

5. **Q: What's the "useful" similarity range I should expect for text embeddings?**
   A: Roughly **0.3 to 0.95** for modern dense embeddings. True 0 and true 1 are rare — modern embedding spaces are dense, nothing is truly orthogonal. So thresholds like "similarity > 0.8" need to be set empirically per corpus, not absolutely. In Audia's demo route, even completely unrelated sentences ("I love pizza" vs "interest rates rose 2%") score ~0.3 cosine.

### Gotchas

- **Mixing embedding models is a category error.** Different models live in incompatible spaces. Don't compare or aggregate across models.
- **Modern embedding spaces are dense.** "Unrelated" still scores ~0.3 cosine. Don't expect 0 for unrelated text.
- **Dim count is NOT a linear quality indicator.** A well-trained 384-dim model can beat a poorly-trained 3072-dim model. Use MTEB.
- **Some models return unnormalized vectors.** If `norm(v) ≠ 1.0`, dot product is biased by magnitude. Use cosine, or normalize first.
- **Embedding the wrong thing.** Embed semantically-meaningful chunks, not random splits. Phase 3.2 covers this.

### Go-deeper resources

- Jay Alammar, [*"The Illustrated Word2Vec"*](https://jalammar.github.io/illustrated-word2vec/) — foundational intuition
- [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — go-to comparison for embedding models
- Cohere blog on embeddings — strong production framing
