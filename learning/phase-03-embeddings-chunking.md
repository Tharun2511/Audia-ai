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

---

## Session 3.2 — Chunking strategies

**Built in Audia:** [src/lib/chunking.ts](../src/lib/chunking.ts) — `chunkTranscript(segments, { targetChars, overlapSegments })` that walks Deepgram's `TranscriptSegment[]`, groups into segment-bounded chunks of target size, and emits chunks with `text`, `segmentIndices`, `speakers`, `startTime`, `endTime`, `charCount`. Demo route [src/app/api/chunk-demo/route.ts](../src/app/api/chunk-demo/route.ts) accepts `?target=` and `?overlap=` query params so you can compare strategies live on a 10-segment sample.

### Concept summary

Chunking splits a source document into smaller, embeddable units. It's forced by three things: context window limits, retrieval granularity (whole-doc retrieval returns mostly noise), and per-query cost. Four canonical strategies — fixed-window (simplest, ignores boundaries), sentence-based (good for prose, inconsistent sizes), recursive (industry default — split at largest natural boundary), and semantic (embed-driven, highest quality, highest cost). Chunk size is the central trade-off: small chunks retrieve precisely but lose context; large chunks are self-contained but waste budget. Overlap (10-20% of chunk size) prevents context loss at boundaries. Tune chunk size empirically via recall@k on a golden set (Phase 6).

### 5 most-likely interview questions

1. **Q: Walk me through how you'd chunk a meeting transcript for RAG.**
   A: Meeting transcripts have natural boundaries from ASR output — segments with speaker, text, and timestamp from Deepgram/Whisper. I'd use a segment-grouped fixed-window strategy: walk the segments, accumulate until target character count (~1200 chars ≈ 300 tokens for conversational), then finalize and start the next chunk with one segment of overlap. Each chunk carries metadata — source transcription ID, segment indices, speakers, time range — so we can cite, filter, or re-rank later. Tune target size via Phase 6 evals.

2. **Q: Recursive vs semantic chunking — when do you use which?**
   A: Recursive is the industry default — cheap, fast, respects natural boundaries (paragraphs → sentences → words) by splitting at the largest one that keeps chunks under target. Semantic chunking embeds every sentence and creates boundaries where adjacent sentences have low cosine similarity (topic shift). Semantic is higher quality but ~100× more expensive at ingest. Start with recursive for any general RAG; escalate to semantic only for high-stakes (legal, medical) or after measuring recursive's retrieval recall is too low.

3. **Q: What's chunk overlap and why does it matter?**
   A: A portion of text (typically 10-20% of chunk size) at the start of each chunk duplicates the end of the previous chunk. Without it, ideas straddling a boundary get split — a question about "why March 15?" might retrieve the chunk with the date OR the chunk with the rationale, but not one with both. Cost is duplicated text = more storage and compute per query. Trade-off worth it for conversational/prose data; less critical for code.

4. **Q: How would you tune chunk size in production?**
   A: Build a golden set of typical user questions paired with the answer chunks they should retrieve (Phase 6 eval discipline). Run retrieval at multiple chunk sizes — 200, 400, 600, 1000 tokens — measure recall@k. Plot recall vs chunk size; pick the knee. Measure latency and storage in parallel. Tune again every few months as corpus and queries shift.

5. **Q: What's "lost in the middle" and how does it affect chunking?**
   A: Empirical finding (Liu et al. 2023) that LLMs pay disproportionately more attention to content at the beginning and end of long contexts, neglecting the middle. It's more a retrieval-ordering problem than a chunking problem — but it informs design: smaller, well-targeted chunks are more robust because you can position the most-relevant ones at the prompt edges. With one 5,000-token chunk you can't recover from middle-position decay; with five 1,000-token chunks you can rank carefully.

### Gotchas

- **Defaulting to fixed-window on prose.** Splits sentences mid-thought. Use recursive.
- **No overlap on conversational data.** Boundary-straddling questions fail.
- **Embedding chunks before deciding chunk strategy.** You'll re-embed when you tune. Decide strategy + measure first.
- **Treating chunk size as a one-time decision.** It's a hyperparameter — re-tune as corpus and query patterns evolve.
- **Forgetting metadata.** A chunk vector without source ID / position / timestamp is useless for citations and impossible to re-rank.
- **Chunking before normalizing whitespace.** `"  Hello.\n\n\nWorld"` and `"Hello. World"` shouldn't produce different chunks.

### Go-deeper resources

- Pinecone, [*"Chunking Strategies for LLM Applications"*](https://www.pinecone.io/learn/chunking-strategies/) — canonical reference
- Greg Kamradt, *"5 Levels of Text Splitting"* — YouTube, 40 min, hands-on
- Liu et al., *"Lost in the Middle"* — arxiv.org/abs/2307.03172, the paper that named the failure mode
- LangChain text splitters docs — industry taxonomy

---

## Session 3.3 — pgvector setup on Neon

**Built in Audia:**
- [src/entity/TranscriptChunk.ts](../src/entity/TranscriptChunk.ts) — entity for chunks (no embedding column, since TypeORM synchronize can't handle pgvector's vector type)
- [src/db/data-source.ts](../src/db/data-source.ts) — `ensurePgvector()` runs once per process: `CREATE EXTENSION IF NOT EXISTS vector` + `ALTER TABLE transcript_chunk ADD COLUMN IF NOT EXISTS embedding vector(768)`. Both idempotent via `IF NOT EXISTS`.
- [src/lib/chunks.ts](../src/lib/chunks.ts) — `saveChunkWithEmbedding()` (TypeORM repo for the row, raw UPDATE for the vector column) + `findSimilarChunks()` (raw SELECT with `embedding <=> $1::vector` cosine distance, ownership filter, optional transcript scoping)
- Wired into [src/app/api/transcribe/route.ts](../src/app/api/transcribe/route.ts) — after the Transcription is saved, transcript is chunked, embeddings generated in parallel via `Promise.all`, chunks persisted with embeddings. Wrapped in try/catch so embedding failures don't block transcription save.
- [src/app/api/search-demo/route.ts](../src/app/api/search-demo/route.ts) — `GET /api/search-demo?q=<question>&k=5&transcript=<uuid>` returns top-k chunks ranked by cosine similarity. End-to-end RAG retrieval, working live.
- Removed throwaway [embed-demo](../src/app/api/embed-demo) route from 3.1.

### Concept summary

pgvector adds vector storage and similarity search to Postgres via a single `CREATE EXTENSION vector;`. It exposes four distance operators — `<->` (L2), `<#>` (negative inner product), `<=>` (cosine distance), `<+>` (L1) — and two ANN index types — IVFFlat (cluster-based, fast build, needs data first) and HNSW (graph-based, slower build, better queries). For Audia today: no index, sequential scan, `<=>` operator, cosine distance. TypeORM's `synchronize: true` doesn't understand the `vector(N)` type, so we work around it by adding the embedding column via raw SQL after sync and reading/writing it via raw queries. The result: every new transcript produces chunks with 768-dim embeddings stored in Postgres, queryable by cosine distance via SQL.

### 5 most-likely interview questions

1. **Q: Why did you choose pgvector over Pinecone or Weaviate?**
   A: Three reasons. Operational simplicity — no new infra, no separate auth, no cross-store consistency to manage; pgvector lives in the same Postgres I already run with the same backups and connection pool. Cost — pgvector is free on Neon; Pinecone starts around $70/month for production. Capability fit — pgvector handles up to hundreds of millions of vectors fine; Audia's projected scale is well under that. At hundreds of millions+ I'd revisit dedicated stores.

2. **Q: How does cosine similarity work in your pgvector queries?**
   A: pgvector's `<=>` operator computes cosine distance (`1 - cos(θ)`), range [0, 2], smaller = more similar. We use it twice — in SELECT to return the distance and in ORDER BY to rank: `ORDER BY embedding <=> $1::vector LIMIT 5`. The query embedding is passed as a string literal `'[0.1, 0.2, ...]'` and cast to vector with `$1::vector`. Convert to similarity in app code as `1 - distance` if you want to display.

3. **Q: Why didn't you build an index immediately?**
   A: Premature indexing has real cost — vector indexes consume memory (HNSW especially, 1.5-3× the raw vectors), slow inserts, and don't pay off until corpus is large enough that sequential scan is too slow. At Audia's <10k vectors, sequential scan completes in single-digit milliseconds; an index would have no perceptible win and would slow every insert. I'll add HNSW in Phase 12 when growth forces it, triggered by measured query latency.

4. **Q: IVFFlat vs HNSW — when do you use which?**
   A: Both are ANN indexes. IVFFlat clusters via k-means into N lists; queries probe closest lists. Fast build, low memory, but requires data present (k-means needs vectors). HNSW is a multi-layer graph; queries walk top-down greedily. Slower build, higher memory, best query speed and recall, grows incrementally. New apps default to HNSW because incremental build is friendlier; IVFFlat suits batch-loaded read-heavy workloads where build-once cost amortizes.

5. **Q: TypeORM's synchronize doesn't understand the vector type. How did you handle it?**
   A: Pragmatic workaround. Entity declared without the embedding column, so synchronize creates the table normally. After init, `ensurePgvector()` runs `CREATE EXTENSION IF NOT EXISTS vector` + `ALTER TABLE ... ADD COLUMN IF NOT EXISTS embedding vector(768)`. Both idempotent via `IF NOT EXISTS`. Reads/writes to the embedding column use raw SQL; everything else uses the repository. Real fix is migrations — that's a deferred infrastructure item. This pattern is common until teams adopt them.

### Gotchas

- **Forgetting the ownership filter on KNN.** `WHERE "userEmail" = $X` is security AND performance — without it, cross-tenant leakage plus a slower scan.
- **Building IVFFlat on an empty table.** Bad cluster centers → bad index. Backfill first, or use HNSW.
- **Using `<#>` on un-normalized vectors.** `<#>` ranks by inner product which is biased by magnitude. Use `<=>` (cosine) or normalize first.
- **Premature indexing.** Don't add HNSW or IVFFlat at <10k vectors — pure overhead.
- **Mixing embedding dimensions.** `vector(768)` columns reject 1536-dim vectors. Adding a column for a new model requires migration.
- **`ALTER TABLE ADD COLUMN vector(N)` without `IF NOT EXISTS`** crashes on the second boot. Always guard.

### Go-deeper resources

- pgvector GitHub README — github.com/pgvector/pgvector
- Neon docs on pgvector — neon.tech/docs/extensions/pgvector
- Supabase blog, *"pgvector v0.5: Faster semantic search with HNSW"*
- Malkov & Yashunin (2018), *"Efficient and robust approximate nearest neighbor search using HNSW graphs"* — the original HNSW paper
