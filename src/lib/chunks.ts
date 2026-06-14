import "server-only";
import { randomUUID } from "crypto";
import { getDatabase } from "@/db/data-source";
import type { TranscriptChunk as TranscriptChunkData } from "@/lib/chunking";

/**
 * pgvector text representation: '[0.1,0.2,0.3,...]'. Comma-separated, square
 * brackets, no spaces required. Cast to ::vector in SQL.
 */
function toVectorLiteral(vec: number[]): string {
    return `[${vec.join(",")}]`;
}

/**
 * Defensive guard against bad embedding values. NaN / Infinity / wrong-length
 * vectors silently corrupt pgvector inserts in some cases — fail loudly at the
 * boundary so the caller's try/catch logs a clear message instead of leaving
 * NULL-embedding orphan rows.
 */
function validateEmbedding(vec: number[], expectedDim: number): void {
    if (!Array.isArray(vec)) throw new Error("embedding is not an array");
    if (vec.length !== expectedDim) {
        throw new Error(`embedding dim mismatch: got ${vec.length}, expected ${expectedDim}`);
    }
    for (let i = 0; i < vec.length; i++) {
        if (!Number.isFinite(vec[i])) {
            throw new Error(`embedding[${i}] is not finite: ${vec[i]}`);
        }
    }
}

/**
 * Persist one chunk + its embedding in a SINGLE atomic INSERT. This replaces
 * the earlier two-step pattern (`repo.save` then raw `UPDATE`) which produced
 * NULL-embedding orphans whenever the UPDATE failed after the row was already
 * written. Now: one statement, either both columns land or neither does.
 *
 * Why we hand-write the INSERT instead of using the TypeORM repo:
 *   - TypeORM doesn't know the `vector` type (Phase 3.3's whole workaround)
 *   - Including the `embedding::vector` cast inline keeps the write atomic
 *   - We give up `@CreateDateColumn` auto-population — set NOW() explicitly
 *   - JSON-encoding `segmentIndices` and `speakers` matches what TypeORM does
 *     for `simple-json` columns under the hood (it serializes to a text column)
 */
export async function saveChunkWithEmbedding(
    chunk: TranscriptChunkData,
    transcriptionId: string,
    userEmail: string,
    chunkIndex: number,
    embedding: number[],
): Promise<{ id: string }> {
    validateEmbedding(embedding, 768);

    const db = await getDatabase();
    const embeddingLiteral = toVectorLiteral(embedding);

    const id = randomUUID();
    await db.query(
        `INSERT INTO transcript_chunk
            (id, "transcriptionId", "userEmail", "chunkIndex", text,
             "segmentIndices", speakers, "startTime", "endTime", "charCount",
             embedding, "createdAt")
         VALUES
            ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::vector, NOW())`,
        [
            id,
            transcriptionId,
            userEmail,
            chunkIndex,
            chunk.text,
            JSON.stringify(chunk.segmentIndices),
            JSON.stringify(chunk.speakers),
            chunk.startTime,
            chunk.endTime,
            chunk.charCount,
            embeddingLiteral,
        ],
    );

    return { id };
}

export type SimilarChunkResult = {
    id: string;
    transcriptionId: string;
    chunkIndex: number;
    text: string;
    segmentIndices: number[];
    speakers: string[];
    startTime: number;
    endTime: number;
    distance: number;       // cosine distance, smaller = more similar
    similarity: number;     // 1 - distance, larger = more similar
};

/**
 * KNN search for the top-k most-similar chunks to a query embedding.
 * Filters by userEmail (ownership) and optionally a single transcriptionId.
 *
 * Uses pgvector's <=> (cosine distance) operator. At Audia's current scale
 * (<10k vectors) sequential scan is fine; add an HNSW index when growth
 * forces it (Phase 12).
 */
export async function findSimilarChunks(
    queryEmbedding: number[],
    userEmail: string,
    opts: { transcriptionId?: string; k?: number } = {},
): Promise<SimilarChunkResult[]> {
    const db = await getDatabase();
    const k = opts.k ?? 5;
    const queryVec = toVectorLiteral(queryEmbedding);

    const params: unknown[] = [queryVec, userEmail];
    let transcriptionClause = "";
    if (opts.transcriptionId) {
        params.push(opts.transcriptionId);
        transcriptionClause = `AND "transcriptionId" = $${params.length}`;
    }
    params.push(k);
    const limitParam = `$${params.length}`;

    const rows = (await db.query(
        `SELECT id, "transcriptionId", "chunkIndex", text,
                "segmentIndices", speakers, "startTime", "endTime",
                (embedding <=> $1::vector) AS distance
         FROM transcript_chunk
         WHERE "userEmail" = $2 AND embedding IS NOT NULL ${transcriptionClause}
         ORDER BY embedding <=> $1::vector
         LIMIT ${limitParam}`,
        params,
    )) as Array<{
        id: string;
        transcriptionId: string;
        chunkIndex: number;
        text: string;
        segmentIndices: number[];
        speakers: string[];
        startTime: number;
        endTime: number;
        distance: string | number;
    }>;

    return rows.map((r) => {
        const distance = Number(r.distance);
        return {
            id: r.id,
            transcriptionId: r.transcriptionId,
            chunkIndex: r.chunkIndex,
            text: r.text,
            segmentIndices: parseJsonColumn<number[]>(r.segmentIndices, []),
            speakers: parseJsonColumn<string[]>(r.speakers, []),
            startTime: r.startTime,
            endTime: r.endTime,
            distance,
            similarity: 1 - distance,
        };
    });
}

/**
 * pgvector's text output format is "[0.1,0.2,0.3]" — parse back to number[].
 * Used when SELECTing the embedding column for in-memory re-ranking.
 */
function parseVector(s: string): number[] {
    return s.slice(1, -1).split(",").map(Number);
}

/**
 * TypeORM's `simple-json` column type stores JSON-encoded strings in a `text`
 * column. When read via the repository, TypeORM auto-deserializes back to the
 * original shape. When read via raw SQL (db.query), the driver returns the raw
 * string — `'["User1"]'`, not `["User1"]`. We parse defensively so callers can
 * always treat the field as the declared TS type.
 *
 * Some drivers/columns may auto-parse (e.g. jsonb), so we handle both cases.
 */
function parseJsonColumn<T>(value: unknown, fallback: T): T {
    if (typeof value === "string") {
        try {
            return JSON.parse(value) as T;
        } catch {
            return fallback;
        }
    }
    return (value as T) ?? fallback;
}

export type CandidateChunk = SimilarChunkResult & {
    embedding: number[];
};

/**
 * Wider top-N retrieval that ALSO returns the embedding column. Used as input
 * to in-memory re-rankers (MMR, cross-encoder, LLM-as-reranker) which need
 * candidate vectors to compute candidate-to-candidate similarities.
 *
 * Typical usage: N = 3-5× the final k. Example: want top-5 after MMR, fetch
 * top-20 here; MMR picks 5 from those 20.
 */
export async function findCandidateChunks(
    queryEmbedding: number[],
    userEmail: string,
    opts: { transcriptionId?: string; n?: number } = {},
): Promise<CandidateChunk[]> {
    const db = await getDatabase();
    const n = opts.n ?? 20;
    const queryVec = toVectorLiteral(queryEmbedding);

    const params: unknown[] = [queryVec, userEmail];
    let transcriptionClause = "";
    if (opts.transcriptionId) {
        params.push(opts.transcriptionId);
        transcriptionClause = `AND "transcriptionId" = $${params.length}`;
    }
    params.push(n);
    const limitParam = `$${params.length}`;

    const rows = (await db.query(
        `SELECT id, "transcriptionId", "chunkIndex", text,
                "segmentIndices", speakers, "startTime", "endTime",
                (embedding <=> $1::vector) AS distance,
                embedding::text AS embedding_text
         FROM transcript_chunk
         WHERE "userEmail" = $2 AND embedding IS NOT NULL ${transcriptionClause}
         ORDER BY embedding <=> $1::vector
         LIMIT ${limitParam}`,
        params,
    )) as Array<{
        id: string;
        transcriptionId: string;
        chunkIndex: number;
        text: string;
        segmentIndices: number[];
        speakers: string[];
        startTime: number;
        endTime: number;
        distance: string | number;
        embedding_text: string;
    }>;

    return rows.map((r) => {
        const distance = Number(r.distance);
        return {
            id: r.id,
            transcriptionId: r.transcriptionId,
            chunkIndex: r.chunkIndex,
            text: r.text,
            segmentIndices: parseJsonColumn<number[]>(r.segmentIndices, []),
            speakers: parseJsonColumn<string[]>(r.speakers, []),
            startTime: r.startTime,
            endTime: r.endTime,
            distance,
            similarity: 1 - distance,
            embedding: parseVector(r.embedding_text),
        };
    });
}

// ── Lexical (keyword) retrieval — the sparse arm of hybrid search (Phase 8.2) ──

export type LexicalChunkResult = {
    id: string;
    transcriptionId: string;
    chunkIndex: number;
    text: string;
    segmentIndices: number[];
    speakers: string[];
    startTime: number;
    endTime: number;
    /** Postgres ts_rank_cd score — used only for ordering this list; RRF fuses on rank, not this value. */
    lexScore: number;
};

/**
 * Keyword search over a user's chunks via Postgres full-text search — the
 * complement to dense/semantic retrieval. Where dense search blurs rare exact
 * tokens (IDs, codes, product names) into a topic vector, this nails them.
 *
 * Implementation note — NO generated tsvector column / GIN index (yet):
 *   We compute `to_tsvector('english', text)` INLINE per query instead of
 *   maintaining a `tsvector GENERATED ALWAYS AS (...) STORED` column + GIN
 *   index. Two reasons:
 *     1. Same "no premature indexing" stance as pgvector's deferred HNSW (3.3)
 *        — at <10k chunks/user (pre-filtered by userEmail) a sequential ts
 *        match is fast enough.
 *     2. Avoids the synchronize-orphan-column bug class (TypeORM `synchronize`
 *        drops columns it doesn't know about — exactly what bit the embedding
 *        column once). A generated column derived from `text` would be safe to
 *        drop+recreate, but inline sidesteps the churn entirely.
 *   Production upgrade when scale demands: add the STORED tsvector column +
 *   GIN index and swap `to_tsvector('english', text)` for that column.
 *
 * websearch_to_tsquery parses arbitrary user input SAFELY (quotes, OR, -neg)
 * and never throws on weird input — unlike to_tsquery/plainto_tsquery. If the
 * query is all stopwords it yields an empty tsquery → zero matches, which is
 * fine (the dense arm still contributes).
 */
export async function findLexicalChunks(
    query: string,
    userEmail: string,
    opts: { n?: number } = {},
): Promise<LexicalChunkResult[]> {
    const db = await getDatabase();
    const n = opts.n ?? 30;

    const rows = (await db.query(
        `WITH q AS (SELECT websearch_to_tsquery('english', $1) AS query)
         SELECT c.id, c."transcriptionId", c."chunkIndex", c.text,
                c."segmentIndices", c.speakers, c."startTime", c."endTime",
                ts_rank_cd(to_tsvector('english', c.text), q.query) AS lex_score
         FROM transcript_chunk c, q
         WHERE c."userEmail" = $2
           AND to_tsvector('english', c.text) @@ q.query
         ORDER BY lex_score DESC
         LIMIT $3`,
        [query, userEmail, n],
    )) as Array<{
        id: string;
        transcriptionId: string;
        chunkIndex: number;
        text: string;
        segmentIndices: number[];
        speakers: string[];
        startTime: number;
        endTime: number;
        lex_score: string | number;
    }>;

    return rows.map((r) => ({
        id: r.id,
        transcriptionId: r.transcriptionId,
        chunkIndex: r.chunkIndex,
        text: r.text,
        segmentIndices: parseJsonColumn<number[]>(r.segmentIndices, []),
        speakers: parseJsonColumn<string[]>(r.speakers, []),
        startTime: r.startTime,
        endTime: r.endTime,
        lexScore: Number(r.lex_score),
    }));
}
