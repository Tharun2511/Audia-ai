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
