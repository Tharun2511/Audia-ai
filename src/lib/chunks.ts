import "server-only";
import { getDatabase } from "@/db/data-source";
import { TranscriptChunk } from "@/entity/TranscriptChunk";
import type { TranscriptChunk as TranscriptChunkData } from "@/lib/chunking";

/**
 * pgvector text representation: '[0.1,0.2,0.3,...]'. Comma-separated, square
 * brackets, no spaces required. Cast to ::vector in SQL.
 */
function toVectorLiteral(vec: number[]): string {
    return `[${vec.join(",")}]`;
}

/**
 * Persist one chunk + its embedding. Uses TypeORM for the row, raw SQL for
 * the vector column (TypeORM doesn't know the vector type).
 */
export async function saveChunkWithEmbedding(
    chunk: TranscriptChunkData,
    transcriptionId: string,
    userEmail: string,
    chunkIndex: number,
    embedding: number[],
): Promise<TranscriptChunk> {
    const db = await getDatabase();
    const repo = db.getRepository(TranscriptChunk);

    const row = repo.create({
        transcriptionId,
        userEmail,
        chunkIndex,
        text: chunk.text,
        segmentIndices: chunk.segmentIndices,
        speakers: chunk.speakers,
        startTime: chunk.startTime,
        endTime: chunk.endTime,
        charCount: chunk.charCount,
    });
    const saved = await repo.save(row);

    await db.query(`UPDATE transcript_chunk SET embedding = $1::vector WHERE id = $2`, [
        toVectorLiteral(embedding),
        saved.id,
    ]);

    return saved;
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
            segmentIndices: r.segmentIndices,
            speakers: r.speakers,
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
            segmentIndices: r.segmentIndices,
            speakers: r.speakers,
            startTime: r.startTime,
            endTime: r.endTime,
            distance,
            similarity: 1 - distance,
            embedding: parseVector(r.embedding_text),
        };
    });
}
