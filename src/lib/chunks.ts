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
         WHERE "userEmail" = $2 ${transcriptionClause}
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
