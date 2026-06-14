import { In } from "typeorm";
import { getDatabase } from "@/db/data-source";
import { Transcription } from "@/entity/Transcription";
import { getCurrentUser } from "@/lib/dal";
import { findLexicalChunks, findSimilarChunks } from "@/lib/chunks";
import { embed } from "@/lib/embeddings";
import { reciprocalRankFusion } from "@/lib/rerank";
import { crossEncoderRerank } from "@/lib/rerank-cross";

/**
 * HYBRID semantic + keyword search across ALL of a user's meetings (Phase 8.2,
 * extending the dense-only search from 8.1).
 *
 *   GET /api/search?q=<query>&k=<final>&n=<per-arm coarse>
 *
 * Pipeline:
 *   1. Run TWO recall arms in parallel over all the user's chunks:
 *        DENSE   — findSimilarChunks  (pgvector cosine; matches MEANING)
 *        LEXICAL — findLexicalChunks  (Postgres FTS; matches EXACT terms)
 *      They have complementary blind spots — dense blurs rare tokens, lexical
 *      misses paraphrase — so together they recall what neither gets alone.
 *   2. RECIPROCAL RANK FUSION merges the two ranked lists on RANK (not score —
 *      cosine and ts_rank are incomparable scales). k=60.
 *   3. CROSS-ENCODER rerank (8.1) re-scores the fused top-N for precision.
 *   4. Attach meeting titles.
 *
 * Note: 8.1's reranker only ever saw what dense recall handed it. Hybrid fixes
 * that RECALL gap upstream — you can't rerank a chunk that was never retrieved.
 */

// Per-arm coarse candidate count, and final results shown.
const DEFAULT_N = 30;
const DEFAULT_K = 8;
// Cap the fused set before the (paid, latency-bearing) cross-encoder pass.
const RERANK_POOL = 30;

export type SearchHit = {
    chunkId: string;
    transcriptionId: string;
    meetingTitle: string | null;
    startTime: number;
    endTime: number;
    speakers: string[];
    text: string;
    /** Cross-encoder relevance in [0, 1]; NaN if the reranker was unavailable. */
    relevanceScore: number;
    /** Which recall arms surfaced this chunk: "semantic", "keyword", or both. */
    sources: string[];
};

/** Common shape both arms normalize to, so RRF + rerank don't care which arm produced an item. */
type Candidate = {
    id: string;
    transcriptionId: string;
    startTime: number;
    endTime: number;
    speakers: string[];
    text: string;
};

export async function GET(req: Request) {
    const user = await getCurrentUser();
    if (!user) return Response.json({ error: "Unauthorized" }, { status: 401 });

    const url = new URL(req.url);
    const q = url.searchParams.get("q")?.trim();
    if (!q) return Response.json({ error: "Missing ?q= query parameter" }, { status: 400 });

    const k = Math.min(Math.max(Number(url.searchParams.get("k") ?? DEFAULT_K), 1), 20);
    const n = Math.min(Math.max(Number(url.searchParams.get("n") ?? DEFAULT_N), k), 60);

    const start = Date.now();

    // ── Recall: dense + lexical, in parallel ──────────────────────────────
    // The dense arm needs the query embedding; the lexical arm works on the raw
    // text. Fire both concurrently — neither depends on the other.
    const queryEmbedding = await embed(q);
    const [dense, lexical] = await Promise.all([
        findSimilarChunks(queryEmbedding, user.email, { k: n }),
        findLexicalChunks(q, user.email, { n }),
    ]);
    const recallMs = Date.now() - start;

    if (dense.length === 0 && lexical.length === 0) {
        return Response.json({
            query: q,
            count: 0,
            hits: [],
            timing: { recallMs, rerankMs: 0, totalMs: Date.now() - start },
            arms: { dense: 0, lexical: 0 },
        });
    }

    // Normalize both arms to the same shape so fusion is arm-agnostic.
    const toCandidate = (c: {
        id: string; transcriptionId: string; startTime: number; endTime: number; speakers: string[]; text: string;
    }): Candidate => ({
        id: c.id,
        transcriptionId: c.transcriptionId,
        startTime: c.startTime,
        endTime: c.endTime,
        speakers: c.speakers,
        text: c.text,
    });

    // ── Fuse on rank (RRF) ────────────────────────────────────────────────
    const fused = reciprocalRankFusion<Candidate>(
        [
            { name: "semantic", items: dense.map(toCandidate) },
            { name: "keyword", items: lexical.map(toCandidate) },
        ],
        (c) => c.id,
    ).slice(0, RERANK_POOL);

    // ── Precision: cross-encoder rerank the fused pool ────────────────────
    const rerankStart = Date.now();
    const reranked = await crossEncoderRerank(q, fused, k);
    const rerankMs = Date.now() - rerankStart;

    // ── Attach meeting titles (ownership-filtered; defense in depth) ──────
    const ids = Array.from(new Set(reranked.map((r) => r.transcriptionId)));
    const db = await getDatabase();
    const repo = db.getRepository(Transcription);
    const meetings = ids.length
        ? await repo.find({ where: { id: In(ids), userEmail: user.email }, select: ["id", "title"] })
        : [];
    const titleById = new Map(meetings.map((m) => [m.id, m.title ?? null]));

    const hits: SearchHit[] = reranked.map((r) => ({
        chunkId: r.id,
        transcriptionId: r.transcriptionId,
        meetingTitle: titleById.get(r.transcriptionId) ?? null,
        startTime: r.startTime,
        endTime: r.endTime,
        speakers: r.speakers,
        text: r.text,
        relevanceScore: r.relevanceScore,
        sources: r.sources,
    }));

    console.log(
        `[search] q="${q.slice(0, 50)}" dense=${dense.length} lexical=${lexical.length} ` +
        `fused=${fused.length} returned=${hits.length} recall=${recallMs}ms rerank=${rerankMs}ms`,
    );

    return Response.json({
        query: q,
        count: hits.length,
        hits,
        timing: { recallMs, rerankMs, totalMs: Date.now() - start },
        arms: { dense: dense.length, lexical: lexical.length },
    });
}
