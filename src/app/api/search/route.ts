import { In } from "typeorm";
import { getDatabase } from "@/db/data-source";
import { Transcription } from "@/entity/Transcription";
import { getCurrentUser } from "@/lib/dal";
import { findCandidateChunks } from "@/lib/chunks";
import { embed } from "@/lib/embeddings";
import { crossEncoderRerank } from "@/lib/rerank-cross";

/**
 * Semantic search across ALL of a user's meetings (Phase 8.1).
 *
 *   GET /api/search?q=<query>&k=<final>&n=<coarse>
 *
 * This is the retrieve-and-rerank pipeline as a USER-FACING surface (vs the
 * chat route, which uses retrieval as hidden plumbing for the LLM):
 *
 *   1. embed(q)                         — bi-encoder query vector
 *   2. findCandidateChunks(n=30)        — STAGE 1: cheap coarse top-N, all transcripts
 *   3. crossEncoderRerank(q, …, k)      — STAGE 2: precise top-k via Jina cross-encoder
 *   4. attach meeting titles            — join transcriptionId → Transcription.title
 *
 * No LLM generation — the ranked results are shown directly to the user, which
 * is exactly why ranking quality (the cross-encoder) suddenly matters here.
 *
 * Note: findCandidateChunks with no transcriptionId already searches across all
 * the user's meetings (ownership-filtered in SQL). Cross-transcript search was
 * a primitive we already had; 8.1 adds the precision stage + the UI surface.
 */

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
    /** Stage-1 bi-encoder cosine similarity — kept for transparency/debugging. */
    biEncoderSimilarity: number;
};

export async function GET(req: Request) {
    const user = await getCurrentUser();
    if (!user) return Response.json({ error: "Unauthorized" }, { status: 401 });

    const url = new URL(req.url);
    const q = url.searchParams.get("q")?.trim();
    if (!q) return Response.json({ error: "Missing ?q= query parameter" }, { status: 400 });

    // k = results shown; n = coarse candidates fed to the reranker (n ≥ k).
    const k = Math.min(Math.max(Number(url.searchParams.get("k") ?? 8), 1), 20);
    const n = Math.min(Math.max(Number(url.searchParams.get("n") ?? 30), k), 60);

    const start = Date.now();

    // ── Stage 1: bi-encoder recall (fast, wide, all transcripts) ──────────
    const queryEmbedding = await embed(q);
    const candidates = await findCandidateChunks(queryEmbedding, user.email, { n });
    const stage1Ms = Date.now() - start;

    if (candidates.length === 0) {
        return Response.json({
            query: q,
            count: 0,
            hits: [],
            timing: { stage1Ms, stage2Ms: 0, totalMs: Date.now() - start },
        });
    }

    // ── Stage 2: cross-encoder precision (slow, narrow) ───────────────────
    const rerankStart = Date.now();
    const reranked = await crossEncoderRerank(q, candidates, k);
    const stage2Ms = Date.now() - rerankStart;

    // ── Attach meeting titles (one query for all distinct meetings) ───────
    // Re-filter by userEmail even though chunks were already ownership-scoped —
    // defense in depth: a title lookup must never leak another user's meeting.
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
        biEncoderSimilarity: r.similarity,
    }));

    console.log(
        `[search] q="${q.slice(0, 50)}" candidates=${candidates.length} returned=${hits.length} ` +
        `stage1=${stage1Ms}ms stage2=${stage2Ms}ms`,
    );

    return Response.json({
        query: q,
        count: hits.length,
        hits,
        timing: { stage1Ms, stage2Ms, totalMs: Date.now() - start },
    });
}
