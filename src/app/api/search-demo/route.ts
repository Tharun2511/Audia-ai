import { getCurrentUser } from "@/lib/dal";
import { embed } from "@/lib/embeddings";
import { findCandidateChunks } from "@/lib/chunks";
import { maximalMarginalRelevance } from "@/lib/rerank";

/**
 * Demo: GET /api/search-demo?q=<question>&k=5&n=20&lambda=0.7&transcript=<uuid>
 *
 * Embeds the query, fetches top-N candidates (with embeddings) from pgvector,
 * then returns BOTH naive top-k AND MMR-reranked top-k side-by-side so you can
 * compare them on the same input.
 *
 * Params:
 *   q          — the query string (required)
 *   k          — final number of results (default 5)
 *   n          — coarse top-N pulled from pgvector before re-ranking (default 20)
 *   lambda     — MMR relevance/diversity knob in [0, 1] (default 0.7)
 *   transcript — scope to a single transcription UUID (optional)
 */
export async function GET(req: Request) {
    const user = await getCurrentUser();
    if (!user) return Response.json({ error: "Unauthorized" }, { status: 401 });

    const url = new URL(req.url);
    const q = url.searchParams.get("q");
    const k = Number(url.searchParams.get("k") ?? 5);
    const n = Number(url.searchParams.get("n") ?? 20);
    const lambda = Number(url.searchParams.get("lambda") ?? 0.7);
    const transcriptionId = url.searchParams.get("transcript") ?? undefined;

    if (!q) {
        return Response.json({ error: "Missing ?q= query parameter" }, { status: 400 });
    }

    const start = Date.now();
    const queryEmbedding = await embed(q);
    const embeddingMs = Date.now() - start;

    const searchStart = Date.now();
    const candidates = await findCandidateChunks(queryEmbedding, user.email, {
        transcriptionId,
        n,
    });
    const searchMs = Date.now() - searchStart;

    const mmrStart = Date.now();
    const mmrPicks = maximalMarginalRelevance(queryEmbedding, candidates, k, lambda);
    const mmrMs = Date.now() - mmrStart;

    const naiveTopK = candidates.slice(0, k);

    const fmt = (r: typeof candidates[number]) => ({
        similarity: r.similarity.toFixed(4),
        chunkIndex: r.chunkIndex,
        timeRange: `${r.startTime.toFixed(1)}s - ${r.endTime.toFixed(1)}s`,
        speakers: r.speakers,
        text: r.text,
    });

    return Response.json({
        query: q,
        scope: transcriptionId ? `transcript=${transcriptionId}` : "all transcripts",
        params: { k, n, lambda },
        timing: {
            embeddingMs,
            vectorSearchMs: searchMs,
            mmrMs,
            totalMs: Date.now() - start,
        },
        candidateCount: candidates.length,
        naiveTopK: naiveTopK.map(fmt),
        mmrReranked: mmrPicks.map(fmt),
    });
}
