import { getCurrentUser } from "@/lib/dal";
import { embed } from "@/lib/embeddings";
import { findSimilarChunks } from "@/lib/chunks";

/**
 * Demo: GET /api/search-demo?q=<question>&k=5&transcript=<uuid>
 *
 * Embeds the question and returns the top-k most-similar chunks across the
 * authenticated user's transcripts (or scoped to a single transcript if
 * ?transcript= is provided). Returns the cosine similarity per chunk.
 */
export async function GET(req: Request) {
    const user = await getCurrentUser();
    if (!user) return Response.json({ error: "Unauthorized" }, { status: 401 });

    const url = new URL(req.url);
    const q = url.searchParams.get("q");
    const k = Number(url.searchParams.get("k") ?? 5);
    const transcriptionId = url.searchParams.get("transcript") ?? undefined;

    if (!q) {
        return Response.json({ error: "Missing ?q= query parameter" }, { status: 400 });
    }

    const start = Date.now();
    const queryEmbedding = await embed(q);
    const embeddingMs = Date.now() - start;

    const searchStart = Date.now();
    const results = await findSimilarChunks(queryEmbedding, user.email, {
        transcriptionId,
        k,
    });
    const searchMs = Date.now() - searchStart;

    return Response.json({
        query: q,
        scope: transcriptionId ? `transcript=${transcriptionId}` : "all transcripts",
        timing: { embeddingMs, searchMs, totalMs: Date.now() - start },
        results: results.map((r) => ({
            similarity: r.similarity.toFixed(4),
            distance: r.distance.toFixed(4),
            transcriptionId: r.transcriptionId,
            chunkIndex: r.chunkIndex,
            timeRange: `${r.startTime.toFixed(1)}s - ${r.endTime.toFixed(1)}s`,
            speakers: r.speakers,
            text: r.text,
        })),
    });
}
