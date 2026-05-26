/**
 * One-off backfill route — chunks + embeddings for transcriptions that have
 * none (or only NULL-embedding orphans).
 *
 * GET  /api/backfill-chunks            — dry run. Lists which transcriptions
 *                                        would be processed and why. Safe,
 *                                        idempotent, browser-friendly.
 * POST /api/backfill-chunks            — actually run it. Sequential per
 *                                        transcription (polite to the embed
 *                                        API); ~2-5s per ~5-chunk transcript.
 *
 * Delete this route after the backfill is done — it's not part of normal
 * product surface.
 *
 * Trigger from DevTools console (auth cookie attaches automatically):
 *   fetch('/api/backfill-chunks', { method: 'POST' }).then(r => r.json()).then(console.log)
 */
import { getDatabase } from "@/db/data-source";
import { Transcription } from "@/entity/Transcription";
import { chunkTranscript } from "@/lib/chunking";
import { saveChunkWithEmbedding } from "@/lib/chunks";
import { getCurrentUser } from "@/lib/dal";
import { embed } from "@/lib/embeddings";

type Status = {
    id: string;
    title: string | null;
    segmentCount: number;
    existingChunks: number;
    existingChunksWithEmbedding: number;
    needsBackfill: boolean;
    reason: "no-chunks" | "all-null-embeddings" | "partial-null-embeddings" | "ok";
};

async function buildStatus(userEmail: string): Promise<{ transcriptions: Transcription[]; status: Status[] }> {
    const db = await getDatabase();
    const repo = db.getRepository(Transcription);

    const transcriptions = await repo.find({
        where: { userEmail },
        order: { createdAt: "ASC" },
    });

    const status: Status[] = [];
    for (const t of transcriptions) {
        const rows = (await db.query(
            `SELECT COUNT(*)::int AS total, COUNT(embedding)::int AS with_embedding
             FROM transcript_chunk WHERE "transcriptionId" = $1`,
            [t.id],
        )) as Array<{ total: number; with_embedding: number }>;

        const total = rows[0]?.total ?? 0;
        const withEmbedding = rows[0]?.with_embedding ?? 0;

        let reason: Status["reason"];
        let needsBackfill: boolean;
        if (total === 0) {
            reason = "no-chunks";
            needsBackfill = true;
        } else if (withEmbedding === 0) {
            reason = "all-null-embeddings";
            needsBackfill = true;
        } else if (withEmbedding < total) {
            reason = "partial-null-embeddings";
            needsBackfill = true;
        } else {
            reason = "ok";
            needsBackfill = false;
        }

        status.push({
            id: t.id,
            title: t.title,
            segmentCount: t.segments?.length ?? 0,
            existingChunks: total,
            existingChunksWithEmbedding: withEmbedding,
            needsBackfill,
            reason,
        });
    }

    return { transcriptions, status };
}

export async function GET() {
    const user = await getCurrentUser();
    if (!user) return Response.json({ error: "Unauthorized" }, { status: 401 });

    const { status } = await buildStatus(user.email);
    const toProcess = status.filter((s) => s.needsBackfill);

    return Response.json({
        dryRun: true,
        message: "POST to this endpoint to actually run the backfill.",
        totalTranscriptions: status.length,
        needsBackfill: toProcess.length,
        wouldProcess: toProcess,
        skipped: status.filter((s) => !s.needsBackfill),
    });
}

export async function POST(req: Request) {
    const user = await getCurrentUser();
    if (!user) return Response.json({ error: "Unauthorized" }, { status: 401 });

    const { transcriptions, status } = await buildStatus(user.email);
    const db = await getDatabase();

    const results: Array<{
        id: string;
        title: string | null;
        ok: boolean;
        chunks?: number;
        error?: string;
    }> = [];

    const start = Date.now();
    console.log(`[backfill] starting for ${user.email}: ${status.filter((s) => s.needsBackfill).length} transcriptions to process`);

    // Sequential per transcription (NOT Promise.all across transcriptions) —
    // each transcription's chunks already parallelize internally. Sequential at
    // the outer level keeps embed-API pressure bounded and makes the per-row
    // log readable.
    for (const t of transcriptions) {
        const stat = status.find((s) => s.id === t.id)!;
        if (!stat.needsBackfill) continue;

        const tStart = Date.now();
        try {
            // Wipe any existing chunks (including NULL-embedding orphans) before
            // recreating. Idempotent — safe to re-run if a previous backfill
            // partially completed.
            if (stat.existingChunks > 0) {
                await db.query(
                    `DELETE FROM transcript_chunk WHERE "transcriptionId" = $1`,
                    [t.id],
                );
            }

            const chunks = chunkTranscript(t.segments);
            if (chunks.length === 0) {
                console.log(`[backfill] ${t.id} ${t.title ?? "(untitled)"}: 0 chunks (empty transcript), skipping`);
                results.push({ id: t.id, title: t.title, ok: true, chunks: 0 });
                continue;
            }

            const embeddings = await Promise.all(chunks.map((c) => embed(c.text)));
            await Promise.all(
                chunks.map((chunk, i) =>
                    saveChunkWithEmbedding(chunk, t.id, user.email, i, embeddings[i]),
                ),
            );

            const elapsed = Date.now() - tStart;
            console.log(`[backfill] ${t.id} ${t.title ?? "(untitled)"}: ${chunks.length} chunks in ${elapsed}ms`);
            results.push({ id: t.id, title: t.title, ok: true, chunks: chunks.length });
        } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            console.warn(`[backfill] ${t.id} failed:`, msg);
            results.push({ id: t.id, title: t.title, ok: false, error: msg });
        }
    }

    const totalElapsed = Date.now() - start;
    const succeeded = results.filter((r) => r.ok).length;
    const failed = results.filter((r) => !r.ok).length;
    console.log(`[backfill] done: ${succeeded} succeeded, ${failed} failed in ${totalElapsed}ms`);

    return Response.json({
        dryRun: false,
        totalTranscriptions: status.length,
        processed: results.length,
        succeeded,
        failed,
        elapsedMs: totalElapsed,
        results,
    });
}
