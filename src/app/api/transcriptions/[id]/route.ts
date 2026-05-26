import { getDatabase } from "@/db/data-source";
import { Transcription } from "@/entity/Transcription";
import { chunkTranscript } from "@/lib/chunking";
import { saveChunkWithEmbedding } from "@/lib/chunks";
import { getCurrentUser } from "@/lib/dal";
import { embed } from "@/lib/embeddings";

type Ctx = { params: Promise<{ id: string }> };

const MAX_SEGMENT_TEXT_LENGTH = 10_000;

export async function PATCH(req: Request, { params }: Ctx) {
    const user = await getCurrentUser();
    if (!user) return Response.json({ error: "Unauthorized" }, { status: 401 });

    const { id } = await params;
    const body = (await req.json()) as {
        speakerMap?: Record<string, string>;
        title?: string;
        /**
         * Sparse, index-keyed text-only edits. Shape: { "5": "corrected text" }.
         * The server applies only the .text field; start/end/speaker are
         * preserved so the audio-seek and active-segment math stay valid.
         */
        segmentTextEdits?: Record<string, string>;
    };

    const db = await getDatabase();
    const repo = db.getRepository(Transcription);
    const record = await repo.findOne({ where: { id, userEmail: user.email } });
    if (!record) return Response.json({ error: "Not found" }, { status: 404 });

    if (body.speakerMap) {
        record.segments = record.segments.map((seg) => ({
            ...seg,
            speaker: body.speakerMap![seg.speaker] ?? seg.speaker,
        }));
    }

    if (body.segmentTextEdits) {
        // Validate first, mutate second — bail before saving anything if any
        // edit is malformed so we never leave the row half-updated.
        for (const [key, value] of Object.entries(body.segmentTextEdits)) {
            const idx = Number(key);
            if (!Number.isInteger(idx) || idx < 0 || idx >= record.segments.length) {
                return Response.json({ error: `Invalid segment index: ${key}` }, { status: 400 });
            }
            if (typeof value !== "string" || value.length > MAX_SEGMENT_TEXT_LENGTH) {
                return Response.json({ error: "Invalid segment text" }, { status: 400 });
            }
        }
        record.segments = record.segments.map((seg, i) => {
            const next = body.segmentTextEdits![String(i)];
            return next === undefined ? seg : { ...seg, text: next };
        });
    }

    if (body.title !== undefined) {
        record.title = body.title || null;
    }

    // Did this PATCH touch any segment content? Only segments affect chunks —
    // title changes don't. We snapshot this BEFORE save so we can decide
    // whether to regenerate chunks after.
    const segmentsChanged = !!(body.speakerMap || body.segmentTextEdits);

    await repo.save(record);

    // ── Chunk invalidation on segment edit (Strategy A: regenerate) ───────
    // If segments changed, the existing chunks + embeddings are stale. Wipe
    // them and re-index from the new segments. Synchronous so the search
    // surface is consistent the moment PATCH returns. Trade-off: edit
    // latency grows by (N chunks × ~200ms embed call) / Promise.all
    // parallelism — typically <500ms on a 10-chunk meeting. Acceptable for
    // an edit interaction (not a hot path).
    //
    // Wrapped in try/catch so a transient embed failure doesn't roll back
    // the segment update — the segments are saved; we'll just have stale
    // chunks until the next edit. Logged loudly so we know it happened.
    if (segmentsChanged) {
        try {
            await db.query(
                `DELETE FROM transcript_chunk WHERE "transcriptionId" = $1`,
                [id],
            );
            const newChunks = chunkTranscript(record.segments);
            const embeddings = await Promise.all(newChunks.map((c) => embed(c.text)));
            await Promise.all(
                newChunks.map((chunk, i) =>
                    saveChunkWithEmbedding(chunk, id, user.email, i, embeddings[i]),
                ),
            );
            console.log(`[PATCH ${id}] regenerated ${newChunks.length} chunks after segment edit`);
        } catch (err) {
            console.warn(`[PATCH ${id}] chunk regeneration failed`, err);
        }
    }

    return Response.json({ segments: record.segments, title: record.title });
}

export async function DELETE(_req: Request, { params }: Ctx) {
    const user = await getCurrentUser();
    if (!user) return Response.json({ error: "Unauthorized" }, { status: 401 });

    const { id } = await params;
    const db = await getDatabase();

    // Cascade delete to chunks BEFORE deleting the parent row. If the
    // chunks delete fails for any reason, we'd rather have leftover chunks
    // pointing at a missing transcription (recoverable, retrievable as
    // orphans) than a deleted transcription with chunks still searchable
    // (would surface stale references via the chat).
    await db.query(`DELETE FROM transcript_chunk WHERE "transcriptionId" = $1 AND "userEmail" = $2`, [
        id,
        user.email,
    ]);

    const repo = db.getRepository(Transcription);
    await repo.delete({ id, userEmail: user.email });
    return Response.json({ success: true });
}
