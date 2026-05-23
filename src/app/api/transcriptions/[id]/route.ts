import { getDatabase } from "@/db/data-source";
import { Transcription } from "@/entity/Transcription";
import { getCurrentUser } from "@/lib/dal";

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
    await repo.save(record);
    return Response.json({ segments: record.segments, title: record.title });
}

export async function DELETE(_req: Request, { params }: Ctx) {
    const user = await getCurrentUser();
    if (!user) return Response.json({ error: "Unauthorized" }, { status: 401 });

    const { id } = await params;
    const db = await getDatabase();
    const repo = db.getRepository(Transcription);
    await repo.delete({ id, userEmail: user.email });
    return Response.json({ success: true });
}
