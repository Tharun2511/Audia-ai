import { getDatabase } from "@/db/data-source";
import { Transcription } from "@/entity/Transcription";

type Ctx = { params: Promise<{ id: string }> };

export async function PATCH(req: Request, { params }: Ctx) {
    const { id } = await params;
    const body = (await req.json()) as { speakerMap?: Record<string, string>; title?: string };

    const db = await getDatabase();
    const repo = db.getRepository(Transcription);
    const record = await repo.findOne({ where: { id } });
    if (!record) return Response.json({ error: "Not found" }, { status: 404 });

    if (body.speakerMap) {
        record.segments = record.segments.map((seg) => ({
            ...seg,
            speaker: body.speakerMap![seg.speaker] ?? seg.speaker,
        }));
    }
    if (body.title !== undefined) {
        record.title = body.title || null;
    }
    await repo.save(record);
    return Response.json({ segments: record.segments, title: record.title });
}

export async function DELETE(_req: Request, { params }: Ctx) {
    const { id } = await params;
    const db = await getDatabase();
    const repo = db.getRepository(Transcription);
    await repo.delete({ id });
    return Response.json({ success: true });
}
