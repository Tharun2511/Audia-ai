import { getDatabase } from "@/db/data-source";
import { Transcription } from "@/entity/Transcription";
import { getCurrentUser } from "@/lib/dal";

type Ctx = { params: Promise<{ id: string }> };

export async function PATCH(req: Request, { params }: Ctx) {
    const user = await getCurrentUser();
    if (!user) return Response.json({ error: "Unauthorized" }, { status: 401 });

    const { id } = await params;
    const body = (await req.json()) as { speakerMap?: Record<string, string>; title?: string };

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
