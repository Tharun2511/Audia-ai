import { getDatabase } from "@/db/data-source";
import { Transcription } from "@/entity/Transcription";
import { summarizeTranscript } from "@/lib/ai";

type Ctx = { params: Promise<{ id: string }> };

export async function POST(_req: Request, { params }: Ctx) {
    const { id } = await params;
    const db = await getDatabase();
    const repo = db.getRepository(Transcription);
    const record = await repo.findOne({ where: { id } });
    if (!record) return Response.json({ error: "Not found" }, { status: 404 });

    record.summary = await summarizeTranscript(record.segments);
    await repo.save(record);
    return Response.json({ summary: record.summary });
}
