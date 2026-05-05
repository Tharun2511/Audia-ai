import { getDatabase } from "@/db/data-source";
import { Transcription } from "@/entity/Transcription";
import { summarizeTranscript } from "@/lib/ai";
import { getCurrentUser } from "@/lib/dal";

type Ctx = { params: Promise<{ id: string }> };

export async function POST(_req: Request, { params }: Ctx) {
    const user = await getCurrentUser();
    if (!user) return Response.json({ error: "Unauthorized" }, { status: 401 });

    const { id } = await params;
    const db = await getDatabase();
    const repo = db.getRepository(Transcription);
    const record = await repo.findOne({ where: { id, userEmail: user.email } });
    if (!record) return Response.json({ error: "Not found" }, { status: 404 });

    record.summary = await summarizeTranscript(record.segments);
    await repo.save(record);
    return Response.json({ summary: record.summary });
}
