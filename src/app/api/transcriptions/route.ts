import { getDatabase } from "@/db/data-source";
import { Transcription } from "@/entity/Transcription";
import { getCurrentUser } from "@/lib/dal";

export async function GET() {
    const user = await getCurrentUser();
    if (!user) return Response.json({ error: "Unauthorized" }, { status: 401 });

    const db = await getDatabase();
    const repo = db.getRepository(Transcription);
    const records = await repo.find({
        where: { userEmail: user.email },
        order: { createdAt: "DESC" },
        take: 20,
    });
    return Response.json(records);
}
