import { getDatabase } from "@/db/data-source";
import { Transcription } from "@/entity/Transcription";
import { getCurrentUser } from "@/lib/dal";

export async function GET() {
    const user = await getCurrentUser();
    if (!user) return Response.json({ error: "Unauthorized" }, { status: 401 });

    const db = await getDatabase();
    const repo = db.getRepository(Transcription);
    // No `take:` cap — the client-side sidebar search needs every row to be
    // findable. Reasonable at small scale (a few hundred rows ≈ a few MB JSON).
    // Revisit when per-user session counts cross ~500: at that point switch to
    // server-side search (Postgres trigram or tsvector index over segments)
    // plus cursor pagination on the sidebar list.
    const records = await repo.find({
        where: { userEmail: user.email },
        order: { createdAt: "DESC" },
    });
    return Response.json(records);
}
