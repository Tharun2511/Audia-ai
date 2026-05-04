import type { NextRequest } from "next/server";
import { getDatabase } from "@/db/data-source";
import { Transcription } from "@/entity/Transcription";

export async function GET(req: NextRequest) {
    const email = req.nextUrl.searchParams.get("email");
    if (!email) return Response.json([]);

    const db = await getDatabase();
    const repo = db.getRepository(Transcription);
    const records = await repo.find({
        where: { userEmail: email },
        order: { createdAt: "DESC" },
        take: 20,
    });
    return Response.json(records);
}
