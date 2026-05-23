import { getDatabase } from "@/db/data-source";
import { Transcription } from "@/entity/Transcription";
import { signAudioUrl } from "@/lib/audio-storage";
import { getCurrentUser } from "@/lib/dal";

type Ctx = { params: Promise<{ id: string }> };

/**
 * Issues a short-lived signed URL the browser can use to fetch this session's
 * audio blob. This is the only place a playable URL ever crosses the auth
 * boundary — clients without a valid session get 401, clients who don't own
 * the row get 404 (not 403, so we don't leak existence).
 *
 * The returned URL expires; the client transparently re-fetches on <audio>
 * onError if a long pause causes it to lapse mid-session.
 */
export async function GET(_req: Request, { params }: Ctx) {
    const user = await getCurrentUser();
    if (!user) return Response.json({ error: "Unauthorized" }, { status: 401 });

    const { id } = await params;
    const db = await getDatabase();
    const repo = db.getRepository(Transcription);
    const record = await repo.findOne({
        where: { id, userEmail: user.email },
        select: { id: true, audioPathname: true },
    });
    if (!record) return Response.json({ error: "Not found" }, { status: 404 });
    if (!record.audioPathname) return Response.json({ error: "No audio" }, { status: 404 });

    const url = await signAudioUrl(record.audioPathname);
    if (!url) return Response.json({ error: "Sign failed" }, { status: 500 });

    return Response.json({ url });
}
