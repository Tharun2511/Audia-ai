import { getDatabase } from "@/db/data-source";
import { Transcription } from "@/entity/Transcription";
import { getCurrentUser } from "@/lib/dal";
import { describeSlide, jointSummary } from "@/lib/vision";

/**
 * Joint multimodal summary (Phase 11.1): fuse a meeting transcript with a slide
 * image shown during it.
 *
 *   POST /api/transcriptions/:id/slide-summary
 *   body: { imageBase64: string, mimeType: string }   (base64 = raw bytes, no data: prefix)
 *   → { slideText, jointSummary }
 *
 * Flow: load the (owned) transcript → VLM reads the slide → text model fuses both.
 * The slide-upload UI is the noted next increment; this is the backend capability.
 */
export async function POST(req: Request, ctx: { params: Promise<{ id: string }> }) {
    const user = await getCurrentUser();
    if (!user) return Response.json({ error: "Unauthorized" }, { status: 401 });

    const { id } = await ctx.params;

    let body: { imageBase64?: string; mimeType?: string };
    try {
        body = (await req.json()) as { imageBase64?: string; mimeType?: string };
    } catch {
        return Response.json({ error: "Invalid JSON body" }, { status: 400 });
    }
    const { imageBase64, mimeType } = body;
    if (!imageBase64 || !mimeType) {
        return Response.json({ error: "imageBase64 and mimeType are required" }, { status: 400 });
    }

    // Load the transcript — ownership-filtered so you can only summarize your own.
    const db = await getDatabase();
    const t = await db.getRepository(Transcription).findOne({
        where: { id, userEmail: user.email },
        select: ["id", "segments"],
    });
    if (!t) return Response.json({ error: "Meeting not found" }, { status: 404 });

    const transcriptText = (t.segments ?? []).map((s) => `${s.speaker}: ${s.text}`).join("\n");

    try {
        // VLM reads the slide, then late-fusion with the transcript.
        const slideText = await describeSlide(imageBase64, mimeType);
        const summary = await jointSummary(transcriptText, slideText);
        return Response.json({ slideText, jointSummary: summary });
    } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.warn("[slide-summary] failed", err);
        return Response.json({ error: `Slide summary failed: ${msg}` }, { status: 502 });
    }
}
