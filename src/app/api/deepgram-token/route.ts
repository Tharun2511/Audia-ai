import { getCurrentUser } from "@/lib/dal";
import { deepgram } from "@/lib/ai";

/**
 * Mint a short-lived Deepgram token for browser-side live transcription (10.2).
 *
 *   POST /api/deepgram-token  →  { token, expiresIn }
 *
 * Why this exists: a serverless Next.js route handler can't hold a persistent
 * WebSocket (it's request/response and ends), so the browser must open the
 * Deepgram live socket DIRECTLY. We never ship the real DEEPGRAM_API_KEY to the
 * client — instead the server (which has the key) grants a token that expires
 * in seconds. Audio then streams browser → Deepgram, never through us.
 *
 * Auth: gated on the Audia session — only logged-in users can mint a token.
 */
export async function POST() {
    const user = await getCurrentUser();
    if (!user) return Response.json({ error: "Unauthorized" }, { status: 401 });

    try {
        // Deepgram v5 SDK: short-lived bearer token (default TTL ~30s).
        const grant = await deepgram.auth.v1.tokens.grant();
        return Response.json({
            token: grant.access_token,
            expiresIn: grant.expires_in,
        });
    } catch (err) {
        console.warn("[deepgram-token] grant failed", err);
        // Graceful: live captions just won't start; batch transcription on stop
        // is unaffected.
        return Response.json({ error: "Could not start live transcription" }, { status: 502 });
    }
}
