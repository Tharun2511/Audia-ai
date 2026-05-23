import "server-only";
import { randomUUID } from "node:crypto";
import { issueSignedToken, presignUrl, put } from "@vercel/blob";

/**
 * Maps an audio MIME type to a file extension. The extension matters because
 * the browser uses it (along with the Content-Type header) to pick a decoder
 * for the <audio> element. WebM/Opus is what MediaRecorder produces in Chrome
 * by default; the others cover what users might upload.
 */
function extensionForMime(mime: string): string {
    if (mime.includes("webm")) return ".webm";
    if (mime.includes("mpeg")) return ".mp3"; // audio/mpeg = MP3
    if (mime.includes("wav")) return ".wav";
    if (mime.includes("m4a") || mime.includes("mp4")) return ".m4a";
    if (mime.includes("ogg")) return ".ogg";
    if (mime.includes("flac")) return ".flac";
    return "";
}

/**
 * Uploads a recording to a PRIVATE Vercel Blob store and returns the pathname.
 *
 * For private blobs, the URL returned by put() is not directly fetchable —
 * playback URLs must be issued per-request via signAudioUrl(). That's why we
 * persist the pathname (stable identifier) rather than the URL (ephemeral).
 *
 * Returns null on missing token or any failure so the calling route can still
 * save the transcription text — losing audio is bad but losing the transcript
 * is worse.
 */
export async function uploadAudio(buffer: Buffer, mimeType: string): Promise<string | null> {
    if (!process.env.BLOB_READ_WRITE_TOKEN) {
        console.warn("[audio-storage] BLOB_READ_WRITE_TOKEN not set — skipping audio persistence");
        return null;
    }

    const pathname = `audio/${randomUUID()}${extensionForMime(mimeType)}`;
    try {
        const blob = await put(pathname, buffer, {
            access: "private",
            contentType: mimeType,
        });
        return blob.pathname;
    } catch (err) {
        console.error("[audio-storage] upload failed:", err);
        return null;
    }
}

const DEFAULT_SIGNED_URL_TTL_MS = 60 * 60 * 1000; // 1 hour

/**
 * Issues a short-lived signed URL the browser can use to fetch a private blob.
 *
 * Flow:
 *   1) issueSignedToken: ask Vercel for a delegation token scoped to (pathname,
 *      get-only, expiry). One network call.
 *   2) presignUrl: locally HMAC the token into a fetchable URL. No network.
 *
 * The URL works only for GET on this exact pathname, only until validUntil.
 * If a user pauses playback past expiry, the <audio> element fires onError —
 * the client just re-fetches and continues.
 */
export async function signAudioUrl(pathname: string, ttlMs: number = DEFAULT_SIGNED_URL_TTL_MS): Promise<string | null> {
    if (!process.env.BLOB_READ_WRITE_TOKEN) return null;
    const validUntil = Date.now() + ttlMs;
    try {
        const signedToken = await issueSignedToken({
            pathname,
            operations: ["get"],
            validUntil,
        });
        const { presignedUrl } = await presignUrl(signedToken, {
            operation: "get",
            pathname,
            access: "private",
            validUntil,
        });
        return presignedUrl;
    } catch (err) {
        console.error("[audio-storage] sign failed:", err);
        return null;
    }
}
