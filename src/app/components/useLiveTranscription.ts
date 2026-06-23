"use client";
import { useCallback, useRef, useState } from "react";

/**
 * Live transcription hook (Phase 10.2). Streams the mic straight to Deepgram's
 * WebSocket from the BROWSER, authenticated by a short-lived token minted by
 * our server (/api/deepgram-token) — so the real API key never reaches the
 * client and no audio proxies through our serverless functions.
 *
 * Why a raw WebSocket (not the @deepgram/sdk client): browsers CANNOT set an
 * Authorization header on a WebSocket handshake, so the SDK's header-based auth
 * fails silently client-side. Deepgram's documented browser method is to pass
 * the credential via the Sec-WebSocket-Protocol subprotocol:
 *     new WebSocket(url, ["token", <token>])
 *
 * Exposes streaming results split into finalized (committed, is_final, speaker-
 * labelled) and interim (the current revisable draft, shown faded). The SAVED
 * transcript still comes from the batch /api/transcribe path on stop.
 */

export type LiveCaption = { speaker: string; text: string };
export type LiveStatus = "idle" | "connecting" | "live" | "error";

const WS_OPEN = 1;

// Minimal shape of a Deepgram live "Results" message (we parse JSON ourselves).
type DGResults = {
    type?: string;
    is_final?: boolean;
    channel?: { alternatives?: { transcript?: string; words?: { speaker?: number }[] }[] };
};

export function useLiveTranscription() {
    const [finalized, setFinalized] = useState<LiveCaption[]>([]);
    const [interim, setInterim] = useState("");
    const [status, setStatus] = useState<LiveStatus>("idle");
    const wsRef = useRef<WebSocket | null>(null);
    const recorderRef = useRef<MediaRecorder | null>(null);

    const reset = useCallback(() => {
        setFinalized([]);
        setInterim("");
        setStatus("idle");
    }, []);

    const start = useCallback(async (stream: MediaStream) => {
        setStatus("connecting");
        try {
            const res = await fetch("/api/deepgram-token", { method: "POST" });
            if (!res.ok) {
                console.warn("[live] token request failed", res.status);
                setStatus("error"); // graceful: recording + batch save still work
                return;
            }
            const { token } = (await res.json()) as { token: string };
            console.log("[live] got token, opening websocket");

            const params = new URLSearchParams({
                model: "nova-2",
                diarize: "true",
                diarize_model: "latest", // v2 diarizer isn't available for streaming
                interim_results: "true",
                punctuate: "true",
                smart_format: "true",
            });
            // Auth via query parameter — the grant token (a bearer JWT) as
            // access_token over the encrypted wss connection.
            //
            // KNOWN RUNTIME NOTE (10.2): in one dev environment all three Deepgram
            // browser-auth methods — subprotocol ["token", jwt], ["bearer", jwt],
            // and this ?access_token query — closed with the SAME transport-level
            // code 1006 (no server close-reason). Identical failure across three
            // auth methods + an earlier ECONNRESET points to the network dropping
            // the Deepgram WSS (proxy / VPN / AV SSL-inspection), NOT the code.
            // Verify on an unrestricted network. Fallback for a fully client-side
            // key: ["token", DEEPGRAM_API_KEY] (documented, but exposes the key).
            params.set("access_token", token);
            const ws = new WebSocket(`wss://api.deepgram.com/v1/listen?${params}`);
            wsRef.current = ws;

            ws.onopen = () => {
                console.log("[live] ws open → starting recorder");
                const rec = new MediaRecorder(stream);
                rec.ondataavailable = (e) => {
                    if (e.data.size > 0 && ws.readyState === WS_OPEN) ws.send(e.data);
                };
                rec.start(250); // 250ms chunks
                recorderRef.current = rec;
                setStatus("live");
            };

            ws.onmessage = (event) => {
                let msg: DGResults;
                try {
                    msg = JSON.parse(event.data as string) as DGResults;
                } catch {
                    return;
                }
                if (msg.type !== "Results") return;
                const alt = msg.channel?.alternatives?.[0];
                const transcript = alt?.transcript?.trim() ?? "";
                if (!transcript) return;
                const speakerIdx = alt?.words?.[0]?.speaker ?? 0;
                if (msg.is_final) {
                    setFinalized((prev) => [...prev, { speaker: `User${speakerIdx + 1}`, text: transcript }]);
                    setInterim("");
                } else {
                    setInterim(transcript); // a revisable draft
                }
            };

            ws.onclose = (e) => {
                // Auth/permission failures surface here as a close code + reason.
                console.warn(`[live] ws closed code=${e.code} reason="${e.reason}"`);
            };
            ws.onerror = () => {
                console.warn("[live] ws error");
                setStatus("error");
            };
        } catch (err) {
            console.warn("[live] start failed", err);
            setStatus("error");
        }
    }, []);

    const stop = useCallback(() => {
        recorderRef.current?.stop();
        recorderRef.current = null;
        const ws = wsRef.current;
        if (ws && ws.readyState === WS_OPEN) {
            // Tell Deepgram to flush + close cleanly.
            try {
                ws.send(JSON.stringify({ type: "CloseStream" }));
            } catch {
                // ignore
            }
            ws.close();
        }
        wsRef.current = null;
        setInterim("");
        setStatus("idle");
    }, []);

    return { finalized, interim, status, start, stop, reset };
}
