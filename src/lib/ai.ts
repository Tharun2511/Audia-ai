import "server-only";
import Groq from "groq-sdk";
import { DeepgramClient } from "@deepgram/sdk";
import type { TranscriptSegment } from "@/entity/Transcription";

export const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
export const deepgram = new DeepgramClient({ apiKey: process.env.DEEPGRAM_API_KEY! });

export async function summarizeTranscript(segments: TranscriptSegment[]): Promise<string | null> {
    if (segments.length === 0) return null;
    const transcriptText = segments.map((s) => `${s.speaker}: ${s.text}`).join("\n");
    try {
        const res = await groq.chat.completions.create({
            messages: [{
                role: "user",
                content: `Summarize this conversation in 1-3 concise bullet points. Focus on key topics, decisions, and action items. Each bullet must start with "• ". Do not add any text before or after the bullets. If the conversation is too short or lacks enough content to summarize meaningfully, reply with exactly: "Not enough conversation to summarize."\n\nTranscript:\n${transcriptText}`,
            }],
            model: "llama-3.1-8b-instant",
            stream: false,
        });
        return res.choices[0]?.message?.content ?? null;
    } catch {
        return null;
    }
}
