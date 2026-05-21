import "server-only";
import Groq from "groq-sdk";
import { DeepgramClient } from "@deepgram/sdk";
import type { TranscriptSegment } from "@/entity/Transcription";
import { computeCost, logUsage } from "./ai-usage";

export const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
export const deepgram = new DeepgramClient({ apiKey: process.env.DEEPGRAM_API_KEY! });

const SUMMARY_SYSTEM_PROMPT = `You are a meeting summarizer. Output 1-3 concise bullet points covering key topics, decisions, and action items. Each bullet must start with "• ". Do not add any text before or after the bullets. If the conversation is too short or lacks enough content to summarize meaningfully, reply with exactly: "Not enough conversation to summarize."`;

export async function summarizeTranscript(segments: TranscriptSegment[]): Promise<string | null> {
    if (segments.length === 0) return null;
    const transcriptText = segments.map((s) => `${s.speaker}: ${s.text}`).join("\n");
    const model = "llama-3.1-8b-instant";
    const start = Date.now();
    try {
        const res = await groq.chat.completions.create({
            messages: [
                { role: "system", content: SUMMARY_SYSTEM_PROMPT },
                { role: "user", content: `Transcript:\n${transcriptText}` },
            ],
            model,
            temperature: 0.2,
            max_tokens: 250,
            stream: false,
        });
        const usage = res.usage;
        if (usage) {
            logUsage({
                label: "summarize",
                model,
                promptTokens: usage.prompt_tokens,
                completionTokens: usage.completion_tokens,
                latencyMs: Date.now() - start,
                cost: computeCost(model, usage.prompt_tokens, usage.completion_tokens),
            });
        }
        return res.choices[0]?.message?.content ?? null;
    } catch {
        return null;
    }
}
