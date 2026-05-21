import "server-only";
import { z } from "zod";
import Groq from "groq-sdk";
import { DeepgramClient } from "@deepgram/sdk";
import type { TranscriptSegment } from "@/entity/Transcription";
import { computeCost, logUsage } from "./ai-usage";

export const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
export const deepgram = new DeepgramClient({ apiKey: process.env.DEEPGRAM_API_KEY! });

const SummaryResponseSchema = z
    .object({
        tooShort: z.boolean(),
        bullets: z.array(z.string()).max(3),
    })
    .refine((data) => data.tooShort || data.bullets.length > 0, {
        message: "If tooShort is false, bullets must be non-empty",
    });

const SUMMARY_SYSTEM_PROMPT = `You are a meeting summarizer. Given a transcript, output a JSON object with two fields:
- "tooShort": true if the transcript is too short or lacks substance to summarize meaningfully; false otherwise.
- "bullets": an array of 1-3 short factual bullet strings covering key topics, decisions, and action items. Each bullet should be a single sentence, no leading symbols. If tooShort is true, this must be an empty array.

Do not include any text outside the JSON object. Do not wrap the JSON in code fences.

Example 1:
Transcript:
Alice: We need to decide on the launch date.
Bob: March 15 works. Marketing is ready.
Alice: March 15 it is. Carol, draft the announcement.
Carol: I'll have a draft by Friday.

Response:
{"tooShort":false,"bullets":["Launch date confirmed for March 15","Marketing team confirmed ready for launch","Carol will draft the launch announcement by Friday"]}

Example 2:
Transcript:
Alice: Hi.
Bob: Hey.

Response:
{"tooShort":true,"bullets":[]}`;

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
            max_tokens: 400,
            response_format: { type: "json_object" },
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
        const rawContent = res.choices[0]?.message?.content;
        if (!rawContent) {
            console.warn("[summary] empty response from provider", { model });
            return null;
        }

        let parsedJson: unknown;
        try {
            parsedJson = JSON.parse(rawContent);
        } catch {
            console.warn("[summary] invalid JSON from provider", { rawContent });
            return null;
        }

        const validated = SummaryResponseSchema.safeParse(parsedJson);
        if (!validated.success) {
            console.warn("[summary] shape mismatch", {
                issues: validated.error.issues,
                rawContent,
            });
            return null;
        }

        if (validated.data.tooShort) return "Not enough conversation to summarize.";
        return validated.data.bullets.map((b) => `• ${b}`).join("\n");
    } catch (err) {
        console.warn("[summary] provider call failed", { err });
        return null;
    }
}
