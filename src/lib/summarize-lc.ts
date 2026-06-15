import "server-only";
import { ChatGroq } from "@langchain/groq";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { SystemMessage } from "@langchain/core/messages";
import type { TranscriptSegment } from "@/entity/Transcription";
import { SUMMARY_SYSTEM_PROMPT, SummaryResponseSchema, type SummaryResult } from "./ai";

/**
 * Phase 9.1 — the SAME summarizer as ai.ts's `summarizeTranscriptStructured`,
 * re-expressed as a LangChain **LCEL chain**, for a side-by-side A/B.
 *
 * The point of this file is to SEE what LangChain abstracts. The primitive
 * (ai.ts) hand-writes the whole sequence:
 *     build messages[] → groq.chat.completions.create({response_format:json_object})
 *     → JSON.parse → Zod safeParse → null on failure        (~25 lines)
 *
 * Here that collapses into a two-step Runnable pipeline:
 *
 *     prompt  |  model.withStructuredOutput(schema)
 *
 * Every LangChain piece implements the `Runnable` interface (.invoke/.stream/
 * .batch), and `.pipe()` wires them so one's output feeds the next — the chain
 * is itself a Runnable. `withStructuredOutput(..., {method:"jsonMode"})` is
 * LangChain's wrapper around the EXACT provider feature the primitive uses
 * (Groq `response_format: json_object`): it requests JSON, parses it, and
 * validates against the Zod schema, returning a typed object or throwing.
 *
 * We reuse ai.ts's prompt AND schema, so this is a FAITHFUL A/B: same model,
 * same system prompt, same json-mode, same validation contract — only the
 * wiring differs (hand-rolled vs LCEL).
 *
 * What LangChain buys here: the request + parse + validate plumbing is one call
 * instead of ~25 lines, and `.stream()`/`.batch()` would come free. What it
 * costs: a dependency + indirection — the json request and the JSON.parse now
 * happen inside the library, not in code you can step through.
 */

const model = new ChatGroq({
    apiKey: process.env.GROQ_API_KEY,
    model: "llama-3.1-8b-instant", // identical to the primitive
    temperature: 0.2,
    maxTokens: 400,
});

// The system prompt contains literal { } (the JSON few-shot examples), so it
// must NOT pass through the template parser — it treats `{x}` as a variable and
// would choke on `{"tooShort":false}`. A fixed SystemMessage keeps those braces
// literal; only the human turn is templated (the `{transcript}` variable).
const prompt = ChatPromptTemplate.fromMessages([
    new SystemMessage(SUMMARY_SYSTEM_PROMPT),
    ["human", "<transcript>\n{transcript}\n</transcript>"],
]);

// jsonMode mirrors the primitive's response_format:{type:"json_object"} —
// deliberately NOT functionCalling: the 8B's tool-calling discipline is weak
// (Phase 7.2), the primitive uses json mode, so we match it for a fair test.
const chain = prompt.pipe(
    model.withStructuredOutput(SummaryResponseSchema, { name: "summary", method: "jsonMode" }),
);

/**
 * Validated structured summary via the LCEL chain — same contract as
 * `summarizeTranscriptStructured`: returns `{ tooShort, bullets }` or null on
 * any provider/parse/shape failure.
 */
export async function summarizeViaLangChain(
    segments: TranscriptSegment[],
): Promise<SummaryResult | null> {
    if (segments.length === 0) return null;
    const transcript = segments.map((s) => `${s.speaker}: ${s.text}`).join("\n");
    try {
        // One invoke runs: format prompt → call model (json mode) → parse → validate.
        const result = await chain.invoke({ transcript });
        return result as SummaryResult;
    } catch (err) {
        console.warn("[summary-lc] chain failed", { err });
        return null;
    }
}
