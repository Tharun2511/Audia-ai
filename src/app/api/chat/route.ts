import { randomUUID } from "crypto";
import type {
    ChatCompletionChunk,
    ChatCompletionCreateParamsStreaming,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
} from "groq-sdk/resources/chat/completions";
import { groq } from "@/lib/ai";
import { computeCost, logUsage } from "@/lib/ai-usage";
import { HISTORY_MESSAGE_LIMIT, loadRecentTurns, saveTurn } from "@/lib/chat-memory";
import { findCandidateChunks } from "@/lib/chunks";
import { getCurrentUser } from "@/lib/dal";
import { embed } from "@/lib/embeddings";
import { CHAT_SYSTEM_PROMPT, wrapUserMessage } from "@/lib/rag-prompt";
import { maximalMarginalRelevance } from "@/lib/rerank";
import { AUDIA_TOOLS, dispatchTool } from "@/lib/tools";

/**
 * Cap on the agentic loop. Rule of thumb: expected_steps + 1 self-correction
 * slot. With 2 tools (listMyMeetings, getMeetingDetails) and the canonical
 * workflow being "list → pick → drill" or "list → answer", we expect ≤2 tool
 * iterations + the final text iteration = 3 productive iterations; 4 leaves
 * one slot for self-correction. Last iteration uses tool_choice="none" to
 * force a text answer. (Phase 7.2 bumped from 3 → 4 to support multi-step.)
 */
const MAX_TOOL_ITERATIONS = 4;

/**
 * Edge-position reordering: most-relevant chunks at positions 0 and k-1
 * (the prompt edges where LLM attention is strongest), least-relevant
 * buried in the middle. Mitigates lost-in-the-middle (Liu et al. 2023).
 *
 * Input order assumed to be by relevance (chunk[0] = most relevant).
 * For [A, B, C, D, E] returns [A, C, E, D, B].
 */
function edgeReorder<T>(items: T[]): T[] {
    const result = new Array<T>(items.length);
    let left = 0;
    let right = items.length - 1;
    for (let i = 0; i < items.length; i++) {
        if (i % 2 === 0) result[left++] = items[i];
        else result[right--] = items[i];
    }
    return result;
}

// groq-sdk@1.1.2 omits stream_options on params and `usage` on chunks, but the
// Groq API supports both (OpenAI-compatible) and emits usage in the final chunk.
type StreamingCreateParamsWithUsage = ChatCompletionCreateParamsStreaming & {
    stream_options?: { include_usage?: boolean };
};
type ChunkWithUsage = ChatCompletionChunk & {
    usage?: { prompt_tokens: number; completion_tokens: number };
};

export async function POST(req: Request) {
    const user = await getCurrentUser();
    if (!user) return Response.json({ error: "Unauthorized" }, { status: 401 });

    const body = (await req.json()) as {
        question?: string;
        transcriptionId?: string;
        /**
         * Opaque conversation key. Absent/null on first turn — server mints a
         * fresh UUID and returns it in the X-Chat-Session header. Subsequent
         * turns echo the same id so the server can load history.
         */
        sessionId?: string | null;
    };
    const question = typeof body.question === "string" ? body.question.trim() : "";
    if (!question) {
        return Response.json({ error: "Missing question" }, { status: 400 });
    }

    // First turn: mint a session id. Subsequent turns reuse the client's.
    const sessionId =
        typeof body.sessionId === "string" && body.sessionId.length > 0
            ? body.sessionId
            : randomUUID();
    const isNewSession = sessionId !== body.sessionId;
    const transcriptionId = body.transcriptionId ?? null;

    // Upgraded from llama-3.1-8b-instant to llama-3.3-70b-versatile (Phase 7.2 fix):
    // the 8B's tool-calling discipline broke at our prompt complexity — it would
    // emit calls in Llama's tag syntax (<function=X>{...}</function>) which Groq's
    // OpenAI-compatible API rejects with `tool_use_failed`. The 70B emits proper
    // tool_calls JSON. Trade-off: TPM rate limit is 12k vs 30k on 8B — fine for
    // typical chat. Both summarizer judge and faithfulness judge already use 70B,
    // so this is consistent.
    const model = "llama-3.3-70b-versatile";
    const start = Date.now();

    // ── RAG retrieval stage ───────────────────────────────────────────────
    // 1. Embed the question
    // 2. Coarse top-N from pgvector (scoped to user + optional transcript)
    // 3. MMR re-rank to top-k for diversity
    // 4. Edge-position reorder for lost-in-the-middle
    const queryEmbedding = await embed(question);
    const candidates = await findCandidateChunks(queryEmbedding, user.email, {
        transcriptionId: transcriptionId ?? undefined,
        n: 20,
    });

    // ── Memory load ───────────────────────────────────────────────────────
    // Rolling buffer: last N turn pairs for this session, oldest-first. Skips
    // the round-trip when we just minted the sessionId. Citations are stripped
    // from assistant turns at load time — see chat-memory.ts for why.
    const history = isNewSession
        ? []
        : await loadRecentTurns(sessionId, user.email, HISTORY_MESSAGE_LIMIT);

    // Diagnostic: log the retrieval + history shape so we can confirm both
    // scoping (RAG) and memory recall (last-N) are working as expected.
    console.log(
        `[chat] session=${sessionId.slice(0, 8)}${isNewSession ? " (new)" : ""} ` +
        `transcriptionId=${transcriptionId ?? "(none)"} candidates=${candidates.length} ` +
        `historyMsgs=${history.length} ` +
        `candidateTranscriptionIds=${JSON.stringify(Array.from(new Set(candidates.map((c) => c.transcriptionId))))}`,
    );

    // ── Empty-retrieval short-circuit ─────────────────────────────────────
    // If no chunks came back (transcript not yet indexed, or no relevant
    // content), don't ask the LLM — it will hallucinate to fill the void.
    // Stream a clear, deterministic message instead. Citation chips render as
    // an empty array, so nothing tries to seek to a nonexistent timestamp.
    if (candidates.length === 0) {
        const msg =
            "I couldn't find any indexed content for this question. " +
            "If this meeting was recorded before the chat feature was enabled, the transcript hasn't been indexed yet — " +
            "try a meeting recorded recently, or re-record this one. " +
            "I won't invent answers from nothing.";

        // Persist both turns even on the deterministic refusal path so the
        // session has a coherent record. Fire-and-forget — a DB hiccup must
        // not block the stream.
        void saveTurn({ sessionId, userEmail: user.email, transcriptionId, role: "user", content: question })
            .then(() => saveTurn({ sessionId, userEmail: user.email, transcriptionId, role: "assistant", content: msg, citations: [] }))
            .catch((err) => console.warn("[chat] persist (empty-retrieval) failed", err));

        const noContextStream = new ReadableStream<Uint8Array>({
            start(controller) {
                controller.enqueue(new TextEncoder().encode(msg));
                controller.close();
            },
        });
        return new Response(noContextStream, {
            headers: {
                "Content-Type": "text/plain; charset=utf-8",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Citations": "[]",
                "X-Chat-Session": sessionId,
                "Access-Control-Expose-Headers": "X-Citations, X-Chat-Session",
            },
        });
    }

    const mmrPicks = maximalMarginalRelevance(queryEmbedding, candidates, 5, 0.7);
    const chunks = edgeReorder(mmrPicks);

    // Build the user message: <context> + <user_input>, both delimited so the
    // system rules can distinguish data from question.
    const userMessage = wrapUserMessage(chunks, question);

    // Citations payload sent in a response header. Compact JSON: only what the
    // client needs to render chips (chunk number, speakers, time range, short
    // text preview, and the startTime for seek-on-click).
    const citationsHeader = JSON.stringify(
        chunks.map((c, i) => ({
            n: i + 1,
            chunkId: c.id,
            transcriptionId: c.transcriptionId,
            speakers: c.speakers,
            startTime: c.startTime,
            endTime: c.endTime,
            preview: c.text.slice(0, 140),
        })),
    );

    // Build the full message list: system, then the rolling-buffer history,
    // then the current user turn wrapped with context. History is already
    // API-shaped from loadRecentTurns (includes assistant-with-tool_calls and
    // role:tool messages from prior turns, with orphan-pair filtering done).
    const messages: ChatCompletionMessageParam[] = [
        { role: "system", content: CHAT_SYSTEM_PROMPT },
        ...(history as ChatCompletionMessageParam[]),
        { role: "user", content: userMessage },
    ];

    // Persist the user turn BEFORE streaming starts. Even if the stream
    // explodes or the client aborts, the question is part of the record. We
    // store the raw question, not the wrapped userMessage — the context block
    // is per-turn ephemera, not history.
    try {
        await saveTurn({ sessionId, userEmail: user.email, transcriptionId, role: "user", content: question });
    } catch (err) {
        // Don't block the stream on a DB hiccup; we'll lose this turn from
        // history but the user still gets an answer.
        console.warn("[chat] persist user turn failed", err);
    }

    const encoder = new TextEncoder();
    const stream = new ReadableStream<Uint8Array>({
        async start(controller) {
            // ── Agentic loop (Phase 7.1) ──────────────────────────────────
            // Up to MAX_TOOL_ITERATIONS rounds of:
            //   1. Stream a completion. Forward delta.content to the client,
            //      accumulate delta.tool_calls separately (they're not user-facing).
            //   2. If the stream ended with tool_calls, execute them, append
            //      their results to `messages`, loop.
            //   3. If the stream ended with content (no tool_calls), we're done.
            // The LAST allowed iteration uses tool_choice="none" so the model
            // is forced to produce a final text answer instead of looping forever.

            let accumulated = "";
            let totalPromptTokens = 0;
            let totalCompletionTokens = 0;
            let streamErr: unknown = null;

            try {
                for (let iter = 0; iter < MAX_TOOL_ITERATIONS; iter++) {
                    if (req.signal.aborted) break;
                    const isLast = iter === MAX_TOOL_ITERATIONS - 1;

                    const params: StreamingCreateParamsWithUsage = {
                        messages,
                        model,
                        tools: AUDIA_TOOLS,
                        tool_choice: isLast ? "none" : "auto",
                        stream: true,
                        stream_options: { include_usage: true },
                    };

                    // Per-iteration accumulators. tool_calls arrive in deltas
                    // keyed by `index`; merge name + arguments fragments.
                    const toolCallsByIndex = new Map<number, {
                        id: string;
                        name: string;
                        arguments: string;
                    }>();
                    let iterContent = "";

                    const aiStream = await groq.chat.completions.create(params, { signal: req.signal });
                    for await (const rawChunk of aiStream) {
                        if (req.signal.aborted) break;
                        const chunk = rawChunk as ChunkWithUsage;
                        const delta = chunk.choices[0]?.delta;

                        if (delta?.content) {
                            iterContent += delta.content;
                            accumulated += delta.content;
                            try {
                                controller.enqueue(encoder.encode(delta.content));
                            } catch {
                                // client disconnected — stop pulling tokens
                                break;
                            }
                        }

                        if (delta?.tool_calls) {
                            for (const tc of delta.tool_calls) {
                                const idx = tc.index ?? 0;
                                const accum = toolCallsByIndex.get(idx) ?? { id: "", name: "", arguments: "" };
                                if (tc.id) accum.id = tc.id;
                                if (tc.function?.name) accum.name = tc.function.name;
                                if (tc.function?.arguments) accum.arguments += tc.function.arguments;
                                toolCallsByIndex.set(idx, accum);
                            }
                        }

                        if (chunk.usage) {
                            totalPromptTokens += chunk.usage.prompt_tokens;
                            totalCompletionTokens += chunk.usage.completion_tokens;
                        }
                    }

                    const toolCalls = [...toolCallsByIndex.entries()]
                        .sort(([a], [b]) => a - b)
                        .map(([, v]) => v)
                        .filter((tc) => tc.id && tc.name);

                    if (toolCalls.length === 0) {
                        // No tool calls this iteration — model produced a final
                        // text answer (or empty, which we treat as done).
                        break;
                    }

                    // Append the assistant's tool-calling turn to in-memory message history,
                    // then execute each tool and append the results. The model needs
                    // to see its own tool_calls turn alongside the tool results on
                    // the next iteration — otherwise it has no reference for what
                    // each tool_call_id corresponds to.
                    const assistantToolCalls: ChatCompletionMessageToolCall[] = toolCalls.map((tc) => ({
                        id: tc.id,
                        type: "function",
                        function: { name: tc.name, arguments: tc.arguments || "{}" },
                    }));
                    messages.push({
                        role: "assistant",
                        // OpenAI's spec: content can be null when only tool_calls are returned.
                        // We pass iterContent in case the model emitted a mix of text + calls.
                        content: iterContent || null,
                        tool_calls: assistantToolCalls,
                    } as ChatCompletionMessageParam);

                    // PERSIST the assistant-with-calls turn (Phase 7.2) so a follow-up
                    // turn can reload working memory of what we called. Order matters:
                    // assistant turn before tool results, with createdAt ascending.
                    try {
                        await saveTurn({
                            sessionId,
                            userEmail: user.email,
                            transcriptionId,
                            role: "assistant",
                            content: iterContent,
                            toolCalls: toolCalls.map((tc) => ({
                                id: tc.id,
                                name: tc.name,
                                arguments: tc.arguments || "{}",
                            })),
                        });
                    } catch (err) {
                        console.warn("[chat] persist assistant tool-calls turn failed", err);
                    }

                    console.log(
                        `[chat] iter=${iter} tool_calls=${toolCalls.map((tc) => `${tc.name}(${tc.arguments.slice(0, 60)})`).join(", ")}`,
                    );

                    // Execute tools sequentially (small N; keeps logs ordered + DB
                    // writes deterministic). Persist each result as a role:tool row.
                    for (const tc of toolCalls) {
                        const result = await dispatchTool(tc.name, tc.arguments, { userEmail: user.email });
                        messages.push({
                            role: "tool",
                            tool_call_id: tc.id,
                            content: result,
                        } as ChatCompletionMessageParam);

                        try {
                            await saveTurn({
                                sessionId,
                                userEmail: user.email,
                                transcriptionId,
                                role: "tool",
                                content: result,
                                toolCallId: tc.id,
                            });
                        } catch (err) {
                            console.warn(`[chat] persist tool-result turn failed (call ${tc.id})`, err);
                        }
                    }
                    // continue loop — next iteration sees tool results in context
                }
            } catch (err) {
                if (!(err instanceof Error && err.name === "AbortError") && !req.signal.aborted) {
                    streamErr = err;
                    console.warn("[chat] stream error", err);
                }
            } finally {
                // Persist the assistant turn with whatever accumulated (even
                // partial on abort — the user saw it; it's part of history).
                if (accumulated.length > 0) {
                    try {
                        await saveTurn({
                            sessionId,
                            userEmail: user.email,
                            transcriptionId,
                            role: "assistant",
                            content: accumulated,
                            citations: chunks.map((c, i) => ({
                                n: i + 1,
                                chunkId: c.id,
                                transcriptionId: c.transcriptionId,
                                speakers: c.speakers,
                                startTime: c.startTime,
                                endTime: c.endTime,
                                preview: c.text.slice(0, 140),
                            })),
                        });
                    } catch (err) {
                        console.warn("[chat] persist assistant turn failed", err);
                    }
                }

                if (totalPromptTokens > 0 || totalCompletionTokens > 0) {
                    logUsage({
                        label: req.signal.aborted ? "chat-aborted" : "chat",
                        model,
                        promptTokens: totalPromptTokens,
                        completionTokens: totalCompletionTokens,
                        latencyMs: Date.now() - start,
                        cost: computeCost(model, totalPromptTokens, totalCompletionTokens),
                    });
                }

                if (streamErr) {
                    try { controller.error(streamErr); } catch {}
                } else {
                    try { controller.close(); } catch {}
                }
            }
        },
    });

    return new Response(stream, {
        headers: {
            "Content-Type": "text/plain; charset=utf-8",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Citations": citationsHeader,
            "X-Chat-Session": sessionId,
            // Expose the custom headers to the client (browsers hide non-standard
            // headers from fetch() by default unless the server announces them).
            "Access-Control-Expose-Headers": "X-Citations, X-Chat-Session",
        },
    });
}
