import "server-only";
import { getDatabase } from "@/db/data-source";
import { ChatMessage } from "@/entity/ChatMessage";

/**
 * How many turn-pairs (user + assistant) to keep in the rolling buffer. 5 is a
 * common production default for chatbot UIs: enough for natural follow-ups
 * ("and Bob?"), few enough that the prompt stays small even on a long chat.
 * Promote to env if we want to tune per-deployment.
 */
export const HISTORY_TURN_PAIRS = 5;
export const HISTORY_MESSAGE_LIMIT = HISTORY_TURN_PAIRS * 2;

export type HistoryMessage = {
    role: "user" | "assistant";
    content: string;
};

/**
 * Load the most recent N messages for a session, oldest-first (the order the
 * LLM API expects). Filters by userEmail for ownership — a leaked sessionId
 * must not surface another user's chat history.
 *
 * We query DESC + LIMIT (so the DB returns only the last N rows even if the
 * session has 1000) and reverse in memory for chronological order.
 */
export async function loadRecentTurns(
    sessionId: string,
    userEmail: string,
    limit = HISTORY_MESSAGE_LIMIT,
): Promise<HistoryMessage[]> {
    const db = await getDatabase();
    const repo = db.getRepository(ChatMessage);

    const rows = await repo.find({
        where: { sessionId, userEmail },
        order: { createdAt: "DESC" },
        take: limit,
        select: ["role", "content"],
    });

    // Strip citation markers ([1], [2], ...) from past assistant turns. Those
    // numbers refer to chunks numbered against THAT turn's retrieval; the
    // current turn's chunks are numbered fresh and conflicting markers would
    // confuse the model into citing the wrong source.
    return rows
        .reverse()
        .map((m) => ({
            role: m.role,
            content: m.role === "assistant" ? stripCitations(m.content) : m.content,
        }));
}

/**
 * Persist one turn. The user turn is saved before streaming starts (so even
 * an aborted/crashed stream leaves the question in history); the assistant
 * turn is saved when the stream closes (with whatever content accumulated).
 */
export async function saveTurn(params: {
    sessionId: string;
    userEmail: string;
    transcriptionId: string | null;
    role: "user" | "assistant";
    content: string;
    citations?: unknown[] | null;
}): Promise<void> {
    const db = await getDatabase();
    const repo = db.getRepository(ChatMessage);
    await repo.save(
        repo.create({
            sessionId: params.sessionId,
            userEmail: params.userEmail,
            transcriptionId: params.transcriptionId,
            role: params.role,
            content: params.content,
            citations: params.citations ?? null,
        }),
    );
}

/**
 * Remove [N] citation markers from text. We keep them in the persisted row
 * (the UI re-renders chips on conversation reload), but strip them when the
 * text becomes LLM input — see loadRecentTurns above for why.
 *
 * Regex matches a digit-only marker like [1], [12]. We tolerate trailing
 * spaces collapsing so "fact [1] [2] confirmed" → "fact  confirmed", then
 * collapse double-spaces.
 */
function stripCitations(text: string): string {
    return text.replace(/\[\d+\]/g, "").replace(/ {2,}/g, " ").trim();
}
