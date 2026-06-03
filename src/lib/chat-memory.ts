import "server-only";
import { getDatabase } from "@/db/data-source";
import { ChatMessage } from "@/entity/ChatMessage";

/**
 * Rolling-buffer conversation memory (Phase 5.1 + Phase 7.2 tool extension).
 *
 * Memory now spans THREE message kinds, not two:
 *   - role: "user"     — what the user typed (raw question, no <context> wrapping)
 *   - role: "assistant"— what the model returned (text + optional tool_calls)
 *   - role: "tool"     — the result a dispatched tool returned, keyed by tool_call_id
 *
 * Why we persist tool turns (added 7.2): without them, a follow-up like
 * "When was that meeting scheduled?" has no way to recall the createdAt that
 * the previous turn's listMyMeetings already fetched — the model would have
 * to re-call the tool every turn. With them, the model sees its own past
 * tool_calls + their results and composes follow-ups from working memory.
 */

export const HISTORY_TURN_PAIRS = 5;
/**
 * We load more rows than HISTORY_TURN_PAIRS suggests because a "turn" may now
 * span 3 messages (user, assistant_with_calls, tool_result) rather than 2.
 * 5 turn-pairs × ~3 messages each = 15 message slots; we use 20 for headroom
 * + safe orphan-pair filtering at the window boundary.
 */
export const HISTORY_MESSAGE_LIMIT = 20;

/**
 * Shapes match what the Groq/OpenAI chat-completions API expects, so the
 * chat route can splat history directly into its `messages` array without
 * re-mapping. The four variants:
 */
export type HistoryMessage =
    | { role: "user"; content: string }
    | { role: "assistant"; content: string }
    | {
          role: "assistant";
          content: string | null;
          tool_calls: Array<{ id: string; type: "function"; function: { name: string; arguments: string } }>;
      }
    | { role: "tool"; tool_call_id: string; content: string };

/**
 * Load the most recent N messages for a session in chronological order, with
 * tool-message orphan filtering at the window boundary. Filters by userEmail
 * for ownership — a leaked sessionId must not surface another user's history.
 *
 * Why orphan filtering: the API requires every role:tool message immediately
 * follow an assistant turn whose tool_calls contain the matching id. If the
 * rolling-window LIMIT cuts mid-pair (the role:tool inside the window but its
 * assistant-with-calls outside), the API rejects the request. We drop any
 * role:tool whose tool_call_id isn't backed by an in-window assistant turn.
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
        select: ["role", "content", "toolCallId", "toolCalls"],
    });

    const chronological = rows.reverse();

    // First pass: collect every tool_call_id we have an in-window assistant for.
    // Those are the only valid anchors for role:tool messages in this window.
    const validToolCallIds = new Set<string>();
    for (const m of chronological) {
        if (m.role === "assistant" && Array.isArray(m.toolCalls)) {
            for (const tc of m.toolCalls) validToolCallIds.add(tc.id);
        }
    }

    // Second pass: build the API-shaped history, skipping orphan role:tool
    // messages and stripping citation markers from assistant text (the [N]
    // refer to per-turn chunk numbering, would confuse the model if carried).
    const out: HistoryMessage[] = [];
    for (const m of chronological) {
        if (m.role === "user") {
            out.push({ role: "user", content: m.content });
        } else if (m.role === "assistant") {
            const cleanContent = stripCitations(m.content);
            if (Array.isArray(m.toolCalls) && m.toolCalls.length > 0) {
                out.push({
                    role: "assistant",
                    content: cleanContent || null,
                    tool_calls: m.toolCalls.map((tc) => ({
                        id: tc.id,
                        type: "function" as const,
                        function: { name: tc.name, arguments: tc.arguments },
                    })),
                });
            } else {
                out.push({ role: "assistant", content: cleanContent });
            }
        } else if (m.role === "tool") {
            if (m.toolCallId && validToolCallIds.has(m.toolCallId)) {
                out.push({ role: "tool", tool_call_id: m.toolCallId, content: m.content });
            }
            // else: orphan, drop silently — the API would reject otherwise
        }
    }
    return out;
}

/**
 * Persist one turn — extended in 7.2 to accept `toolCalls` (for assistant
 * turns that emit calls) and `toolCallId` (for role:tool result turns).
 *
 * Ordering discipline: the chat route MUST save the assistant-with-calls
 * turn BEFORE saving each role:tool result, because loadRecentTurns
 * orders by createdAt and the API needs pairs in that order.
 */
export async function saveTurn(params: {
    sessionId: string;
    userEmail: string;
    transcriptionId: string | null;
    role: "user" | "assistant" | "tool";
    content: string;
    citations?: unknown[] | null;
    toolCallId?: string | null;
    toolCalls?: Array<{ id: string; name: string; arguments: string }> | null;
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
            toolCallId: params.toolCallId ?? null,
            toolCalls: params.toolCalls ?? null,
        }),
    );
}

/**
 * Remove [N] citation markers from text. We keep them in the persisted row
 * (the UI re-renders chips on conversation reload), but strip them when the
 * text becomes LLM input — the numbers refer to per-turn chunk numbering,
 * would mislead the model into citing the wrong source in the next turn.
 */
function stripCitations(text: string): string {
    return text.replace(/\[\d+\]/g, "").replace(/ {2,}/g, " ").trim();
}
