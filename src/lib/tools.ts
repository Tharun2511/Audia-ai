import "server-only";
import { getDatabase } from "@/db/data-source";
import { Transcription } from "@/entity/Transcription";
import type { ChatCompletionTool } from "groq-sdk/resources/chat/completions";

/**
 * Function-calling tools registered with the chat model (Phase 7.1).
 *
 * Three pieces live here together on purpose:
 *   1. The JSON-schema declaration the API gets via `tools: [...]`
 *   2. The dispatcher that runs when the model emits a tool_call
 *   3. The ownership-aware implementation of each tool
 *
 * Keeping schema next to dispatcher prevents a class of drift: rename the
 * function, the schema name moves with it; change a parameter, the validator
 * sees the change. If we split them across files later, the routing-prompt
 * (description) and the dispatch must move together.
 *
 * Note on `description` strings: those ARE the prompt the model reads at
 * decision time. Specific positive AND negative examples beat one-line summaries.
 */

// ── Tool: listMyMeetings ──────────────────────────────────────────────────

const listMyMeetingsSchema: ChatCompletionTool = {
    type: "function",
    function: {
        name: "listMyMeetings",
        // The first sentence describes capability; the rest is decision-time
        // routing — when to call this and, crucially, when NOT to. Vague
        // descriptions cause over-calling on greetings or under-calling on
        // valid queries.
        description:
            "List the user's recorded meetings with their metadata — id, title, date created, duration. " +
            "Use this whenever the user asks about meetings as ENTITIES — which meetings exist, when a meeting was, how long it ran, the title of a meeting, or any follow-up question about meeting metadata (e.g. 'what meetings did I have this week?', 'when was the React meeting?', 'how long was Sprint Planning?', 'show my recent meetings'). " +
            "ALSO call this again on follow-up questions that reference a previously mentioned meeting (the user is asking for more details about something you already listed). " +
            "Do NOT call this to answer questions about the CONTENT inside a specific meeting (what was said, decisions made, action items) — for those, answer from the numbered sources already in your prompt. " +
            "Do NOT call this for greetings or general questions about how Audia works.",
        parameters: {
            type: "object",
            properties: {
                limit: {
                    type: "integer",
                    description: "Max number of meetings to return. Default 10, max 50.",
                },
                since: {
                    type: "string",
                    description:
                        "Optional ISO 8601 date (e.g. '2026-05-01' or '2026-05-19T00:00:00Z'). " +
                        "If provided, only meetings created at or after this date are returned.",
                },
            },
            required: [],
        },
    },
};

type ListMyMeetingsArgs = {
    limit?: number;
    since?: string;
};

async function listMyMeetings(
    args: ListMyMeetingsArgs,
    ctx: { userEmail: string },
): Promise<unknown> {
    const limit = Math.min(Math.max(args.limit ?? 10, 1), 50);
    const db = await getDatabase();
    const repo = db.getRepository(Transcription);

    const qb = repo
        .createQueryBuilder("t")
        .where("t.userEmail = :email", { email: ctx.userEmail })
        .orderBy("t.createdAt", "DESC")
        .take(limit)
        .select(["t.id", "t.title", "t.duration", "t.createdAt"]);

    if (args.since) {
        const sinceDate = new Date(args.since);
        if (Number.isNaN(sinceDate.getTime())) {
            // The model passed an unparseable date — return a structured error
            // it can read and self-correct (don't throw). Tool error handling
            // is part of the prompt loop: the more legible the error, the
            // more likely the model fixes its next call.
            return { error: `Invalid 'since' value: ${args.since}. Expected ISO 8601 date.` };
        }
        qb.andWhere("t.createdAt >= :since", { since: sinceDate });
    }

    const rows = await qb.getMany();
    return {
        count: rows.length,
        // Return null for missing values — never placeholder strings. The model
        // would echo "(untitled)" verbatim into its answer, which leaks
        // implementation detail. Presentation (e.g. "an untitled meeting") is
        // the prompt's job, not the tool's.
        meetings: rows.map((r) => ({
            id: r.id,
            title: r.title,
            createdAt: r.createdAt.toISOString(),
            durationSec: r.duration ?? null,
        })),
    };
}

// ── Registry + dispatcher ─────────────────────────────────────────────────

export const AUDIA_TOOLS: ChatCompletionTool[] = [listMyMeetingsSchema];

type ToolCtx = { userEmail: string };

/**
 * Execute a tool by name. Returns a JSON-serialized string suitable for the
 * `content` field of a `role: "tool"` message back to the model.
 *
 * Unknown tools, bad argument JSON, and thrown errors all become structured
 * `{ error: ... }` results — the model gets a chance to self-correct on the
 * next iteration instead of crashing the loop.
 */
export async function dispatchTool(
    name: string,
    rawArgs: string,
    ctx: ToolCtx,
): Promise<string> {
    let args: unknown;
    try {
        args = JSON.parse(rawArgs || "{}");
    } catch {
        return JSON.stringify({ error: `Tool "${name}" received invalid JSON arguments.` });
    }

    try {
        switch (name) {
            case "listMyMeetings":
                return JSON.stringify(await listMyMeetings(args as ListMyMeetingsArgs, ctx));
            default:
                // Hallucinated tool name — return a list of valid names so the
                // model can self-correct. Better than a generic "unknown".
                const available = AUDIA_TOOLS
                    .map((t) => t.function?.name)
                    .filter((n): n is string => !!n)
                    .join(", ");
                return JSON.stringify({
                    error: `Tool "${name}" is not available. Available tools: ${available}.`,
                });
        }
    } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        return JSON.stringify({ error: `Tool "${name}" failed: ${msg}` });
    }
}
