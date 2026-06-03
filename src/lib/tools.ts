import "server-only";
import { MoreThanOrEqual } from "typeorm";
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

    // Use repo.find instead of QueryBuilder + .select() — the explicit select
    // syntax with camelCase column names like `t.createdAt` can produce
    // partially-populated entity instances where some columns load as
    // undefined, depending on TypeORM version. The crash we saw in production
    // ("Cannot read properties of null reading toISOString") was exactly this.
    // repo.find loads every declared field correctly.
    // Build the where clause incrementally so optional date filter only attaches
    // when present. FindOptionsWhere<T> tolerates partial shape + FindOperator
    // values, which is exactly what MoreThanOrEqual produces.
    type WhereClause = { userEmail: string; createdAt?: ReturnType<typeof MoreThanOrEqual<Date>> };
    const where: WhereClause = { userEmail: ctx.userEmail };

    if (args.since) {
        const sinceDate = new Date(args.since);
        if (Number.isNaN(sinceDate.getTime())) {
            // The model passed an unparseable date — return a structured error
            // it can read and self-correct (don't throw). Tool error handling
            // is part of the prompt loop: the more legible the error, the
            // more likely the model fixes its next call.
            return { error: `Invalid 'since' value: ${args.since}. Expected ISO 8601 date.` };
        }
        where.createdAt = MoreThanOrEqual(sinceDate);
    }

    const rows = await repo.find({
        where,
        order: { createdAt: "DESC" },
        take: limit,
    });

    return {
        count: rows.length,
        // Defensive null/undefined guard on every field — the model gets a
        // structured value either way, never crashes on edge cases. Title null
        // = "an untitled meeting" gets handled by the prompt, not here.
        meetings: rows.map((r) => ({
            id: r.id,
            title: r.title ?? null,
            createdAt: r.createdAt instanceof Date
                ? r.createdAt.toISOString()
                : (r.createdAt ?? null),
            durationSec: r.duration ?? null,
        })),
    };
}

// ── Tool: getMeetingDetails ───────────────────────────────────────────────

const getMeetingDetailsSchema: ChatCompletionTool = {
    type: "function",
    function: {
        name: "getMeetingDetails",
        description:
            "Get detailed metadata + cached summary for ONE specific meeting by its id. " +
            "Use this when the user asks for more detail about a meeting you've already listed " +
            "(e.g. 'tell me more about Q3 Pricing Review', 'who was in that meeting?', 'what was the summary of Sprint Planning?') — " +
            "typically called AFTER listMyMeetings has returned a list of meetings with their ids. " +
            "" +
            "CRITICAL — getting the transcriptionId right: " +
            "(1) The transcriptionId MUST be copied verbatim from a previous listMyMeetings tool result's `meetings[].id` field. " +
            "(2) DO NOT invent, guess, or use placeholder/example UUIDs like 123e4567-e89b-12d3-a456-426614174000. " +
            "(3) DO NOT make up UUIDs that 'look right' — only IDs that appeared in a prior tool result are valid. " +
            "(4) If you haven't already called listMyMeetings this conversation, CALL THAT FIRST to get real ids. " +
            "(5) When the user names a meeting (e.g. 'JavaScript Engine Overview'), match it to the corresponding `title` in the listMyMeetings result and use THAT object's `id`. " +
            "" +
            "Do NOT call this to retrieve raw transcript chunks — answer content questions from the <context> sources already in your prompt.",
        parameters: {
            type: "object",
            properties: {
                transcriptionId: {
                    type: "string",
                    description:
                        "UUID copied verbatim from a prior listMyMeetings result's meetings[].id field. " +
                        "Must be a real UUID returned by a tool call in this conversation — NEVER a placeholder/example UUID, NEVER invented. " +
                        "If unsure which meeting the user means, call listMyMeetings first and match by title.",
                },
            },
            required: ["transcriptionId"],
        },
    },
};

type GetMeetingDetailsArgs = {
    transcriptionId?: string;
};

async function getMeetingDetails(
    args: GetMeetingDetailsArgs,
    ctx: { userEmail: string },
): Promise<unknown> {
    if (!args.transcriptionId || typeof args.transcriptionId !== "string") {
        return { error: "transcriptionId is required and must be a string." };
    }

    const db = await getDatabase();
    const repo = db.getRepository(Transcription);

    // Ownership filter — never return another user's meeting even with a leaked id.
    const t = await repo.findOne({
        where: { id: args.transcriptionId, userEmail: ctx.userEmail },
        select: ["id", "title", "duration", "createdAt", "summary", "segments"],
    });

    if (!t) {
        return { error: `No meeting found with id ${args.transcriptionId}. Did you call listMyMeetings first to get a valid id?` };
    }

    // Distinct speakers from segments — cheap aggregation, useful for the model.
    const speakers = Array.from(new Set(t.segments?.map((s) => s.speaker) ?? []));

    return {
        id: t.id,
        title: t.title,
        createdAt: t.createdAt.toISOString(),
        durationSec: t.duration ?? null,
        speakerCount: speakers.length,
        speakers,
        summary: t.summary,
        // Don't return full segments — model doesn't need raw transcript through
        // this tool (that's what RAG <context> is for); avoids ballooning the
        // tool result message into the context budget.
    };
}

// ── Registry + dispatcher ─────────────────────────────────────────────────

export const AUDIA_TOOLS: ChatCompletionTool[] = [listMyMeetingsSchema, getMeetingDetailsSchema];

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
            case "getMeetingDetails":
                return JSON.stringify(await getMeetingDetails(args as GetMeetingDetailsArgs, ctx));
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
