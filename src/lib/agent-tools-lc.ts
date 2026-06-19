import "server-only";
import { z } from "zod";
import { tool } from "@langchain/core/tools";
import { AUDIA_TOOLS, dispatchTool } from "./tools";

/**
 * Phase 9.3 — the tool-format bridge.
 *
 * AUDIA_TOOLS (tools.ts) are OpenAI/Groq JSON-schema *specs* — data describing
 * tools, perfect for `groq.chat.completions.create({ tools })`. But LangGraph's
 * `ToolNode` / `ChatGroq.bindTools` need runnable LangChain `tool()` objects —
 * things that can EXECUTE. So we wrap each as a LangChain tool whose execute fn
 * calls the SAME `dispatchTool` the hand-rolled loop uses — reusing the
 * implementation + ownership scoping, just satisfying LangChain's interface.
 *
 * We reuse the ROUTING DESCRIPTIONS verbatim from AUDIA_TOOLS (the description
 * is the prompt the model reads at decision time — single source of truth, so
 * the graph agent routes identically to the hand-rolled one). The arg Zod
 * schemas are small and restated here (LangChain wants Zod; our registry holds
 * JSON schema) — the one accepted bit of duplication.
 *
 * Built PER-REQUEST with the authed user's email so every tool call stays
 * ownership-scoped — the closure captures `userEmail`.
 */

function descriptionOf(name: string): string {
    const spec = AUDIA_TOOLS.find((t) => t.function?.name === name);
    return spec?.function?.description ?? name;
}

export function buildAudiaTools(userEmail: string) {
    const ctx = { userEmail };

    const listMyMeetings = tool(
        // dispatchTool returns a JSON string → becomes the ToolMessage content.
        async (args) => dispatchTool("listMyMeetings", JSON.stringify(args ?? {}), ctx),
        {
            name: "listMyMeetings",
            description: descriptionOf("listMyMeetings"),
            schema: z.object({
                limit: z.number().int().optional().describe("Max meetings to return. Default 10, max 50."),
                since: z.string().optional().describe("Optional ISO 8601 date; only meetings at/after it."),
            }),
        },
    );

    const getMeetingDetails = tool(
        async (args) => dispatchTool("getMeetingDetails", JSON.stringify(args ?? {}), ctx),
        {
            name: "getMeetingDetails",
            description: descriptionOf("getMeetingDetails"),
            schema: z.object({
                transcriptionId: z
                    .string()
                    .describe("UUID copied verbatim from a prior listMyMeetings result's meetings[].id."),
            }),
        },
    );

    return [listMyMeetings, getMeetingDetails];
}
