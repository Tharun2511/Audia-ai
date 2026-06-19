import { HumanMessage } from "@langchain/core/messages";
import { getCurrentUser } from "@/lib/dal";
import { buildChatGraph, buildReactAgent, summarizeRun } from "@/lib/chat-graph-agent";

/**
 * Phase 9.3 — the LangGraph chat agent as a PARALLEL route, for A/B vs the
 * hand-rolled loop in /api/chat. Same model + tools, expressed as a graph.
 *
 *   POST /api/chat-graph   body: { question: string, mode?: "explicit" | "react" }
 *
 * `mode` picks which build to run:
 *   "explicit" (default) — the hand-built StateGraph (control path)
 *   "react"              — createReactAgent (easy path)
 * Both should fire the same tool and return an equivalent answer — that's the
 * point: the graph reproduces the primitive loop, declaratively.
 *
 * Scope: the agentic tool-loop only (no RAG/memory/streaming). Best probed with
 * meeting-entity questions, e.g. "what meetings did I have this week?".
 * Non-streaming JSON so the A/B is easy to read.
 *
 * Quick test (replace COOKIE with a dev session cookie):
 *   curl -X POST http://localhost:3000/api/chat-graph -H "Content-Type: application/json" \
 *     -H "Cookie: COOKIE" -d '{"question":"what meetings did I have?","mode":"explicit"}'
 */
export async function POST(req: Request) {
    const user = await getCurrentUser();
    if (!user) return Response.json({ error: "Unauthorized" }, { status: 401 });

    let body: { question?: string; mode?: string };
    try {
        body = (await req.json()) as { question?: string; mode?: string };
    } catch {
        return Response.json({ error: "Invalid JSON body" }, { status: 400 });
    }

    const question = typeof body.question === "string" ? body.question.trim() : "";
    if (!question) return Response.json({ error: "Missing question" }, { status: 400 });
    const mode = body.mode === "react" ? "react" : "explicit";

    const start = Date.now();
    // Build the graph per-request so the tools are scoped to this user.
    const graph = mode === "react" ? buildReactAgent(user.email) : buildChatGraph(user.email);

    try {
        // Both builds accept { messages } and bound the cycle with recursionLimit
        // (the graph form of MAX_TOOL_ITERATIONS).
        const result = await graph.invoke(
            { messages: [new HumanMessage(question)] },
            { recursionLimit: 8 },
        );
        const { answer, toolsFired } = summarizeRun(result.messages);

        console.log(`[chat-graph] mode=${mode} toolsFired=[${toolsFired.join(", ")}] ms=${Date.now() - start}`);

        return Response.json({
            mode,
            answer,
            toolsFired,
            messageCount: result.messages.length,
        });
    } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.warn("[chat-graph] graph run failed", err);
        return Response.json({ error: `Graph run failed: ${msg}` }, { status: 500 });
    }
}
