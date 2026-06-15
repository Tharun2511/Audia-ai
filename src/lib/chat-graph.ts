import { AIMessage, BaseMessage, ToolMessage } from "@langchain/core/messages";
import { Annotation, StateGraph, START, END } from "@langchain/langgraph";

/**
 * Phase 9.2 — Audia's chat agent, expressed as a LangGraph `StateGraph`
 * SKELETON. This is the structure we'll fill in for real in 9.3 (real model +
 * AUDIA_TOOLS + a route). Today the model + tool nodes are STUBBED so the graph
 * runs end-to-end and you can SEE the four core primitives:
 *
 *   • State + channels + reducer  → how turns accumulate
 *   • nodes                       → units of work (model / tools)
 *   • conditional edge            → branch: "more tool calls?" → loop : END
 *   • the back-edge (cycle)       → tools → model = your hand-rolled while-loop
 *
 * Map to the imperative loop in app/api/chat/route.ts:
 *   messages[] + .push()              →  the `messages` channel + append reducer
 *   while (iter < MAX_TOOL_ITERATIONS) →  the back-edge tools→model + recursionLimit
 *   if (toolCalls) execute; else break →  the `shouldContinue` conditional edge
 *   dispatchTool(...) then push        →  the `tools` node returning { messages: [...] }
 */

// ── State: the typed object threaded through every node ───────────────────
// Each key is a "channel"; the reducer says how a node's partial update MERGES.
export const ChatState = Annotation.Root({
    // The conversation. A node returns { messages: [oneNewTurn] } and the
    // APPEND reducer concatenates it — the declarative version of messages.push().
    // (This is what the prebuilt MessagesAnnotation gives you; we spell it out
    // so the reducer is visible. Real MessagesAnnotation also de-dupes by id.)
    messages: Annotation<BaseMessage[]>({
        reducer: (current, update) => current.concat(update),
        default: () => [],
    }),
    // The user's question. No reducer → DEFAULT behavior = overwrite (last wins).
    question: Annotation<string>(),
});

type ChatStateType = typeof ChatState.State;

// ── Node: model (STUBBED) ─────────────────────────────────────────────────
// In 9.3 this calls llama-3.3-70b with AUDIA_TOOLS. Today it fakes the ReAct
// decision deterministically: if it hasn't called a tool yet, "decide" to call
// one; once a tool result is in state, produce the final answer. That single
// tool round is enough to exercise the cycle (model → tools → model → END).
function modelNode(state: ChatStateType): Partial<ChatStateType> {
    const toolRounds = state.messages.filter((m) => m.getType() === "tool").length;

    if (toolRounds === 0) {
        // First pass: emit an assistant turn that "wants" a tool — exactly the
        // shape Groq returns when the model emits tool_calls.
        return {
            messages: [
                new AIMessage({
                    content: "",
                    tool_calls: [{ id: "call_1", name: "listMyMeetings", args: { limit: 5 } }],
                }),
            ],
        };
    }

    // A tool result has come back → produce the final text answer.
    return {
        messages: [
            new AIMessage(`(stub) Answer to "${state.question}", grounded in the tool result above.`),
        ],
    };
}

// ── Node: tools (STUBBED) ─────────────────────────────────────────────────
// In 9.3 this dispatches via tools.ts. Today it returns a fake ToolMessage per
// requested call. Note it RETURNS { messages: [...] } — the reducer appends;
// the node never mutates state directly.
function toolNode(state: ChatStateType): Partial<ChatStateType> {
    const last = state.messages[state.messages.length - 1] as AIMessage;
    const calls = last.tool_calls ?? [];
    const results = calls.map(
        (c) => new ToolMessage({ content: `(stub) result for ${c.name}`, tool_call_id: c.id ?? "" }),
    );
    return { messages: results };
}

// ── Conditional edge: the branch (loop or stop?) ──────────────────────────
// Reads the latest message; if the model asked for tools, route to the tools
// node (→ which loops back to model); otherwise we're done. This is the graph
// form of `if (toolCalls.length) { ... } else break`.
function shouldContinue(state: ChatStateType): "tools" | "end" {
    const last = state.messages[state.messages.length - 1];
    if (last?.getType() === "ai" && ((last as AIMessage).tool_calls?.length ?? 0) > 0) {
        return "tools";
    }
    return "end";
}

// ── Assemble the graph ────────────────────────────────────────────────────
//   START → model → (tools? → model | END)
// The `tools → model` edge is the back-edge: the cycle that replaces the loop.
export const chatGraphSkeleton = new StateGraph(ChatState)
    .addNode("model", modelNode)
    .addNode("tools", toolNode)
    .addEdge(START, "model")
    .addConditionalEdges("model", shouldContinue, { tools: "tools", end: END })
    .addEdge("tools", "model")
    .compile();
