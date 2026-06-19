import "server-only";
import { ChatGroq } from "@langchain/groq";
import { AIMessage, BaseMessage, SystemMessage } from "@langchain/core/messages";
import { StateGraph, START, END, MessagesAnnotation, MemorySaver } from "@langchain/langgraph";
import { ToolNode, createReactAgent } from "@langchain/langgraph/prebuilt";
import { buildAudiaTools } from "./agent-tools-lc";

/**
 * Phase 9.4 — the checkpointer. A module-level singleton so the saved State
 * survives ACROSS requests (each request rebuilds the graph for per-user tools,
 * but they all share this saver). MemorySaver = RAM (fine for dev; lost on
 * restart). Production swap: PostgresSaver on the same Neon DB — one line.
 * The graph is keyed by thread_id at invoke time, so conversations stay separate.
 */
const checkpointer = new MemorySaver();

/**
 * Phase 9.3 — Audia's chat agent as a REAL LangGraph agent, built TWO ways:
 *   buildChatGraph()   — explicit StateGraph (the control path)
 *   buildReactAgent()  — createReactAgent     (the easy path, ~3 lines)
 *
 * Both reuse the SAME tools (buildAudiaTools → dispatchTool) and model, so they
 * route identically to the hand-rolled loop in app/api/chat/route.ts. Built
 * per-request (tools close over userEmail for ownership scoping).
 *
 * Scope (Phase 9.3): the AGENTIC LOOP only — model ↔ tools ↔ cycle. No RAG /
 * memory / streaming (orthogonal, already-mastered plumbing); a retrieve node
 * would slot in upstream of the model node as a 1-node extension.
 */

const MODEL = "llama-3.3-70b-versatile"; // same model as the hand-rolled chat route

export const AGENT_SYSTEM_PROMPT =
    "You are Audia's meeting assistant. Use the tools to answer questions about the " +
    "user's meetings — which meetings exist, when they were, how long, and their " +
    "details/summary. Call listMyMeetings to enumerate meetings; call getMeetingDetails " +
    "for one meeting using an id returned by a prior listMyMeetings call. Answer concisely " +
    "from the tool results. Never invent meetings or ids.";

// ── The easy path: createReactAgent builds the whole agent StateGraph ───────
// One call = model node + ToolNode + the conditional cycle, prebuilt.
export function buildReactAgent(userEmail: string) {
    const tools = buildAudiaTools(userEmail);
    const llm = new ChatGroq({ apiKey: process.env.GROQ_API_KEY, model: MODEL, temperature: 0.2 });
    // checkpointSaver gives the prebuilt agent the same per-thread memory.
    return createReactAgent({ llm, tools, prompt: AGENT_SYSTEM_PROMPT, checkpointSaver: checkpointer });
}

// ── The control path: the explicit StateGraph (9.2's skeleton, now real) ────
// Same shape as the hand-rolled loop: model → (tools? → model | END).
export function buildChatGraph(userEmail: string) {
    const tools = buildAudiaTools(userEmail);
    // bindTools attaches the tool schemas so the model can emit tool_calls.
    const model = new ChatGroq({ apiKey: process.env.GROQ_API_KEY, model: MODEL, temperature: 0.2 }).bindTools(tools);
    const toolNode = new ToolNode(tools); // prebuilt: runs the tools in the last AI message's tool_calls

    // Model node: prepend the system prompt each call (it lives outside state),
    // invoke the tool-bound model, append its turn. (MessagesAnnotation's reducer
    // appends + de-dupes by id — the prebuilt version of 9.2's hand-spelled channel.)
    async function modelNode(state: typeof MessagesAnnotation.State): Promise<Partial<typeof MessagesAnnotation.State>> {
        const response = await model.invoke([new SystemMessage(AGENT_SYSTEM_PROMPT), ...state.messages]);
        return { messages: [response] };
    }

    // The conditional edge: more tool calls? loop : stop. (= `if (toolCalls) … else break`.)
    function shouldContinue(state: typeof MessagesAnnotation.State): "tools" | "end" {
        const last = state.messages[state.messages.length - 1] as AIMessage;
        return (last?.tool_calls?.length ?? 0) > 0 ? "tools" : "end";
    }

    return new StateGraph(MessagesAnnotation)
        .addNode("model", modelNode)
        .addNode("tools", toolNode)
        .addEdge(START, "model")
        .addConditionalEdges("model", shouldContinue, { tools: "tools", end: END })
        .addEdge("tools", "model") // the back-edge = the cycle
        .compile({ checkpointer }); // ← persist State per thread_id (Phase 9.4)
}

/** Extract the final text answer + the names of every tool the run fired. */
export function summarizeRun(messages: BaseMessage[]): { answer: string; toolsFired: string[] } {
    const last = messages[messages.length - 1];
    const answer = typeof last?.content === "string" ? last.content : JSON.stringify(last?.content ?? "");
    const toolsFired = messages
        .filter((m): m is AIMessage => m instanceof AIMessage && (m.tool_calls?.length ?? 0) > 0)
        .flatMap((m) => m.tool_calls!.map((c) => c.name));
    return { answer, toolsFired };
}
