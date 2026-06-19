/**
 * Phase 9.3 — live smoke test of the LangGraph agent MECHANICS (real model +
 * bindTools + ToolNode + the cycle), both ways, with STUB tools.
 *
 *   npm run graph:agent-demo
 *
 * Why stub tools: the real tools (agent-tools-lc → tools.ts → TypeORM entities)
 * can't be imported under tsx — esbuild doesn't emit the decorator metadata
 * TypeORM needs (ColumnTypeUndefinedError). That's a toolchain limit, not an
 * agent bug; the DB-backed agent ships in /api/chat-graph and runs under Next.
 * Here we prove the LangGraph wiring + Groq tool-calling work end-to-end:
 * real model emits tool_calls → ToolNode runs the (stub) tool → loops → answer.
 */
import { z } from "zod";
import { tool } from "@langchain/core/tools";
import { ChatGroq } from "@langchain/groq";
import { AIMessage, BaseMessage, HumanMessage, SystemMessage } from "@langchain/core/messages";
import { StateGraph, START, END, MessagesAnnotation } from "@langchain/langgraph";
import { ToolNode, createReactAgent } from "@langchain/langgraph/prebuilt";

const FAKE_MEETINGS = JSON.stringify({
    count: 2,
    meetings: [
        { id: "m1", title: "JavaScript Engine Overview", durationSec: 66 },
        { id: "m2", title: "Front End Interview Prep", durationSec: 49 },
    ],
});

const stubTools = [
    tool(async () => FAKE_MEETINGS, {
        name: "listMyMeetings",
        description: "List the user's recorded meetings (id, title, duration).",
        schema: z.object({ limit: z.number().int().optional() }),
    }),
];

const SYS = "Use listMyMeetings to answer questions about the user's meetings. Answer concisely.";
const MODEL = "llama-3.3-70b-versatile";

function buildExplicit() {
    const model = new ChatGroq({ apiKey: process.env.GROQ_API_KEY, model: MODEL, temperature: 0.2 }).bindTools(stubTools);
    const toolNode = new ToolNode(stubTools);
    async function modelNode(s: typeof MessagesAnnotation.State): Promise<Partial<typeof MessagesAnnotation.State>> {
        return { messages: [await model.invoke([new SystemMessage(SYS), ...s.messages])] };
    }
    function cont(s: typeof MessagesAnnotation.State): "tools" | "end" {
        const last = s.messages[s.messages.length - 1] as AIMessage;
        return (last?.tool_calls?.length ?? 0) > 0 ? "tools" : "end";
    }
    return new StateGraph(MessagesAnnotation)
        .addNode("model", modelNode)
        .addNode("tools", toolNode)
        .addEdge(START, "model")
        .addConditionalEdges("model", cont, { tools: "tools", end: END })
        .addEdge("tools", "model")
        .compile();
}

function buildReact() {
    const llm = new ChatGroq({ apiKey: process.env.GROQ_API_KEY, model: MODEL, temperature: 0.2 });
    return createReactAgent({ llm, tools: stubTools, prompt: SYS });
}

async function run(name: string, graph: { invoke: (i: { messages: BaseMessage[] }, c: object) => Promise<{ messages: BaseMessage[] }> }, q: string) {
    const { messages } = await graph.invoke({ messages: [new HumanMessage(q)] }, { recursionLimit: 8 });
    const toolsFired = messages
        .filter((m): m is AIMessage => m instanceof AIMessage && (m.tool_calls?.length ?? 0) > 0)
        .flatMap((m) => m.tool_calls!.map((c) => c.name));
    const last = messages[messages.length - 1];
    const answer = typeof last?.content === "string" ? last.content : JSON.stringify(last?.content);
    console.log(`\n=== ${name} ===\n  toolsFired : [${toolsFired.join(", ")}]\n  answer     : ${answer}`);
}

async function main() {
    const q = "What meetings did I have? List them.";
    console.log(`Question: "${q}"  (real llama-3.3-70b, STUB tools)`);
    await run("explicit StateGraph", buildExplicit(), q);
    await run("createReactAgent", buildReact(), q);
    console.log("\nSame model+tools loop, two builds — explicit StateGraph (control) vs createReactAgent (easy).");
}

main()
    .then(() => process.exit(0))
    .catch((err) => {
        console.error(err);
        process.exit(1);
    });
