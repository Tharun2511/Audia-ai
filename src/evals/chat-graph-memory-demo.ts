/**
 * Phase 9.4 — prove the checkpointer (memory) and an interrupt (HITL), live,
 * self-contained (stub tools, MemorySaver, no DB → runs under tsx).
 *
 *   npm run graph:memory-demo
 *
 * Part 1 (memory): one graph, two turns on the SAME thread — the follow-up
 *   "how long was the first one?" only works because the checkpointer loaded
 *   the prior turn. A fresh thread can't answer it → proof it's the memory.
 * Part 2 (interrupt): a node pauses via interrupt(); we resume with Command.
 */
import { z } from "zod";
import { tool } from "@langchain/core/tools";
import { ChatGroq } from "@langchain/groq";
import { AIMessage, BaseMessage, HumanMessage, SystemMessage } from "@langchain/core/messages";
import { Annotation, StateGraph, START, END, MessagesAnnotation, MemorySaver, Command, interrupt } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";

const MODEL = "llama-3.3-70b-versatile";

// ── Part 1: a memory-enabled agent (stub tool) ────────────────────────────
function buildMemoryGraph() {
    const FAKE = JSON.stringify({
        count: 2,
        meetings: [
            { id: "m1", title: "JavaScript Engine Overview", durationSec: 66 },
            { id: "m2", title: "Front End Interview Prep", durationSec: 49 },
        ],
    });
    const tools = [
        tool(async () => FAKE, {
            name: "listMyMeetings",
            description: "List the user's meetings with their durations.",
            schema: z.object({}),
        }),
    ];
    const model = new ChatGroq({ apiKey: process.env.GROQ_API_KEY, model: MODEL, temperature: 0.2 }).bindTools(tools);
    const toolNode = new ToolNode(tools);
    const SYS = "Use listMyMeetings for meeting questions. Use earlier turns of the conversation. Answer in one sentence.";

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
        .compile({ checkpointer: new MemorySaver() }); // ← memory
}

function lastText(messages: BaseMessage[]): string {
    const last = messages[messages.length - 1];
    return typeof last?.content === "string" ? last.content : JSON.stringify(last?.content);
}

async function memoryDemo() {
    const g = buildMemoryGraph();
    console.log("── Part 1: memory (checkpointer + thread_id) ──");

    const t1 = { configurable: { thread_id: "t1" }, recursionLimit: 8 };
    const a1 = await g.invoke({ messages: [new HumanMessage("What meetings did I have? List them.")] }, t1);
    console.log(`  [thread t1] Q1 "what meetings did I have?"\n    → ${lastText(a1.messages)}`);

    // Same thread → the checkpointer loads Q1's turns; "the first one" has meaning.
    const a2 = await g.invoke({ messages: [new HumanMessage("How long was the first one?")] }, t1);
    console.log(`  [thread t1] Q2 "how long was the first one?"\n    → ${lastText(a2.messages)}   (remembered)`);

    // Fresh thread → no memory → it cannot resolve "the first one". In practice
    // the model flails (re-calls the tool trying to find "the first one") and
    // hits the recursion cap — a vivid demonstration that the memory is what
    // made Q2 answerable. We catch it so the demo continues.
    try {
        const a3 = await g.invoke(
            { messages: [new HumanMessage("How long was the first one?")] },
            { configurable: { thread_id: "fresh" }, recursionLimit: 8 },
        );
        console.log(`  [thread fresh] same Q2, no history\n    → ${lastText(a3.messages)}   (no memory)`);
    } catch (err) {
        const msg = err instanceof Error ? err.message.split("\n")[0] : String(err);
        console.log(`  [thread fresh] same Q2, no history\n    → could not resolve "the first one" without memory; the agent looped to the cap (${msg})`);
    }
}

// ── Part 2: an interrupt (human-in-the-loop) ──────────────────────────────
const ApprovalState = Annotation.Root({
    action: Annotation<string>(),
    approved: Annotation<string>(),
});

function buildInterruptGraph() {
    async function ask(s: typeof ApprovalState.State): Promise<Partial<typeof ApprovalState.State>> {
        // Pauses here, persists, and waits. On resume, interrupt() RETURNS the
        // value passed via Command({ resume }).
        const decision = interrupt(`Approve action "${s.action}"? (yes/no)`);
        return { approved: String(decision) };
    }
    return new StateGraph(ApprovalState)
        .addNode("ask", ask)
        .addEdge(START, "ask")
        .addEdge("ask", END)
        .compile({ checkpointer: new MemorySaver() });
}

async function interruptDemo() {
    console.log("\n── Part 2: interrupt (human-in-the-loop) ──");
    try {
        const g = buildInterruptGraph();
        const cfg = { configurable: { thread_id: "i1" } };

        const paused = await g.invoke({ action: "delete meeting 'Q3 Review'" }, cfg);
        const interrupts = (paused as Record<string, unknown>).__interrupt__;
        console.log(`  invoke → PAUSED, waiting for human. interrupt payload:`);
        console.log(`    ${JSON.stringify(interrupts)}`);

        const resumed = await g.invoke(new Command({ resume: "yes" }), cfg);
        console.log(`  resume with Command({resume:"yes"}) → approved = "${resumed.approved}"`);
    } catch (err) {
        console.log(`  (interrupt demo skipped: ${err instanceof Error ? err.message : String(err)})`);
    }
}

async function main() {
    await memoryDemo();
    await interruptDemo();
    console.log("\nMemory = checkpointer saves State per thread_id. HITL = interrupt() pauses, Command resumes.");
}

main()
    .then(() => process.exit(0))
    .catch((err) => {
        console.error(err);
        process.exit(1);
    });
