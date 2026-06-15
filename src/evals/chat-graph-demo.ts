/**
 * Phase 9.2 — run the chat-graph SKELETON and watch the cycle fire.
 *
 *   npm run graph:demo
 *
 * Prints (a) each node as it fires (streamMode "updates") and (b) the final
 * accumulated message list — proving the conditional-edge CYCLE ran
 * (model → tools → model → END) and the APPEND reducer accumulated every turn.
 * Stubbed model/tools (no API key needed); 9.3 swaps the stubs for real.
 */
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { chatGraphSkeleton } from "@/lib/chat-graph";

async function main() {
    const question = "What meetings did I have this week?";
    const input = { question, messages: [new HumanMessage(question)] };

    console.log(`Invoking chat-graph skeleton with: "${question}"\n`);

    console.log("— node firings (streamMode: updates) —");
    for await (const step of await chatGraphSkeleton.stream(input, {
        streamMode: "updates",
        recursionLimit: 10,
    })) {
        console.log(`  ${Object.keys(step)[0]} fired`);
    }

    const final = await chatGraphSkeleton.invoke(input, { recursionLimit: 10 });
    console.log("\n— final messages (append reducer accumulated every turn) —");
    final.messages.forEach((m, i) => {
        const calls = m instanceof AIMessage && m.tool_calls?.length
            ? ` tool_calls=[${m.tool_calls.map((c) => c.name).join(", ")}]`
            : "";
        console.log(`  [${i}] ${m.getType()}: ${JSON.stringify(m.content)}${calls}`);
    });

    console.log("\nFlow: human → model(tool_calls) → tools → model(answer) → END");
    console.log("The back-edge tools→model IS the cycle; the reducer did every .push() for us.");
}

main()
    .then(() => process.exit(0))
    .catch((err) => {
        console.error(err);
        process.exit(1);
    });
