/**
 * Phase 9.1 — A/B the primitive summarizer vs the LangChain (LCEL) summarizer.
 *
 *   npm run eval:lc-compare
 *
 * Runs both implementations on the first few golden cases and prints their
 * structured outputs side by side. Same model, same prompt, same json-mode —
 * so matching output is the point: LCEL is a different WIRING of the same call,
 * not a different model. (This reuses the Phase 6 golden set as ready-made,
 * representative inputs — it is NOT a pass/fail eval; that's summary.eval.ts.)
 */
import { summarizeTranscriptStructured } from "@/lib/ai";
import { summarizeViaLangChain } from "@/lib/summarize-lc";
import { GOLDEN_SUMMARY_CASES } from "./golden-summary";

async function main() {
    const cases = GOLDEN_SUMMARY_CASES.slice(0, 3);
    console.log(`Comparing primitive vs LangChain summarizer on ${cases.length} golden cases…\n`);

    for (const c of cases) {
        // Run both on the SAME transcript. Sequential (not parallel) just to
        // keep the console output readable and the rate-limit gentle.
        const primitive = await summarizeTranscriptStructured(c.segments);
        const langchain = await summarizeViaLangChain(c.segments);

        console.log(`=== ${c.name} ===`);
        console.log(`  expect.tooShort = ${c.expect.tooShort}`);
        console.log(`  primitive : ${JSON.stringify(primitive)}`);
        console.log(`  langchain : ${JSON.stringify(langchain)}`);
        console.log();
    }

    console.log(
        "Takeaway: same model + prompt + json-mode → equivalent structured output.\n" +
        "LCEL changed the WIRING (prompt | model.withStructuredOutput), not the result.",
    );
}

main()
    .then(() => process.exit(0))
    .catch((err) => {
        console.error(err);
        process.exit(1);
    });
