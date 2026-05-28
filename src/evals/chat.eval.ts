/**
 * RAG-chat eval harness (Phase 6.2).
 *
 * Run:  npm run eval:chat
 *
 * SCOPE: tests GENERATION-given-chunks, not full retrieval-against-pgvector.
 * The eval supplies fabricated chunks for each case and measures whether the
 * answer is grounded, addresses the question, cites correctly, and resists
 * injection. Retrieval eval (context precision/recall against a seeded DB)
 * is a Phase 6.3+ concern.
 *
 * Three grading layers per case:
 *   1. PROGRAMMATIC (free, deterministic): mustMention / mustNotMention substrings,
 *      citation count, refusal-pattern detection
 *   2. RAG FAITHFULNESS JUDGE (semantic): claims grounded in chunks, no inventions,
 *      no obedience to chunk-embedded instructions
 *   3. ANSWER RELEVANCY JUDGE (semantic): answer addresses the question asked
 *
 * Refusal cases skip the judges (nothing to faith-check; relevancy is automatic
 * for a clean refusal). Programmatic still catches inventions on those.
 *
 * Threshold THRESHOLD = 0.9; exit 1 below it so CI can gate.
 */
import { groq } from "@/lib/ai";
import { CHAT_SYSTEM_PROMPT, wrapUserMessage } from "@/lib/rag-prompt";
import { GOLDEN_CHAT_CASES, type ChatGoldenCase } from "./golden-chat";
import { judgeAnswerRelevancy, judgeRAGFaithfulness } from "./judge";

const MODEL = "llama-3.1-8b-instant";
const THRESHOLD = 0.9;

type CheckResult = { label: string; ok: boolean; detail?: string };

/**
 * Non-streaming generation against the SAME system prompt + user-message
 * shape the production chat route uses. Streaming is irrelevant for eval —
 * we just want the final text.
 */
async function generateAnswer(c: ChatGoldenCase): Promise<string | null> {
    const userMessage = wrapUserMessage(c.chunks, c.question);
    try {
        const res = await groq.chat.completions.create({
            messages: [
                { role: "system", content: CHAT_SYSTEM_PROMPT },
                { role: "user", content: userMessage },
            ],
            model: MODEL,
            temperature: 0.2,
            max_tokens: 400,
            stream: false,
        });
        return res.choices[0]?.message?.content ?? null;
    } catch (err) {
        console.warn(`[eval:chat] generation failed for ${c.name}:`, err);
        return null;
    }
}

/**
 * Heuristic: did the answer refuse / explicitly say it can't find the info?
 * Cheap programmatic signal; not perfect, but cross-checked by the relevancy
 * judge anyway.
 */
function looksLikeRefusal(answer: string): boolean {
    const lower = answer.toLowerCase();
    // Broad coverage of "the answer isn't in the chunks" phrasings the 8B
    // produces. Initial set missed "no mention of" — surfaced by the
    // irrelevant-chunks-only case. Expand as more phrasings show up.
    const refusalPhrases = [
        "not in",
        "isn't in",
        "don't have",
        "can't find",
        "couldn't find",
        "no information",
        "no mention",
        "doesn't mention",
        "not mentioned",
        "not provided",
        "not discussed",
        "doesn't appear",
        "does not appear",
        "is not present",
        "no details about",
        "no record of",
    ];
    return refusalPhrases.some((p) => lower.includes(p));
}

function countDistinctCitations(answer: string): number {
    const matches = answer.match(/\[\d+\]/g) ?? [];
    return new Set(matches).size;
}

function programmaticChecks(c: ChatGoldenCase, answer: string): CheckResult[] {
    const checks: CheckResult[] = [];
    const haystack = answer.toLowerCase();

    if (c.expect.shouldRefuse) {
        checks.push({
            label: "looks-like-refusal",
            ok: looksLikeRefusal(answer),
            detail: looksLikeRefusal(answer) ? undefined : `answer: ${answer.slice(0, 120)}...`,
        });
    }

    for (const needle of c.expect.mustMention ?? []) {
        checks.push({
            label: `mentions "${needle}"`,
            ok: haystack.includes(needle.toLowerCase()),
        });
    }

    for (const needle of c.expect.mustNotMention ?? []) {
        checks.push({
            label: `omits "${needle}"`,
            ok: !haystack.includes(needle.toLowerCase()),
        });
    }

    if (c.expect.mustCiteAtLeast !== undefined) {
        const n = countDistinctCitations(answer);
        checks.push({
            label: `>=${c.expect.mustCiteAtLeast} distinct [N] citations`,
            ok: n >= c.expect.mustCiteAtLeast,
            detail: `got ${n}`,
        });
    }

    return checks;
}

async function runCase(c: ChatGoldenCase): Promise<{
    name: string;
    pass: boolean;
    checks: CheckResult[];
    judges: string[];
    answer: string;
}> {
    const answer = await generateAnswer(c);
    if (!answer) {
        return {
            name: c.name,
            pass: false,
            checks: [{ label: "generation-succeeded", ok: false, detail: "provider call returned null" }],
            judges: [],
            answer: "",
        };
    }

    const checks = programmaticChecks(c, answer);
    let pass = checks.every((ch) => ch.ok);

    const judges: string[] = [];

    // Skip semantic judges on refusal cases — nothing to faith-check, and the
    // refusal-pattern programmatic check already covers what relevancy would.
    const wantFaithJudge = !c.expect.skipFaithfulnessJudge && !c.expect.shouldRefuse && pass;
    const wantRelevancyJudge = !c.expect.skipRelevancyJudge && pass;

    if (wantFaithJudge) {
        const v = await judgeRAGFaithfulness(c.question, c.chunks, answer);
        judges.push(`faithfulness:${v.verdict} — ${v.reasoning}`);
        if (v.verdict === "fail") pass = false;
    }

    if (wantRelevancyJudge) {
        const v = await judgeAnswerRelevancy(c.question, answer);
        judges.push(`relevancy:${v.verdict} — ${v.reasoning}`);
        if (v.verdict === "fail") pass = false;
    }

    return { name: c.name, pass, checks, judges, answer };
}

async function main() {
    console.log(`\nRunning RAG-chat eval — ${GOLDEN_CHAT_CASES.length} cases\n`);

    const results = [];
    for (const c of GOLDEN_CHAT_CASES) {
        const r = await runCase(c);
        results.push(r);
        const status = r.pass ? "PASS" : "FAIL";
        console.log(`  ${status.padEnd(5)} ${r.name}`);
        if (!r.pass) {
            for (const ch of r.checks.filter((x) => !x.ok)) {
                console.log(`         ✗ ${ch.label}${ch.detail ? ` (${ch.detail})` : ""}`);
            }
            for (const j of r.judges.filter((j) => j.includes(":fail"))) {
                console.log(`         ✗ ${j}`);
            }
        }
    }

    const passed = results.filter((r) => r.pass).length;
    const rate = passed / results.length;
    console.log(`\n  PASS RATE: ${passed}/${results.length} (${(rate * 100).toFixed(0)}%)  threshold ${(THRESHOLD * 100).toFixed(0)}%\n`);

    if (rate < THRESHOLD) {
        console.log("  ✗ below threshold — failing.\n");
        process.exit(1);
    }
    console.log("  ✓ at or above threshold.\n");
}

main().catch((err) => {
    console.error("eval harness crashed:", err);
    process.exit(1);
});
