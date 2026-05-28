/**
 * Summarizer eval harness (Phase 6.1).
 *
 * Run:  npm run eval
 * (wired as: node --import tsx --conditions=react-server --env-file=.env src/evals/summary.eval.ts)
 *
 *   --conditions=react-server  → makes `import "server-only"` resolve to a no-op
 *                                so we can import @/lib/ai outside Next's runtime
 *   --env-file=.env            → loads GROQ_API_KEY (a standalone script doesn't
 *                                inherit Next's automatic .env loading)
 *   tsx                        → TypeScript + tsconfig `@/*` path resolution
 *
 * Each case runs two grading layers:
 *   1. PROGRAMMATIC (deterministic, free): did it return a valid structured
 *      result? does tooShort match? <=3 bullets? required facts present?
 *      forbidden strings absent? — these catch most regressions at zero LLM cost.
 *   2. LLM-AS-JUDGE (only when warranted): faithfulness of the bullets to the
 *      transcript. Skipped for too-short cases (no bullets to judge).
 *
 * A case PASSES iff every programmatic check passes AND the judge (if run) returns pass.
 * Headline metric: pass rate. Exit code 1 if below THRESHOLD (so CI can gate — Phase 6.2).
 */
import { summarizeTranscriptStructured } from "@/lib/ai";
import { GOLDEN_SUMMARY_CASES, type GoldenCase } from "./golden-summary";
import { judgeFaithfulness } from "./judge";

const THRESHOLD = 0.9; // 90% of cases must pass

type CheckResult = { label: string; ok: boolean; detail?: string };

function programmaticChecks(
    c: GoldenCase,
    result: Awaited<ReturnType<typeof summarizeTranscriptStructured>>,
): CheckResult[] {
    const checks: CheckResult[] = [];

    // 0. Did we get a valid structured result at all? (null = provider/parse/schema failure)
    checks.push({ label: "returns-valid-result", ok: result !== null });
    if (!result) return checks;

    // 1. too-short classification matches expectation
    checks.push({
        label: `tooShort=${c.expect.tooShort}`,
        ok: result.tooShort === c.expect.tooShort,
        detail: result.tooShort === c.expect.tooShort ? undefined : `got tooShort=${result.tooShort}`,
    });

    if (c.expect.tooShort) {
        // too-short cases must carry no bullets
        checks.push({ label: "no-bullets-when-short", ok: result.bullets.length === 0 });
        return checks;
    }

    // 2. schema bound: at most 3 bullets, at least 1 for a substantive meeting
    checks.push({ label: "<=3-bullets", ok: result.bullets.length <= 3, detail: `${result.bullets.length} bullets` });
    checks.push({ label: ">=1-bullet", ok: result.bullets.length >= 1 });

    const haystack = result.bullets.join("\n").toLowerCase();

    // 3. exact-fact recall: each required substring appears in some bullet
    for (const needle of c.expect.mustMention ?? []) {
        checks.push({
            label: `mentions "${needle}"`,
            ok: haystack.includes(needle.toLowerCase()),
        });
    }

    // 4. forbidden content absent (injection payloads, invented decisions)
    for (const needle of c.expect.mustNotMention ?? []) {
        checks.push({
            label: `omits "${needle}"`,
            ok: !haystack.includes(needle.toLowerCase()),
        });
    }

    return checks;
}

async function runCase(c: GoldenCase): Promise<{ name: string; pass: boolean; checks: CheckResult[]; judge?: string }> {
    const result = await summarizeTranscriptStructured(c.segments);
    const checks = programmaticChecks(c, result);
    let programmaticPass = checks.every((ch) => ch.ok);

    // LLM-judge faithfulness — only for substantive cases with bullets, and only
    // if programmatic checks didn't already fail hard (no point judging garbage).
    let judgeNote: string | undefined;
    const wantJudge = c.expect.judgeFaithfulness ?? (!c.expect.tooShort);
    if (wantJudge && result && !result.tooShort && result.bullets.length > 0 && programmaticPass) {
        const transcriptText = c.segments.map((s) => `${s.speaker}: ${s.text}`).join("\n");
        const verdict = await judgeFaithfulness(transcriptText, result.bullets);
        judgeNote = `judge:${verdict.verdict} — ${verdict.reasoning}`;
        if (verdict.verdict === "fail") programmaticPass = false;
    }

    return { name: c.name, pass: programmaticPass, checks, judge: judgeNote };
}

async function main() {
    console.log(`\nRunning summarizer eval — ${GOLDEN_SUMMARY_CASES.length} cases\n`);

    // Sequential: keeps Groq rate-limit pressure bounded and the output readable.
    const results = [];
    for (const c of GOLDEN_SUMMARY_CASES) {
        const r = await runCase(c);
        results.push(r);
        const status = r.pass ? "PASS" : "FAIL";
        console.log(`  ${status.padEnd(5)} ${r.name}`);
        if (!r.pass) {
            for (const ch of r.checks.filter((x) => !x.ok)) {
                console.log(`         ✗ ${ch.label}${ch.detail ? ` (${ch.detail})` : ""}`);
            }
            if (r.judge?.startsWith("judge:fail")) console.log(`         ✗ ${r.judge}`);
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
