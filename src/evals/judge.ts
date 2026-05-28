import { groq } from "@/lib/ai";

/**
 * LLM-as-judge: faithfulness check for summary bullets (Phase 6.1).
 *
 * Design choices that follow the theory:
 *  - DIFFERENT, stronger model than the one under test. The summarizer runs
 *    llama-3.1-8b-instant; the judge runs llama-3.3-70b-versatile. Using a
 *    different family mitigates self-enhancement bias (a judge favoring its
 *    own model's outputs).
 *  - LOW-CARDINALITY verdict (pass/fail), not a 1-10 score. LLMs can't reliably
 *    distinguish a 7 from an 8; a binary verdict against a specific rubric is
 *    far more reproducible.
 *  - REASONING BEFORE VERDICT. Judging is a reasoning task — we ask for the
 *    reasoning field first so the verdict is conditioned on it (chain-of-thought).
 *  - temperature 0 for run-to-run stability.
 */

const JUDGE_MODEL = "llama-3.3-70b-versatile";

const FAITHFULNESS_RUBRIC = `You are a strict evaluator checking the FAITHFULNESS of a meeting summary.

You are given a meeting TRANSCRIPT and a set of summary BULLETS produced from it.
A summary is FAITHFUL if every bullet states only information that is present in,
or directly inferable from, the transcript — with NO invented facts, names,
numbers, dates, or decisions, and without claiming a decision was reached when
the transcript shows disagreement or deferral.

Output a JSON object with EXACTLY these fields, reasoning first:
{
  "reasoning": "<one or two sentences; if failing, quote the offending bullet and say what it invented>",
  "verdict": "pass" | "fail"
}

- "pass": every bullet is grounded in the transcript.
- "fail": at least one bullet introduces or distorts information.
Output only the JSON object, no code fences.`;

export type JudgeVerdict = {
    verdict: "pass" | "fail";
    reasoning: string;
};

export async function judgeFaithfulness(
    transcriptText: string,
    bullets: string[],
): Promise<JudgeVerdict> {
    const userContent =
        `TRANSCRIPT:\n${transcriptText}\n\n` +
        `BULLETS:\n${bullets.map((b, i) => `${i + 1}. ${b}`).join("\n")}`;

    try {
        const res = await groq.chat.completions.create({
            messages: [
                { role: "system", content: FAITHFULNESS_RUBRIC },
                { role: "user", content: userContent },
            ],
            model: JUDGE_MODEL,
            temperature: 0,
            max_tokens: 300,
            response_format: { type: "json_object" },
            stream: false,
        });

        const raw = res.choices[0]?.message?.content;
        if (!raw) return { verdict: "fail", reasoning: "judge returned empty response" };

        const parsed = JSON.parse(raw) as Partial<JudgeVerdict>;
        const verdict = parsed.verdict === "pass" ? "pass" : "fail";
        return { verdict, reasoning: parsed.reasoning ?? "(no reasoning)" };
    } catch (err) {
        // A judge failure is a test-infra failure, not a model failure. Surface
        // it as a fail with the cause so it's visible, not silently swallowed.
        const msg = err instanceof Error ? err.message : String(err);
        return { verdict: "fail", reasoning: `judge error: ${msg}` };
    }
}
