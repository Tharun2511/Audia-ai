# Phase 6 — Evals

## Session 6.1 — Eval theory: offline/online, metric families, golden sets, LLM-as-judge

**Built in Audia:**
- Refactored [src/lib/ai.ts](../src/lib/ai.ts): split `summarizeTranscriptStructured()` (returns validated `{ tooShort, bullets }` — the eval seam) from `summarizeTranscript()` (display-string wrapper, unchanged public contract)
- [src/evals/golden-summary.ts](../src/evals/golden-summary.ts) — 15 held-out cases (none reuse the prompt's few-shot examples), each tagged with the behavior it probes + `expect` criteria (`tooShort`, `mustMention`, `mustNotMention`)
- [src/evals/judge.ts](../src/evals/judge.ts) — faithfulness LLM-judge on **llama-3.3-70b-versatile** (different + stronger than the 8B under test → mitigates self-enhancement bias); pass/fail verdict, reasoning-before-verdict, temperature 0
- [src/evals/summary.eval.ts](../src/evals/summary.eval.ts) — runner: programmatic checks first (free/deterministic), judge only on substantive cases that pass programmatic; per-case table + pass rate; exits 1 below 90% threshold (CI gate for 6.2)
- `npm run eval` → `node --import tsx --conditions=react-server --env-file=.env src/evals/summary.eval.ts`; installed `tsx` + `server-only` devDeps. **First run: 15/15.**

### Concept summary

Evals are the unit tests of non-deterministic systems: there's no `===` for "is this summary good," so you swap exact-match for graded measurement, and the craft is choosing a grader you trust. Offline evals (fixed golden set, pre-merge/CI) catch regressions before ship; online evals (production traffic, feedback signals) catch what the offline set missed — and feed back into it as a flywheel. Four metric families, in order of reach: programmatic checks (schema/exact-match/regex — cheap, exact, always first), LLM-as-judge (semantic, modern default for open-ended output), reference-based BLEU/ROUGE (n-gram overlap — weak for LLM output because they score words not meaning), human eval (gold standard, used to validate the automated judge). Golden sets win on coverage over volume and must hold out the prompt's few-shot examples. LLM-as-judge needs a specific rubric, low-cardinality verdicts (pass/fail beats 1–10), reasoning before score, a different/stronger judge model, and one-time human validation (Cohen's kappa). For RAG, eval retrieval (context precision/recall) separately from generation (faithfulness/answer-relevancy) so a wrong answer is diagnosable. Pass rate is the headline regression number — but 100% on a fresh set usually means the set is too easy, not that the system is perfect.

### 5 most-likely interview questions

1. **Q: How do you evaluate an LLM feature when outputs are non-deterministic?**
   A: Graded measurement against a held-out golden set, not exact-match. Programmatic checks for anything structured (did it parse, pass schema, set the boolean right, include the required fact, omit the injection payload) — free and exact, run first. LLM-as-judge for the semantic parts that programmatic can't express (is this free-text answer faithful to the source). Track pass rate across commits as the regression signal, gate CI on a threshold. Offline pre-merge plus online feedback signals in production that feed failures back into the golden set.

2. **Q: Why not BLEU or ROUGE?**
   A: They score n-gram overlap with a reference, so they punish correct paraphrases — "March 15" and "the fifteenth of March" share no n-grams but mean the same thing. They were designed for machine translation where staying close to reference wording is the goal; for open-ended summarization or QA they don't track quality. I use programmatic checks for structured properties and an LLM-judge for semantic faithfulness instead. I'd only reach for ROUGE if I had gold reference summaries and wanted a cheap directional signal.

3. **Q: How can you trust an LLM to grade another LLM?**
   A: Treat the judge as a measurement instrument you calibrate before relying on. Specific rubric, not "rate 1–10" — a binary pass/fail against an explicit definition (e.g. "fail if any bullet states a fact not in the transcript"). A different, stronger model as judge (70B judging 8B) to avoid self-enhancement bias. Reasoning before verdict so the score is conditioned on an explanation. Then validate against ~20 human spot-checks once — if agreement (Cohen's kappa) is high, trust it in CI. Also mind position bias (run both orders in pairwise) and verbosity bias (rubric rewards conciseness).

4. **Q: How do you eval a RAG pipeline specifically?**
   A: Separate retrieval from generation, because a wrong answer has two root causes. Retrieval: context precision (of retrieved chunks, how many relevant) and context recall (of relevant chunks, how many retrieved). Generation: faithfulness (answer only claims what context supports) and answer relevancy (answer addresses the question). That's the RAGAS framing. If faithfulness is low the generation prompt is hallucinating; if context recall is low the retrieval is missing chunks — fix chunking/embedding/N. "The answer was wrong because the right chunk was never retrieved" is actionable; "the answer was wrong" is not.

5. **Q: Your eval suite is green at 100% — are you done?**
   A: No. A fresh golden set passing 100% usually means it isn't adversarial enough, not that the system is perfect. Next I harden the set with cases I expect to fail — long multi-number meetings, subtle injections, ambiguous too-short calls — and feed real production failures back in. Pass rate is only meaningful relative to a set that can actually fail. The goal isn't a green dashboard; it's a set that catches the regressions I care about.

### Gotchas

- **Testing on few-shot examples.** The model has seen them in the prompt — you're measuring memorization. Hold eval cases out. (Audia's golden set deliberately avoids the Alice/March-15 and PWNED few-shots.)
- **1–10 scoring.** LLMs can't reliably distinguish 7 from 8. Use pass/fail or 1–3.
- **Same model as judge and subject.** Self-enhancement bias inflates scores. Different family.
- **No rubric.** "Rate quality" is noise. Define each verdict by observable properties.
- **LLM-judging what a substring check answers.** Wasteful and less reliable. Programmatic first, judge only for semantics.
- **One blended score for RAG.** Can't diagnose. Split retrieval vs generation metrics.
- **Treating 100% as success.** Often the set is too easy. Harden until it can fail.
- **Running server-only code as a script.** `import "server-only"` won't resolve outside Next (it's bundler-provided, so `MODULE_NOT_FOUND` until you `npm i -D server-only`); needs `--conditions=react-server` to hit the no-op export, plus tsx for `@/*` paths and `--env-file=.env` for keys.

### Go-deeper resources

- Hamel Husain, [*"Your AI Product Needs Evals"*](https://hamel.dev/blog/posts/evals/) — the best practitioner essay on eval-driven development
- Zheng et al. 2023, [*"Judging LLM-as-a-Judge"*](https://arxiv.org/abs/2306.05685) — MT-Bench/Chatbot Arena; read the biases section
- [RAGAS docs](https://docs.ragas.io/en/stable/concepts/metrics/index.html) — the four RAG-eval metrics
- OpenAI [Evals](https://github.com/openai/evals) and [`promptfoo`](https://www.promptfoo.dev/) — production eval frameworks to know by name (we built primitives first)
