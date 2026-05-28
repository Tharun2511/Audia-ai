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

---

## Session 6.2 — Building an eval harness: CI gates, RAG eval, the flywheel

**Built in Audia:**
- Shared prompt seam [src/lib/rag-prompt.ts](../src/lib/rag-prompt.ts) — extracted `CHAT_SYSTEM_PROMPT` + `buildContextBlock` + `wrapUserMessage` from `chat/route.ts` so eval and product can't drift on prompt text. Updated [chat/route.ts](../src/app/api/chat/route.ts) to import from it.
- [src/evals/golden-chat.ts](../src/evals/golden-chat.ts) — 10 RAG cases: single-chunk recall, multi-chunk synthesis, refusal-out-of-scope, injection-in-chunk, disagreement-surface-both, exact-number recall, irrelevant-chunks refusal, speaker attribution, action-item synthesis, role-change injection.
- Extended [src/evals/judge.ts](../src/evals/judge.ts) with `judgeRAGFaithfulness` (chunks-aware faithfulness) and `judgeAnswerRelevancy` (does the answer address the question). Refactored to shared `callJudge()` helper.
- [src/evals/chat.eval.ts](../src/evals/chat.eval.ts) — three layers: programmatic (substring + citation count + refusal heuristic), faithfulness judge, relevancy judge. Skips judges on refusal cases. Exits 1 below 90%.
- `npm run eval:chat`, `npm run eval:all`.
- [.github/workflows/eval.yml](../../.github/workflows/eval.yml) — CI gate at 90%: runs both suites on every PR, requires `secrets.GROQ_API_KEY`, skips on doc-only changes, concurrency-cancels superseded runs.
- **Live flywheel demo:** first chat eval run was 6/10. Diagnosed → 2 real prompt bugs (didn't preserve exact numbers; invented decisions on deferral) + 2 eval defects (refusal heuristic incomplete; "arr" was a false-positive magnet substring). Fixed all four. **Re-ran: 10/10.** The two new prompt rules (verbatim-preservation + pending-decision handling) are real product improvements, not eval-paper.

### Concept summary

Phase 6.1 built the first harness; 6.2 makes evals load-bearing in three ways. One, the eval flywheel — every production failure becomes a new golden-set case before being fixed, so the set grows with real failure patterns instead of imagined ones. Two, CI integration — `npm run eval:all` runs on every PR with a hard gate at 90% (threshold set to baseline minus 5–10 points for judge noise tolerance), blocking merge when AI quality regresses. Three, RAG-specific eval depth — the four RAGAS metrics, separating retrieval (context precision/recall) from generation (faithfulness/answer relevancy) so a wrong answer is diagnosable. Audia 6.2 implements the generation half (faithfulness + relevancy) against fabricated chunks; full retrieval eval needs DB seeding and is deferred to 6.3+. The key engineering decision is extracting `CHAT_SYSTEM_PROMPT` into a shared module — eval and product MUST share the prompt or the eval can pass on a stale copy. Pairwise judging beats pointwise for comparing prompt v1 vs v2; judge validation against ~20 human spot-checks (measured by Cohen's kappa) is what makes the judge itself trustworthy. Scaling patterns at high volume — PR fast subset + nightly full suite + opt-in `[full-eval]` keyword — exist; Audia doesn't need them yet at Groq's free tier.

### 5 most-likely interview questions

1. **Q: Walk me through how you build an eval harness from scratch.**
   A: "Start by reading the model's outputs — 50–100 real examples before defining any metric. Categorize failures into buckets (invention, off-topic, formatting breaks, injection susceptibility). Build one metric per bucket. Construct a held-out golden set of 15–30 cases each probing one bucket, including adversarial cases. Run programmatic checks first — schema, substring, citation count — free, exact, fail fast. Layer an LLM-judge on top only for what programmatic can't express. Use a different, stronger judge model than the system under test for self-enhancement-bias avoidance. Wire `npm run eval` for local iteration and a CI step that gates merge below a threshold. Then start the flywheel: every production failure becomes a new case."

2. **Q: How do you handle CI eval gates without flaky failures?**
   A: "Threshold set to baseline minus 5–10 points to tolerate judge noise — if my eval baseline is 100%, I gate at 90%. Tighter than that and judge variance fails honest PRs; looser and the gate stops catching real regressions. I also skip the eval workflow on doc-only PRs via `paths-ignore`, cancel superseded runs via `concurrency` so only the latest commit's eval matters, and post the score-vs-baseline as a PR comment for soft signal even when the hard gate passes. When the gate fails, the right move is to investigate the failing cases — sometimes they're real regressions, sometimes the judge is being flaky on the same case and the rubric needs hardening."

3. **Q: Why split RAG eval into context-precision/recall and faithfulness/relevancy?**
   A: "Different root causes. If faithfulness drops, generation is hallucinating — fix the prompt or grounding rules. If answer-relevancy drops, generation is answering the wrong question — fix the prompt. If context-precision drops, retrieval is letting noise through — tune the re-ranker (MMR λ, k). If context-recall drops, the right chunks never reach the re-ranker — fix coarse N, chunking, or the embedding model. Without that separation, 'the answer was wrong' tells you nothing; with it, you walk the diagnostic chain backward to a specific knob. Audia 6.2 implements the generation half (faithfulness, relevancy) — those work against fabricated chunks. Retrieval eval needs a seeded pgvector with ground-truth chunk-relevance labels per question; different infrastructure, separate phase."

4. **Q: Your eval is at 100%. Should you trust the harness?**
   A: "Not yet. Three things have to be true to trust it. One: the golden set must be hard enough to fail — 100% on a fresh set usually means the set is too easy, not that the system is perfect. Validate by adding adversarial cases until something fails, then fix it. Two: the judge must be validated against human judgment — spot-check ~20 verdicts by hand and compute Cohen's kappa; above 0.7 is substantial agreement. Until that's measured, every 'pass' is unverified. Three: the flywheel must be running — every production failure feeds back into the set. A static eval becomes a coverage-fiction over time."

5. **Q: How would you scale this to a 1000-case suite where every PR is expensive?**
   A: "Three layers, cheap-to-expensive. PR-level: run a curated 10–20 case fast subset that catches the most common regressions; gate on it; full suite is opt-in via a `[full-eval]` keyword in the PR title for risky changes. Nightly: run the full suite, post a Slack alert if pass rate drops by more than X. Production: sample 1% of real responses through the LLM-judge, trend the score over time — that's online eval, complementary to offline. Pattern is 'cheap on every PR, expensive when it matters.' Audia doesn't need any of this today — we're on Groq's free tier at ~25 calls per run, effectively $0 — but at gpt-4o judge prices and 1000 cases this matters."

### Gotchas (additions to 6.1's list)

- **Eval and product prompt drift.** Putting `CHAT_SYSTEM_PROMPT` in two files lets evals pass against a stale copy. Extract to a shared module (Audia: `src/lib/rag-prompt.ts`).
- **Substring assertions that are too broad.** `"arr"` matches "carry," "narrative," "guarantee." Use specific markers, word-boundary regex, or whole-word matching.
- **Skipping judge validation.** A judge that's never been audited against humans is an uncalibrated instrument. Every "pass" is unverified.
- **Running the same eval on every push, not every PR.** Push-scoped runs burn budget on intermediate commits; use `on: pull_request` for the "block bad merges" goal.
- **Threshold too tight.** 95%+ means judge noise fails PRs unfairly; teams turn off the gate. Threshold ≈ baseline minus 5–10 points.
- **No flywheel — static golden set.** After ~3 months it stops mapping to reality. Discipline: every prod failure → new case before fixing the bug.

### Operational notes

- **GitHub Actions setup needed:** add `GROQ_API_KEY` as a repository secret (Settings → Secrets and variables → Actions → New repository secret). After the first run completes, add the workflow as a Required Status Check (Settings → Branches → Branch protection rules → main) for the gate to actually block merges.
- **Cost on Groq free tier:** ~25 calls per `eval:all` run = effectively $0. Daily limit (100k tokens for 70B) accommodates ~9 full runs/day; 30 PRs/week trivial.

### Go-deeper resources

- Hamel Husain, [*"A Field Guide to Rapidly Improving AI Products"*](https://hamel.dev/blog/posts/field-guide/) — eval flywheel + "look at your data" rule
- Eugene Yan, [*"Patterns for Building LLM-based Systems & Products — Evals"*](https://eugeneyan.com/writing/llm-patterns/#evals-to-measure-performance) — the eval decision tree
- [RAGAS docs](https://docs.ragas.io/en/stable/concepts/metrics/index.html) — the four metrics, when each fires
- [GitHub Actions docs on required status checks](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches#require-status-checks-before-merging) — the wire-up for making the gate actually block
