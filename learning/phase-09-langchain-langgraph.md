# Phase 9 — Agent orchestration: LangChain & LangGraph

> Inserted at Tharun's request after Phase 8. LangGraph.js, in Audia. The thread: **which framework for which use case**, and refactor the hand-rolled chat agent into a graph **as a parallel path** (keep the primitive). Original Phases 9–12 shifted to 10–13.

## Session 9.1 — LangChain fundamentals & the framework map

**Built in Audia:**
- Installed `@langchain/core` (^1.1.49) + `@langchain/groq` (^1.2.1).
- Exported `SUMMARY_SYSTEM_PROMPT` + `SummaryResponseSchema` from [ai.ts](../src/lib/ai.ts) (single source of truth → fair A/B).
- New [src/lib/summarize-lc.ts](../src/lib/summarize-lc.ts) — the summarizer re-expressed as an **LCEL chain**: `ChatPromptTemplate.pipe(model.withStructuredOutput(schema, {method:"jsonMode"}))`. Same model (`llama-3.1-8b-instant`), same prompt, same json-mode, same Zod contract as the primitive — only the wiring differs.
- New [src/evals/compare-lc.ts](../src/evals/compare-lc.ts) + `npm run eval:lc-compare` — runs primitive vs LCEL on 3 golden cases side by side. **Result: equivalent output** (budget-numbers + greeting-only identical; hiring-decision same shape, minor LLM wording). `npx tsc --noEmit` clean.

### Concept summary

LangChain is **not a monolith and not intelligence** — it's a toolkit of composable **components** (models, prompts, output parsers, retrievers, tools) + hundreds of **integrations**, wired with a uniform composition interface. Post-v1 (Oct 2025) it's split: `@langchain/core` (base abstractions + the `Runnable` interface), provider packages (`@langchain/groq`, …), `@langchain/langgraph` (orchestration), `@langchain/community` (long-tail integrations). **LCEL** = every component implements `Runnable` (`.invoke`/`.stream`/`.batch`) and `.pipe()` composes them so one's output feeds the next; the chain is itself a Runnable, so streaming/batching/retries/tracing come uniformly. LangChain's real value is **standardization** (swap providers in one line; plumbing for free), not capability — it can't do anything the primitives can't. The **framework map** (the decision rule): linear pipeline → LCEL; standard tool-using agent → `createAgent` (which runs *on* LangGraph); need to control the agent's flow (mid-run state, human-in-the-loop, conditional retries, multi-agent) → drop to LangGraph `StateGraph`; want full control / zero deps → primitives. `createAgent` and `StateGraph` aren't rivals — `createAgent` *is* a prebuilt StateGraph you drop below when you outgrow it; the old `AgentExecutor` is deprecated (maintenance until Dec 2026). The cost of all this: a dependency + indirection — proven live when the LCEL summarizer's call **bypassed our `logUsage` instrumentation** (the framework owns the call; you'd use LangSmith/callbacks instead).

### 5 most-likely interview questions

1. **Q: Is LangGraph a replacement for LangChain?**
   A: "No — they're layered, not rivals. Since v1 (Oct 2025) LangChain is the high-level component + integration layer (models, retrievers, tools, LCEL composition, the `createAgent` factory); LangGraph is the low-level orchestration runtime underneath — `StateGraph` with cycles, persistence, streaming, human-in-the-loop. `createAgent` actually runs on LangGraph. The thing LangGraph *replaced* is the old `AgentExecutor`, which is deprecated. So: LangChain for components, LangGraph for agent control flow, and they compose."

2. **Q: What is LCEL / a Runnable, and why does it matter?**
   A: "Every LangChain piece implements the `Runnable` interface — `.invoke`, `.stream`, `.batch` — and LCEL composes them with `.pipe()`: output of one feeds the next, and the whole chain is itself a Runnable. Why it matters: streaming, batching, retries, and tracing come uniformly across any chain without writing them per step. It's not that it can do something I can't by hand — it's that the plumbing is standardized. The trade-off is a layer of indirection between me and the raw model call."

3. **Q: When would you use LangChain vs LangGraph vs just writing it yourself?**
   A: "Linear pipeline — prompt to model to parser — use an LCEL chain. A standard tool-using agent — use `createAgent`. The moment I need to control the agent's flow — intercept state mid-run, add a human-approval step, conditional retries, multi-agent handoffs — I drop to a LangGraph `StateGraph`. And if I want zero dependencies and full control, or I'm learning, I write the primitive. The rule: reach down a layer only when the one above stops fitting."

4. **Q: What does LangChain actually buy you, and what does it cost?**
   A: "Buys: 600+ integrations (swap providers in a line), a uniform interface so streaming/batching/retries are free, LCEL composition, and the ecosystem — LangSmith tracing, prebuilt agents. Costs: a dependency with real breaking-change churn, indirection that makes 'why did it do that' harder to debug, and 'magic' defaults you didn't write. I saw the cost concretely — re-expressing our summarizer in LCEL, the call bypassed our custom `logUsage` because the library owns the request now; I'd have to wire LangSmith or a callback to get that observability back."

5. **Q: You re-expressed your summarizer in LCEL and got the same output. So what's the point?**
   A: "Exactly the point — same model, prompt, and json-mode, so the *result* is equivalent; LCEL only changed the wiring (`prompt | model.withStructuredOutput`). What it demonstrates is that the framework is a wiring choice, not a capability one. The value shows up at scale — uniform streaming/batching, provider swap, tracing — and the cost shows up immediately — the dependency and the lost custom instrumentation. For a single hand-tuned call the primitive is arguably clearer; the framework pays off when you have many chains or need the ecosystem."

### Gotchas

- **Curly braces in prompt templates.** `ChatPromptTemplate` treats `{x}` as a variable — a system prompt with literal JSON (`{"tooShort":false}`) breaks it. Use a fixed `SystemMessage` (no templating) or double the braces `{{ }}`.
- **`withStructuredOutput` method matters.** Default is `functionCalling`; we used `{method:"jsonMode"}` to mirror the primitive's `response_format:json_object` (and because 8B's tool-calling is weak — 7.2). Pick the method that matches the provider feature you intend.
- **The framework bypasses your custom instrumentation.** `logUsage` never fired for the LCEL call. Observability moves to LangSmith / callback handlers once LangChain owns the call.
- **Don't reach for the framework reflexively.** For one hand-tuned call, the primitive is clearer. LCEL earns its keep with many chains, provider-swap needs, or the ecosystem.

### Operational notes

- **Run the A/B:** `npm run eval:lc-compare` — primitive vs LCEL on 3 golden cases. Equivalent output confirms it's a wiring change.
- **Deferred to 9.2+:** LangGraph (`@langchain/langgraph`) — StateGraph core (9.2), refactor the chat agent (9.3), persistence/HITL/streaming (9.4).
- **Decision rule to memorize:** *Linear → LCEL · standard agent → `createAgent` · control the flow → `StateGraph` · full control/zero-deps → primitives.*
