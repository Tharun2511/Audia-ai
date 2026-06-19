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

---

## Session 9.2 — LangGraph core: StateGraph, nodes, edges, state & reducers

**Built in Audia:**
- Installed `@langchain/langgraph` (^1.4.2).
- New [src/lib/chat-graph.ts](../src/lib/chat-graph.ts) — Audia's chat agent as a `StateGraph` **skeleton**: `ChatState` via `Annotation.Root` (a `messages` channel with an **append reducer** + a `question` channel with default/overwrite); STUB `model` + `tools` nodes; `shouldContinue` conditional edge; back-edge `tools → model` (the cycle). Compiled, runnable. 9.3 swaps stubs for the real model + AUDIA_TOOLS.
- New [src/evals/chat-graph-demo.ts](../src/evals/chat-graph-demo.ts) + `npm run graph:demo`. **Ran clean:** node firings `model → tools → model`; final messages `human → ai(tool_calls=listMyMeetings) → tool → ai(answer)` — the conditional cycle + append reducer, visible. `tsc` clean.

### Concept summary

LangGraph is the runtime for what LCEL **can't** express — loops and branches. A **`StateGraph`** is nodes (units of work) + edges (control flow) + a typed **State** threaded through every node. A **node** is `(state) => Partial<State>`: it reads state, does work, and returns *only the keys it changes* — it never mutates state directly. An **edge** wires A→B unconditionally; a **conditional edge** runs a router on the state and picks the next node — this is how you get branching and the **cycle** (a back-edge `tools → model`). `START`/`END` are entry/exit. The genuinely new piece is **state channels + reducers**: each State key is a channel, and its **reducer** defines how a node's partial update merges — default is **overwrite**, but a `messages` channel uses an **append** reducer so returning `{ messages: [turn] }` concatenates (the declarative form of `messages.push()`; `MessagesAnnotation` is the prebuilt version, which also de-dupes by id). The cycle needs a bound — **`recursionLimit`** is LangGraph's `MAX_TOOL_ITERATIONS`. The 1:1 map to the hand-rolled loop: `messages[]`+`.push()` → messages channel + append reducer; `while(iter<MAX)` → back-edge + recursionLimit; `if(toolCalls) else break` → conditional edge; `dispatchTool` then push → the tools node returning `{messages:[...]}`. Why bother when the loop works: the flow is **declared not buried** (readable/inspectable), you can **stream per-node**, and — the real payoff (9.4) — you can **checkpoint the State and interrupt at a node** (persistence + human-in-the-loop), which a raw `while`-loop can't give you cleanly.

### 5 most-likely interview questions

1. **Q: What's a StateGraph, in one breath?**
   A: "Nodes + edges + a shared typed state. Nodes are functions that read the state and return a partial update; edges declare what runs next; a conditional edge branches on the state, which is how you get loops. You compile the graph and invoke or stream it. It's a state machine for an agent — the control flow is declared as a graph instead of buried in an imperative loop."

2. **Q: What's a reducer and why does it matter?**
   A: "Each key in the state is a channel, and its reducer defines how a node's partial update merges into it. The default is overwrite — last write wins. For a message list you want append, so the reducer concatenates: a node returns `{ messages: [newTurn] }` and the reducer does the push. That's the formalization of 'how does this turn get added to history' — in my hand-rolled loop I called `messages.push()` manually; the reducer makes it declarative and automatic every iteration."

3. **Q: How does a graph express a loop? Isn't a graph acyclic?**
   A: "Not in LangGraph — it allows cycles deliberately. The loop is a back-edge: my model node has a conditional edge that routes to a tools node when the model emits tool calls, and the tools node has an edge straight back to the model. That `tools → model` back-edge is the cycle — it's exactly my `while (tool_calls) { execute; loop }`. The `recursionLimit` bounds it, same role as `MAX_TOOL_ITERATIONS`, so a model that never stops calling tools throws instead of spinning forever."

4. **Q: You already had a working loop. Why rewrite it as a graph?**
   A: "Three reasons, and the third is the real one. First, the control flow is declared as a graph instead of tangled in imperative branches — readable and inspectable. Second, I can stream per-node and visualize it. Third — and this is what a raw loop can't give cleanly — the state is a first-class object the runtime owns, so I can checkpoint it per conversation thread (persistence/memory for free) and interrupt the graph mid-run for human approval, then resume. That's Phase 9.4. For a simple loop the graph is overhead; it earns its keep when you need those runtime features."

5. **Q: A node returns `{ messages: [x] }` — does that replace the whole state?**
   A: "No. A node returns only the channels it wants to change, and each channel's reducer merges the update. For `messages` with an append reducer, returning `{ messages: [x] }` appends x; the `question` channel is untouched. That partial-update-plus-reducer model is why nodes compose cleanly — they don't need to know or reconstruct the full state, just declare their delta."

### Gotchas

- **Default reducer overwrites.** Forget to set an append reducer on `messages` and each node *replaces* the history instead of extending it. Use `MessagesAnnotation` or an explicit `concat` reducer.
- **Nodes return deltas, not full state.** Return only changed channels; the reducer merges. Returning the whole state (or mutating `state` in place) fights the model.
- **Cycles need `recursionLimit`.** A back-edge with no bound + a model that keeps calling tools = infinite loop. Set the limit (= your MAX_TOOL_ITERATIONS).
- **Conditional-edge return values must map to real targets.** `shouldContinue` returns `"tools"`/`"end"`; the `pathMap` `{tools:"tools", end:END}` maps those to nodes/END. A typo routes nowhere.

### Operational notes

- **Run it:** `npm run graph:demo` — prints node firings (`model→tools→model`) + the accumulated messages. Stubbed, no API key.
- **9.3 turns the skeleton real:** model node → llama-3.3-70b with `AUDIA_TOOLS`; tools node → `dispatchTool`; wire to `/api/chat-graph` as a parallel route; A/B vs the primitive loop.
- **`MessagesAnnotation`** is the prebuilt messages channel (append + de-dupe by id). We spelled the reducer out by hand for visibility; 9.3 may switch to the prebuilt.

---

## Session 9.3 — The real agentic graph (refactor the chat agent)

**Built in Audia:**
- [src/lib/agent-tools-lc.ts](../src/lib/agent-tools-lc.ts) — `buildAudiaTools(userEmail)`: the **tool-format bridge**. Wraps `AUDIA_TOOLS` (JSON-schema specs) as runnable LangChain `tool()` objects whose execute fn calls the same `dispatchTool` — reuses descriptions verbatim (single-source routing prompt) + ownership scoping (closure over `userEmail`).
- [src/lib/chat-graph-agent.ts](../src/lib/chat-graph-agent.ts) — the chat agent **two ways**: `buildChatGraph()` (explicit `StateGraph` over `MessagesAnnotation`: model node `ChatGroq.bindTools(tools)` + prebuilt `ToolNode` + conditional cycle) and `buildReactAgent()` (`createReactAgent({llm,tools,prompt})`, ~3 lines). Both per-request, same model (`llama-3.3-70b-versatile`) + tools.
- [src/app/api/chat-graph/route.ts](../src/app/api/chat-graph/route.ts) — parallel POST route (`{question, mode}`), A/B surface vs the hand-rolled `/api/chat`. Non-streaming JSON; the primitive loop stays untouched.
- [src/evals/chat-graph-agent-demo.ts](../src/evals/chat-graph-agent-demo.ts) + `npm run graph:agent-demo` — live smoke test (real model, STUB tools). **Ran clean: both builds fired `listMyMeetings` and returned identical answers.** `tsc` clean.

### Concept summary

Turning 9.2's stub skeleton real is three swaps: the model node becomes **`ChatGroq(...).bindTools(tools)`** (LangChain's wrapper over the `tools:` array we hand-passed to Groq — now the model really emits `tool_calls`); the tools node becomes a prebuilt **`ToolNode(tools)`** (reads the last AI message's `tool_calls`, executes the matching tools, returns `ToolMessage`s — the graph form of the `for (tc of toolCalls) dispatchTool(...)` block); and the messages channel uses **`MessagesAnnotation`** (the prebuilt append+dedupe reducer). The one real wrinkle is the **tool-format bridge**: `AUDIA_TOOLS` are OpenAI/Groq JSON-schema *specs* (data), but `ToolNode`/`bindTools` need runnable LangChain `tool()` objects, so we wrap each as a `tool()` whose execute calls `dispatchTool` — reusing implementation + ownership, restating only the small arg schema in Zod, and reusing the description verbatim so routing is identical. Tools need `userEmail`, so the graph is **built per-request** (closure). The headline lesson, shown live: **`createReactAgent` and the explicit `StateGraph` produced byte-identical answers** — because `createReactAgent` *is* a prebuilt StateGraph (model + ToolNode + conditional cycle); you drop to the explicit graph only when you need to customize the flow (9.4: checkpoints, interrupts). One operational note from the run: the 70B **called `listMyMeetings` twice** before answering (benign over-calling, bounded by `recursionLimit=8`) — the same repeated-call pattern flagged in 7.2, which a fingerprint check would suppress.

### 5 most-likely interview questions

1. **Q: Your tools were OpenAI-format JSON specs; LangGraph needs runnable tools. How'd you bridge that?**
   A: "I wrapped each registry tool as a LangChain `tool()` whose execute function calls my existing `dispatchTool` — so the implementation and ownership scoping are reused, I just satisfy LangChain's runnable interface. I reuse the description verbatim (it's the routing prompt — single source of truth, so the graph routes identically), and restate only the small arg schema in Zod since LangChain wants Zod and my registry holds JSON schema. The tools are built per-request in a closure over the authed user's email so every call stays ownership-scoped."

2. **Q: `createReactAgent` vs building the StateGraph yourself — when each?**
   A: "`createReactAgent` is a prebuilt StateGraph — model node, ToolNode, conditional cycle, done in one call. I proved it: my explicit graph and `createReactAgent` returned identical answers on the same question. Use the prebuilt for a standard tool-using agent. Drop to the explicit StateGraph the moment you need to customize the flow — intercept or transform state between nodes, add a human-approval interrupt, conditional retries, multi-agent routing. The explicit graph is the same thing with the lid off."

3. **Q: What does `bindTools` actually do?**
   A: "It attaches the tool schemas to the chat model so the model can emit `tool_calls` — it's LangChain's wrapper over the `tools:` parameter I used to hand-pass to `groq.chat.completions.create`. After `bindTools`, calling `model.invoke(messages)` may return an AIMessage with `tool_calls` populated, which my conditional edge then routes to the ToolNode."

4. **Q: How is the loop bounded, and did you observe it mattering?**
   A: "`recursionLimit` on invoke — the graph form of MAX_TOOL_ITERATIONS; it throws if the graph takes too many node-steps. It mattered in the demo: the model called `listMyMeetings` twice before answering — benign over-calling, but without a bound a model that keeps re-calling would spin. The limit caught it; a sharper fix is fingerprinting repeated identical calls and short-circuiting, which I'd noted as a loose end earlier."

5. **Q: Why couldn't you run the DB-backed agent in your tsx demo?**
   A: "TypeORM relies on `emitDecoratorMetadata` to infer column types from the entity decorators, and tsx/esbuild doesn't emit that metadata — so importing the entities throws `ColumnTypeUndefinedError`. It's a toolchain limitation, not an agent bug: the DB-backed code runs fine under Next (whose compiler emits the metadata), which is why the real agent lives in the route and my CLI smoke test uses stub tools to exercise just the LangGraph + Groq mechanics."

### Gotchas

- **Tool-format mismatch.** OpenAI/Groq JSON specs ≠ runnable LangChain tools. `ToolNode`/`bindTools` need `tool()` objects. Wrap, don't pass the spec.
- **Ownership scoping.** Tools need `userEmail` — build them per-request in a closure; don't hoist a global tool set that ignores the caller.
- **tsx can't run TypeORM entities.** No `emitDecoratorMetadata` under esbuild → `ColumnTypeUndefinedError`. DB-backed agent code is verified via the Next route, not the tsx evals.
- **`recursionLimit` is mandatory for a cyclic agent.** It's MAX_TOOL_ITERATIONS; over-calling models will hit it.
- **`createReactAgent` system-prompt param is `prompt`** (string/SystemMessage/fn) in current LangGraph.js — older `messageModifier`/`stateModifier` are renamed/deprecated.

### Operational notes

- **Run the live mechanics:** `npm run graph:agent-demo` (real model, stub tools — no DB). Both builds fire the tool + answer identically.
- **A/B the real thing:** POST `/api/chat-graph` (`{question, mode:"explicit"|"react"}`) vs `/api/chat` with a meeting-entity question, under `next dev`. Same tool fires.
- **Scope:** agentic loop only — no RAG/memory/streaming in the graph route (orthogonal). A retrieve node would slot upstream of the model node. Streaming + persistence are 9.4.
- **The primitive `/api/chat` is untouched** — "built it by hand AND as a graph," A/B-able, interview-ready.
