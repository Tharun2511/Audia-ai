# Phase 7 — Tool use & agents

## Session 7.1 — Function-calling: schemas, dispatch, the agentic loop

**Built in Audia:**
- New [src/lib/tools.ts](../src/lib/tools.ts) — `AUDIA_TOOLS` registry + `dispatchTool(name, rawArgs, ctx)`. First tool: `listMyMeetings(limit?, since?)` returns the user's recent meetings (id, title, date, duration). Schema + dispatcher + implementation co-located so they can't drift. Unknown names + bad-JSON args + exceptions all return structured `{ error: ... }` so the model self-corrects.
- [src/app/api/chat/route.ts](../src/app/api/chat/route.ts) — replaced the single streaming call with an **agentic loop**, MAX_ITERATIONS=3, last iteration uses `tool_choice="none"` to force a final text answer. The streaming controller forwards `delta.content` to the client mid-iteration; `delta.tool_calls` deltas merge by `index` into an accumulator until the stream closes.
- [src/lib/rag-prompt.ts](../src/lib/rag-prompt.ts) — added a `TOOLS:` section to `CHAT_SYSTEM_PROMPT` with the routing rule: call tools only when `<context>` can't answer the question; never call for content questions about the current meeting.
- **Eval safety:** ran existing suites post-change — summarizer 15/15, RAG chat 10/10 still hold. The tool-availability hint in the system prompt didn't disturb non-tool behavior.

### Concept summary

Through Phase 6, Audia's chat had one fixed shape: question in → server retrieves chunks → model answers. The model is a *language* engine — it cannot look up data the server didn't pre-fetch or take actions. Function-calling is the protocol that lets the model ask for those actions through a structured channel. The protocol is three messages, two LLM calls: user-with-tools → model-emits-tool_calls → app-executes → tool-result-back → model-finishes-with-text. Tool schemas declare name + description + JSON-Schema parameters; the **description IS the prompt the model reads at decision time** — vague descriptions cause under-calling or over-calling; explicit positive AND negative examples are the lever. `tool_choice` controls behavior: `"auto"` (default), `"required"`, `{name:"X"}`, or `"none"` (used on the LAST iteration of an agentic loop to force a text answer). The agentic loop is just function-calling repeated: while the model returns tool_calls, execute and append; when it returns content, you're done. **LangChain agents, the Assistants API, and Claude's computer use are all built on this exact six-line loop** — the hard parts are tool design, observability, and exit conditions, not the loop. Cost: 2 LLM calls per tool round, so the loop is quadratic-ish in tool usage; MAX_ITERATIONS bounds it. Streaming and tool_calls coexist — `delta.content` and `delta.tool_calls` arrive on independent channels in the same stream; the application forwards content to the client AND accumulates tool_calls (merged by `index`) until the stream closes.

### 5 most-likely interview questions

1. **Q: Walk me through how function calling works at the API level.**
   A: "Three messages, two LLM calls. First call: client sends the user message plus a `tools` array of schemas — each schema is `{name, description, JSON-Schema parameters}`. The model either returns text content (we're done) or emits a structured `tool_calls` field with `{id, name, arguments}`. If it called tools, the app executes them, then sends a second call with the message history plus the assistant's tool_calls turn AND a `role:'tool'` message per result, with `tool_call_id` referencing the original call's id. The model uses those results to compose a final text answer. The id linking is critical — without it the second call has no anchor for which result goes with which call."

2. **Q: How does the model decide whether to call a tool?**
   A: "Next-token sampling. The model's vocabulary includes a tool-call token; at each generation step it picks the highest-likelihood next token — sometimes text, sometimes the tool-call sequence. The decision is driven by the tool description (which IS a prompt the model reads at inference time), the system prompt (whether it instructs to use tools), and the recent conversation as in-context learning. That's why description quality is the single biggest lever — vague descriptions cause under-calling on cases that should fire, and over-calling on greetings. Production teams iterate on descriptions with explicit positive AND negative examples ('Use this for X. Do NOT use this for Y.')."

3. **Q: What's the difference between function calling and an agent?**
   A: "An agent is function-calling in a while-loop. Single-step function calling is one tool round: call → execute → answer. The agentic loop is the same protocol repeated until the model returns content with no tool_calls, capped by MAX_ITERATIONS. The 'agent' is just the loop plus sensible exit conditions; LangChain agents, the OpenAI Assistants API, Claude's computer use — all built on this exact pattern. The interesting engineering is in tool design (sharp descriptions, narrow parameter surface), observability (every tool call logged), and exit conditions (max iterations, repeated-failure detection, `tool_choice='none'` on the last iteration to force a final answer). The loop itself is six lines."

4. **Q: What's the most common failure mode of tool use in production?**
   A: "Routing failures — the model calls when it shouldn't or doesn't call when it should. Symptoms: tool fires on 'hello' (over-calling) or never fires on a question that needed it (under-calling). Cause: vague tool descriptions. Fix: rewrite the description with explicit positive AND negative examples — 'Use this when X. Do NOT use this for Y.' The description is the routing prompt, not metadata. After that, bad arguments (model passing wrong types or hallucinated IDs) and hallucinated tool names (model invents a tool name) are the next two. Both have the same fix shape: validate at the dispatch layer and return structured `{ error: ... }` so the model can self-correct on the next iteration. Errors as data, not crashes."

5. **Q: How do streaming and tool calls coexist?**
   A: "Provider APIs stream `delta.content` and `delta.tool_calls` on independent channels in the same stream. The application forwards `delta.content` to the client immediately — that's the user-facing streaming UX. It accumulates `delta.tool_calls` server-side: tool_calls arrive as fragments keyed by `index`, with id and name typically in the first chunk and arguments arriving in pieces; merge by index into an accumulator. When the upstream stream closes, if accumulators are non-empty, you have tool calls to execute — dispatch them, append results, start the next iteration. The user sees text from each iteration stream live; tool execution happens in the gap between iterations, which feels like the assistant 'thinking out loud' then continuing. Same Phase-2 streaming primitives extended for tool dispatch."

### Gotchas

- **Vague tool descriptions** are the #1 production failure cause. Lead with the capability, but ALWAYS include the negative case ("Do NOT call for greetings or content questions").
- **No MAX_ITERATIONS** = runaway loops that burn cost AND context budget. Always cap.
- **No `tool_choice="none"` on the last iteration** = model can keep asking for tools at the cap, never producing a final answer. Always force text on the boundary.
- **Throwing instead of returning structured errors** = the loop crashes when it should self-correct. Wrap dispatchers in try/catch and return `{ error: "human-readable" }`.
- **Forgetting to append the assistant's tool_calls turn to message history** = the model sees tool results with no anchor. The next-iteration messages array MUST include both the assistant's tool_calls turn AND the role:"tool" results in the right order.
- **Streaming `delta.tool_calls` to the client** = leaks internal protocol to the user. Accumulate server-side; only forward `delta.content`.
- **One mega-tool with 15 parameters** = the model can't fill it correctly. Prefer multiple narrow tools (1-3 parameters each).
- **No observability** = "why did the agent do that?" becomes unanswerable. Log every tool call: name, args, result, duration, iteration index.

### Operational notes

- **MAX_ITERATIONS=3 for Audia today.** Workload-specific; research agents need 10-20, RAG-style chat needs 2-3, structured extraction is often 1.
- **Cost on Groq free tier:** ~2-3 LLM calls per tool-using turn, ~$0 (still in the free band). At gpt-4o prices, a 3-iteration agentic turn is ~$0.03 — line-item-relevant at any scale.
- **Try it manually** (no automated eval yet for tool routing):
  - "What meetings did I have this week?" → should call `listMyMeetings({since:'<ISO date>'})`
  - "Summarize this meeting" (inside a session) → should NOT call any tool, should answer from `<context>` chunks
  - "Hi" → should NOT call any tool
  - Server log shows `[chat] iter=0 tool_calls=listMyMeetings({"since":"..."})` when tool fires

### Go-deeper resources

- [OpenAI guide, *"Function calling"*](https://platform.openai.com/docs/guides/function-calling) — the canonical primer; Groq's API is OpenAI-compatible so this is literally the spec
- [Anthropic, *"Tool use with Claude"*](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — the "tools are prompts in disguise" framing is the most important part
- Hamel Husain, [*"Building & evaluating tool-using LLMs"*](https://hamel.dev/blog/posts/tool-use/) — what goes wrong in practice and how to measure routing

---

## Session 7.2 — Multi-step agents & ReAct: tool-result memory, failure modes, planning

**Built in Audia:**
- **`ChatMessage` entity extended** — `role` adds `"tool"`, plus `toolCallId` (nullable varchar) and `toolCalls` (nullable simple-json) columns. TypeORM synchronize adds via standard ALTER TABLE.
- **[chat-memory.ts](../src/lib/chat-memory.ts) rewritten** — `HistoryMessage` is now a union of API-shaped variants (user / assistant text / assistant-with-tool_calls / role:tool). `loadRecentTurns` returns them chronologically with **orphan-pair filtering** at the window edge (collects valid tool_call_ids from in-window assistants, drops role:tool messages without an anchor). `saveTurn` accepts new optional `toolCallId` + `toolCalls` fields.
- **[tools.ts](../src/lib/tools.ts)** — second tool `getMeetingDetails(transcriptionId)` returns title, createdAt, duration, distinct speakers + count, cached summary. Ownership filter (`userEmail`) prevents cross-user leaks even with a leaked id. Composes with `listMyMeetings` for the canonical "list → drill" pattern.
- **[chat/route.ts](../src/app/api/chat/route.ts)** — persists assistant-with-tool_calls turn + each role:tool result inside the loop, in order (assistant first, then tool results — chronological reload reconstructs the pair correctly). `MAX_TOOL_ITERATIONS` bumped 3 → 4 to accommodate genuine two-step workflows + one self-correction slot. History from `loadRecentTurns` splats directly into the messages array — no remapping needed.
- **Eval calibration** (small, surgical): dropped `mustCiteAtLeast` on `action-item-owner` (case probes attribution, citation is tested elsewhere); faithfulness rubric now explicitly classifies source metadata (speaker labels, timestamps) as part of the source. **Both suites pass: 15/15 + 10/10.**

### Concept summary

Multi-step agents are function calling in a loop with persistent tool-result memory. The loop is six lines; the architectural difference from single-step is twofold: (1) ≥2 tools so genuine composition is possible, and (2) persisting the assistant's tool_calls turn AND each role:tool result in conversation memory so subsequent turns can reference fetched data without re-calling. Without that persistence, follow-ups have amnesia about prior tool output. ReAct (Yao et al. 2023) is the original pattern — Thought / Action / Observation interleaved — and modern function-calling APIs encode it implicitly: `delta.content` = thought, `tool_calls` = action, `role:"tool"` = observation. The four canonical multi-step failure modes are cascading errors (bad data compounds), stuck loops (model emits identical calls), hallucinated state (model "remembers" things not actually in context), and off-task drift (new in multi-step — chains of reasoning steps wander from the user's goal). Guards: structured `{error}` returns, repeated-call detection, MAX_ITERATIONS with `tool_choice="none"` on the last iteration, restating the user's goal each iteration. Reactive agents (default for Audia) decide one step at a time; planning agents emit a sequence first. Reach for planning only when reactive drifts on 5+ step queries — the planning step costs an upfront LLM call. Audia's MAX = expected_steps + 1, so 4 for two-tool chat. Order preservation matters: the API requires role:tool to immediately follow its anchor assistant turn; save assistant first (lower createdAt), then each tool result.

### 5 most-likely interview questions

1. **Q: Walk me through your multi-step agent architecture.**
   A: "Function calling in a loop with persistent tool-result memory. The loop iterates: model returns text (done) or tool_calls (execute, append results, call model again). Up to MAX_ITERATIONS — Audia uses 4 for the two-tool chat workload (expected_steps=2, plus one self-correction slot). Last iteration uses `tool_choice='none'` to force text. Tool results and the assistant's tool_calls turn both persist to ChatMessage so follow-ups have working memory of what was fetched. Without that persistence, every follow-up triggers a redundant re-call. LangChain agents and Anthropic's tool use are built on this same six-line pattern — the engineering work is exit conditions, tool descriptions, and observability, not the loop."

2. **Q: ReAct — what is it and is it still relevant?**
   A: "Yao et al. 2023 — the prompting pattern that interleaved Thought, Action, and Observation per iteration. It taught us that exposing reasoning helps the model self-correct. Modern function-calling APIs encode ReAct implicitly: `delta.content` carries any thoughts the model wants to surface, `tool_calls` carries actions, `role:'tool'` messages carry observations. The loop iterates them automatically. You don't write 'Thought:' tags anymore, but the underlying structure IS ReAct. If someone asks 'is ReAct still used' — yes, you just don't see the prose because the protocol does the same work."

3. **Q: What's the difference between reactive and planning agents?**
   A: "Reactive agents decide one step at a time from current context — the loop iterates ReAct. Planning agents emit a sequence-of-tool-calls plan FIRST, then execute it (may replan mid-execution). Reactive is the default — cheaper (no upfront planning call), simpler code, robust on short tasks. Planning becomes worth the extra LLM call on long structured tasks where reactive starts drifting off-task — typically 5+ step queries. Production systems often nest: short reactive loops inside an overall plan. Audia is pure reactive because the workload is 1–3 step chat queries; I'd add planning only if I observed measurable off-task drift."

4. **Q: How do you persist tool results across turns?**
   A: "Extend the conversation-memory table with two new columns: `toolCallId` (nullable, populated on `role='tool'` rows) and `toolCalls` (nullable JSON, populated on assistant rows that emitted calls). Each iteration of the loop saves the assistant-with-tool_calls turn FIRST (lower createdAt), then each tool result (higher createdAt) — order matters because the API requires `role:tool` messages to immediately follow their anchor assistant. On load, the rolling buffer returns these chronologically, with **orphan-pair filtering**: collect all valid tool_call_ids from in-window assistant turns, then drop any `role:tool` whose id isn't backed (handles the case where the LIMIT cuts mid-pair at the window boundary). Without that filtering, a broken pair would cause the API to reject the next call."

5. **Q: What are the multi-step-specific failure modes you guard against?**
   A: "Four canonical ones. **Cascading errors** — bad data on iteration 1 becomes args for iteration 2; guard with structured `{error}` returns so the model self-corrects. **Stuck loops** — model emits identical (name + args) twice; guard with duplicate detection that returns 'you just called this; try something else.' **Hallucinated state** — model claims it 'remembers' tool data that's actually rolled out of the window; guard with tool-result persistence + a system rule 'never assert info that didn't come from a tool result this conversation.' **Off-task drift** — new in multi-step, chains of reasoning wander from the user's goal; guard with restating the user's question each iteration. All four compound with cost: every iteration is 2 LLM calls, so bound MAX_ITERATIONS aggressively and instrument cost per task."

### Gotchas (additions to 7.1's list)

- **Persisting role:tool without the matching assistant turn.** API rejects. Always save assistant-with-tool_calls FIRST, then each tool result. Order is enforced via `createdAt` ascending on the next load.
- **Window-edge orphan pairs.** Rolling buffer cuts mid-pair → role:tool inside but anchor outside → API rejects. Orphan-pair filtering in `loadRecentTurns` collects valid tool_call_ids from in-window assistants and drops unbacked tool messages.
- **Treating tool results as user-facing.** They're DATA the model integrates into natural language. Never stream their JSON to the client; never let their raw shape leak into the assistant's text response.
- **MAX_ITERATIONS scaling with tool count.** Number of tools ≠ expected steps per task. Anchor MAX to "how many calls does a typical query genuinely need," usually 1–3, not 10. Bump only when you see real symptoms.
- **Single-tool agents.** A one-tool agent can't multi-step in any meaningful sense — every task is "call the one tool or don't." For genuine ReAct workflows you need ≥2 tools that compose.
- **Reactive-vs-planning confusion in interviews.** Default IS reactive; planning is an escalation. Don't say "all agents are planning agents" — they're not. Say "reactive by default, planning when reactive drifts."

### Operational notes

- **Database migration on first deploy:** TypeORM synchronize adds the new ChatMessage columns automatically (`toolCallId`, `toolCalls`). The CHECK constraint from 7.1's hardening lives on `transcript_chunk`, not `chat_message` — no conflict.
- **Try it manually:** ask *"What meetings did I have this week?"* → expect listMyMeetings tool call → meeting title in response. Then follow up *"Tell me more about [that meeting]"* → expect getMeetingDetails to fire with the id from turn 1's tool result (no longer needs to re-list).
- **Server log on multi-step:** look for two `[chat] iter=N tool_calls=...` lines in one turn — that's a genuine multi-step request firing two iterations of the loop.

### Go-deeper resources

- Yao et al. 2023, [*"ReAct: Synergizing Reasoning and Acting in Language Models"*](https://arxiv.org/abs/2210.03629) — original ReAct paper; skim sections 1–3
- Anthropic, [*"Building effective agents"*](https://www.anthropic.com/research/building-effective-agents) — most up-to-date practitioner essay; the workflow vs agent distinction is the interview-grade content
- LangChain, [*"Why agents fail"*](https://blog.langchain.dev/why-agents-fail/) — known taxonomy of multi-step breakage patterns
