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
