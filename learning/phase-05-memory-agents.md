# Phase 5 — Memory & Agents

## Session 5.1 — Conversation memory: rolling buffer, summary buffer, vector memory

**Built in Audia:**
- New [src/entity/ChatMessage.ts](../src/entity/ChatMessage.ts) — one row per turn, grouped by `sessionId`; composite index `(sessionId, createdAt)` for the hot "last N turns" query; `citations` stored as `simple-json` for UI replay
- New [src/lib/chat-memory.ts](../src/lib/chat-memory.ts) — `loadRecentTurns()` (DESC + LIMIT + reverse for chronological order, strips `[N]` markers from assistant text), `saveTurn()`, constant `HISTORY_TURN_PAIRS = 5`
- Updated [chat/route.ts](../src/app/api/chat/route.ts) — accepts `sessionId | null` in body, mints UUID on null, loads last-5 turn-pairs, persists user turn before stream + assistant turn in `finally`, returns sessionId in `X-Chat-Session` header. System prompt extended with CONVERSATION RULES + cross-turn `[N]`-marker caveat
- Updated [ChatPanel.tsx](../src/app/components/ChatPanel.tsx) — `sessionId` state, echoes back on subsequent fetches, reads `X-Chat-Session` from response, new Reset (↻) icon button that aborts in-flight stream + clears messages + clears sessionId
- Registered `ChatMessage` in [data-source.ts](../src/db/data-source.ts) entities list

### Concept summary

LLMs are stateless: every call is independent, conversation is an illusion the client maintains by replaying past turns. This makes "memory" a client-owned prompt-construction discipline, not a model feature. Three families: rolling buffer (last N turns verbatim, cheap, exact), summary buffer (compress old turns to a running paragraph; bounded but lossy and adds a second LLM call), vector memory (embed turns, retrieve only the relevant ones; scales to thousands of turns). Production hybrid layers rolling + vector. Audia 5.1 ships rolling buffer at N=5 turn pairs — appropriate for short chatbot sessions. Cost shape matters: unbounded history is O(N²) input tokens across N turns; capped is O(N×K). At gpt-4o prices and 10k sessions/day, that's a ~$700k/year delta. Memory is orthogonal to RAG retrieval: each turn does its own fresh chunk lookup; `[N]` citation markers from prior turns must be stripped from LLM input (they referred to that turn's chunk numbering, not the current turn's). Persist user turn before streaming; assistant turn in `finally` — two distinct rows, never the two-step update pattern that caused Phase 4.2's orphan bug.

### 5 most-likely interview questions

1. **Q: How does conversation memory actually work, given the model is stateless?**
   A: The model is stateless every call — no server-side conversation object. "Memory" lives in the client's `messages[]` array, which the client re-sends in every request. The client picks a strategy for which past turns to include: rolling buffer keeps the last N raw, summary buffer compresses old turns into a paragraph, vector memory retrieves only the *relevant* past turns. The choice is a cost-vs-fidelity trade-off. Audia ships rolling-buffer last-5 because sessions are short and exact recall of recent context matters; on a long-session product we'd reach for vector memory.

2. **Q: Walk me through the cost growth shape.**
   A: Unbounded rolling history is O(N²) in total input tokens — turn n sends roughly n×t tokens of history, summed across N turns equals `t × N(N+1)/2`. Capped at last-K it's O(N×K). At llama-3.1-8b-instant ($0.05/M input), 10k sessions/day of 20 turns ≈ $5.5k/year capped vs $19k/year unbounded. On gpt-4o ($2.50/M) it's $275k vs $960k. The asymptotic gap is what every production team eventually optimizes — usually with a hybrid of rolling buffer + vector memory once sessions get long.

3. **Q: Why are you using a rolling buffer and not summary-buffer memory?**
   A: Three reasons. One: Audia sessions are typically 1-10 turn pairs — summarization isn't paying for itself at that length. Two: summary buffer adds a *second* LLM call per turn (or every K turns), introducing both extra cost and extra latency on the critical path. Three: it adds a new failure mode — summary drift, where a subtle misread compounds across regenerations and silently corrupts the running summary. Rolling buffer is exact-recall, cheap, and has no drift surface. We'd revisit if median session length crosses ~15 turns or context-window pressure became measurable.

4. **Q: How do you handle citations in past turns when sending them as history?**
   A: We strip `[N]` markers from assistant turns before they become LLM input. The markers referred to *that turn's* chunk numbering at retrieval time; the current turn does a fresh retrieval with a different numbering, so leaving them in would mislead the model into citing the wrong chunk. We keep the markers in the persisted row though — the UI re-renders chips on conversation reload using the citations metadata we stored alongside the turn. It's a "load-time strip, render-time restore" pattern.

5. **Q: What's the security boundary on sessionId? What stops me from guessing one and reading another user's chat?**
   A: Two layers. One: sessionId is a server-minted UUIDv4 — 122 bits of entropy, infeasible to guess. Two: every history query filters `WHERE sessionId = ? AND userEmail = ?` — even if a sessionId leaked, the row lookup is bounded to the requesting user's email pulled from the session cookie. Same defense pattern as the chunk retrieval's ownership filter — multi-tenant isolation at the SQL layer, not at the application layer. If we ever introduced session-sharing as a feature (multiple users in one chat), the model would change explicitly via an ACL row, not by trusting client-supplied sessionId.

### Gotchas

- **Forgetting to strip `[N]` markers from history.** Symptom: model cites wrong chunks confidently. Past markers refer to past numberings; sending them across turns is the bug.
- **Not filtering history by `userEmail`.** Leaked sessionId becomes full conversation exfiltration. Always `WHERE sessionId = ? AND userEmail = ?`.
- **Two-step write for assistant turn.** Same trap as Phase 4.2's orphan-NULL-embedding bug — don't persist a placeholder then update. Save in `finally` once with the accumulated content.
- **Storing the wrapped userMessage instead of the raw question.** The context block is per-turn ephemera; storing it bloats history and re-injects stale chunks into future prompts.
- **Unbounded rolling buffer "to be safe."** Scales O(N²) in cost AND latency (TTFT grows with prompt length). Cap is mandatory.
- **Summary buffer at short session lengths.** The extra LLM call eats your latency budget AND introduces summary drift for sessions too short to benefit. Not free.
- **Mixing memory and retrieval mentally.** They're orthogonal layers: retrieval supplies the *data* per turn, memory supplies the *dialogue* across turns. Each turn does both.

### Go-deeper resources

- LangChain docs, [*"Memory in LangGraph"*](https://langchain-ai.github.io/langgraph/concepts/memory/) — clearest framing of the three families
- OpenAI cookbook, [*"How to handle long conversations with the chat model"*](https://cookbook.openai.com/examples/how_to_handle_long_conversations) — practical truncation patterns
- Anthropic, [*"Prompt caching"*](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — the cost lever every chat product eventually pulls; pairs with stable system prompts + bounded history
- Liu et al. 2023, [*"Lost in the Middle"*](https://arxiv.org/abs/2307.03172) — relevant for memory too: long history puts important early turns in the U-curve dead zone
