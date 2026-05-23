# Phase 2 — Streaming UX

## Session 2.1 — Cancellation, partial-save, full streaming lifecycle

**Built in Audia:**
- **Server** ([chat/route.ts](../src/app/api/chat/route.ts)): forwarded `req.signal` to `groq.chat.completions.create(params, { signal })` so client abort cancels the upstream call and stops the token meter. Restructured the streaming loop with `try/catch/finally` so `chatRecord.response` is *always* persisted — partial on abort, full on clean close. Wrapped `controller.enqueue` in `try` to handle the race where the client closes while the server is mid-enqueue. Differentiated `AbortError` from real errors (only `console.warn` on real ones). Log label `chat-aborted` distinguishes abort calls from clean calls for Phase 12 dashboards.
- **Client** ([ChatPanel.tsx](../src/app/components/ChatPanel.tsx)): added `abortControllerRef`, `stop()` function, conditional send/stop button render (stop button uses `error.main` red), `AbortError` differentiated in catch (no error UI shown on intentional stop, partial content remains visible). Removed `loading` from TextField disable so users can compose next message while current one streams.

### Concept summary

Streaming is not an add-on; it's a direct expression of how LLMs produce output (autoregressive, one token at a time). The engineering work is plumbing — getting each token from provider SDK → server `ReadableStream` → client `reader.read()` → DOM render, while handling the full lifecycle including cancellation. The defining shift from naive to production code is treating **abort as first-class control flow, not exception**. Concretely: forward `AbortSignal` from client fetch through `req.signal` to the upstream SDK call (so cancellation propagates and stops the token meter), persist accumulated response in `finally` (so partials aren't lost), and differentiate `AbortError` from real errors everywhere it's caught. Audia chose raw `ReadableStream` over SSE (no framing needed), WebSockets (no bidirectional), and Vercel AI SDK (primitives-first to feel what's being abstracted).

### 5 most-likely interview questions

1. **Q: How does cancellation work end-to-end in your streaming chat?**
   A: One AbortController on the client. Its signal goes to `fetch`, which becomes `req.signal` on the server. We forward `req.signal` to the Groq SDK as its `signal` option. A single `.abort()` cascades: client fetch rejects with AbortError → HTTP connection closes → server's `req.signal.aborted` is true → SDK cancels the upstream Groq call → Groq stops generating. The server's `finally` block persists whatever response accumulated and logs usage with a `chat-aborted` label.

2. **Q: Why does streaming feel faster than non-streaming if total time is the same?**
   A: Streaming optimizes **time to first token (TTFT)**, not total generation time. Non-streaming shows a blank screen for the full duration, then a wall of text. Streaming shows the first words within roughly the model's prompt-processing time (often ~200ms), and text continues to appear at the model's token-emission rate (Groq ~500 TPS, OpenAI ~50 TPS). Users tolerate a 20-second streaming response but not a 3-second blank screen. Wall-clock to *last* token is the same; wall-clock to *first* token is dramatically different.

3. **Q: What's the difference between SSE and raw HTTP chunked streaming?**
   A: SSE is HTTP chunked streaming with extra framing — `data: <payload>\n\n` per event, browser exposes `EventSource` API with auto-reconnect. Raw chunked is just bytes. For single-direction LLM streams with a custom client, raw chunked is simpler and saves framing overhead per chunk. SSE wins when you want named event types, server-sent reconnection, or `EventSource` ergonomics. OpenAI uses SSE; Audia uses raw chunked.

4. **Q: A user reports that hitting stop doesn't actually save tokens. How do you fix it?**
   A: That means the abort signal isn't being forwarded to the upstream LLM call. The fix is one line: `await provider.create(params, { signal: req.signal })` — pass the request's abort signal to the SDK as a stream option. Without this, the client abort closes the HTTP connection but the server's `for await` loop keeps reading from Groq's stream, generating tokens nobody will see, and billing continues. With it, the SDK propagates the cancellation to Groq's API, generation stops mid-token, and the meter halts.

5. **Q: Why does your server save the response inside `finally`?**
   A: To guarantee partial-save on abort and errors. The `for await` loop terminates three ways: clean close, error, or abort-via-signal. If we save *after* the loop, we lose the accumulated response on the abort and error paths. `finally` runs in all three cases. The persisted partial has real value — cost accounting accuracy via the usage logger, resume-ability when we add conversation memory in Phase 5, and observability (a spike in saved-but-partial responses is a signal that users are aborting a lot — investigate UX or latency).

### Gotchas

- **Forgetting to forward `req.signal` to the upstream SDK call.** Client aborts, server keeps reading, you keep paying. Pure cost-control hygiene.
- **`save()` outside `finally`.** Loses partial response on abort. Persistence belongs in `finally`.
- **Treating `AbortError` like a real error.** Shows users a "something went wrong" red message when they intentionally stopped. Distinguish `err.name === "AbortError"` in every catch.
- **Disabling the input field while streaming.** Breaks multi-turn flow. Just check `loading` inside the submit handler.
- **Buffering tokens before re-rendering.** Negates streaming entirely. Re-render on each `reader.read()` chunk.
- **Wrapping `controller.enqueue` without a try.** There's a race where the client closes during enqueue; without the try, that error bubbles out of `start()` instead of letting you break out of the loop cleanly.

### Go-deeper resources

- MDN: [*Streams API*](https://developer.mozilla.org/en-US/docs/Web/API/Streams_API) and [*AbortController*](https://developer.mozilla.org/en-US/docs/Web/API/AbortController)
- Vercel AI SDK docs — to see what a library abstracts when you don't write it yourself
- Theo (t3.gg) — "How to stream responses" YouTube videos for visual pattern reinforcement
