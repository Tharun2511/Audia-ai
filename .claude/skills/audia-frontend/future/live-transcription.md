# Live transcription — UI patterns

Future feature: real-time captions appear while recording. This doc describes the visual + accessibility pattern so any session that adds it lands consistently.

## Surface

A new main-pane state, `LiveTranscribing` (or a sub-mode of `Recording`):

```
┌──────────────────────────────────────────────┐
│ ● REC 02:14                  [Stop] [Pause]  │
│                                              │
│ ╭─ Waveform ──────────────────────────────╮  │
│ │  ▁▂▃▅▇▆▄▃▂▁  (live, RAF-driven)         │  │
│ ╰──────────────────────────────────────────╯  │
│                                              │
│ Live captions                                │
│   0:14  So the main thing is — we're going   │
│   0:21  to ship phase 4.3 next sprint, and…  │
│   0:28  the timeline depends on…             │
│                                              │
└──────────────────────────────────────────────┘
```

## Components

- **Recording header**: red blinking dot (existing `rec-blink` keyframe) + timer in Geist Mono + Stop / Pause buttons.
- **Waveform**: `<canvas>` + `requestAnimationFrame`. **No library.** Heights driven from `AnalyserNode.getByteFrequencyData()`. Color: `primary.main` at varying opacity by bar height.
- **Caption list**: `<Box aria-live="polite" aria-atomic="false">` with `<Caption>` items. Each has a timestamp (Geist Mono, `text.secondary`) and the text (`text.primary`).
- New final segments fade in with `.animate-fade-in`. Interim/partial segments are visually italicized (`fontStyle: "italic"`, `text.disabled`) and **not** announced — only final segments enter the live region.

## Visual rules

- Caption width cap: 720px (reading width). Center the column inside the main pane.
- Timestamps left, text right. Don't right-align timestamps to the text.
- Auto-scroll the caption list to keep the latest segment visible, but **pause auto-scroll** if the user manually scrolls up (so they can read earlier segments). Resume on a small "Jump to latest" chip that appears in the bottom-right when paused.
- The waveform is **decorative**, not informational. Don't put critical info inside it.

## Accessibility

- `aria-live="polite"` on the caption container.
- Only final segments announced. Interim segments visible but with `aria-hidden="true"`.
- Reduced motion: waveform should still render (it's the recording indicator), but use a static bar pattern instead of animated bars. The pulse-ring on the record button also disables. See [../motion.md](../motion.md).
- Timer: announce changes only every minute, not every second.

## Sketch — data flow (verify during implementation)

- Browser captures audio via `MediaRecorder` or `AudioWorklet`.
- Stream chunks to a route handler that pipes to Deepgram (already in dependencies) with `interim_results: true`.
- Server-Sent Events back to the client (or WebSocket if Deepgram requires).
- Client appends interim → replaces with final on `is_final: true`.

When implementing, read the relevant `node_modules/next/dist/docs/` and the Deepgram SDK docs. Don't trust training-time API surface.

## After recording stops

Transition to `Processing` while the final transcript + summary are persisted, then to `SessionView`. The captions already shown should match the final transcript — no flash of empty state, no "loading" placeholder over content the user already saw.

## Edge cases to design for

- **Network drop mid-stream**: show an inline warning chip ("Reconnecting…") above the captions, retry transparently. Don't tear down the UI.
- **Permission denied**: show an `EmptyState` ([../patterns/empty-state.tsx](../patterns/empty-state.tsx)) with a "Grant microphone access" CTA that re-prompts.
- **No speech detected for 30s**: don't auto-stop. Show a subtle dot status ("Listening…") and keep the timer running. User decides when to stop.
- **Very long sessions (>1h)**: caption list virtualizes (TanStack Virtual or a simple windowed render) — don't render 1000+ DOM nodes.

## Don't reinvent

- The Recording state already has the timer + pulse-ring + Stop button. Extend it; don't duplicate.
- Reuse `TranscriptPanel`'s segment rendering for the live caption list if shape allows.
