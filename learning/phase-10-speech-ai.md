# Phase 10 — Speech AI

## Session 10.1 — ASR architectures, CTC, Whisper internals

**Built in Audia:**
- [Transcription.ts](../src/entity/Transcription.ts) — `TranscriptSegment` gains an optional `confidence?: number` (avg ASR word-confidence). `segments` is a `simple-json` column, so no migration.
- [transcribe/route.ts](../src/app/api/transcribe/route.ts) — `DeepgramWord` gains `confidence`; `parseSegments` now keeps a running mean of word confidence per segment and stores it (was discarding it).
- [TranscriptPanel.tsx](../src/app/components/TranscriptPanel.tsx) — segments below `LOW_CONFIDENCE = 0.7` show a muted `~NN%` chip (warning-tinted, as a genuine *state* signal) with a tooltip — "the ASR told us it's unsure here." `tsc` clean.

### Concept summary (definition · type · example)

ASR maps `audio → tokens`, and the input is far longer than the output (hundreds of frames/sec vs a few tokens), so the central problem is **alignment** with no frame↔token labels. Audio is first turned into a **log-Mel spectrogram** (≈25ms frames × mel frequency bands — a 2D image of sound). Three architecture families solve alignment differently: **CTC** emits a token (or a special **blank**) at every frame independently, then collapses repeats+blanks — alignment-free, encoder-only, streamable but no cross-token context; **RNN-T / transducer** adds a prediction network (a built-in LM) so outputs condition on prior outputs — the streaming-first, context-aware choice (voice assistants, Deepgram-class); **attention encoder-decoder (AED/seq2seq)** has a decoder attend over the whole encoded spectrogram autoregressively (same shape as an LLM) — best offline accuracy but **not natively streaming** (Whisper). **Whisper** is an AED trained on ~680k hrs of weakly-supervised audio, 30s log-Mel windows, multitask via special tokens (transcribe/translate/timestamps/language-ID). Accuracy is scored with **WER = (S+D+I)/N** (word edit distance ÷ reference length; lower better). Every ASR decoder is **probabilistic**, so each word carries a **confidence** ∈ [0,1] — low ≈ likely error — which Audia was throwing away and now keeps. **Streaming vs batch**: batch transcribes a finished recording (Audia today), streaming emits revised partials live (10.2) and structurally favors RNN-T over Whisper's AED. Audia uses **Deepgram nova-2** — proprietary, hosted, streaming-capable, diarizing, with per-word confidence — over self-hosted Whisper.

### 5 most-likely interview questions

1. **Q: The three ASR architecture families and their trade-off?**
   A: "CTC, RNN-T/transducer, and attention encoder-decoder. CTC predicts a token or a blank at each frame independently and collapses — alignment-free and streamable, but no cross-token context. RNN-T adds a prediction network so it conditions on prior tokens — the streaming + context-aware sweet spot, which is why voice assistants and Deepgram-class models use it. Attention encoder-decoders — Whisper — attend over the whole clip autoregressively for the best offline accuracy, but they're not natively streaming. The axis is streaming/latency vs offline accuracy."

2. **Q: How does CTC handle alignment without frame-level labels?**
   A: "It introduces a blank token and predicts a distribution over tokens-plus-blank at every frame, scoring all alignments that collapse to the target. The collapse rule is: merge consecutive duplicates, then drop blanks — so 'h-h-∅-e-l-l-∅-l-o' becomes 'hello'. The blank is what lets it represent 'no new token here' and separate real repeats (the ∅ between the two l's). That's why CTC needs no frame↔token alignment in the training data."

3. **Q: What is Whisper, architecturally, and why isn't it ideal for live captions?**
   A: "An attention encoder-decoder Transformer: the encoder reads a 30-second log-Mel spectrogram, the decoder generates tokens attending over it, with special tokens selecting the task — transcribe, translate, timestamps, language ID. It's robust because it was trained on ~680k hours of weakly-supervised web audio. It's not ideal for live captioning because the AED wants the whole window before decoding — it's batch-oriented; for streaming you'd reach for a transducer."

4. **Q: How is ASR accuracy measured?**
   A: "Word Error Rate — (substitutions + deletions + insertions) / reference word count, the word-level edit distance normalized by reference length. Lower is better, 0 is perfect. 'let's ship on friday' vs 'lets ship friday' is one substitution plus one deletion over four reference words = 0.5, 50% WER. WER is the standard, though it's punctuation/casing-sensitive unless you normalize first."

5. **Q: What's ASR confidence and how did you use it in Audia?**
   A: "The decoder is probabilistic, so every word comes with a confidence in [0,1] — the model's certainty. Low confidence correlates with likely errors: noise, crosstalk, rare terms. Audia was discarding Deepgram's per-word confidence; I now average it per segment, store it on the segment, and flag segments under 0.7 in the transcript with a muted percentage chip + tooltip — so the user sees where the transcription is shaky instead of trusting every word equally."

### Gotchas

- **Alignment is the core ASR problem** — input frames ≫ output tokens with no labels. CTC (blank+collapse), RNN-T (prediction net), AED (attention) are three answers.
- **Streaming ≠ accuracy** — AED (Whisper) is most accurate offline but not streaming; transducers trade a little accuracy for live partials.
- **WER is sensitive to normalization** — casing/punctuation inflate it; normalize before comparing.
- **Confidence is per-word** — averaging to a segment is a (useful) simplification; a single low word can hide in a high-average segment.
- **`confidence` is optional** — legacy rows + golden-set test segments lack it; UI must guard `typeof === "number"`.

### Operational notes

- **Verify:** record (or upload) audio, especially something noisy/with a rare term → low-confidence segments show a `~NN%` chip + tooltip. Server stores `confidence` on each segment.
- **Deepgram** model is `nova-2`, `diarize + smart_format + punctuate`; per-word `confidence` comes back by default in `results.channels[0].alternatives[0].words[]`.
- **Deferred / extensions:** a `wordErrorRate()` utility + transcription eval (needs ground-truth references); per-*word* (not per-segment) confidence highlighting; Phase 10.2 = diarization deep-dive + **streaming/live transcription** (where the architecture choice — transducer vs AED — actually bites).

---

## Session 10.2 — Diarization & streaming / live transcription

**Built in Audia:**
- [src/app/api/deepgram-token/route.ts](../src/app/api/deepgram-token/route.ts) — POST, auth-gated; `deepgram.auth.v1.tokens.grant()` → short-lived bearer token. The keystone: lets the browser stream directly without ever seeing the real key.
- [src/app/components/useLiveTranscription.ts](../src/app/components/useLiveTranscription.ts) — client hook: fetch token → `DeepgramClient.listen.v1.connect({ model, Authorization, diarize, interim_results, queryParams:{diarize_model:"latest"} })` → `MediaRecorder` chunks → `socket.sendMedia` → parse `Results` into `finalized` (is_final, speaker-labelled) + `interim` (draft).
- [HomeClient.tsx](../src/app/HomeClient.tsx) — `live.start(stream)` on record, `live.stop()` on stop, `live.reset()` on new; batch `/api/transcribe` save unchanged.
- [MainPaneStates.tsx](../src/app/components/MainPaneStates.tsx) — `RecordingState` renders live captions (finalized solid, interim faded/italic).
- `tsc` clean (caught: SDK types `diarize`/`interim_results`/`punctuate`/`smart_format` as **string** query values, not booleans). **Runtime (browser + mic + WS) verification pending — see gotchas.**

### Concept summary (definition · type · example)

Live transcription needs two things batch didn't: **streaming ASR** and **streaming diarization**, both over a **WebSocket**. *Diarization* = "who spoke when" (clustering-based vs end-to-end neural; offline vs streaming — streaming is harder because it can't look ahead). *Streaming ASR* emits **interim** results (`is_final:false`, revisable drafts) then **final** ones (`is_final:true`, committed) once **endpointing** detects a pause; this trades accuracy/stability for latency, which is exactly why streaming favors **RNN-T over Whisper's AED** (10.1). Transport is a **WebSocket** (full-duplex: audio up, transcripts down, simultaneously). The load-bearing architecture decision: a **serverless Next.js route handler can't host a persistent WebSocket**, so the server mints a **short-lived token** (a normal HTTP call) and the **browser opens the Deepgram socket directly** — audio never proxies through us and the real key never reaches the client. In Audia, live captions are real-time UX during recording; the **saved** transcript still comes from the accurate **batch** `/api/transcribe` on stop. (Streaming diarization note: `diarize_model=v2` isn't available for streaming → pin `latest`.)

### 5 most-likely interview questions

1. **Q: How do you do live transcription in a serverless app — doesn't streaming need a WebSocket the server can't hold?**
   A: "Right — a serverless route handler is request/response, it can't keep a socket open. So you flip it: the server mints a short-lived token over plain HTTP, and the browser opens the speech provider's WebSocket *directly* with that token. Audio streams browser→provider, never through your functions, and the real API key never reaches the client — only a token that expires in seconds. That's the ephemeral-token pattern; it's how you stream from serverless."

2. **Q: Interim vs final results — what's the difference and how do you render them?**
   A: "Interim results are revisable drafts the model emits as audio arrives, flagged `is_final:false`; finals are committed once the segment stabilizes — usually when endpointing detects a pause — flagged `is_final:true`. You render interim faded/italic and promote it to solid on final, so the user sees words appear live and then lock. The trade is latency vs stability: snappier finals mean more mid-sentence corrections."

3. **Q: Why is streaming diarization harder than offline?**
   A: "Offline diarization can cluster voice embeddings over the *whole* recording before deciding who's who; streaming has to assign a speaker the moment each word arrives with no future context, so it can label early speech wrong and only correct later. It's a no-look-ahead constraint. Practically, providers even ship different diarizers for streaming — Deepgram's v2 diarizer isn't available for streaming, so you pin `diarize_model=latest`."

4. **Q: Why does streaming favor a transducer over Whisper?**
   A: "Whisper is an attention encoder-decoder — it wants the whole window before decoding, so it's batch-oriented. Streaming needs a model that emits as audio arrives and revises, which is the transducer/RNN-T design. So the same accuracy-vs-latency axis from ASR architectures decides it: Whisper for best offline accuracy, a transducer for live."

5. **Q: In Audia, the saved transcript and the live captions can differ — why, and is that a bug?**
   A: "Not a bug — deliberate. The live captions come from the streaming socket (interim/final, optimized for latency); the *saved* record comes from re-transcribing the full audio in batch on stop, which is more accurate because it has the whole clip. Live is real-time UX; batch is the source of truth. Separating them keeps the persistence path simple and the stored transcript high-quality."

### Gotchas

- **⚠️ Runtime-unverified spots (tsc-clean, but not browser-tested):** (1) the token **auth header** — grant tokens are `Bearer`; if Deepgram rejects, try `Token ${token}`. (2) The v5 SDK's WebSocket running in the **browser bundle** — it's a server-oriented SDK; if it pulls Node deps, fall back to a raw browser `WebSocket` to `wss://api.deepgram.com/v1/listen?...`. (3) The connect→attach-handlers→open **sequence** — we guard both orders (`on("open")` + a `readyState === OPEN` check).
- **SDK query params are strings, not booleans** — `diarize: "true"`, not `diarize: true` (tsc caught this).
- **Serverless can't hold a WebSocket** — don't try to proxy audio through a route handler; mint a token, stream client-direct.
- **Streaming diarizer ≠ batch diarizer** — pin `diarize_model=latest` for streaming (v2 errors).
- **Two MediaRecorders on one stream** — the live hook runs its own recorder; pausing the *batch* recorder doesn't pause live captions (minor, acceptable).

### Operational notes

- **Verify (browser):** record → captions appear live (interim faded → finalize solid, speaker-labelled). Server log: `[deepgram-token]` grant. On stop → existing batch transcription saves the canonical record.
- **`/api/deepgram-token`** returns a ~30s token; the browser must connect promptly.
- **Deferred:** persist the live transcript instead of re-batching (would need assembling final segments + timing); pause/resume wired into the live socket; the raw-WebSocket fallback if the SDK isn't browser-friendly at runtime.
- **🐢 Phase 10 (Speech AI) COMPLETE** — ASR internals + confidence (10.1) → diarization + live streaming (10.2).
