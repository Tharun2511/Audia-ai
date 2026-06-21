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
