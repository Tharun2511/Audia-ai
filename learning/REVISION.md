# Audia AI — Master Revision Document

> **How to use this document:** This is your night-before-the-interview cram doc. It teaches every concept end-to-end, not just reminds you. Read it cover-to-cover ~once per month while job hunting. The cheat sheets (`phase-NN-*.md`) are 5-minute walking-in-the-door reviews; this is the 4–5 hour deep refresh.
>
> **Recurring callouts:** **📐 Math you must memorize** · **⚖️ Trade-off tables** · **🎯 Defense talking points** (interviewer challenges + comebacks) · **⚠️ Common pitfalls**

---

## Table of contents

**Part I — Fundamentals**
- [§1. How an LLM decides the next token](#1-how-an-llm-decides-the-next-token) ✅ *(Phase 0.1)*
- [§2. The model API surface — sampling params, streaming, cost math](#2-the-model-api-surface--sampling-params-streaming-cost-math) ✅ *(Phase 0.2)*
- §3. Prompt engineering as a discipline *(Phase 1.1 — TBD)*
- §4. Prompt injection & output safety *(Phase 1.2 — TBD)*

**Part II — Real-time UX**
- §5. Streaming patterns: SSE, ReadableStream, AbortController *(Phase 2 — TBD)*

**Part III — Retrieval (RAG)**
- §6. Embeddings: vector spaces, cosine, model choice *(Phase 3.1 — TBD)*
- §7. Chunking strategies *(Phase 3.2 — TBD)*
- §8. pgvector + indexing (IVFFlat vs HNSW) *(Phase 3.3 — TBD)*
- §9. Retrieval: top-k, MMR, hybrid, lost-in-the-middle *(Phase 4.1 — TBD)*
- §10. Generation with retrieval: templates, citations, hallucination control *(Phase 4.2 — TBD)*

**Part IV — Conversation state**
- §11. Memory strategies *(Phase 5 — TBD)*

**Part V — Measurement**
- §12. Eval theory *(Phase 6.1 — TBD)*
- §13. Building an eval harness *(Phase 6.2 — TBD)*

**Part VI — Agents**
- §14. Tool use & function calling *(Phase 7.1 — TBD)*
- §15. Multi-step agents & ReAct *(Phase 7.2 — TBD)*
- §16. MCP — Model Context Protocol *(Phase 7.3 — TBD)*

**Part VII — Search**
- §17. Re-ranking with cross-encoders *(Phase 8.1 — TBD)*
- §18. Hybrid search & query expansion *(Phase 8.2 — TBD)*

**Part VIII — Speech**
- §19. ASR architectures, Whisper, diarization *(Phase 9.1 — TBD)*
- §20. Streaming ASR *(Phase 9.2 — TBD)*

**Part IX — Multimodal**
- §21. Vision-language models, OCR *(Phase 10 — TBD)*

**Part X — Fine-tuning**
- §22. When to fine-tune *(Phase 11.1 — TBD)*
- §23. Dataset preparation *(Phase 11.2 — TBD)*
- §24. Running a fine-tune *(Phase 11.3 — TBD)*

**Part XI — Production AI Ops**
- §25. Observability, prompt versioning, cost tracking *(Phase 12.1 — TBD)*
- §26. Production hardening: caching, rate limiting, guardrails *(Phase 12.2 — TBD)*

**Appendices**
- [A. Glossary](#appendix-a-glossary)
- [B. One-page architecture diagrams](#appendix-b-architecture-diagrams) *(grows over time)*
- [C. "Defend your Audia" — system design talking points](#appendix-c-defend-your-audia) *(grows over time)*

---

## §1. How an LLM decides the next token

### 1.1 The whole game in one sentence

> An LLM is a function `f(token_sequence) → probability_distribution_over_vocabulary` trained to maximize the likelihood of the next token. Generation is that function applied repeatedly: sample one token, append it, run again.

That's it. ChatGPT, Claude, Llama 3 — all of them. "Conversation" is an illusion produced by feeding the model `[system message, user message, "assistant:"]` and letting it predict tokens until it predicts a stop token. The model has **zero memory** between API calls. State lives in the caller (you), not the model.

This is the single most important sentence in the entire document. Internalize it; almost every architectural decision in AI engineering descends from this fact.

### 1.2 The end-to-end pipeline

```
"Audia is great"
       │
       ▼
[1] Tokenization        →  [21034, 689, 318, 1049]    (subword IDs)
       │
       ▼
[2] Embedding lookup    →  4 vectors ∈ ℝ^d_model       (one per token)
       │
       ▼
[3] Positional info     →  RoPE rotates Q,K vectors    (so order matters)
       │
       ▼
[4] N transformer blocks (e.g. 32 layers)
       │   each block:  attention → residual → LayerNorm
       │                MLP       → residual → LayerNorm
       ▼
[5] Final projection    →  logits ∈ ℝ^vocab_size       (V ≈ 128k)
       │
       ▼
[6] Sampling (temperature → top-p → sample)  →  next_token_id
       │
       ▼
[7] Append to context, GOTO [1]            (autoregressive loop)
```

Memorize this seven-step picture. In every system design interview, your first move is drawing it on the whiteboard.

### 1.3 Tokens — the unit of work

Modern LLMs don't operate on characters or words. They operate on **sub-word tokens** produced by Byte-Pair Encoding (BPE) or SentencePiece. The vocabulary is a fixed set of ~100,000–130,000 byte sequences chosen during pre-training to compress the training corpus efficiently.

Example: `"Audia is great"` tokenizes (in GPT-4's tokenizer) approximately as `["Aud", "ia", " is", " great"]` → IDs `[21034, 689, 318, 1049]`. Note the leading spaces are part of the token — that's why " great" is a single token but "Great" at the start of a sentence might be two.

**Why subword?** Three failure modes BPE solves simultaneously:
- **Character-level:** sequences become 5–10× longer. Attention is O(n²) in sequence length, so this is expensive.
- **Word-level:** vocabulary explodes (millions of words across all languages), every typo or rare name is OOV (out-of-vocabulary), no graceful handling.
- **BPE:** common words like "the" become a single token; rare words like "anthropomorphize" split into pieces; novel tokens like new product names compose from sub-pieces. ~4 chars/token in English, ~3 chars/token in code, much worse for non-Latin scripts.

**Practical:**
- LLM cost is **per token**, never per word.
- 1,000 English words ≈ 1,300 tokens. 1,000 lines of Python ≈ 2,500 tokens.
- Foreign names ("Vreddy", "Sundararajan") often consume 3–5 tokens — a meaningful cost in chat applications.
- Tool to inspect: [tiktokenizer.vercel.app](https://tiktokenizer.vercel.app). Paste any Audia transcript and see the tokens.

**📐 Math you must memorize:**
- English: ~4 chars/token → roughly `tokens = chars / 4`
- Cost: `total_cost = (input_tokens × input_rate) + (output_tokens × output_rate)`
- Output tokens cost 2–5× input tokens on most APIs (because output is serial, compute-heavier)

### 1.4 Embeddings — integers become vectors

Once you have token IDs, you need to turn them into something a neural network can do math on. The answer is the **embedding matrix** `E`:

```
E ∈ ℝ^(vocab_size × d_model)

token_id 21034  →  E[21034]  →  a vector ∈ ℝ^d_model
```

For Llama 3 8B, `d_model = 4096`. The matrix `E` is just a lookup table — but it's **learned during training**. Backpropagation adjusts `E` so that tokens appearing in similar contexts end up with similar vectors.

**Why this is profound:** the geometry of `E` *is* semantic meaning. Vectors for "Paris" and "France" are close. Vectors for "king" and "queen" differ by approximately the same direction as "man" and "woman" — the famous `king − man + woman ≈ queen` result. This is the same mechanism that powers the embedding-based retrieval you'll build in Phase 3 (RAG). Embeddings *are* semantic geometry.

In modern models, the same matrix `E` is often re-used (transposed) as the output projection in step [5] — this is called "tied embeddings" or "weight tying." It saves parameters and tends to help quality.

**🎯 Defense talking point** — Interviewer asks: "Where does the model store its 'knowledge'?" Answer: "Distributed across all the weights — the embedding matrix encodes vocabulary-level semantics, the MLP layers in each transformer block encode factual associations (this has been shown by neurosurgery work like ROME), and attention layers encode contextual relationships. There's no single 'knowledge table.'"

### 1.5 Positional encoding — order matters

If you only had token embeddings, `"Dog bit man"` and `"Man bit dog"` would look identical to the model (just three vectors in either order — attention is permutation-invariant). We need to inject position information.

Three approaches you should be able to name:

| Approach | Used by | How it works |
|---|---|---|
| **Absolute learned** | GPT-2, BERT | Learn a separate position embedding for each position 0..max_len, add to token embedding |
| **RoPE (Rotary)** | Llama, Mistral, most modern | Rotate `Q` and `K` vectors by an angle θ_i that depends on position i. The dot product `Q·K` then depends only on *relative* position |
| **ALiBi** | BLOOM | Add a linear bias `−m·|i−j|` to attention scores; tokens further apart get smaller scores |

**Why RoPE won:** it generalizes to sequence lengths longer than seen during training (with techniques like NTK-aware scaling) and the math is beautiful — pure rotations preserve vector norms.

For Audia: Llama-3.1 uses RoPE. You don't need to implement it, but if asked "how does the model know token order?" — say RoPE rotates query/key vectors by position-dependent angles, making attention sensitive to relative position.

### 1.6 Attention — the heart of the matter

> **Reading guide:** the **intuition** below is must-know. The **math** section is marked `[OPTIONAL]` — skip it on first read; you can answer 90% of AI Engineer interview questions about attention without the derivation. Come back to the math only if you're targeting ML Engineer / Research roles.

#### The intuition (must-know)

Attention is a soft, differentiable **content-addressable lookup**. For each token in the sequence, the model figures out *how relevant every other token is to it right now*, then produces a representation that's a blend of all of them weighted by that relevance.

Think of it like a meeting:
- Each person has a **question** (Q) they're trying to answer
- Each person also has an **advertisement** (K) of what they know about
- Each person has **content** (V) they can contribute
- For each person, the system computes a weighted average of everyone's content, weighted by how well their question matches each person's advertisement

That's it. **You don't need the matrix math** to discuss this in an interview. You need:
- The word "attention"
- The acronyms Q, K, V (query, key, value)
- "Softmax turns scores into probabilities"
- "Causal mask prevents looking at future tokens — that's what makes generation autoregressive"
- "Multi-head means doing this in parallel with different learned projections, so different heads can focus on different patterns (syntax, co-reference, etc.)"

If asked to write the formula on a whiteboard, "I know it's `softmax(QKᵀ / √d_k) · V` but I'd have to look up the exact derivation" is a perfectly acceptable AI engineer answer. **Almost no production AI engineer can derive attention from scratch.**

#### [OPTIONAL] The math — skip on first read

For each input token, project it through three learned linear maps to produce **Query**, **Key**, and **Value** vectors:

```
Q = X · W_Q       Q ∈ ℝ^(n × d_k)
K = X · W_K       K ∈ ℝ^(n × d_k)
V = X · W_V       V ∈ ℝ^(n × d_v)
```

where `X` is the input matrix (n tokens × d_model), `W_Q, W_K, W_V` are learned weight matrices, and `n` is sequence length. The attention output is:

```
                    Q · K^T
Attention(Q, K, V) = softmax( ────── ) · V
                     √d_k
```

**Phrase-by-phrase:**

- `Q · K^T` — every query dot-products with every key. Result is an `n × n` matrix where entry `(i, j)` is "how much does token i's query match token j's key?"
- `/ √d_k` — variance normalization. Without it, softmax saturates for large d_k and gradients die.
- `softmax(...)` — row-wise normalize to probabilities.
- `... · V` — weighted sum of value vectors.

#### (back to must-know)

Think of attention as a soft, differentiable **content-addressable lookup**:

- Each token has a **question** (Q) it wants answered
- Each token also has an **advertisement** (K) of what it knows
- Each token has **content** (V) it can contribute
- For each token, we compute a weighted average of all tokens' content, weighted by how well the asking token's question matches each answering token's advertisement

It's a generalization of a database query: instead of a hard match (`WHERE k = q`), every key contributes proportionally to how well it matches.

#### Multi-head attention

A single attention operation has one "perspective." Real models run **H attention operations in parallel**, each with its own `(W_Q, W_K, W_V)` triple:

```
head_i = Attention(X·W_Q^i, X·W_K^i, X·W_V^i)    for i = 1..H
MultiHead = Concat(head_1, ..., head_H) · W_O
```

Different heads learn different patterns: one might track syntax, another co-reference, another long-range dependency. The dimensions usually divide such that `d_model = H × d_k`. For Llama 3 8B: `d_model = 4096`, `H = 32`, so `d_k = 128` per head.

**[OPTIONAL] Math:**
```
H = d_model / d_k          (number of attention heads)
For Llama 3 8B:  H = 4096 / 128 = 32
For Llama 3 70B: H = 8192 / 128 = 64
```
*You only need to know multi-head exists. The arithmetic is for trivia rounds.*

#### Causal mask — the trick that makes generation work

During training and generation of decoder-only LMs (i.e. every chat model you'll touch), we add a **causal mask** that sets all "future" positions to `−∞` *before* softmax:

```
                      Q · K^T
mask_score(i,j) = (────────)  + M(i,j),  where M(i,j) = 0 if j ≤ i, else −∞
                       √d_k
```

After softmax, the `−∞` entries become 0. Token `i` can only attend to tokens 1..i. This is what makes it **autoregressive** / left-to-right: the model literally cannot peek at future tokens because the math forbids it. (Encoder models like BERT don't have this mask — they see the whole sequence bidirectionally. That's why BERT is great at classification but can't generate.)

**🎯 Defense talking point** — "How is generation possible if attention sees everything?" Answer: "Decoder-only LMs use a causal mask that zeros out attention scores for future positions before softmax. During training this lets us train all positions in parallel — each position predicts the next token having only seen prior tokens. During inference we generate one token at a time, but with KV caching we reuse the previously-computed K and V vectors, so each new step only needs to compute one new row of attention."

#### KV caching (engineering aside)

At inference, when generating token `n+1`, you don't recompute `K` and `V` for tokens 1..n — those don't change. They're cached in GPU memory. Only the new token's Q, K, V need computing, and Q dot-products against the cached K. This is the **KV cache** and it's *why generation can scale to long contexts at all*. Memory cost is roughly `2 × n_layers × n × d_model × bytes_per_param` — for Llama 3 8B at 8k context, that's a few GB per concurrent request.

### 1.7 The full transformer block

One block does:
```
x' = LayerNorm(x)
x = x + MultiHeadAttention(x')      # residual
x' = LayerNorm(x)
x = x + MLP(x')                     # residual
```

The MLP is a two-layer feedforward network with a wider hidden dim (typically 4× d_model):
```
MLP(x) = W_2 · activation(W_1 · x + b_1) + b_2
```

Activation is usually SwiGLU or GELU in modern models. The MLP is where most of the parameters live (about 2/3 of the model). **Common metaphor:** attention = communication across tokens; MLP = per-token computation. Attention mixes; MLP thinks.

Stack N of these blocks (Llama 3 8B has 32; 70B has 80). The final layer projects from `d_model` back to `vocab_size` to produce **logits** — one unnormalized score per vocabulary token.

### 1.8 Sampling — turning logits into a token

You now have a vector of `vocab_size` logits. You need to pick one token.

**⚖️ Sampling strategies trade-off table:**

| Strategy | What it does | Use when | Avoid when |
|---|---|---|---|
| **Greedy** (argmax) | Always pick highest-logit token | Code generation, fact retrieval, when determinism matters | Open-ended generation — produces repetitive, dull text |
| **Temperature** | Divide logits by T before softmax. T<1 peakier, T>1 flatter | Universal — the main creativity knob | T=0 is just greedy; very high T (>1.5) becomes incoherent |
| **Top-k** | Keep only top-k tokens by probability, zero the rest | Bound the tail; classic default k=40 | When confidence varies a lot — fixed k is too rigid |
| **Top-p (nucleus)** | Keep smallest set whose cumulative prob ≥ p | Modern default, adapts to model confidence | Rarely a wrong choice; combine with temperature |
| **Min-p** | Keep tokens with prob ≥ p × top_prob | Newer, simpler than top-p, good results | Less standard, may not be supported |

In practice: **temperature → top-p → sample.** Common defaults are T=0.7, top-p=0.9. For structured output (JSON, code, summaries), drop temperature to 0.0–0.3.

**📐 The temperature formula:**
```
softmax_T(x_i) = exp(x_i / T) / Σ_j exp(x_j / T)
```
- `T → 0`: softmax becomes argmax (deterministic, greedy)
- `T = 1`: softmax in its natural form
- `T → ∞`: softmax approaches uniform (random over vocab)

### 1.9 Autoregressive generation — the loop

Once you sample one token, you append it to the context and run the whole pipeline again. This is what "autoregressive" means and it has three crucial implications:

1. **Generation is serial.** Each output token requires one full forward pass. You cannot generate token 5 in parallel with token 3 — token 5 depends on token 4 depends on token 3, etc. This is why **output is slow** while **prompt processing is fast** (the prompt's tokens can be processed in parallel because they all already exist).

2. **The model has no memory between API calls.** To "remember" a prior turn, you must include it in the prompt. Every chat API call processes the full conversation from scratch. This linear cost growth is why memory strategies (Phase 5) exist.

3. **Streaming is "free."** Since the model produces tokens one at a time anyway, sending each one to the client as it's generated is just network plumbing — there's no extra cost or complexity in the model. (Phase 2 covers the engineering side.)

### 1.10 What this means for Audia specifically

The refactor we just did in [src/lib/ai.ts](../src/lib/ai.ts) was small in code but conceptually large:

```ts
messages: [
  { role: "system", content: SUMMARY_SYSTEM_PROMPT },
  { role: "user",   content: `Transcript:\n${transcriptText}` },
],
temperature: 0.2,
max_tokens: 250,
```

**Why each choice:**

- **System role for the instruction.** During training, models see thousands of examples where system messages contain rules and personas, user messages contain inputs. They learn to weight system content as "the operator's intent" and user content as "the input data." Putting our instruction in `user` previously meant the model couldn't cleanly separate "summarize" (intent) from the transcript text (data). This also weakens prompt injection resistance — if the instruction is in user-space, malicious transcript content can override it more easily (we'll harden this in Phase 1.2).

- **Temperature 0.2.** Summarization is structured factual output. We want the same transcript to produce roughly the same summary on repeated calls. T=0.2 means logits are scaled by 1/0.2 = 5×, making the softmax sharply prefer the top-1 token. We don't use T=0 (fully greedy) because a tiny amount of stochasticity can rescue the model from degenerate top tokens; the difference is small but consistent in practice.

- **max_tokens: 250.** Three bullets at ~50 tokens each is ~150 tokens. 250 is a safe upper bound. Without this cap, a misbehaving prompt could rant for thousands of tokens and burn cost. Note: `max_tokens` is **output only** — it does not affect context window math.

- **Extracted `SUMMARY_SYSTEM_PROMPT` as a module constant.** Sets up Phase 6 (evals) — to A/B test prompt versions, you need them addressable, versioned, and comparable.

### 1.11 Common pitfalls in this section

**⚠️ Pitfalls:**

- **Confusing context window with `max_tokens`.** Context window = input + output combined. `max_tokens` caps only the output. If your input is 100k tokens and the model has a 128k context window, you can generate at most 28k tokens.
- **Trusting the default temperature.** Most SDKs default to 1.0. Always set it explicitly for production. The number of "model is hallucinating" bugs caused by an unset temperature is shocking.
- **Believing the model "knows" things between calls.** Every API call is stateless. If you saw "remembering" in a chat UI, the caller is sending prior turns back to the model in `messages`.
- **Treating `system` as a hard contract.** Models give system messages more weight, but a determined user message can still override (prompt injection). Phase 1.2 covers defenses.
- **Underestimating tokenization quirks.** A name like "Vreddy" might be 3 tokens; the same email "vreddy@smart-structures.com" might be 12+ tokens. Localized text (Tamil, Hindi) can be 2–3× more expensive than English. Always benchmark with real data, not toy examples.

### 1.12 Defense talking points for §1

**🎯 Q: "Walk me through what happens when a user types a message in ChatGPT."**
A: "The client sends the conversation history including the new user message to the API. The server runs tokenization — typically BPE — producing a sequence of integer IDs. Those IDs index into the embedding matrix, giving one vector per token. Positional information is added — modern models use RoPE which rotates Q and K vectors by position-dependent angles. The sequence passes through stacked transformer blocks, each performing multi-head causal self-attention (softmax(QKᵀ/√d_k)·V with future-position masking) followed by an MLP, with residual connections and layer norm. The final layer projects to vocab-size logits. We apply sampling — typically temperature scaling then nucleus sampling — to pick one token. That token is appended to the context, and we repeat until a stop token or max output length. The whole conversation is reprocessed on each turn because the model is stateless."

**🎯 Q: "Why √d_k in attention?"** *[OPTIONAL — only if interviewer pushes for ML internals depth]*
A: "It's variance normalization for the softmax. Without scaling, dot products grow with the head dimension and the softmax saturates — gradients die. Honestly, this is a corner I haven't implemented; I treat the model as a system component and focus on prompts, retrieval, and evaluation."

*(The honest "I haven't implemented it" answer is fine. AI Engineer interviewers respect calibrated knowledge over fake depth.)*

**🎯 Q: "How does temperature affect output?"**
A: "Temperature scales the logits before softmax — softmax_T(x_i) = exp(x_i/T) / Σ exp(x_j/T). T < 1 sharpens the distribution toward the top tokens, making output more deterministic. T > 1 flattens it, making output more diverse. T → 0 approaches argmax (greedy). For structured output like JSON or summaries I use 0.1–0.3; for creative writing 0.7–1.0."

**🎯 Q: "Audia's chat doesn't remember earlier turns. How would you fix that?"**
A: "Since the model is stateless, memory has to be implemented by the caller. Three approaches with different trade-offs: rolling buffer (include the last N turns verbatim — simple, bounded cost, loses old context); summary buffer (LLM-summarize older turns to compress them — more complex, lossy but bounded); vector memory (embed past turns, retrieve relevant ones per query — best for long-running assistants, adds latency and infrastructure). For Audia's transcript chat, I'd start with rolling buffer of last 10 turns since most chats are short, and only escalate to summary buffer if conversations regularly exceed that."

---

## §2. The model API surface — sampling params, streaming, cost math

### 2.1 Sampling parameters — the production knobs

Every chat-completion API accepts a set of sampling parameters that shape how tokens are chosen from the model's output distribution. They're inference-time only — they don't affect the model's weights, just how we pick a token from its predicted probabilities.

| Param | Range | What it does | Set it when |
|---|---|---|---|
| `temperature` | 0–2 | Sharpens/flattens distribution | Always — universal control |
| `top_p` (nucleus) | 0–1 | Keep smallest token set with cumulative prob ≥ p | Modern default, leave at 0.9 |
| `top_k` | 1–∞ | Hard top-k filter | Rarely set; legacy from older models |
| `max_tokens` | 1–∞ | Cap output length | **Always set in production** (prevents runaway cost) |
| `stop` | string list | Halt generation when substring is produced | Chat — prevents the model from role-playing the user |
| `frequency_penalty` | 0–2 | Penalize tokens that already appeared (proportional) | Long-form creative writing — anti-repetition |
| `presence_penalty` | 0–2 | Flat penalty for any token already used | Brainstorming — push toward novel topics |
| `seed` | int | Fix RNG for reproducible sampling | Evals (Phase 6), debugging |

**The two you'll set on every production call:** `temperature` and `max_tokens`. The rest are situational.

**Practical notes for interviews:**

- **Stop sequences fix prompt leakage.** If a chat model is hallucinating fake `User: ...` turns at the end of its responses, set `stop: ["\nUser:", "\n\nHuman:"]`. Common production fix; interviewers know this.
- **`seed` is best-effort.** GPU non-determinism (parallel reductions in CUDA aren't deterministic across runs) means even with a fixed seed you may see slightly different outputs. Expect "very close," not bit-identical.
- **Combining params:** in practice, `temperature` + `top_p` is the universal default. `top_k` was common in older models (GPT-2 era) but has fallen out of favor — nucleus sampling adapts better to varying confidence.
- **`max_tokens` is output-only.** It does NOT control input length. Context window = input + output combined.

### 2.2 Streaming — the autoregressive loop made visible

**This is not a feature bolted on top of LLMs — it's a direct expression of how they work.**

The model generates tokens one at a time (autoregressive, from §1). You have two choices for the API surface:

- **No streaming:** generate all tokens server-side, buffer them, send the complete response. User sees a blank screen for 5–30 seconds, then a wall of text.
- **Streaming:** as each token is generated, push it to the client immediately. User sees text appearing at the model's actual generation rate.

**Streaming does NOT make the model faster.** Same tokens generated either way. It changes *perceived latency* — and that perceived latency is the entire difference between "delightful UX" and "feels broken."

#### Three streaming protocols you should be able to name

| Protocol | How it works | Used by |
|---|---|---|
| **Server-Sent Events (SSE)** | HTTP/1.1 standard. `data: ...\n\n` framed events. Browser consumes via `EventSource` API | OpenAI, Anthropic |
| **HTTP chunked / ReadableStream** | Plain HTTP with `Transfer-Encoding: chunked`. Raw bytes per chunk. Lower-level than SSE | Audia's chat endpoint |
| **WebSockets** | Bidirectional, full-duplex. Used when client needs to interrupt or send mid-stream | Real-time chat with voice/interrupt |

#### What streaming looks like in code

From Audia's [chat route](../src/app/api/chat/route.ts) (simplified):

```ts
const aiStream = await groq.chat.completions.create({
    messages, model, stream: true,
    stream_options: { include_usage: true },
});

for await (const chunk of aiStream) {
    const text = chunk.choices[0]?.delta?.content ?? "";
    if (text) controller.enqueue(encoder.encode(text));
    if (chunk.usage) usage = chunk.usage;  // last chunk carries usage
}
```

`aiStream` is an **async iterator**. Each iteration is "the model has produced more tokens since last time" — `delta.content` is the incremental text. Forward it to the client and the user sees it appear in real time.

**`stream_options: { include_usage: true }`** is the magic flag. Without it, usage data is omitted from streaming responses. With it, the final chunk has `choices: []` and a `usage` object. You must check `chunk.usage` on every iteration because you don't know which chunk is last.

**🎯 Defense talking point — interviewer asks: "How does streaming work?"**
A: "The model is generating tokens one at a time anyway — streaming just sends each one to the client as it's produced, instead of buffering until done. The provider returns an async iterator; you consume it in a loop and forward each delta to the client over an open HTTP connection (chunked encoding or SSE). It doesn't change cost or generation speed — only perceived latency. Users will tolerate a 20-second response if they see it appearing; they won't tolerate a 3-second blank screen."

### 2.3 Cost math — the formula and why output costs more

The universal formula:

```
total_cost = (input_tokens × input_rate) + (output_tokens × output_rate)
```

Rates are quoted as **dollars per million tokens**. Two rates per model — input and output.

#### Why output costs 2–5× input

- **Input** is processed in **one forward pass** — all tokens in parallel. GPU loves this; throughput is high.
- **Output** is **N forward passes** — one per generated token, serial. GPU is underutilized; throughput drops.
- Same compute per pass, but output forces serial work. Providers price this throughput hit into the output rate.

**📐 Sample rates (May 2026, USD per 1M tokens):**

| Model | Input | Output | Notes |
|---|---|---|---|
| Groq `llama-3.1-8b-instant` | $0.05 | $0.08 | Audia's pick — basically free |
| Gemini 2.0 Flash | $0.075 | $0.30 | Free tier with rate limits |
| GPT-4o-mini | $0.15 | $0.60 | OpenAI's cheap tier |
| Claude Sonnet 4 | $3 | $15 | Strong reasoning, much pricier |
| Claude Opus 4.7 | $15 | $75 | Frontier model |

#### Audia's economics

A 5-minute meeting transcript ≈ 1,000 tokens. A summary call:
- Input: 1,000 × $0.05 / 1,000,000 = **$0.00005**
- Output: 200 × $0.08 / 1,000,000 = **$0.000016**
- **Total: ~$0.00007 per summary**

At 1,000 summaries/day → $0.07/day. At 1,000,000 summaries/day → $70/day. **Cheap by AI standards.**

#### Three interview-worthy cost implications

1. **Long contexts scale cost linearly with input length.** A 50k-token input is 50× more expensive than 1k input. This is the #1 reason RAG exists — instead of stuffing everything into context, retrieve only what's relevant. Phase 4.
2. **Streaming doesn't change cost** — only perceived latency. Same token count either way.
3. **Prompt caching** (Anthropic, OpenAI, recently Groq) — if your prompt prefix is identical across calls (system prompt, retrieved docs), providers cache the KV-state of those tokens and charge ~10× less for re-reads. Real production optimization. Phase 12.

### 2.4 What this means in Audia

We added [src/lib/ai-usage.ts](../src/lib/ai-usage.ts) — a pricing table + cost calculator + structured logger. Every call to Groq now logs:

```
[ai-usage] summarize model=llama-3.1-8b-instant in=523 out=87 latency=843ms cost=$0.0000095
```

This instrumentation is what Phase 6 (evals) and Phase 12 (AI Ops) will build on. The cost number is dollars; the latency is wall-clock; the token counts ground tokenization in reality (you can now SEE that "Vreddy" really does cost more than you'd think for foreign names).

**Why this matters for the curriculum:** by Phase 12 we'll have a full observability stack — Langfuse or Helicone style. Today's logger is the seed of that stack. Every production AI system has equivalent telemetry; you'll never ship a customer-facing LLM call without it.

### 2.5 Common pitfalls

**⚠️ Pitfalls:**

- **Forgetting `stream_options: { include_usage: true }`** — usage data silently omitted from streaming responses. You'll think the SDK is broken. It isn't; you need the flag.
- **Setting `max_tokens` too low for the task.** A summary truncated at 50 tokens looks like a model bug; it's just `max_tokens` cutoff.
- **Believing `seed` gives bit-identical reproducibility.** It doesn't, due to GPU non-determinism. Useful for "mostly the same" outputs in evals; not for hash-equality.
- **Thinking streaming saves money.** It saves perceived latency, nothing else. Cost is identical.
- **Counting words instead of tokens for cost estimation.** Use a tokenizer (tiktokenizer.vercel.app) or log actual usage. Word count ≠ token count, especially for code, names, and non-English text.

### 2.6 Defense talking points for §2

**🎯 Q: "How would you keep your model spend under control in production?"**
A: "Four levers, in order of effort: (1) always set `max_tokens` so a runaway call has a known ceiling; (2) measure per-call cost and latency — instrument every call with a structured logger so I can see who's expensive; (3) use prompt caching for stable prefixes — Anthropic and OpenAI both support this and it's a ~10× saving on the cached portion; (4) for retrieval-heavy use cases, switch from 'stuff everything in context' to RAG so input length stops scaling with corpus size."

**🎯 Q: "Why is streaming worth the engineering complexity?"**
A: "It's not extra complexity — it's actually simpler than buffering, because the model already produces tokens serially. You just forward each delta to the client over a persistent HTTP connection. The win is purely UX: users tolerate a 20-second streaming response but not a 3-second blank screen. For any non-trivial output length, streaming is the default."

**🎯 Q: "What params do you set on every production call and why?"**
A: "Temperature — controls determinism, set per task (0.1–0.3 for structured output, 0.7+ for creative). Max_tokens — bounds cost ceiling, set to roughly 1.5× the expected output length. Stop sequences — for chat, prevents the model from hallucinating fake user turns. Stream_options.include_usage — so I can log cost per call. Seed — only for evals where I need reproducibility."

---

*(Sections 3–26 will be filled in as we progress through phases.)*

---

## Appendix A. Glossary

*(Alphabetical. Grows with each session.)*

- **Async iterator** — JavaScript pattern (`for await ... of`) for consuming streams. Each iteration yields one chunk. Used by Groq/OpenAI SDKs to expose streamed responses.
- **Autoregressive** — a generation process where each output depends on previous outputs. Token n+1 is sampled from a distribution conditioned on tokens 1..n. All chat LLMs are autoregressive.
- **Attention** — the mechanism that lets each token's representation be computed as a weighted average of other tokens' representations, weighted by learned relevance (`softmax(QKᵀ/√d_k)·V`).
- **BPE (Byte-Pair Encoding)** — sub-word tokenization algorithm that learns a vocabulary of byte sequences from a corpus by repeatedly merging the most frequent pair of adjacent symbols.
- **Causal mask** — a triangular matrix added to attention scores before softmax that sets future positions to −∞, ensuring each token can only attend to itself and prior tokens.
- **Context window** — the maximum number of tokens (input + output combined) the model can process in a single forward pass. Llama 3.1: 128k. Claude: 200k+.
- **d_model** — the dimensionality of token embeddings throughout the network. Llama 3 8B: 4096.
- **d_k** — the dimensionality of each attention head's key/query vectors. `d_model / num_heads`.
- **Embedding** — a dense vector representation of a token (or text chunk in RAG contexts). Tokens with similar usage end up with geometrically nearby embeddings.
- **Frequency penalty** — sampling parameter that reduces the logit of each token proportionally to how often it has already appeared. Reduces repetition in long-form output. Range 0–2.
- **Greedy decoding** — always picking the argmax of the logits. Deterministic but often repetitive.
- **KV cache** — at inference, the previously-computed Key and Value vectors are cached so generating each new token only requires one new attention computation rather than recomputing for the whole sequence.
- **Logits** — the model's final-layer output before softmax. Unnormalized scores over the vocabulary.
- **`max_tokens`** — output length cap. Counts only generated tokens, not input. Always set in production to bound cost.
- **MLP (in transformer)** — the two-layer feedforward block inside each transformer layer. Most of the parameters live here.
- **Multi-head attention** — running H parallel attention operations with different learned projections, then concatenating the outputs.
- **Nucleus sampling (top-p)** — keep the smallest set of tokens whose cumulative probability exceeds p, sample from that. p=0.9 is common.
- **Presence penalty** — flat logit penalty for any token that has already appeared. Encourages topic diversity. Range 0–2.
- **Prompt caching** — provider feature where stable prompt prefixes are cached server-side, charging ~10× less for the cached portion on subsequent calls. Real production optimization.
- **ReadableStream** — Web API for emitting bytes incrementally to an HTTP response. Used by Audia's chat route to forward LLM tokens as they arrive.
- **RoPE (Rotary Position Embedding)** — modern positional encoding that rotates Q and K vectors by position-dependent angles, making attention sensitive to relative position.
- **Sampling** — the process of choosing one token from the logits distribution. Combines temperature scaling and/or top-p/top-k filtering.
- **Seed** — integer that fixes the RNG used during sampling. Best-effort reproducibility (GPU non-determinism prevents bit-identical guarantees). Useful for evals.
- **Softmax** — function that converts a vector of real numbers to a probability distribution: `softmax(x_i) = exp(x_i) / Σ exp(x_j)`.
- **SSE (Server-Sent Events)** — HTTP/1.1 streaming protocol where each event is a `data: ...\n\n` framed chunk. Browser API: `EventSource`. Standard for OpenAI/Anthropic streaming responses.
- **Stateless** — the model retains no information between API calls. All context must be in the prompt.
- **Stop sequence** — string(s) which, when generated, halt the model. Provider-side mechanism. Use `["\nUser:", "\n\nHuman:"]` for chat to prevent the model from role-playing the user.
- **Stop token** — a special token (e.g. `<|endoftext|>`) that the model emits when it judges generation complete. Different from a stop *sequence*.
- **Streaming** — sending each generated token to the client as it's produced. Does not change cost or generation speed — only perceived latency.
- **`stream_options.include_usage`** — flag required to receive token usage data in streamed responses. Without it, usage is omitted; with it, the final chunk carries a `usage` field.
- **System message** — the role-tagged prompt segment containing instructions, persona, and constraints. Models are trained to weight system content as operator intent.
- **Temperature** — a scalar that divides logits before softmax. <1 sharpens, >1 flattens. Controls output randomness.
- **Top-k sampling** — keep only the k highest-probability tokens, zero the rest, renormalize.
- **Tokenization** — converting a text string into a sequence of integer token IDs using a fixed vocabulary.
- **Usage object** — provider response field containing `{ prompt_tokens, completion_tokens, total_tokens }`. Source of truth for billing and instrumentation.

---

## Appendix B. Architecture diagrams

*(Will accumulate as we build features in Audia. Each phase adds the system-design-whiteboard view of what we built.)*

---

## Appendix C. "Defend your Audia"

*(System design talking points for interview rounds. Each entry: scenario · architecture choice · trade-offs · how to defend if questioned.)*

**Coming after Phase 4 (full RAG):**
- The chat-with-transcript pipeline
- Why pgvector and not Pinecone

**Coming after Phase 7 (agents):**
- Cross-session agent architecture
- Why no LangChain
