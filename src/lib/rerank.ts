import { cosineSimilarity } from "@/lib/vector-math";

/**
 * Maximal Marginal Relevance (Carbonell & Goldstein 1998).
 *
 * ─────────────────────────────────────────────────────────────────────────
 * THE CORE IDEA — read this once before reading the code.
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Suppose vector search gave you 20 candidate chunks ranked by similarity to
 * a question. Naive top-k just takes the first 5. Problem: those 5 might all
 * be near-duplicates of each other — say, 5 paraphrases from a single moment
 * in the meeting where someone repeated themselves. You've used 5 slots in
 * the LLM context budget for the same idea.
 *
 * MMR fixes this by picking iteratively, one item per round. Each round, for
 * every remaining candidate, it computes a score that:
 *   - REWARDS being relevant to the query
 *   - PENALIZES being too similar to whatever we've already picked
 *
 *   score(candidate) =  λ · relevance_to_query
 *                       − (1 − λ) · max_similarity_to_already_picked
 *
 *                       ↑                         ↑
 *                       "close to query is good"  "close to stuff I already
 *                                                  picked is BAD — I'd be
 *                                                  picking a duplicate"
 *
 * λ ("lambda") is the knob in [0, 1] controlling the trade-off:
 *   λ = 1.0   pure relevance  (collapses to top-k by similarity)
 *   λ = 0.7   relevance-leaning  (production default for Audia)
 *   λ = 0.5   balanced
 *   λ = 0.0   pure diversity  (ignores query entirely — almost never useful)
 *
 * Cost: O(N × k) cosine ops. For N=20, k=5, that's ~100 ops — negligible.
 *
 * @param queryEmbedding  The query vector (from `embed(question)`)
 * @param candidates      Items with a `.embedding` field; typically the wider
 *                        top-N from vector search (e.g. N=20 if final k=5)
 * @param k               How many items to return
 * @param lambda          Relevance vs. diversity knob in [0, 1]; default 0.7
 */
export function maximalMarginalRelevance<T extends { embedding: number[] }>(
    queryEmbedding: number[],
    candidates: T[],
    k: number,
    lambda = 0.7,
): T[] {
    // ── Edge cases ───────────────────────────────────────────────────────
    // Nothing to pick from, or caller asked for zero items.
    if (candidates.length === 0 || k === 0) return [];

    // Optimization: if we have at most k candidates AND λ ≥ 1 (pure
    // relevance — no diversity penalty), the algorithm would just return
    // everything in the order it came (already sorted by relevance from the
    // upstream vector search). Skip the work.
    if (candidates.length <= k && lambda >= 1) return candidates.slice(0, k);

    // ── Step 1: precompute candidate-to-query similarities ───────────────
    // Every candidate's relevance to the query is reused MANY times below
    // (once per candidate per round of picking). Computing it once up front
    // avoids redundant work in the loops.
    //
    // queryRelevance[i] = cosine similarity between candidate i and the query.
    const queryRelevance = candidates.map((c) => cosineSimilarity(queryEmbedding, c.embedding));

    // The output list we'll build up, one pick per round.
    const selected: T[] = [];

    // Track WHICH candidate indices have already been picked, by their
    // position in the input `candidates` array. We use a Set rather than
    // mutating `candidates` because:
    //   - Set.has() is O(1) — fast "have we picked this?" check
    //   - The function stays pure (no mutation of caller's data)
    const selectedIndices = new Set<number>();

    // ── Step 2: the first pick is special ────────────────────────────────
    // Nothing in `selected` yet, so the "max similarity to already-selected"
    // term is undefined. The diversity penalty doesn't exist on the first
    // round, so we just pick the most relevant candidate — equivalent to
    // top-1 of pure top-k retrieval.
    let firstIdx = 0;
    for (let i = 1; i < candidates.length; i++) {
        if (queryRelevance[i] > queryRelevance[firstIdx]) firstIdx = i;
    }
    selected.push(candidates[firstIdx]);
    selectedIndices.add(firstIdx);

    // ── Step 3: each subsequent pick ─────────────────────────────────────
    // Repeat until we have k items, OR we've picked every candidate (which
    // can happen if k > candidates.length).
    while (selected.length < k && selectedIndices.size < candidates.length) {
        // Track this round's best candidate as we scan all remaining ones.
        let bestIdx = -1;
        let bestScore = -Infinity;

        // Scan every candidate to find the one with the highest MMR score.
        // (We can't precompute MMR scores once because the diversity term
        // DEPENDS on what's already in `selected`, which changes each round.
        // The relevance term IS precomputed — see queryRelevance above.)
        for (let i = 0; i < candidates.length; i++) {
            // Skip candidates we've already picked.
            if (selectedIndices.has(i)) continue;

            // First half of MMR's score: how relevant is THIS candidate to
            // the query? We computed it in step 1 — just look it up.
            const relevance = queryRelevance[i];

            // Second half: how similar is this candidate to the MOST similar
            // item we've ALREADY picked?
            //
            // KEY DETAIL: this is the MAX, not the AVERAGE. Why max? Because
            // if ANY single already-picked item is very similar, this
            // candidate is effectively a duplicate of THAT item — even if
            // it's different from the others. We want to detect "have I
            // already covered this idea?" — and one near-duplicate is enough
            // to say yes.
            //
            // Initialize at -Infinity so the first comparison always wins.
            let maxSimToSelected = -Infinity;
            for (const s of selected) {
                const sim = cosineSimilarity(candidates[i].embedding, s.embedding);
                if (sim > maxSimToSelected) maxSimToSelected = sim;
            }

            // The MMR score. Plain reading:
            //   "reward relevance, penalize redundancy"
            // λ controls how much each side counts. At λ = 0.7, relevance
            // is weighted 0.7 and the diversity penalty 0.3 — roughly 2.3×
            // more weight on relevance.
            const mmrScore = lambda * relevance - (1 - lambda) * maxSimToSelected;

            // Track the best so far for this round.
            if (mmrScore > bestScore) {
                bestScore = mmrScore;
                bestIdx = i;
            }
        }

        // Defensive: should be unreachable given the while-loop guards, but
        // belt-and-suspenders for future modifications that might filter
        // candidates inside the loop.
        if (bestIdx === -1) break;

        // Commit this round's winner. Now the NEXT round's diversity penalty
        // calculations will include similarity to this item too.
        selected.push(candidates[bestIdx]);
        selectedIndices.add(bestIdx);
    }

    // We've picked k items (or all available, whichever came first).
    return selected;
}

/**
 * Reciprocal Rank Fusion (Cormack et al. 2009) — the fusion step of hybrid
 * search (Phase 8.2).
 *
 * ─────────────────────────────────────────────────────────────────────────
 * THE PROBLEM IT SOLVES.
 * ─────────────────────────────────────────────────────────────────────────
 * Dense retrieval scores are cosine similarities (~0..1); lexical (BM25 /
 * ts_rank) scores are unbounded and corpus-dependent. They live on
 * INCOMPARABLE scales, so you can't just add them. Normalizing (min-max,
 * z-score) is fiddly and distribution-sensitive.
 *
 * RRF sidesteps all of that by fusing on RANK, not score:
 *
 *     RRF(d) = Σ over lists  1 / (k + rank_list(d))            rank is 1-based
 *
 * Properties:
 *   - SCALE-INVARIANT: only a document's POSITION in each list matters, so the
 *     two systems' raw score magnitudes are irrelevant. (This is also why our
 *     lexical arm doesn't need true BM25 — ts_rank_cd's ordering is all we use.)
 *   - NO TUNING: k=60 is the field default and "just works"; the lists don't
 *     even have to be related.
 *   - REWARDS AGREEMENT: a doc ranked high in BOTH lists outscores one that's
 *     #1 in only one. The k constant flattens the head so rank-1 doesn't
 *     dominate (rank 1 → 1/61 ≈ 0.0164; rank 2 → 1/62 ≈ 0.0161).
 *
 * @param lists  named ranked lists (each already sorted best-first)
 * @param keyFn  stable identity for an item (so the same chunk across lists fuses)
 * @param k      smoothing constant; 60 is the standard default
 */
export type Fused<T> = T & {
    rrfScore: number;
    /** Which input lists contributed this item, e.g. ["semantic", "keyword"]. */
    sources: string[];
};

export function reciprocalRankFusion<T>(
    lists: { name: string; items: T[] }[],
    keyFn: (item: T) => string,
    k = 60,
): Fused<T>[] {
    const acc = new Map<string, { item: T; score: number; sources: Set<string> }>();

    for (const { name, items } of lists) {
        items.forEach((item, i) => {
            const key = keyFn(item);
            const rank = i + 1; // 1-based: the top item is rank 1, not 0
            const contribution = 1 / (k + rank);
            const existing = acc.get(key);
            if (existing) {
                existing.score += contribution;
                existing.sources.add(name);
            } else {
                acc.set(key, { item, score: contribution, sources: new Set([name]) });
            }
        });
    }

    return [...acc.values()]
        .map((e) => ({ ...e.item, rrfScore: e.score, sources: [...e.sources] }))
        .sort((a, b) => b.rrfScore - a.rrfScore);
}
