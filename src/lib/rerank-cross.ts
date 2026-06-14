import "server-only";

/**
 * Cross-encoder re-ranking — Stage 2 of the retrieve-and-rerank pipeline (Phase 8.1).
 *
 * ─────────────────────────────────────────────────────────────────────────
 * BI-ENCODER vs CROSS-ENCODER — read once before the code.
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Everything in chunks.ts is a BI-ENCODER: the query and each chunk are
 * embedded SEPARATELY into 768-dim vectors, then compared by cosine
 * (pgvector's <=>). It's fast (vectors precomputed, ANN over the whole corpus)
 * but lossy — the query and the document never "see" each other, so subtle
 * signals (negation, directionality, exact-phrase match) get flattened.
 *
 * A CROSS-ENCODER feeds the query AND one document TOGETHER through a
 * transformer in a single pass, letting their tokens attend to each other, and
 * emits ONE relevance score. Far more precise — but nothing can be
 * precomputed (the score depends on the query), so it costs one forward pass
 * PER candidate. You can't run it over millions of chunks.
 *
 * Hence the two-stage funnel:
 *   Stage 1 (bi-encoder, findCandidateChunks): millions → coarse top-N   FAST, recall
 *   Stage 2 (cross-encoder, THIS file):        N → precise top-k          SLOW, precision
 * The cross-encoder is affordable ONLY because stage 1 already narrowed the field.
 *
 * Provider: Jina Reranker API (a purpose-built cross-encoder, free tier).
 * It returns `relevance_score` already normalized to [0, 1] via sigmoid — unlike
 * a raw BGE cross-encoder, which emits raw logits you'd have to sigmoid yourself.
 * We rank WITHIN a query by this score; it is not comparable across queries.
 *
 * Graceful degradation: if JINA_API_KEY is missing or the API errors, we fall
 * back to the bi-encoder order (stage-1 ranking). Search still works — it just
 * loses the precision lift. A dead reranker must never take down search.
 */

const JINA_RERANK_URL = "https://api.jina.ai/v1/rerank";

// jina-reranker-v2-base-multilingual: a cross-encoder supporting 100+ languages.
// Good default for free-tier reranking; English-only models exist too.
const RERANK_MODEL = "jina-reranker-v2-base-multilingual";

/** A candidate with its cross-encoder relevance score attached. */
export type Reranked<T> = T & {
    /** Cross-encoder relevance in [0, 1]. NaN when the reranker was unavailable. */
    relevanceScore: number;
};

/**
 * Re-rank `candidates` against `query` with a cross-encoder, returning the
 * top-`topK` by relevance. The input is assumed to be the coarse top-N from a
 * bi-encoder retrieval (e.g. findCandidateChunks).
 *
 * @param query       the user's search query
 * @param candidates  coarse top-N candidates; each must expose `.text`
 * @param topK        how many to keep after re-ranking
 */
export async function crossEncoderRerank<T extends { text: string }>(
    query: string,
    candidates: T[],
    topK: number,
): Promise<Reranked<T>[]> {
    if (candidates.length === 0 || topK === 0) return [];

    const apiKey = process.env.JINA_API_KEY;

    // No key configured → degrade to bi-encoder order. Score is NaN so the UI
    // can tell "not reranked" apart from a genuine 0.0 relevance.
    if (!apiKey) {
        console.warn("[rerank-cross] JINA_API_KEY not set — falling back to bi-encoder order");
        return fallback(candidates, topK);
    }

    try {
        const res = await fetch(JINA_RERANK_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${apiKey}`,
            },
            body: JSON.stringify({
                model: RERANK_MODEL,
                query,
                documents: candidates.map((c) => c.text),
                top_n: topK,
                // We already hold the documents in `candidates`; map results back
                // by index instead of paying to echo the texts over the wire.
                return_documents: false,
            }),
        });

        if (!res.ok) {
            const body = await res.text();
            console.warn(`[rerank-cross] Jina rerank failed: ${res.status} ${body} — falling back`);
            return fallback(candidates, topK);
        }

        const data = (await res.json()) as {
            // Jina returns hits sorted by relevance, each referencing the input
            // document by its ORIGINAL index in `documents`.
            results?: Array<{ index: number; relevance_score: number }>;
        };

        const results = data.results ?? [];
        if (results.length === 0) {
            console.warn("[rerank-cross] Jina returned no results — falling back");
            return fallback(candidates, topK);
        }

        // Map Jina's index-referenced, relevance-sorted results back onto our
        // candidate objects, preserving the cross-encoder ordering.
        return results
            .filter((r) => Number.isInteger(r.index) && r.index >= 0 && r.index < candidates.length)
            .map((r) => ({ ...candidates[r.index], relevanceScore: r.relevance_score }));
    } catch (err) {
        console.warn("[rerank-cross] Jina rerank threw — falling back to bi-encoder order", err);
        return fallback(candidates, topK);
    }
}

/**
 * Bi-encoder-order fallback: keep the candidates' incoming order (already
 * sorted by cosine from stage 1), tag them with NaN so callers/UI can show
 * "reranking unavailable" rather than a fake score.
 */
function fallback<T>(candidates: T[], topK: number): Reranked<T>[] {
    return candidates.slice(0, topK).map((c) => ({ ...c, relevanceScore: NaN }));
}
