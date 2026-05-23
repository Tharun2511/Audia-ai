import { embed } from "@/lib/embeddings";
import { cosineSimilarity, dotProduct, norm } from "@/lib/vector-math";

const SENTENCES = [
    "The cat sat on the mat.",
    "A feline rested on a rug.",
    "Interest rates rose two percent.",
    "The central bank hiked borrowing costs.",
    "I love pizza.",
];

export async function GET() {
    const vectors = await Promise.all(SENTENCES.map((s) => embed(s)));

    const normChecks = vectors.map((v, i) => ({ i, sentence: SENTENCES[i], norm: norm(v).toFixed(4) }));

    const pairs: Array<{ a: string; b: string; cosine: string; dot: string }> = [];
    for (let i = 0; i < SENTENCES.length; i++) {
        for (let j = i + 1; j < SENTENCES.length; j++) {
            pairs.push({
                a: SENTENCES[i],
                b: SENTENCES[j],
                cosine: cosineSimilarity(vectors[i], vectors[j]).toFixed(4),
                dot: dotProduct(vectors[i], vectors[j]).toFixed(4),
            });
        }
    }

    pairs.sort((a, b) => Number(b.cosine) - Number(a.cosine));

    return Response.json({
        dimensions: vectors[0].length,
        normChecks,
        pairsRankedBySimilarity: pairs,
    });
}
