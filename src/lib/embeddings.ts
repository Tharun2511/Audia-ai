import "server-only";

// text-embedding-004 was deprecated; current model is gemini-embedding-2 (multimodal,
// flexible output dim 128-3072 with 768/1536/3072 recommended).
const GEMINI_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent";
const EMBED_DIM = 768;

type EmbedResponse = { embedding: { values: number[] } };

export async function embed(text: string): Promise<number[]> {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) throw new Error("GEMINI_API_KEY is not set");

    const res = await fetch(`${GEMINI_EMBED_URL}?key=${apiKey}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            content: { parts: [{ text }] },
            outputDimensionality: EMBED_DIM,
        }),
    });

    if (!res.ok) {
        const body = await res.text();
        throw new Error(`Gemini embed failed: ${res.status} ${body}`);
    }

    const data = (await res.json()) as EmbedResponse;
    const vec = data.embedding?.values;
    if (!Array.isArray(vec) || vec.length !== EMBED_DIM) {
        throw new Error(`Unexpected embedding shape: length=${vec?.length}`);
    }
    return vec;
}

export const EMBEDDING_DIM = EMBED_DIM;
