export function dotProduct(a: number[], b: number[]): number {
    if (a.length !== b.length) throw new Error(`dimension mismatch: ${a.length} vs ${b.length}`);
    let sum = 0;
    for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
    return sum;
}

export function norm(a: number[]): number {
    let sumSq = 0;
    for (let i = 0; i < a.length; i++) sumSq += a[i] * a[i];
    return Math.sqrt(sumSq);
}

export function cosineSimilarity(a: number[], b: number[]): number {
    const denom = norm(a) * norm(b);
    if (denom === 0) return 0;
    return dotProduct(a, b) / denom;
}

export function l2Distance(a: number[], b: number[]): number {
    if (a.length !== b.length) throw new Error(`dimension mismatch: ${a.length} vs ${b.length}`);
    let sumSq = 0;
    for (let i = 0; i < a.length; i++) {
        const diff = a[i] - b[i];
        sumSq += diff * diff;
    }
    return Math.sqrt(sumSq);
}
