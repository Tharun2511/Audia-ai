import "server-only";

export type ModelPricing = { input: number; output: number };

// USD per 1 million tokens. May 2026 rates from Groq.
export const GROQ_PRICING: Record<string, ModelPricing> = {
    "llama-3.1-8b-instant": { input: 0.05, output: 0.08 },
    "llama-3.3-70b-versatile": { input: 0.59, output: 0.79 },
};

export function computeCost(model: string, promptTokens: number, completionTokens: number): number {
    const pricing = GROQ_PRICING[model];
    if (!pricing) return 0;
    return (promptTokens * pricing.input + completionTokens * pricing.output) / 1_000_000;
}

export type UsageEvent = {
    label: string;
    model: string;
    promptTokens: number;
    completionTokens: number;
    latencyMs: number;
    cost: number;
};

export function logUsage(event: UsageEvent): void {
    console.log(
        `[ai-usage] ${event.label} model=${event.model} ` +
        `in=${event.promptTokens} out=${event.completionTokens} ` +
        `latency=${event.latencyMs}ms cost=$${event.cost.toFixed(6)}`
    );
}
