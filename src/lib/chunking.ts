import type { TranscriptSegment } from "@/entity/Transcription";

export type TranscriptChunk = {
    text: string;              // concatenated speaker-prefixed text
    segmentIndices: number[];  // indices into source TranscriptSegment[]
    speakers: string[];        // unique speakers in this chunk
    startTime: number;         // first segment's start (seconds)
    endTime: number;           // last segment's end (seconds)
    charCount: number;         // approximate size; ~4 chars/token English
};

// Defaults tuned for conversational transcripts.
// 1200 chars ≈ 300 tokens — small enough for precise retrieval, large enough to hold one topic.
const DEFAULT_TARGET_CHARS = 1200;
const DEFAULT_OVERLAP_SEGMENTS = 1;

const formatSegment = (s: TranscriptSegment) => `${s.speaker}: ${s.text}`;

function buildChunk(segments: TranscriptSegment[], indices: number[]): TranscriptChunk {
    const included = indices.map((i) => segments[i]);
    const text = included.map(formatSegment).join("\n");
    return {
        text,
        segmentIndices: indices.slice(),
        speakers: Array.from(new Set(included.map((s) => s.speaker))),
        startTime: included[0].start,
        endTime: included[included.length - 1].end,
        charCount: text.length,
    };
}

export function chunkTranscript(
    segments: TranscriptSegment[],
    opts: { targetChars?: number; overlapSegments?: number } = {},
): TranscriptChunk[] {
    if (segments.length === 0) return [];

    const targetChars = opts.targetChars ?? DEFAULT_TARGET_CHARS;
    const overlapSegments = opts.overlapSegments ?? DEFAULT_OVERLAP_SEGMENTS;

    const chunks: TranscriptChunk[] = [];
    let bufferIndices: number[] = [];
    let bufferChars = 0;

    for (let i = 0; i < segments.length; i++) {
        bufferIndices.push(i);
        bufferChars += formatSegment(segments[i]).length + 1; // +1 for joining newline

        const isLast = i === segments.length - 1;
        if (bufferChars >= targetChars || isLast) {
            chunks.push(buildChunk(segments, bufferIndices));

            if (!isLast) {
                // Carry over the last N segments into the next chunk as overlap
                const overlapStart = Math.max(0, bufferIndices.length - overlapSegments);
                bufferIndices = bufferIndices.slice(overlapStart);
                bufferChars = bufferIndices.reduce(
                    (sum, idx) => sum + formatSegment(segments[idx]).length + 1,
                    0,
                );
            }
        }
    }

    return chunks;
}
