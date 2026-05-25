import { chunkTranscript } from "@/lib/chunking";
import type { TranscriptSegment } from "@/entity/Transcription";

const SAMPLE_TRANSCRIPT: TranscriptSegment[] = [
    { speaker: "Alice", text: "We need to talk about the Q4 launch.", start: 0, end: 3 },
    { speaker: "Bob", text: "Sure, what's on your mind?", start: 3, end: 5 },
    { speaker: "Alice", text: "Marketing says we need a March 15 launch date.", start: 5, end: 10 },
    { speaker: "Bob", text: "Is that realistic given the engineering backlog?", start: 10, end: 14 },
    { speaker: "Alice", text: "We discussed it with engineering. They can hit it if we deprioritize the analytics feature.", start: 14, end: 22 },
    { speaker: "Bob", text: "What about the customer support tooling? That was on the critical path too.", start: 22, end: 28 },
    { speaker: "Alice", text: "We can launch with the existing tooling and add the new features in May.", start: 28, end: 36 },
    { speaker: "Carol", text: "I think we should also consider the hiring situation. We need at least two more engineers before peak season.", start: 36, end: 46 },
    { speaker: "Bob", text: "Carol's right. Let's add hiring as a parallel track. I'll talk to recruiting tomorrow.", start: 46, end: 54 },
    { speaker: "Alice", text: "Great. Action items: Bob talks to recruiting, I'll deprioritize analytics, Carol draws up the hiring plan. Let's reconvene Friday.", start: 54, end: 66 },
];

export async function GET(req: Request) {
    const url = new URL(req.url);
    const targetChars = Number(url.searchParams.get("target") ?? 400);
    const overlap = Number(url.searchParams.get("overlap") ?? 1);

    const chunks = chunkTranscript(SAMPLE_TRANSCRIPT, {
        targetChars,
        overlapSegments: overlap,
    });

    return Response.json({
        params: { targetChars, overlapSegments: overlap },
        totalSegments: SAMPLE_TRANSCRIPT.length,
        totalCharsInTranscript: SAMPLE_TRANSCRIPT.reduce(
            (sum, s) => sum + s.speaker.length + s.text.length + 2,
            0,
        ),
        chunkCount: chunks.length,
        chunks: chunks.map((c, i) => ({
            index: i,
            charCount: c.charCount,
            segmentRange: c.segmentIndices.length > 1
                ? `[${c.segmentIndices[0]}..${c.segmentIndices[c.segmentIndices.length - 1]}]`
                : `[${c.segmentIndices[0]}]`,
            timeRange: `${c.startTime.toFixed(1)}s - ${c.endTime.toFixed(1)}s`,
            speakers: c.speakers,
            text: c.text,
        })),
    });
}
