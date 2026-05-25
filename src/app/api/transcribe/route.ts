import { getDatabase } from "@/db/data-source";
import { Transcription } from "@/entity/Transcription";
import type { TranscriptSegment } from "@/entity/Transcription";
import { deepgram, summarizeTranscript } from "@/lib/ai";
import { uploadAudio } from "@/lib/audio-storage";
import { chunkTranscript } from "@/lib/chunking";
import { saveChunkWithEmbedding } from "@/lib/chunks";
import { getCurrentUser } from "@/lib/dal";
import { embed } from "@/lib/embeddings";

type DeepgramWord = {
    word: string;
    start: number;
    end: number;
    speaker: number;
    punctuated_word?: string;
};

function parseSegments(words: DeepgramWord[]): TranscriptSegment[] {
    if (words.length === 0) return [];

    const segments: TranscriptSegment[] = [];
    let currentSpeaker = words[0].speaker ?? 0;
    let currentWords: string[] = [];
    let segStart = words[0].start;
    let segEnd = words[0].end;

    for (const word of words) {
        const speaker = word.speaker ?? 0;
        const displayWord = word.punctuated_word ?? word.word;

        if (speaker !== currentSpeaker) {
            segments.push({
                speaker: `User${currentSpeaker + 1}`,
                text: currentWords.join(" "),
                start: segStart,
                end: segEnd,
            });
            currentSpeaker = speaker;
            currentWords = [displayWord];
            segStart = word.start;
            segEnd = word.end;
        } else {
            currentWords.push(displayWord);
            segEnd = word.end;
        }
    }

    if (currentWords.length > 0) {
        segments.push({
            speaker: `User${currentSpeaker + 1}`,
            text: currentWords.join(" "),
            start: segStart,
            end: segEnd,
        });
    }

    return segments;
}

export async function POST(req: Request) {
    const user = await getCurrentUser();
    if (!user) return Response.json({ error: "Unauthorized" }, { status: 401 });

    const formData = await req.formData();
    const audioFile = formData.get("audio") as File | null;

    if (!audioFile) {
        return Response.json({ error: "No audio file provided" }, { status: 400 });
    }

    const arrayBuffer = await audioFile.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    // Upload and transcribe in parallel: both consume the same buffer, neither
    // depends on the other. This makes audio persistence essentially free in
    // latency — the user already waits for Deepgram, and upload finishes in
    // that same window.
    const [audioPathname, result] = await Promise.all([
        uploadAudio(buffer, audioFile.type || "audio/webm"),
        deepgram.listen.v1.media.transcribeFile(buffer, {
            model: "nova-2",
            diarize: true,
            smart_format: true,
            punctuate: true,
        }),
    ]);

    const words = ((result as unknown as { results?: { channels?: { alternatives?: { words?: DeepgramWord[] }[] }[] } })?.results?.channels?.[0]?.alternatives?.[0]?.words ?? []) as DeepgramWord[];
    const segments = parseSegments(words);
    const duration = ((result as unknown as { metadata?: { duration?: number } })?.metadata?.duration) ?? 0;

    const summary = await summarizeTranscript(segments);

    const db = await getDatabase();
    const repo = db.getRepository(Transcription);
    const record = repo.create({ duration, segments, userEmail: user.email, summary, audioPathname });
    await repo.save(record);

    // Chunk + embed + persist. Parallel embed calls keep latency bounded by the
    // slowest single call, not the sum. If embedding fails, log and continue —
    // the transcription itself already saved; chunks can be backfilled later.
    try {
        const chunks = chunkTranscript(segments);
        const embeddings = await Promise.all(chunks.map((c) => embed(c.text)));
        await Promise.all(
            chunks.map((chunk, i) =>
                saveChunkWithEmbedding(chunk, record.id, user.email, i, embeddings[i]),
            ),
        );
        console.log(`[transcribe] saved ${chunks.length} chunks for ${record.id}`);
    } catch (err) {
        console.warn(`[transcribe] chunk+embed failed for ${record.id}`, err);
    }

    return Response.json({ id: record.id, segments, duration, summary, audioPathname });
}
