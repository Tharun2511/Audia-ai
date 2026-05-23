"use client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Box from "@mui/material/Box";
import Chip from "@mui/material/Chip";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import type { TranscriptSegment } from "@/entity/Transcription";
import { formatDuration } from "./utils";
import AudioPlayer from "./AudioPlayer";
import SummaryBlock from "./SummaryBlock";
import TitleInput from "./TitleInput";
import TranscriptPanel from "./TranscriptPanel";
import ChatPanel from "./ChatPanel";

interface Props {
    id: string;
    title: string | null;
    summary: string | null;
    duration: number;
    segments: TranscriptSegment[];
    audioPathname: string | null;
    createdAt: string;
    isJustCompleted?: boolean;
    onTitleSave: (title: string | null) => void;
    onSegmentsUpdate: (segments: TranscriptSegment[]) => void;
}

/**
 * Returns the index of the segment that contains `currentTime`, or -1 if none.
 * "Contains" = the segment with the latest start <= currentTime. We walk from
 * the end so the common case (recent segments during forward playback) is O(1).
 */
function findActiveSegmentIndex(segments: TranscriptSegment[], currentTime: number): number {
    for (let i = segments.length - 1; i >= 0; i--) {
        if (segments[i].start <= currentTime) return i;
    }
    return -1;
}

export default function SessionView({
    id,
    title,
    summary,
    duration,
    segments,
    audioPathname,
    createdAt,
    isJustCompleted,
    onTitleSave,
    onSegmentsUpdate,
}: Props) {
    const date = new Date(createdAt);
    const audioRef = useRef<HTMLAudioElement>(null);
    const [currentTime, setCurrentTime] = useState(0);

    /**
     * Signed playback URL — null while loading or when audio isn't available.
     * Fetched from the auth-gated /audio-url endpoint each time `id` changes,
     * and re-fetched on <audio> onError to recover from URL expiry.
     */
    const [signedAudioUrl, setSignedAudioUrl] = useState<string | null>(null);
    const audioRetryRef = useRef(false);

    const fetchSignedUrl = useCallback(async (): Promise<string | null> => {
        try {
            const res = await fetch(`/api/transcriptions/${id}/audio-url`);
            if (!res.ok) return null;
            const data = await res.json();
            return data.url ?? null;
        } catch {
            return null;
        }
    }, [id]);

    // When the session changes (or its audio appears for the first time),
    // fetch a fresh signed URL. The `cancelled` flag prevents a stale fetch
    // from a previously-selected session landing in this session's <audio>.
    useEffect(() => {
        if (!audioPathname) {
            setSignedAudioUrl(null);
            return;
        }
        let cancelled = false;
        audioRetryRef.current = false;
        fetchSignedUrl().then((url) => {
            if (!cancelled) setSignedAudioUrl(url);
        });
        return () => { cancelled = true; };
    }, [audioPathname, fetchSignedUrl]);

    // Recomputes ~4x/sec while audio plays, but the rendered child only
    // re-renders when the *index* changes (typically once per segment), not on
    // every timeupdate tick.
    const activeSegmentIndex = useMemo(
        () => findActiveSegmentIndex(segments, currentTime),
        [segments, currentTime]
    );

    const seekTo = useCallback((startSeconds: number) => {
        const audio = audioRef.current;
        if (!audio) return;
        audio.currentTime = startSeconds;
        // .play() can reject if the user hasn't interacted with the page yet
        // (autoplay policy). The seek itself succeeds; we silently ignore the
        // play-rejection because the user can hit the play button themselves.
        if (audio.paused) audio.play().catch(() => { });
    }, []);

    return (
        <Stack spacing={2.5} sx={{ animation: "fade-in 250ms ease both" }}>
            {/* Page-level header — title + meta. Renaming here renames the whole session. */}
            <Box>
                <TitleInput
                    transcriptionId={id}
                    initialTitle={title}
                    fallback="Untitled session"
                    onSaved={onTitleSave}
                    variant="title"
                />
                <Stack direction="row" spacing={1.5} sx={{ alignItems: "center", flexWrap: "wrap", mt: 1.25 }}>
                    <Typography variant="caption" sx={{ fontSize: 11, color: "text.disabled", fontFamily: "var(--font-geist-mono), monospace" }}>
                        {formatDuration(duration)}
                    </Typography>
                    <Typography variant="caption" sx={{ fontSize: 11, color: "text.disabled" }}>·</Typography>
                    <Typography variant="caption" sx={{ fontSize: 11, color: "text.disabled" }}>
                        {date.toLocaleString()}
                    </Typography>
                    {isJustCompleted && (
                        <Chip
                            label="Just completed"
                            size="small"
                            sx={{ height: 18, fontSize: 10, bgcolor: "primary.main", color: "primary.contrastText", "& .MuiChip-label": { px: 1 } }}
                        />
                    )}
                </Stack>
            </Box>

            <SummaryBlock summary={summary} />

            {/*
             * The audio player is "sticky" from this position onward — when
             * you scroll into the transcript, it pins to the top of the
             * viewport so the controls + progress + active-segment context
             * stay visible while reading. This is the standard pattern for
             * any media-paired-with-text product (Otter, Loom, Descript).
             *
             * `top` accounts for the mobile AppBar (56px) which is also
             * sticky; on desktop a small gap gives breathing room from the
             * viewport edge.
             *
             * `zIndex: 2` keeps it above the transcript card shadows beneath.
             */}
            {audioPathname && (
                <Box
                    sx={{
                        position: "sticky",
                        top: { xs: 56, lg: 12 },
                        zIndex: 2,
                    }}
                >
                    <AudioPlayer
                        src={signedAudioUrl}
                        durationSeconds={duration}
                        currentTime={currentTime}
                        onTimeUpdate={setCurrentTime}
                        audioRef={audioRef}
                        onError={async () => {
                            // The signed URL likely expired (1h TTL). Retry once
                            // with a fresh URL; if that also fails, give up so we
                            // don't loop forever on a genuinely broken blob.
                            if (audioRetryRef.current) return;
                            audioRetryRef.current = true;
                            const fresh = await fetchSignedUrl();
                            if (fresh) setSignedAudioUrl(fresh);
                        }}
                    />
                </Box>
            )}

            <TranscriptPanel
                key={id}
                segments={segments}
                transcriptionId={id}
                onSegmentsUpdate={onSegmentsUpdate}
                title={title}
                summary={summary}
                duration={duration}
                createdAt={createdAt}
                activeSegmentIndex={signedAudioUrl ? activeSegmentIndex : -1}
                onSegmentSeek={signedAudioUrl ? seekTo : undefined}
            />

            {segments.length > 0 && <ChatPanel segments={segments} />}
        </Stack>
    );
}
