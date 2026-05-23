"use client";
import { useState } from "react";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import CircularProgress from "@mui/material/CircularProgress";
import IconButton from "@mui/material/IconButton";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import ReplayIcon from "@mui/icons-material/Replay";
import { formatDuration } from "./utils";

/**
 * Cycle order is "most-common first" — most users want 1× or faster for
 * skimming, with 0.5× as the fallback for re-listening hard sections.
 */
const PLAYBACK_SPEEDS = [1, 1.25, 1.5, 2, 0.5] as const;

interface Props {
    /** Signed playback URL. `null` triggers the loading skeleton. */
    src: string | null;
    /**
     * Authoritative duration in seconds. Comes from Deepgram, NOT the
     * <audio>.duration property — which returns Infinity for WebM/Opus
     * recordings produced by MediaRecorder (no Cues block). Trusting the
     * browser here is what broke the progress bar before.
     */
    durationSeconds: number;
    /** Current playback time (lifted to parent for transcript highlighting). */
    currentTime: number;
    /** Reports back from <audio> timeupdate/seeked events. */
    onTimeUpdate: (time: number) => void;
    /** Imperative handle so the parent can seek when a transcript line is clicked. */
    audioRef: React.RefObject<HTMLAudioElement | null>;
    /** Fired on <audio> error — parent uses this to refetch an expired signed URL. */
    onError?: () => void;
}

export default function AudioPlayer({
    src,
    durationSeconds,
    currentTime,
    onTimeUpdate,
    audioRef,
    onError,
}: Props) {
    const [playing, setPlaying] = useState(false);
    const [ended, setEnded] = useState(false);
    const [speedIndex, setSpeedIndex] = useState(0);
    const playbackRate = PLAYBACK_SPEEDS[speedIndex];

    const togglePlay = () => {
        const audio = audioRef.current;
        if (!audio) return;
        if (audio.paused) {
            // If playback finished, restart from the top instead of failing silently.
            if (ended || audio.currentTime >= durationSeconds - 0.05) {
                audio.currentTime = 0;
                setEnded(false);
            }
            audio.play().catch(() => { /* autoplay blocked — user can retry */ });
        } else {
            audio.pause();
        }
    };

    const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
        const audio = audioRef.current;
        if (!audio || durationSeconds <= 0) return;
        const rect = e.currentTarget.getBoundingClientRect();
        const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        audio.currentTime = pct * durationSeconds;
        setEnded(false);
    };

    const cycleSpeed = () => {
        const next = (speedIndex + 1) % PLAYBACK_SPEEDS.length;
        setSpeedIndex(next);
        const audio = audioRef.current;
        if (audio) audio.playbackRate = PLAYBACK_SPEEDS[next];
    };

    const progressPct = durationSeconds > 0
        ? Math.min(100, (currentTime / durationSeconds) * 100)
        : 0;

    // ── Loading state ──────────────────────────────────────────────
    // Same outer shape as the loaded state so there's no layout jump when
    // src arrives. A small spinner + explicit text — the user reported
    // confusing the missing player for an upload failure.
    if (!src) {
        return (
            <Card sx={{ px: 2, py: 1.5 }}>
                <Stack direction="row" spacing={1.5} sx={{ alignItems: "center", minHeight: 40 }}>
                    <CircularProgress size={18} thickness={5} sx={{ color: "primary.main" }} />
                    <Typography variant="body2" sx={{ color: "text.secondary" }}>
                        Loading audio…
                    </Typography>
                </Stack>
            </Card>
        );
    }

    // ── Loaded state ───────────────────────────────────────────────
    return (
        <Card sx={{ px: 2, py: 1.5, overflow: "hidden" }}>
            {/* The actual <audio> is non-visual — we drive it programmatically. */}
            <Box
                component="audio"
                ref={audioRef}
                src={src}
                preload="metadata"
                onPlay={() => { setPlaying(true); setEnded(false); }}
                onPause={() => setPlaying(false)}
                onEnded={() => { setPlaying(false); setEnded(true); }}
                onTimeUpdate={(e) => onTimeUpdate((e.currentTarget as HTMLAudioElement).currentTime)}
                onSeeked={(e) => onTimeUpdate((e.currentTarget as HTMLAudioElement).currentTime)}
                onError={onError}
                sx={{ display: "none" }}
            />

            <Stack direction="row" spacing={2} sx={{ alignItems: "center", minHeight: 40 }}>
                <IconButton
                    onClick={togglePlay}
                    aria-label={ended ? "Replay" : playing ? "Pause" : "Play"}
                    sx={{
                        bgcolor: "primary.main",
                        color: "primary.contrastText",
                        width: 40,
                        height: 40,
                        flexShrink: 0,
                        "&:hover": { bgcolor: "primary.dark" },
                    }}
                >
                    {ended ? <ReplayIcon /> : playing ? <PauseIcon /> : <PlayArrowIcon />}
                </IconButton>

                {/* Progress bar — clickable, grows on hover for affordance. */}
                <Box
                    onClick={handleSeek}
                    role="slider"
                    aria-label="Playback progress"
                    aria-valuemin={0}
                    aria-valuemax={Math.round(durationSeconds)}
                    aria-valuenow={Math.round(currentTime)}
                    sx={{
                        flex: 1,
                        height: 6,
                        bgcolor: "divider",
                        borderRadius: 999,
                        cursor: "pointer",
                        position: "relative",
                        overflow: "hidden",
                        transition: "height 120ms",
                        "&:hover": { height: 8 },
                    }}
                >
                    <Box
                        sx={{
                            position: "absolute",
                            left: 0,
                            top: 0,
                            bottom: 0,
                            width: `${progressPct}%`,
                            bgcolor: "primary.main",
                            // Linear transition keeps the fill smooth despite the
                            // 4Hz updates from timeupdate.
                            transition: "width 250ms linear",
                        }}
                    />
                </Box>

                <Typography
                    variant="caption"
                    sx={{
                        fontFamily: "var(--font-geist-mono), monospace",
                        color: "text.secondary",
                        minWidth: 92,
                        textAlign: "right",
                        flexShrink: 0,
                        fontSize: 12,
                    }}
                >
                    {formatDuration(currentTime)} / {formatDuration(durationSeconds)}
                </Typography>

                <Box
                    component="button"
                    type="button"
                    onClick={cycleSpeed}
                    aria-label={`Playback speed: ${playbackRate}× (click to change)`}
                    sx={{
                        border: 1,
                        borderColor: "divider",
                        bgcolor: "transparent",
                        borderRadius: 999,
                        px: 1.25,
                        py: 0.5,
                        fontSize: 11,
                        fontWeight: 700,
                        color: "text.secondary",
                        cursor: "pointer",
                        fontFamily: "inherit",
                        minWidth: 44,
                        flexShrink: 0,
                        transition: "border-color 120ms, color 120ms",
                        "&:hover": { borderColor: "primary.main", color: "primary.main" },
                        "&:focus-visible": {
                            outline: "2px solid",
                            outlineColor: "primary.main",
                            outlineOffset: 2,
                        },
                    }}
                >
                    {playbackRate}×
                </Box>
            </Stack>
        </Card>
    );
}
