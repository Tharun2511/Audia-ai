"use client";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import GraphicEqIcon from "@mui/icons-material/GraphicEq";
import MicIcon from "@mui/icons-material/Mic";
import PauseIcon from "@mui/icons-material/Pause";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import StopIcon from "@mui/icons-material/Stop";
import { buildColorMap, formatDuration } from "./utils";
import FileUpload from "./FileUpload";

/**
 * "Hub home" — shown whenever no session is selected and we're not recording.
 * This is the place to start a new session — record live or upload a file.
 */
export function ReadyState({
    error,
    onStart,
    onUpload,
    disabled,
}: {
    error: string | null;
    onStart: () => void;
    onUpload: (file: File) => Promise<void>;
    disabled: boolean;
}) {
    return (
        <Stack
            spacing={4}
            sx={{
                minHeight: 520,
                alignItems: "center",
                justifyContent: "center",
                textAlign: "center",
                px: 3,
                animation: "fade-in 250ms ease both",
            }}
        >
            <Stack spacing={1.5} sx={{ alignItems: "center" }}>
                <Typography variant="h5" sx={{ fontWeight: 700, color: "text.primary", textWrap: "balance" }}>
                    Start a new session
                </Typography>
                <Typography variant="body2" sx={{ color: "text.secondary", maxWidth: 420 }}>
                    Record a conversation in your browser, or upload an existing audio file. We'll transcribe it and identify each speaker.
                </Typography>
            </Stack>

            {/* Big record CTA */}
            <Button
                onClick={onStart}
                disabled={disabled}
                size="large"
                variant="contained"
                startIcon={<MicIcon />}
                sx={{
                    px: 5,
                    py: 1.75,
                    fontSize: 16,
                    borderRadius: 999,
                    boxShadow: "0 12px 28px -10px color-mix(in srgb, var(--mui-palette-primary-main) 55%, transparent)",
                    transition: "transform 80ms ease, box-shadow 200ms ease",
                    "&:hover": { boxShadow: "0 14px 30px -8px color-mix(in srgb, var(--mui-palette-primary-main) 70%, transparent)" },
                    "&:active": { transform: "scale(0.98)" },
                }}
            >
                Start Recording
            </Button>

            <Stack direction="row" spacing={1.5} sx={{ alignItems: "center", width: "100%", maxWidth: 360 }}>
                <Box sx={{ flex: 1, height: 1, bgcolor: "divider" }} />
                <Typography variant="caption" sx={{ fontSize: 10, color: "text.disabled", letterSpacing: "0.2em", textTransform: "uppercase" }}>
                    or
                </Typography>
                <Box sx={{ flex: 1, height: 1, bgcolor: "divider" }} />
            </Stack>

            <Box sx={{ width: "100%", maxWidth: 360 }}>
                <FileUpload onUpload={onUpload} disabled={disabled} />
            </Box>

            {error && (
                <Box sx={{ maxWidth: 420, p: 2, borderRadius: 2, border: 1, borderColor: "error.main", bgcolor: "color-mix(in srgb, var(--mui-palette-error-main) 6%, transparent)" }}>
                    <Typography variant="caption" sx={{ color: "error.main", lineHeight: 1.6 }}>
                        {error}
                    </Typography>
                </Box>
            )}
        </Stack>
    );
}

/** Shown while the user is actively recording — or paused. */
export function RecordingState({
    elapsed,
    isPaused,
    onPause,
    onResume,
    onStop,
    liveCaptions = [],
    interimCaption = "",
}: {
    elapsed: number;
    isPaused: boolean;
    onPause: () => void;
    onResume: () => void;
    onStop: () => void;
    /** Real-time captions (10.2) — finalized speaker-labelled segments. */
    liveCaptions?: { speaker: string; text: string }[];
    /** The current in-flight (revisable) caption draft. */
    interimCaption?: string;
}) {
    const colorMap = buildColorMap([...new Set(liveCaptions.map((c) => c.speaker))]);
    const hasCaptions = liveCaptions.length > 0 || interimCaption.length > 0;
    return (
        <Stack
            spacing={3.5}
            sx={{
                minHeight: 520,
                alignItems: "center",
                justifyContent: "center",
                textAlign: "center",
                px: 3,
                animation: "fade-in 250ms ease both",
            }}
        >
            {/* Pulsing rings — only while actively recording */}
            <Box sx={{ position: "relative", width: 160, height: 160, display: "flex", alignItems: "center", justifyContent: "center" }}>
                {!isPaused && (
                    <>
                        <Box sx={{ position: "absolute", inset: 0, borderRadius: "50%", border: "2px solid", borderColor: "color-mix(in srgb, var(--mui-palette-error-main) 40%, transparent)", animation: "pulse-ring 1.6s cubic-bezier(0.4, 0, 0.6, 1) infinite" }} />
                        <Box sx={{ position: "absolute", inset: 0, borderRadius: "50%", border: "2px solid", borderColor: "color-mix(in srgb, var(--mui-palette-error-main) 25%, transparent)", animation: "pulse-ring 1.6s cubic-bezier(0.4, 0, 0.6, 1) 0.5s infinite" }} />
                    </>
                )}
                <Box
                    sx={{
                        width: 104,
                        height: 104,
                        borderRadius: "50%",
                        bgcolor: isPaused
                            ? "color-mix(in srgb, var(--mui-palette-error-main) 55%, transparent)"
                            : "error.main",
                        color: "common.white",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        boxShadow: isPaused
                            ? "0 10px 24px -10px color-mix(in srgb, var(--mui-palette-error-main) 35%, transparent)"
                            : "0 14px 32px -8px color-mix(in srgb, var(--mui-palette-error-main) 55%, transparent)",
                        transition: "background-color 200ms, box-shadow 200ms",
                    }}
                >
                    {isPaused ? (
                        <PauseIcon sx={{ fontSize: 44 }} />
                    ) : (
                        <GraphicEqIcon sx={{ fontSize: 40, animation: "rec-blink 1.2s ease-in-out infinite" }} />
                    )}
                </Box>
            </Box>

            <Stack spacing={0.5} sx={{ alignItems: "center" }}>
                <Typography className="tabular-nums" sx={{ fontFamily: "var(--font-geist-mono), monospace", fontSize: 32, fontWeight: 700, color: "text.primary", letterSpacing: 1 }}>
                    {formatDuration(elapsed)}
                </Typography>
                <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
                    {!isPaused && (
                        <Box sx={{ width: 8, height: 8, borderRadius: "50%", bgcolor: "error.main", animation: "rec-blink 1s ease-in-out infinite" }} />
                    )}
                    <Typography variant="body2" sx={{ color: isPaused ? "warning.main" : "error.main", fontWeight: 500 }}>
                        {isPaused ? "Paused" : "Recording in progress"}
                    </Typography>
                </Stack>
            </Stack>

            {/* Actions — pause/resume is the primary action, stop is the final/red one */}
            <Stack direction="row" spacing={1.5} sx={{ alignItems: "center", flexWrap: "wrap", justifyContent: "center" }}>
                {isPaused ? (
                    <Button
                        onClick={onResume}
                        variant="contained"
                        size="large"
                        startIcon={<PlayArrowIcon />}
                        sx={{
                            px: 3.5,
                            py: 1.5,
                            borderRadius: 999,
                            transition: "transform 80ms ease",
                            "&:active": { transform: "scale(0.98)" },
                        }}
                    >
                        Resume
                    </Button>
                ) : (
                    <Button
                        onClick={onPause}
                        variant="outlined"
                        size="large"
                        startIcon={<PauseIcon />}
                        sx={{
                            px: 3.5,
                            py: 1.5,
                            borderRadius: 999,
                            transition: "transform 80ms ease",
                            "&:active": { transform: "scale(0.98)" },
                        }}
                    >
                        Pause
                    </Button>
                )}
                <Button
                    onClick={onStop}
                    variant="contained"
                    color="error"
                    size="large"
                    startIcon={<StopIcon />}
                    sx={{
                        px: 3.5,
                        py: 1.5,
                        borderRadius: 999,
                        transition: "transform 80ms ease, background-color 200ms",
                        "&:active": { transform: "scale(0.98)" },
                    }}
                >
                    Stop
                </Button>
            </Stack>

            {/* Live captions (10.2): finalized segments solid, the in-flight
                draft faded/italic — the streaming interim-vs-final distinction made visible. */}
            {hasCaptions && (
                <Box
                    sx={{
                        width: "100%",
                        maxWidth: 560,
                        maxHeight: 200,
                        overflowY: "auto",
                        textAlign: "left",
                        borderRadius: 2,
                        border: 1,
                        borderColor: "divider",
                        bgcolor: "background.paper",
                        p: 2,
                    }}
                >
                    <Typography variant="overline" sx={{ fontWeight: 700, letterSpacing: "0.18em", color: "text.disabled", display: "block", mb: 1 }}>
                        Live captions
                    </Typography>
                    <Stack spacing={0.75}>
                        {liveCaptions.map((c, i) => (
                            <Typography key={i} variant="body2" sx={{ lineHeight: 1.5 }}>
                                <Box component="span" sx={{ fontWeight: 700, color: colorMap[c.speaker] ?? "text.secondary", mr: 0.75 }}>
                                    {c.speaker}:
                                </Box>
                                <Box component="span" sx={{ color: "text.primary" }}>{c.text}</Box>
                            </Typography>
                        ))}
                        {interimCaption && (
                            <Typography variant="body2" sx={{ color: "text.disabled", fontStyle: "italic", lineHeight: 1.5 }}>
                                {interimCaption}
                            </Typography>
                        )}
                    </Stack>
                </Box>
            )}
        </Stack>
    );
}

/** Shown while audio is being transcribed/summarized after stop. */
export function ProcessingState() {
    return (
        <Stack
            spacing={2.5}
            sx={{
                minHeight: 520,
                alignItems: "center",
                justifyContent: "center",
                textAlign: "center",
                px: 3,
                animation: "fade-in 250ms ease both",
            }}
        >
            <Box sx={{ display: "flex", alignItems: "flex-end", gap: 0.75, height: 56 }}>
                {[0, 0.1, 0.2, 0.3, 0.4].map((delay, i) => (
                    <Box
                        key={i}
                        sx={{
                            width: 10,
                            height: "100%",
                            borderRadius: 1,
                            background: "linear-gradient(to top, var(--mui-palette-primary-main), var(--mui-palette-primary-light))",
                            animation: `bar-wave 0.8s ease-in-out ${delay}s infinite`,
                        }}
                    />
                ))}
            </Box>
            <Typography variant="h6" sx={{ color: "text.primary", fontWeight: 600 }}>
                Analyzing audio…
            </Typography>
            <Typography variant="body2" sx={{ color: "text.secondary", maxWidth: 380 }}>
                Identifying speakers, transcribing, generating a summary. Usually under 30 seconds.
            </Typography>
        </Stack>
    );
}
