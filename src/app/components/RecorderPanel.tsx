"use client";
import Box from "@mui/material/Box";
import IconButton from "@mui/material/IconButton";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import Alert from "@mui/material/Alert";
import StopIcon from "@mui/icons-material/Stop";
import MicIcon from "@mui/icons-material/Mic";
import { formatDuration } from "./utils";

type Status = "idle" | "recording" | "processing" | "done" | "error";

interface Props {
    status: Status;
    elapsed: number;
    error: string | null;
    onStart: () => void;
    onStop: () => void;
}

export default function RecorderPanel({ status, elapsed, error, onStart, onStop }: Props) {
    const isRecording = status === "recording";
    const isProcessing = status === "processing";
    const isDone = status === "done";
    const isError = status === "error";

    return (
        <Stack spacing={2.5} sx={{ alignItems: "center", width: "100%" }}>
            <Typography
                variant="overline"
                sx={{ fontWeight: 700, letterSpacing: "0.25em", color: "text.disabled", lineHeight: 1, userSelect: "none" }}
            >
                Voice Capture
            </Typography>

            {isRecording ? (
                <Box sx={{ position: "relative", width: 128, height: 128, display: "flex", alignItems: "center", justifyContent: "center" }}>
                    <Box
                        sx={{
                            position: "absolute",
                            inset: 0,
                            borderRadius: "50%",
                            border: "2px solid",
                            borderColor: "rgba(252, 165, 165, 1)",
                            animation: "pulse-ring 1.6s cubic-bezier(0.4, 0, 0.6, 1) infinite",
                        }}
                    />
                    <Box
                        sx={{
                            position: "absolute",
                            inset: 0,
                            borderRadius: "50%",
                            border: "2px solid",
                            borderColor: "rgba(254, 202, 202, 1)",
                            animation: "pulse-ring 1.6s cubic-bezier(0.4, 0, 0.6, 1) 0.5s infinite",
                        }}
                    />
                    <IconButton
                        onClick={onStop}
                        sx={{
                            position: "relative",
                            zIndex: 10,
                            width: 80,
                            height: 80,
                            bgcolor: "#ef4444",
                            color: "#fff",
                            boxShadow: "0 10px 25px -5px rgba(254,202,202,0.8)",
                            display: "flex",
                            flexDirection: "column",
                            gap: 0.25,
                            "&:hover": { bgcolor: "#dc2626" },
                        }}
                        aria-label="Stop recording"
                    >
                        <StopIcon sx={{ fontSize: 18 }} />
                        <Typography variant="caption" sx={{ fontFamily: "var(--font-geist-mono), monospace", fontWeight: 700, color: "#fff", lineHeight: 1 }}>
                            {formatDuration(elapsed)}
                        </Typography>
                    </IconButton>
                </Box>
            ) : isProcessing ? (
                <Stack spacing={2} sx={{ alignItems: "center", height: 128, justifyContent: "center" }}>
                    <Box sx={{ display: "flex", alignItems: "flex-end", gap: 0.5, height: 40 }}>
                        {[0, 0.1, 0.2, 0.3, 0.4].map((delay, i) => (
                            <Box
                                key={i}
                                sx={{
                                    width: 8,
                                    height: "100%",
                                    borderRadius: 0.5,
                                    background: "linear-gradient(to top, #5b21b6, #a78bfa)",
                                    animation: `bar-wave 0.8s ease-in-out ${delay}s infinite`,
                                }}
                            />
                        ))}
                    </Box>
                    <Typography variant="caption" color="text.secondary">
                        Analyzing audio…
                    </Typography>
                </Stack>
            ) : (
                <Stack spacing={1.5} sx={{ alignItems: "center", height: 128, justifyContent: "center" }}>
                    <IconButton
                        onClick={onStart}
                        sx={{
                            width: 80,
                            height: 80,
                            background: "linear-gradient(135deg, #5b21b6, #4f46e5)",
                            color: "#fff",
                            boxShadow: "0 10px 25px -5px rgba(196,181,253,0.8)",
                            "&:hover": {
                                background: "linear-gradient(135deg, #4c1d95, #4338ca)",
                                boxShadow: "0 12px 28px -4px rgba(167,139,250,0.85)",
                            },
                        }}
                        aria-label={isDone ? "Start a new session" : isError ? "Retry recording" : "Start recording"}
                    >
                        {isDone || isError ? (
                            <Typography variant="caption" sx={{ fontWeight: 700, letterSpacing: "0.15em", color: "#fff" }}>
                                {isDone ? "NEW" : "RETRY"}
                            </Typography>
                        ) : (
                            <MicIcon sx={{ fontSize: 32 }} />
                        )}
                    </IconButton>
                    {(isDone || isError) && (
                        <Typography variant="caption" sx={{ color: "text.disabled", fontSize: 10 }}>
                            {isDone ? "Start a new session" : "Try again"}
                        </Typography>
                    )}
                </Stack>
            )}

            {isRecording && (
                <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
                    <Box
                        sx={{
                            width: 8,
                            height: 8,
                            borderRadius: "50%",
                            bgcolor: "#ef4444",
                            animation: "rec-blink 1s ease-in-out infinite",
                        }}
                    />
                    <Typography variant="caption" sx={{ color: "#ef4444", fontWeight: 500 }}>
                        Recording in progress
                    </Typography>
                </Stack>
            )}

            {isError && error && (
                <Alert severity="error" variant="outlined" sx={{ maxWidth: 240, fontSize: 12, py: 0.5 }}>
                    {error}
                </Alert>
            )}
        </Stack>
    );
}
