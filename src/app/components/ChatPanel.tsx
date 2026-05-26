"use client";
import { useState, useRef, useEffect } from "react";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import Chip from "@mui/material/Chip";
import IconButton from "@mui/material/IconButton";
import InputAdornment from "@mui/material/InputAdornment";
import Stack from "@mui/material/Stack";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";
import Tooltip from "@mui/material/Tooltip";
import ChatBubbleOutlineIcon from "@mui/icons-material/ChatBubbleOutlined";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import SendIcon from "@mui/icons-material/Send";
import StopIcon from "@mui/icons-material/Stop";
import type { TranscriptSegment } from "@/entity/Transcription";
import { formatDuration } from "./utils";

type Citation = {
    n: number;                  // 1-indexed chunk number (matches [N] in answer text)
    chunkId: string;
    transcriptionId: string;
    speakers: string[];
    startTime: number;
    endTime: number;
    preview: string;
};

type Message = {
    role: "user" | "assistant";
    content: string;
    streaming?: boolean;
    citations?: Citation[];
};

const QUICK_PROMPTS = [
    "What were the action items?",
    "What key decisions were made?",
    "Summarize each speaker's points",
    "Were there any unresolved questions?",
];

interface Props {
    transcriptionId: string;
    segments: TranscriptSegment[];
    onCitationClick?: (startTime: number) => void;
}

export default function ChatPanel({ transcriptionId, segments, onCitationClick }: Props) {
    const [open, setOpen] = useState(false);
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const bottomRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);
    const abortControllerRef = useRef<AbortController | null>(null);

    const stop = () => {
        abortControllerRef.current?.abort();
    };

    useEffect(() => {
        if (messages.length > 0) {
            bottomRef.current?.scrollIntoView({ behavior: "smooth" });
        }
    }, [messages]);

    useEffect(() => {
        if (open) setTimeout(() => inputRef.current?.focus(), 60);
    }, [open]);

    const send = async (text: string) => {
        const question = text.trim();
        if (!question || loading || segments.length === 0) return;

        setInput("");
        setMessages((prev) => [
            ...prev,
            { role: "user", content: question },
            { role: "assistant", content: "", streaming: true },
        ]);
        setLoading(true);

        const ctrl = new AbortController();
        abortControllerRef.current = ctrl;

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                signal: ctrl.signal,
                headers: { "Content-Type": "application/json" },
                // RAG protocol: client sends only a reference (transcriptionId) +
                // the question. Server does retrieval, selects relevant chunks,
                // and includes them in the prompt. No more "ship the whole
                // transcript" pattern — scales to long meetings.
                body: JSON.stringify({ question, transcriptionId }),
            });

            if (!res.body) throw new Error("No stream");

            // Citations come back in a response header before the stream body.
            // Parse once; attach to the assistant message so the renderer can
            // turn [N] markers into clickable chips.
            const citationsRaw = res.headers.get("X-Citations");
            const citations: Citation[] = citationsRaw ? JSON.parse(citationsRaw) : [];
            setMessages((prev) =>
                prev.map((m, i) => (i === prev.length - 1 ? { ...m, citations } : m))
            );

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let accumulated = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                accumulated += decoder.decode(value, { stream: true });
                const snap = accumulated;
                setMessages((prev) =>
                    prev.map((m, i) => (i === prev.length - 1 ? { ...m, content: snap } : m))
                );
            }
        } catch (err) {
            const wasAborted = err instanceof Error && err.name === "AbortError";
            if (!wasAborted) {
                setMessages((prev) =>
                    prev.map((m, i) =>
                        i === prev.length - 1
                            ? { ...m, content: "Something went wrong. Please try again.", streaming: false }
                            : m
                    )
                );
            }
            // On abort, leave the partial content as-is — the user stopped on purpose.
        } finally {
            abortControllerRef.current = null;
            setMessages((prev) =>
                prev.map((m, i) => (i === prev.length - 1 ? { ...m, streaming: false } : m))
            );
            setLoading(false);
        }
    };

    const questionCount = messages.filter((m) => m.role === "user").length;
    const noTranscript = segments.length === 0;

    /**
     * Renders streamed text with [N] markers replaced by clickable citation
     * chips. Unknown N (citation hallucinated by the model, or markers from a
     * partial stream where the next chunk is still arriving) render as plain
     * "[N]" text so nothing breaks.
     */
    const renderWithCitations = (text: string, citations?: Citation[]) => {
        if (!citations || citations.length === 0) {
            return <Box component="span" sx={{ whiteSpace: "pre-wrap" }}>{text}</Box>;
        }
        const byNumber = new Map(citations.map((c) => [c.n, c]));
        // Split on [N] markers while KEEPING the markers as separate tokens.
        const parts = text.split(/(\[\d+\])/g);
        return (
            <Box component="span" sx={{ whiteSpace: "pre-wrap" }}>
                {parts.map((part, idx) => {
                    const m = part.match(/^\[(\d+)\]$/);
                    if (!m) return <span key={idx}>{part}</span>;
                    const n = Number(m[1]);
                    const cite = byNumber.get(n);
                    if (!cite) return <span key={idx}>{part}</span>;
                    const speakerLabel = cite.speakers.join(", ");
                    const timeLabel = formatDuration(cite.startTime);
                    return (
                        <Tooltip
                            key={idx}
                            arrow
                            title={
                                <Box sx={{ p: 0.5 }}>
                                    <Box sx={{ fontWeight: 600, fontSize: 11, opacity: 0.8 }}>
                                        {speakerLabel} · {timeLabel}
                                    </Box>
                                    <Box sx={{ fontSize: 12, mt: 0.5 }}>
                                        {cite.preview}{cite.preview.length >= 140 ? "…" : ""}
                                    </Box>
                                </Box>
                            }
                        >
                            <Box
                                component="span"
                                onClick={() => onCitationClick?.(cite.startTime)}
                                sx={{
                                    display: "inline-flex",
                                    alignItems: "center",
                                    justifyContent: "center",
                                    minWidth: 18,
                                    height: 18,
                                    px: 0.5,
                                    mx: 0.25,
                                    borderRadius: 1,
                                    fontSize: 11,
                                    fontWeight: 600,
                                    lineHeight: 1,
                                    bgcolor: "info.main",
                                    color: "info.contrastText",
                                    cursor: onCitationClick ? "pointer" : "default",
                                    transition: "background 120ms",
                                    "&:hover": onCitationClick ? { bgcolor: "info.dark" } : undefined,
                                    verticalAlign: "middle",
                                }}
                            >
                                {n}
                            </Box>
                        </Tooltip>
                    );
                })}
            </Box>
        );
    };

    return (
        <Card sx={{ overflow: "hidden" }}>
            <Box sx={{ height: 3, background: "linear-gradient(to right, var(--mui-palette-info-dark), var(--mui-palette-info-main), var(--mui-palette-info-light))" }} />

            {/* Header */}
            <Stack
                direction="row"
                spacing={1}
                onClick={() => setOpen((v) => !v)}
                sx={{
                    alignItems: "center",
                    justifyContent: "space-between",
                    px: 2,
                    py: 1.75,
                    cursor: "pointer",
                    "&:hover": { bgcolor: "action.hover" },
                    transition: "background 150ms",
                }}
            >
                <Stack direction="row" spacing={1.25} sx={{ alignItems: "center", flex: 1, minWidth: 0 }}>
                    <Box
                        sx={{
                            width: 22,
                            height: 22,
                            borderRadius: "50%",
                            bgcolor: "info.main",
                            color: "info.contrastText",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            flexShrink: 0,
                            opacity: 0.9,
                        }}
                    >
                        <ChatBubbleOutlineIcon sx={{ fontSize: 12 }} />
                    </Box>

                    <Typography variant="body2" sx={{ fontWeight: 600, color: "text.primary" }}>
                        Ask about this transcript
                    </Typography>

                    {questionCount > 0 && (
                        <Chip
                            label={questionCount}
                            size="small"
                            sx={{
                                height: 18,
                                fontSize: 10,
                                fontWeight: 600,
                                bgcolor: "info.main",
                                color: "info.contrastText",
                                "& .MuiChip-label": { px: 0.75 },
                            }}
                        />
                    )}
                </Stack>

                <IconButton
                    size="small"
                    sx={{ color: "text.disabled", flexShrink: 0 }}
                    aria-label={open ? "Collapse" : "Expand"}
                >
                    <ExpandMoreIcon
                        sx={{ transition: "transform 200ms", transform: open ? "rotate(180deg)" : "rotate(0deg)" }}
                    />
                </IconButton>
            </Stack>

            {open && (
                <Box sx={{ borderTop: 1, borderColor: "divider" }}>
                    {messages.length === 0 && (
                        <Box sx={{ px: 2, pt: 1.5, pb: 1, display: "flex", flexWrap: "wrap", gap: 1 }}>
                            {QUICK_PROMPTS.map((q) => (
                                <Chip
                                    key={q}
                                    label={q}
                                    onClick={() => send(q)}
                                    disabled={loading || noTranscript}
                                    sx={{
                                        borderRadius: 999,
                                        border: 1,
                                        borderColor: "divider",
                                        color: "text.secondary",
                                        bgcolor: "transparent",
                                        "&:hover": { borderColor: "info.light", color: "info.main", bgcolor: "action.hover" },
                                    }}
                                />
                            ))}
                        </Box>
                    )}

                    {messages.length > 0 && (
                        <Stack spacing={1.5} sx={{ px: 2, py: 1.5, maxHeight: 288, overflowY: "auto" }}>
                            {messages.map((msg, i) => (
                                <Box
                                    key={i}
                                    sx={{ display: "flex", justifyContent: msg.role === "user" ? "flex-end" : "flex-start" }}
                                >
                                    <Box
                                        sx={{
                                            maxWidth: "85%",
                                            borderRadius: 3,
                                            px: 1.75,
                                            py: 1.25,
                                            fontSize: 14,
                                            lineHeight: 1.6,
                                            ...(msg.role === "user"
                                                ? { bgcolor: "info.main", color: "info.contrastText", borderTopRightRadius: 4 }
                                                : { bgcolor: "action.hover", border: 1, borderColor: "divider", color: "text.primary", borderTopLeftRadius: 4 }),
                                        }}
                                    >
                                        {msg.role === "assistant" && msg.streaming && msg.content === "" ? (
                                            <Box sx={{ display: "inline-flex", alignItems: "center", gap: 0.5, py: 0.25 }}>
                                                {[0, 0.2, 0.4].map((delay, j) => (
                                                    <Box
                                                        key={j}
                                                        sx={{
                                                            width: 6,
                                                            height: 6,
                                                            borderRadius: "50%",
                                                            bgcolor: "text.disabled",
                                                            animation: `rec-blink 1.2s ease-in-out ${delay}s infinite`,
                                                        }}
                                                    />
                                                ))}
                                            </Box>
                                        ) : (
                                            <>
                                                {msg.role === "assistant"
                                                    ? renderWithCitations(msg.content, msg.citations)
                                                    : <Box component="span" sx={{ whiteSpace: "pre-wrap" }}>{msg.content}</Box>}
                                                {msg.streaming && (
                                                    <Box
                                                        component="span"
                                                        sx={{
                                                            display: "inline-block",
                                                            width: "2px",
                                                            height: 14,
                                                            bgcolor: "currentColor",
                                                            ml: 0.25,
                                                            verticalAlign: "middle",
                                                            opacity: 0.6,
                                                            animation: "rec-blink 1s ease-in-out infinite",
                                                        }}
                                                    />
                                                )}
                                            </>
                                        )}
                                    </Box>
                                </Box>
                            ))}
                            <Box ref={bottomRef} />
                        </Stack>
                    )}

                    {/* Input row */}
                    <Box sx={{ display: "flex", gap: 1, px: 2, py: 1.5, ...(messages.length > 0 ? { borderTop: 1, borderColor: "divider" } : {}) }}>
                        <TextField
                            inputRef={inputRef}
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={(e) => {
                                if (e.key === "Enter" && !e.shiftKey) {
                                    e.preventDefault();
                                    send(input);
                                }
                            }}
                            placeholder={noTranscript ? "Record or upload a session first…" : loading ? "Streaming response… press stop or wait" : "Ask anything about this transcript…"}
                            disabled={noTranscript}
                            size="small"
                            fullWidth
                            slotProps={{
                                input: {
                                    endAdornment: (
                                        <InputAdornment position="end">
                                            {loading ? (
                                                <IconButton
                                                    onClick={stop}
                                                    size="small"
                                                    edge="end"
                                                    sx={{
                                                        bgcolor: "error.main",
                                                        color: "error.contrastText",
                                                        "&:hover": { bgcolor: "error.dark" },
                                                    }}
                                                    aria-label="Stop"
                                                >
                                                    <StopIcon sx={{ fontSize: 16 }} />
                                                </IconButton>
                                            ) : (
                                                <IconButton
                                                    onClick={() => send(input)}
                                                    disabled={!input.trim() || noTranscript}
                                                    size="small"
                                                    edge="end"
                                                    sx={{
                                                        bgcolor: "info.main",
                                                        color: "info.contrastText",
                                                        "&:hover": { bgcolor: "info.dark" },
                                                        "&.Mui-disabled": { bgcolor: "action.disabledBackground", color: "action.disabled" },
                                                    }}
                                                    aria-label="Send"
                                                >
                                                    <SendIcon sx={{ fontSize: 16 }} />
                                                </IconButton>
                                            )}
                                        </InputAdornment>
                                    ),
                                },
                            }}
                        />
                    </Box>
                </Box>
            )}
        </Card>
    );
}
