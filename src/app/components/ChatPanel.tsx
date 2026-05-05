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
import ChatBubbleOutlineIcon from "@mui/icons-material/ChatBubbleOutlined";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import SendIcon from "@mui/icons-material/Send";
import type { TranscriptSegment } from "@/entity/Transcription";
import { formatDuration } from "./utils";
import TitleInput from "./TitleInput";

type Message = {
    role: "user" | "assistant";
    content: string;
    streaming?: boolean;
};

const QUICK_PROMPTS = [
    "What were the action items?",
    "What key decisions were made?",
    "Summarize each speaker's points",
    "Were there any unresolved questions?",
];

interface Props {
    segments: TranscriptSegment[];
    transcriptionId?: string;
    currentTitle?: string | null;
    onTitleSave?: (title: string | null) => void;
}

function buildPrompt(segments: TranscriptSegment[], question: string): string {
    const transcript = segments
        .map((s) => `${s.speaker} (${formatDuration(s.start)}): ${s.text}`)
        .join("\n");
    return `You are a helpful assistant analyzing a conversation transcript. Answer the question concisely and clearly based only on what's in the transcript.\n\nTranscript:\n${transcript}\n\nQuestion: ${question}`;
}

export default function ChatPanel({ segments, transcriptionId, currentTitle, onTitleSave }: Props) {
    const [open, setOpen] = useState(false);
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const bottomRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

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

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ prompt: buildPrompt(segments, question) }),
            });

            if (!res.body) throw new Error("No stream");

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
        } catch {
            setMessages((prev) =>
                prev.map((m, i) =>
                    i === prev.length - 1
                        ? { ...m, content: "Something went wrong. Please try again.", streaming: false }
                        : m
                )
            );
        } finally {
            setMessages((prev) =>
                prev.map((m, i) => (i === prev.length - 1 ? { ...m, streaming: false } : m))
            );
            setLoading(false);
        }
    };

    const questionCount = messages.filter((m) => m.role === "user").length;
    const noTranscript = segments.length === 0;

    return (
        <Card sx={{ overflow: "hidden" }}>
            <Box sx={{ height: 3, background: "linear-gradient(to right, #0369a1, #0ea5e9, #38bdf8)" }} />

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
                    "&:hover": { bgcolor: "#f8fafc" },
                    transition: "background 150ms",
                }}
            >
                <Stack direction="row" spacing={1.25} sx={{ alignItems: "center", flex: 1, minWidth: 0 }}>
                    <Box
                        sx={{
                            width: 22,
                            height: 22,
                            borderRadius: "50%",
                            bgcolor: "#f0f9ff",
                            border: 1,
                            borderColor: "#bae6fd",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            flexShrink: 0,
                            color: "#0369a1",
                        }}
                    >
                        <ChatBubbleOutlineIcon sx={{ fontSize: 12 }} />
                    </Box>

                    {transcriptionId && onTitleSave ? (
                        <Box onClick={(e) => e.stopPropagation()}>
                            <TitleInput
                                transcriptionId={transcriptionId}
                                initialTitle={currentTitle ?? null}
                                fallback="Untitled — click to name"
                                onSaved={onTitleSave}
                            />
                        </Box>
                    ) : (
                        <Typography variant="body2" sx={{ fontWeight: 600, color: "text.primary" }}>
                            Ask about this transcript
                        </Typography>
                    )}

                    {questionCount > 0 && (
                        <Chip
                            label={questionCount}
                            size="small"
                            sx={{
                                height: 18,
                                fontSize: 10,
                                fontWeight: 600,
                                bgcolor: "#f0f9ff",
                                color: "#0369a1",
                                border: 1,
                                borderColor: "#bae6fd",
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
                                        color: "#475569",
                                        bgcolor: "transparent",
                                        "&:hover": { borderColor: "#7dd3fc", color: "#0369a1", bgcolor: "#f0f9ff" },
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
                                                ? { bgcolor: "#0369a1", color: "#fff", borderTopRightRadius: 4 }
                                                : { bgcolor: "#f8fafc", border: 1, borderColor: "divider", color: "#334155", borderTopLeftRadius: 4 }),
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
                                                            bgcolor: "#cbd5e1",
                                                            animation: `rec-blink 1.2s ease-in-out ${delay}s infinite`,
                                                        }}
                                                    />
                                                ))}
                                            </Box>
                                        ) : (
                                            <>
                                                <Box component="span" sx={{ whiteSpace: "pre-wrap" }}>
                                                    {msg.content}
                                                </Box>
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
                            placeholder={noTranscript ? "Record or upload a session first…" : "Ask anything about this transcript…"}
                            disabled={loading || noTranscript}
                            size="small"
                            fullWidth
                            slotProps={{
                                input: {
                                    endAdornment: (
                                        <InputAdornment position="end">
                                            <IconButton
                                                onClick={() => send(input)}
                                                disabled={!input.trim() || loading || noTranscript}
                                                size="small"
                                                edge="end"
                                                sx={{
                                                    bgcolor: "#0369a1",
                                                    color: "#fff",
                                                    "&:hover": { bgcolor: "#0284c7" },
                                                    "&.Mui-disabled": { bgcolor: "#e2e8f2", color: "#94a3b8" },
                                                }}
                                                aria-label="Send"
                                            >
                                                <SendIcon sx={{ fontSize: 16 }} />
                                            </IconButton>
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
