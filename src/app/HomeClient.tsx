"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import AppBar from "@mui/material/AppBar";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Container from "@mui/material/Container";
import Divider from "@mui/material/Divider";
import Paper from "@mui/material/Paper";
import Stack from "@mui/material/Stack";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import LogoutIcon from "@mui/icons-material/Logout";
import type { TranscriptSegment } from "@/entity/Transcription";
import { logout } from "@/app/actions/auth";
import { BrandLogo } from "./components/BrandLogo";
import RecorderPanel from "./components/RecorderPanel";
import FileUpload from "./components/FileUpload";
import SummaryBlock from "./components/SummaryBlock";
import TranscriptPanel from "./components/TranscriptPanel";
import ChatPanel from "./components/ChatPanel";
import SessionHistory from "./components/SessionHistory";

type Status = "idle" | "recording" | "processing" | "done" | "error";

type HistoryRecord = {
    id: string;
    title: string | null;
    summary: string | null;
    duration: number;
    segments: TranscriptSegment[];
    createdAt: string;
};

interface Props {
    userEmail: string;
    userName: string | null;
}

const SIDEBAR_WIDTH = 288;

export default function HomeClient({ userEmail, userName }: Props) {
    const [status, setStatus] = useState<Status>("idle");
    const [segments, setSegments] = useState<TranscriptSegment[]>([]);
    const [currentTranscriptId, setCurrentTranscriptId] = useState<string | null>(null);
    const [currentTitle, setCurrentTitle] = useState<string | null>(null);
    const [currentSummary, setCurrentSummary] = useState<string | null>(null);
    const [elapsed, setElapsed] = useState(0);
    const [error, setError] = useState<string | null>(null);
    const [history, setHistory] = useState<HistoryRecord[]>([]);
    const [historyLoaded, setHistoryLoaded] = useState(false);
    const [historyLoading, setHistoryLoading] = useState(false);

    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const chunksRef = useRef<Blob[]>([]);
    const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

    const loadHistory = useCallback(async () => {
        setHistoryLoading(true);
        try {
            const res = await fetch("/api/transcriptions");
            if (!res.ok) throw new Error(`Server returned ${res.status}`);
            setHistory(await res.json());
            setHistoryLoaded(true);
        } catch {
            toast.error("Couldn't load your transcripts. Try refreshing.");
        } finally {
            setHistoryLoading(false);
        }
    }, []);

    useEffect(() => {
        loadHistory();
    }, [loadHistory]);

    const startRecording = async () => {
        setError(null);
        setSegments([]);
        setCurrentTranscriptId(null);
        setCurrentTitle(null);
        setCurrentSummary(null);
        setElapsed(0);

        let stream: MediaStream;
        try {
            stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch (err) {
            const isDenied =
                err instanceof DOMException &&
                (err.name === "NotAllowedError" || err.name === "PermissionDeniedError");
            setError(
                isDenied
                    ? "Microphone access blocked. Click the lock icon in your browser's address bar → Site settings → Microphone → Allow, then refresh."
                    : "Could not access microphone. Make sure a microphone is connected and try again."
            );
            setStatus("error");
            return;
        }

        const mediaRecorder = new MediaRecorder(stream);
        mediaRecorderRef.current = mediaRecorder;
        chunksRef.current = [];

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) chunksRef.current.push(e.data);
        };

        mediaRecorder.onstop = async () => {
            stream.getTracks().forEach((t) => t.stop());
            clearInterval(timerRef.current!);
            setStatus("processing");

            const blob = new Blob(chunksRef.current, {
                type: chunksRef.current[0]?.type ?? "audio/webm",
            });
            const formData = new FormData();
            formData.append("audio", blob, "recording.webm");

            try {
                const res = await fetch("/api/transcribe", { method: "POST", body: formData });
                const data = await res.json();
                if (!res.ok) { setError(data.error ?? "Transcription failed"); setStatus("error"); return; }
                setSegments(data.segments);
                setCurrentTranscriptId(data.id);
                setCurrentSummary(data.summary ?? null);
                setStatus("done");
                loadHistory();
            } catch {
                setError("Network error. Please try again.");
                setStatus("error");
            }
        };

        mediaRecorder.start(250);
        setStatus("recording");
        timerRef.current = setInterval(() => setElapsed((e) => e + 1), 1000);
    };

    const stopRecording = () => mediaRecorderRef.current?.stop();

    const handleFileUpload = async (file: File) => {
        setError(null);
        setSegments([]);
        setCurrentTranscriptId(null);
        setCurrentTitle(null);
        setCurrentSummary(null);
        setStatus("processing");

        const formData = new FormData();
        formData.append("audio", file, file.name);

        try {
            const res = await fetch("/api/transcribe", { method: "POST", body: formData });
            const data = await res.json();
            if (!res.ok) { setError(data.error ?? "Transcription failed"); setStatus("error"); return; }
            setSegments(data.segments);
            setCurrentTranscriptId(data.id);
            setCurrentSummary(data.summary ?? null);
            setStatus("done");
            loadHistory();
        } catch {
            setError("Network error. Please try again.");
            setStatus("error");
        }
    };

    const showTranscript = status === "done" || status === "processing";

    return (
        <Box sx={{ minHeight: "100vh", bgcolor: "background.default", display: { lg: "flex" } }}>

            {/* ── DESKTOP SIDEBAR ── */}
            <Paper
                square
                elevation={0}
                component="aside"
                sx={{
                    display: { xs: "none", lg: "flex" },
                    width: SIDEBAR_WIDTH,
                    flexShrink: 0,
                    flexDirection: "column",
                    borderRight: 1,
                    borderColor: "divider",
                    position: "sticky",
                    top: 0,
                    height: "100vh",
                    overflow: "auto",
                }}
            >
                {/* Logo */}
                <Box sx={{ px: 3, py: 2.5, borderBottom: 1, borderColor: "divider" }}>
                    <Stack direction="row" spacing={1.25} sx={{ alignItems: "center" }}>
                        <BrandLogo size={28} />
                        <Typography variant="subtitle1" sx={{ fontWeight: 700, color: "text.primary" }}>
                            Audia
                        </Typography>
                    </Stack>
                    <Typography variant="caption" sx={{ display: "block", color: "text.disabled", mt: 1, fontSize: 10, lineHeight: 1.5 }}>
                        Record conversations — speakers identified automatically.
                    </Typography>
                </Box>

                {/* Recorder + upload */}
                <Stack spacing={3} sx={{ flex: 1, alignItems: "center", justifyContent: "center", px: 3, py: 4 }}>
                    <RecorderPanel
                        status={status}
                        elapsed={elapsed}
                        error={error}
                        onStart={startRecording}
                        onStop={stopRecording}
                    />

                    <Stack direction="row" spacing={1.5} sx={{ alignItems: "center", width: "100%" }}>
                        <Divider sx={{ flex: 1 }} />
                        <Typography variant="caption" sx={{ fontSize: 10, color: "text.disabled", letterSpacing: "0.18em", textTransform: "uppercase" }}>
                            or
                        </Typography>
                        <Divider sx={{ flex: 1 }} />
                    </Stack>

                    <FileUpload
                        onUpload={handleFileUpload}
                        disabled={status === "recording" || status === "processing"}
                    />
                </Stack>

                {/* Account */}
                <Box sx={{ px: 3, py: 2.5, borderTop: 1, borderColor: "divider" }}>
                    <Typography variant="overline" sx={{ display: "block", fontWeight: 700, letterSpacing: "0.18em", color: "text.disabled", mb: 0.25, lineHeight: 1.2 }}>
                        Signed in as
                    </Typography>
                    {userName && (
                        <Typography variant="body2" sx={{ fontWeight: 600, color: "text.primary", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                            {userName}
                        </Typography>
                    )}
                    <Typography variant="caption" sx={{ display: "block", color: userName ? "text.secondary" : "text.primary", fontWeight: userName ? 400 : 500, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        {userEmail}
                    </Typography>
                    <Box component="form" action={logout} sx={{ mt: 1 }}>
                        <Button type="submit" size="small" startIcon={<LogoutIcon sx={{ fontSize: 14 }} />} sx={{ color: "primary.main", textTransform: "none", fontWeight: 500, px: 0, "&:hover": { bgcolor: "transparent", color: "primary.dark" } }}>
                            Sign out
                        </Button>
                    </Box>
                </Box>
            </Paper>

            {/* ── MAIN AREA ── */}
            <Box sx={{ flex: 1, minWidth: 0 }}>

                {/* Mobile top bar */}
                <AppBar
                    position="sticky"
                    elevation={0}
                    color="inherit"
                    sx={{
                        display: { xs: "block", lg: "none" },
                        bgcolor: "background.paper",
                        borderBottom: 1,
                        borderColor: "divider",
                    }}
                >
                    <Toolbar sx={{ minHeight: 56, px: 2, justifyContent: "space-between" }}>
                        <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
                            <BrandLogo size={24} />
                            <Typography variant="body2" sx={{ fontWeight: 700, color: "text.primary" }}>
                                Audia
                            </Typography>
                        </Stack>
                        <Box component="form" action={logout}>
                            <Button type="submit" variant="outlined" size="small" startIcon={<LogoutIcon sx={{ fontSize: 14 }} />}>
                                Sign out
                            </Button>
                        </Box>
                    </Toolbar>
                </AppBar>

                {/* Mobile recorder + upload */}
                <Box sx={{ display: { xs: "block", lg: "none" }, px: 2, pt: 2.5, pb: 1 }}>
                    <Stack spacing={1.5}>
                        <Paper sx={{ p: 3, borderRadius: 4, border: 1, borderColor: "divider" }}>
                            <RecorderPanel
                                status={status}
                                elapsed={elapsed}
                                error={error}
                                onStart={startRecording}
                                onStop={stopRecording}
                            />
                        </Paper>
                        <Paper sx={{ px: 2, py: 2, borderRadius: 4, border: 1, borderColor: "divider" }}>
                            <Typography variant="overline" sx={{ display: "block", fontWeight: 700, letterSpacing: "0.2em", color: "text.disabled", mb: 1.5 }}>
                                Upload File
                            </Typography>
                            <FileUpload
                                onUpload={handleFileUpload}
                                disabled={status === "recording" || status === "processing"}
                            />
                        </Paper>
                    </Stack>
                </Box>

                {/* Content */}
                <Container maxWidth="md" component="main" sx={{ py: { xs: 3, lg: 5 }, px: { xs: 2, lg: 5 } }}>
                    <Stack spacing={4}>
                        {/* Live session */}
                        {showTranscript && (
                            <Box component="section" sx={{ animation: "fade-in 250ms ease both" }}>
                                <Stack direction="row" spacing={1.5} sx={{ alignItems: "center", mb: 1.5 }}>
                                    <Typography variant="overline" component="h2" sx={{ fontWeight: 700, letterSpacing: "0.25em", color: "text.disabled", lineHeight: 1, flexShrink: 0 }}>
                                        Live Session
                                    </Typography>
                                    <Divider sx={{ flex: 1 }} />
                                    {status === "done" && (
                                        <Typography variant="caption" sx={{ fontSize: 10, color: "primary.main", fontWeight: 600, flexShrink: 0 }}>
                                            ✓ Complete
                                        </Typography>
                                    )}
                                </Stack>

                                <Stack spacing={1.5}>
                                    {status === "done" && currentTranscriptId && (
                                        <SummaryBlock
                                            transcriptionId={currentTranscriptId}
                                            initialSummary={currentSummary}
                                            onSaved={setCurrentSummary}
                                        />
                                    )}

                                    <TranscriptPanel
                                        segments={segments}
                                        transcriptionId={currentTranscriptId ?? undefined}
                                        currentTitle={currentTitle}
                                        onSegmentsUpdate={setSegments}
                                        onTitleSave={setCurrentTitle}
                                        isProcessing={status === "processing"}
                                    />

                                    {status === "done" && segments.length > 0 && (
                                        <ChatPanel
                                            segments={segments}
                                            transcriptionId={currentTranscriptId ?? undefined}
                                            currentTitle={currentTitle}
                                            onTitleSave={setCurrentTitle}
                                        />
                                    )}
                                </Stack>
                            </Box>
                        )}

                        {/* Archive */}
                        <SessionHistory
                            history={history}
                            loading={historyLoading}
                            loaded={historyLoaded}
                            onDelete={(id) => setHistory((prev) => prev.filter((r) => r.id !== id))}
                            onSegmentsUpdate={(id, updated) =>
                                setHistory((prev) =>
                                    prev.map((r) => (r.id === id ? { ...r, segments: updated } : r))
                                )
                            }
                            onTitleUpdate={(id, title) =>
                                setHistory((prev) =>
                                    prev.map((r) => (r.id === id ? { ...r, title } : r))
                                )
                            }
                            onSummaryUpdate={(id, summary) =>
                                setHistory((prev) =>
                                    prev.map((r) => (r.id === id ? { ...r, summary } : r))
                                )
                            }
                        />
                    </Stack>
                </Container>
            </Box>
        </Box>
    );
}
