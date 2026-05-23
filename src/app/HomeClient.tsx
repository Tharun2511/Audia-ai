"use client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import AppBar from "@mui/material/AppBar";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Container from "@mui/material/Container";
import Drawer from "@mui/material/Drawer";
import IconButton from "@mui/material/IconButton";
import List from "@mui/material/List";
import Stack from "@mui/material/Stack";
import Toolbar from "@mui/material/Toolbar";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import AddIcon from "@mui/icons-material/Add";
import LogoutOutlinedIcon from "@mui/icons-material/LogoutOutlined";
import MenuOutlinedIcon from "@mui/icons-material/MenuOutlined";
import type { TranscriptSegment } from "@/entity/Transcription";
import { logout } from "@/app/actions/auth";
import { BrandLogo } from "./components/BrandLogo";
import SessionListItem from "./components/SessionListItem";
import SessionView from "./components/SessionView";
import SidebarSearch from "./components/SidebarSearch";
import ThemeToggle from "./components/ThemeToggle";
import { ReadyState, RecordingState, ProcessingState } from "./components/MainPaneStates";

type Status = "idle" | "recording" | "paused" | "processing" | "done" | "error";

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

const SIDEBAR_WIDTH = 320;

export default function HomeClient({ userEmail, userName }: Props) {
    const [status, setStatus] = useState<Status>("idle");
    const [elapsed, setElapsed] = useState(0);
    const [error, setError] = useState<string | null>(null);
    const [history, setHistory] = useState<HistoryRecord[]>([]);
    const [historyLoaded, setHistoryLoaded] = useState(false);
    const [historyLoading, setHistoryLoading] = useState(false);
    const [selectedId, setSelectedId] = useState<string | null>(null);
    const [justCompletedId, setJustCompletedId] = useState<string | null>(null);
    const [drawerOpen, setDrawerOpen] = useState(false);
    const [searchQuery, setSearchQuery] = useState("");

    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const chunksRef = useRef<Blob[]>([]);
    const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

    /* ---------- Data loading ---------- */
    const loadHistory = useCallback(async (): Promise<HistoryRecord[]> => {
        setHistoryLoading(true);
        try {
            const res = await fetch("/api/transcriptions");
            if (!res.ok) throw new Error(`Server returned ${res.status}`);
            const data: HistoryRecord[] = await res.json();
            setHistory(data);
            setHistoryLoaded(true);
            return data;
        } catch {
            toast.error("Couldn't load your transcripts. Try refreshing.");
            return [];
        } finally {
            setHistoryLoading(false);
        }
    }, []);

    useEffect(() => {
        loadHistory();
    }, [loadHistory]);

    /* ---------- Selection / patches ---------- */
    const selectedRecord = useMemo(
        () => history.find((r) => r.id === selectedId) ?? null,
        [history, selectedId]
    );

    /* ---------- Search ---------- */
    const filteredHistory = useMemo(() => {
        const q = searchQuery.trim().toLowerCase();
        if (!q) return history;
        return history.filter((r) => {
            if (r.title?.toLowerCase().includes(q)) return true;
            if (r.summary?.toLowerCase().includes(q)) return true;
            return r.segments.some(
                (s) =>
                    s.speaker.toLowerCase().includes(q) ||
                    s.text.toLowerCase().includes(q)
            );
        });
    }, [history, searchQuery]);
    const isFiltering = searchQuery.trim().length > 0;

    /* ---------- Delete with Undo ----------
     * Optimistic remove + 5s window to undo before the server-side delete
     * actually fires. If the user navigates away in the window, we flush
     * pending deletes so the UI on next visit isn't out of sync with the DB.
     */
    const pendingDeletesRef = useRef(new Map<string, ReturnType<typeof setTimeout>>());

    const sortByDateDesc = useCallback((records: HistoryRecord[]): HistoryRecord[] => {
        return [...records].sort(
            (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
        );
    }, []);

    const requestDelete = useCallback((id: string) => {
        const record = history.find((r) => r.id === id);
        if (!record) return;

        setHistory((prev) => prev.filter((r) => r.id !== id));
        if (selectedId === id) setSelectedId(null);

        const commit = async () => {
            pendingDeletesRef.current.delete(id);
            try {
                const res = await fetch(`/api/transcriptions/${id}`, { method: "DELETE" });
                if (!res.ok) throw new Error();
            } catch {
                setHistory((prev) => sortByDateDesc([record, ...prev]));
                toast.error(`Couldn't delete "${record.title ?? "Untitled session"}".`);
            }
        };

        const timer = setTimeout(commit, 5000);
        pendingDeletesRef.current.set(id, timer);

        toast.success(`Deleted "${record.title ?? "Untitled session"}"`, {
            duration: 5000,
            action: {
                label: "Undo",
                onClick: () => {
                    const t = pendingDeletesRef.current.get(id);
                    if (!t) return; // Already committed — no-op
                    clearTimeout(t);
                    pendingDeletesRef.current.delete(id);
                    setHistory((prev) => sortByDateDesc([record, ...prev]));
                },
            },
        });
    }, [history, selectedId, sortByDateDesc]);

    useEffect(() => {
        const pending = pendingDeletesRef.current;
        return () => {
            // On unmount (logout, route change), flush pending deletes so the
            // server eventually matches what the user saw on screen.
            pending.forEach((timer, id) => {
                clearTimeout(timer);
                fetch(`/api/transcriptions/${id}`, { method: "DELETE" }).catch(() => { });
            });
            pending.clear();
        };
    }, []);

    const patchRecord = useCallback((id: string, patch: Partial<HistoryRecord>) => {
        setHistory((prev) => prev.map((r) => (r.id === id ? { ...r, ...patch } : r)));
    }, []);

    /* ---------- Recording ---------- */
    const startRecording = async () => {
        setError(null);
        setSelectedId(null);
        setJustCompletedId(null);
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
                if (!res.ok) {
                    setError(data.error ?? "Transcription failed");
                    setStatus("idle");
                    return;
                }
                setStatus("idle");
                setJustCompletedId(data.id);
                const fresh = await loadHistory();
                if (fresh.find((r) => r.id === data.id)) setSelectedId(data.id);
            } catch {
                setError("Network error. Please try again.");
                setStatus("idle");
            }
        };

        mediaRecorder.start(250);
        setStatus("recording");
        timerRef.current = setInterval(() => setElapsed((e) => e + 1), 1000);
    };

    const pauseRecording = () => {
        const rec = mediaRecorderRef.current;
        if (!rec || rec.state !== "recording") return;
        rec.pause();
        if (timerRef.current) {
            clearInterval(timerRef.current);
            timerRef.current = null;
        }
        setStatus("paused");
    };

    const resumeRecording = () => {
        const rec = mediaRecorderRef.current;
        if (!rec || rec.state !== "paused") return;
        rec.resume();
        timerRef.current = setInterval(() => setElapsed((e) => e + 1), 1000);
        setStatus("recording");
    };

    const stopRecording = () => {
        const rec = mediaRecorderRef.current;
        if (!rec || rec.state === "inactive") return;
        // mediaRecorder.stop() works from both "recording" and "paused" states.
        rec.stop();
    };

    /* ---------- File upload ---------- */
    const handleFileUpload = async (file: File) => {
        setError(null);
        setSelectedId(null);
        setJustCompletedId(null);
        setStatus("processing");

        const formData = new FormData();
        formData.append("audio", file, file.name);

        try {
            const res = await fetch("/api/transcribe", { method: "POST", body: formData });
            const data = await res.json();
            if (!res.ok) {
                setError(data.error ?? "Transcription failed");
                setStatus("idle");
                return;
            }
            setStatus("idle");
            setJustCompletedId(data.id);
            const fresh = await loadHistory();
            if (fresh.find((r) => r.id === data.id)) setSelectedId(data.id);
        } catch {
            setError("Network error. Please try again.");
            setStatus("idle");
        }
    };

    const startNew = () => {
        setSelectedId(null);
        setError(null);
        setDrawerOpen(false);
    };

    /* ---------- What does the main pane show? ---------- */
    let mainView: "recording" | "processing" | "session" | "ready";
    if (status === "recording" || status === "paused") mainView = "recording";
    else if (status === "processing") mainView = "processing";
    else if (selectedRecord) mainView = "session";
    else mainView = "ready";

    /* ---------- Sidebar JSX (reused desktop + mobile drawer) ---------- */
    const sidebarContent = (
        <Stack sx={{ width: SIDEBAR_WIDTH, height: "100%", bgcolor: "background.paper", borderRight: 1, borderColor: "divider" }}>

            {/* Top: brand + theme + sign out */}
            <Stack direction="row" spacing={1} sx={{ alignItems: "center", justifyContent: "space-between", px: 2.5, py: 2, borderBottom: 1, borderColor: "divider" }}>
                <Stack direction="row" spacing={1.25} sx={{ alignItems: "center" }}>
                    <BrandLogo size={28} />
                    <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>Audia</Typography>
                </Stack>
                <Stack direction="row" spacing={0.5} sx={{ alignItems: "center" }}>
                    <ThemeToggle />
                    <Tooltip title="Sign out">
                        <Box component="form" action={logout} sx={{ display: "inline-flex" }}>
                            <IconButton type="submit" size="small" aria-label="Sign out">
                                <LogoutOutlinedIcon fontSize="small" />
                            </IconButton>
                        </Box>
                    </Tooltip>
                </Stack>
            </Stack>

            {/* User identity */}
            <Box sx={{ px: 2.5, py: 1.5, borderBottom: 1, borderColor: "divider" }}>
                {userName && (
                    <Typography variant="body2" sx={{ fontWeight: 600, color: "text.primary", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        {userName}
                    </Typography>
                )}
                <Typography variant="caption" sx={{ display: "block", color: "text.secondary", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {userEmail}
                </Typography>
            </Box>

            {/* New session CTA — only when there's something to leave (a selected session, recording, or processing) */}
            {mainView !== "ready" && (
                <Box sx={{ px: 2, py: 1.75, borderBottom: 1, borderColor: "divider" }}>
                    <Button
                        onClick={startNew}
                        variant="contained"
                        color="primary"
                        fullWidth
                        startIcon={<AddIcon />}
                        disabled={status === "recording" || status === "paused" || status === "processing"}
                        sx={{ borderRadius: 999, py: 1.1 }}
                    >
                        New Session
                    </Button>
                </Box>
            )}

            {/* Sessions list */}
            <Stack direction="row" spacing={1} sx={{ alignItems: "center", justifyContent: "space-between", px: 2.5, pt: 2, pb: 1, flexShrink: 0 }}>
                <Typography variant="overline" sx={{ fontWeight: 700, letterSpacing: "0.18em", color: "text.disabled" }}>
                    Sessions
                </Typography>
                {historyLoaded && (
                    <Typography variant="caption" sx={{ fontSize: 10, color: "text.disabled" }}>
                        {isFiltering ? `${filteredHistory.length} of ${history.length}` : history.length}
                    </Typography>
                )}
            </Stack>
            {historyLoaded && history.length > 0 && (
                <Box sx={{ px: 1.5, pb: 1, flexShrink: 0 }}>
                    <SidebarSearch value={searchQuery} onChange={setSearchQuery} />
                </Box>
            )}
            <Box sx={{ flex: 1, minHeight: 0, overflowY: "auto", px: 1.5, pb: 2 }}>
                {historyLoading && history.length === 0 ? (
                    <Stack sx={{ alignItems: "center", py: 4 }}>
                        <Box sx={{ display: "flex", alignItems: "flex-end", gap: 0.5, height: 24 }}>
                            {[0, 0.1, 0.2, 0.3, 0.4].map((delay, i) => (
                                <Box key={i} sx={{ width: 4, height: "100%", borderRadius: 0.5, bgcolor: "primary.light", animation: `bar-wave 0.8s ease-in-out ${delay}s infinite` }} />
                            ))}
                        </Box>
                    </Stack>
                ) : history.length === 0 ? (
                    <Box sx={{ px: 1, py: 3, textAlign: "center" }}>
                        <Typography variant="caption" sx={{ color: "text.disabled" }}>
                            No sessions yet.
                        </Typography>
                    </Box>
                ) : filteredHistory.length === 0 ? (
                    <Stack spacing={1.25} sx={{ px: 1, py: 3, alignItems: "center", textAlign: "center" }}>
                        <Typography variant="caption" sx={{ color: "text.disabled" }}>
                            No sessions match &ldquo;{searchQuery.trim()}&rdquo;
                        </Typography>
                        <Button onClick={() => setSearchQuery("")} size="small" variant="text" sx={{ fontSize: 11, minHeight: 0, py: 0.5 }}>
                            Clear search
                        </Button>
                    </Stack>
                ) : (
                    <List disablePadding sx={{ display: "flex", flexDirection: "column", gap: 0.5 }}>
                        {filteredHistory.map((record) => (
                            <SessionListItem
                                key={record.id}
                                record={record}
                                selected={record.id === selectedId}
                                onSelect={(id) => { setSelectedId(id); setDrawerOpen(false); }}
                                onDelete={requestDelete}
                            />
                        ))}
                    </List>
                )}
            </Box>
        </Stack>
    );

    /* ---------- Main pane content ---------- */
    let mainContent: React.ReactNode;
    if (mainView === "recording") {
        mainContent = (
            <RecordingState
                elapsed={elapsed}
                isPaused={status === "paused"}
                onPause={pauseRecording}
                onResume={resumeRecording}
                onStop={stopRecording}
            />
        );
    } else if (mainView === "processing") {
        mainContent = <ProcessingState />;
    } else if (mainView === "session" && selectedRecord) {
        mainContent = (
            <SessionView
                id={selectedRecord.id}
                title={selectedRecord.title}
                summary={selectedRecord.summary}
                duration={selectedRecord.duration}
                segments={selectedRecord.segments}
                createdAt={selectedRecord.createdAt}
                isJustCompleted={selectedRecord.id === justCompletedId}
                onTitleSave={(t) => patchRecord(selectedRecord.id, { title: t })}
                onSegmentsUpdate={(s) => patchRecord(selectedRecord.id, { segments: s })}
            />
        );
    } else {
        mainContent = (
            <ReadyState
                error={error}
                onStart={startRecording}
                onUpload={handleFileUpload}
                disabled={status === "recording" || status === "paused" || status === "processing"}
            />
        );
    }

    return (
        <Box sx={{ minHeight: "100vh", bgcolor: "background.default", display: { lg: "flex" } }}>

            {/* Mobile app bar */}
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
                        <IconButton onClick={() => setDrawerOpen(true)} size="small" aria-label="Open menu">
                            <MenuOutlinedIcon />
                        </IconButton>
                        <BrandLogo size={24} />
                        <Typography variant="body2" sx={{ fontWeight: 700 }}>Audia</Typography>
                    </Stack>
                    <Stack direction="row" spacing={0.5} sx={{ alignItems: "center" }}>
                        <ThemeToggle />
                        <Box component="form" action={logout} sx={{ display: "inline-flex" }}>
                            <IconButton type="submit" size="small" aria-label="Sign out">
                                <LogoutOutlinedIcon sx={{ fontSize: 18 }} />
                            </IconButton>
                        </Box>
                    </Stack>
                </Toolbar>
            </AppBar>

            {/* Mobile drawer (sidebar) */}
            <Drawer
                anchor="left"
                open={drawerOpen}
                onClose={() => setDrawerOpen(false)}
                sx={{ display: { xs: "block", lg: "none" }, "& .MuiDrawer-paper": { width: SIDEBAR_WIDTH, boxSizing: "border-box" } }}
            >
                {sidebarContent}
            </Drawer>

            {/* Desktop sidebar (sticky) */}
            <Box
                component="aside"
                sx={{
                    display: { xs: "none", lg: "block" },
                    width: SIDEBAR_WIDTH,
                    flexShrink: 0,
                    position: "sticky",
                    top: 0,
                    height: "100vh",
                }}
            >
                {sidebarContent}
            </Box>

            {/* Main pane */}
            <Box component="main" sx={{ flex: 1, minWidth: 0 }}>
                <Container maxWidth="md" sx={{ py: { xs: 3, lg: 5 }, px: { xs: 2, lg: 5 } }}>
                    {mainContent}
                </Container>
            </Box>
        </Box>
    );
}
