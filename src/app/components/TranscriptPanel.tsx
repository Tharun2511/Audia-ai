"use client";
import { useState } from "react";
import { toast } from "sonner";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import Chip from "@mui/material/Chip";
import CircularProgress from "@mui/material/CircularProgress";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import Stack from "@mui/material/Stack";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";
import ArrowDropDownIcon from "@mui/icons-material/ArrowDropDown";
import ArticleOutlinedIcon from "@mui/icons-material/ArticleOutlined";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import CheckIcon from "@mui/icons-material/Check";
import EditIcon from "@mui/icons-material/Edit";
import EditOutlinedIcon from "@mui/icons-material/EditOutlined";
import FileDownloadOutlinedIcon from "@mui/icons-material/FileDownloadOutlined";
import TextSnippetOutlinedIcon from "@mui/icons-material/TextSnippetOutlined";
import IconButton from "@mui/material/IconButton";
import type { TranscriptSegment } from "@/entity/Transcription";
import {
    downloadAsFile,
    formatTranscriptAsMd,
    formatTranscriptAsTxt,
    suggestedFilename,
} from "@/lib/transcript-export";
import { buildColorMap, formatDuration } from "./utils";

interface Props {
    segments: TranscriptSegment[];
    transcriptionId?: string;
    onSegmentsUpdate?: (segments: TranscriptSegment[]) => void;
    isProcessing?: boolean;
    /** Optional context — when provided, enables the Download menu. */
    title?: string | null;
    summary?: string | null;
    duration?: number;
    createdAt?: string;
    /**
     * Index of the segment currently being played. -1 = none. When >= 0 and
     * onSegmentSeek is also defined, that segment is rendered with a highlight.
     */
    activeSegmentIndex?: number;
    /**
     * If provided, each segment becomes clickable and clicking calls this with
     * the segment's start time (in seconds). Lifts the seek into the parent
     * which owns the audio element.
     */
    onSegmentSeek?: (startSeconds: number) => void;
}

export default function TranscriptPanel({
    segments,
    transcriptionId,
    onSegmentsUpdate,
    isProcessing,
    title,
    summary,
    duration,
    createdAt,
    activeSegmentIndex = -1,
    onSegmentSeek,
}: Props) {
    const uniqueSpeakers = [...new Set(segments.map((s) => s.speaker))];
    const colorMap = buildColorMap(uniqueSpeakers);

    const [copied, setCopied] = useState(false);
    const [showRenamer, setShowRenamer] = useState(false);
    const [speakerNames, setSpeakerNames] = useState<Record<string, string>>(
        () => Object.fromEntries(uniqueSpeakers.map((s) => [s, s]))
    );
    const [saving, setSaving] = useState(false);
    const [downloadAnchor, setDownloadAnchor] = useState<HTMLElement | null>(null);

    /*
     * Per-segment text editing state. Only one segment can be in edit mode
     * at a time — multiple in-progress edits would create ambiguous save
     * semantics and a more complex commit flow.
     *
     *   editingIndex   index of the segment currently being edited; null = none
     *   editingText    the draft text shown in the TextField
     *   editingSaving  true while the PATCH is in flight
     */
    const [editingIndex, setEditingIndex] = useState<number | null>(null);
    const [editingText, setEditingText] = useState("");
    const [editingSaving, setEditingSaving] = useState(false);

    const startEdit = (i: number) => {
        setEditingIndex(i);
        setEditingText(segments[i].text);
    };

    const cancelEdit = () => {
        setEditingIndex(null);
        setEditingText("");
    };

    const saveEdit = async () => {
        if (editingIndex === null || !transcriptionId || !onSegmentsUpdate) return;
        const trimmed = editingText.trim();
        // If the user wiped the field and saved, treat as "no change" rather
        // than persisting an empty segment — empty bubbles look broken.
        if (trimmed.length === 0) {
            cancelEdit();
            return;
        }
        // No-op if text didn't change.
        if (trimmed === segments[editingIndex].text) {
            cancelEdit();
            return;
        }
        setEditingSaving(true);
        try {
            const res = await fetch(`/api/transcriptions/${transcriptionId}`, {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    segmentTextEdits: { [editingIndex]: trimmed },
                }),
            });
            if (!res.ok) throw new Error(`Server returned ${res.status}`);
            const { segments: updated } = await res.json();
            onSegmentsUpdate(updated);
            toast.success("Segment updated");
            cancelEdit();
        } catch {
            toast.error("Couldn't save changes. Try again.");
        } finally {
            setEditingSaving(false);
        }
    };

    const handleSaveRenames = async () => {
        if (!transcriptionId || !onSegmentsUpdate) return;
        const speakerMap = Object.fromEntries(
            Object.entries(speakerNames).filter(([orig, renamed]) => orig !== renamed)
        );
        if (Object.keys(speakerMap).length === 0) { setShowRenamer(false); return; }
        setSaving(true);
        try {
            const res = await fetch(`/api/transcriptions/${transcriptionId}`, {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ speakerMap }),
            });
            if (!res.ok) throw new Error(`Server returned ${res.status}`);
            const { segments: updated } = await res.json();
            onSegmentsUpdate(updated);
            setShowRenamer(false);
            const renamedCount = Object.keys(speakerMap).length;
            toast.success(renamedCount === 1 ? "Speaker renamed" : `${renamedCount} speakers renamed`);
        } catch {
            toast.error("Couldn't rename speakers. Try again.");
        } finally {
            setSaving(false);
        }
    };

    const handleCopy = async () => {
        const text = segments
            .map((s) => `${s.speaker} (${formatDuration(s.start)}): ${s.text}`)
            .join("\n\n");
        try {
            await navigator.clipboard.writeText(text);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch {
            toast.error("Couldn't copy to clipboard.");
        }
    };

    const downloadEnabled = createdAt !== undefined && duration !== undefined;

    const handleDownload = (format: "txt" | "md") => {
        if (!downloadEnabled) return;
        const opts = {
            title: title ?? null,
            summary: summary ?? null,
            duration: duration!,
            createdAt: createdAt!,
            segments,
        };
        const content = format === "txt" ? formatTranscriptAsTxt(opts) : formatTranscriptAsMd(opts);
        const filename = suggestedFilename(title ?? null, createdAt!, format);
        const mimeType = format === "txt" ? "text/plain" : "text/markdown";
        try {
            downloadAsFile(content, filename, mimeType);
            toast.success(`Downloaded ${filename}`);
        } catch {
            toast.error("Couldn't generate the download.");
        }
        setDownloadAnchor(null);
    };

    return (
        <Card sx={{ overflow: "hidden" }}>
            {/* Header */}
            <Stack
                direction="row"
                spacing={1}
                sx={{ alignItems: "center", justifyContent: "space-between", px: 2, py: 1.75, borderBottom: 1, borderColor: "divider" }}
            >
                <Typography variant="overline" sx={{ fontWeight: 700, letterSpacing: "0.2em", color: "text.disabled", lineHeight: 1, flexShrink: 0 }}>
                    Transcript
                </Typography>
                {!isProcessing && segments.length > 0 && (
                    <Stack direction="row" spacing={1} sx={{ flexShrink: 0 }}>
                        <Button
                            onClick={handleCopy}
                            variant="outlined"
                            size="small"
                            startIcon={copied ? <CheckIcon /> : <ContentCopyIcon />}
                            sx={{ color: copied ? "primary.main" : "text.secondary" }}
                        >
                            {copied ? "Copied" : "Copy"}
                        </Button>
                        {downloadEnabled && (
                            <Button
                                onClick={(e) => setDownloadAnchor(e.currentTarget)}
                                variant="outlined"
                                size="small"
                                startIcon={<FileDownloadOutlinedIcon />}
                                endIcon={<ArrowDropDownIcon />}
                                aria-haspopup="menu"
                                aria-expanded={Boolean(downloadAnchor)}
                                aria-controls={downloadAnchor ? "transcript-download-menu" : undefined}
                                sx={{ color: "text.secondary" }}
                            >
                                Download
                            </Button>
                        )}
                    </Stack>
                )}
            </Stack>

            <Menu
                id="transcript-download-menu"
                anchorEl={downloadAnchor}
                open={Boolean(downloadAnchor)}
                onClose={() => setDownloadAnchor(null)}
                anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
                transformOrigin={{ vertical: "top", horizontal: "right" }}
                slotProps={{ paper: { sx: { minWidth: 260, mt: 0.5, borderRadius: 2 } } }}
            >
                <MenuItem onClick={() => handleDownload("txt")} sx={{ py: 1.25 }}>
                    <ListItemIcon sx={{ color: "text.secondary" }}>
                        <TextSnippetOutlinedIcon fontSize="small" />
                    </ListItemIcon>
                    <ListItemText
                        primary=".txt"
                        secondary="Plain text — opens anywhere"
                        slotProps={{
                            primary: { sx: { fontWeight: 600, fontSize: 14 } },
                            secondary: { sx: { fontSize: 11, color: "text.disabled" } },
                        }}
                    />
                </MenuItem>
                <MenuItem onClick={() => handleDownload("md")} sx={{ py: 1.25 }}>
                    <ListItemIcon sx={{ color: "text.secondary" }}>
                        <ArticleOutlinedIcon fontSize="small" />
                    </ListItemIcon>
                    <ListItemText
                        primary=".md"
                        secondary="Markdown — Notion, GitHub, Slack"
                        slotProps={{
                            primary: { sx: { fontWeight: 600, fontSize: 14 } },
                            secondary: { sx: { fontSize: 11, color: "text.disabled" } },
                        }}
                    />
                </MenuItem>
            </Menu>

            <Box sx={{ p: 2 }}>
                {isProcessing ? (
                    <Stack direction="row" spacing={1.5} sx={{ alignItems: "center", justifyContent: "center", py: 3 }}>
                        <Box sx={{ display: "flex", alignItems: "flex-end", gap: "3px", height: 24 }}>
                            {[0, 0.1, 0.2, 0.3, 0.4].map((delay, i) => (
                                <Box
                                    key={i}
                                    sx={{
                                        width: 6,
                                        height: "100%",
                                        borderRadius: 0.5,
                                        background: "linear-gradient(to top, var(--mui-palette-primary-main), var(--mui-palette-primary-light))",
                                        animation: `bar-wave 0.8s ease-in-out ${delay}s infinite`,
                                    }}
                                />
                            ))}
                        </Box>
                        <Typography variant="body2" color="text.secondary">
                            Processing audio…
                        </Typography>
                    </Stack>
                ) : segments.length === 0 ? (
                    <Typography variant="body2" sx={{ color: "text.disabled", fontStyle: "italic", textAlign: "center", py: 3 }}>
                        No speech detected.
                    </Typography>
                ) : (
                    <>
                        {/* Speaker legend + rename trigger */}
                        <Stack direction="row" spacing={1} sx={{ alignItems: "center", justifyContent: "space-between", mb: 2 }}>
                            <Stack direction="row" spacing={1.5} sx={{ flexWrap: "wrap" }}>
                                {uniqueSpeakers.map((sp) => (
                                    <Stack key={sp} direction="row" spacing={0.75} sx={{ alignItems: "center" }}>
                                        <Box sx={{ width: 10, height: 10, borderRadius: "50%", bgcolor: colorMap[sp], flexShrink: 0 }} />
                                        <Typography variant="caption" sx={{ color: "text.secondary", fontWeight: 500 }}>
                                            {sp}
                                        </Typography>
                                    </Stack>
                                ))}
                            </Stack>
                            {transcriptionId && onSegmentsUpdate && (
                                <Button
                                    onClick={() => setShowRenamer((v) => !v)}
                                    variant={showRenamer ? "contained" : "outlined"}
                                    size="small"
                                    startIcon={<EditIcon />}
                                    sx={{ flexShrink: 0 }}
                                >
                                    Rename
                                </Button>
                            )}
                        </Stack>

                        {/* Renamer */}
                        {showRenamer && (
                            <Box sx={{ bgcolor: "action.hover", border: 1, borderColor: "divider", borderRadius: 3, p: 2, mb: 2 }}>
                                <Typography variant="overline" sx={{ fontWeight: 700, letterSpacing: "0.2em", color: "primary.main", display: "block", mb: 1.5 }}>
                                    Rename Speakers
                                </Typography>
                                <Stack spacing={1.5}>
                                    {uniqueSpeakers.map((sp) => (
                                        <Stack key={sp} direction="row" spacing={1.5} sx={{ alignItems: "center" }}>
                                            <Stack direction="row" spacing={1} sx={{ alignItems: "center", flexShrink: 0, minWidth: 80 }}>
                                                <Box sx={{ width: 10, height: 10, borderRadius: "50%", bgcolor: colorMap[sp] }} />
                                                <Typography variant="caption" sx={{ fontWeight: 500, color: "text.secondary" }}>
                                                    {sp}
                                                </Typography>
                                            </Stack>
                                            <Typography sx={{ color: "text.disabled", fontSize: 12 }}>→</Typography>
                                            <TextField
                                                value={speakerNames[sp] ?? sp}
                                                onChange={(e) => setSpeakerNames((prev) => ({ ...prev, [sp]: e.target.value }))}
                                                size="small"
                                                sx={{ flex: 1, maxWidth: 200 }}
                                            />
                                        </Stack>
                                    ))}
                                    <Stack direction="row" spacing={1} sx={{ pt: 0.5 }}>
                                        <Button
                                            onClick={handleSaveRenames}
                                            disabled={saving}
                                            variant="contained"
                                            size="small"
                                            startIcon={saving ? <CircularProgress size={12} sx={{ color: "inherit" }} /> : null}
                                        >
                                            {saving ? "Saving…" : "Save"}
                                        </Button>
                                        <Button onClick={() => setShowRenamer(false)} size="small" sx={{ color: "text.secondary" }}>
                                            Cancel
                                        </Button>
                                    </Stack>
                                </Stack>
                            </Box>
                        )}

                        {/* Conversation bubbles */}
                        <Stack spacing={1.5}>
                            {segments.map((seg, i) => {
                                const isEditing = editingIndex === i;
                                const canEdit = Boolean(transcriptionId && onSegmentsUpdate);
                                // While editing this segment, suppress the
                                // card-level click-to-seek so the user doesn't
                                // accidentally jump audio while typing.
                                const cardSeekable = onSegmentSeek && !isEditing;
                                return (
                                <Card
                                    key={i}
                                    onClick={cardSeekable ? () => onSegmentSeek!(seg.start) : undefined}
                                    role={cardSeekable ? "button" : undefined}
                                    tabIndex={cardSeekable ? 0 : undefined}
                                    onKeyDown={cardSeekable ? (e) => {
                                        if (e.key === "Enter" || e.key === " ") {
                                            e.preventDefault();
                                            onSegmentSeek!(seg.start);
                                        }
                                    } : undefined}
                                    sx={{
                                        borderRadius: 1.5,
                                        borderLeftStyle: "solid",
                                        borderLeftWidth: i === activeSegmentIndex ? 4 : 3,
                                        borderLeftColor: colorMap[seg.speaker] ?? "text.disabled",
                                        transition: "background-color 150ms, border-left-width 150ms, box-shadow 150ms",
                                        ...(cardSeekable ? {
                                            cursor: "pointer",
                                            "&:hover": { bgcolor: "action.hover" },
                                            "&:focus-visible": {
                                                outline: "2px solid",
                                                outlineColor: "primary.main",
                                                outlineOffset: 2,
                                            },
                                        } : {}),
                                        ...(i === activeSegmentIndex && !isEditing ? {
                                            bgcolor: "action.selected",
                                        } : {}),
                                        ...(isEditing ? {
                                            // Primary-colored ring overlays the speaker color
                                            // — speaker identity is preserved on the left bar,
                                            // edit context is signaled by the outer ring.
                                            boxShadow: (theme) => `0 0 0 2px ${theme.vars?.palette?.primary?.main ?? theme.palette.primary.main}`,
                                            bgcolor: "background.paper",
                                        } : {}),
                                    }}
                                >
                                    <Box sx={{ px: 2, py: 1.5 }}>
                                        <Stack
                                            direction="row"
                                            sx={{ alignItems: "center", justifyContent: "space-between", mb: 0.75 }}
                                        >
                                            <Stack direction="row" spacing={1.25} sx={{ alignItems: "center" }}>
                                                <Typography
                                                    variant="caption"
                                                    sx={{ fontWeight: 700, color: colorMap[seg.speaker] ?? "text.secondary" }}
                                                >
                                                    {seg.speaker}
                                                </Typography>
                                                <Chip
                                                    label={formatDuration(seg.start)}
                                                    size="small"
                                                    sx={{
                                                        height: 18,
                                                        fontSize: 10,
                                                        fontFamily: "var(--font-geist-mono), monospace",
                                                        color: i === activeSegmentIndex ? "primary.main" : "text.disabled",
                                                        bgcolor: "action.hover",
                                                        fontWeight: i === activeSegmentIndex ? 700 : 400,
                                                        "& .MuiChip-label": { px: 0.75 },
                                                    }}
                                                />
                                            </Stack>
                                            {/*
                                             * Edit affordance — subtle by default (low opacity), brightens
                                             * on row-hover for desktop discoverability; always visible on
                                             * touch via `@media (hover: none)`. stopPropagation prevents
                                             * the card's seek handler from firing on click.
                                             */}
                                            {canEdit && !isEditing && (
                                                <IconButton
                                                    onClick={(e) => { e.stopPropagation(); startEdit(i); }}
                                                    size="small"
                                                    aria-label="Edit segment text"
                                                    sx={{
                                                        color: "text.disabled",
                                                        opacity: 0.35,
                                                        transition: "opacity 150ms, color 150ms",
                                                        ".MuiCard-root:hover &": { opacity: 1 },
                                                        "@media (hover: none)": { opacity: 1 },
                                                        "&:hover": { color: "primary.main" },
                                                    }}
                                                >
                                                    <EditOutlinedIcon sx={{ fontSize: 14 }} />
                                                </IconButton>
                                            )}
                                        </Stack>
                                        {isEditing ? (
                                            <Stack spacing={1} onClick={(e) => e.stopPropagation()}>
                                                <TextField
                                                    value={editingText}
                                                    onChange={(e) => setEditingText(e.target.value)}
                                                    onKeyDown={(e) => {
                                                        if (e.key === "Escape") {
                                                            e.preventDefault();
                                                            cancelEdit();
                                                        } else if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                                                            e.preventDefault();
                                                            saveEdit();
                                                        }
                                                    }}
                                                    autoFocus
                                                    multiline
                                                    minRows={2}
                                                    fullWidth
                                                    size="small"
                                                    disabled={editingSaving}
                                                />
                                                <Stack direction="row" spacing={1}>
                                                    <Button
                                                        onClick={saveEdit}
                                                        disabled={editingSaving}
                                                        variant="contained"
                                                        size="small"
                                                        startIcon={editingSaving ? <CircularProgress size={12} sx={{ color: "inherit" }} /> : null}
                                                    >
                                                        {editingSaving ? "Saving…" : "Save"}
                                                    </Button>
                                                    <Button
                                                        onClick={cancelEdit}
                                                        disabled={editingSaving}
                                                        size="small"
                                                        sx={{ color: "text.secondary" }}
                                                    >
                                                        Cancel
                                                    </Button>
                                                    <Typography
                                                        variant="caption"
                                                        sx={{ color: "text.disabled", alignSelf: "center", fontSize: 10 }}
                                                    >
                                                        ⌘+Enter to save · Esc to cancel
                                                    </Typography>
                                                </Stack>
                                            </Stack>
                                        ) : (
                                            <Typography variant="body2" sx={{ color: "text.primary", lineHeight: 1.65 }}>
                                                {seg.text}
                                            </Typography>
                                        )}
                                    </Box>
                                </Card>
                                );
                            })}
                        </Stack>
                    </>
                )}
            </Box>
        </Card>
    );
}
