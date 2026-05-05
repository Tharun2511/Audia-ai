"use client";
import { useState, useEffect } from "react";
import { toast } from "sonner";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import Chip from "@mui/material/Chip";
import CircularProgress from "@mui/material/CircularProgress";
import Stack from "@mui/material/Stack";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import CheckIcon from "@mui/icons-material/Check";
import EditIcon from "@mui/icons-material/Edit";
import type { TranscriptSegment } from "@/entity/Transcription";
import { buildColorMap, formatDuration } from "./utils";
import TitleInput from "./TitleInput";

interface Props {
    segments: TranscriptSegment[];
    transcriptionId?: string;
    currentTitle?: string | null;
    onSegmentsUpdate?: (segments: TranscriptSegment[]) => void;
    onTitleSave?: (title: string | null) => void;
    isProcessing?: boolean;
}

export default function TranscriptPanel({
    segments,
    transcriptionId,
    currentTitle,
    onSegmentsUpdate,
    onTitleSave,
    isProcessing,
}: Props) {
    const [copied, setCopied] = useState(false);
    const [showRenamer, setShowRenamer] = useState(false);
    const [speakerNames, setSpeakerNames] = useState<Record<string, string>>({});
    const [saving, setSaving] = useState(false);

    const uniqueSpeakers = [...new Set(segments.map((s) => s.speaker))];
    const colorMap = buildColorMap(uniqueSpeakers);

    useEffect(() => {
        setSpeakerNames(Object.fromEntries(uniqueSpeakers.map((s) => [s, s])));
        setShowRenamer(false);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [segments]);

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

    return (
        <Card sx={{ overflow: "hidden" }}>
            {/* Header */}
            <Stack
                direction="row"
                spacing={1}
                sx={{ alignItems: "center", justifyContent: "space-between", px: 2, py: 1.75, borderBottom: 1, borderColor: "divider" }}
            >
                <Stack direction="row" spacing={1.5} sx={{ alignItems: "center", minWidth: 0, flex: 1 }}>
                    <Typography variant="overline" sx={{ fontWeight: 700, letterSpacing: "0.2em", color: "text.disabled", lineHeight: 1, flexShrink: 0 }}>
                        Transcript
                    </Typography>
                    {transcriptionId && onTitleSave && (
                        <TitleInput
                            transcriptionId={transcriptionId}
                            initialTitle={currentTitle ?? null}
                            fallback="Untitled — click to name"
                            onSaved={onTitleSave}
                        />
                    )}
                </Stack>
                {!isProcessing && segments.length > 0 && (
                    <Button
                        onClick={handleCopy}
                        variant="outlined"
                        size="small"
                        startIcon={copied ? <CheckIcon /> : <ContentCopyIcon />}
                        sx={{ color: copied ? "primary.main" : "text.secondary", flexShrink: 0 }}
                    >
                        {copied ? "Copied" : "Copy"}
                    </Button>
                )}
            </Stack>

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
                            {segments.map((seg, i) => (
                                <Card
                                    key={i}
                                    sx={{
                                        borderRadius: 1.5,
                                        borderLeftStyle: "solid",
                                        borderLeftWidth: 3,
                                        borderLeftColor: colorMap[seg.speaker] ?? "text.disabled",
                                    }}
                                >
                                    <Box sx={{ px: 2, py: 1.5 }}>
                                        <Stack direction="row" spacing={1.25} sx={{ alignItems: "center", mb: 0.75 }}>
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
                                                    color: "text.disabled",
                                                    bgcolor: "action.hover",
                                                    "& .MuiChip-label": { px: 0.75 },
                                                }}
                                            />
                                        </Stack>
                                        <Typography variant="body2" sx={{ color: "text.primary", lineHeight: 1.65 }}>
                                            {seg.text}
                                        </Typography>
                                    </Box>
                                </Card>
                            ))}
                        </Stack>
                    </>
                )}
            </Box>
        </Card>
    );
}
