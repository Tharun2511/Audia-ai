"use client";
import { useState } from "react";
import { toast } from "sonner";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import Collapse from "@mui/material/Collapse";
import IconButton from "@mui/material/IconButton";
import Stack from "@mui/material/Stack";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import DeleteOutlinedIcon from "@mui/icons-material/DeleteOutlined";
import type { TranscriptSegment } from "@/entity/Transcription";
import { buildColorMap, formatDuration } from "./utils";
import TitleInput from "./TitleInput";
import SummaryBlock from "./SummaryBlock";
import TranscriptPanel from "./TranscriptPanel";
import ChatPanel from "./ChatPanel";

type HistoryRecord = {
    id: string;
    title: string | null;
    summary: string | null;
    duration: number;
    segments: TranscriptSegment[];
    createdAt: string;
};

interface Props {
    record: HistoryRecord;
    onDelete: (id: string) => void;
    onSegmentsUpdate: (id: string, segments: TranscriptSegment[]) => void;
    onTitleUpdate: (id: string, title: string | null) => void;
    onSummaryUpdate: (id: string, summary: string | null) => void;
}

export default function HistoryEntry({
    record,
    onDelete,
    onSegmentsUpdate,
    onTitleUpdate,
    onSummaryUpdate,
}: Props) {
    const [expanded, setExpanded] = useState(false);
    const [deleting, setDeleting] = useState(false);

    const speakers = [...new Set(record.segments.map((s) => s.speaker))];
    const colorMap = buildColorMap(speakers);

    const date = new Date(record.createdAt);
    const relativeTime = (() => {
        const diff = Date.now() - date.getTime();
        const mins = Math.floor(diff / 60000);
        const hours = Math.floor(diff / 3600000);
        const days = Math.floor(diff / 86400000);
        if (mins < 60) return `${mins}m ago`;
        if (hours < 24) return `${hours}h ago`;
        if (days < 7) return `${days}d ago`;
        return date.toLocaleDateString();
    })();

    const handleDelete = async (e: React.MouseEvent) => {
        e.stopPropagation();
        setDeleting(true);
        try {
            const res = await fetch(`/api/transcriptions/${record.id}`, { method: "DELETE" });
            if (!res.ok) throw new Error(`Server returned ${res.status}`);
            onDelete(record.id);
        } catch {
            toast.error("Couldn't delete this transcript. Try again.");
            setDeleting(false);
        }
    };

    return (
        <Card
            sx={{
                overflow: "hidden",
                transition: "box-shadow 200ms, border-color 200ms",
                ...(expanded
                    ? { borderColor: "#ddd6fe", boxShadow: "0 8px 16px -4px rgba(91,33,182,0.08)" }
                    : { "&:hover": { borderColor: "#ddd6fe", boxShadow: "0 8px 16px -4px rgba(91,33,182,0.08)" } }),
            }}
        >
            {expanded && (
                <Box sx={{ height: 2, background: "linear-gradient(to right, #5b21b6, #4f46e5)" }} />
            )}

            {/* Header row */}
            <Stack
                direction="row"
                spacing={2}
                onClick={() => setExpanded((v) => !v)}
                sx={{ alignItems: "center", justifyContent: "space-between", px: 2, py: 1.75, cursor: "pointer" }}
            >
                <Box sx={{ minWidth: 0, flex: 1 }}>
                    <Box onClick={(e) => e.stopPropagation()}>
                        <TitleInput
                            transcriptionId={record.id}
                            initialTitle={record.title}
                            fallback={`${date.toLocaleString()} (Rename title)`}
                            onSaved={(t) => onTitleUpdate(record.id, t)}
                        />
                    </Box>
                    <Stack direction="row" spacing={1} sx={{ alignItems: "center", mt: 0.75, flexWrap: "wrap" }}>
                        <Typography variant="caption" sx={{ fontSize: 10, color: "text.disabled" }}>
                            {relativeTime}
                        </Typography>
                        <Typography variant="caption" sx={{ fontSize: 10, color: "divider" }}>·</Typography>
                        <Typography variant="caption" sx={{ fontSize: 10, fontFamily: "var(--font-geist-mono), monospace", color: "text.disabled" }}>
                            {formatDuration(record.duration)}
                        </Typography>
                        {speakers.length > 0 && (
                            <>
                                <Typography variant="caption" sx={{ fontSize: 10, color: "divider" }}>·</Typography>
                                <Stack direction="row" spacing={0.75} sx={{ alignItems: "center" }}>
                                    {speakers.map((sp) => (
                                        <Tooltip key={sp} title={sp}>
                                            <Box sx={{ width: 8, height: 8, borderRadius: "50%", bgcolor: colorMap[sp], flexShrink: 0 }} />
                                        </Tooltip>
                                    ))}
                                    <Typography variant="caption" sx={{ fontSize: 10, color: "text.disabled" }}>
                                        {speakers.join(", ")}
                                    </Typography>
                                </Stack>
                            </>
                        )}
                    </Stack>
                </Box>

                <Stack direction="row" spacing={0.5} sx={{ alignItems: "center", flexShrink: 0 }}>
                    <Tooltip title={deleting ? "Deleting…" : "Delete transcript"}>
                        <span>
                            <IconButton
                                onClick={handleDelete}
                                disabled={deleting}
                                size="small"
                                sx={{ color: "text.disabled", "&:hover": { color: "error.main", bgcolor: "rgba(239,68,68,0.08)" } }}
                                aria-label="Delete transcript"
                            >
                                <DeleteOutlinedIcon fontSize="small" />
                            </IconButton>
                        </span>
                    </Tooltip>
                    <ExpandMoreIcon
                        sx={{
                            color: "text.disabled",
                            transition: "transform 200ms",
                            transform: expanded ? "rotate(180deg)" : "rotate(0deg)",
                        }}
                    />
                </Stack>
            </Stack>

            {/* Expanded detail */}
            <Collapse in={expanded} unmountOnExit>
                <Box sx={{ borderTop: 1, borderColor: "divider", px: 2, pb: 2, pt: 2 }}>
                    <Stack spacing={1.5}>
                        <SummaryBlock
                            transcriptionId={record.id}
                            initialSummary={record.summary}
                            onSaved={(s) => onSummaryUpdate(record.id, s)}
                        />
                        <TranscriptPanel
                            segments={record.segments}
                            transcriptionId={record.id}
                            onSegmentsUpdate={(updated) => onSegmentsUpdate(record.id, updated)}
                        />
                        <ChatPanel segments={record.segments} />
                    </Stack>
                </Box>
            </Collapse>
        </Card>
    );
}
