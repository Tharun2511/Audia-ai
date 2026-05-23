"use client";
import Box from "@mui/material/Box";
import IconButton from "@mui/material/IconButton";
import ListItemButton from "@mui/material/ListItemButton";
import Stack from "@mui/material/Stack";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import DeleteOutlinedIcon from "@mui/icons-material/DeleteOutlined";
import type { TranscriptSegment } from "@/entity/Transcription";
import { buildColorMap, formatDuration } from "./utils";

type HistoryRecord = {
    id: string;
    title: string | null;
    duration: number;
    segments: TranscriptSegment[];
    createdAt: string;
};

interface Props {
    record: HistoryRecord;
    selected: boolean;
    onSelect: (id: string) => void;
    onDelete: (id: string) => void;
}

function relativeTime(iso: string): string {
    const date = new Date(iso);
    const diff = Date.now() - date.getTime();
    const mins = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);
    if (mins < 60) return `${mins}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;
    return date.toLocaleDateString();
}

export default function SessionListItem({ record, selected, onSelect, onDelete }: Props) {
    const speakers = [...new Set(record.segments.map((s) => s.speaker))];
    const colorMap = buildColorMap(speakers);

    const handleDelete = (e: React.MouseEvent) => {
        e.stopPropagation();
        onDelete(record.id);
    };

    return (
        <ListItemButton
            onClick={() => onSelect(record.id)}
            selected={selected}
            sx={{
                px: 2,
                py: 1.25,
                borderRadius: 2,
                alignItems: "flex-start",
                gap: 1.25,
                "&.Mui-selected": {
                    bgcolor: "action.selected",
                    "&:hover": { bgcolor: "action.selected" },
                    borderLeft: 3,
                    borderColor: "primary.main",
                    pl: 1.625,
                },
            }}
        >
            <Box sx={{ flex: 1, minWidth: 0 }}>
                <Typography
                    variant="body2"
                    sx={{
                        fontWeight: 600,
                        color: record.title ? "text.primary" : "text.disabled",
                        fontStyle: record.title ? "normal" : "italic",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                    }}
                >
                    {record.title ?? "Untitled"}
                </Typography>
                <Stack direction="row" spacing={0.75} sx={{ alignItems: "center", mt: 0.5, flexWrap: "wrap" }}>
                    <Typography variant="caption" sx={{ fontSize: 10, color: "text.disabled" }}>
                        {relativeTime(record.createdAt)}
                    </Typography>
                    <Typography variant="caption" sx={{ fontSize: 10, color: "text.disabled" }}>·</Typography>
                    <Typography variant="caption" sx={{ fontSize: 10, fontFamily: "var(--font-geist-mono), monospace", color: "text.disabled" }}>
                        {formatDuration(record.duration)}
                    </Typography>
                    {speakers.length > 0 && (
                        <>
                            <Typography variant="caption" sx={{ fontSize: 10, color: "text.disabled" }}>·</Typography>
                            <Stack direction="row" spacing={0.25} sx={{ alignItems: "center" }}>
                                {speakers.slice(0, 4).map((sp) => (
                                    <Box key={sp} sx={{ width: 6, height: 6, borderRadius: "50%", bgcolor: colorMap[sp] }} />
                                ))}
                            </Stack>
                        </>
                    )}
                </Stack>
            </Box>

            <Tooltip title="Delete">
                <span>
                    <IconButton
                        onClick={handleDelete}
                        size="small"
                        sx={{
                            color: "text.disabled",
                            opacity: 0,
                            transition: "opacity 150ms",
                            ".MuiListItemButton-root:hover &": { opacity: 1 },
                            // Touch devices have no hover state — keep the delete
                            // button visible so it's actually reachable on mobile.
                            "@media (hover: none)": { opacity: 1 },
                            "&:hover": { color: "error.main", bgcolor: "rgba(239,68,68,0.08)" },
                        }}
                        aria-label="Delete transcript"
                    >
                        <DeleteOutlinedIcon sx={{ fontSize: 16 }} />
                    </IconButton>
                </span>
            </Tooltip>
        </ListItemButton>
    );
}
