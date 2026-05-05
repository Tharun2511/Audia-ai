"use client";
import Box from "@mui/material/Box";
import Chip from "@mui/material/Chip";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import type { TranscriptSegment } from "@/entity/Transcription";
import { formatDuration } from "./utils";
import SummaryBlock from "./SummaryBlock";
import TranscriptPanel from "./TranscriptPanel";
import ChatPanel from "./ChatPanel";

interface Props {
    id: string;
    title: string | null;
    summary: string | null;
    duration: number;
    segments: TranscriptSegment[];
    createdAt: string;
    isJustCompleted?: boolean;
    onTitleSave: (title: string | null) => void;
    onSegmentsUpdate: (segments: TranscriptSegment[]) => void;
    onSummaryUpdate: (summary: string | null) => void;
}

export default function SessionView({
    id,
    title,
    summary,
    duration,
    segments,
    createdAt,
    isJustCompleted,
    onTitleSave,
    onSegmentsUpdate,
    onSummaryUpdate,
}: Props) {
    const date = new Date(createdAt);

    return (
        <Stack spacing={2.5} sx={{ animation: "fade-in 250ms ease both" }}>
            {/* Header strip */}
            <Box>
                <Stack direction="row" spacing={1.5} sx={{ alignItems: "center", flexWrap: "wrap" }}>
                    <Typography variant="caption" sx={{ fontSize: 10, color: "text.disabled", fontFamily: "var(--font-geist-mono), monospace" }}>
                        {formatDuration(duration)}
                    </Typography>
                    <Typography variant="caption" sx={{ fontSize: 10, color: "text.disabled" }}>·</Typography>
                    <Typography variant="caption" sx={{ fontSize: 10, color: "text.disabled" }}>
                        {date.toLocaleString()}
                    </Typography>
                    {isJustCompleted && (
                        <Chip
                            label="Just completed"
                            size="small"
                            sx={{ height: 18, fontSize: 10, bgcolor: "primary.main", color: "primary.contrastText", "& .MuiChip-label": { px: 1 } }}
                        />
                    )}
                </Stack>
            </Box>

            {/* Summary */}
            <SummaryBlock
                transcriptionId={id}
                initialSummary={summary}
                onSaved={onSummaryUpdate}
            />

            {/* Transcript */}
            <TranscriptPanel
                segments={segments}
                transcriptionId={id}
                currentTitle={title}
                onSegmentsUpdate={onSegmentsUpdate}
                onTitleSave={onTitleSave}
            />

            {/* Chat */}
            {segments.length > 0 && (
                <ChatPanel
                    segments={segments}
                    transcriptionId={id}
                    currentTitle={title}
                    onTitleSave={onTitleSave}
                />
            )}
        </Stack>
    );
}
