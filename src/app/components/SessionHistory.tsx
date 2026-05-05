"use client";
import Box from "@mui/material/Box";
import Divider from "@mui/material/Divider";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import type { TranscriptSegment } from "@/entity/Transcription";
import HistoryEntry from "./HistoryEntry";

type HistoryRecord = {
    id: string;
    title: string | null;
    summary: string | null;
    duration: number;
    segments: TranscriptSegment[];
    createdAt: string;
};

interface Props {
    history: HistoryRecord[];
    loading: boolean;
    loaded: boolean;
    onDelete: (id: string) => void;
    onSegmentsUpdate: (id: string, segments: TranscriptSegment[]) => void;
    onTitleUpdate: (id: string, title: string | null) => void;
    onSummaryUpdate: (id: string, summary: string | null) => void;
}

export default function SessionHistory({
    history,
    loading,
    loaded,
    onDelete,
    onSegmentsUpdate,
    onTitleUpdate,
    onSummaryUpdate,
}: Props) {
    if (!loading && !loaded) return null;

    return (
        <Box component="section">
            <Stack direction="row" spacing={1.5} sx={{ alignItems: "center", mb: 2 }}>
                <Typography variant="overline" component="h2" sx={{ fontWeight: 700, letterSpacing: "0.25em", color: "text.disabled", lineHeight: 1, flexShrink: 0 }}>
                    Session Archive
                </Typography>
                <Divider sx={{ flex: 1 }} />
                {!loading && (
                    <Typography variant="caption" sx={{ fontSize: 10, color: "text.disabled", flexShrink: 0 }}>
                        {history.length} {history.length === 1 ? "session" : "sessions"}
                    </Typography>
                )}
            </Stack>

            {loading ? (
                <Stack direction="row" sx={{ alignItems: "center", justifyContent: "center", py: 8 }}>
                    <Box sx={{ display: "flex", alignItems: "flex-end", gap: 0.5, height: 32 }}>
                        {[0, 0.1, 0.2, 0.3, 0.4].map((delay, i) => (
                            <Box
                                key={i}
                                sx={{
                                    width: 6,
                                    height: "100%",
                                    borderRadius: 0.5,
                                    bgcolor: "#ddd6fe",
                                    animation: `bar-wave 0.8s ease-in-out ${delay}s infinite`,
                                }}
                            />
                        ))}
                    </Box>
                </Stack>
            ) : history.length === 0 ? (
                <Box sx={{ border: "2px dashed", borderColor: "divider", borderRadius: 3, p: 5, textAlign: "center" }}>
                    <Typography variant="body2" color="text.disabled">
                        No sessions recorded yet.
                    </Typography>
                    <Typography variant="caption" color="text.disabled" sx={{ display: "block", mt: 0.5 }}>
                        Hit record in the sidebar to get started.
                    </Typography>
                </Box>
            ) : (
                <Stack spacing={1}>
                    {history.map((record) => (
                        <HistoryEntry
                            key={record.id}
                            record={record}
                            onDelete={onDelete}
                            onSegmentsUpdate={onSegmentsUpdate}
                            onTitleUpdate={onTitleUpdate}
                            onSummaryUpdate={onSummaryUpdate}
                        />
                    ))}
                </Stack>
            )}
        </Box>
    );
}
