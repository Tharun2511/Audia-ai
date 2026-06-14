"use client";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Chip from "@mui/material/Chip";
import IconButton from "@mui/material/IconButton";
import List from "@mui/material/List";
import ListItemButton from "@mui/material/ListItemButton";
import Stack from "@mui/material/Stack";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import CloseIcon from "@mui/icons-material/Close";
import type { SearchHit } from "@/app/api/search/route";
import { buildColorMap, formatDuration } from "./utils";

interface Props {
    query: string;
    hits: SearchHit[];
    loading: boolean;
    onClose: () => void;
    onOpenHit: (transcriptionId: string) => void;
}

/**
 * Main-pane view for semantic search results across ALL meetings (Phase 8.1).
 *
 * Unlike chat, there's no LLM here — the ranked chunk hits are shown directly.
 * That's the whole reason ranking quality matters: the user reads result #1
 * first, so the cross-encoder's precision is visible, not hidden behind an LLM.
 *
 * Each hit shows its cross-encoder relevance (the purple % chip), the meeting
 * it came from, the timestamp, and a snippet. Clicking opens that meeting.
 */
export default function SearchResults({ query, hits, loading, onClose, onOpenHit }: Props) {
    return (
        <Stack spacing={2.5} sx={{ animation: "fade-in 250ms ease both" }}>
            {/* Header: query echo + count + close */}
            <Stack direction="row" spacing={1.5} sx={{ alignItems: "flex-start", justifyContent: "space-between" }}>
                <Box sx={{ minWidth: 0 }}>
                    <Typography variant="overline" sx={{ fontWeight: 700, letterSpacing: "0.18em", color: "text.disabled" }}>
                        Search results
                    </Typography>
                    <Typography variant="h6" sx={{ fontWeight: 700, lineHeight: 1.3, wordBreak: "break-word" }}>
                        &ldquo;{query}&rdquo;
                    </Typography>
                    {!loading && (
                        <Typography variant="caption" sx={{ color: "text.secondary" }}>
                            {hits.length === 0
                                ? "No matches across your meetings"
                                : `${hits.length} ${hits.length === 1 ? "moment" : "moments"} · ranked by relevance`}
                        </Typography>
                    )}
                </Box>
                <Tooltip title="Close search">
                    <IconButton onClick={onClose} size="small" aria-label="Close search results">
                        <CloseIcon fontSize="small" />
                    </IconButton>
                </Tooltip>
            </Stack>

            {/* Loading — the app's 5-bar wave idiom */}
            {loading ? (
                <Stack sx={{ alignItems: "center", py: 8 }} spacing={2}>
                    <Box sx={{ display: "flex", alignItems: "flex-end", gap: 0.5, height: 28 }}>
                        {[0, 0.1, 0.2, 0.3, 0.4].map((delay, i) => (
                            <Box
                                key={i}
                                sx={{
                                    width: 5,
                                    height: "100%",
                                    borderRadius: 0.5,
                                    bgcolor: "primary.light",
                                    animation: `bar-wave 0.8s ease-in-out ${delay}s infinite`,
                                }}
                            />
                        ))}
                    </Box>
                    <Typography variant="caption" sx={{ color: "text.disabled" }}>
                        Searching across your meetings…
                    </Typography>
                </Stack>
            ) : hits.length === 0 ? (
                /* Empty — semantic-search-specific guidance */
                <Stack spacing={1.5} sx={{ alignItems: "center", textAlign: "center", py: 8, px: 2 }}>
                    <Typography variant="body2" sx={{ color: "text.secondary", fontWeight: 500 }}>
                        Nothing matched &ldquo;{query}&rdquo;
                    </Typography>
                    <Typography variant="caption" sx={{ color: "text.disabled", maxWidth: 320, lineHeight: 1.6 }}>
                        Semantic search matches meaning, not exact words — try rephrasing, or record a
                        meeting that covers this topic first.
                    </Typography>
                    <Button onClick={onClose} size="small" variant="text" sx={{ mt: 0.5 }}>
                        Close
                    </Button>
                </Stack>
            ) : (
                <List disablePadding sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
                    {hits.map((hit) => {
                        const speakers = [...new Set(hit.speakers)];
                        const colorMap = buildColorMap(speakers);
                        const hasScore = Number.isFinite(hit.relevanceScore);
                        const pct = hasScore ? Math.round(hit.relevanceScore * 100) : null;
                        // Which recall arm(s) surfaced this hit — "both" = found by
                        // semantic AND keyword search (the strongest hybrid signal).
                        const sourceLabel = hit.sources.length >= 2 ? "both" : hit.sources[0] ?? "";

                        return (
                            <ListItemButton
                                key={hit.chunkId}
                                onClick={() => onOpenHit(hit.transcriptionId)}
                                sx={{
                                    display: "block",
                                    px: 2,
                                    py: 1.5,
                                    borderRadius: 2,
                                    border: 1,
                                    borderColor: "divider",
                                    "&:hover": { borderColor: "primary.main", bgcolor: "action.hover" },
                                }}
                            >
                                {/* Top row: meeting title + relevance chip */}
                                <Stack direction="row" spacing={1} sx={{ alignItems: "center", justifyContent: "space-between" }}>
                                    <Typography
                                        variant="body2"
                                        sx={{
                                            fontWeight: 600,
                                            color: hit.meetingTitle ? "text.primary" : "text.disabled",
                                            fontStyle: hit.meetingTitle ? "normal" : "italic",
                                            overflow: "hidden",
                                            textOverflow: "ellipsis",
                                            whiteSpace: "nowrap",
                                            minWidth: 0,
                                        }}
                                    >
                                        {hit.meetingTitle ?? "Untitled session"}
                                    </Typography>
                                    {pct !== null && (
                                        <Chip
                                            label={`${pct}%`}
                                            size="small"
                                            sx={{
                                                flexShrink: 0,
                                                height: 20,
                                                fontSize: 10,
                                                fontWeight: 700,
                                                fontFamily: "var(--font-geist-mono), monospace",
                                                bgcolor: "primary.main",
                                                color: "primary.contrastText",
                                                "& .MuiChip-label": { px: 0.875 },
                                            }}
                                        />
                                    )}
                                </Stack>

                                {/* Snippet — 2-line clamp */}
                                <Typography
                                    variant="body2"
                                    sx={{
                                        mt: 0.75,
                                        color: "text.secondary",
                                        lineHeight: 1.55,
                                        display: "-webkit-box",
                                        WebkitLineClamp: 2,
                                        WebkitBoxOrient: "vertical",
                                        overflow: "hidden",
                                    }}
                                >
                                    {hit.text}
                                </Typography>

                                {/* Footer: timestamp + speaker dots */}
                                <Stack direction="row" spacing={0.75} sx={{ alignItems: "center", mt: 0.75, flexWrap: "wrap" }}>
                                    <Typography
                                        className="tabular-nums"
                                        variant="caption"
                                        sx={{ fontSize: 10, fontFamily: "var(--font-geist-mono), monospace", color: "text.disabled" }}
                                    >
                                        {formatDuration(hit.startTime)}–{formatDuration(hit.endTime)}
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
                                    {sourceLabel && (
                                        <>
                                            <Typography variant="caption" sx={{ fontSize: 10, color: "text.disabled" }}>·</Typography>
                                            <Typography
                                                variant="caption"
                                                sx={{
                                                    fontSize: 10,
                                                    color: sourceLabel === "both" ? "primary.main" : "text.disabled",
                                                    fontWeight: sourceLabel === "both" ? 600 : 400,
                                                }}
                                            >
                                                {sourceLabel}
                                            </Typography>
                                        </>
                                    )}
                                </Stack>
                            </ListItemButton>
                        );
                    })}
                </List>
            )}
        </Stack>
    );
}
