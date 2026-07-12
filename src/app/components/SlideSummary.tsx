"use client";
import { useRef, useState } from "react";
import { toast } from "sonner";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import Collapse from "@mui/material/Collapse";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import CircularProgress from "@mui/material/CircularProgress";
import AddPhotoAlternateOutlinedIcon from "@mui/icons-material/AddPhotoAlternateOutlined";

interface Props {
    transcriptionId: string;
}

type SlideResult = { slideText: string; jointSummary: string | null };

/** Read a File into raw base64 (strip the "data:...;base64," prefix). */
function fileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            const result = reader.result as string;
            resolve(result.slice(result.indexOf(",") + 1));
        };
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(file);
    });
}

/**
 * Multimodal slide summary (Phase 11.1). Upload a slide image shown during the
 * meeting; a vision model reads it and it's fused with the transcript into a
 * joint summary. Purely additive to the session view.
 */
export default function SlideSummary({ transcriptionId }: Props) {
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<SlideResult | null>(null);
    const [showSlideText, setShowSlideText] = useState(false);
    const inputRef = useRef<HTMLInputElement>(null);

    const handleFile = async (file: File | undefined) => {
        if (!file) return;
        if (!file.type.startsWith("image/")) {
            toast.error("Please choose an image (PNG, JPG, …).");
            return;
        }
        if (file.size > 20 * 1024 * 1024) {
            toast.error("Image is too large (max 20 MB).");
            return;
        }
        setLoading(true);
        setResult(null);
        try {
            const imageBase64 = await fileToBase64(file);
            const res = await fetch(`/api/transcriptions/${transcriptionId}/slide-summary`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ imageBase64, mimeType: file.type }),
            });
            if (!res.ok) throw new Error(`Server returned ${res.status}`);
            const data: SlideResult = await res.json();
            setResult(data);
        } catch {
            toast.error("Couldn't summarize that slide. Try again.");
        } finally {
            setLoading(false);
            if (inputRef.current) inputRef.current.value = ""; // allow re-selecting the same file
        }
    };

    return (
        <Card sx={{ overflow: "hidden" }}>
            <Stack
                direction="row"
                spacing={1}
                sx={{ alignItems: "center", justifyContent: "space-between", px: 2, py: 1.75, borderBottom: 1, borderColor: "divider" }}
            >
                <Typography variant="overline" sx={{ fontWeight: 700, letterSpacing: "0.2em", color: "text.disabled", lineHeight: 1 }}>
                    Slides
                </Typography>
                <Button
                    onClick={() => inputRef.current?.click()}
                    variant="outlined"
                    size="small"
                    disabled={loading}
                    startIcon={loading ? <CircularProgress size={14} sx={{ color: "inherit" }} /> : <AddPhotoAlternateOutlinedIcon />}
                    sx={{ color: "text.secondary", flexShrink: 0 }}
                >
                    {loading ? "Reading slide…" : result ? "Add another slide" : "Add a slide"}
                </Button>
                <input
                    ref={inputRef}
                    type="file"
                    accept="image/*"
                    hidden
                    onChange={(e) => handleFile(e.target.files?.[0])}
                />
            </Stack>

            <Box sx={{ p: 2 }}>
                {!result && !loading && (
                    <Typography variant="body2" sx={{ color: "text.disabled", fontStyle: "italic" }}>
                        Add a slide image shown during this meeting — it&apos;ll be read and combined with the transcript into a joint summary.
                    </Typography>
                )}

                {result && (
                    <Stack spacing={1.5}>
                        <Box>
                            <Typography variant="caption" sx={{ fontWeight: 700, color: "primary.main", display: "block", mb: 0.5 }}>
                                Joint summary (transcript + slide)
                            </Typography>
                            <Typography variant="body2" sx={{ color: "text.primary", lineHeight: 1.65, whiteSpace: "pre-wrap" }}>
                                {result.jointSummary ?? "Couldn't generate a joint summary for this slide."}
                            </Typography>
                        </Box>

                        <Box>
                            <Button
                                onClick={() => setShowSlideText((v) => !v)}
                                size="small"
                                variant="text"
                                sx={{ px: 0, minHeight: 0, fontSize: 12, color: "text.secondary" }}
                            >
                                {showSlideText ? "Hide" : "Show"} what the slide said
                            </Button>
                            <Collapse in={showSlideText}>
                                <Typography
                                    variant="body2"
                                    sx={{ mt: 0.75, p: 1.5, borderRadius: 2, bgcolor: "action.hover", color: "text.secondary", lineHeight: 1.6, whiteSpace: "pre-wrap" }}
                                >
                                    {result.slideText}
                                </Typography>
                            </Collapse>
                        </Box>
                    </Stack>
                )}
            </Box>
        </Card>
    );
}
