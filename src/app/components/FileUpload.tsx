"use client";
import { useState, useRef, useEffect } from "react";
import Box from "@mui/material/Box";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import LibraryMusicIcon from "@mui/icons-material/LibraryMusic";

interface Props {
    onUpload: (file: File) => Promise<void>;
    disabled: boolean;
}

const ACCEPTED = ["audio/mpeg", "audio/wav", "audio/webm", "audio/mp4", "audio/ogg", "audio/flac", "audio/x-m4a"];

function isAudioFile(file: File): boolean {
    return ACCEPTED.includes(file.type) || /\.(mp3|wav|webm|m4a|ogg|flac)$/i.test(file.name);
}

export default function FileUpload({ onUpload, disabled }: Props) {
    const [dragging, setDragging] = useState(false);
    const [fileName, setFileName] = useState<string | null>(null);
    const [formatError, setFormatError] = useState(false);
    const inputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        if (!disabled) {
            setFileName(null);
            setFormatError(false);
        }
    }, [disabled]);

    const handleFile = async (file: File) => {
        if (!isAudioFile(file)) {
            setFormatError(true);
            return;
        }
        setFormatError(false);
        setFileName(file.name);
        await onUpload(file);
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setDragging(false);
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
    };

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        if (!disabled) setDragging(true);
    };

    return (
        <Box sx={{ width: "100%" }}>
            <Box
                role="button"
                tabIndex={disabled ? -1 : 0}
                onClick={() => !disabled && inputRef.current?.click()}
                onKeyDown={(e) => e.key === "Enter" && !disabled && inputRef.current?.click()}
                onDragOver={handleDragOver}
                onDragEnter={handleDragOver}
                onDragLeave={() => setDragging(false)}
                onDrop={handleDrop}
                sx={{
                    width: "100%",
                    borderRadius: 3,
                    border: "2px dashed",
                    px: 2,
                    py: 2.5,
                    textAlign: "center",
                    transition: "all 150ms",
                    userSelect: "none",
                    outline: "none",
                    ...(disabled
                        ? { opacity: 0.4, cursor: "not-allowed", borderColor: "divider" }
                        : dragging
                            ? { borderColor: "primary.main", bgcolor: "action.hover", cursor: "copy" }
                            : {
                                borderColor: "divider",
                                cursor: "pointer",
                                "&:hover": { borderColor: "primary.light", bgcolor: "action.hover" },
                                "&:focus-visible": { borderColor: "primary.main" },
                            }),
                }}
            >
                <input
                    ref={inputRef}
                    type="file"
                    accept=".mp3,.wav,.webm,.m4a,.ogg,.flac,audio/*"
                    style={{ display: "none" }}
                    disabled={disabled}
                    onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file) handleFile(file);
                        e.target.value = "";
                    }}
                />

                {fileName ? (
                    <Stack spacing={0.5} sx={{ alignItems: "center" }}>
                        <Box sx={{ display: "flex", alignItems: "flex-end", gap: "3px", height: 16, mb: 0.5 }}>
                            {[0, 0.1, 0.2].map((delay, i) => (
                                <Box
                                    key={i}
                                    sx={{
                                        width: 4,
                                        height: "100%",
                                        borderRadius: 0.5,
                                        bgcolor: "primary.main",
                                        animation: `bar-wave 0.7s ease-in-out ${delay}s infinite`,
                                    }}
                                />
                            ))}
                        </Box>
                        <Typography
                            variant="caption"
                            sx={{
                                fontWeight: 500,
                                color: "primary.main",
                                maxWidth: "100%",
                                px: 1,
                                overflow: "hidden",
                                textOverflow: "ellipsis",
                                whiteSpace: "nowrap",
                            }}
                        >
                            {fileName}
                        </Typography>
                        <Typography variant="caption" sx={{ fontSize: 10, color: "text.disabled" }}>
                            Uploading…
                        </Typography>
                    </Stack>
                ) : dragging ? (
                    <Stack spacing={0.5} sx={{ alignItems: "center" }}>
                        <LibraryMusicIcon sx={{ fontSize: 28, color: "primary.main" }} />
                        <Typography variant="caption" sx={{ fontWeight: 600, color: "primary.main" }}>
                            Drop to upload
                        </Typography>
                    </Stack>
                ) : (
                    <Stack spacing={0.5} sx={{ alignItems: "center" }}>
                        <UploadFileIcon sx={{ fontSize: 24, color: "text.disabled" }} />
                        <Typography variant="caption" sx={{ fontWeight: 500, color: "text.secondary" }}>
                            Drop audio file or{" "}
                            <Box component="span" sx={{ color: "primary.main" }}>
                                browse
                            </Box>
                        </Typography>
                        <Typography variant="caption" sx={{ fontSize: 10, color: "text.disabled" }}>
                            mp3 · wav · webm · m4a · ogg
                        </Typography>
                    </Stack>
                )}
            </Box>

            {formatError && (
                <Typography variant="caption" sx={{ display: "block", color: "error.main", mt: 0.75, textAlign: "center", fontSize: 10 }}>
                    Unsupported file type. Please upload an audio file.
                </Typography>
            )}
        </Box>
    );
}
