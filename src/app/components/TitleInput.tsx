"use client";
import { useState } from "react";
import { toast } from "sonner";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import IconButton from "@mui/material/IconButton";
import Stack from "@mui/material/Stack";
import TextField from "@mui/material/TextField";
import Typography, { type TypographyProps } from "@mui/material/Typography";
import EditIcon from "@mui/icons-material/Edit";
import CloseIcon from "@mui/icons-material/Close";

type Variant = "default" | "title";

interface Props {
    transcriptionId: string;
    initialTitle: string | null;
    fallback: string;
    onSaved: (title: string | null) => void;
    /**
     * "default" — inline body-text size for use inside card headers.
     * "title"   — large session title for use at the top of SessionView.
     */
    variant?: Variant;
}

const STYLE: Record<Variant, {
    typographyVariant: TypographyProps["variant"];
    typographyWeight: number;
    fieldWidth: number | string;
    fieldSize: "small" | "medium";
    iconSize: number;
}> = {
    default: { typographyVariant: "body2", typographyWeight: 600, fieldWidth: 200, fieldSize: "small", iconSize: 12 },
    title:   { typographyVariant: "h5",    typographyWeight: 700, fieldWidth: "100%", fieldSize: "medium", iconSize: 18 },
};

export default function TitleInput({ transcriptionId, initialTitle, fallback, onSaved, variant = "default" }: Props) {
    const [editing, setEditing] = useState(false);
    const [value, setValue] = useState(initialTitle ?? "");
    const [saving, setSaving] = useState(false);
    const style = STYLE[variant];

    const save = async () => {
        const trimmed = value.trim();
        setSaving(true);
        try {
            const res = await fetch(`/api/transcriptions/${transcriptionId}`, {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ title: trimmed || null }),
            });
            if (!res.ok) throw new Error(`Server returned ${res.status}`);
            onSaved(trimmed || null);
            setEditing(false);
        } catch {
            toast.error("Couldn't save the title.");
        } finally {
            setSaving(false);
        }
    };

    const cancel = () => {
        setValue(initialTitle ?? "");
        setEditing(false);
    };

    if (editing) {
        return (
            <Stack
                direction="row"
                spacing={1}
                sx={{ alignItems: "center", width: variant === "title" ? "100%" : undefined }}
                onClick={(e) => e.stopPropagation()}
            >
                <TextField
                    autoFocus
                    value={value}
                    onChange={(e) => setValue(e.target.value)}
                    onKeyDown={(e) => {
                        if (e.key === "Enter") save();
                        if (e.key === "Escape") cancel();
                    }}
                    placeholder="Name this session…"
                    size={style.fieldSize}
                    sx={{ width: style.fieldWidth, maxWidth: variant === "title" ? 480 : undefined }}
                />
                <Button onClick={save} disabled={saving} variant="contained" size={style.fieldSize}>
                    {saving ? "…" : "Save"}
                </Button>
                <IconButton onClick={cancel} size={style.fieldSize} aria-label="Cancel">
                    <CloseIcon fontSize="small" />
                </IconButton>
            </Stack>
        );
    }

    return (
        <Box
            component="button"
            type="button"
            onClick={(e: React.MouseEvent) => { e.stopPropagation(); setEditing(true); }}
            aria-label={initialTitle ? `Rename "${initialTitle}"` : "Name this session"}
            sx={{
                bgcolor: "transparent",
                border: 0,
                p: 0,
                cursor: "pointer",
                display: "inline-flex",
                alignItems: "center",
                gap: variant === "title" ? 1.25 : 0.75,
                textAlign: "left",
                color: "inherit",
                "&:hover .edit-pencil": { color: "primary.main", opacity: 1 },
                "&:focus-visible": {
                    outline: "2px solid",
                    outlineColor: "primary.main",
                    outlineOffset: 4,
                    borderRadius: 4,
                },
            }}
        >
            <Typography
                variant={style.typographyVariant}
                sx={{
                    fontWeight: style.typographyWeight,
                    color: initialTitle ? "text.primary" : "text.disabled",
                    fontStyle: initialTitle ? "normal" : "italic",
                    lineHeight: 1.2,
                }}
            >
                {initialTitle ?? fallback}
            </Typography>
            {/*
             * Pencil is subtly visible by default (35% opacity), brightens on
             * row-hover for desktop, and is always visible on touch devices
             * via @media (hover: none). Discovery without clutter.
             */}
            <EditIcon
                className="edit-pencil"
                sx={{
                    fontSize: style.iconSize,
                    color: "text.disabled",
                    opacity: 0.35,
                    transition: "opacity 150ms, color 150ms",
                    "@media (hover: none)": { opacity: 1 },
                }}
            />
        </Box>
    );
}
