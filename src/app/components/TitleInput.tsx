"use client";
import { useState } from "react";
import { toast } from "sonner";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import IconButton from "@mui/material/IconButton";
import Stack from "@mui/material/Stack";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";
import EditIcon from "@mui/icons-material/Edit";
import CloseIcon from "@mui/icons-material/Close";

interface Props {
    transcriptionId: string;
    initialTitle: string | null;
    fallback: string;
    onSaved: (title: string | null) => void;
}

export default function TitleInput({ transcriptionId, initialTitle, fallback, onSaved }: Props) {
    const [editing, setEditing] = useState(false);
    const [value, setValue] = useState(initialTitle ?? "");
    const [saving, setSaving] = useState(false);

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
                sx={{ alignItems: "center" }}
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
                    size="small"
                    sx={{ width: 180 }}
                />
                <Button onClick={save} disabled={saving} variant="contained" size="small">
                    {saving ? "…" : "Save"}
                </Button>
                <IconButton onClick={cancel} size="small" aria-label="Cancel">
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
            sx={{
                bgcolor: "transparent",
                border: 0,
                p: 0,
                cursor: "pointer",
                display: "inline-flex",
                alignItems: "center",
                gap: 0.75,
                textAlign: "left",
                color: "inherit",
                "&:hover .edit-pencil": { color: "#c4b5fd" },
            }}
        >
            <Typography
                variant="body2"
                sx={{
                    fontWeight: 600,
                    color: initialTitle ? "text.primary" : "text.disabled",
                    fontStyle: initialTitle ? "normal" : "italic",
                }}
            >
                {initialTitle ?? fallback}
            </Typography>
            <EditIcon className="edit-pencil" sx={{ fontSize: 12, color: "transparent", transition: "color 150ms" }} />
        </Box>
    );
}
