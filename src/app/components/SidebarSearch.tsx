"use client";
import { useEffect, useRef } from "react";
import IconButton from "@mui/material/IconButton";
import InputAdornment from "@mui/material/InputAdornment";
import TextField from "@mui/material/TextField";
import ClearIcon from "@mui/icons-material/Close";
import SearchIcon from "@mui/icons-material/Search";

interface Props {
    value: string;
    onChange: (value: string) => void;
}

/**
 * Sidebar session-search input. Owns its own ref and the global `/` shortcut
 * so the parent doesn't have to. The shortcut is ignored while the user is
 * typing in any input/textarea/contenteditable so it never steals keystrokes.
 */
export default function SidebarSearch({ value, onChange }: Props) {
    const inputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            if (e.key !== "/") return;
            const t = e.target as HTMLElement | null;
            if (
                t &&
                (t.tagName === "INPUT" ||
                    t.tagName === "TEXTAREA" ||
                    t.isContentEditable)
            ) {
                return;
            }
            e.preventDefault();
            inputRef.current?.focus();
            inputRef.current?.select();
        };
        window.addEventListener("keydown", handler);
        return () => window.removeEventListener("keydown", handler);
    }, []);

    return (
        <TextField
            inputRef={inputRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={(e) => {
                if (e.key === "Escape" && value) {
                    e.preventDefault();
                    onChange("");
                }
            }}
            placeholder="Search sessions…"
            size="small"
            type="search"
            slotProps={{
                input: {
                    startAdornment: (
                        <InputAdornment position="start">
                            <SearchIcon sx={{ fontSize: 18, color: "text.disabled" }} />
                        </InputAdornment>
                    ),
                    endAdornment: value ? (
                        <InputAdornment position="end">
                            <IconButton
                                onClick={() => {
                                    onChange("");
                                    inputRef.current?.focus();
                                }}
                                size="small"
                                edge="end"
                                aria-label="Clear search"
                                sx={{ color: "text.disabled" }}
                            >
                                <ClearIcon sx={{ fontSize: 16 }} />
                            </IconButton>
                        </InputAdornment>
                    ) : null,
                },
            }}
            sx={{
                "& .MuiOutlinedInput-root": {
                    fontSize: 13,
                    bgcolor: "action.hover",
                    "& fieldset": { border: "none" },
                },
            }}
        />
    );
}
