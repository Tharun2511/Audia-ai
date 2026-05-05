"use client";
import { useEffect, useState } from "react";
import IconButton from "@mui/material/IconButton";
import Tooltip from "@mui/material/Tooltip";
import { useColorScheme } from "@mui/material/styles";
import LightModeIcon from "@mui/icons-material/LightModeOutlined";
import DarkModeIcon from "@mui/icons-material/DarkModeOutlined";

/**
 * Toggles between light and dark. Uses MUI's color-scheme machinery:
 *   - Stored in localStorage as `mui-mode`.
 *   - Applied via a class on <html>; CSS vars switch instantly with no re-render.
 *
 * On first paint, useColorScheme() may return undefined (SSR) — render a
 * placeholder so layout doesn't shift on hydration.
 */
export default function ThemeToggle({ size = "small" }: { size?: "small" | "medium" | "large" }) {
    const { mode, systemMode, setMode } = useColorScheme();
    const [mounted, setMounted] = useState(false);
    useEffect(() => setMounted(true), []);

    if (!mounted) {
        return <IconButton size={size} disabled aria-hidden sx={{ visibility: "hidden" }}><LightModeIcon /></IconButton>;
    }

    const resolved = mode === "system" ? systemMode : mode;
    const isDark = resolved === "dark";
    const next = isDark ? "light" : "dark";

    return (
        <Tooltip title={isDark ? "Switch to light" : "Switch to dark"}>
            <IconButton onClick={() => setMode(next)} size={size} aria-label={`Activate ${next} mode`}>
                {isDark ? <LightModeIcon fontSize="small" /> : <DarkModeIcon fontSize="small" />}
            </IconButton>
        </Tooltip>
    );
}
