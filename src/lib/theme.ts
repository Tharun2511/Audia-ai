"use client";
import { createTheme } from "@mui/material/styles";

/**
 * Audia theme — light + dark color schemes.
 *
 * IMPORTANT: with `cssVariables: { colorSchemeSelector: "class" }`, every
 * component override below uses `theme.vars.palette.*` instead of
 * `theme.palette.*`. The `vars` form returns a CSS variable string like
 * `var(--mui-palette-background-paper)` which the browser re-resolves when
 * the .mui-color-scheme-* class changes. Using `palette.*` directly would
 * bake in the static light-mode hex at theme-construction time and never
 * swap.
 */
export const theme = createTheme({
    cssVariables: { colorSchemeSelector: "class" },
    colorSchemes: {
        light: {
            palette: {
                primary: { main: "#5b21b6", dark: "#4c1d95", light: "#7c3aed", contrastText: "#ffffff" },
                secondary: { main: "#4f46e5", dark: "#4338ca", light: "#818cf8", contrastText: "#ffffff" },
                error: { main: "#dc2626", light: "#ef4444", dark: "#b91c1c" },
                success: { main: "#059669", light: "#10b981", dark: "#047857" },
                info: { main: "#0ea5e9", light: "#38bdf8", dark: "#0284c7" },
                background: { default: "#f1f4f9", paper: "#ffffff" },
                text: { primary: "#1e1b4b", secondary: "#64748b", disabled: "#94a3b8" },
                divider: "#e2e8f2",
            },
        },
        dark: {
            palette: {
                primary: { main: "#a78bfa", dark: "#8b5cf6", light: "#c4b5fd", contrastText: "#1e1b4b" },
                secondary: { main: "#818cf8", dark: "#6366f1", light: "#a5b4fc", contrastText: "#1e1b4b" },
                error: { main: "#f87171", light: "#fca5a5", dark: "#ef4444" },
                success: { main: "#34d399", light: "#6ee7b7", dark: "#10b981" },
                info: { main: "#38bdf8", light: "#7dd3fc", dark: "#0ea5e9" },
                background: { default: "#0b0a17", paper: "#15132a" },
                text: { primary: "#f1f4f9", secondary: "#cbd5e1", disabled: "#64748b" },
                divider: "rgba(167,139,250,0.16)",
            },
        },
    },
    shape: { borderRadius: 12 },
    typography: {
        fontFamily: "var(--font-geist-sans), system-ui, -apple-system, sans-serif",
        button: { textTransform: "none", fontWeight: 600 },
        h1: { fontWeight: 700 },
        h2: { fontWeight: 700 },
        h3: { fontWeight: 700 },
    },
    components: {
        MuiButton: {
            defaultProps: { disableElevation: true },
            styleOverrides: {
                root: { borderRadius: 12, paddingTop: 10, paddingBottom: 10 },
            },
        },
        MuiTextField: {
            defaultProps: { variant: "outlined", fullWidth: true, size: "medium" },
        },
        MuiOutlinedInput: {
            styleOverrides: {
                root: ({ theme: t }) => ({
                    borderRadius: 12,
                    backgroundColor: t.vars.palette.background.paper,
                    "& .MuiOutlinedInput-notchedOutline": { borderColor: t.vars.palette.divider },
                    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: t.vars.palette.primary.light },
                    "&.Mui-focused .MuiOutlinedInput-notchedOutline": { borderColor: t.vars.palette.primary.main, borderWidth: 2 },
                }),
            },
        },
        MuiInputLabel: {
            styleOverrides: {
                root: ({ theme: t }) => ({ color: t.vars.palette.text.secondary, "&.Mui-focused": { color: t.vars.palette.primary.main } }),
            },
        },
        MuiPaper: {
            defaultProps: { elevation: 0 },
            styleOverrides: { root: { backgroundImage: "none" } },
        },
        MuiCard: {
            defaultProps: { elevation: 0 },
            styleOverrides: {
                root: ({ theme: t }) => ({
                    borderRadius: 16,
                    border: `1px solid ${t.vars.palette.divider}`,
                    backgroundColor: t.vars.palette.background.paper,
                }),
            },
        },
        MuiChip: {
            styleOverrides: { root: { borderRadius: 8, fontWeight: 500 } },
        },
        MuiIconButton: {
            styleOverrides: { root: { borderRadius: 8 } },
        },
        MuiTooltip: {
            styleOverrides: {
                tooltip: ({ theme: t }) => ({
                    backgroundColor: t.vars.palette.text.primary,
                    color: t.vars.palette.background.paper,
                    fontSize: 12,
                    borderRadius: 8,
                    padding: "6px 10px",
                }),
                arrow: ({ theme: t }) => ({ color: t.vars.palette.text.primary }),
            },
        },
        MuiDivider: {
            styleOverrides: {
                root: ({ theme: t }) => ({ borderColor: t.vars.palette.divider }),
            },
        },
    },
});
