"use client";
import { createTheme } from "@mui/material/styles";

export const theme = createTheme({
    palette: {
        mode: "light",
        primary: { main: "#5b21b6", dark: "#4c1d95", light: "#7c3aed", contrastText: "#ffffff" },
        secondary: { main: "#4f46e5", dark: "#4338ca", light: "#818cf8", contrastText: "#ffffff" },
        error: { main: "#dc2626", light: "#ef4444", dark: "#b91c1c" },
        success: { main: "#059669", light: "#10b981", dark: "#047857" },
        info: { main: "#0ea5e9", light: "#38bdf8", dark: "#0284c7" },
        background: { default: "#f1f4f9", paper: "#ffffff" },
        text: { primary: "#1e1b4b", secondary: "#64748b", disabled: "#94a3b8" },
        divider: "#e2e8f2",
    },
    shape: {
        borderRadius: 12,
    },
    typography: {
        fontFamily: "var(--font-geist-sans), system-ui, -apple-system, sans-serif",
        button: { textTransform: "none", fontWeight: 600 },
        h1: { fontWeight: 700, color: "#1e1b4b" },
        h2: { fontWeight: 700, color: "#1e1b4b" },
        h3: { fontWeight: 700, color: "#1e1b4b" },
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
                root: {
                    borderRadius: 12,
                    backgroundColor: "#ffffff",
                    "& .MuiOutlinedInput-notchedOutline": { borderColor: "#e2e8f2" },
                    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: "#c4b5fd" },
                    "&.Mui-focused .MuiOutlinedInput-notchedOutline": { borderColor: "#6d28d9", borderWidth: 2 },
                },
            },
        },
        MuiInputLabel: {
            styleOverrides: {
                root: { color: "#64748b", "&.Mui-focused": { color: "#6d28d9" } },
            },
        },
        MuiPaper: {
            defaultProps: { elevation: 0 },
            styleOverrides: {
                root: { backgroundImage: "none" },
            },
        },
        MuiCard: {
            defaultProps: { elevation: 0 },
            styleOverrides: {
                root: {
                    borderRadius: 16,
                    border: "1px solid #e2e8f2",
                    backgroundColor: "#ffffff",
                },
            },
        },
        MuiChip: {
            styleOverrides: {
                root: { borderRadius: 8, fontWeight: 500 },
            },
        },
        MuiIconButton: {
            styleOverrides: {
                root: { borderRadius: 8 },
            },
        },
        MuiTooltip: {
            styleOverrides: {
                tooltip: {
                    backgroundColor: "#1e1b4b",
                    fontSize: 12,
                    borderRadius: 8,
                    padding: "6px 10px",
                },
                arrow: { color: "#1e1b4b" },
            },
        },
        MuiDivider: {
            styleOverrides: {
                root: { borderColor: "#e2e8f2" },
            },
        },
    },
});
