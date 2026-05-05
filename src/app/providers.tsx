"use client";
import type { ReactNode } from "react";
import { AppRouterCacheProvider } from "@mui/material-nextjs/v15-appRouter";
import { ThemeProvider, useColorScheme } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import { Toaster } from "sonner";
import { theme } from "@/lib/theme";

function ThemedToaster() {
    const { mode, systemMode } = useColorScheme();
    const resolved = mode === "system" ? systemMode : mode;
    return (
        <Toaster
            theme={resolved === "dark" ? "dark" : "light"}
            richColors
            closeButton
            position="top-right"
            toastOptions={{ duration: 4000 }}
        />
    );
}

export default function AppProviders({ children }: { children: ReactNode }) {
    return (
        <AppRouterCacheProvider options={{ key: "mui" }}>
            <ThemeProvider theme={theme} defaultMode="system">
                <CssBaseline />
                {children}
                <ThemedToaster />
            </ThemeProvider>
        </AppRouterCacheProvider>
    );
}
