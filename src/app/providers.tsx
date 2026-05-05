"use client";
import type { ReactNode } from "react";
import { AppRouterCacheProvider } from "@mui/material-nextjs/v15-appRouter";
import { ThemeProvider } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import { Toaster } from "sonner";
import { theme } from "@/lib/theme";

export default function AppProviders({ children }: { children: ReactNode }) {
    return (
        <AppRouterCacheProvider options={{ key: "mui" }}>
            <ThemeProvider theme={theme}>
                <CssBaseline />
                {children}
                <Toaster
                    richColors
                    closeButton
                    position="top-right"
                    toastOptions={{ duration: 4000 }}
                />
            </ThemeProvider>
        </AppRouterCacheProvider>
    );
}
