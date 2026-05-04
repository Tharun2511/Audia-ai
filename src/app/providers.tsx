"use client";
import type { ReactNode } from "react";
import { Toaster } from "sonner";

export default function AppProviders({ children }: { children: ReactNode }) {
    return (
        <>
            {children}
            <Toaster
                richColors
                closeButton
                position="top-right"
                toastOptions={{ duration: 4000 }}
            />
        </>
    );
}
