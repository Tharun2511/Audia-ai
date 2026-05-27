import type { ReactNode } from "react";
import Box from "@mui/material/Box";

/**
 * Caps main-pane content at 1440px and centers on large screens.
 * Use as the outermost wrapper inside any main-pane view that isn't the sidebar.
 *
 * Per-surface narrower caps (transcript = 720, settings = 720) go INSIDE
 * this container, not as a replacement for it.
 */
export default function ShellContainer({ children }: { children: ReactNode }) {
    return (
        <Box
            sx={{
                width: "100%",
                maxWidth: 1440,
                mx: "auto",
                px: { xs: 2, sm: 3, md: 4 },
            }}
        >
            {children}
        </Box>
    );
}
