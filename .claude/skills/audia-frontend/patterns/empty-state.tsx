import type { ReactNode } from "react";
import Box from "@mui/material/Box";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";

interface EmptyStateProps {
    icon?: ReactNode;
    title: string;
    description?: string;
    action?: ReactNode;
}

/**
 * Empty / no-results state. Use anywhere a list is empty by design or by filtering.
 *
 * Visual rules:
 * - Centered in its container, never edge-aligned.
 * - Icon is decorative — pass an MUI icon with sx={{ fontSize: 48 }} when calling.
 * - One primary action max. Two = decision fatigue.
 */
export default function EmptyState({ icon, title, description, action }: EmptyStateProps) {
    return (
        <Box
            sx={{
                py: 6,
                px: 3,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
            }}
        >
            <Stack spacing={2} sx={{ alignItems: "center", textAlign: "center", maxWidth: 360 }}>
                {icon && (
                    <Box
                        aria-hidden
                        sx={{
                            color: "text.disabled",
                            display: "inline-flex",
                            lineHeight: 1,
                        }}
                    >
                        {icon}
                    </Box>
                )}
                <Typography variant="h6" sx={{ color: "text.primary" }}>
                    {title}
                </Typography>
                {description && (
                    <Typography variant="body2" sx={{ color: "text.secondary" }}>
                        {description}
                    </Typography>
                )}
                {action && <Box sx={{ mt: 1 }}>{action}</Box>}
            </Stack>
        </Box>
    );
}
