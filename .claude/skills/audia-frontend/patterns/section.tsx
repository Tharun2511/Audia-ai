import type { ReactNode } from "react";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";

interface SectionProps {
    title: string;
    description?: string;
    action?: ReactNode;
    children: ReactNode;
}

/**
 * A page section with title + optional description + optional action.
 * Use on settings pages, dashboards, the public session page, and anywhere
 * content stacks into labeled blocks.
 *
 * Compose with `Row` (below) for label/value/action rows.
 */
export function Section({ title, description, action, children }: SectionProps) {
    return (
        <Stack component="section" spacing={2} sx={{ maxWidth: 720, width: "100%" }}>
            <Stack direction="row" spacing={2} sx={{ alignItems: "flex-start" }}>
                <Box sx={{ flex: 1 }}>
                    <Typography variant="h6" sx={{ color: "text.primary", lineHeight: 1.3 }}>
                        {title}
                    </Typography>
                    {description && (
                        <Typography variant="body2" sx={{ color: "text.secondary", mt: 0.5 }}>
                            {description}
                        </Typography>
                    )}
                </Box>
                {action}
            </Stack>
            <Card sx={{ overflow: "hidden" }}>
                {children}
            </Card>
        </Stack>
    );
}

interface RowProps {
    label: string;
    value?: ReactNode;
    action?: ReactNode;
}

/** A single label/value/action row inside a Section. Auto-divides between rows. */
export function Row({ label, value, action }: RowProps) {
    return (
        <Stack
            direction={{ xs: "column", sm: "row" }}
            spacing={1}
            sx={{
                alignItems: { sm: "center" },
                justifyContent: "space-between",
                px: 2.5,
                py: 2,
                borderBottom: 1,
                borderColor: "divider",
                "&:last-of-type": { borderBottom: 0 },
            }}
        >
            <Box>
                <Typography variant="body2" sx={{ fontWeight: 600, color: "text.primary" }}>
                    {label}
                </Typography>
                {typeof value !== "undefined" && (
                    <Typography variant="body2" sx={{ color: "text.secondary", mt: 0.25 }}>
                        {value}
                    </Typography>
                )}
            </Box>
            {action}
        </Stack>
    );
}
