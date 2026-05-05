import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome";

interface Props {
    summary: string | null;
}

export default function SummaryBlock({ summary }: Props) {
    const isShort = summary?.trim() === "Not enough conversation to summarize.";
    const bullets =
        summary && !isShort
            ? summary
                .split("\n")
                .map((l) => l.trim())
                .filter((l) => l.length > 0)
                .map((l) => l.replace(/^([•\-*]|\d+[.)]) */, "").trim())
                .filter((l) => l.length > 0)
            : [];

    return (
        <Card sx={{ overflow: "hidden" }}>
            <Box sx={{ height: 3, background: "linear-gradient(to right, var(--mui-palette-primary-dark), var(--mui-palette-primary-main), var(--mui-palette-secondary-main))" }} />
            <CardContent sx={{ p: 2 }}>
                <Stack direction="row" spacing={1} sx={{ alignItems: "center", mb: 1.5 }}>
                    <AutoAwesomeIcon sx={{ fontSize: 14, color: "primary.main" }} />
                    <Typography variant="overline" sx={{ fontWeight: 700, letterSpacing: "0.2em", color: "primary.main", lineHeight: 1 }}>
                        AI Summary
                    </Typography>
                </Stack>

                {bullets.length > 0 ? (
                    <List dense disablePadding>
                        {bullets.map((b, i) => (
                            <ListItem key={i} disableGutters sx={{ alignItems: "flex-start", py: 0.75 }}>
                                <Typography component="span" sx={{ color: "primary.main", fontSize: 12, mt: "3px", mr: 1.25, flexShrink: 0 }}>
                                    ▸
                                </Typography>
                                <Typography variant="body2" sx={{ color: "text.primary", lineHeight: 1.6 }}>
                                    {b}
                                </Typography>
                            </ListItem>
                        ))}
                    </List>
                ) : isShort ? (
                    <Typography variant="body2" sx={{ color: "text.disabled", fontStyle: "italic" }}>
                        Not enough conversation to summarize.
                    </Typography>
                ) : (
                    <Typography variant="body2" sx={{ color: "text.disabled", fontStyle: "italic" }}>
                        No summary available for this session.
                    </Typography>
                )}
            </CardContent>
        </Card>
    );
}
