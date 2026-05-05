import { redirect } from "next/navigation";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import { readSession } from "@/lib/session";
import { BrandLogo } from "@/app/components/BrandLogo";
import LoginForm from "@/app/components/LoginForm";

export default async function LoginPage() {
    const session = await readSession();
    if (session?.userId) redirect("/");

    return (
        <Box
            sx={{
                minHeight: "100vh",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                bgcolor: "background.default",
                px: 2,
                py: 5,
            }}
        >
            <Card sx={{ width: "100%", maxWidth: 440, overflow: "hidden", boxShadow: "0 20px 50px -20px rgba(91,33,182,0.18)" }}>
                <Box sx={{ height: 4, background: "linear-gradient(to right, #5b21b6, #4f46e5)" }} />
                <CardContent sx={{ p: 4 }}>
                    <Stack direction="row" spacing={1.25} sx={{ alignItems: "center", mb: 3 }}>
                        <BrandLogo size={32} />
                        <Typography variant="h6" component="span" sx={{ fontWeight: 700 }}>
                            Audia
                        </Typography>
                    </Stack>

                    <Typography variant="h5" sx={{ fontWeight: 700 }}>
                        Welcome back
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5, mb: 3 }}>
                        Log in to access your transcripts.
                    </Typography>

                    <LoginForm />
                </CardContent>
            </Card>
        </Box>
    );
}
