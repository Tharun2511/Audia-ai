"use client";
import NextLink from "next/link";
import { useActionState, useEffect } from "react";
import { toast } from "sonner";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Link from "@mui/material/Link";
import Stack from "@mui/material/Stack";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";
import { login } from "@/app/actions/auth";
import PasswordInput from "./PasswordInput";

export default function LoginForm() {
    const [state, action, pending] = useActionState(login, undefined);

    useEffect(() => {
        if (!state) return;
        if (state.message) {
            toast.error(state.message);
        } else if (state.errors && Object.keys(state.errors).length > 0) {
            toast.error("Please fix the highlighted fields.");
        }
    }, [state]);

    return (
        <Box component="form" action={action} noValidate>
            <Stack spacing={2.5}>
                <TextField
                    label="Email"
                    name="email"
                    type="email"
                    placeholder="you@example.com"
                    defaultValue={state?.values?.email}
                    autoComplete="email"
                    required
                    error={!!state?.errors?.email?.length}
                    helperText={state?.errors?.email?.[0]}
                />
                <PasswordInput
                    label="Password"
                    name="password"
                    autoComplete="current-password"
                    required
                    error={!!state?.errors?.password?.length}
                    helperText={state?.errors?.password?.[0]}
                />

                <Button
                    type="submit"
                    variant="contained"
                    color="primary"
                    size="large"
                    disabled={pending}
                    fullWidth
                >
                    {pending ? "Logging in…" : "Log in →"}
                </Button>

                <Typography variant="body2" align="center" color="text.secondary">
                    Don&apos;t have an account?{" "}
                    <Link component={NextLink} href="/signup" sx={{ color: "primary.main", fontWeight: 500, textDecoration: "none", "&:hover": { textDecoration: "underline" } }}>
                        Sign up
                    </Link>
                </Typography>
            </Stack>
        </Box>
    );
}
