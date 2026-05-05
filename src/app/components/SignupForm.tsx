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
import { signup } from "@/app/actions/auth";
import PasswordInput from "./PasswordInput";

export default function SignupForm() {
    const [state, action, pending] = useActionState(signup, undefined);

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
                    label="Name"
                    name="name"
                    type="text"
                    placeholder="Optional"
                    defaultValue={state?.values?.name}
                    autoComplete="name"
                    error={!!state?.errors?.name?.length}
                    helperText={state?.errors?.name?.[0]}
                />
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
                    placeholder="At least 6 characters"
                    autoComplete="new-password"
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
                    {pending ? "Creating account…" : "Create account →"}
                </Button>

                <Typography variant="body2" align="center" color="text.secondary">
                    Already have an account?{" "}
                    <Link component={NextLink} href="/login" sx={{ color: "primary.main", fontWeight: 500, textDecoration: "none", "&:hover": { textDecoration: "underline" } }}>
                        Log in
                    </Link>
                </Typography>
            </Stack>
        </Box>
    );
}
