import * as z from "zod";

export const SignupFormSchema = z.object({
    name: z
        .string()
        .trim()
        .max(100, { error: "Name must be 100 characters or fewer." })
        .optional(),
    email: z
        .email({ error: "Enter a valid email address." })
        .trim()
        .toLowerCase()
        .max(320, { error: "Email is too long." }),
    password: z
        .string()
        .min(6, { error: "Password must be at least 6 characters." })
        .max(200, { error: "Password is too long." }),
});

export const LoginFormSchema = z.object({
    email: z.email({ error: "Enter a valid email address." }).trim().toLowerCase(),
    password: z.string().min(1, { error: "Password is required." }),
});

export type SignupFormState =
    | {
        errors?: { name?: string[]; email?: string[]; password?: string[] };
        message?: string;
        values?: { name?: string; email?: string };
    }
    | undefined;

export type LoginFormState =
    | {
        errors?: { email?: string[]; password?: string[] };
        message?: string;
        values?: { email?: string };
    }
    | undefined;
