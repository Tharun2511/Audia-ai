"use server";

import bcrypt from "bcryptjs";
import { redirect } from "next/navigation";
import { getDatabase } from "@/db/data-source";
import { User } from "@/entity/User";
import { createSession, deleteSession } from "@/lib/session";
import {
    LoginFormSchema,
    SignupFormSchema,
    type LoginFormState,
    type SignupFormState,
} from "@/lib/validation";

const BCRYPT_COST = 10;

export async function signup(_prev: SignupFormState, formData: FormData): Promise<SignupFormState> {
    const raw = {
        name: (formData.get("name") as string | null) ?? undefined,
        email: formData.get("email"),
        password: formData.get("password"),
    };

    const parsed = SignupFormSchema.safeParse(raw);
    if (!parsed.success) {
        return {
            errors: z_flatten(parsed.error),
            values: { name: typeof raw.name === "string" ? raw.name : undefined, email: typeof raw.email === "string" ? raw.email : undefined },
        };
    }
    const { name, email, password } = parsed.data;

    const db = await getDatabase();
    const userRepo = db.getRepository(User);

    const existing = await userRepo.findOne({ where: { email } });
    if (existing) {
        return {
            message: "An account with that email already exists.",
            values: { name: name ?? undefined, email },
        };
    }

    const passwordHash = await bcrypt.hash(password, BCRYPT_COST);
    const user = userRepo.create({ email, passwordHash, name: name ?? null });
    await userRepo.save(user);

    await createSession(user.id);
    redirect("/");
}

export async function login(_prev: LoginFormState, formData: FormData): Promise<LoginFormState> {
    const parsed = LoginFormSchema.safeParse({
        email: formData.get("email"),
        password: formData.get("password"),
    });
    if (!parsed.success) {
        return {
            errors: z_flatten(parsed.error),
            values: { email: typeof formData.get("email") === "string" ? (formData.get("email") as string) : undefined },
        };
    }
    const { email, password } = parsed.data;

    const db = await getDatabase();
    const userRepo = db.getRepository(User);
    const user = await userRepo.findOne({ where: { email } });

    // Run bcrypt even if the user doesn't exist, to keep response time roughly
    // constant — prevents a timing attack that could enumerate registered emails.
    const placeholderHash = "$2a$10$invalid.invalid.invalid.invalid.invalid.invalid.invalid.invalid";
    const ok = await bcrypt.compare(password, user?.passwordHash ?? placeholderHash);

    if (!user || !ok) {
        return { message: "Invalid email or password.", values: { email } };
    }

    await createSession(user.id);
    redirect("/");
}

export async function logout(): Promise<void> {
    await deleteSession();
    redirect("/login");
}

// Zod 4 returns ZodError; we want { fieldName: string[] }.
function z_flatten(error: import("zod").ZodError): Record<string, string[]> {
    const out: Record<string, string[]> = {};
    for (const issue of error.issues) {
        const key = issue.path[0]?.toString() ?? "_";
        (out[key] ??= []).push(issue.message);
    }
    return out;
}
