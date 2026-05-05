import "server-only";
import { SignJWT, jwtVerify, type JWTPayload } from "jose";
import { cookies } from "next/headers";

const SESSION_COOKIE = "session";
const SESSION_DURATION_MS = 7 * 24 * 60 * 60 * 1000; // 7 days

export type SessionPayload = JWTPayload & {
    userId: string;
    expiresAt: number; // unix ms
};

function getKey(): Uint8Array {
    const secret = process.env.SESSION_SECRET;
    if (!secret) throw new Error("SESSION_SECRET is not set");
    return new TextEncoder().encode(secret);
}

export async function encrypt(payload: SessionPayload): Promise<string> {
    return new SignJWT(payload)
        .setProtectedHeader({ alg: "HS256" })
        .setIssuedAt()
        .setExpirationTime("7d")
        .sign(getKey());
}

export async function decrypt(token: string | undefined): Promise<SessionPayload | null> {
    if (!token) {
        console.log("[auth] decrypt: no token in request");
        return null;
    }
    try {
        const { payload } = await jwtVerify(token, getKey(), { algorithms: ["HS256"] });
        return payload as SessionPayload;
    } catch (err) {
        console.log("[auth] decrypt: jwtVerify failed:", err instanceof Error ? err.message : err);
        return null;
    }
}

export async function createSession(userId: string): Promise<void> {
    const expiresAt = Date.now() + SESSION_DURATION_MS;
    const token = await encrypt({ userId, expiresAt });
    const store = await cookies();
    store.set(SESSION_COOKIE, token, {
        httpOnly: true,
        secure: process.env.NODE_ENV === "production",
        sameSite: "lax",
        path: "/",
        expires: new Date(expiresAt),
    });
}

export async function deleteSession(): Promise<void> {
    const store = await cookies();
    store.delete(SESSION_COOKIE);
}

export async function readSession(): Promise<SessionPayload | null> {
    const store = await cookies();
    const token = store.get(SESSION_COOKIE)?.value;
    return decrypt(token);
}
