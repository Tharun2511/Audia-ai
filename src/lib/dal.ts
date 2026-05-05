import "server-only";
import { cache } from "react";
import { redirect } from "next/navigation";
import { readSession } from "@/lib/session";
import { getDatabase } from "@/db/data-source";
import { User } from "@/entity/User";

export const verifySession = cache(async (): Promise<{ userId: string }> => {
    const session = await readSession();
    if (!session?.userId) redirect("/login");
    return { userId: session.userId };
});

export const getCurrentUser = cache(async (): Promise<{ id: string; email: string; name: string | null } | null> => {
    const session = await readSession();
    if (!session?.userId) return null;

    const db = await getDatabase();
    const repo = db.getRepository(User);
    const user = await repo.findOne({
        where: { id: session.userId },
        select: { id: true, email: true, name: true },
    });
    return user ?? null;
});
