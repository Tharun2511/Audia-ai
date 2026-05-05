import { NextResponse, type NextRequest } from "next/server";
import { decrypt } from "@/lib/session";

const PUBLIC_ROUTES = new Set(["/login", "/signup"]);

export default async function proxy(req: NextRequest) {
    const path = req.nextUrl.pathname;
    const isPublic = PUBLIC_ROUTES.has(path);

    const token = req.cookies.get("session")?.value;
    const session = await decrypt(token);
    const isAuthed = !!session?.userId;

    if (!isPublic && !isAuthed) {
        return NextResponse.redirect(new URL("/login", req.nextUrl));
    }
    if (isPublic && isAuthed) {
        return NextResponse.redirect(new URL("/", req.nextUrl));
    }
    return NextResponse.next();
}

export const config = {
    matcher: ["/((?!api|_next/static|_next/image|.*\\.svg$|.*\\.ico$).*)"],
};
