/**
 * Audia's MCP (Model Context Protocol) HTTP transport (Phase 7.3).
 *
 *   POST /api/mcp
 *
 * Spec: Model Context Protocol 2025-06-18, Streamable HTTP transport.
 * Body: a JSON-RPC 2.0 request (or notification). Response: a JSON-RPC 2.0
 * response (or 204 for notifications). Streaming SSE responses (for
 * server-initiated notifications) are NOT yet implemented — Audia's tool
 * surface today is synchronous request/response only.
 *
 * Auth: reuses Audia's existing session-cookie auth via getCurrentUser().
 * This means MCP-via-browser-fetch works AND curl-with-cookie works for
 * testing. Claude Desktop / Cursor integration would need a bearer-token
 * path layered on later (Phase 12 OAuth work).
 *
 * Quick test (replace COOKIE with your dev session cookie):
 *
 *   curl -X POST http://localhost:3000/api/mcp \
 *     -H "Content-Type: application/json" \
 *     -H "Cookie: COOKIE" \
 *     -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{}}}'
 *
 *   curl ... -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'
 *
 *   curl ... -d '{"jsonrpc":"2.0","id":3,"method":"tools/call",
 *     "params":{"name":"listMyMeetings","arguments":{"limit":5}}}'
 */
import { getCurrentUser } from "@/lib/dal";
import { handleMcpMessage, parseErrorResponse } from "@/lib/mcp";

export async function POST(req: Request) {
    // Auth FIRST — every MCP method (even initialize) requires a logged-in
    // user because Audia's tools are user-scoped. An unauthenticated client
    // can't even discover what tools exist; that's deliberate.
    const user = await getCurrentUser();
    if (!user) {
        return new Response("Unauthorized", { status: 401 });
    }

    // Parse body. Parse failures are reported as JSON-RPC -32700 (per spec).
    let body: unknown;
    try {
        body = await req.json();
    } catch {
        return Response.json(parseErrorResponse(), { status: 200 });
    }

    // Spec note: MCP also supports BATCH requests (an array of requests).
    // For Audia's surface today, single requests only — easier to read and
    // we have no client today that batches. If batch is needed: detect
    // Array.isArray(body), map over handleMcpMessage, return filtered array
    // (drop nulls from notifications).
    const response = await handleMcpMessage(body, { userEmail: user.email });

    // Notifications: spec says server MUST NOT respond. Return 204 No Content.
    if (response === null) {
        return new Response(null, { status: 204 });
    }

    return Response.json(response, { status: 200 });
}
