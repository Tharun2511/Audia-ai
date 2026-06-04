import "server-only";
import { AUDIA_TOOLS, dispatchTool } from "@/lib/tools";

/**
 * Audia's MCP (Model Context Protocol) server logic — pure JSON-RPC 2.0
 * handlers, no HTTP/auth concerns (those live in the API route).
 *
 * MCP is Anthropic's open standard (Nov 2024) for connecting LLM applications
 * to external tools/data/prompts. Audia exposes its tool dispatcher here so
 * any compliant MCP client (Claude Desktop, Cursor, third-party agents) can
 * query the user's meetings the same way Audia's own chat does.
 *
 * Today's surface: tools only (the most common MCP primitive). resources and
 * prompts can come later (Phase 12) — same JSON-RPC envelope, additional
 * `resources/list` + `prompts/list` methods.
 *
 * Auth: the route layer authenticates the request via getCurrentUser() and
 * passes the userEmail in `ctx`. Every tool call is scoped to that user;
 * cross-tenant isolation is enforced by the existing tool implementations,
 * not by MCP-level logic. Same ownership boundary as Audia's chat.
 */

// ── Protocol constants ────────────────────────────────────────────────────

const PROTOCOL_VERSION = "2025-06-18";
const SERVER_INFO = {
    name: "audia-mcp",
    version: "0.1.0",
} as const;

// ── JSON-RPC 2.0 type surface ─────────────────────────────────────────────

export type JsonRpcRequest = {
    jsonrpc: "2.0";
    id?: string | number | null;          // absent = notification (no response)
    method: string;
    params?: unknown;
};

export type JsonRpcSuccess = {
    jsonrpc: "2.0";
    id: string | number | null;
    result: unknown;
};

export type JsonRpcError = {
    jsonrpc: "2.0";
    id: string | number | null;
    error: { code: number; message: string; data?: unknown };
};

export type JsonRpcResponse = JsonRpcSuccess | JsonRpcError;

// JSON-RPC 2.0 reserved error codes (https://www.jsonrpc.org/specification#error_object)
const PARSE_ERROR = -32700;
const INVALID_REQUEST = -32600;
const METHOD_NOT_FOUND = -32601;
const INVALID_PARAMS = -32602;
const INTERNAL_ERROR = -32603;

// ── Context passed by the route layer ─────────────────────────────────────

export type McpCtx = { userEmail: string };

// ── Public entry point ────────────────────────────────────────────────────

/**
 * Handle one JSON-RPC message. Returns:
 *   - a JsonRpcResponse for requests (with an `id`)
 *   - null for notifications (no `id`, per spec — server MUST NOT respond)
 *
 * Errors thrown by handlers are caught and translated to INTERNAL_ERROR so
 * the caller always gets a well-formed envelope, never an unhandled exception.
 */
export async function handleMcpMessage(
    message: unknown,
    ctx: McpCtx,
): Promise<JsonRpcResponse | null> {
    // Shape-check first — the route already JSON.parsed the body, but the
    // shape is still untrusted. Anything that doesn't match the request
    // envelope gets a -32600 Invalid Request.
    if (!isJsonRpcRequest(message)) {
        return errorResponse(null, INVALID_REQUEST, "Invalid Request — must conform to JSON-RPC 2.0");
    }

    const { id, method, params } = message;
    const isNotification = id === undefined;

    try {
        switch (method) {
            case "initialize":
                return successResponse(id ?? null, handleInitialize());

            case "notifications/initialized":
                // Client signals it has finished setup. Spec: server MUST NOT
                // respond to notifications. Return null so the route returns 204.
                return null;

            case "tools/list":
                return successResponse(id ?? null, handleToolsList());

            case "tools/call": {
                if (!isToolsCallParams(params)) {
                    return errorResponse(id ?? null, INVALID_PARAMS, "tools/call requires { name: string, arguments?: object }");
                }
                const result = await handleToolsCall(params, ctx);
                return successResponse(id ?? null, result);
            }

            // Unknown methods. For requests, return method-not-found; for
            // notifications, silently drop (per spec).
            default:
                if (isNotification) return null;
                return errorResponse(id ?? null, METHOD_NOT_FOUND, `Method not found: ${method}`);
        }
    } catch (err) {
        // Internal handler bug — surface as JSON-RPC -32603 so the client
        // gets a well-formed error envelope instead of a 500.
        if (isNotification) return null;
        const msg = err instanceof Error ? err.message : String(err);
        return errorResponse(id ?? null, INTERNAL_ERROR, `Internal error: ${msg}`);
    }
}

// ── Method handlers ───────────────────────────────────────────────────────

/**
 * initialize — the handshake. Client sends its desired protocolVersion +
 * capabilities; server responds with its own. Audia declares only the
 * `tools` capability today; `resources` and `prompts` are future work.
 */
function handleInitialize() {
    return {
        protocolVersion: PROTOCOL_VERSION,
        capabilities: {
            tools: {},
            // resources: {}, prompts: {} — add when those primitives land.
        },
        serverInfo: SERVER_INFO,
    };
}

/**
 * tools/list — convert AUDIA_TOOLS (Groq/OpenAI function-calling shape) into
 * MCP tool definitions. The conversion is straightforward because both formats
 * standardize on JSON Schema for `inputSchema`/`parameters` — the same schema
 * works for Groq function calling AND MCP tools/list.
 *
 * Reusing AUDIA_TOOLS means Audia's chat and the MCP server expose the SAME
 * tool definitions by construction; they can never drift.
 */
function handleToolsList() {
    return {
        tools: AUDIA_TOOLS.map((t) => {
            const fn = t.function;
            // ChatCompletionTool's `function` is technically optional in the
            // SDK type, but every AUDIA_TOOLS entry sets it. Narrow defensively.
            if (!fn?.name) {
                throw new Error("Tool registry has an entry without a function.name");
            }
            return {
                name: fn.name,
                description: fn.description ?? "",
                inputSchema: fn.parameters ?? { type: "object", properties: {}, required: [] },
            };
        }),
    };
}

/**
 * tools/call — dispatch via the SAME tools.ts dispatcher Audia's chat uses.
 * The MCP wire format passes `arguments` as a JSON object; tools.ts's
 * dispatchTool expects a JSON string (matches Groq's tool_call.arguments
 * shape). Stringify here to bridge.
 *
 * MCP wraps the tool's return string into a `content` array with typed
 * blocks (text, image, audio, embedded resources). For Audia's text-only
 * tool results, one text block is sufficient.
 *
 * `isError: true` is set when the dispatcher returned a structured `{error}`
 * payload — see tools.ts. This lets MCP clients distinguish "tool ran but
 * returned an error" from "the protocol itself failed" (which would be a
 * JSON-RPC error response instead).
 */
async function handleToolsCall(
    params: { name: string; arguments?: Record<string, unknown> },
    ctx: McpCtx,
) {
    const argsJson = JSON.stringify(params.arguments ?? {});
    const resultText = await dispatchTool(params.name, argsJson, { userEmail: ctx.userEmail });

    // Did the dispatcher signal an error? Surface it via isError so MCP
    // clients can render it differently from a successful result.
    let isError = false;
    try {
        const parsed = JSON.parse(resultText) as { error?: unknown };
        if (parsed && typeof parsed.error === "string") isError = true;
    } catch {
        // Tool returned non-JSON (unusual but allowed) — treat as success.
    }

    return {
        content: [{ type: "text" as const, text: resultText }],
        isError,
    };
}

// ── Envelope helpers ──────────────────────────────────────────────────────

function successResponse(id: string | number | null, result: unknown): JsonRpcSuccess {
    return { jsonrpc: "2.0", id, result };
}

function errorResponse(
    id: string | number | null,
    code: number,
    message: string,
    data?: unknown,
): JsonRpcError {
    return { jsonrpc: "2.0", id, error: { code, message, ...(data !== undefined ? { data } : {}) } };
}

// ── Shape guards ──────────────────────────────────────────────────────────

function isJsonRpcRequest(x: unknown): x is JsonRpcRequest {
    if (typeof x !== "object" || x === null) return false;
    const m = x as Record<string, unknown>;
    if (m.jsonrpc !== "2.0") return false;
    if (typeof m.method !== "string") return false;
    return true;
}

function isToolsCallParams(x: unknown): x is { name: string; arguments?: Record<string, unknown> } {
    if (typeof x !== "object" || x === null) return false;
    const p = x as Record<string, unknown>;
    if (typeof p.name !== "string") return false;
    if (p.arguments !== undefined && (typeof p.arguments !== "object" || p.arguments === null)) return false;
    return true;
}

// Useful for the route layer to mark a leak-prevention boundary:
// the parse-error case happens BEFORE we have a request id to echo back.
export function parseErrorResponse(): JsonRpcError {
    return errorResponse(null, PARSE_ERROR, "Parse error — body is not valid JSON");
}
