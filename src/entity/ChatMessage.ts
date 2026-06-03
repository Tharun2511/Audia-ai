import { Column, CreateDateColumn, Entity, Index, PrimaryGeneratedColumn } from "typeorm";

/**
 * One turn in a conversation. `sessionId` is the grouping key — there's no
 * separate `ChatSession` entity yet; a session is just "all messages sharing
 * a sessionId." Promote to its own table when we need session listing,
 * titles, or cross-transcript chats (5.2+).
 *
 * The composite (sessionId, createdAt) index is the only one that matters
 * here — the hot query is "load this session's last N turns in order."
 *
 * `citations` is JSON metadata for the [N] chips the UI rendered next to an
 * assistant turn. We persist it for replay (load past conversation → re-show
 * chips) but STRIP it from the prompt we send to the model, because past [N]
 * markers refer to chunks numbered against past retrievals — the current
 * turn's chunks are numbered fresh and conflicting markers would mislead
 * the model into citing the wrong chunks.
 */
@Entity({ name: "chat_message" })
@Index(["sessionId", "createdAt"])
@Index(["userEmail"])
export class ChatMessage {
    @PrimaryGeneratedColumn("uuid")
    id!: string;

    @Column({ type: "uuid" })
    sessionId!: string;

    // Denormalized for ownership filtering without a join on every history load.
    @Column({ type: "varchar", length: 320 })
    userEmail!: string;

    // Nullable so we can support cross-transcript chats later (5.2+) without a
    // schema change. Today every session is bound to one transcript.
    @Column({ type: "uuid", nullable: true })
    transcriptionId!: string | null;

    @Column({ type: "varchar", length: 16 })
    role!: "user" | "assistant" | "tool";

    @Column({ type: "text" })
    content!: string;

    @Column({ type: "simple-json", nullable: true })
    citations!: unknown[] | null;

    // Populated only on role="tool" — references the assistant tool_call.id
    // this row is the result of. Lets us reconstruct the assistant→tool pairing
    // when loading history (the API requires role:tool messages immediately
    // follow their assistant turn with a matching tool_call_id).
    @Column({ type: "varchar", length: 64, nullable: true })
    toolCallId!: string | null;

    // Populated only on role="assistant" turns that emitted tool_calls.
    // Each entry: { id, name, arguments } — matches the OpenAI/Groq shape so
    // we can round-trip the assistant turn back into the messages array
    // verbatim on subsequent calls.
    @Column({ type: "simple-json", nullable: true })
    toolCalls!: Array<{ id: string; name: string; arguments: string }> | null;

    @CreateDateColumn()
    createdAt!: Date;
}
