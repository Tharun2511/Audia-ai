import "reflect-metadata";
import { DataSource } from "typeorm";
import { ChatMessage } from "../entity/ChatMessage";
import { Transcription } from "../entity/Transcription";
import { TranscriptChunk } from "../entity/TranscriptChunk";
import { User } from "../entity/User";

export const AppDataSource = new DataSource({
    type: "postgres",
    url: process.env.DATABASE_URL,
    ssl: {
        rejectUnauthorized: false // Required for Neon
    },
    synchronize: true,
    logging: false,
    entities: [Transcription, TranscriptChunk, User, ChatMessage],
});

let pgvectorReady = false;

/**
 * Post-init schema hardening for the embedding column.
 *
 * As of Phase 7.2 fix: the `embedding` column is now declared on the
 * TranscriptChunk entity (`@Column("vector", {length: 768, nullable: true})`),
 * so TypeORM 0.3.27+'s native pgvector support handles column creation +
 * preservation across synchronize. We no longer ADD COLUMN here manually —
 * doing it post-sync was the bug (synchronize saw the column as orphan,
 * dropped it, then we re-added it empty, wiping all embeddings every restart).
 *
 * What still happens here:
 *   1. CREATE EXTENSION IF NOT EXISTS vector — needed at least once on a
 *      fresh DB so TypeORM can create vector columns. Idempotent.
 *   2. SELF-HEAL: delete any NULL-embedding rows. With the fix in place these
 *      shouldn't appear, but the defense-in-depth check stays — if anything
 *      ever lands NULL we want it cleaned + loud-logged on startup.
 *   3. CHECK CONSTRAINT: belt on top of the entity-level `nullable: false`
 *      (TODO once historical NULLs are backfilled). For now `nullable: true`
 *      because legacy rows may exist; constraint stays as the eventual lock.
 */
async function ensurePgvector(ds: DataSource) {
    if (pgvectorReady) return;

    // The extension MUST exist before TypeORM tries to synchronize a vector
    // column on a fresh DB. Idempotent, so cheap to call every startup.
    await ds.query("CREATE EXTENSION IF NOT EXISTS vector;");

    // Self-heal: count then delete any NULL-embedding rows.
    const nullCheck = (await ds.query(
        `SELECT COUNT(*)::int AS n FROM transcript_chunk WHERE embedding IS NULL`,
    )) as Array<{ n: number }>;
    const nullCount = nullCheck[0]?.n ?? 0;
    if (nullCount > 0) {
        console.warn(
            `[pgvector] cleaning ${nullCount} NULL-embedding chunk row(s). ` +
            `These rows were unreachable (retrieval filters embedding IS NOT NULL). ` +
            `Re-run /api/backfill-chunks POST to regenerate them.`,
        );
        await ds.query(`DELETE FROM transcript_chunk WHERE embedding IS NULL`);
    }

    // DB-level CHECK constraint: any future INSERT without embedding fails
    // loudly even if the entity-level nullable: true allows it through.
    // Tightened to NOT NULL on the entity in a follow-up once we've confirmed
    // no remaining legacy NULLs in production.
    await ds.query(`
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'transcript_chunk_embedding_not_null'
            ) THEN
                ALTER TABLE transcript_chunk
                    ADD CONSTRAINT transcript_chunk_embedding_not_null
                    CHECK (embedding IS NOT NULL);
            END IF;
        END $$;
    `);

    pgvectorReady = true;
    console.log("pgvector ready: extension enabled, entity-managed column, NOT NULL constraint active.");
}

// This helper prevents Next.js from creating too many connections during development
export const getDatabase = async () => {
    if (!AppDataSource.isInitialized) {
        await AppDataSource.initialize();
        console.log("Database connected successfully!");
    }
    await ensurePgvector(AppDataSource);
    return AppDataSource;
};
