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
 * Ensures pgvector is enabled and the embedding column exists + is protected
 * on transcript_chunk. Runs once per process.
 *
 * Three things happen here, in order:
 *   1. Enable the vector extension + add the embedding column. Idempotent.
 *   2. SELF-HEAL: NULL-embedding rows are unreachable (every retrieval filters
 *      `AND embedding IS NOT NULL`) and only ever come from a write path that
 *      bypassed `saveChunkWithEmbedding`. Delete them every startup, loud-log
 *      the count so we know they were here. Recover via /api/backfill-chunks.
 *   3. DB-LEVEL ENFORCEMENT: a CHECK constraint that future INSERTs must
 *      include a non-NULL embedding. Defense-in-depth on top of the atomic
 *      INSERT — if some future code path ever forgets to populate it, the DB
 *      rejects it at write time instead of letting it silently land as NULL.
 *      CHECK constraints are invisible to TypeORM synchronize, so they survive
 *      hot reloads and entity changes.
 */
async function ensurePgvector(ds: DataSource) {
    if (pgvectorReady) return;

    await ds.query("CREATE EXTENSION IF NOT EXISTS vector;");
    await ds.query(
        `ALTER TABLE transcript_chunk ADD COLUMN IF NOT EXISTS embedding vector(768);`,
    );

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

    // DB-level enforcement: any future INSERT without embedding fails loudly.
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
    console.log("pgvector ready: extension enabled, column ensured, NOT NULL constraint active.");
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
