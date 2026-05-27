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
 * Ensures pgvector is enabled and the embedding column exists on transcript_chunk.
 * Runs once per process. TypeORM synchronize doesn't know about the vector type,
 * so we ALTER TABLE ourselves after sync has created the rest of the table.
 */
async function ensurePgvector(ds: DataSource) {
    if (pgvectorReady) return;
    await ds.query("CREATE EXTENSION IF NOT EXISTS vector;");
    await ds.query(
        `ALTER TABLE transcript_chunk ADD COLUMN IF NOT EXISTS embedding vector(768);`,
    );
    pgvectorReady = true;
    console.log("pgvector enabled and embedding column ensured.");
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
