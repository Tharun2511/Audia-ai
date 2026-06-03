import { Check, Column, CreateDateColumn, Entity, Index, PrimaryGeneratedColumn } from "typeorm";

/**
 * One embeddable unit of a transcript.
 *
 * The `embedding` column uses TypeORM 0.3.27+ native pgvector support —
 * declaring it explicitly on the entity is what stops synchronize from
 * dropping it on every cold start (the bug we hit in Phase 4.2 → 7.2 that
 * empirically destroyed embeddings on every restart).
 *
 * `select: false` keeps repo.find() from auto-pulling the 3KB vector on
 * every read; the raw SQL paths in src/lib/chunks.ts explicitly select
 * `embedding::text` when they actually need it.
 */
@Entity({ name: "transcript_chunk" })
@Index(["userEmail"])
@Index(["transcriptionId"])
// Named CHECK constraint so synchronize matches it to the existing DB
// constraint and stops the drop-then-recreate churn we saw on every cold
// start. The name MUST match what ensurePgvector creates (kept consistent
// across both for the migration cutover in Phase 12 to be painless).
@Check("transcript_chunk_embedding_not_null", `"embedding" IS NOT NULL`)
export class TranscriptChunk {
    @PrimaryGeneratedColumn("uuid")
    id!: string;

    @Column({ type: "uuid" })
    transcriptionId!: string;

    // Denormalized for ownership filtering without a join on every KNN query.
    @Column()
    userEmail!: string;

    @Column("int")
    chunkIndex!: number;

    @Column("text")
    text!: string;

    @Column("simple-json")
    segmentIndices!: number[];

    @Column("simple-json")
    speakers!: string[];

    @Column("float")
    startTime!: number;

    @Column("float")
    endTime!: number;

    @Column("int")
    charCount!: number;

    @Column("vector", { length: 768, nullable: true, select: false })
    embedding!: number[] | null;

    @CreateDateColumn()
    createdAt!: Date;
}
