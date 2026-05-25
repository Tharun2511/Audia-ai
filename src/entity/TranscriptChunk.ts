import { Column, CreateDateColumn, Entity, Index, PrimaryGeneratedColumn } from "typeorm";

/**
 * One embeddable unit of a transcript. The `embedding` column is intentionally
 * NOT declared here — TypeORM's synchronize doesn't understand pgvector's
 * `vector(N)` type. We ADD COLUMN it via raw SQL after sync (see ensurePgvector
 * in db/data-source.ts) and read/write it via raw queries.
 */
@Entity({ name: "transcript_chunk" })
@Index(["userEmail"])
@Index(["transcriptionId"])
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

    @CreateDateColumn()
    createdAt!: Date;
}
