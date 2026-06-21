import { Column, CreateDateColumn, Entity, PrimaryGeneratedColumn } from "typeorm";

export type TranscriptSegment = {
  speaker: string;
  text: string;
  start: number;
  end: number;
  /**
   * Average ASR word-confidence for this segment, in [0, 1] (Phase 10.1).
   * Optional: legacy rows + non-ASR code paths (golden-set test segments)
   * won't have it. Low values flag likely transcription errors in the UI.
   */
  confidence?: number;
};

@Entity({ name: "transcription" })
export class Transcription {
  @PrimaryGeneratedColumn("uuid")
  id!: string;

  @Column({ nullable: true })
  userEmail!: string;

  @Column({ nullable: true, type: "varchar" })
  title!: string | null;

  @Column("float", { nullable: true })
  duration!: number;

  @Column({ type: "simple-json" })
  segments!: TranscriptSegment[];

  @Column({ nullable: true, type: "text" })
  summary!: string | null;

  /**
   * Vercel Blob pathname (e.g. "audio/9f0e1d35-b8c7.webm") — NOT the playable
   * URL. The store is private, so a fresh signed URL has to be issued each
   * time a client wants to play this recording. Pathname is the canonical
   * identifier we keep; URLs are derivative + ephemeral.
   *
   * Nullable for legacy rows (no audio uploaded) and graceful degradation when
   * BLOB_READ_WRITE_TOKEN isn't configured.
   */
  @Column({ nullable: true, type: "text" })
  audioPathname!: string | null;

  @CreateDateColumn()
  createdAt!: Date;
}
