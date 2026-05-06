import { Column, CreateDateColumn, Entity, PrimaryGeneratedColumn } from "typeorm";

export type TranscriptSegment = {
  speaker: string;
  text: string;
  start: number;
  end: number;
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

  @CreateDateColumn()
  createdAt!: Date;
}
