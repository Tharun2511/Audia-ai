import { Column, CreateDateColumn, Entity, Index, PrimaryGeneratedColumn } from "typeorm";

@Entity()
export class User {
    @PrimaryGeneratedColumn("uuid")
    id!: string;

    @Index({ unique: true })
    @Column({ type: "varchar", length: 320 })
    email!: string;

    @Column({ type: "varchar", length: 255 })
    passwordHash!: string;

    @Column({ type: "varchar", length: 100, nullable: true })
    name!: string | null;

    @CreateDateColumn()
    createdAt!: Date;
}
