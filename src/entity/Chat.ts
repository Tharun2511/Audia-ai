import { Entity, PrimaryGeneratedColumn, Column, CreateDateColumn } from "typeorm";

@Entity({ name: "chat" })
export class Chat {
    @PrimaryGeneratedColumn("uuid")
    id!: string;

    @Column("text")
    prompt!: string;

    @Column("text")
    response!: string;

    @CreateDateColumn()
    createdAt!: Date;
}
