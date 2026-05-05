import "reflect-metadata";
import { DataSource } from "typeorm";
import { Chat } from "../entity/Chat";
import { Transcription } from "../entity/Transcription";
import { User } from "../entity/User";

export const AppDataSource = new DataSource({
    type: "postgres",
    url: process.env.DATABASE_URL || process.env.DATABSE_URL,
    ssl: {
        rejectUnauthorized: false // Required for Neon
    },
    synchronize: true,
    logging: false,
    entities: [Chat, Transcription, User],
});

// This helper prevents Next.js from creating too many connections during development
export const getDatabase = async () => {
    if (!AppDataSource.isInitialized) {
        await AppDataSource.initialize();
        console.log("Database connected successfully!");
    }
    return AppDataSource;
};
