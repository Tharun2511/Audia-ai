import { getDatabase } from "@/db/data-source";
import { Chat } from "@/entity/Chat";
import { groq } from "@/lib/ai";
import { getCurrentUser } from "@/lib/dal";

export async function POST(req: Request) {
    const user = await getCurrentUser();
    if (!user) return Response.json({ error: "Unauthorized" }, { status: 401 });

    const { prompt } = await req.json();

    const db = await getDatabase();
    const chatRepo = db.getRepository(Chat);
    const chatRecord = chatRepo.create({ prompt, response: "" });
    await chatRepo.save(chatRecord);

    const encoder = new TextEncoder();
    const stream = new ReadableStream<Uint8Array>({
        async start(controller) {
            try {
                const aiStream = await groq.chat.completions.create({
                    messages: [{ role: "user", content: prompt }],
                    model: "llama-3.1-8b-instant",
                    stream: true,
                });

                let fullResponse = "";
                for await (const chunk of aiStream) {
                    const text = chunk.choices[0]?.delta?.content ?? "";
                    if (text) {
                        fullResponse += text;
                        controller.enqueue(encoder.encode(text));
                    }
                }

                chatRecord.response = fullResponse;
                await chatRepo.save(chatRecord);
            } catch (err) {
                controller.error(err);
                return;
            }
            controller.close();
        },
    });

    return new Response(stream, {
        headers: {
            "Content-Type": "text/plain; charset=utf-8",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    });
}
