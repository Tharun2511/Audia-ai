import { redirect } from "next/navigation";
import { getCurrentUser } from "@/lib/dal";
import HomeClient from "./HomeClient";

export default async function Home() {
    const user = await getCurrentUser();
    if (!user) redirect("/login");

    return <HomeClient userEmail={user.email} userName={user.name} />;
}
