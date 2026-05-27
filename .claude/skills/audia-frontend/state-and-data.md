# State & data — RSC discipline, Server Actions, boundaries

The single biggest source of bugs in modern Next.js apps is **using client-component patterns where server-component patterns apply**. This doc is the rule book.

## The default is Server Component

Every file under `src/app/` is a Server Component unless it starts with `"use client"`. You should be writing Server Components by default.

### When you MUST add `"use client"`

Any of these features force a Client Component:
- `useState`, `useReducer`
- `useEffect`, `useLayoutEffect`
- `useContext` (the consumer; provider also needs to be client)
- `useRef`
- Event handlers (`onClick`, `onChange`, `onSubmit`, …)
- Browser APIs (`window`, `document`, `localStorage`, `navigator`, `IntersectionObserver`, …)
- Class components
- Any of the above-using third-party library

### Lift the client boundary down, not up

```
❌ Bad:  "use client" on the whole page
                     ↓
        Page (client)
        ├── Header (forced client)
        ├── List (forced client, doesn't need to be)
        └── InteractiveButton (needs client)

✅ Good: "use client" only on the interactive leaf
        Page (server)
        ├── Header (server)
        ├── List (server, fetches data directly)
        └── InteractiveButton ("use client")
```

The list can stay a server component and fetch data directly. Only the button — which needs `onClick` — is client.

### Server Component fetches data; passes to client as props

```tsx
// page.tsx — Server Component
import { db } from "@/db/data-source";
import SessionListClient from "./SessionListClient";

export default async function Page() {
  const sessions = await db.session.findMany({ where: { ... } });
  return <SessionListClient sessions={sessions} />;
}
```

Don't `useEffect(() => fetch("/api/sessions"))` inside a client component for data the server already has.

## Mutations: Server Actions

For any "user submits a form / clicks a button that changes server state" — use a Server Action, not a client `fetch("/api/...")`.

### File location

Server Actions live in `src/app/actions/<name>.ts`. Each file starts with `"use server"`. Audia already does this with [src/app/actions/auth.ts](../../../src/app/actions/auth.ts).

### Form pattern

```tsx
// app/actions/title.ts
"use server";
import { revalidatePath } from "next/cache";
import { z } from "zod";

const Schema = z.object({ id: z.string(), title: z.string().min(1).max(200) });

export async function renameSession(prev: unknown, formData: FormData) {
  const parsed = Schema.safeParse(Object.fromEntries(formData));
  if (!parsed.success) return { error: parsed.error.issues[0].message };

  // …auth check, ownership check, DB update…

  revalidatePath(`/`);
  return { ok: true };
}
```

```tsx
// component.tsx
"use client";
import { useActionState } from "react";
import { useFormStatus } from "react-dom";
import { renameSession } from "@/app/actions/title";

function SubmitButton() {
  const { pending } = useFormStatus();
  return (
    <Button type="submit" disabled={pending} aria-busy={pending}>
      {pending ? "Saving…" : "Save"}
    </Button>
  );
}

export function RenameForm({ id, initial }: { id: string; initial: string }) {
  const [state, action] = useActionState(renameSession, null);

  return (
    <form action={action} noValidate>
      <input type="hidden" name="id" value={id} />
      <TextField name="title" defaultValue={initial} error={!!state?.error} helperText={state?.error} />
      <SubmitButton />
    </form>
  );
}
```

Key points:
- `useActionState` gives you the latest action result.
- `useFormStatus` gives you `pending` — must be called inside a component that's a child of the `<form>`.
- The action `revalidatePath`s so the page re-renders with fresh data; no manual cache management.

## Optimistic UI with `useOptimistic`

For mutations where the response should feel instant — title edits, chat sends, marking a session shared — use `useOptimistic`. The user sees their change immediately; if the server fails, it rolls back.

```tsx
"use client";
import { useOptimistic, useTransition } from "react";

export function Title({ id, current }: { id: string; current: string }) {
  const [optimistic, setOptimistic] = useOptimistic(current, (_, next: string) => next);
  const [, startTransition] = useTransition();

  async function save(next: string) {
    startTransition(async () => {
      setOptimistic(next);                 // UI updates immediately
      const result = await renameSession(id, next);
      if (result?.error) {
        toast.error("Couldn't rename. Reverting.");
        // rollback happens automatically when the action ends without revalidating
      }
    });
  }

  return <TitleInput value={optimistic} onSave={save} />;
}
```

Use cases in Audia:
- Title rename (the most obvious win).
- Chat message send — show your message in the thread immediately, then stream the assistant's reply.
- Future: pin/unpin a session, mark a session public.

## Suspense & streaming

Server Components can be `async`. Wrap them in `<Suspense>` to stream their content while the rest of the page renders.

```tsx
// page.tsx
import { Suspense } from "react";

export default function Page({ params }: { params: Promise<{ id: string }> }) {
  return (
    <Stack spacing={2}>
      <Title id={params.then(p => p.id)} />
      <Suspense fallback={<Skeleton variant="rectangular" height={160} />}>
        <SummaryAsync id={params.then(p => p.id)} />
      </Suspense>
      <Suspense fallback={<TranscriptSkeleton />}>
        <TranscriptAsync id={params.then(p => p.id)} />
      </Suspense>
    </Stack>
  );
}
```

Each Suspense boundary streams independently. The user sees the title immediately, summary when ready, transcript when ready.

## Route-level boundaries

Co-locate with `page.tsx`:

| File | Purpose |
|---|---|
| `loading.tsx` | Shown while the page server component is fetching. Should match the page's skeleton layout. |
| `error.tsx` | Shown when an uncaught error bubbles up. Must be a client component (has `reset()`). |
| `not-found.tsx` | Shown when `notFound()` is called or a dynamic param doesn't match. |

```tsx
// app/[id]/error.tsx
"use client";
export default function Error({ error, reset }: { error: Error; reset: () => void }) {
  return (
    <Stack spacing={2} sx={{ p: 4, alignItems: "center" }}>
      <Typography variant="h6">Something went wrong</Typography>
      <Typography variant="body2" color="text.secondary">{error.message}</Typography>
      <Button onClick={reset} variant="contained">Try again</Button>
    </Stack>
  );
}
```

```tsx
// app/[id]/not-found.tsx
export default function NotFound() {
  return (
    <EmptyState
      title="Session not found"
      description="It may have been deleted or the link is incorrect."
      action={<Button component={Link} href="/">Back to your sessions</Button>}
    />
  );
}
```

## Lists need three states

Every list-bearing surface must handle:
1. **Loading** — skeleton matching the row shape.
2. **Empty** — `EmptyState` primitive ([patterns/empty-state.tsx](patterns/empty-state.tsx)) with a primary action ("+ New session").
3. **Error** — inline retry, ideally the `error.tsx` boundary picks it up.

Use the `ListStates` wrapper in [patterns/list-states.tsx](patterns/list-states.tsx) for the common case.

## When to use Context

Context is fine for **truly app-wide concerns**: theme (MUI already), color scheme (MUI's `useColorScheme`), auth user (a thin provider in a `"use client"` boundary).

Context is **wrong** for: server-fetched data (pass via props from server component), form state (the form owns it), modal open/close state (lift to the lowest common ancestor).

## When NOT to use Server Actions

- For pure reads — those are Server Components.
- For third-party API proxies that need to stream — use a Route Handler.
- For webhooks — Route Handlers only.

## Pitfalls

- **`"use client"` at the top of a page**. Now the whole page is client; you've lost RSC benefits.
- **`useEffect(() => fetch("/api/X"))` for data you control.** Move the fetch to the Server Component.
- **Reaching into client state from a Server Component.** Server Components don't have hooks. Pass data the other direction.
- **Forgetting `revalidatePath`** in a Server Action. The mutation lands in the DB but the UI doesn't refresh.
- **Async Client Components**. Client Components can't be async. Use `useEffect` + state, or refactor the parent to be a Server Component.
- **Passing functions from Server → Client**. Components can pass props that are JSON-serializable plus a few things. **Functions, classes, Dates** generally don't cross. Functions specifically: Server Actions (`"use server"`) can be passed; arbitrary functions cannot.
