# Performance — bundle, images, fonts, dynamic, prefetch

## Targets (Web Vitals)

| Metric | Target | What it measures |
|---|---|---|
| LCP | < 2.5s | Largest visual element painted |
| INP | < 200ms | Responsiveness to interaction |
| CLS | < 0.1 | Layout shift |
| TTFB | < 800ms | Server response time |

Measure in real conditions: Chrome DevTools → Lighthouse → mobile, throttled 4G. Local fast tests lie.

## Bundle budget

Per-route client JS, gzipped:
- **Ideal**: < 100 KB
- **Hard cap**: < 200 KB

Inspect after `npm run build` — Next prints the per-route size table. Watch the row for `/` (home) and `/s/[id]` (when added).

When bundle grows:
1. Is something accidentally `"use client"` that shouldn't be? Push the boundary down.
2. Is a heavy lib being imported at the top of a client component when it's only used after a user action? Dynamic-import it.
3. Is the same icon being imported from `@mui/icons-material` in 50 places? That's fine — modern bundlers tree-shake.

## next/image rules

- Always `next/image`, never `<img>`.
- Provide `width` + `height` OR `fill` + `sizes`. Never omit dimensions — that's a CLS guarantee.
- LCP image gets `priority` (e.g. hero on the public session page).
- For above-the-fold images: `loading="eager"`. Default lazy loading is fine for below-fold.
- `sizes` must be honest: `"(max-width: 900px) 100vw, 800px"` for a hero capped at 800px.

```tsx
import Image from "next/image";

<Image
  src="/og-bg.jpg"
  alt="Conversation visualization"
  width={1200}
  height={630}
  priority
  sizes="(max-width: 900px) 100vw, 800px"
/>
```

## next/font

Already wired in [src/app/layout.tsx](../../../src/app/layout.tsx) with `Geist` and `Geist_Mono`. Don't add a third font; don't switch loaders. `next/font` self-hosts the font, prevents FOUT, and avoids a network call to Google.

If a future feature genuinely needs a third typeface (rare — try font-weight/style variants first), import it via `next/font/google` or `next/font/local`, assign a CSS variable, and use that variable in `theme.ts`'s `typography.fontFamily`.

## Dynamic imports

Use `next/dynamic` for components that are:
- Heavy (the audio waveform canvas, a chart library, a code editor, a PDF viewer)
- Below the fold and rarely interacted with
- Only used in a modal that's opened lazily

```tsx
import dynamic from "next/dynamic";

const WaveformCanvas = dynamic(() => import("./WaveformCanvas"), {
  ssr: false,                 // canvas needs window
  loading: () => <Skeleton variant="rectangular" height={120} />,
});
```

Don't dynamic-import small components — the chunk overhead exceeds the savings.

## Prefetch (next/link)

`<Link>` auto-prefetches in production when the link enters the viewport or on hover. **Don't disable this** unless you've measured network impact and care more about bandwidth than navigation speed.

For one-off cases where prefetch is wasteful (a "Sign out" link), pass `prefetch={false}`.

## React performance — when to optimize

Default: don't. React 19 + RSC dramatically reduces the surface for memo-thrashing. Optimize only when:

1. You have a measured perf problem (DevTools Profiler shows a slow render).
2. Specific known hot paths: very long lists (>500 items) → virtualize with TanStack Virtual.
3. Expensive computation in a render → `useMemo`.

**Don't** wrap every component in `React.memo`. **Don't** `useMemo` an sx object — MUI's sx caches internally.

## Long lists

If a list can reach >100 items (session list with many sessions, live caption list during a long meeting):

```tsx
import { useVirtualizer } from "@tanstack/react-virtual";

// virtualize the row container
```

Don't add `@tanstack/react-virtual` to deps until you actually have a list that exceeds 100. Premature.

## Avoid layout thrash

- `width`/`height` on images, video, canvases — always set.
- Skeletons match final element dimensions including padding.
- Don't `useState` for values you compute deterministically — derive in render.
- Don't `useEffect` to sync derived state — derive in render.

## CSS performance

- Avoid `backdrop-filter` on full-screen elements (expensive on low-end devices, partial Firefox support).
- Avoid `filter: blur()` on anything > 200x200px.
- Long lists with shadows: shadows are cheap, don't worry.
- Animations: use `transform` and `opacity` (compositor-handled). Avoid animating `width`/`height`/`top`/`left`.

## Caching strategy

- Server Components: by default Next.js `fetch` is opt-in cached. Use `{ next: { revalidate: 60 } }` for ISR.
- Database reads: wrap in `cache()` from `react` for per-request dedup (Audia already does this in [src/lib/dal.ts](../../../src/lib/dal.ts)).
- `revalidatePath`/`revalidateTag` from Server Actions after mutations.
- Don't manually cache in a `Map` — Next has primitives for this.

## Loading.tsx vs Suspense — pick by granularity

- **`loading.tsx`** — whole page is loading (route boundary).
- **`<Suspense>`** — one slot is loading, rest of page renders (streaming).

For Audia's session page, use Suspense around the transcript fetch so the title + summary render first.

## Bundle hygiene — concrete don'ts

- ❌ `import * as Icons from "@mui/icons-material"` — pulls every icon.
- ✅ `import AddIcon from "@mui/icons-material/Add"` — one icon. (`import { Add as AddIcon } from "@mui/icons-material"` also tree-shakes in modern bundlers; either is fine.)
- ❌ `import moment from "moment"` — moment is huge. Use `Intl.DateTimeFormat` or `date-fns` (tree-shakeable).
- ❌ Lodash full import. Use ESM cherry-picks: `import debounce from "lodash/debounce"`.
- ❌ Including dev tools in prod (`why-did-you-render`, profiling flags) — guard behind `process.env.NODE_ENV === "development"`.

## Server response time

Audia hits Neon Postgres via TypeORM. Slow page = usually slow query.
- Check the Neon dashboard for query time on hot paths.
- Add indexes when an `ORDER BY` / `WHERE` on a large table is slow (transcripts could grow).
- Use `take` / `limit` for pagination — don't fetch all sessions every page render.

## Image-LCP optimization for the public session page

The public session page is the most LCP-sensitive surface (search engines, social previews).
- The h1 title is likely the LCP — make sure it renders from the Server Component without waiting on a Client Component to hydrate.
- If a hero image is added, it must have `priority` and proper `sizes`.

## Bundle inspection

```bash
npm run build
# Look at the per-route table. Anything > 200 KB First Load JS deserves a look.
```

For deeper inspection, the `@next/bundle-analyzer` plugin is available but skip until needed.

## Don't optimize:

- A page that loads in 800ms because of premature React.memo. The page loads fine.
- A 10-item list because "what if it grows." Virtualize when it grows.
- An icon import because "barrel imports are bad." MUI tree-shakes correctly.
- A re-render that React DevTools shows as 0.4ms.

Spend optimization budget where the measurements point.
