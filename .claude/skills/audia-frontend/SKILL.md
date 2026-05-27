---
name: audia-frontend
description: Audia frontend rules + patterns — MUI v9 only (no Tailwind), three-hue purple-family palette, Geist Sans, hub-and-detail shell capped at 1440 on large screens. Activates on any UI/UX work: adding/editing components, theming, layout, responsiveness, motion, accessibility, empty states, or the live-transcription and shared public-session surfaces. Read this file first on every UI task; pull in theme.md / components.md / layout.md / responsive.md / motion.md / a11y.md / future/* / patterns/* only as needed.
---

# Audia frontend — binding spec

You are the frontend engineer for Audia. Every UI decision passes through the rules below before code is written. When a rule conflicts with default training-time intuition, the rule wins.

## What Audia is

A meeting transcription app. Users record/upload audio, get a transcript + AI summary, then chat with the transcript (RAG). The product feel target: **iOS / Things / Arc** — soft, harmonious, premier. Not Linear-sharp. Not Stripe-loud.

## The 5 binding rules

**1. MUI v9 is the only UI system.**
- Use `@mui/material` + `@mui/icons-material` for every component, every layout, every style.
- Tailwind utility classes are **forbidden** in new code. Tailwind is scheduled for removal.
- Inline styles, CSS modules, and `styled-components` are also forbidden. Use `sx` (object form). For extraction, use `styled()` from `@mui/material/styles`.
- One escape hatch: a CSS variable in `globals.css` is OK *only* when a server component needs a theme-conditional value (see `--audia-card-shadow`).

**2. The palette is 3 hues + neutrals. Never add a fourth.**
- `primary` purple, `secondary` indigo, `info` cyan (reserved for the chat surface). Plus text/background neutrals.
- `success`, `warning`, `error` exist in the palette but are **state-only** — never used as decorative accents.
- Read exact tokens from [theme.md](theme.md). Never hardcode hex; use `palette.*` paths in `sx` or `var(--mui-palette-*)` in raw CSS.

**3. Dark mode uses `theme.vars.palette.*`, not `theme.palette.*`.**
- In component `styleOverrides` and anywhere static: write `theme.vars.palette.primary.main`. The plain `palette.*` form bakes the light hex at theme-construction time and **never swaps** to dark.
- In `sx` you can use string paths like `"primary.main"` and MUI handles it.
- In server components, function-form `sx={(t) => …}` is banned (RSC can't pass functions). Use plain object `sx` with string paths, or a CSS variable.
- `useColorScheme()` returns `undefined` during SSR — gate dark-aware UI behind a `mounted` state.

**4. Radii and shape are settled. Don't redesign them.**
- Base radius `12`, Card `16`, Chip `8`, IconButton `8`. Not up for debate.
- No box-shadow elevations on cards. Surfaces are distinguished by `divider` borders + background tone.
- Buttons are `textTransform: "none"`, `fontWeight: 600`, no elevation. Don't override.

**5. Layout is hub-and-detail; content caps at ~1440 on big screens.**
- Sidebar 320px (left) + main pane (right). Mobile: drawer + AppBar.
- Main pane is **state-driven**, not stacked sections. One of `Ready`, `Recording`, `Processing`, `SessionView` at a time.
- On screens ≥1600px, the main pane content caps at 1440px and centers. Sidebar stays 320. Use [`patterns/shell-container.tsx`](patterns/shell-container.tsx).
- Recorder/upload affordances live in `Ready` state. **Never in the sidebar.** Sidebar is pure navigation.

**6. Server-first. RSC by default; `"use client"` only when forced.**
- Every file is a Server Component unless it starts with `"use client"`. Adopt the boundary as low in the tree as possible — wrap the interactive *leaf*, not the whole page.
- Mutations go through **Server Actions** in `src/app/actions/*.ts` + `useActionState` + `useFormStatus`. Not `fetch("/api/...")` from a Client Component.
- Snappy edits (title rename, chat send) use **`useOptimistic`** for instant UI + automatic rollback on failure.
- Data fetching lives in Server Components — never `useEffect(() => fetch("/api/..."))` for data you own.
- Co-locate route boundaries: `loading.tsx`, `error.tsx`, `not-found.tsx`.
- Full rules + examples in [state-and-data.md](state-and-data.md).

## Definition of done for a UI task

Before reporting any UI task as complete, walk this checklist:

- [ ] No hardcoded hex. Every color comes from `palette.*` or `var(--mui-palette-*)`.
- [ ] Works in light AND dark. Toggle flipped, confirmed.
- [ ] Works at 360px, 768px, 1280px, 1920px. No horizontal scroll, no clipped controls.
- [ ] Keyboard reaches every action. Focus ring is visible on `:focus-visible`.
- [ ] Toasts via `sonner` for silent fetch failures. No `alert()`, no inline error banners for transient issues.
- [ ] No Tailwind classes added. No `styled-components`. No inline `style={{…}}` (except CSS variables).
- [ ] No `useEffect` to fetch data in pages — Server Components or Server Actions / Route Handlers.

If a check is impossible (e.g. can't run dev server in this environment), say so explicitly. Don't claim "tested" without testing.

## When to read each sub-doc

| Sub-doc | Read when |
|---|---|
| [theme.md](theme.md) | Adjusting palette, gradients, dark-mode behavior, or adding a `components` override. |
| [components.md](components.md) | Choosing an MUI component, writing `sx`, z-index, truncation, or wiring forms / dialogs / drawers / menus. |
| [layout.md](layout.md) | Adding a new page or surface, a settings section, or anywhere with cards-in-a-grid. |
| [responsive.md](responsive.md) | Anything touching breakpoints, mobile drawer, big-screen behavior, or touch targets. |
| [motion.md](motion.md) | Animating something, building a skeleton, or wiring the waveform / live transcription. |
| [a11y.md](a11y.md) | Forms, live updates, focus traps, color-coded states, keyboard shortcuts. |
| [state-and-data.md](state-and-data.md) | Anything stateful or async: forms, mutations, optimistic UI, Suspense, loading/error boundaries. |
| [performance.md](performance.md) | Adding a heavy dep, an image, a long list, or chasing a Lighthouse score. |
| [polish.md](polish.md) | Reaching for "make it feel premium" — the small details that signal craft. |
| [future/live-transcription.md](future/live-transcription.md) | Touching recording, streaming captions, waveform, or live-region announcements. |
| [future/public-session.md](future/public-session.md) | Building the shared / public read-only session URL surface. |
| [patterns/](patterns/) | Need a ready-made primitive — `EmptyState`, `Section`, `ShellContainer`, `ListStates`. Copy them; don't reinvent. |

## Hard prohibitions

- **No Tailwind classes** (`className="…"` with utility classes) in new code. If you find them in existing code, plan a removal pass — don't extend them.
- **No hardcoded hex** in component code. Theme tokens only.
- **No new icon libraries.** `@mui/icons-material` covers ~2000 icons.
- **No Framer Motion / motion / gsap** unless a task genuinely needs gestures or spring physics. Default to MUI transitions + CSS keyframes ([motion.md](motion.md)).
- **No new font families.** Geist Sans (body) + Geist Mono (timestamps/durations) only.
- **No native HTML form validation popups.** Use `noValidate` + Zod + `helperText` + sonner toast.
- **No new accent colors.** Want emphasis? Use weight, scale, or the purple→indigo gradient. Not a new hue.

## When the user asks for "something cooler"

Cooler in Audia means: more harmonious, more confident in the three-hue palette, more breathing room — not more saturated, not more shadows, not more colors. If you're tempted to add a teal or pink to "spice it up," stop. Reach for the purple→indigo gradient instead. See [theme.md](theme.md#gradients-use-sparingly).

## Read Next.js 16 docs before writing app code

From [AGENTS.md](../../../AGENTS.md): this is Next.js 16, not the Next.js in your training. Read `node_modules/next/dist/docs/` before writing route handlers, server actions, or `params` / `cookies` usage. `params` is a Promise. `cookies()` is async. `proxy.ts` replaces `middleware.ts`. Heed deprecation notices.

## Existing source-of-truth files

- Palette + tokens + component overrides: [src/lib/theme.ts](../../../src/lib/theme.ts)
- Providers (theme + sonner + RSC cache): [src/app/providers.tsx](../../../src/app/providers.tsx)
- Global keyframes + scrollbar + CSS variables: [src/app/globals.css](../../../src/app/globals.css)
- Brand constants (logo SVG, favicon source): [src/app/components/brand.ts](../../../src/app/components/brand.ts)
- All components: [src/app/components/](../../../src/app/components/) (NOT `src/components/`)
