# Public session pages — shareable URLs

Future feature: read-only public URL for a session, e.g. `/s/<id>`. Marketing-grade, separate from the app shell.

## Route

`src/app/s/[id]/page.tsx` — outside the authenticated app shell. Use a route group like `src/app/(public)/s/[id]/page.tsx` if you want to share a public-only layout.

Auth: optional. The URL itself is the share token; if a session is marked `public`, anyone with the URL can view. Validate in the page server component — never trust client.

## Layout

**No sidebar. No app chrome.** This is a brand-forward, public surface.

```
┌─────────────────────────────────────────────────┐
│  Audia                          [Sign in] [⋯]   │  ← thin top nav
│                                                 │
│             Q1 Planning Call                    │  ← h1, centered, 800px column
│             42 minutes · May 12                 │  ← meta
│                                                 │
│         [Listen] [Read transcript] [Copy URL]   │  ← actions
│                                                 │
│  ╭─ AI Summary ─────────────────────────╮       │
│  │ • shipped phase 4.2                  │       │
│  │ • ship 4.3 next sprint               │       │
│  ╰───────────────────────────────────────╯       │
│                                                 │
│  ╭─ Transcript ─────────────────────────╮       │
│  │ 0:00  Alright let's start...         │       │
│  │ 0:14  So the main thing is...        │       │
│  ╰───────────────────────────────────────╯       │
│                                                 │
│  Created with Audia · ↗ Try it free             │  ← subtle footer CTA
└─────────────────────────────────────────────────┘
```

## Rules

- Top nav: brand logo (links to `/`), "Sign in" link (if not authed), a share IconButton menu (Copy URL, Download transcript).
- Content column: **800px max-width**, centered. Single column. No multi-pane.
- The summary card and transcript card use the same components as the app — `SummaryBlock`, `TranscriptPanel` — in read-only mode. No Copy buttons inside cards if the top-nav Copy is the canonical action.
- **No chat panel.** Chat is an authenticated feature.
- Background: `background.default`. No gradient mesh, no hero illustration. Audia is harmonious.
- Footer: one line of `text.disabled` + a single text-link CTA.

## OG / social

Place an `opengraph-image.tsx` next to the page:

```tsx
// src/app/s/[id]/opengraph-image.tsx
import { ImageResponse } from "next/og";

export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default async function OG({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  // fetch title from DB; render brand gradient on the left, title + meta on the right
  return new ImageResponse(/* JSX */);
}
```

Reuse the canonical brand gradient (see [../theme.md](../theme.md#gradients-use-sparingly)).

Read `node_modules/next/dist/docs/` for OG image specifics in Next.js 16 — the API has shifted.

## Performance

- Public pages should be **statically renderable** where possible (session content doesn't change after the session ends). Use `generateStaticParams` if the public-session set is enumerable, or ISR with a short revalidate.
- No client-side data fetching. Everything via Server Components.
- Lazy-load the audio player below the fold — if a player is shown at all.

## Discoverability

- `robots: { index: true, follow: true }` in page metadata.
- Page `<title>`: `{session title} · Audia`.
- Twitter card: `summary_large_image` with the OG image.

## Accessibility

- Transcript is the primary content. Reachable by keyboard, proper heading hierarchy (`<h1>` title, `<h2>` Summary, `<h2>` Transcript).
- Embedded audio: proper `<audio controls>` with `aria-label="Session recording"`.
- "Copy URL" action: announce success via sonner.

## What this surface is NOT

- Not a marketing landing page — that's separate.
- Not a paywall — public is public.
- Not editable. View-only. If the visitor is the owner, show a thin "You own this session — open in Audia" link in the top nav.

## Privacy

When a session goes from private → public, surface a confirmation dialog in the app:

> Make "Q1 planning call" public?
> Anyone with the URL will be able to view the transcript and AI summary. You can revert this anytime.

Default the dialog button to Cancel (focus on Cancel), not Confirm — destructive defaults are an anti-pattern.
