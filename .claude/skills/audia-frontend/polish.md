# Polish — the small things that signal premier

Premier feel is a hundred small details, consistently applied. Most live in [globals.css](../../../src/app/globals.css); a few are runtime patterns.

## Already done in globals.css

Reuse these — don't re-implement.

| Polish detail | Where | How to use |
|---|---|---|
| Brand-colored text selection | `::selection` rule | Automatic everywhere. |
| Brand-colored caret in inputs | `input, textarea` rule | Automatic. |
| iOS tap-flash killed | `-webkit-tap-highlight-color: transparent` | Automatic. |
| Stable scrollbar gutter (no layout shift) | `scrollbar-gutter: stable` on html | Automatic. |
| Reduced-motion catch-all | `@media (prefers-reduced-motion: reduce)` | Add new keyframes to the catch-all rule. |
| Tabular numerals for timers | `.tabular-nums` utility class | `<Typography className="tabular-nums">02:14</Typography>` |
| iOS safe-area at the bottom | `.safe-bottom` utility class | Use on sticky bottom controls / bottom sheets. |
| Multi-line ellipsis truncation | `.line-clamp` utility class | `<Box className="line-clamp" sx={{ ['--lines' as any]: 2 }}>{longText}</Box>` |

## Truncation

### Single-line ellipsis
```tsx
<Typography
  noWrap
  sx={{ minWidth: 0 /* required when inside a flex parent */ }}
>
  {longTitle}
</Typography>
```

### Multi-line ellipsis
```tsx
<Box className="line-clamp" sx={{ ['--lines' as any]: 2 }}>
  {longDescription}
</Box>
```

If the truncated content is interactive, **add `title={fullText}` or a Tooltip** so the full text is recoverable.

## Hover discipline

Only apply hover effects on devices that can hover:

```tsx
sx={{
  "@media (hover: hover)": {
    "&:hover": { bgcolor: "action.hover" },
  },
}}
```

Without this, mobile shows hover state on tap and it sticks until you tap elsewhere — looks broken.

## Active press feedback

For primary CTAs and clickable cards on mobile, a tiny scale-down on active feels premium:

```tsx
sx={{
  transition: "transform 80ms ease",
  "&:active": { transform: "scale(0.98)" },
}}
```

Keep the scale small (0.96–0.99). Don't combine with MUI's ripple — pick one.

## Focus-visible

MUI ships proper `:focus-visible` focus rings. Don't disable them. If you must customize:

```tsx
sx={{
  "&:focus-visible": {
    outline: "2px solid",
    outlineColor: "primary.main",
    outlineOffset: 2,
  },
  "&:focus:not(:focus-visible)": { outline: "none" },
}}
```

## Backdrop blur — use sparingly

For the mobile drawer backdrop and sticky AppBar, a subtle blur reads premium:

```tsx
// AppBar
sx={{
  position: "sticky",
  top: 0,
  backdropFilter: "blur(8px) saturate(180%)",
  bgcolor: "color-mix(in srgb, var(--mui-palette-background-default) 80%, transparent)",
}}
```

Don't apply blur to large surfaces — it's expensive on low-end devices and on Firefox.

## Drawer top handle (when used as bottom sheet)

```tsx
<Drawer
  anchor="bottom"
  PaperProps={{ sx: { borderTopLeftRadius: 16, borderTopRightRadius: 16 } }}
>
  <Box sx={{ display: "flex", justifyContent: "center", pt: 1.5 }}>
    <Box sx={{ width: 36, height: 4, borderRadius: 2, bgcolor: "divider" }} />
  </Box>
  {/* content */}
</Drawer>
```

Small drag handle reads as "draggable bottom sheet" to anyone who's used iOS.

## Cursor pointer on every clickable

MUI buttons handle this. Custom clickable rows (a `ButtonBase`-wrapped row) get it for free. **Plain divs with `onClick`** don't — and they shouldn't exist (see [a11y.md](a11y.md)).

## Number / date formatting

Don't write `${minutes}:${seconds}` manually for durations. Centralize formatters:

```ts
// src/app/components/utils.ts (where it likely lives)
export const formatDuration = (sec: number) =>
  `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, "0")}`;

export const formatDate = (d: Date) =>
  new Intl.DateTimeFormat(undefined, { dateStyle: "medium" }).format(d);

export const formatRelative = (d: Date) =>
  new Intl.RelativeTimeFormat(undefined, { numeric: "auto" }).format(/* …compute… */);
```

`Intl.*` is built into the browser; no library needed.

## Logo / illustration in both schemes

When adding any new SVG / image asset that has color:
- Either test it in both light and dark mode and confirm it reads well, OR
- Provide two variants and swap via the `.mui-color-scheme-dark` class.

The BrandLogo uses option (a) — one SVG, with an optional white border ring shown only in dark mode via a CSS class hook. Mimic that pattern when you can.

## Animation discipline (review of motion.md key rules)

- Page transitions: **no**. Content streams in; navigation is instant.
- Skeletons: match final radius, stop within 300ms of data arrival.
- Stagger lists: max 5 items, max 50ms between.
- Springs / bouncy physics: **no**. Smooth, not bouncy.

## Forms — quiet polish

- Autocomplete attributes filled in (`autoComplete="email" / "current-password" / "new-password"`). Password managers reward you.
- `type="email"` / `type="tel"` / `type="number"` set semantically so mobile keyboards adapt.
- Submit-on-Enter works without you wiring it — `<form>` handles it.
- The pending state should be visible **without layout shift** — keep button size constant, change content with `<CircularProgress size={16}/>` swapped in.

## Print stylesheet (nice to have)

Add a `@media print` rule when shipping the public session page:

```css
@media print {
  .no-print { display: none !important; }
  body { background: white; color: black; }
  a { color: black; text-decoration: underline; }
}
```

Tag chrome (sidebar, share buttons, footer) with `className="no-print"` so transcripts print cleanly. Not blocking work — do it the first time someone tries to print and it looks bad.

## Tiny details that make a difference

- `<wbr>` inside long unbroken strings (URLs, IDs) so they break gracefully.
- `text-wrap: balance` on h1/h2/h3 (modern browsers wrap headings more attractively):
  ```tsx
  <Typography variant="h1" sx={{ textWrap: "balance" }}>
  ```
- `text-wrap: pretty` on long body paragraphs (avoids orphans):
  ```tsx
  <Typography variant="body1" sx={{ textWrap: "pretty" }}>
  ```
- `min-width: 0` on flex children that contain text — prevents overflow blowups when the text is long.
- `overflow-wrap: anywhere` on user-generated long strings.

## When the user says "make it feel more premium"

Don't reach for more color, more shadow, or more motion. Reach for:
1. More **consistency** — are all card radii really 16? All button heights matching?
2. More **breathing room** — bump section spacing one tier (`spacing={3}` → `spacing={4}`).
3. More **typographic care** — `textWrap: "balance"` on titles, tabular numerals on metrics.
4. The **active press feedback** above on every clickable.
5. A **subtle backdrop blur** on the AppBar.

That's it. The palette and shape are already premier; the polish is in restraint.
