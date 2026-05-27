# Responsive — breakpoints, mobile, large screens

## Breakpoints (MUI defaults — don't override)

| Breakpoint | Width | Audia behavior |
|---|---|---|
| `xs` | 0 – 599 | Sidebar = Drawer. AppBar visible. Single column. |
| `sm` | 600 – 899 | Same as `xs` plus more padding. |
| `md` | 900 – 1199 | Sidebar permanent. AppBar hidden. |
| `lg` | 1200 – 1535 | Same as `md`. |
| `xl` | 1536+ | Content caps at 1440 inside main pane (centered). |

`md` (900px) is the **sidebar-permanent threshold**. Below = drawer. At or above = permanent.

## Test resolutions (every UI task)

| Width | Device class | What to verify |
|---|---|---|
| 360 | small mobile | No horizontal scroll. Tap targets ≥ 44×44. Drawer works. |
| 768 | tablet portrait | Still drawer. Spacing comfortable. |
| 1280 | laptop | Sidebar permanent. Content fills main pane. |
| 1440 | large laptop | Same as 1280; content uses ~1100 of main pane width. |
| 1920 | desktop | Main pane content caps at 1440, centered. Sidebar still 320. |
| 2560 | 4K / wide | Sidebar still 320. Vast empty area both sides of capped content. Looks intentional, not broken. |

If you can't run the dev server, say so in the response and ask the user to verify. Don't claim "tested on mobile" without actually checking.

## Responsive sx pattern

```tsx
sx={{
  p: { xs: 2, md: 3, xl: 4 },
  flexDirection: { xs: "column", md: "row" },
  fontSize: { xs: 14, md: 16 },
}}
```

Omit intermediate breakpoints when they match a neighbor — keeps it readable. Don't write `{ xs: 2, sm: 2, md: 3 }` when `{ xs: 2, md: 3 }` works.

## Touch targets

- Minimum tap area: 44×44 px (WCAG 2.5.5). MUI's default `IconButton` is 40 — bump to `size="large"` (48) on mobile.
- Spacing between adjacent tap targets: ≥ 8px.
- Hero record button on mobile: ≥ 72px height for confident tapping.

## Mobile-specific patterns

- **AppBar height**: 56 (default). Don't change.
- **Drawer**: 320px wide on mobile, slides in from left. Optional `BackdropProps={{ sx: { backdropFilter: "blur(2px)" } }}` for premier feel.
- **Bottom sheet** (future need): `Drawer anchor="bottom"` with rounded top corners — `PaperProps={{ sx: { borderTopLeftRadius: 16, borderTopRightRadius: 16 } }}`.
- **Sticky controls**: when a primary action would otherwise scroll out of view (long forms), pin it to the bottom on `<sm` with a `position: "sticky", bottom: 0, bgcolor: "background.paper", borderTop: 1, borderColor: "divider"` wrapper.

## Large-screen patterns

- Use [patterns/shell-container.tsx](patterns/shell-container.tsx) for the 1440 cap.
- Don't right-align controls to the screen edge on 4K — keep them inside the cap.
- For future multi-column dashboards: 2-col grid at `xl` (≥1536). If a 3-col layout is genuinely warranted at ≥1920, add a custom breakpoint in `theme.breakpoints.values` rather than hardcoding `@media`.

## Avoid

- Custom breakpoints (`@media (min-width: 1100px)`) — use `theme.breakpoints.up("md")` or sx breakpoint objects.
- Fixed pixel widths on cards. Use `flex` / `grid` / `maxWidth` with breakpoints.
- `100vw` on anything but full-screen overlays (it ignores scrollbar width and causes horizontal scroll).
- `vh` on mobile heights (iOS Safari toolbar shifts cause jumps) — prefer `dvh` (`100dvh`) for full-height surfaces.

## Mobile-first vs desktop-first

Default to **mobile-first**: write the `xs` value as the base, then add `md` overrides. It produces cleaner CSS and forces you to think about constraints first.

```tsx
// ✅ Mobile-first
sx={{ p: 2, flexDirection: "column", md: { p: 3, flexDirection: "row" } }}
// (the actual sx breakpoint syntax — equivalent below)
sx={{ p: { xs: 2, md: 3 }, flexDirection: { xs: "column", md: "row" } }}
```

## Image / media responsiveness

- Use Next.js `<Image>` for any image asset. Set `sizes` honestly: `"(max-width: 900px) 100vw, 720px"`.
- Avatars: fixed size (`width={40} height={40}`), don't make them fluid.
- Video / audio elements: `width: "100%", maxWidth: 720` for in-content media.

## Checklist before "responsive done"

- [ ] DevTools at 360px — no horizontal scroll, no clipped tap targets.
- [ ] DevTools at 768px — drawer still works, content reflows comfortably.
- [ ] DevTools at 1280px — sidebar permanent, no awkward gaps.
- [ ] DevTools at 1920px — content centered with cap, sidebar pinned left.
- [ ] Resize between breakpoints — no janky jumps, smooth transitions.
