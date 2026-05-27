# Theme — palette, tokens, dark mode

Single source of truth: [src/lib/theme.ts](../../../src/lib/theme.ts). This file explains the decisions behind it and how to change it safely.

## The 3-hue discipline

| Hue | MUI slot | Role |
|---|---|---|
| Purple (`#5b21b6` / `#a78bfa`) | `primary` | Brand identity, primary actions, AI surfaces (summary card accent, brand gradient). |
| Indigo (`#4f46e5` / `#818cf8`) | `secondary` | Companion to primary in gradients. Rare standalone use. |
| Cyan (`#0ea5e9` / `#38bdf8`) | `info` | **Reserved for the chat panel only.** Visually separates Q&A from the AI summary card. |

State colors (`success`, `warning`, `error`) exist but are **state-only** — pills, alerts, validation messages. Never used to color a hero element, section header, or decorative accent.

If a feature feels like it "wants" a new color, use **weight + scale + gradient** instead:
- The canonical brand gradient: `primary.dark` → `primary.main` → `secondary.main` (see SummaryBlock's accent strip).
- Increase font-weight or size before reaching for color.
- Use `text.secondary` vs `text.primary` for hierarchy, not a new hue.

## Tokens

### Light
- `primary.main` `#5b21b6` · `primary.dark` `#4c1d95` · `primary.light` `#7c3aed`
- `secondary.main` `#4f46e5` · `secondary.dark` `#4338ca` · `secondary.light` `#818cf8`
- `info.main` `#0ea5e9` · `info.light` `#38bdf8` · `info.dark` `#0284c7`
- `background.default` `#f1f4f9` · `background.paper` `#ffffff`
- `text.primary` `#1e1b4b` · `text.secondary` `#64748b` · `text.disabled` `#94a3b8`
- `divider` `#e2e8f2`

### Dark
- `primary.main` `#a78bfa` · `primary.dark` `#8b5cf6` · `primary.light` `#c4b5fd`
- `secondary.main` `#818cf8` · `secondary.dark` `#6366f1` · `secondary.light` `#a5b4fc`
- `info.main` `#38bdf8` · `info.light` `#7dd3fc` · `info.dark` `#0ea5e9`
- `background.default` `#0b0a17` (near-black with purple tint) · `background.paper` `#15132a`
- `text.primary` `#f1f4f9` · `text.secondary` `#cbd5e1` · `text.disabled` `#64748b`
- `divider` `rgba(167,139,250,0.16)`

Why brighter purples in dark mode: `#5b21b6` would disappear against `#0b0a17`. Inverted toward `#a78bfa` for WCAG AA contrast on dark.

## Typography

- Body / UI: `var(--font-geist-sans)` (loaded in `layout.tsx`).
- Timestamps, durations, code-ish numbers: `var(--font-geist-mono)`.
- Heading weights: h1/h2/h3 = 700. Default body = 400. Buttons = 600.
- `textTransform: "none"` everywhere. No ALL-CAPS buttons.
- `variant="overline"` *is* allowed for tiny section labels — `letterSpacing: "0.2em"`, `fontWeight: 700`, `color: "primary.main"`. See SummaryBlock pattern.

## Shape

- Base radius: 12 (theme.shape.borderRadius).
- Cards: 16. Chips, IconButtons: 8. Inputs: 12.
- No box-shadow elevation. `MuiPaper.defaultProps.elevation = 0` and `backgroundImage: "none"`.
- Surfaces stack via `background.paper` over `background.default` + a 1px `divider` border.

## Gradients (use sparingly)

The single canonical brand gradient:

```css
linear-gradient(to right,
  var(--mui-palette-primary-dark),
  var(--mui-palette-primary-main),
  var(--mui-palette-secondary-main)
)
```

Acceptable uses:
- The BrandLogo circle fill.
- A 3px accent strip on the top edge of AI-generated content cards (SummaryBlock pattern).
- A very subtle hover glow on the primary CTA.

**Forbidden uses:**
- Full-bleed page backgrounds.
- Card body fills (use `background.paper`).
- Button fills (use `primary.main`).

The gradient is an *accent*, not a *surface*.

## Dark mode — the gotchas

These have bitten this repo before. Read carefully.

### `theme.vars.palette.*` in static overrides

When `cssVariables: { colorSchemeSelector: "class" }` is set, MUI emits CSS variables that swap when `.mui-color-scheme-dark` is applied to `<html>`. But this only works if your override references the variable, not the resolved value.

```ts
// ❌ baked-in light hex; never swaps
MuiCard: { styleOverrides: { root: ({ theme }) => ({
  border: `1px solid ${theme.palette.divider}`,
}) } }

// ✅ CSS variable; swaps on class change
MuiCard: { styleOverrides: { root: ({ theme }) => ({
  border: `1px solid ${theme.vars.palette.divider}`,
}) } }
```

### `sx` accepts string paths

In `sx`, write the token path. MUI resolves it correctly per scheme.

```tsx
<Box sx={{ color: "primary.main", bgcolor: "background.paper", borderColor: "divider" }} />
```

### No function-form `sx` in server components

```tsx
// ❌ Server component → Client component cannot accept functions
<Card sx={(t) => ({ boxShadow: t.shadows[4] })} />

// ✅ Plain object, CSS variable for theme-conditional bits
<Card sx={{ boxShadow: "var(--audia-card-shadow)" }} />
```

`--audia-card-shadow` lives in `globals.css` and swaps under `.mui-color-scheme-dark`. Use the same pattern for any theme-conditional value a server component needs.

### Hydration guard for `useColorScheme`

```tsx
const [mounted, setMounted] = useState(false);
useEffect(() => setMounted(true), []);
const { mode } = useColorScheme();
// don't render mode-dependent UI until mounted, or SSR will mismatch
```

## Adding a new component override

1. Add the override to `components` in [src/lib/theme.ts](../../../src/lib/theme.ts) — not in a one-off component.
2. Use `theme.vars.palette.*` in any function-form override that touches color / border / background.
3. Verify both schemes by flipping the toggle.
4. Verify SSR doesn't flash the wrong colors — `<InitColorSchemeScript />` in `layout.tsx` handles this; don't remove it.

## Existing CSS variables (globals.css)

| Variable | Purpose | Swaps |
|---|---|---|
| `--audia-card-shadow` | Soft card shadow for server-component contexts | Light: purple-tinted soft drop. Dark: deeper black. |

Add new CSS variables only when a server component needs a theme-conditional value. Otherwise prefer MUI tokens.

## Scrollbar

Custom 5px scrollbar lives in `globals.css`, scheme-aware. Don't override per-component.

## Summary

3 hues, no more. Tokens via theme. `theme.vars.*` in static overrides. `palette.*` string paths in `sx`. Gradient is an accent, not a surface.
