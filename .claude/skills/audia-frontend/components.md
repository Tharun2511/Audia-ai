# Components — MUI patterns & sx rules

Audia's components live in [src/app/components/](../../../src/app/components/). Read this before adding a new one.

## Picking the right MUI component

| Need | MUI component | Notes |
|---|---|---|
| Container with title + body | `Card` + `CardContent` | Title goes outside the card when the card represents *content*; inside (in CardHeader) only for settings-style sections. |
| Vertical layout, even spacing | `Stack spacing={2}` | Prefer Stack over Box+gap unless you need grid. |
| Horizontal layout | `Stack direction="row"` | With `alignItems`, `justifyContent`. |
| Multi-column responsive grid | `Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "1fr 1fr" }, gap: 2 }}` | MUI Grid v2 also fine; raw CSS grid keeps API surface smaller. |
| Inline label / pill | `Chip` | Rounded 8. Use `size="small"` in dense rows. |
| Status pill | `Chip` `color="primary"` `variant="outlined"` | Outlined preferred for state; filled chips compete with cards. |
| Tooltip | `Tooltip` | Already styled (dark text-primary bg). Use `arrow` only on dense surfaces. |
| Confirmation modal | `Dialog` | `DialogTitle` / `DialogContent` / `DialogActions`. |
| Side panel | `Drawer` | `anchor="right"`, `PaperProps={{ sx: { width: { xs: "100%", sm: 480 } } }}`. |
| Menu | `Menu` + `MenuItem` | Anchor to an IconButton. Use for context actions. |
| Empty list | `EmptyState` primitive | See [patterns/empty-state.tsx](patterns/empty-state.tsx). |
| Page section | `Section` primitive | See [patterns/section.tsx](patterns/section.tsx). |

## sx — the rules

1. **Object form preferred.** Function form `sx={(t) => …}` cannot cross the RSC boundary.
2. **String paths for tokens.** `color: "primary.main"`, `bgcolor: "background.paper"`, `borderColor: "divider"`.
3. **Spacing uses the scale.** `p: 2` = 16px. `gap: 1` = 8px. Never `padding: "16px"`.
4. **Responsive via breakpoint object.** `{ xs: 1, sm: 2, md: 3 }` — not media queries in CSS.
5. **No `!important`.** If something's overriding you, fix it at the theme override layer in `theme.ts`.

```tsx
<Card sx={{
  p: { xs: 2, md: 3 },
  borderColor: "divider",
  bgcolor: "background.paper",
}}>
  <Typography variant="overline" sx={{ color: "primary.main" }}>
    Summary
  </Typography>
</Card>
```

## Buttons

```tsx
// Primary CTA
<Button variant="contained" color="primary" size="large">Start recording</Button>

// Secondary action
<Button variant="outlined" color="primary">Upload file</Button>

// Tertiary / inline
<Button variant="text" color="primary">Cancel</Button>

// Icon-only
<IconButton aria-label="Copy transcript"><ContentCopyIcon /></IconButton>
```

Rules:
- Always provide `aria-label` on icon-only buttons.
- Destructive actions use `color="error"` with `variant="outlined"`, never filled red.
- `size="large"` only for the hero record button. Everything else is default.
- `disableElevation` is set globally; don't override.
- For loading states: `disabled={pending}` + a leading `<CircularProgress size={16} />` or use the existing `aria-busy` pattern.

## Inputs

```tsx
<TextField
  label="Session title"
  value={title}
  onChange={(e) => setTitle(e.target.value)}
  error={!!error}
  helperText={error ?? "Renaming syncs immediately"}
/>
```

Rules:
- `variant="outlined"` is the global default (set in theme).
- Always pair `error` with `helperText`. Surface submission-time errors in a sonner toast too.
- `noValidate` on `<form>` — Zod handles validation, not the browser.
- Use `PasswordInput` (existing) for password fields; don't reinvent the show/hide toggle.
- Don't use `placeholder` as a label. Both can coexist for hint text.

## Cards

The canonical Audia card pattern:

```tsx
<Card sx={{ overflow: "hidden" }}>
  {/* Optional 3px accent strip — reserved for AI-generated content surfaces */}
  <Box sx={{
    height: 3,
    background: "linear-gradient(to right, var(--mui-palette-primary-dark), var(--mui-palette-primary-main), var(--mui-palette-secondary-main))",
  }} />
  <CardContent sx={{ p: 2 }}>
    <Stack direction="row" spacing={1} sx={{ alignItems: "center", mb: 1.5 }}>
      <AutoAwesomeIcon sx={{ fontSize: 14, color: "primary.main" }} />
      <Typography variant="overline" sx={{
        fontWeight: 700,
        letterSpacing: "0.2em",
        color: "primary.main",
        lineHeight: 1,
      }}>
        Section label
      </Typography>
    </Stack>
    {/* body */}
  </CardContent>
</Card>
```

- `overline` + 0.2em letter-spacing for section labels.
- Border + paper bg distinguishes cards. No box-shadow.
- The accent strip is reserved for AI surfaces (summary; future: cited chat answers).

## Lists

```tsx
<List dense disablePadding>
  {items.map((item) => (
    <ListItem key={item.id} disableGutters sx={{ py: 0.75 }}>
      <Typography variant="body2">{item.text}</Typography>
    </ListItem>
  ))}
</List>
```

For session navigation (the sidebar list), use the existing `SessionListItem` — it has the active-state styling already.

## Dialogs

```tsx
<Dialog open={open} onClose={() => setOpen(false)} maxWidth="sm" fullWidth>
  <DialogTitle>Delete this session?</DialogTitle>
  <DialogContent>
    <DialogContentText>This can't be undone.</DialogContentText>
  </DialogContent>
  <DialogActions>
    <Button onClick={() => setOpen(false)}>Cancel</Button>
    <Button onClick={handleDelete} color="error" variant="contained">Delete</Button>
  </DialogActions>
</Dialog>
```

- `fullWidth` + `maxWidth="sm"` for confirmations. `"md"` for content-heavy dialogs.
- On mobile, use `fullScreen={useMediaQuery(theme.breakpoints.down("sm"))}` for content-heavy dialogs; confirmations stay centered.

## Drawers

```tsx
<Drawer
  anchor="right"
  open={open}
  onClose={() => setOpen(false)}
  PaperProps={{ sx: { width: { xs: "100%", sm: 480 } } }}
>
  {/* content */}
</Drawer>
```

For the mobile sidebar, the existing pattern in `HomeClient.tsx` is canonical — match it for any new drawer.

## Menus

```tsx
<IconButton aria-label="More actions" onClick={(e) => setAnchor(e.currentTarget)}>
  <MoreHorizIcon />
</IconButton>
<Menu anchorEl={anchor} open={!!anchor} onClose={() => setAnchor(null)}>
  <MenuItem onClick={handleRename}>Rename</MenuItem>
  <MenuItem onClick={handleDelete} sx={{ color: "error.main" }}>Delete</MenuItem>
</Menu>
```

- Anchor to an IconButton with explicit `aria-label`.
- Destructive items: only color the *text* error.main, not the whole row.

## Toasts (sonner)

```tsx
import { toast } from "sonner";

try {
  await something();
  toast.success("Saved");
} catch (e) {
  toast.error("Couldn't save", { description: e instanceof Error ? e.message : undefined });
}
```

- Already mounted in `providers.tsx` via `ThemedToaster`.
- Use for every silent fetch failure and async success worth confirming.
- `position="top-right"`, `richColors`, `closeButton`, `duration: 4000` — don't change per-toast.

## Forbidden patterns

| Forbidden | Use instead |
|---|---|
| `className="flex gap-4"` (Tailwind) | `<Stack direction="row" spacing={2}>` |
| `style={{ color: "#5b21b6" }}` | `sx={{ color: "primary.main" }}` |
| `<button>` (raw HTML) | `<Button>` or `<IconButton>` |
| `<input>` (raw HTML) | `<TextField>` |
| `useEffect(() => fetch(...), [])` | Server Component or Server Action |
| `alert(...)` / `confirm(...)` | `toast` or `Dialog` |
| New icon library | `@mui/icons-material` |
| Custom font import | Geist Sans / Geist Mono only |

## Z-index discipline

Never hardcode `zIndex: 9999`. Use `theme.zIndex.*`:

| Token | Value | Use |
|---|---|---|
| `theme.zIndex.mobileStepper` | 1000 | Mobile steppers |
| `theme.zIndex.fab` | 1050 | Floating action button |
| `theme.zIndex.speedDial` | 1050 | Speed dial |
| `theme.zIndex.appBar` | 1100 | App bar |
| `theme.zIndex.drawer` | 1200 | Drawer |
| `theme.zIndex.modal` | 1300 | Modal / Dialog |
| `theme.zIndex.snackbar` | 1400 | Snackbar / Toast |
| `theme.zIndex.tooltip` | 1500 | Tooltip |

```tsx
sx={{ position: "sticky", top: 0, zIndex: "appBar" }}
```

If you need a custom layer between two known layers, use the value `+ 1`:

```tsx
sx={(t) => ({ zIndex: t.zIndex.appBar + 1 })}
// (client component only — function-form sx)
```

Never use 4-digit numbers like `999`/`9999`. They will fight MUI's stack.

## Text truncation

### Single line
```tsx
<Typography noWrap sx={{ minWidth: 0 }}>{longTitle}</Typography>
```

`minWidth: 0` is required when the typography sits inside a flex container — without it, text overflows the flex track instead of truncating.

### Multi-line (clamp)
The `.line-clamp` utility class is wired in [globals.css](../../../src/app/globals.css):

```tsx
<Box className="line-clamp" sx={{ ['--lines' as any]: 2 }}>
  {longDescription}
</Box>
```

If the truncated text is meaningful, add a `title={fullText}` attribute or wrap in `<Tooltip>` so users can recover the full content.

## Lists need three states

Every list-bearing surface must handle:

1. **Loading** — skeleton matching the row shape.
2. **Empty** — `EmptyState` primitive ([patterns/empty-state.tsx](patterns/empty-state.tsx)) with one action.
3. **Error** — inline retry, or let the route's `error.tsx` boundary catch it.

For client-side fetched lists, use `ListStates` ([patterns/list-states.tsx](patterns/list-states.tsx)):

```tsx
<ListStates
  data={results}
  loading={pending}
  error={error}
  emptyTitle="No matches"
  emptyDescription="Try a different keyword."
  renderRow={(s) => <SessionListItem key={s.id} session={s} />}
/>
```

Server-Component-fetched lists handle empty/loaded inline; `loading.tsx` covers loading; `error.tsx` covers error.

## MUI v9 — banned components / APIs

These existed in older MUI but are wrong here:

| Forbidden | Why | Use instead |
|---|---|---|
| `Grid` v1 (`Grid container` + `Grid item`) | Deprecated. | `Grid` v2 (no `item`) or raw CSS grid via `Box sx={{ display: "grid" }}`. |
| `Hidden` | Removed in v5+. | sx breakpoint object: `sx={{ display: { xs: "none", md: "block" } }}`. |
| `makeStyles` / `withStyles` | Removed (was @mui/styles). | `sx` or `styled()` from `@mui/material/styles`. |
| `createMuiTheme` | Renamed. | `createTheme`. |
| `Box component="span"` for everything | Unnecessary `<Box>` everywhere. | Use the right semantic element directly. |

## When to create a new component file

- Used in 2+ places? Extract.
- 100+ lines and has its own state? Extract.
- Otherwise inline. Don't pre-extract.

New components go in [src/app/components/](../../../src/app/components/), PascalCase filename matching the export name.
