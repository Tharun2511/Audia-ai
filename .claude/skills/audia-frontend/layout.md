# Layout — shell, sidebar, sections, containers

## The app shell

Audia has one app shell (the authenticated app at `/`) and one marketing-grade shell (the future public session page at `/s/[id]`). This doc covers the app shell.

```
┌─────────────────────────────────────────────────────────┐
│  ┌─────────┐  ┌────────────────────────────────────┐    │
│  │ Sidebar │  │ Main pane                          │    │
│  │ 320px   │  │ (capped at 1440 + centered ≥1600)  │    │
│  │         │  │                                    │    │
│  │ brand   │  │ Ready | Recording | Processing |   │    │
│  │ theme   │  │ SessionView                        │    │
│  │ user    │  │                                    │    │
│  │ list... │  │                                    │    │
│  └─────────┘  └────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### Rules

- Sidebar is **exactly 320px** on desktop. Don't make it resizable; don't make it 280 or 360 ad hoc.
- Main pane is **state-driven**, not stacked. One of `Ready`, `Recording`, `Processing`, `SessionView` at a time.
- Recorder/upload lives in `Ready`, never in the sidebar.
- The sidebar's "+ New Session" button is **hidden** when `mainView === "ready"` (redundant with the big record button).
- Sidebar contents (top → bottom): brand row + theme toggle + sign-out · user identity · "+ New Session" button (conditional) · scrollable session list.

### Mobile (<900px)

- Sidebar → `Drawer` triggered by an `AppBar` hamburger.
- Drawer width: 320px (same as desktop). Design language stays consistent across breakpoints.
- AppBar sticky to top: brand on the left, menu trigger on the right.

## Session view structure (inside SessionView)

Content ordering:
1. **TitleInput** at the top with `variant="title"` (h5, prominent, editable). The session's title is its identity — top of the surface, not buried in any card header.
2. Meta row: duration · date · "Just completed" chip if applicable.
3. SummaryBlock (purple gradient accent strip, AutoAwesome icon).
4. TranscriptPanel (no title in header — just an "Transcript" overline + Copy button).
5. ChatPanel (no title in header — just "Ask about this transcript").

Title used to live in TranscriptPanel/ChatPanel headers. It's been removed because it made cards feel disconnected.

## Section primitive

For surfaces that are *not* the SessionView (settings, account, public session, future dashboards), use the `Section` primitive in [patterns/section.tsx](patterns/section.tsx):

```tsx
<Section
  title="Account"
  description="Your profile and security settings."
>
  <Row label="Email" value={user.email} />
  <Row label="Password" action={<Button>Change</Button>} />
</Section>
```

A page is then just a `<Stack spacing={4}>` of Sections inside a `ShellContainer`.

## Container (the 1440 cap)

On screens ≥1600px, main pane content **caps at 1440 and centers**. Use [patterns/shell-container.tsx](patterns/shell-container.tsx):

```tsx
<ShellContainer>
  {/* your page content */}
</ShellContainer>
```

Sidebar stays 320 and pinned left. The cap applies only to the main pane.

Per-surface narrower caps (transcript = 720, settings = 720, public session = 800) go **inside** ShellContainer, not as a replacement.

| Surface | Inner cap | Why |
|---|---|---|
| Transcript text | 720 | Reading comfort. |
| Chat message column | 720 | Reading comfort. |
| Summary card | Fills main pane (≤1440) | Glanceable, not for reading. |
| Settings sections | 720 | Forms read better narrow. |
| Public session page | 800 | Marketing-grade, single column. |
| Sidebar | 320 fixed | Identity. |

If a surface caps narrower than the main pane, **center it inside the main pane** — don't left-align.

## Spacing scale

`theme.spacing(1)` = 8px. Use only these values:

| Value | Use |
|---|---|
| `0.5` (4px) | Tight icon-text gaps |
| `1` (8px) | Element gaps in a row |
| `1.5` (12px) | Compact stack |
| `2` (16px) | Card body padding |
| `3` (24px) | Section internal spacing |
| `4` (32px) | Between sections |
| `5+` | Marketing / hero only |

No half-step values like `padding: "13px"`. Stay on the scale.

## When the user asks for a new top-level surface

1. Does it belong in the app shell (sidebar + main pane) or stand alone (public session, marketing)?
   - Standalone: see [future/public-session.md](future/public-session.md).
2. If in the app shell: add it as a new `mainView` value OR as a new route segment (e.g. `/settings`).
3. Wrap content in `<ShellContainer>`.
4. Compose with `Section` primitives.
5. Run through [responsive.md](responsive.md) checklist at 360 / 768 / 1280 / 1920.

## When the user asks for "more density"

Audia is content-forward, not dashboard-dense. Before reducing padding:
- Try `dense` props on List/MenuItem/Toolbar first.
- Try `size="small"` on TextField/Button/Chip.
- Try collapsing meta rows into a single line with `·` separators.

Only drop card padding (`p: 2` → `p: 1.5`) as a last resort, and never below `p: 1`.
