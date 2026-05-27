# Motion — when, how, how much

Motion in Audia is **assistive**, not decorative. Every animation should answer "what state just changed?" or "where did this come from?" If you can't answer, it doesn't ship.

## Defaults

- **Library**: none. Use MUI built-in transitions (`Fade`, `Grow`, `Collapse`, `Slide`, `Zoom`) and CSS keyframes in `globals.css`.
- **No Framer Motion / motion / gsap** unless a task genuinely needs gesture (drag, swipe) or spring physics. Audia doesn't today.
- **Durations**:
  - 150ms — micro (button press, hover)
  - 250ms — standard (Fade in card, Collapse list)
  - 400ms — page-level enter
  - Anything longer feels sluggish.
- **Easings**: use the theme's `transitions.easing`:
  - `easeInOut` — symmetric (toggles, fade in/out).
  - `easeOut` — enter (Slide, Grow on appearance).
  - `easeIn` — exit (rare; usually skip).
  - `sharp` — sliders, tap responses.

## Existing keyframes (globals.css)

Reuse, don't duplicate:

| Keyframe | Use |
|---|---|
| `pulse-ring` | Expanding ring around the record button during recording. |
| `rec-blink` | Blinking red dot for recording state. |
| `bar-wave` | Equalizer-style bars for audio activity / loading. |
| `fade-in` | Generic "this just appeared" — bound to class `.animate-fade-in`. |

Adding a new keyframe? Justify it. Most needs are met by `Fade` / `Grow` / `Collapse`.

## Patterns

### Card appears (e.g. summary loaded)

```tsx
<Fade in={!!summary} timeout={250}>
  <div><SummaryBlock summary={summary} /></div>
</Fade>
```

Wrap in a `<div>` because `Fade` clones the child and needs a real DOM element.

### List item slides in

```tsx
<Collapse in={mounted} timeout={300}>
  <SessionListItem />
</Collapse>
```

### Page transitions

Don't animate full-page transitions. Next.js navigation should feel instant. Animate the *content* (skeleton → loaded) instead.

### Loading skeleton

```tsx
<Skeleton variant="rectangular" height={120} sx={{ borderRadius: 2 }} />
```

- Match the final element's `borderRadius` (use `2` for cards = 16px, `1` for inputs).
- 2–4 skeletons per visible region, not a screenful.
- Stop the skeleton as soon as data arrives — never show longer than 300ms after data is ready.

### Streaming text (live transcription, chat tokens)

See [future/live-transcription.md](future/live-transcription.md) for the full pattern. Brief: append text + apply `.animate-fade-in` to each new chunk.

### Hover / focus

- Buttons: built-in hover handled by MUI. Don't override unless adding the brand-gradient glow on the primary CTA.
- Cards: no hover effect by default. If clickable (session list item), wrap in `ButtonBase` and let it handle ripple + focus ring.

### The brand gradient glow (single allowed flourish)

For the primary CTA on hero surfaces:

```tsx
<Button
  variant="contained"
  sx={{
    position: "relative",
    "&:hover::after": {
      content: '""',
      position: "absolute",
      inset: -2,
      borderRadius: "inherit",
      background: "linear-gradient(to right, var(--mui-palette-primary-dark), var(--mui-palette-primary-main), var(--mui-palette-secondary-main))",
      opacity: 0.35,
      filter: "blur(8px)",
      zIndex: -1,
      transition: "opacity 200ms ease",
    },
  }}
>
  Start recording
</Button>
```

Use sparingly — once per surface max.

## Reduced motion

Respect `prefers-reduced-motion`. MUI transitions already honor this when you use `Fade` / `Grow` / etc. For custom CSS keyframes, always add:

```css
@media (prefers-reduced-motion: reduce) {
  .animate-fade-in, .pulse-ring, .rec-blink, .bar-wave {
    animation: none !important;
  }
}
```

Add to the global rule whenever you add a new keyframe. **Do not** ship a keyframe without a reduced-motion fallback.

## The 7-second rule

If a process takes > 7 seconds (uploading, transcribing, summarizing), motion isn't enough — show **progress**:
- Determinate `LinearProgress` if you have a percentage.
- Stepper (Upload → Transcribe → Summarize) if you have stages.
- Otherwise indeterminate bar **plus** descriptive copy ("Transcribing 12 minutes of audio…").

Never show only a spinner for > 7 seconds.

## Forbidden motion

- **Page-load animations.** Content appears instantly when SSR'd. Don't `Fade in` the whole page.
- **Bouncy / spring physics** on UI controls. iOS-soft means smooth, not bouncy.
- **Stagger** on more than 5 items (theatrical, slow).
- **Parallax** on scroll. Premier doesn't mean theme park.
- **Auto-playing motion** in the marketing/public surface unless tied to user interaction.
- **Looping animations** that aren't tied to an active state. Recording = looping is fine (state is "active"). Idle UI loops are not.

## Decision tree

1. Is this state change visible without motion? → No motion needed.
2. Is the state change ambiguous (could the user miss it)? → 250ms Fade or Collapse.
3. Is this a route-level change? → No motion; rely on instant navigation + content skeleton.
4. Is the user waiting? → Determinate progress + copy, not just a spinner.
5. Is this gesture-driven (drag, swipe)? → Now you may need a motion library. Stop and discuss with the user.
