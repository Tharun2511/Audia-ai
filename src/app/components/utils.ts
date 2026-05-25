/**
 * 10 visually-distinct speaker colors with strong perceptual separation.
 * Each is paired by hue family — primary, then accent — so that even if a
 * meeting has all 10 represented, adjacent speakers don't blend.
 *
 * Beyond 10 speakers we cycle back to the start. The visual collision is
 * acceptable because 10-speaker meetings are vanishingly rare and the
 * speaker name labels keep the bubbles distinguishable.
 */
export const SPEAKER_COLORS = [
  "#6d28d9", // violet
  "#0369a1", // sky
  "#047857", // emerald
  "#b45309", // amber
  "#9d174d", // pink
  "#1d4ed8", // indigo
  "#15803d", // green
  "#c2410c", // orange
  "#7e22ce", // purple
  "#0891b2", // cyan
];

export function buildColorMap(speakers: string[]): Record<string, string> {
  return Object.fromEntries(
    speakers.map((sp, i) => [sp, SPEAKER_COLORS[i % SPEAKER_COLORS.length]])
  );
}

export function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}
