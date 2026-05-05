/**
 * Audia brand constants + favicon SVG generator.
 *
 * Server-safe (no React, no hooks) so it can be imported by app/icon.tsx
 * without forcing client-bundle inclusion. The matching React component
 * <BrandLogo /> imports from here.
 */

export const BRAND = {
    primary: "#5b21b6",
    secondary: "#4f46e5",
    inner: "#ffffff",
    /** All values below are on a 32-unit canvas. */
    canvas: 32,
    /** Half the canvas → perfect circle. Drop to ~8 for a rounded square. */
    cornerRadius: 16,
    innerCircleRadius: 5,
} as const;

/**
 * Favicon SVG string. The `<linearGradient id="g">` is fine here because the
 * favicon is its own document — no ID collision with the page's BrandLogo.
 */
export function brandLogoSvgString(): string {
    return [
        `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${BRAND.canvas} ${BRAND.canvas}">`,
        `<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">`,
        `<stop offset="0%" stop-color="${BRAND.primary}"/>`,
        `<stop offset="100%" stop-color="${BRAND.secondary}"/>`,
        `</linearGradient></defs>`,
        `<rect width="${BRAND.canvas}" height="${BRAND.canvas}" rx="${BRAND.cornerRadius}" fill="url(#g)"/>`,
        `<circle cx="${BRAND.canvas / 2}" cy="${BRAND.canvas / 2}" r="${BRAND.innerCircleRadius}" fill="${BRAND.inner}"/>`,
        `</svg>`,
    ].join("");
}
