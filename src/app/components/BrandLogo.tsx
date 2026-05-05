/**
 * Audia brand mark — single source of truth.
 *
 * Used by:
 *   - <BrandLogo />          (React component, rendered inside pages)
 *   - app/icon.tsx           (favicon route, serves the same SVG to browsers)
 *
 * Change values here → both surfaces update together.
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

interface BrandLogoProps {
    /** Rendered pixel size. Internal proportions are preserved via viewBox. */
    size?: number;
    /** Optional accessible label. Pass empty string to mark as decorative. */
    title?: string;
}

export function BrandLogo({ size = 32, title = "Audia" }: BrandLogoProps) {
    return (
        <svg
            xmlns="http://www.w3.org/2000/svg"
            width={size}
            height={size}
            viewBox={`0 0 ${BRAND.canvas} ${BRAND.canvas}`}
            role={title ? "img" : "presentation"}
            aria-label={title || undefined}
            aria-hidden={title ? undefined : true}
        >
            <defs>
                <linearGradient id="audia-brand-grad" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stopColor={BRAND.primary} />
                    <stop offset="100%" stopColor={BRAND.secondary} />
                </linearGradient>
            </defs>
            <rect
                width={BRAND.canvas}
                height={BRAND.canvas}
                rx={BRAND.cornerRadius}
                fill="url(#audia-brand-grad)"
            />
            <circle
                cx={BRAND.canvas / 2}
                cy={BRAND.canvas / 2}
                r={BRAND.innerCircleRadius}
                fill={BRAND.inner}
            />
        </svg>
    );
}

/**
 * Returns the brand mark as an SVG string. Used by the favicon route handler,
 * which can't render React. Mirrors <BrandLogo /> exactly.
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
