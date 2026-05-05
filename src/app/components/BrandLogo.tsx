"use client";
import { useId } from "react";
import { BRAND } from "./brand";

/**
 * Audia brand mark — React component.
 *
 * Two reasons this is a client component:
 *   1. useId() guarantees a unique gradient ID per instance, so multiple
 *      BrandLogos on the same page (e.g. desktop sidebar + mobile app bar)
 *      don't share an ID and break each other's `url(#...)` references.
 *   2. The dark-mode white border is rendered as a normal SVG element with a
 *      class. globals.css shows it only when an ancestor has the
 *      `.mui-color-scheme-dark` class. No JS color-scheme guard, no hydration
 *      flash, no theme-prop drilling.
 *
 * Constants and the favicon string generator live in ./brand.ts (server-safe).
 */

interface BrandLogoProps {
    /** Rendered pixel size. Internal proportions are preserved via viewBox. */
    size?: number;
    /** Optional accessible label. Pass empty string to mark as decorative. */
    title?: string;
}

export function BrandLogo({ size = 32, title = "Audia" }: BrandLogoProps) {
    // useId() returns ":r1:" — sanitize for the SVG id/url(#...) reference.
    const reactId = useId().replace(/[^a-zA-Z0-9_-]/g, "");
    const gradId = `audia-grad-${reactId}`;

    return (
        <svg
            xmlns="http://www.w3.org/2000/svg"
            width={size}
            height={size}
            viewBox={`0 0 ${BRAND.canvas} ${BRAND.canvas}`}
            role={title ? "img" : "presentation"}
            aria-label={title || undefined}
            aria-hidden={title ? undefined : true}
            className="brand-logo-svg"
        >
            <defs>
                <linearGradient id={gradId} x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stopColor={BRAND.primary} />
                    <stop offset="100%" stopColor={BRAND.secondary} />
                </linearGradient>
            </defs>
            <rect
                width={BRAND.canvas}
                height={BRAND.canvas}
                rx={BRAND.cornerRadius}
                fill={`url(#${gradId})`}
            />
            {/* Dark-mode-only white border ring. Hidden by default in CSS. */}
            <circle
                className="brand-logo-dark-ring"
                cx={BRAND.canvas / 2}
                cy={BRAND.canvas / 2}
                r={BRAND.canvas / 2 - 1}
                fill="none"
                stroke="#ffffff"
                strokeWidth={1.5}
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

// Re-export so existing imports `import { BrandLogo, brandLogoSvgString } from "./BrandLogo"` still work.
export { BRAND, brandLogoSvgString } from "./brand";
