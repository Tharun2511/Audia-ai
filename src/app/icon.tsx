import { brandLogoSvgString, BRAND } from "@/app/components/BrandLogo";

export const contentType = "image/svg+xml";
export const size = { width: BRAND.canvas, height: BRAND.canvas };

/**
 * Favicon route. Returns the brand mark as inline SVG, sourced from the same
 * constants as <BrandLogo />. Update src/app/components/BrandLogo.tsx to
 * change either.
 */
export default function Icon(): Response {
    return new Response(brandLogoSvgString(), {
        headers: {
            "Content-Type": "image/svg+xml",
            // Cache for an hour at the CDN, revalidate forever in browsers
            // until the URL changes (which it won't unless we add a query param).
            "Cache-Control": "public, max-age=3600, immutable",
        },
    });
}
