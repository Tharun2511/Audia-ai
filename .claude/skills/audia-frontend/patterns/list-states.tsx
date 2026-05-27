import type { ReactNode } from "react";
import Skeleton from "@mui/material/Skeleton";
import Stack from "@mui/material/Stack";
import EmptyState from "./empty-state";

interface ListStatesProps<T> {
    data: T[] | undefined;
    loading: boolean;
    error?: Error | null;
    emptyTitle: string;
    emptyDescription?: string;
    emptyAction?: ReactNode;
    skeletonCount?: number;
    skeletonHeight?: number;
    renderRow: (item: T, index: number) => ReactNode;
}

/**
 * Wraps a list-bearing surface with Loading / Empty / Error states.
 *
 * Renders, in order of precedence:
 *  1. Error → throws, so the nearest error.tsx boundary picks it up.
 *  2. Loading → N skeletons matching row height.
 *  3. Empty → EmptyState primitive with the given title/description/action.
 *  4. Loaded → maps over data with renderRow.
 *
 * Use anywhere you fetch a collection client-side. Server Components should
 * handle this inline since they have direct data access — this wrapper exists
 * for the client-fetch case (e.g. paginated lists, search results).
 */
export default function ListStates<T>({
    data,
    loading,
    error,
    emptyTitle,
    emptyDescription,
    emptyAction,
    skeletonCount = 3,
    skeletonHeight = 72,
    renderRow,
}: ListStatesProps<T>) {
    if (error) {
        // Let the route-level error.tsx boundary handle it.
        throw error;
    }

    if (loading) {
        return (
            <Stack spacing={1.5}>
                {Array.from({ length: skeletonCount }, (_, i) => (
                    <Skeleton
                        key={i}
                        variant="rectangular"
                        height={skeletonHeight}
                        sx={{ borderRadius: 2 }}
                    />
                ))}
            </Stack>
        );
    }

    if (!data || data.length === 0) {
        return (
            <EmptyState
                title={emptyTitle}
                description={emptyDescription}
                action={emptyAction}
            />
        );
    }

    return <Stack spacing={1}>{data.map(renderRow)}</Stack>;
}
