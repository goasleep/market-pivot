import { lazy, Suspense } from "react";
import { Folder, Loader2 } from "lucide-react";
import type { Artifact } from "@/types";

const ArtifactCollection = lazy(() =>
  import("./ArtifactCollection").then((module) => ({
    default: module.ArtifactCollection,
  })),
);

export function LazyArtifactCollection({
  artifacts,
}: {
  artifacts: Artifact[];
}) {
  return (
    <Suspense
      fallback={
        <div className="flex min-h-24 max-w-md items-center justify-center gap-3 rounded-xl border border-dashed bg-amber-50/30 p-4 text-sm text-muted-foreground">
          <span className="relative flex h-9 w-9 items-center justify-center rounded-xl bg-white text-amber-600 shadow-sm">
            <Folder className="h-4 w-4" />
            <Loader2 className="absolute -right-1 -top-1 h-3.5 w-3.5 animate-spin" />
          </span>
          正在整理交付文件…
        </div>
      }
    >
      <ArtifactCollection artifacts={artifacts} />
    </Suspense>
  );
}
