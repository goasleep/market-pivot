import { lazy, Suspense } from "react";
import { FileText, Loader2 } from "lucide-react";
import type { Artifact } from "@/types";

const ArtifactCard = lazy(() =>
  import("./ArtifactCard").then((module) => ({ default: module.ArtifactCard })),
);

export function LazyArtifactCard({ artifact }: { artifact: Artifact }) {
  return (
    <Suspense
      fallback={
        <div className="flex min-h-24 items-center justify-center gap-3 rounded-xl border border-dashed bg-blue-50/30 p-4 text-sm text-muted-foreground">
          <span className="relative flex h-9 w-9 items-center justify-center rounded-xl bg-white text-primary shadow-sm">
            <FileText className="h-4 w-4" />
            <Loader2 className="absolute -right-1 -top-1 h-3.5 w-3.5 animate-spin" />
          </span>
          正在加载富预览…
        </div>
      }
    >
      <ArtifactCard artifact={artifact} />
    </Suspense>
  );
}
