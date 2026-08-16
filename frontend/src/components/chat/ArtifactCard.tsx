import { useState } from "react";
import { createPortal } from "react-dom";
import {
  Download,
  ExternalLink,
  FileCode2,
  FileText,
  Image,
} from "lucide-react";
import type { Artifact } from "@/types";

interface ArtifactCardProps {
  artifact: Artifact;
}

function iconFor(mimeType: string) {
  if (mimeType === "text/html")
    return <FileCode2 className="h-7 w-7 text-sky-600" />;
  if (mimeType.startsWith("image/"))
    return <Image className="h-7 w-7 text-violet-600" />;
  return <FileText className="h-7 w-7 text-blue-600" />;
}

function formatSize(size: number) {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}

export function ArtifactCard({ artifact }: ArtifactCardProps) {
  const [previewOpen, setPreviewOpen] = useState(false);

  return (
    <>
      <div className="flex w-full max-w-md items-center gap-3 rounded-xl border bg-background p-3 shadow-sm">
        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-muted">
          {iconFor(artifact.mime_type)}
        </div>
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-semibold" title={artifact.name}>
            {artifact.name}
          </div>
          <div className="mt-0.5 text-xs text-muted-foreground">
            {artifact.mime_type === "text/html"
              ? "HTML"
              : artifact.mime_type === "text/markdown"
                ? "Markdown"
                : artifact.mime_type}
            {" · "}
            {formatSize(artifact.size_bytes)}
          </div>
          <div className="mt-2 flex items-center gap-3 text-xs">
            <button
              type="button"
              className="inline-flex items-center gap-1 text-primary hover:underline"
              onClick={() => setPreviewOpen(true)}
            >
              <ExternalLink className="h-3.5 w-3.5" />
              预览
            </button>
            <a
              className="inline-flex items-center gap-1 text-primary hover:underline"
              href={artifact.download_url}
              download
            >
              <Download className="h-3.5 w-3.5" />
              下载
            </a>
          </div>
        </div>
      </div>
      {previewOpen &&
        typeof document !== "undefined" &&
        createPortal(
          <div className="fixed inset-0 z-[70]">
            <button
              type="button"
              aria-label="关闭预览"
              className="absolute inset-0 bg-slate-950/25"
              onClick={() => setPreviewOpen(false)}
            />
            <aside className="absolute inset-y-0 right-0 flex w-[min(92vw,920px)] flex-col border-l bg-background shadow-2xl">
              <header className="flex h-14 shrink-0 items-center gap-3 border-b px-4">
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-muted">
                  {iconFor(artifact.mime_type)}
                </div>
                <div className="min-w-0 flex-1">
                  <div
                    className="truncate text-sm font-semibold"
                    title={artifact.name}
                  >
                    {artifact.name}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {artifact.mime_type === "text/html"
                      ? "HTML"
                      : artifact.mime_type === "text/markdown"
                        ? "Markdown"
                        : artifact.mime_type}
                  </div>
                </div>
                <a
                  className="rounded-md p-2 text-muted-foreground hover:bg-muted hover:text-foreground"
                  href={artifact.download_url}
                  download
                  aria-label="下载产物"
                  title="下载"
                >
                  <Download className="h-4 w-4" />
                </a>
                <button
                  type="button"
                  className="rounded-md p-2 text-muted-foreground hover:bg-muted hover:text-foreground"
                  onClick={() => setPreviewOpen(false)}
                  aria-label="关闭预览"
                  title="关闭"
                >
                  <span className="text-xl leading-none">×</span>
                </button>
              </header>
              <div className="min-h-0 flex-1 bg-white">
                <iframe
                  title={`预览 ${artifact.name}`}
                  src={artifact.preview_url}
                  className="h-full w-full border-0"
                  sandbox="allow-scripts"
                />
              </div>
            </aside>
          </div>,
          document.body,
        )}
    </>
  );
}
