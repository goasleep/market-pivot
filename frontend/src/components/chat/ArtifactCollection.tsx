import { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import {
  ChevronDown,
  ChevronRight,
  Download,
  FileArchive,
  Folder,
  FolderOpen,
  X,
} from "lucide-react";
import type { Artifact } from "@/types";
import { cn } from "@/lib/utils";
import { assetTypeLabel } from "@/lib/assets";
import {
  ArtifactCard,
  ArtifactPreview,
  artifactIcon,
  formatArtifactSize,
  formatArtifactType,
} from "./ArtifactCard";

interface ArtifactCollectionProps {
  artifacts: Artifact[];
}

interface ArtifactGroup {
  id: string;
  label: string;
  artifacts: Artifact[];
}

function inferredTicker(artifact: Artifact) {
  if (artifact.ticker) return artifact.ticker;
  return artifact.name.match(/(?:^|[^0-9])(\d{6})(?:[^0-9]|$)/)?.[1] || "";
}

export function groupArtifacts(artifacts: Artifact[]): ArtifactGroup[] {
  const groups = new Map<string, ArtifactGroup>();
  for (const artifact of artifacts) {
    const ticker = inferredTicker(artifact);
    const id = ticker || "other";
    const assetType = artifact.asset_type
      ? assetTypeLabel(artifact.asset_type)
      : undefined;
    const group = groups.get(id) || {
      id,
      label: ticker
        ? `${ticker}${assetType ? ` · ${assetType}` : ""}`
        : "其他文件",
      artifacts: [],
    };
    group.artifacts.push(artifact);
    groups.set(id, group);
  }
  return [...groups.values()];
}

function totalSize(artifacts: Artifact[]) {
  return artifacts.reduce(
    (total, artifact) => total + Math.max(0, artifact.size_bytes || 0),
    0,
  );
}

function ArtifactFolder({
  artifacts,
  onOpen,
}: {
  artifacts: Artifact[];
  onOpen: () => void;
}) {
  const groups = groupArtifacts(artifacts);
  return (
    <button
      type="button"
      onClick={onOpen}
      className="group flex w-full max-w-md items-center gap-3 rounded-xl border bg-background p-3 text-left shadow-sm transition-colors hover:border-primary/35 hover:bg-primary/[0.025] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
    >
      <span className="relative flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-amber-50 text-amber-600 dark:bg-amber-950/30 dark:text-amber-400">
        <Folder className="h-7 w-7" strokeWidth={1.6} />
        <span className="absolute -bottom-1 -right-1 rounded-full border-2 border-background bg-foreground px-1.5 py-0.5 text-[9px] font-semibold leading-none text-background">
          {artifacts.length}
        </span>
      </span>
      <span className="min-w-0 flex-1">
        <span className="block text-sm font-semibold">全部文件</span>
        <span className="mt-0.5 block text-xs text-muted-foreground">
          {groups.length} 个分组 · {formatArtifactSize(totalSize(artifacts))}
        </span>
      </span>
      <span className="rounded-md border bg-background px-2.5 py-1.5 text-xs font-medium text-foreground transition-colors group-hover:border-primary/30 group-hover:text-primary">
        浏览文件
      </span>
    </button>
  );
}

function FileTree({
  groups,
  selectedId,
  onSelect,
}: {
  groups: ArtifactGroup[];
  selectedId: string;
  onSelect: (artifact: Artifact) => void;
}) {
  const [collapsed, setCollapsed] = useState<Set<string>>(() => new Set());

  const toggleGroup = (groupId: string) => {
    setCollapsed((current) => {
      const next = new Set(current);
      if (next.has(groupId)) next.delete(groupId);
      else next.add(groupId);
      return next;
    });
  };

  return (
    <div className="space-y-1 p-3">
      {groups.map((group) => {
        const isCollapsed = collapsed.has(group.id);
        return (
          <div key={group.id}>
            <button
              type="button"
              onClick={() => toggleGroup(group.id)}
              className="flex w-full items-center gap-2 rounded-lg px-2 py-2 text-left text-sm font-medium hover:bg-muted"
              aria-expanded={!isCollapsed}
            >
              {isCollapsed ? (
                <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
              ) : (
                <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
              )}
              {isCollapsed ? (
                <Folder className="h-4 w-4 text-amber-500" />
              ) : (
                <FolderOpen className="h-4 w-4 text-amber-500" />
              )}
              <span className="min-w-0 flex-1 truncate">{group.label}</span>
              <span className="text-[11px] tabular-nums text-muted-foreground">
                {group.artifacts.length}
              </span>
            </button>
            {!isCollapsed && (
              <div className="ml-[1.15rem] border-l border-border/80 pl-2">
                {group.artifacts.map((artifact) => {
                  const selected = artifact.artifact_id === selectedId;
                  return (
                    <button
                      key={artifact.artifact_id}
                      type="button"
                      onClick={() => onSelect(artifact)}
                      className={cn(
                        "my-0.5 flex w-full items-center gap-2 rounded-lg px-2 py-2 text-left transition-colors",
                        selected
                          ? "bg-primary/10 text-primary"
                          : "text-foreground hover:bg-muted",
                      )}
                    >
                      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-background shadow-sm ring-1 ring-border/70">
                        {artifactIcon(artifact.mime_type, "h-4 w-4")}
                      </span>
                      <span className="min-w-0 flex-1">
                        <span
                          className="block truncate text-xs font-medium"
                          title={artifact.name}
                        >
                          {artifact.name}
                        </span>
                        <span className="mt-0.5 block text-[10px] text-muted-foreground">
                          {formatArtifactType(artifact.mime_type)} ·{" "}
                          {formatArtifactSize(artifact.size_bytes)}
                        </span>
                      </span>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function ArtifactBrowser({
  artifacts,
  onClose,
}: {
  artifacts: Artifact[];
  onClose: () => void;
}) {
  const groups = useMemo(() => groupArtifacts(artifacts), [artifacts]);
  const [selected, setSelected] = useState(artifacts[0]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  if (!selected) return null;

  return createPortal(
    <div className="fixed inset-0 z-[70]">
      <button
        type="button"
        aria-label="关闭全部文件"
        className="absolute inset-0 bg-slate-950/30 backdrop-blur-[1px]"
        onClick={onClose}
      />
      <aside className="absolute inset-y-0 right-0 flex w-[min(96vw,1120px)] flex-col border-l bg-background shadow-2xl animate-in slide-in-from-right duration-200">
        <header className="flex h-16 shrink-0 items-center gap-3 border-b px-4">
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-amber-50 text-amber-600 dark:bg-amber-950/30 dark:text-amber-400">
            <FileArchive className="h-5 w-5" />
          </span>
          <div className="min-w-0 flex-1">
            <div className="text-sm font-semibold">全部文件</div>
            <div className="text-xs text-muted-foreground">
              {artifacts.length} 个文件 · {groups.length} 个分组 ·{" "}
              {formatArtifactSize(totalSize(artifacts))}
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-2 text-muted-foreground hover:bg-muted hover:text-foreground"
            aria-label="关闭文件浏览器"
            title="关闭"
          >
            <X className="h-5 w-5" />
          </button>
        </header>
        <div className="grid min-h-0 flex-1 md:grid-cols-[320px_minmax(0,1fr)]">
          <nav className="min-h-0 overflow-y-auto border-b bg-muted/15 md:border-b-0 md:border-r">
            <div className="sticky top-0 z-10 border-b bg-background/90 px-4 py-2.5 text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground backdrop-blur">
              文件树
            </div>
            <FileTree
              groups={groups}
              selectedId={selected.artifact_id}
              onSelect={setSelected}
            />
          </nav>
          <section className="flex min-h-0 min-w-0 flex-col">
            <div className="flex min-h-14 shrink-0 items-center gap-3 border-b px-4 py-2">
              <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-muted">
                {artifactIcon(selected.mime_type, "h-4 w-4")}
              </span>
              <div className="min-w-0 flex-1">
                <div
                  className="truncate text-sm font-medium"
                  title={selected.name}
                >
                  {selected.name}
                </div>
                <div className="text-[11px] text-muted-foreground">
                  {formatArtifactType(selected.mime_type)} ·{" "}
                  {formatArtifactSize(selected.size_bytes)}
                </div>
              </div>
              <a
                href={selected.download_url}
                download
                className="inline-flex items-center gap-1.5 rounded-lg border bg-background px-3 py-2 text-xs font-medium hover:border-primary/30 hover:text-primary"
              >
                <Download className="h-3.5 w-3.5" />
                下载
              </a>
            </div>
            <div className="min-h-0 flex-1 bg-white">
              <ArtifactPreview artifact={selected} />
            </div>
          </section>
        </div>
      </aside>
    </div>,
    document.body,
  );
}

export function ArtifactCollection({ artifacts }: ArtifactCollectionProps) {
  const [browserOpen, setBrowserOpen] = useState(false);
  const htmlArtifacts = artifacts.filter(
    (artifact) => artifact.mime_type === "text/html",
  );
  const otherArtifacts = artifacts.filter(
    (artifact) => artifact.mime_type !== "text/html",
  );

  if (artifacts.length === 1) {
    return <ArtifactCard artifact={artifacts[0]} />;
  }

  return (
    <div className="w-full space-y-2">
      {htmlArtifacts.map((artifact) => (
        <ArtifactCard key={artifact.artifact_id} artifact={artifact} />
      ))}
      {otherArtifacts.length > 0 && (
        <>
          <ArtifactFolder
            artifacts={otherArtifacts}
            onOpen={() => setBrowserOpen(true)}
          />
          {browserOpen && (
            <ArtifactBrowser
              artifacts={otherArtifacts}
              onClose={() => setBrowserOpen(false)}
            />
          )}
        </>
      )}
    </div>
  );
}
