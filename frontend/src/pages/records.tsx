import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Archive, CalendarDays, ExternalLink, FileText, Layers3, RefreshCw, Search } from "lucide-react";
import { getArtifacts } from "@/api";
import type { Artifact } from "@/types";
import { ArtifactCard } from "@/components/chat/ArtifactCard";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

function formatDate(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "时间未知";
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function reportTitle(artifact: Artifact) {
  return artifact.name.replace(/\.html?$/i, "");
}

function reportSource(source?: string) {
  if (source === "chat") return "Chat 分析";
  if (source === "analysis") return "Agent Analysis";
  return source || "Agent 生成";
}

type ArtifactCategory = {
  key: string;
  label: string;
};

const mimeLabels: Record<string, string> = {
  "text/html": "HTML",
  "text/markdown": "Markdown",
  "text/csv": "CSV",
  "text/plain": "TXT",
  "application/json": "JSON",
  "application/pdf": "PDF",
  "application/xml": "XML",
  "application/yaml": "YAML",
};

function formatForArtifact(artifact: Artifact) {
  const metadataFormat = artifact.metadata?.format;
  if (typeof metadataFormat === "string" && metadataFormat.trim()) {
    const format = metadataFormat.trim().toLowerCase().replace(/^\./, "");
    return format === "md" ? "Markdown" : format.toUpperCase();
  }
  if (mimeLabels[artifact.mime_type]) return mimeLabels[artifact.mime_type];
  if (artifact.mime_type.startsWith("image/")) return "图片";
  if (artifact.mime_type.startsWith("video/")) return "视频";
  if (artifact.mime_type.startsWith("audio/")) return "音频";
  const extension = artifact.name.split(".").pop()?.trim();
  return extension ? extension.toUpperCase() : "其他";
}

function categoryForArtifact(artifact: Artifact): ArtifactCategory {
  const label = formatForArtifact(artifact);
  return { key: label.toLowerCase(), label };
}

export function RecordsPage() {
  const navigate = useNavigate();
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [query, setQuery] = useState("");
  const [activeCategory, setActiveCategory] = useState("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadArtifacts = async () => {
    setLoading(true);
    setError(null);
    try {
      setArtifacts(await getArtifacts());
    } catch (err) {
      setError(err instanceof Error ? err.message : "产物加载失败");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadArtifacts();
  }, []);

  const searchedArtifacts = useMemo(() => {
    const text = query.trim().toLowerCase();
    if (!text) return artifacts;
    return artifacts.filter((artifact) =>
      `${artifact.name} ${artifact.ticker || ""} ${artifact.asset_type || ""} ${artifact.source || ""} ${categoryForArtifact(artifact).label}`
        .toLowerCase()
        .includes(text),
    );
  }, [artifacts, query]);

  const categories = useMemo(() => {
    const available = new Map<string, ArtifactCategory>();
    searchedArtifacts.forEach((artifact) => {
      const category = categoryForArtifact(artifact);
      available.set(category.key, category);
    });
    return Array.from(available.values());
  }, [searchedArtifacts]);

  const categoryCounts = useMemo(() => {
    const counts = new Map<string, number>();
    searchedArtifacts.forEach((artifact) => {
      const category = categoryForArtifact(artifact);
      counts.set(category.key, (counts.get(category.key) || 0) + 1);
    });
    return counts;
  }, [searchedArtifacts]);

  const filteredArtifacts = useMemo(
    () =>
      activeCategory === "all"
        ? searchedArtifacts
        : searchedArtifacts.filter((artifact) => categoryForArtifact(artifact).key === activeCategory),
    [activeCategory, searchedArtifacts],
  );

  return (
    <div className="space-y-6 p-6">
      <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
        <div>
          <div className="eyebrow">Research library</div>
          <h1 className="mt-2 text-2xl font-bold">产物库</h1>
          <p className="mt-1 text-sm text-muted-foreground">按文件格式整理生成产物，统一预览、下载和追溯来源。</p>
        </div>
        <Button variant="outline" onClick={() => void loadArtifacts()} disabled={loading}>
          <RefreshCw className={`mr-2 h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          刷新产物
        </Button>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <Card className="border-violet-100 bg-gradient-to-br from-white to-violet-50/70">
          <CardContent className="flex items-center gap-4 p-5">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-violet-100 text-violet-600"><FileText className="h-5 w-5" /></div>
            <div><p className="text-sm text-muted-foreground">产物总数</p><p className="text-2xl font-semibold">{artifacts.length}</p></div>
          </CardContent>
        </Card>
        <Card className="border-blue-100 bg-gradient-to-br from-white to-blue-50/70">
          <CardContent className="flex items-center gap-4 p-5">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-blue-100 text-blue-600"><Archive className="h-5 w-5" /></div>
            <div><p className="text-sm text-muted-foreground">当前展示</p><p className="text-2xl font-semibold">{filteredArtifacts.length}</p></div>
          </CardContent>
        </Card>
        <Card className="border-cyan-100 bg-gradient-to-br from-white to-cyan-50/70">
          <CardContent className="flex items-center gap-4 p-5">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-cyan-100 text-cyan-600"><CalendarDays className="h-5 w-5" /></div>
            <div className="min-w-0"><p className="text-sm text-muted-foreground">最近生成</p><p className="truncate text-sm font-semibold">{artifacts[0] ? formatDate(artifacts[0].created_at) : "暂无产物"}</p></div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardContent className="p-4">
          <div className="relative">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input className="pl-9" placeholder="搜索产物名称、标的代码、格式或来源" value={query} onChange={(event) => setQuery(event.target.value)} />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="flex flex-col gap-3 p-4 sm:flex-row sm:items-center">
          <div className="flex shrink-0 items-center gap-2 text-sm font-semibold text-foreground">
            <Layers3 className="h-4 w-4 text-primary" />
            产物分类
          </div>
          <Tabs value={activeCategory} onValueChange={setActiveCategory} className="min-w-0 flex-1">
            <TabsList className="h-auto w-full justify-start gap-1 overflow-x-auto bg-muted/60 p-1">
              <TabsTrigger value="all" className="shrink-0 gap-1.5">
                全部
                <span className="text-[10px] text-muted-foreground">{searchedArtifacts.length}</span>
              </TabsTrigger>
              {categories.map((category) => (
                <TabsTrigger key={category.key} value={category.key} className="shrink-0 gap-1.5">
                  {category.label}
                  <span className="text-[10px] text-muted-foreground">{categoryCounts.get(category.key)}</span>
                </TabsTrigger>
              ))}
            </TabsList>
          </Tabs>
        </CardContent>
      </Card>

      {error && <Card className="border-destructive"><CardContent className="pt-6 text-sm text-destructive">{error}</CardContent></Card>}

      <Card>
        <CardHeader>
          <CardTitle>产物列表</CardTitle>
          <CardDescription>每张卡片代表一个生成文件；关联的对话仅作为追溯入口。</CardDescription>
        </CardHeader>
        <CardContent>
          {loading && <p className="text-sm text-muted-foreground">正在加载产物…</p>}
          {!loading && filteredArtifacts.length === 0 && <p className="text-sm text-muted-foreground">暂无匹配的产物。</p>}
          <div className="grid gap-4 xl:grid-cols-2">
            {filteredArtifacts.map((artifact) => (
              <article key={artifact.artifact_id} className="rounded-2xl border bg-background/70 p-4 transition hover:border-primary/40 hover:shadow-md">
                <div className="flex items-start gap-3">
                  <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-violet-100 text-violet-600"><FileText className="h-5 w-5" /></div>
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <h2 className="truncate text-base font-semibold" title={reportTitle(artifact)}>{reportTitle(artifact)}</h2>
                      <Badge variant="secondary">{categoryForArtifact(artifact).label}</Badge>
                    </div>
                    <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                      {artifact.ticker && <span className="font-medium text-foreground">{artifact.ticker}</span>}
                      {artifact.asset_type && <Badge variant="outline">{artifact.asset_type.toUpperCase()}</Badge>}
                      <span>{reportSource(artifact.source)}</span>
                      <span>·</span>
                      <span>{formatDate(artifact.created_at)}</span>
                    </div>
                  </div>
                </div>

                <div className="mt-4 rounded-xl bg-muted/30 p-1">
                  <ArtifactCard artifact={artifact} />
                </div>

                {artifact.conversation_id && (
                  <div className="mt-3 flex justify-end">
                    <Button variant="ghost" size="sm" onClick={() => navigate(`/chat?conversation=${encodeURIComponent(artifact.conversation_id || "")}`)}>
                      <ExternalLink className="mr-2 h-3.5 w-3.5" />查看关联对话
                    </Button>
                  </div>
                )}
              </article>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
