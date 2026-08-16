import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Archive, CalendarDays, ExternalLink, FileText, RefreshCw, Search } from "lucide-react";
import { getArtifacts } from "@/api";
import type { Artifact } from "@/types";
import { ArtifactCard } from "@/components/chat/ArtifactCard";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";

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

export function RecordsPage() {
  const navigate = useNavigate();
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadReports = async () => {
    setLoading(true);
    setError(null);
    try {
      setArtifacts(await getArtifacts());
    } catch (err) {
      setError(err instanceof Error ? err.message : "报告加载失败");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadReports();
  }, []);

  const filteredArtifacts = useMemo(() => {
    const text = query.trim().toLowerCase();
    if (!text) return artifacts;
    return artifacts.filter((artifact) =>
      `${artifact.name} ${artifact.ticker || ""} ${artifact.asset_type || ""} ${artifact.source || ""}`
        .toLowerCase()
        .includes(text),
    );
  }, [artifacts, query]);

  return (
    <div className="space-y-6 p-6">
      <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
        <div>
          <div className="eyebrow">Research library</div>
          <h1 className="mt-2 text-2xl font-bold">研究报告</h1>
          <p className="mt-1 text-sm text-muted-foreground">以报告为中心沉淀每次分析结果，统一预览、下载和追溯来源。</p>
        </div>
        <Button variant="outline" onClick={() => void loadReports()} disabled={loading}>
          <RefreshCw className={`mr-2 h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          刷新报告
        </Button>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <Card className="border-violet-100 bg-gradient-to-br from-white to-violet-50/70">
          <CardContent className="flex items-center gap-4 p-5">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-violet-100 text-violet-600"><FileText className="h-5 w-5" /></div>
            <div><p className="text-sm text-muted-foreground">报告总数</p><p className="text-2xl font-semibold">{artifacts.length}</p></div>
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
            <div className="min-w-0"><p className="text-sm text-muted-foreground">最近生成</p><p className="truncate text-sm font-semibold">{artifacts[0] ? formatDate(artifacts[0].created_at) : "暂无报告"}</p></div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardContent className="p-4">
          <div className="relative">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input className="pl-9" placeholder="搜索报告名称、标的代码或来源" value={query} onChange={(event) => setQuery(event.target.value)} />
          </div>
        </CardContent>
      </Card>

      {error && <Card className="border-destructive"><CardContent className="pt-6 text-sm text-destructive">{error}</CardContent></Card>}

      <Card>
        <CardHeader>
          <CardTitle>报告列表</CardTitle>
          <CardDescription>每张卡片代表一份独立研究报告；报告关联的对话仅作为追溯入口。</CardDescription>
        </CardHeader>
        <CardContent>
          {loading && <p className="text-sm text-muted-foreground">正在加载报告…</p>}
          {!loading && filteredArtifacts.length === 0 && <p className="text-sm text-muted-foreground">暂无匹配的研究报告。</p>}
          <div className="grid gap-4 xl:grid-cols-2">
            {filteredArtifacts.map((artifact) => (
              <article key={artifact.artifact_id} className="rounded-2xl border bg-background/70 p-4 transition hover:border-primary/40 hover:shadow-md">
                <div className="flex items-start gap-3">
                  <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-violet-100 text-violet-600"><FileText className="h-5 w-5" /></div>
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <h2 className="truncate text-base font-semibold" title={reportTitle(artifact)}>{reportTitle(artifact)}</h2>
                      <Badge variant="secondary">研究报告</Badge>
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
