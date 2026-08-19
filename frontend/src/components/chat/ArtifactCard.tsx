import { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import type { EChartsOption } from "echarts";
import {
  Download,
  ExternalLink,
  FileCode2,
  FileText,
  Image,
  Film,
  Loader2,
} from "lucide-react";
import type { Artifact } from "@/types";
import { A2UIRenderer, createMarkdownSurface, EChart } from "./A2UIRenderer";

interface ArtifactCardProps {
  artifact: Artifact;
}

function iconFor(mimeType: string) {
  if (mimeType === "text/html")
    return <FileCode2 className="h-7 w-7 text-sky-600" />;
  if (mimeType.startsWith("image/"))
    return <Image className="h-7 w-7 text-violet-600" />;
  if (mimeType.startsWith("video/"))
    return <Film className="h-7 w-7 text-fuchsia-600" />;
  return <FileText className="h-7 w-7 text-blue-600" />;
}

function formatSize(size: number) {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}

function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let quoted = false;

  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    const nextCharacter = text[index + 1];

    if (character === '"') {
      if (quoted && nextCharacter === '"') {
        cell += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (character === "," && !quoted) {
      row.push(cell);
      cell = "";
    } else if ((character === "\n" || character === "\r") && !quoted) {
      if (character === "\r" && nextCharacter === "\n") index += 1;
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += character;
    }
  }

  if (cell || row.length) {
    row.push(cell);
    rows.push(row);
  }
  while (rows.length > 1 && rows[rows.length - 1]?.length === 1 && rows[rows.length - 1]?.[0] === "") {
    rows.pop();
  }
  return rows;
}

function formatArtifactType(mimeType: string) {
  if (mimeType === "text/html") return "HTML";
  if (mimeType === "text/markdown") return "Markdown";
  if (mimeType === "text/csv") return "CSV";
  if (mimeType === "application/json") return "JSON";
  return mimeType;
}

type JsonObject = Record<string, unknown>;
type ChartPoint = { label: string; value: number };

function asObject(value: unknown): JsonObject | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as JsonObject
    : null;
}

function numberValue(value: unknown): number | null {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) ? number : null;
}

function formatMetric(value: unknown, digits = 2) {
  const number = numberValue(value);
  return number === null ? "暂无" : number.toLocaleString("zh-CN", { maximumFractionDigits: digits });
}

function formatPercent(value: unknown) {
  const number = numberValue(value);
  return number === null ? "暂无" : `${(number * 100).toFixed(2)}%`;
}

function equityPoints(result: JsonObject): ChartPoint[] {
  if (!Array.isArray(result.equity_curve)) return [];
  return result.equity_curve.flatMap((item, index) => {
    const point = asObject(item);
    const value = numberValue(point?.value);
    return value === null ? [] : [{ label: String(point?.date || index), value }];
  }).slice(-250);
}

function ExperimentJsonPreview({ document, result, points }: { document: JsonObject; result: JsonObject; points: ChartPoint[] }) {
  const trades = Array.isArray(result.trades)
    ? result.trades.map(asObject).filter((item): item is JsonObject => Boolean(item))
    : [];
  const { equityOption, drawdownOption } = useMemo(() => {
    const labels = points.map((point) => point.label);
    const values = points.map((point) => point.value);
    const valueByDate = new Map(points.map((point) => [point.label, point.value]));
    const tradeMarkers = trades.flatMap((trade) => {
      const date = String(trade.date || "");
      const value = valueByDate.get(date);
      if (value === undefined) return [];
      const isBuy = String(trade.action || "").toLowerCase() === "buy";
      return [{
        name: isBuy ? "买入" : "卖出",
        coord: [date, value],
        value: isBuy ? "买" : "卖",
        itemStyle: { color: isBuy ? "#16a34a" : "#dc2626" },
      }];
    });
    const equityOption: EChartsOption = {
      animation: false,
      tooltip: { trigger: "axis" },
      grid: { left: 58, right: 20, top: 24, bottom: 42 },
      xAxis: { type: "category", boundaryGap: false, data: labels, axisLabel: { hideOverlap: true } },
      yAxis: { type: "value", scale: true },
      series: [{
        type: "line",
        data: values,
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 3, color: "#2563eb" },
        areaStyle: { color: "#2563eb", opacity: 0.12 },
        markPoint: tradeMarkers.length ? {
          symbol: "pin",
          symbolSize: 42,
          data: tradeMarkers,
          label: { color: "#fff", fontSize: 11 },
        } : undefined,
      }],
    };
    let peak = 0;
    const drawdowns = values.map((value) => {
      peak = Math.max(peak, value);
      return peak ? Number((((value - peak) / peak) * 100).toFixed(4)) : 0;
    });
    const drawdownOption: EChartsOption = {
      animation: false,
      tooltip: { trigger: "axis", valueFormatter: (value) => `${Number(value).toFixed(2)}%` },
      grid: { left: 58, right: 20, top: 24, bottom: 42 },
      xAxis: { type: "category", boundaryGap: false, data: labels, axisLabel: { hideOverlap: true } },
      yAxis: { type: "value", max: 0, axisLabel: { formatter: "{value}%" } },
      series: [{
        type: "line",
        data: drawdowns,
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2, color: "#dc2626" },
        areaStyle: { color: "#dc2626", opacity: 0.1 },
      }],
    };
    return { equityOption, drawdownOption };
  }, [points, trades]);

  const ticker = String(result.ticker || (Array.isArray(result.tickers) ? result.tickers.join(", ") : "实验组合"));
  const strategy = asObject(document.strategy_spec);
  return (
    <div className="space-y-4 bg-background p-4">
      <div>
        <div className="text-sm font-semibold">实验结果 · {String(strategy?.name || "未命名策略")}</div>
        <div className="mt-1 text-xs text-muted-foreground">
          {ticker} · {String(result.start_date || "")} 至 {String(result.end_date || "")} · {String(document.experiment_id || "")}
        </div>
      </div>
      <div className="grid gap-2 sm:grid-cols-5">
        {[
          ["总收益", formatPercent(result.total_return)],
          ["最大回撤", formatPercent(result.max_drawdown)],
          ["夏普", formatMetric(result.sharpe_ratio)],
          ["最终资产", formatMetric(result.final_value)],
          ["成交笔数", formatMetric(result.total_trades, 0)],
        ].map(([label, value]) => (
          <div key={label} className="rounded-lg border bg-card px-3 py-2">
            <div className="text-[11px] text-muted-foreground">{label}</div>
            <div className="mt-1 text-sm font-semibold tabular-nums">{value}</div>
          </div>
        ))}
      </div>
      <div className="rounded-lg border bg-card p-2">
        <div className="px-2 pt-1 text-xs font-medium text-muted-foreground">回测资产曲线（标记买入/卖出）</div>
        <EChart option={equityOption} height={280} ariaLabel="回测资产曲线" />
      </div>
      <div className="rounded-lg border bg-card p-2">
        <div className="px-2 pt-1 text-xs font-medium text-muted-foreground">回撤曲线</div>
        <EChart option={drawdownOption} height={230} ariaLabel="回测回撤曲线" />
      </div>
      <details className="rounded-lg border bg-card">
        <summary className="cursor-pointer px-3 py-2 text-xs font-medium text-muted-foreground">查看原始 JSON</summary>
        <pre className="max-h-72 overflow-auto border-t p-3 text-[11px] leading-relaxed">{JSON.stringify(document, null, 2)}</pre>
      </details>
    </div>
  );
}

function JsonArtifactPreview({ content }: { content: string }) {
  const parsed = useMemo(() => {
    try {
      return { value: JSON.parse(content) as unknown, error: null };
    } catch {
      return { value: null, error: "JSON 内容格式无效。" };
    }
  }, [content]);
  const document = asObject(parsed.value);
  const result = asObject(document?.result);
  const points = result ? equityPoints(result) : [];
  if (parsed.error) return <div className="p-5 text-sm text-destructive">{parsed.error}</div>;
  if (document && result && document.experiment_id && points.length >= 2) {
    return <ExperimentJsonPreview document={document} result={result} points={points} />;
  }
  return <pre className="h-full overflow-auto whitespace-pre-wrap bg-background p-5 text-xs leading-relaxed">{JSON.stringify(parsed.value, null, 2)}</pre>;
}

function TextArtifactPreview({ artifact }: { artifact: Artifact }) {
  const [content, setContent] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const isMarkdown = artifact.mime_type === "text/markdown";

  useEffect(() => {
    const controller = new AbortController();
    setContent(null);
    setError(null);

    fetch(artifact.preview_url, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.text();
      })
      .then((text) => setContent(text.replace(/^\uFEFF/, "")))
      .catch((reason: unknown) => {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        setError("文件内容加载失败，请尝试下载文件。");
      });

    return () => controller.abort();
  }, [artifact.preview_url]);

  if (error) {
    return <div className="flex h-full items-center justify-center p-6 text-sm text-destructive">{error}</div>;
  }
  if (content === null) {
    return (
      <div className="flex h-full items-center justify-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" />
        正在加载预览…
      </div>
    );
  }
  if (isMarkdown) {
    return (
      <div className="h-full overflow-auto bg-background p-5">
        <A2UIRenderer messages={createMarkdownSurface(content, `artifact-${artifact.artifact_id}`)} />
      </div>
    );
  }
  if (artifact.mime_type === "application/json") {
    return <JsonArtifactPreview content={content} />;
  }

  const rows = parseCsv(content);
  const headers = rows[0] || [];
  const dataRows = rows.slice(1);
  if (!headers.length) {
    return <div className="p-5 text-sm text-muted-foreground">CSV 文件为空。</div>;
  }

  return (
    <div className="h-full overflow-auto bg-background p-4">
      <div className="mb-3 text-xs text-muted-foreground">
        {dataRows.length.toLocaleString("zh-CN")} 行 · {headers.length.toLocaleString("zh-CN")} 列
      </div>
      <div className="overflow-x-auto rounded-lg border">
        <table className="w-full min-w-max text-left text-xs">
          <thead className="sticky top-0 bg-muted/95 text-muted-foreground">
            <tr>
              {headers.map((header, index) => (
                <th key={`${header}-${index}`} className="whitespace-nowrap px-3 py-2 font-medium">
                  {header || `列 ${index + 1}`}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {dataRows.map((row, rowIndex) => (
              <tr key={rowIndex} className="border-t border-border/70 even:bg-muted/20">
                {headers.map((_, columnIndex) => (
                  <td key={columnIndex} className="whitespace-pre-wrap px-3 py-2 align-top">
                    {row[columnIndex] || ""}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function previewContent(artifact: Artifact) {
  if (artifact.mime_type === "text/markdown" || artifact.mime_type === "text/csv" || artifact.mime_type === "application/json") {
    return <TextArtifactPreview artifact={artifact} />;
  }
  if (artifact.mime_type.startsWith("image/")) {
    return (
      <div className="flex h-full items-center justify-center overflow-auto bg-slate-950/5 p-4">
        <img src={artifact.preview_url} alt={artifact.name} className="max-h-full max-w-full object-contain" />
      </div>
    );
  }
  if (artifact.mime_type.startsWith("video/")) {
    return (
      <div className="flex h-full items-center justify-center bg-slate-950 p-4">
        <video src={artifact.preview_url} controls className="max-h-full max-w-full" />
      </div>
    );
  }
  return (
    <iframe
      title={`预览 ${artifact.name}`}
      src={artifact.preview_url}
      className="h-full w-full border-0"
      sandbox="allow-scripts"
    />
  );
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
            {formatArtifactType(artifact.mime_type)}
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
                    {formatArtifactType(artifact.mime_type)}
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
                {previewContent(artifact)}
              </div>
            </aside>
          </div>,
          document.body,
        )}
    </>
  );
}
