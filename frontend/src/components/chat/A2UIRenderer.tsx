import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import type { EChartsOption } from "echarts";
import type { ECharts } from "echarts/core";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export interface A2UIMessage {
  version?: "v0.9";
  createSurface?: {
    surfaceId: string;
    catalogId: string;
    sendDataModel?: boolean;
  };
  updateComponents?: {
    surfaceId: string;
    components: A2UIComponent[];
  };
  updateDataModel?: {
    surfaceId: string;
    path?: string;
    value?: unknown;
  };
  deleteSurface?: { surfaceId: string };
}

interface A2UIComponent {
  id: string;
  component: string;
  children?: string[];
  [key: string]: unknown;
}

interface SurfaceState {
  id: string;
  catalogId?: string;
  components: Map<string, A2UIComponent>;
  model: Record<string, unknown>;
}

export interface A2UIAction {
  name: string;
  surfaceId: string;
  context: Record<string, unknown>;
}

export function createMarkdownSurface(
  text: string,
  surfaceId: string,
): A2UIMessage[] {
  return [
    {
      version: "v0.9",
      createSurface: {
        surfaceId,
        catalogId: "https://a-share-agent.local/a2ui/catalog/v0.9",
      },
    },
    {
      version: "v0.9",
      updateComponents: {
        surfaceId,
        components: [
          { id: "root", component: "Markdown", text: { path: "/text" } },
        ],
      },
    },
    {
      version: "v0.9",
      updateDataModel: { surfaceId, path: "/", value: { text } },
    },
  ];
}

export function MarkdownInline({ text }: { text: string }) {
  const lines = text.split("\n");
  return (
    <>
      {lines.map((line, index) => (
        <span key={index}>
          {inlineMarkdown(line)}
          {index < lines.length - 1 && <br />}
        </span>
      ))}
    </>
  );
}

interface A2UIRendererProps {
  messages: A2UIMessage[];
  onAction?: (action: A2UIAction) => void;
}

type Binding = { path: string } | { literalString: string };

function isBinding(value: unknown): value is Binding {
  return Boolean(
    value &&
    typeof value === "object" &&
    ("path" in value || "literalString" in value),
  );
}

function getAtPath(value: unknown, path: string, scope?: unknown): unknown {
  if (!path) return scope ?? value;
  const source = path.startsWith("/") ? value : scope;
  if (source === undefined) return undefined;
  return path
    .replace(/^\//, "")
    .split("/")
    .filter(Boolean)
    .map((part) => part.replace(/~1/g, "/").replace(/~0/g, "~"))
    .reduce<unknown>((current, part) => {
      if (current && typeof current === "object") {
        return (current as Record<string, unknown>)[part];
      }
      return undefined;
    }, source);
}

function resolveValue(
  value: unknown,
  model: unknown,
  scope?: unknown,
): unknown {
  if (!isBinding(value)) return value;
  if ("literalString" in value) return value.literalString;
  return getAtPath(model, value.path, scope);
}

function displayValue(value: unknown): string {
  if (value === undefined || value === null) return "";
  if (typeof value === "string") return value;
  if (typeof value === "number") return value.toLocaleString("zh-CN");
  return JSON.stringify(value);
}

function cloneAndSet(
  root: Record<string, unknown>,
  path: string,
  value: unknown,
) {
  const next = structuredClone(root);
  const keys = path.replace(/^\//, "").split("/").filter(Boolean);
  if (!keys.length) return (value || {}) as Record<string, unknown>;
  let cursor: Record<string, unknown> = next;
  keys.forEach((key, index) => {
    const decoded = key.replace(/~1/g, "/").replace(/~0/g, "~");
    if (index === keys.length - 1) {
      cursor[decoded] = value;
    } else {
      if (!cursor[decoded] || typeof cursor[decoded] !== "object") {
        cursor[decoded] = {};
      }
      cursor = cursor[decoded] as Record<string, unknown>;
    }
  });
  return next;
}

function buildSurfaces(messages: A2UIMessage[]): SurfaceState[] {
  const surfaces = new Map<string, SurfaceState>();
  messages.forEach((message) => {
    if (message.createSurface) {
      const { surfaceId, catalogId } = message.createSurface;
      surfaces.set(surfaceId, {
        id: surfaceId,
        catalogId,
        components: new Map(),
        model: {},
      });
    }
    if (message.updateComponents) {
      const update = message.updateComponents;
      const surface = surfaces.get(update.surfaceId) || {
        id: update.surfaceId,
        components: new Map<string, A2UIComponent>(),
        model: {},
      };
      update.components.forEach((component) =>
        surface.components.set(component.id, component),
      );
      surfaces.set(update.surfaceId, surface);
    }
    if (message.updateDataModel) {
      const update = message.updateDataModel;
      const surface = surfaces.get(update.surfaceId) || {
        id: update.surfaceId,
        components: new Map<string, A2UIComponent>(),
        model: {},
      };
      surface.model =
        update.path && update.path !== "/"
          ? cloneAndSet(surface.model, update.path, update.value)
          : ((update.value || {}) as Record<string, unknown>);
      surfaces.set(update.surfaceId, surface);
    }
    if (message.deleteSurface) surfaces.delete(message.deleteSurface.surfaceId);
  });
  return Array.from(surfaces.values());
}

export function A2UIRenderer({ messages, onAction }: A2UIRendererProps) {
  const baseSurfaces = useMemo(() => buildSurfaces(messages), [messages]);
  const [models, setModels] = useState<Record<string, Record<string, unknown>>>(
    {},
  );

  useEffect(() => {
    setModels(
      Object.fromEntries(
        baseSurfaces.map((surface) => [surface.id, surface.model]),
      ),
    );
  }, [baseSurfaces]);

  if (!baseSurfaces.length) return null;

  return (
    <div className="w-full min-w-0 max-w-full space-y-3">
      {baseSurfaces.map((surface) => (
        <SurfaceRenderer
          key={surface.id}
          surface={{ ...surface, model: models[surface.id] || surface.model }}
          onModelChange={(path, value) =>
            setModels((current) => ({
              ...current,
              [surface.id]: cloneAndSet(
                current[surface.id] || surface.model,
                path,
                value,
              ),
            }))
          }
          onAction={onAction}
        />
      ))}
    </div>
  );
}

function SurfaceRenderer({
  surface,
  onModelChange,
  onAction,
}: {
  surface: SurfaceState;
  onModelChange: (path: string, value: unknown) => void;
  onAction?: (action: A2UIAction) => void;
}) {
  const root = surface.components.get("root");
  if (!root) return null;
  return (
    <div
      className="animate-in fade-in slide-in-from-bottom-1 duration-300"
      data-a2ui-surface={surface.id}
      data-a2ui-catalog={surface.catalogId}
    >
      <RenderComponent
        component={root}
        surface={surface}
        scope={surface.model}
        onModelChange={onModelChange}
        onAction={onAction}
      />
    </div>
  );
}

function CollapsibleRenderer({
  component,
  surface,
  scope,
  onModelChange,
  onAction,
}: {
  component: A2UIComponent;
  surface: SurfaceState;
  scope: unknown;
  onModelChange: (path: string, value: unknown) => void;
  onAction?: (action: A2UIAction) => void;
}) {
  const defaultExpanded = component.defaultExpanded !== false;
  const [expanded, setExpanded] = useState(defaultExpanded);
  const children = (component.children || [])
    .map((id) => surface.components.get(id))
    .filter((item): item is A2UIComponent => Boolean(item));

  useEffect(() => {
    setExpanded(defaultExpanded);
  }, [surface.id, component.id, defaultExpanded]);

  return (
    <section className="overflow-hidden rounded-xl border bg-card shadow-sm">
      <button
        type="button"
        className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left transition-colors hover:bg-muted/30"
        aria-expanded={expanded}
        onClick={() => setExpanded((current) => !current)}
      >
        <span className="text-base font-semibold">
          {String(component.title || "查看详情")}
        </span>
        <ChevronDown
          className={cn(
            "h-5 w-5 shrink-0 text-muted-foreground transition-transform",
            expanded && "rotate-180",
          )}
        />
      </button>
      {expanded && (
        <div className="border-t px-4 py-4">
          <div className="space-y-3">
            {children.map((child) => (
              <RenderComponent
                key={child.id}
                component={child}
                surface={surface}
                scope={scope}
                onModelChange={onModelChange}
                onAction={onAction}
              />
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

export function EChart({
  option,
  height = 280,
  ariaLabel,
}: {
  option: EChartsOption;
  height?: number;
  ariaLabel: string;
}) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return undefined;
    let cancelled = false;
    let chart: ECharts | undefined;
    const resize = () => chart?.resize();
    void import("@/lib/echarts").then(({ init }) => {
      if (cancelled) return;
      chart = init(container, undefined, { renderer: "canvas" });
      chart.setOption(option);
      window.addEventListener("resize", resize);
    });
    return () => {
      cancelled = true;
      window.removeEventListener("resize", resize);
      chart?.dispose();
    };
  }, [option]);

  return (
    <div
      ref={containerRef}
      className="w-full"
      style={{ height }}
      role="img"
      aria-label={ariaLabel}
    />
  );
}

function LineChart({ points }: { points: unknown }) {
  const validPoints = (Array.isArray(points) ? points : [])
    .map((point) => {
      if (!point || typeof point !== "object") return null;
      const record = point as Record<string, unknown>;
      const value = Number(record.value);
      if (!Number.isFinite(value)) return null;
      return { label: String(record.label || ""), value };
    })
    .filter(
      (point): point is { label: string; value: number } => point !== null,
    );

  if (validPoints.length < 2) {
    return <p className="text-xs text-muted-foreground">暂无足够的走势数据</p>;
  }

  const values = validPoints.map((point) => point.value);
  const option = useMemo<EChartsOption>(
    () => ({
      animation: false,
      tooltip: { trigger: "axis" },
      grid: { left: 52, right: 20, top: 18, bottom: 42 },
      xAxis: {
        type: "category",
        boundaryGap: false,
        data: validPoints.map((point) => point.label),
        axisLabel: { hideOverlap: true },
      },
      yAxis: { type: "value", scale: true },
      series: [
        {
          type: "line",
          data: values,
          smooth: true,
          showSymbol: false,
          lineStyle: { width: 3, color: "#2563eb" },
          itemStyle: { color: "#2563eb" },
          areaStyle: { color: "#2563eb", opacity: 0.12 },
        },
      ],
    }),
    [validPoints, values],
  );

  return (
    <div className="rounded-lg border border-border/70 bg-background/50 p-2">
      <div className="flex items-center justify-between px-1 text-xs text-muted-foreground">
        <span>{validPoints[0].label || "起始日"}</span>
        <span>最新 {validPoints[validPoints.length - 1].value.toFixed(2)}</span>
        <span>{validPoints[validPoints.length - 1].label || "最新日"}</span>
      </div>
      <EChart option={option} height={260} ariaLabel="历史收盘价走势" />
    </div>
  );
}

function MultiLineChart({
  series,
  ariaLabel,
}: {
  series: unknown;
  ariaLabel: string;
}) {
  const palette = ["#2563eb", "#7c3aed", "#0891b2", "#ea580c"];
  const validSeries = (Array.isArray(series) ? series : [])
    .map((item, seriesIndex) => {
      if (!item || typeof item !== "object") return null;
      const record = item as Record<string, unknown>;
      const points = (Array.isArray(record.points) ? record.points : [])
        .map((point) => {
          if (!point || typeof point !== "object") return null;
          const pointRecord = point as Record<string, unknown>;
          const value = Number(pointRecord.value);
          if (!Number.isFinite(value)) return null;
          return { label: String(pointRecord.label || ""), value };
        })
        .filter(
          (point): point is { label: string; value: number } => point !== null,
        );
      if (points.length < 2) return null;
      return { name: String(record.name || `策略 ${seriesIndex + 1}`), points };
    })
    .filter(
      (
        item,
      ): item is {
        name: string;
        points: Array<{ label: string; value: number }>;
      } => item !== null,
    );

  const labels = validSeries[0]?.points.map((point) => point.label) || [];
  const option = useMemo<EChartsOption>(
    () => ({
      animation: false,
      color: palette,
      tooltip: { trigger: "axis" },
      legend: {
        type: "scroll",
        top: 4,
        left: 8,
        right: 8,
        textStyle: { fontSize: 11 },
      },
      grid: { left: 54, right: 20, top: 52, bottom: 44 },
      xAxis: {
        type: "category",
        boundaryGap: false,
        data: labels,
        axisLabel: { hideOverlap: true },
      },
      yAxis: { type: "value", scale: true },
      series: validSeries.map((item) => ({
        name: item.name,
        type: "line",
        data: item.points.map((point) => point.value),
        smooth: false,
        showSymbol: false,
        lineStyle: { width: 2 },
      })),
    }),
    [labels, validSeries],
  );

  if (!validSeries.length) {
    return (
      <p className="text-xs text-muted-foreground">暂无足够的对比曲线数据</p>
    );
  }
  return (
    <div className="min-w-0 overflow-hidden rounded-lg border border-border/70 bg-background/50 p-2">
      <EChart option={option} height={320} ariaLabel={ariaLabel} />
    </div>
  );
}

interface NormalizedTrade {
  date: string;
  action: "buy" | "sell";
  price: number;
  shares: number;
  amount: number;
}

interface NormalizedTradeStrategy {
  name: string;
  trades: NormalizedTrade[];
}

function formatTradeTooltip(params: unknown): string {
  const items = Array.isArray(params) ? params : [params];
  const first = items[0];
  const firstRecord =
    first && typeof first === "object"
      ? (first as Record<string, unknown>)
      : undefined;
  const lines = [String(firstRecord?.axisValue || "")];
  items.forEach((item) => {
    if (!item || typeof item !== "object") return;
    const record = item as Record<string, unknown>;
    const data = record.data;
    const dataRecord =
      data && typeof data === "object" && !Array.isArray(data)
        ? (data as Record<string, unknown>)
        : undefined;
    const value = dataRecord?.value ?? data;
    const values = Array.isArray(value) ? value : [value];
    const price = Number(values[values.length - 1]);
    if (!Number.isFinite(price)) return;
    let line = `${String(record.seriesName || "价格")}：${price.toFixed(3)}`;
    const shares = Number(dataRecord?.shares || 0);
    const amount = Number(dataRecord?.amount || 0);
    if (shares > 0) line += ` · ${shares.toLocaleString("zh-CN")} 份`;
    if (amount > 0)
      line += ` · ¥${amount.toLocaleString("zh-CN", { maximumFractionDigits: 2 })}`;
    lines.push(line);
  });
  return lines.filter(Boolean).join("\n");
}

function TradeChart({
  pricePoints,
  strategies,
  ariaLabel,
}: {
  pricePoints: unknown;
  strategies: unknown;
  ariaLabel: string;
}) {
  const prices = useMemo(
    () =>
      (Array.isArray(pricePoints) ? pricePoints : [])
        .map((point) => {
          if (!point || typeof point !== "object") return null;
          const record = point as Record<string, unknown>;
          const value = Number(record.value);
          if (!Number.isFinite(value)) return null;
          return { label: String(record.label || ""), value };
        })
        .filter(
          (point): point is { label: string; value: number } => point !== null,
        ),
    [pricePoints],
  );
  const normalizedStrategies = useMemo<NormalizedTradeStrategy[]>(
    () =>
      (Array.isArray(strategies) ? strategies : [])
        .map((strategy, strategyIndex) => {
          if (!strategy || typeof strategy !== "object") return null;
          const record = strategy as Record<string, unknown>;
          const trades = (Array.isArray(record.trades) ? record.trades : [])
            .map((trade) => {
              if (!trade || typeof trade !== "object") return null;
              const tradeRecord = trade as Record<string, unknown>;
              const actionValue = String(
                tradeRecord.action || "",
              ).toLowerCase();
              const action = actionValue.endsWith("buy")
                ? "buy"
                : actionValue.endsWith("sell")
                  ? "sell"
                  : null;
              const price = Number(tradeRecord.price);
              if (!action || !Number.isFinite(price)) return null;
              return {
                date: String(tradeRecord.date || ""),
                action,
                price,
                shares: Number(tradeRecord.shares || 0),
                amount: Number(tradeRecord.amount || 0),
              };
            })
            .filter((trade): trade is NormalizedTrade => trade !== null);
          return {
            name: String(record.name || `策略 ${strategyIndex + 1}`),
            trades,
          };
        })
        .filter(
          (strategy): strategy is NormalizedTradeStrategy => strategy !== null,
        ),
    [strategies],
  );
  const [selectedName, setSelectedName] = useState("");
  const selectedStrategy =
    normalizedStrategies.find((strategy) => strategy.name === selectedName) ||
    normalizedStrategies[0];
  const buyTrades =
    selectedStrategy?.trades.filter((trade) => trade.action === "buy") || [];
  const sellTrades =
    selectedStrategy?.trades.filter((trade) => trade.action === "sell") || [];
  const option = useMemo<EChartsOption>(
    () => ({
      animation: false,
      color: ["#2563eb", "#dc2626", "#16a34a"],
      tooltip: {
        trigger: "axis",
        renderMode: "richText",
        axisPointer: { type: "cross" },
        formatter: formatTradeTooltip,
      },
      legend: { top: 4, left: 8 },
      grid: { left: 58, right: 22, top: 48, bottom: 68 },
      dataZoom: [
        { type: "inside", start: 0, end: 100 },
        { type: "slider", start: 0, end: 100, height: 20, bottom: 12 },
      ],
      xAxis: {
        type: "category",
        boundaryGap: false,
        data: prices.map((point) => point.label),
        axisLabel: { hideOverlap: true },
      },
      yAxis: { type: "value", scale: true, name: "价格" },
      series: [
        {
          name: "标的价格",
          type: "line",
          data: prices.map((point) => point.value),
          showSymbol: false,
          lineStyle: { width: 2, color: "#2563eb" },
        },
        {
          name: "买入",
          type: "scatter",
          data: buyTrades.map((trade) => ({
            value: [trade.date, trade.price],
            shares: trade.shares,
            amount: trade.amount,
          })),
          symbol: "triangle",
          symbolSize: 18,
          itemStyle: { color: "#dc2626" },
          label: {
            show: true,
            formatter: "买",
            position: "top",
            color: "#dc2626",
          },
        },
        {
          name: "卖出",
          type: "scatter",
          data: sellTrades.map((trade) => ({
            value: [trade.date, trade.price],
            shares: trade.shares,
            amount: trade.amount,
          })),
          symbol: "triangle",
          symbolRotate: 180,
          symbolSize: 18,
          itemStyle: { color: "#16a34a" },
          label: {
            show: true,
            formatter: "卖",
            position: "bottom",
            color: "#16a34a",
          },
        },
      ],
    }),
    [buyTrades, prices, sellTrades],
  );

  if (prices.length < 2 || !selectedStrategy) {
    return (
      <p className="text-xs text-muted-foreground">暂无足够的价格或策略数据</p>
    );
  }

  return (
    <div className="min-w-0 space-y-2 overflow-hidden rounded-lg border border-border/70 bg-background/50 p-2">
      <div className="flex flex-wrap items-center justify-between gap-2 px-1 text-xs text-muted-foreground">
        <label className="flex items-center gap-2">
          <span>策略</span>
          <select
            className="max-w-[24rem] rounded-md border bg-background px-2 py-1 text-foreground"
            value={selectedStrategy.name}
            onChange={(event) => setSelectedName(event.target.value)}
          >
            {normalizedStrategies.map((strategy) => (
              <option key={strategy.name} value={strategy.name}>
                {strategy.name}（{strategy.trades.length} 次成交）
              </option>
            ))}
          </select>
        </label>
        <span>
          买入 {buyTrades.length} 次 · 卖出 {sellTrades.length} 次
        </span>
      </div>
      <EChart option={option} height={380} ariaLabel={ariaLabel} />
    </div>
  );
}

interface NormalizedBenchmarkStrategy {
  name: string;
  assetPoints: Array<{ label: string; value: number }>;
  marketPoints: Array<{ label: string; value: number }>;
}

function normalizeChartPoints(value: unknown) {
  return (Array.isArray(value) ? value : [])
    .map((point) => {
      if (!point || typeof point !== "object") return null;
      const record = point as Record<string, unknown>;
      const number = Number(record.value);
      if (!Number.isFinite(number)) return null;
      return { label: String(record.label || ""), value: number };
    })
    .filter(
      (point): point is { label: string; value: number } => point !== null,
    );
}

function BenchmarkChart({
  strategies,
  assetLabel,
  marketLabel,
  ariaLabel,
}: {
  strategies: unknown;
  assetLabel: string;
  marketLabel: string;
  ariaLabel: string;
}) {
  const normalizedStrategies = useMemo<NormalizedBenchmarkStrategy[]>(
    () =>
      (Array.isArray(strategies) ? strategies : [])
        .map((strategy, index) => {
          if (!strategy || typeof strategy !== "object") return null;
          const record = strategy as Record<string, unknown>;
          const assetPoints = normalizeChartPoints(record.assetPoints);
          const marketPoints = normalizeChartPoints(record.marketPoints);
          if (assetPoints.length < 2 || marketPoints.length < 2) return null;
          return {
            name: String(record.name || `策略 ${index + 1}`),
            assetPoints,
            marketPoints,
          };
        })
        .filter(
          (strategy): strategy is NormalizedBenchmarkStrategy =>
            strategy !== null,
        ),
    [strategies],
  );
  const [selectedName, setSelectedName] = useState("");
  const selectedStrategy =
    normalizedStrategies.find((strategy) => strategy.name === selectedName) ||
    normalizedStrategies[0];
  const labels = useMemo(
    () =>
      Array.from(
        new Set([
          ...(selectedStrategy?.assetPoints.map((point) => point.label) || []),
          ...(selectedStrategy?.marketPoints.map((point) => point.label) || []),
        ]),
      ).sort(),
    [selectedStrategy],
  );
  const option = useMemo<EChartsOption>(() => {
    const assetByDate = new Map(
      selectedStrategy?.assetPoints.map((point) => [point.label, point.value]),
    );
    const marketByDate = new Map(
      selectedStrategy?.marketPoints.map((point) => [point.label, point.value]),
    );
    return {
      animation: false,
      color: ["#2563eb", "#ea580c"],
      tooltip: { trigger: "axis", axisPointer: { type: "cross" } },
      legend: { top: 4, left: 8 },
      grid: { left: 58, right: 22, top: 48, bottom: 68 },
      dataZoom: [
        { type: "inside", start: 0, end: 100 },
        { type: "slider", start: 0, end: 100, height: 20, bottom: 12 },
      ],
      xAxis: {
        type: "category",
        boundaryGap: false,
        data: labels,
        axisLabel: { hideOverlap: true },
      },
      yAxis: { type: "value", scale: true, name: "归一化净值" },
      series: [
        {
          name: assetLabel,
          type: "line",
          data: labels.map((label) => assetByDate.get(label) ?? null),
          connectNulls: true,
          showSymbol: false,
          lineStyle: { width: 2 },
        },
        {
          name: marketLabel,
          type: "line",
          data: labels.map((label) => marketByDate.get(label) ?? null),
          connectNulls: true,
          showSymbol: false,
          lineStyle: { width: 2 },
        },
      ],
    };
  }, [assetLabel, labels, marketLabel, selectedStrategy]);

  if (!selectedStrategy) {
    return (
      <p className="text-xs text-muted-foreground">
        暂无可用的同期大盘策略曲线
      </p>
    );
  }
  return (
    <div className="min-w-0 space-y-2 overflow-hidden rounded-lg border border-border/70 bg-background/50 p-2">
      <label className="flex items-center gap-2 px-1 text-xs text-muted-foreground">
        <span>策略</span>
        <select
          className="max-w-[24rem] rounded-md border bg-background px-2 py-1 text-foreground"
          value={selectedStrategy.name}
          onChange={(event) => setSelectedName(event.target.value)}
        >
          {normalizedStrategies.map((strategy) => (
            <option key={strategy.name} value={strategy.name}>
              {strategy.name}
            </option>
          ))}
        </select>
      </label>
      <EChart option={option} height={380} ariaLabel={ariaLabel} />
    </div>
  );
}

function safeArtifactUrl(value: unknown, suffix: "preview" | "download") {
  const url = displayValue(value).trim();
  return /^\/api\/artifacts\/[A-Za-z0-9_-]+\/(preview|download)$/.test(url) &&
    url.endsWith(`/${suffix}`)
    ? url
    : "";
}

function RenderComponent({
  component,
  surface,
  scope,
  onModelChange,
  onAction,
}: {
  component: A2UIComponent;
  surface: SurfaceState;
  scope: unknown;
  onModelChange: (path: string, value: unknown) => void;
  onAction?: (action: A2UIAction) => void;
}) {
  const resolve = (value: unknown) => resolveValue(value, surface.model, scope);
  const children = (component.children || [])
    .map((id) => surface.components.get(id))
    .filter((item): item is A2UIComponent => Boolean(item));
  const renderChildren = () =>
    children.map((child) => (
      <RenderComponent
        key={child.id}
        component={child}
        surface={surface}
        scope={scope}
        onModelChange={onModelChange}
        onAction={onAction}
      />
    ));

  switch (component.component) {
    case "Markdown":
      return <MarkdownContent text={displayValue(resolve(component.text))} />;
    case "Text": {
      const variant = String(component.variant || "body");
      return (
        <p
          className={cn(
            "whitespace-pre-wrap text-sm leading-relaxed",
            variant === "h3" && "text-base font-semibold",
            variant === "h4" && "text-sm font-semibold",
            variant === "metric" && "text-xl font-semibold",
            variant === "caption" && "text-xs text-muted-foreground",
            component.tone === "positive" && "text-green-500",
            component.tone === "negative" && "text-red-500",
          )}
        >
          {displayValue(resolve(component.text))}
        </p>
      );
    }
    case "CodeBlock": {
      const code = displayValue(resolve(component.code));
      const language = displayValue(resolve(component.language)) || "text";
      return (
        <div className="overflow-hidden rounded-lg border border-slate-700 bg-slate-950 text-slate-100">
          <div className="border-b border-slate-800 px-3 py-1.5 font-mono text-[11px] uppercase tracking-wide text-slate-400">
            {language}
          </div>
          <pre className="max-h-[32rem] overflow-auto p-3 text-xs leading-relaxed">
            <code className="font-mono">{code}</code>
          </pre>
        </div>
      );
    }
    case "Row":
      return (
        <div className="flex flex-wrap items-center gap-3">
          {renderChildren()}
        </div>
      );
    case "Column":
      return <div className="flex flex-col gap-2">{renderChildren()}</div>;
    case "Card":
      return (
        <div className="rounded-xl border bg-card p-4 shadow-sm">
          <div className="space-y-3">{renderChildren()}</div>
        </div>
      );
    case "Collapsible":
      return (
        <CollapsibleRenderer
          component={component}
          surface={surface}
          scope={scope}
          onModelChange={onModelChange}
          onAction={onAction}
        />
      );
    case "Section":
      return (
        <section className="rounded-lg border border-border/70 bg-background/40 p-3">
          {Boolean(component.title) && (
            <h4 className="mb-2 text-xs font-semibold text-muted-foreground">
              {String(component.title)}
            </h4>
          )}
          <div className="space-y-2">{renderChildren()}</div>
        </section>
      );
    case "Badge": {
      const tone = String(resolve(component.tone) || "secondary");
      const variant =
        tone === "sell" || tone === "strong_sell"
          ? "destructive"
          : tone === "buy" || tone === "strong_buy"
            ? "success"
            : "secondary";
      return (
        <Badge variant={variant}>{displayValue(resolve(component.text))}</Badge>
      );
    }
    case "Progress":
      return (
        <div className="h-2 overflow-hidden rounded-full bg-muted">
          <div
            className="h-full rounded-full bg-primary transition-all"
            style={{
              width: `${Math.max(0, Math.min(100, Number(resolve(component.value) || 0)))}%`,
            }}
          />
        </div>
      );
    case "ScoreBar": {
      const value = Number(resolve(component.value) || 0);
      return (
        <div className="space-y-1">
          <div className="flex justify-between text-xs">
            <span>{String(component.label || "")}</span>
            <span>
              {value >= 0 ? "+" : ""}
              {value.toFixed(0)}
            </span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-muted">
            <div
              className={cn(
                "h-full rounded-full",
                value >= 0 ? "bg-primary" : "bg-destructive",
              )}
              style={{ width: `${Math.min(100, Math.abs(value))}%` }}
            />
          </div>
        </div>
      );
    }
    case "PipelineStep": {
      const status = String(component.status || "pending");
      const completed = ["done", "complete", "completed"].includes(status);
      const failed = status === "failed";
      const skipped = status === "skipped";
      const statusClass = completed
        ? "border-green-500 bg-green-500/10 text-green-500"
        : failed
          ? "border-destructive bg-destructive/10 text-destructive"
          : skipped
            ? "border-amber-500 bg-amber-500/10 text-amber-600"
            : status === "running"
              ? "border-primary bg-primary/10 text-primary"
              : "border-border text-muted-foreground";
      const icon = completed
        ? "✓"
        : failed
          ? "×"
          : skipped
            ? "–"
            : status === "running"
              ? "•"
              : "○";
      const detail = String(component.detail || "");
      return (
        <div className="flex items-start gap-2 text-xs">
          <span
            className={cn(
              "flex h-6 w-6 shrink-0 items-center justify-center rounded-full border text-[10px]",
              statusClass,
            )}
          >
            {icon}
          </span>
          <span className="min-w-0">
            <span className="block">{String(component.label || "")}</span>
            {detail && (
              <span
                className={cn(
                  "mt-0.5 block text-[11px]",
                  failed ? "text-destructive" : "text-muted-foreground",
                )}
              >
                {detail}
              </span>
            )}
          </span>
        </div>
      );
    }
    case "List": {
      const items = resolve(component.items);
      if (!Array.isArray(items)) return null;
      const template = component.itemTemplate
        ? surface.components.get(String(component.itemTemplate))
        : undefined;
      return (
        <div className="space-y-1">
          {Boolean(component.title) && (
            <p className="text-xs font-medium text-muted-foreground">
              {String(component.title)}
            </p>
          )}
          {items.map((item, index) =>
            template ? (
              <RenderComponent
                key={index}
                component={template}
                surface={surface}
                scope={item}
                onModelChange={onModelChange}
                onAction={onAction}
              />
            ) : (
              <div
                key={index}
                className="rounded-md bg-muted/50 px-2 py-1 text-xs"
              >
                {displayValue(item)}
              </div>
            ),
          )}
        </div>
      );
    }
    case "StrategyItem":
      return (
        <div className="rounded-lg border px-3 py-2">
          <div className="flex items-center justify-between text-sm font-medium">
            <span>{displayValue(resolve(component.name))}</span>
            {Boolean(resolve(component.active)) && (
              <Badge variant="success">启用</Badge>
            )}
          </div>
          <p className="mt-1 text-xs text-muted-foreground">
            {displayValue(resolve(component.description))}
          </p>
        </div>
      );
    case "SearchResultItem": {
      const link = displayValue(resolve(component.link)).trim();
      const title = displayValue(resolve(component.title)) || link;
      const snippet = displayValue(resolve(component.snippet));
      const source = displayValue(resolve(component.source));
      const date = displayValue(resolve(component.date));
      const isExternalLink = /^https?:\/\//i.test(link);
      return (
        <article className="rounded-lg border px-3 py-2">
          {isExternalLink ? (
            <a
              href={link}
              target="_blank"
              rel="noreferrer"
              className="text-sm font-medium text-primary underline underline-offset-2 hover:opacity-80"
            >
              {title}
            </a>
          ) : (
            <p className="text-sm font-medium">{title}</p>
          )}
          {(source || date) && (
            <p className="mt-1 text-xs text-muted-foreground">
              {[source, date].filter(Boolean).join(" · ")}
            </p>
          )}
          {snippet && (
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
              {snippet}
            </p>
          )}
        </article>
      );
    }
    case "StatusItem": {
      const status = displayValue(resolve(component.status));
      return (
        <div className="flex items-center justify-between rounded-md bg-muted/50 px-2 py-1 text-xs">
          <span>{displayValue(resolve(component.label))}</span>
          <Badge
            variant={
              status === "closed"
                ? "success"
                : status === "open"
                  ? "destructive"
                  : "warning"
            }
          >
            {status}
          </Badge>
        </div>
      );
    }
    case "Sparkline": {
      const values = (resolve(component.values) as number[]) || [];
      if (values.length < 2)
        return <p className="text-xs text-muted-foreground">暂无走势数据</p>;
      return (
        <EChart
          option={{
            animation: false,
            grid: { left: 4, right: 4, top: 4, bottom: 4 },
            xAxis: {
              type: "category",
              show: false,
              data: values.map((_, index) => index),
            },
            yAxis: { type: "value", show: false, scale: true },
            series: [
              {
                type: "line",
                data: values,
                smooth: true,
                showSymbol: false,
                lineStyle: { width: 2, color: "#2563eb" },
                areaStyle: { color: "#2563eb", opacity: 0.1 },
              },
            ],
          }}
          height={80}
          ariaLabel="近期走势"
        />
      );
    }
    case "LineChart":
      return <LineChart points={resolve(component.points)} />;
    case "MultiLineChart":
      return (
        <MultiLineChart
          series={resolve(component.series)}
          ariaLabel={
            displayValue(resolve(component.ariaLabel)) || "多策略对比曲线"
          }
        />
      );
    case "TradeChart":
      return (
        <TradeChart
          pricePoints={resolve(component.pricePoints)}
          strategies={resolve(component.strategies)}
          ariaLabel={
            displayValue(resolve(component.ariaLabel)) || "回测策略实际买卖点"
          }
        />
      );
    case "BenchmarkChart":
      return (
        <BenchmarkChart
          strategies={resolve(component.strategies)}
          assetLabel={displayValue(resolve(component.assetLabel)) || "当前标的"}
          marketLabel={
            displayValue(resolve(component.marketLabel)) || "同期大盘"
          }
          ariaLabel={
            displayValue(resolve(component.ariaLabel)) ||
            "当前标的与同期大盘同策略净值对比"
          }
        />
      );
    case "ArtifactLink": {
      const previewUrl = safeArtifactUrl(
        resolve(component.previewUrl),
        "preview",
      );
      const downloadUrl = safeArtifactUrl(
        resolve(component.downloadUrl),
        "download",
      );
      const size = Number(resolve(component.size) || 0);
      const name = displayValue(resolve(component.name)) || "研究成果";
      return (
        <div className="flex min-w-0 flex-wrap items-center gap-2 rounded-lg border border-border/70 bg-background/50 px-3 py-2 text-xs">
          <span className="min-w-0 flex-1 truncate font-medium" title={name}>
            {name}
          </span>
          {size > 0 && (
            <span className="text-muted-foreground">
              {(size / 1024).toFixed(1)} KB
            </span>
          )}
          {previewUrl && (
            <a
              className="rounded-md border px-2 py-1 text-primary hover:bg-muted"
              href={previewUrl}
              target="_blank"
              rel="noreferrer"
            >
              预览
            </a>
          )}
          {downloadUrl && (
            <a
              className="rounded-md bg-primary px-2 py-1 text-primary-foreground hover:opacity-90"
              href={downloadUrl}
              download
            >
              下载
            </a>
          )}
          {!previewUrl && !downloadUrl && (
            <span className="text-destructive">链接无效</span>
          )}
        </div>
      );
    }
    case "Activity": {
      const status = displayValue(resolve(component.status));
      const error = displayValue(resolve(component.error));
      const failed = status === "failed";
      return (
        <div
          className={cn(
            "flex flex-wrap items-center gap-2 rounded-lg border px-3 py-2 text-xs",
            failed
              ? "border-destructive/50 bg-destructive/5 text-destructive"
              : "border-border/70 bg-background/40 text-muted-foreground",
          )}
          title={error || undefined}
        >
          <span
            className={cn(
              "h-2 w-2 rounded-full",
              status === "completed"
                ? "bg-green-500"
                : failed
                  ? "bg-destructive"
                  : "animate-pulse bg-primary",
            )}
          />
          <span>已调用数据工具：{displayValue(resolve(component.name))}</span>
          <span className="ml-auto">
            {status === "completed"
              ? "完成"
              : status === "running"
                ? "执行中"
                : status}
          </span>
          {error && <span className="basis-full pl-4">原因：{error}</span>}
        </div>
      );
    }
    case "DataTable": {
      const rows = resolve(component.rows);
      const columns = Array.isArray(component.columns)
        ? (component.columns as Array<{ key: string; label: string }>)
        : [];
      if (!Array.isArray(rows)) return null;
      return (
        <div className="overflow-x-auto rounded-lg border">
          <table className="w-full min-w-[560px] text-left text-xs">
            <thead className="bg-muted/60 text-muted-foreground">
              <tr>
                {columns.map((column) => (
                  <th
                    key={column.key}
                    className="whitespace-nowrap px-3 py-2 font-medium"
                  >
                    {column.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, index) => (
                <tr key={index} className="border-t border-border/70">
                  {columns.map((column) => (
                    <td
                      key={column.key}
                      className="whitespace-nowrap px-3 py-2"
                    >
                      {displayValue(
                        (row as Record<string, unknown>)[column.key],
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    }
    case "Button": {
      const event = (
        component.action as
          | { event?: { name?: string; context?: Record<string, unknown> } }
          | undefined
      )?.event;
      return (
        <Button
          onClick={() =>
            event?.name &&
            onAction?.({
              name: event.name,
              surfaceId: surface.id,
              context: resolveContext(event.context || {}, surface.model),
            })
          }
        >
          {component.text
            ? displayValue(resolve(component.text))
            : children.map((child) => (
                <RenderComponent
                  key={child.id}
                  component={child}
                  surface={surface}
                  scope={scope}
                  onModelChange={onModelChange}
                  onAction={onAction}
                />
              ))}
        </Button>
      );
    }
    case "TextField": {
      const binding = component.value as Binding | undefined;
      const path = binding && "path" in binding ? binding.path : "";
      return (
        <Input
          aria-label={String(component.label || "")}
          placeholder={String(component.label || "")}
          value={displayValue(resolve(component.value))}
          onChange={(event) => path && onModelChange(path, event.target.value)}
        />
      );
    }
    case "ChoicePicker": {
      const binding = component.value as Binding | undefined;
      const path = binding && "path" in binding ? binding.path : "";
      const options = Array.isArray(component.options)
        ? (component.options as Array<{ label: string; value: string }>)
        : [];
      return (
        <select
          className="h-9 rounded-md border border-input bg-background px-3 text-sm"
          value={displayValue(resolve(component.value))}
          onChange={(event) => path && onModelChange(path, event.target.value)}
        >
          {options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      );
    }
    default:
      return (
        <div className="rounded-md border border-dashed px-3 py-2 text-xs text-muted-foreground">
          暂不支持的 A2UI 组件：{component.component}
        </div>
      );
  }
}

function resolveContext(context: Record<string, unknown>, model: unknown) {
  return Object.fromEntries(
    Object.entries(context).map(([key, value]) => [
      key,
      resolveValue(value, model),
    ]),
  );
}

function MarkdownContent({ text }: { text: string }) {
  const lines = text.split("\n");
  const blocks: ReactNode[] = [];
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) {
      index += 1;
      continue;
    }
    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      const level = heading[1].length;
      const className =
        level <= 2
          ? "mt-3 text-base font-semibold"
          : "mt-2 text-sm font-semibold";
      blocks.push(
        <div key={index} className={className}>
          {inlineMarkdown(heading[2])}
        </div>,
      );
      index += 1;
      continue;
    }
    if (
      /^\s*\|/.test(line) &&
      index + 1 < lines.length &&
      /^\s*\|?\s*:?-{3,}/.test(lines[index + 1])
    ) {
      const tableLines: string[] = [line];
      index += 2;
      while (index < lines.length && /^\s*\|/.test(lines[index])) {
        tableLines.push(lines[index]);
        index += 1;
      }
      blocks.push(<MarkdownTable key={`table-${index}`} lines={tableLines} />);
      continue;
    }
    if (/^\s*[-*]\s+/.test(line)) {
      const items: string[] = [];
      while (index < lines.length && /^\s*[-*]\s+/.test(lines[index])) {
        items.push(lines[index].replace(/^\s*[-*]\s+/, ""));
        index += 1;
      }
      blocks.push(
        <ul key={`ul-${index}`} className="ml-4 list-disc space-y-1 text-sm">
          {items.map((item, itemIndex) => (
            <li key={itemIndex}>{inlineMarkdown(item)}</li>
          ))}
        </ul>,
      );
      continue;
    }
    if (/^\s*\d+\.\s+/.test(line)) {
      const items: string[] = [];
      while (index < lines.length && /^\s*\d+\.\s+/.test(lines[index])) {
        items.push(lines[index].replace(/^\s*\d+\.\s+/, ""));
        index += 1;
      }
      blocks.push(
        <ol key={`ol-${index}`} className="ml-4 list-decimal space-y-1 text-sm">
          {items.map((item, itemIndex) => (
            <li key={itemIndex}>{inlineMarkdown(item)}</li>
          ))}
        </ol>,
      );
      continue;
    }
    const paragraph: string[] = [line];
    index += 1;
    while (
      index < lines.length &&
      lines[index].trim() &&
      !/^(#{1,6})\s|^\s*[-*]\s+|^\s*\d+\.\s+|^\s*\|/.test(lines[index])
    ) {
      paragraph.push(lines[index]);
      index += 1;
    }
    blocks.push(
      <p
        key={`p-${index}`}
        className="whitespace-pre-wrap text-sm leading-relaxed"
      >
        {inlineMarkdown(paragraph.join("\n"))}
      </p>,
    );
  }
  return <div className="space-y-3">{blocks}</div>;
}

function MarkdownTable({ lines }: { lines: string[] }) {
  const cells = (line: string) =>
    line
      .split("|")
      .slice(1, -1)
      .map((cell) => cell.trim());
  const headers = cells(lines[0]);
  return (
    <div className="overflow-x-auto rounded-lg border">
      <table className="w-full text-left text-xs">
        <thead className="bg-muted/60">
          <tr>
            {headers.map((header, index) => (
              <th
                key={index}
                className="whitespace-nowrap px-3 py-2 font-medium"
              >
                {inlineMarkdown(header)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {lines.slice(1).map((line, rowIndex) => (
            <tr key={rowIndex} className="border-t border-border/70">
              {cells(line).map((cell, cellIndex) => (
                <td key={cellIndex} className="whitespace-nowrap px-3 py-2">
                  {inlineMarkdown(cell)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function inlineMarkdown(text: string): ReactNode {
  const linkPattern =
    /(\[[^\]\n]+\]\((https?:\/\/[^)\s]+)\)|https?:\/\/[^\s<]+)/g;
  const nodes: ReactNode[] = [];
  let cursor = 0;
  let match: RegExpExecArray | null;
  let key = 0;

  while ((match = linkPattern.exec(text))) {
    if (match.index > cursor) {
      nodes.push(...inlineMarkdownTokens(text.slice(cursor, match.index), key));
      key += 1;
    }

    const markdownLabel = match[1];
    const rawUrl = markdownLabel ? match[2] : match[0];
    const { url, suffix } = trimUrlSuffix(rawUrl);
    const label = markdownLabel
      ? markdownLabel.slice(1, markdownLabel.lastIndexOf("]("))
      : url;
    nodes.push(
      <a
        key={`link-${key}`}
        href={url}
        target="_blank"
        rel="noreferrer"
        className="break-all text-primary underline underline-offset-2 hover:opacity-80"
      >
        {markdownLabel ? inlineMarkdownTokens(label, key) : label}
      </a>,
    );
    key += 1;
    if (suffix) nodes.push(<span key={`suffix-${key}`}>{suffix}</span>);
    key += 1;
    cursor = match.index + match[0].length;
  }

  if (cursor < text.length) {
    nodes.push(...inlineMarkdownTokens(text.slice(cursor), key));
  }
  return nodes.length ? nodes : inlineMarkdownTokens(text, 0);
}

function inlineMarkdownTokens(text: string, keyOffset: number): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((part, index) => {
    const key = keyOffset + index;
    if (part.startsWith("**") && part.endsWith("**"))
      return <strong key={key}>{part.slice(2, -2)}</strong>;
    if (part.startsWith("`") && part.endsWith("`"))
      return (
        <code key={key} className="rounded bg-muted px-1">
          {part.slice(1, -1)}
        </code>
      );
    return <span key={key}>{part}</span>;
  });
}

function trimUrlSuffix(rawUrl: string): { url: string; suffix: string } {
  const match = rawUrl.match(/^(.*?)([),.，。；：！？、》）】]*)$/u);
  const url = match?.[1] || rawUrl;
  return { url, suffix: match?.[2] || "" };
}
