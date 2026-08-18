import { useState } from "react";
import { runBacktest, runBacktestExperiment } from "@/api";
import type { AssetType, BacktestMode, BacktestResult } from "@/types";

export function BacktestPage() {
  const [mode, setMode] = useState<BacktestMode>("portfolio");
  const [assetType, setAssetType] = useState<AssetType>("etf");
  const [tickers, setTickers] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  const symbols = tickers.split(",").map((item) => item.trim()).filter(Boolean);
  const payload = {
    mode, asset_type: assetType, start_date: startDate, end_date: endDate,
    ...(symbols.length > 1 ? { tickers: symbols } : { ticker: symbols[0] }),
  };

  const run = async (withAgent: boolean) => {
    setRunning(true); setError(null);
    try {
      const response = withAgent
        ? (await runBacktestExperiment({ ...payload, objective: "设计一个短中线、控制回撤的 ETF 组合策略" })).result
        : await runBacktest(payload);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "回测失败");
    } finally {
      setRunning(false);
    }
  };

  return <main className="space-y-6 p-6">
    <h1 className="text-2xl font-bold">回测实验</h1>
    <p className="text-sm text-muted-foreground">支持单标的、标的池和组合账户回测。</p>
    <section className="space-y-4 rounded-xl border bg-card p-5">
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <label className="space-y-2 text-sm">模式<select className="flex h-10 w-full rounded-md border bg-transparent px-3" value={mode} onChange={(e) => setMode(e.target.value as BacktestMode)}><option value="single">单标的</option><option value="pool">标的池</option><option value="portfolio">组合</option></select></label>
        <label className="space-y-2 text-sm">资产类型<select className="flex h-10 w-full rounded-md border bg-transparent px-3" value={assetType} onChange={(e) => setAssetType(e.target.value as AssetType)}><option value="etf">ETF</option><option value="lof">LOF</option><option value="stock">股票</option></select></label>
        <label className="space-y-2 text-sm md:col-span-2">标的代码（逗号分隔）<input className="flex h-10 w-full rounded-md border bg-transparent px-3" placeholder="510300,159915,512100" value={tickers} onChange={(e) => setTickers(e.target.value)} /></label>
        <label className="space-y-2 text-sm">开始日期<input className="flex h-10 w-full rounded-md border bg-transparent px-3" type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} /></label>
        <label className="space-y-2 text-sm">结束日期<input className="flex h-10 w-full rounded-md border bg-transparent px-3" type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} /></label>
      </div>
      <div className="flex gap-3"><button className="rounded-md bg-primary px-4 py-2 text-primary-foreground disabled:opacity-50" disabled={running || !symbols.length || !startDate || !endDate} onClick={() => run(false)}>{running ? "回测中…" : "运行回测"}</button><button className="rounded-md border px-4 py-2 disabled:opacity-50" disabled={running || !symbols.length || !startDate || !endDate} onClick={() => run(true)}>Agent 设计并保存实验</button></div>
    </section>
    {error && <p className="rounded-md border border-destructive p-4 text-sm text-destructive">{error}</p>}
    {result && <section className="grid gap-4 md:grid-cols-4"><Metric title="最终资产" value={`¥${result.final_value.toLocaleString()}`} /><Metric title="总收益率" value={`${(result.total_return * 100).toFixed(2)}%`} /><Metric title="最大回撤" value={`${(result.max_drawdown * 100).toFixed(2)}%`} /><Metric title="交易次数" value={String(result.total_trades)} /></section>}
  </main>;
}

function Metric({ title, value }: { title: string; value: string }) {
  return <div className="rounded-xl border bg-card p-4"><p className="text-xs text-muted-foreground">{title}</p><p className="mt-2 text-xl font-bold">{value}</p></div>;
}
