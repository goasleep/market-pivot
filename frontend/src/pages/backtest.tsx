import { useState } from "react";
import { deployBacktestExperiment, runBacktest, runBacktestExperiment } from "@/api";
import type { AssetType, BacktestExperimentResult, BacktestMode, BacktestResult, StrategyDeployment } from "@/types";

export function BacktestPage() {
  const [mode, setMode] = useState<BacktestMode>("portfolio");
  const [assetType, setAssetType] = useState<AssetType>("etf");
  const [tickers, setTickers] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [experiment, setExperiment] = useState<BacktestExperimentResult | null>(null);
  const [deployment, setDeployment] = useState<StrategyDeployment | null>(null);
  const [accountId, setAccountId] = useState("");
  const [accountName, setAccountName] = useState("");
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
      if (withAgent) {
        const response = await runBacktestExperiment({
          ...payload,
          objective: `设计一个短中线、控制回撤的 ${assetType.toUpperCase()} ${mode === "portfolio" ? "组合" : "交易"}策略`,
        });
        setExperiment(response);
        setResult(response.result);
        setAccountId(`paper_${response.experiment_id.replace(/[^A-Za-z0-9_-]/g, "_").slice(-20)}`);
        setAccountName(`${String(response.strategy_spec.name || "Agent 策略")} 模拟盘`);
      } else {
        setExperiment(null);
        setResult(await runBacktest(payload));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "回测失败");
    } finally {
      setRunning(false);
    }
  };

  const deploy = async () => {
    if (!experiment || !accountId.trim()) return;
    setRunning(true); setError(null);
    try {
      setDeployment(await deployBacktestExperiment(experiment.experiment_id, {
        account_id: accountId.trim(),
        account_name: accountName.trim() || undefined,
        mode: "confirm",
      }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "部署失败");
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
    {experiment && <section className="space-y-4 rounded-xl border bg-card p-5"><div><h2 className="font-semibold">部署到独立模拟盘</h2><p className="text-sm text-muted-foreground">实验 {experiment.experiment_id} 的策略将以不可变快照部署；默认逐单确认，可同时创建多个账户。</p></div><div className="grid gap-3 md:grid-cols-2"><label className="space-y-2 text-sm">账户 ID<input className="flex h-10 w-full rounded-md border bg-transparent px-3" value={accountId} onChange={(e) => setAccountId(e.target.value)} /></label><label className="space-y-2 text-sm">账户名称<input className="flex h-10 w-full rounded-md border bg-transparent px-3" value={accountName} onChange={(e) => setAccountName(e.target.value)} /></label></div><button className="rounded-md bg-primary px-4 py-2 text-primary-foreground disabled:opacity-50" disabled={running || !accountId.trim()} onClick={deploy}>创建并启用模拟盘</button>{deployment && <p className="text-sm text-emerald-700">已部署 {deployment.strategy_name} → <a className="underline" href={`/automation/${deployment.account_id}`}>{deployment.account_id}</a></p>}</section>}
  </main>;
}

function Metric({ title, value }: { title: string; value: string }) {
  return <div className="rounded-xl border bg-card p-4"><p className="text-xs text-muted-foreground">{title}</p><p className="mt-2 text-xl font-bold">{value}</p></div>;
}
