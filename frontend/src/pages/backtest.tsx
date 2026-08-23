import { useEffect, useState } from "react";
import { getStrategies, runBacktest, type StrategyInfo } from "@/api";
import type {
  AssetType,
  BacktestMode,
  BacktestResult,
} from "@/types";
import {
  ArrowRight,
  BarChart3,
  FlaskConical,
  LineChart,
  Shield,
  Wallet,
} from "lucide-react";
import { PageHeader, PageShell, MetricCard } from "@/components/layout/Page";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export function BacktestPage() {
  const [mode, setMode] = useState<BacktestMode>("portfolio");
  const [assetType, setAssetType] = useState<AssetType>("etf");
  const [strategy, setStrategy] = useState("bull_trend");
  const [strategies, setStrategies] = useState<StrategyInfo[]>([]);
  const [tickers, setTickers] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    getStrategies().then(setStrategies).catch(() => setStrategies([]));
  }, []);

  const symbols = tickers
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  const payload = {
    mode,
    asset_type: assetType,
    start_date: startDate,
    end_date: endDate,
    strategy,
    ...(symbols.length > 1 ? { tickers: symbols } : { ticker: symbols[0] }),
  };

  const run = async () => {
    setRunning(true);
    setError(null);
    try {
      setResult(await runBacktest(payload));
    } catch (err) {
      setError(err instanceof Error ? err.message : "回测失败");
    } finally {
      setRunning(false);
    }
  };

  return (
    <PageShell>
      <PageHeader
        eyebrow="Strategy Lab"
        title="策略回测"
        description="在一致的数据快照和成交规则下验证单标的、标的池或组合策略。Agent 研究请从对话入口发起。"
        icon={FlaskConical}
      />
      <Card>
        <CardHeader>
          <CardTitle>实验参数</CardTitle>
          <CardDescription>
            本页只执行确定性回测，不调用 Agent。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <label className="space-y-2 text-sm">
              <Label>模式</Label>
              <select
                className="field-surface"
                value={mode}
                onChange={(e) => setMode(e.target.value as BacktestMode)}
              >
                <option value="single">单标的</option>
                <option value="pool">标的池</option>
                <option value="portfolio">组合</option>
              </select>
            </label>
            <label className="space-y-2 text-sm">
              <Label>资产类型</Label>
              <select
                className="field-surface"
                value={assetType}
                onChange={(e) => setAssetType(e.target.value as AssetType)}
              >
                <option value="etf">ETF</option>
                <option value="lof">LOF</option>
                <option value="stock">股票</option>
              </select>
            </label>
            <label className="space-y-2 text-sm md:col-span-2">
              <Label>标的代码（逗号分隔）</Label>
              <Input
                placeholder="510300,159915,512100"
                value={tickers}
                onChange={(e) => setTickers(e.target.value)}
              />
            </label>
            <label className="space-y-2 text-sm">
              <Label>确定性策略</Label>
              <select
                className="field-surface"
                value={strategy}
                onChange={(e) => setStrategy(e.target.value)}
              >
                {strategies.length === 0 && (
                  <option value="bull_trend">牛市趋势策略</option>
                )}
                {strategies.map((item) => (
                  <option key={item.name} value={item.name}>
                    {item.display_name}
                  </option>
                ))}
              </select>
            </label>
            <label className="space-y-2 text-sm">
              <Label>开始日期</Label>
              <Input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
              />
            </label>
            <label className="space-y-2 text-sm">
              <Label>结束日期</Label>
              <Input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
              />
            </label>
          </div>
          <div className="flex flex-wrap gap-3">
            <Button
              disabled={running || !symbols.length || !startDate || !endDate}
              onClick={run}
            >
              <LineChart className="h-4 w-4" />
              {running ? "回测中…" : "运行回测"}
            </Button>
          </div>
        </CardContent>
      </Card>
      {error && (
        <p className="rounded-xl border border-destructive/30 bg-destructive/5 p-4 text-sm text-destructive">
          {error}
        </p>
      )}
      {result && (
        <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <MetricCard
            label="最终资产"
            value={`¥${result.final_value.toLocaleString()}`}
            icon={Wallet}
          />
          <MetricCard
            label="总收益率"
            value={`${(result.total_return * 100).toFixed(2)}%`}
            icon={BarChart3}
            tone={result.total_return < 0 ? "negative" : "positive"}
          />
          <MetricCard
            label="最大回撤"
            value={`${(result.max_drawdown * 100).toFixed(2)}%`}
            icon={Shield}
            tone="warning"
          />
          <MetricCard
            label="交易次数"
            value={String(result.total_trades)}
            icon={ArrowRight}
          />
        </section>
      )}
    </PageShell>
  );
}
