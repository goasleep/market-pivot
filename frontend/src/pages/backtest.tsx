import { useState } from "react";
import {
  deployBacktestExperiment,
  runBacktest,
  runBacktestExperiment,
} from "@/api";
import type {
  AssetType,
  BacktestExperimentResult,
  BacktestMode,
  BacktestResult,
  StrategyDeployment,
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
  const [tickers, setTickers] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [experiment, setExperiment] = useState<BacktestExperimentResult | null>(
    null,
  );
  const [deployment, setDeployment] = useState<StrategyDeployment | null>(null);
  const [accountId, setAccountId] = useState("");
  const [accountName, setAccountName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  const symbols = tickers
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  const payload = {
    mode,
    asset_type: assetType,
    start_date: startDate,
    end_date: endDate,
    ...(symbols.length > 1 ? { tickers: symbols } : { ticker: symbols[0] }),
  };

  const run = async (withAgent: boolean) => {
    setRunning(true);
    setError(null);
    try {
      if (withAgent) {
        const response = await runBacktestExperiment({
          ...payload,
          objective: `设计一个短中线、控制回撤的 ${assetType.toUpperCase()} ${mode === "portfolio" ? "组合" : "交易"}策略`,
        });
        setExperiment(response);
        setResult(response.result);
        setAccountId(
          `paper_${response.experiment_id.replace(/[^A-Za-z0-9_-]/g, "_").slice(-20)}`,
        );
        setAccountName(
          `${String(response.strategy_spec.name || "Agent 策略")} 模拟盘`,
        );
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
    setRunning(true);
    setError(null);
    try {
      setDeployment(
        await deployBacktestExperiment(experiment.experiment_id, {
          account_id: accountId.trim(),
          account_name: accountName.trim() || undefined,
          mode: "confirm",
        }),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "部署失败");
    } finally {
      setRunning(false);
    }
  };

  return (
    <PageShell>
      <PageHeader
        eyebrow="Strategy Lab"
        title="回测与策略实验"
        description="在一致的数据快照和成交规则下验证单标的、标的池或组合策略，再决定是否部署到独立模拟盘。"
        icon={FlaskConical}
      />
      <Card>
        <CardHeader>
          <CardTitle>实验参数</CardTitle>
          <CardDescription>
            Agent 实验会设计并保存策略；普通回测直接使用当前参数执行。
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
              onClick={() => run(false)}
            >
              <LineChart className="h-4 w-4" />
              {running ? "回测中…" : "运行回测"}
            </Button>
            <Button
              variant="outline"
              disabled={running || !symbols.length || !startDate || !endDate}
              onClick={() => run(true)}
            >
              <FlaskConical className="h-4 w-4" />
              Agent 设计并保存实验
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
      {experiment && (
        <Card>
          <CardHeader>
            <CardTitle>部署到独立模拟盘</CardTitle>
            <CardDescription>
              实验 {experiment.experiment_id}{" "}
              的策略将以不可变快照部署；默认逐单确认，可同时创建多个账户。
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-3 md:grid-cols-2">
              <label className="space-y-2 text-sm">
                <Label>账户 ID</Label>
                <Input
                  value={accountId}
                  onChange={(e) => setAccountId(e.target.value)}
                />
              </label>
              <label className="space-y-2 text-sm">
                <Label>账户名称</Label>
                <Input
                  value={accountName}
                  onChange={(e) => setAccountName(e.target.value)}
                />
              </label>
            </div>
            <Button disabled={running || !accountId.trim()} onClick={deploy}>
              创建并启用模拟盘
              <ArrowRight className="h-4 w-4" />
            </Button>
            {deployment && (
              <p className="rounded-xl border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-700">
                已部署 {deployment.strategy_name} →{" "}
                <a
                  className="font-medium underline"
                  href={`/automation/${deployment.account_id}`}
                >
                  {deployment.account_id}
                </a>
              </p>
            )}
          </CardContent>
        </Card>
      )}
    </PageShell>
  );
}
