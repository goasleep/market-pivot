import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { executeAnalysisInSimulation, streamAnalysis } from "@/api";
import type { AnalysisResult, AssetType, SSEProgress } from "@/types";
import { ArtifactCard } from "@/components/chat/ArtifactCard";
import { Loader2, Search, Send } from "lucide-react";

const STAGE_LABELS: Record<string, string> = {
  market_data: "市场数据获取",
  technical: "技术分析",
  fundamentals: "基本面分析",
  sentiment: "舆情分析",
  debate: "多空辩论",
  risk: "风控评估",
  portfolio: "投资组合决策",
};

export function AnalysisPage() {
  const [ticker, setTicker] = useState("");
  const [assetType, setAssetType] = useState<AssetType>("stock");
  const [holdingPeriodDays, setHoldingPeriodDays] = useState("20");
  const [availableCapital, setAvailableCapital] = useState("");
  const [maxLossPct, setMaxLossPct] = useState("5");
  const [currentPositionPct, setCurrentPositionPct] = useState("0");
  const [entryPrice, setEntryPrice] = useState("");
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<SSEProgress[]>([]);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [executing, setExecuting] = useState(false);

  const handleRun = () => {
    if (!ticker.trim()) return;
    setRunning(true);
    setProgress([]);
    setResult(null);
    setError(null);

    streamAnalysis(
      ticker.trim(),
      {
        asset_type: assetType,
        holding_period_days: Number(holdingPeriodDays) || undefined,
        available_capital: Number(availableCapital) || undefined,
        max_loss_pct: Number(maxLossPct) / 100 || undefined,
        current_position_pct: Number(currentPositionPct) / 100 || undefined,
        entry_price: Number(entryPrice) || undefined,
      },
      (data) => setProgress((prev) => [...prev, data]),
      (data) => {
        setResult(data);
        setRunning(false);
      },
      () => {
        setError("连接中断");
        setRunning(false);
      }
    );
  };

  const progressPct = running ? (progress.length / 7) * 100 : result ? 100 : 0;

  const handleExecute = async () => {
    if (!result || result.decision === "hold") return;
    if (!window.confirm(`确认将 ${result.asset_type} ${result.ticker} 的 ${result.decision} 决策提交到模拟账户吗？`)) return;
    setExecuting(true);
    setError(null);
    try {
      await executeAnalysisInSimulation("default", result);
      setError("Agent 决策已提交到 default 模拟账户");
    } catch (err) {
      setError(err instanceof Error ? err.message : "模拟执行失败");
    } finally {
      setExecuting(false);
    }
  };

  return (
    <div className="space-y-6 p-6">
      <div>
        <h1 className="text-2xl font-bold">Agent Analysis</h1>
        <p className="text-sm text-muted-foreground">输入股票或场内基金代码，启动多智能体协作分析</p>
      </div>

      {/* Input */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">标的分析</CardTitle>
          <CardDescription>股票示例 600519；ETF 示例 510300；LOF 示例 166009</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            <div className="space-y-2">
              <Label htmlFor="asset-type">标的类型</Label>
              <select id="asset-type" className="flex h-10 w-full rounded-md border border-input bg-transparent px-3 text-sm" value={assetType} onChange={(e) => setAssetType(e.target.value as AssetType)} disabled={running}>
                <option value="stock">股票</option>
                <option value="etf">ETF</option>
                <option value="lof">LOF</option>
              </select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="ticker">股票 / 基金代码</Label>
              <Input
                id="ticker"
                placeholder={assetType === "stock" ? "000737" : assetType === "etf" ? "510300" : "166009"}
                value={ticker}
                onChange={(e) => setTicker(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && !running && handleRun()}
                disabled={running}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="holding-period">预计持有天数</Label>
              <Input id="holding-period" type="number" min="1" value={holdingPeriodDays} onChange={(e) => setHoldingPeriodDays(e.target.value)} disabled={running} />
            </div>
            <div className="space-y-2">
              <Label htmlFor="capital">可投入资金（可选）</Label>
              <Input id="capital" type="number" placeholder="100000" value={availableCapital} onChange={(e) => setAvailableCapital(e.target.value)} disabled={running} />
            </div>
            <div className="space-y-2">
              <Label htmlFor="max-loss">最大可接受亏损 (%)</Label>
              <Input id="max-loss" type="number" min="0" max="100" value={maxLossPct} onChange={(e) => setMaxLossPct(e.target.value)} disabled={running} />
            </div>
            <div className="space-y-2">
              <Label htmlFor="position">当前仓位 (%)</Label>
              <Input id="position" type="number" min="0" max="100" value={currentPositionPct} onChange={(e) => setCurrentPositionPct(e.target.value)} disabled={running} />
            </div>
            <div className="space-y-2">
              <Label htmlFor="entry-price">持仓成本价（可选）</Label>
              <Input id="entry-price" type="number" placeholder="无持仓可留空" value={entryPrice} onChange={(e) => setEntryPrice(e.target.value)} disabled={running} />
            </div>
          </div>
          <div className="mt-4">
            <Button onClick={handleRun} disabled={running || !ticker.trim()}>
              {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
              {running ? "分析中..." : "开始分析"}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Progress */}
      {(running || progress.length > 0) && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Agent 工作进度</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <Progress value={progressPct} />
            <div className="space-y-2">
              {progress.map((p, i) => (
                <div key={i} className="flex items-center gap-3 text-sm">
                  <Badge variant="success" className="h-5 w-5 justify-center p-0 text-xs">
                    {i + 1}
                  </Badge>
                  <span className="text-muted-foreground">{STAGE_LABELS[p.stage] || p.stage}</span>
                  <span>{p.message}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Result */}
      {result && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">分析结果 - {result.ticker}</CardTitle>
              <Badge
                variant={
                  result.decision === "buy" ? "success" :
                  result.decision === "sell" ? "destructive" : "secondary"
                }
                className="text-sm"
              >
                {result.decision === "buy" ? "买入" : result.decision === "sell" ? "卖出" : "持有"}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
              <div>
                <Label className="text-xs text-muted-foreground">置信度</Label>
                <p className="text-lg font-semibold">{(result.confidence * 100).toFixed(1)}%</p>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">建议入场价</Label>
                <p className="text-lg font-semibold">
                  {result.entry_price ? `¥${result.entry_price}` : "—"}
                </p>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">目标价</Label>
                <p className="text-lg font-semibold">
                  {result.target_price ? `¥${result.target_price}` : "—"}
                </p>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">止损价</Label>
                <p className="text-lg font-semibold">
                  {result.stop_loss ? `¥${result.stop_loss}` : "—"}
                </p>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">止盈价</Label>
                <p className="text-lg font-semibold">
                  {result.take_profit ? `¥${result.take_profit}` : "—"}
                </p>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">建议仓位</Label>
                <p className="text-lg font-semibold">
                  {result.position_size ? `${(result.position_size * 100).toFixed(0)}%` : "—"}
                </p>
              </div>
            </div>
            <Separator />
            <div className="flex justify-end">
              <Button variant="outline" onClick={handleExecute} disabled={executing || result.decision === "hold"}>
                {executing ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Send className="mr-2 h-4 w-4" />}
                {result.decision === "hold" ? "持有，不生成订单" : "执行到模拟盘"}
              </Button>
            </div>
            <Separator />
            <div>
              <Label className="text-xs text-muted-foreground">决策推理</Label>
              <p className="mt-1 text-sm leading-relaxed">{result.reasoning}</p>
            </div>
            {result.data_status && (
              <div className="rounded-md border bg-muted/30 p-3 text-xs text-muted-foreground">
                数据状态：{Object.entries(result.data_status).map(([key, value]) => `${key}=${String(value)}`).join(" · ")}
              </div>
            )}
            {result.dashboard && (
              <div className="grid gap-3 text-sm md:grid-cols-2">
                <div className="rounded-md border p-3"><Label className="text-xs text-muted-foreground">趋势与量价</Label><p>{result.dashboard.data_perspective.trend_status || "—"} · {result.dashboard.data_perspective.volume_analysis || "—"}</p></div>
                <div className="rounded-md border p-3"><Label className="text-xs text-muted-foreground">执行计划</Label><p>{result.dashboard.battle_plan.position_strategy || "—"}</p><p className="text-xs text-muted-foreground">{result.dashboard.battle_plan.action_items.join("；")}</p></div>
                <div className="rounded-md border p-3 md:col-span-2"><Label className="text-xs text-muted-foreground">价格依据</Label><p>入场：{result.dashboard.battle_plan.entry_explanation || "—"}</p><p>止损：{result.dashboard.battle_plan.stop_loss_explanation || "—"}</p><p>止盈：{result.dashboard.battle_plan.take_profit_explanation || "—"}</p></div>
                <div className="rounded-md border p-3 md:col-span-2"><Label className="text-xs text-muted-foreground">本次策略</Label><p>{result.dashboard.strategy_plan.name || "—"} · {result.dashboard.strategy_plan.thesis || "—"}</p><p className="text-xs text-muted-foreground">入场条件：{result.dashboard.strategy_plan.entry_conditions.join("；") || "—"}</p><p className="text-xs text-muted-foreground">退出条件：{result.dashboard.strategy_plan.exit_conditions.join("；") || "—"}</p></div>
                <div className="rounded-md border p-3 md:col-span-2"><Label className="text-xs text-muted-foreground">阶段计划</Label><p>{result.dashboard.phase_decision.pre_market || "—"} {result.dashboard.phase_decision.intraday || "—"} {result.dashboard.phase_decision.post_market || "—"}</p></div>
              </div>
            )}
            {result.artifacts && result.artifacts.length > 0 && (
              <div className="space-y-2">
                <Label className="text-xs text-muted-foreground">分析产物</Label>
                <div className="flex flex-wrap gap-3">
                  {result.artifacts.map((artifact) => (
                    <ArtifactCard key={artifact.artifact_id} artifact={artifact} />
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {error && (
        <Card className="border-destructive">
          <CardContent className="pt-6">
            <p className="text-sm text-destructive">{error}</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
