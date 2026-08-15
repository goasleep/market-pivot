import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { TrendingUp, Wallet, Activity, Brain, Plus, X } from "lucide-react";
import { getAutomationRuns, getAutomationTask, getMarketQuote, getPortfolio, getStrategies, getSystemStatus, openSimulationStream, type StrategyInfo } from "@/api";
import type { AgentRunSummary, AssetType, AutomationTask, Portfolio } from "@/types";

interface WatchItem {
  ticker: string;
  assetType: AssetType;
  alertPrice?: number;
}

export function DashboardPage() {
  const [strategies, setStrategies] = useState<StrategyInfo[]>([]);
  const [breakers, setBreakers] = useState<Record<string, string>>({});
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [automation, setAutomation] = useState<AutomationTask | null>(null);
  const [runs, setRuns] = useState<AgentRunSummary[]>([]);
  const [watchlist, setWatchlist] = useState<WatchItem[]>(() => {
    try {
      const parsed = JSON.parse(localStorage.getItem("a-share-agent:watchlist") || "[]");
      return parsed.map((item: WatchItem | string) => typeof item === "string" ? { ticker: item, assetType: "stock" } : item);
    } catch {
      return [];
    }
  });
  const [watchInput, setWatchInput] = useState("");
  const [watchType, setWatchType] = useState<AssetType>("etf");
  const [alertPrice, setAlertPrice] = useState("");
  const [quotes, setQuotes] = useState<Record<string, Record<string, unknown>>>({});

  useEffect(() => {
    localStorage.setItem("a-share-agent:watchlist", JSON.stringify(watchlist));
  }, [watchlist]);

  const addWatch = () => {
    const code = watchInput.trim().replace(/^(sh|sz|bj)/i, "").padStart(6, "0");
    if (!/^\d{6}$/.test(code) || watchlist.some((item) => item.ticker === code && item.assetType === watchType)) return;
    setWatchlist((items) => [...items, { ticker: code, assetType: watchType, alertPrice: Number(alertPrice) || undefined }]);
    setWatchInput("");
    setAlertPrice("");
  };

  const refreshWatchlist = async () => {
    const entries = await Promise.all(watchlist.map(async (item) => {
      try {
        const result = await getMarketQuote(item.ticker, item.assetType);
        return [item.ticker, result.quote] as const;
      } catch {
        return [item.ticker, {}] as const;
      }
    }));
    setQuotes(Object.fromEntries(entries));
  };

  useEffect(() => {
    getStrategies().then(setStrategies).catch(() => {});
    getSystemStatus().then(setBreakers).catch(() => {});
    const refresh = () => {
      getPortfolio().then(setPortfolio).catch(() => {});
      getAutomationTask().then(setAutomation).catch(() => {});
      getAutomationRuns().then(setRuns).catch(() => {});
    };
    refresh();
    const socket = openSimulationStream("default", (event) => {
      if (event.type.startsWith("agent.") || event.type === "daily.settled" || event.type === "automation.updated") refresh();
    });
    const timer = window.setInterval(refresh, 30_000);
    return () => {
      socket.close();
      window.clearInterval(timer);
    };
  }, []);

  const breakerColors: Record<string, "success" | "destructive" | "warning"> = {
    closed: "success",
    open: "destructive",
    half_open: "warning",
  };
  const breakerLabels: Record<string, string> = {
    closed: "Closed",
    open: "Open",
    half_open: "Half-Open",
  };
  return (
    <div className="space-y-6 p-6">
      <div>
        <h1 className="text-2xl font-bold">Dashboard</h1>
        <p className="text-sm text-muted-foreground">A 股 Agent 模拟交易系统概览</p>
      </div>

      {/* Stats cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">总资产</CardTitle>
            <Wallet className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">¥{(portfolio?.total_value || 0).toLocaleString(undefined, { maximumFractionDigits: 2 })}</div>
            <p className="text-xs text-muted-foreground">{portfolio?.current_date || "模拟账户"}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">累计收益</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${(portfolio?.total_pnl || 0) < 0 ? "text-destructive" : "text-emerald-600"}`}>¥{(portfolio?.total_pnl || 0).toFixed(2)}</div>
            <p className="text-xs text-muted-foreground">{((portfolio?.total_return_pct || 0) * 100).toFixed(2)}%</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">持仓股票</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{portfolio?.positions.length || 0}</div>
            <p className="text-xs text-muted-foreground">模拟账户持仓</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Agent 状态</CardTitle>
            <Brain className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <Badge variant={automation?.config.enabled ? "success" : "secondary"}>{automation?.config.enabled ? "运行中" : "未启用"}</Badge>
              <span className="text-xs text-muted-foreground">{automation?.config.mode || "observe"} · {runs[0]?.status || "暂无运行"}</span>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader><CardTitle className="text-base">自选标的</CardTitle><CardDescription>保存股票、ETF 或 LOF 代码，作为后续分析和提醒入口。</CardDescription></CardHeader>
        <CardContent className="space-y-3">
          <div className="grid gap-2 md:grid-cols-[1fr_120px_120px_auto]"><Input placeholder="例如 510300、600519" value={watchInput} onChange={(e) => setWatchInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && addWatch()} /><select className="h-10 rounded-md border bg-background px-3 text-sm" value={watchType} onChange={(e) => setWatchType(e.target.value as AssetType)}><option value="stock">股票</option><option value="etf">ETF</option><option value="lof">LOF</option></select><Input type="number" placeholder="提醒价（可选）" value={alertPrice} onChange={(e) => setAlertPrice(e.target.value)} /><Button onClick={addWatch}><Plus className="mr-1 h-4 w-4" />加入</Button></div>
          {watchlist.length === 0 ? <p className="text-sm text-muted-foreground">暂无自选标的</p> : <div className="space-y-2">{watchlist.map((item) => { const quote = quotes[item.ticker] || {}; const price = Number(quote.price || 0); const triggered = item.alertPrice && price > 0 && price >= item.alertPrice; return <div key={`${item.assetType}-${item.ticker}`} className="flex items-center justify-between rounded-md border px-3 py-2 text-sm"><div><span className="font-medium">{item.ticker}</span><span className="ml-2 text-xs text-muted-foreground">{item.assetType.toUpperCase()}</span>{item.alertPrice && <span className="ml-2 text-xs text-muted-foreground">提醒 ≥ ¥{item.alertPrice}</span>}</div><div className="flex items-center gap-3"><span className={triggered ? "font-semibold text-emerald-600" : "text-muted-foreground"}>{price ? `¥${price.toFixed(3)} ${Number(quote.pct_chg || 0) >= 0 ? "+" : ""}${Number(quote.pct_chg || 0).toFixed(2)}%` : "暂无行情"}</span><button aria-label={`移除 ${item.ticker}`} onClick={() => setWatchlist((items) => items.filter((current) => !(current.ticker === item.ticker && current.assetType === item.assetType)))}><X className="h-3 w-3" /></button></div></div> })}</div>}
          <div className="flex items-center justify-between"><p className="text-xs text-muted-foreground">自选列表保存在本机；提醒价在刷新行情后显示触发状态。</p><Button variant="outline" size="sm" onClick={refreshWatchlist} disabled={!watchlist.length}>刷新行情</Button></div>
        </CardContent>
      </Card>

      {/* Agent pipeline overview */}
      <Card>
        <CardHeader>
          <CardTitle>Agent 工作流</CardTitle>
          <CardDescription>多智能体协作流程概览</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-center gap-3">
            {[
              "市场数据", "技术分析", "基本面分析", "舆情分析",
              "多空辩论", "风控评估", "投资组合", "交易决策",
            ].map((stage, i) => (
              <div key={stage} className="flex items-center gap-3">
                <div className="flex items-center gap-2 rounded-md border px-3 py-1.5">
                  <span className="flex h-5 w-5 items-center justify-center rounded-full bg-muted text-xs">
                    {i + 1}
                  </span>
                  <span className="text-sm">{stage}</span>
                </div>
                {i < 7 && <span className="text-muted-foreground">→</span>}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Recent activity placeholder */}
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Trading Strategies</CardTitle>
            <CardDescription>YAML strategy engine</CardDescription>
          </CardHeader>
          <CardContent>
            {strategies.length === 0 ? (
              <p className="text-sm text-muted-foreground">Loading...</p>
            ) : (
              <div className="space-y-2">
                {strategies.map((s) => (
                  <div
                    key={s.name}
                    className="flex items-center justify-between rounded-md border px-3 py-2"
                  >
                    <div>
                      <span className="text-sm font-medium">{s.display_name}</span>
                      <span className="ml-2 text-xs text-muted-foreground">{s.category}</span>
                    </div>
                    {s.default_active && (
                      <Badge variant="success" className="text-xs">Active</Badge>
                    )}
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Circuit Breakers</CardTitle>
            <CardDescription>Data source health</CardDescription>
          </CardHeader>
          <CardContent>
            {Object.keys(breakers).length === 0 ? (
              <p className="text-sm text-muted-foreground">Loading...</p>
            ) : (
              <div className="space-y-2">
                {Object.entries(breakers).map(([name, state]) => (
                  <div
                    key={name}
                    className="flex items-center justify-between rounded-md border px-3 py-2"
                  >
                    <span className="text-sm">{name}</span>
                    <Badge
                      variant={breakerColors[state] || "secondary"}
                      className="text-xs"
                    >
                      {breakerLabels[state] || state}
                    </Badge>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
