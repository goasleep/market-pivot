import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { TrendingUp, TrendingDown, Wallet, Activity, Brain, Shield, Zap } from "lucide-react";
import { getStrategies, getSystemStatus, type StrategyInfo } from "@/api";

export function DashboardPage() {
  const [strategies, setStrategies] = useState<StrategyInfo[]>([]);
  const [breakers, setBreakers] = useState<Record<string, string>>({});

  useEffect(() => {
    getStrategies().then(setStrategies).catch(() => {});
    getSystemStatus().then(setBreakers).catch(() => {});
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
            <div className="text-2xl font-bold">¥1,000,000</div>
            <p className="text-xs text-muted-foreground">初始资金</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">累计收益</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-muted-foreground">¥0</div>
            <p className="text-xs text-muted-foreground">0.00%</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">持仓股票</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">0</div>
            <p className="text-xs text-muted-foreground">无持仓</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Agent 状态</CardTitle>
            <Brain className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <Badge variant="success">就绪</Badge>
              <span className="text-xs text-muted-foreground">DeepSeek</span>
            </div>
          </CardContent>
        </Card>
      </div>

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
