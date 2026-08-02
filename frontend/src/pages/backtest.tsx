import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { runBacktest } from "@/api";
import type { BacktestResult } from "@/types";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { Loader2, Play } from "lucide-react";

export function BacktestPage() {
  const [form, setForm] = useState({
    ticker: "",
    startDate: "",
    endDate: "",
    initialCapital: "1000000",
    decisionInterval: "1",
  });
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleRun = async () => {
    setRunning(true);
    setError(null);
    setResult(null);
    try {
      const res = await runBacktest({
        ticker: form.ticker,
        start_date: form.startDate,
        end_date: form.endDate,
        initial_capital: Number(form.initialCapital),
        decision_interval: Number(form.decisionInterval),
      });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "回测失败");
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="space-y-6 p-6">
      <div>
        <h1 className="text-2xl font-bold">Backtest</h1>
        <p className="text-sm text-muted-foreground">使用历史数据验证 Agent 策略表现</p>
      </div>

      {/* Form */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">回测参数</CardTitle>
          <CardDescription>设置股票、时间范围和初始资金</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            <div className="space-y-2">
              <Label htmlFor="bt-ticker">股票代码</Label>
              <Input
                id="bt-ticker"
                placeholder="000737"
                value={form.ticker}
                onChange={(e) => setForm({ ...form, ticker: e.target.value })}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="bt-start">开始日期</Label>
              <Input
                id="bt-start"
                type="date"
                value={form.startDate}
                onChange={(e) => setForm({ ...form, startDate: e.target.value })}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="bt-end">结束日期</Label>
              <Input
                id="bt-end"
                type="date"
                value={form.endDate}
                onChange={(e) => setForm({ ...form, endDate: e.target.value })}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="bt-capital">初始资金 (¥)</Label>
              <Input
                id="bt-capital"
                type="number"
                value={form.initialCapital}
                onChange={(e) => setForm({ ...form, initialCapital: e.target.value })}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="bt-interval">决策间隔 (天)</Label>
              <Input
                id="bt-interval"
                type="number"
                min="1"
                value={form.decisionInterval}
                onChange={(e) => setForm({ ...form, decisionInterval: e.target.value })}
              />
            </div>
          </div>
          <div className="mt-4">
            <Button onClick={handleRun} disabled={running || !form.ticker || !form.startDate || !form.endDate}>
              {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              {running ? "回测中..." : "开始回测"}
            </Button>
          </div>
        </CardContent>
      </Card>

      {error && (
        <Card className="border-destructive">
          <CardContent className="pt-6">
            <p className="text-sm text-destructive">{error}</p>
          </CardContent>
        </Card>
      )}

      {/* Results */}
      {result && (
        <>
          {/* Stats */}
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-xs text-muted-foreground">初始资金</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-xl font-bold">¥{result.initial_capital.toLocaleString()}</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-xs text-muted-foreground">最终市值</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-xl font-bold">¥{result.final_value.toLocaleString()}</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-xs text-muted-foreground">总收益率</CardTitle>
              </CardHeader>
              <CardContent>
                <p className={`text-xl font-bold ${result.total_return >= 0 ? "text-chart-1" : "text-destructive"}`}>
                  {result.total_return >= 0 ? "+" : ""}{(result.total_return * 100).toFixed(2)}%
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-xs text-muted-foreground">最大回撤</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-xl font-bold text-destructive">
                  -{(result.max_drawdown * 100).toFixed(2)}%
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-xs text-muted-foreground">交易次数</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-xl font-bold">{result.total_trades}</p>
              </CardContent>
            </Card>
          </div>

          {/* Equity curve */}
          {result.equity_curve.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">资金曲线</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={result.equity_curve}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="date" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                    <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "hsl(var(--card))",
                        border: "1px solid hsl(var(--border))",
                        borderRadius: "6px",
                      }}
                    />
                    <ReferenceLine y={result.initial_capital} stroke="hsl(var(--muted-foreground))" strokeDasharray="3 3" />
                    <Line
                      type="monotone"
                      dataKey="value"
                      stroke="hsl(var(--chart-1))"
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          {/* Trade log */}
          {result.trades.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">交易记录</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b text-left text-muted-foreground">
                        <th className="pb-2 pr-4">日期</th>
                        <th className="pb-2 pr-4">操作</th>
                        <th className="pb-2 pr-4">股票</th>
                        <th className="pb-2 pr-4 text-right">股数</th>
                        <th className="pb-2 pr-4 text-right">价格</th>
                        <th className="pb-2 pr-4 text-right">金额</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.trades.map((t, i) => (
                        <tr key={i} className="border-b">
                          <td className="py-2 pr-4">{t.date}</td>
                          <td className="py-2 pr-4">
                            <Badge variant={t.action === "buy" ? "success" : "destructive"}>
                              {t.action === "buy" ? "买入" : "卖出"}
                            </Badge>
                          </td>
                          <td className="py-2 pr-4">{t.ticker}</td>
                          <td className="py-2 pr-4 text-right">{t.shares.toLocaleString()}</td>
                          <td className="py-2 pr-4 text-right">¥{t.price.toFixed(2)}</td>
                          <td className="py-2 pr-4 text-right">¥{t.amount.toLocaleString()}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          )}

          {!result.equity_curve.length && (
            <Card>
              <CardContent className="pt-6">
                <p className="text-sm text-muted-foreground">暂无回测数据</p>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
