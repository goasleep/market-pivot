import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { getPortfolio, resetPortfolio } from "@/api";
import type { Portfolio } from "@/types";
import { RefreshCw, RotateCcw, Wallet, TrendingUp, TrendingDown } from "lucide-react";

export function PortfolioPage() {
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchPortfolio = async () => {
    setLoading(true);
    try {
      const data = await getPortfolio();
      setPortfolio(data);
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPortfolio();
  }, []);

  const handleReset = async () => {
    await resetPortfolio();
    fetchPortfolio();
  };

  if (loading || !portfolio) {
    return (
      <div className="p-6">
        <p className="text-sm text-muted-foreground">加载中...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Portfolio</h1>
          <p className="text-sm text-muted-foreground">虚拟账户持仓管理</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={fetchPortfolio}>
            <RefreshCw className="h-4 w-4" />
            刷新
          </Button>
          <Button variant="outline" size="sm" onClick={handleReset}>
            <RotateCcw className="h-4 w-4" />
            重置
          </Button>
        </div>
      </div>

      {/* Summary */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">总资产</CardTitle>
            <Wallet className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">¥{portfolio.total_value.toLocaleString()}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">可用现金</CardTitle>
            <Wallet className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">¥{portfolio.cash.toLocaleString()}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">当日盈亏</CardTitle>
            {portfolio.daily_pnl >= 0 ? (
              <TrendingUp className="h-4 w-4 text-chart-1" />
            ) : (
              <TrendingDown className="h-4 w-4 text-destructive" />
            )}
          </CardHeader>
          <CardContent>
            <p
              className={`text-2xl font-bold ${
                portfolio.daily_pnl >= 0 ? "text-chart-1" : "text-destructive"
              }`}
            >
              {portfolio.daily_pnl >= 0 ? "+" : ""}¥{portfolio.daily_pnl.toLocaleString()}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Positions */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">持仓明细</CardTitle>
          <CardDescription>当前持仓的股票列表</CardDescription>
        </CardHeader>
        <CardContent>
          {portfolio.positions.length === 0 ? (
            <p className="text-sm text-muted-foreground">暂无持仓</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-left text-muted-foreground">
                    <th className="pb-2 pr-4">股票代码</th>
                    <th className="pb-2 pr-4 text-right">持仓股数</th>
                    <th className="pb-2 pr-4 text-right">成本价</th>
                    <th className="pb-2 pr-4 text-right">现价</th>
                    <th className="pb-2 pr-4 text-right">市值</th>
                    <th className="pb-2 pr-4 text-right">盈亏</th>
                    <th className="pb-2 pr-4 text-right">收益率</th>
                  </tr>
                </thead>
                <tbody>
                  {portfolio.positions.map((pos) => (
                    <tr key={pos.ticker} className="border-b">
                      <td className="py-2 pr-4 font-medium">{pos.ticker}</td>
                      <td className="py-2 pr-4 text-right">{pos.shares.toLocaleString()}</td>
                      <td className="py-2 pr-4 text-right">¥{pos.avg_cost.toFixed(2)}</td>
                      <td className="py-2 pr-4 text-right">¥{pos.current_price.toFixed(2)}</td>
                      <td className="py-2 pr-4 text-right">¥{pos.market_value.toLocaleString()}</td>
                      <td className={`py-2 pr-4 text-right ${pos.pnl >= 0 ? "text-chart-1" : "text-destructive"}`}>
                        {pos.pnl >= 0 ? "+" : ""}¥{pos.pnl.toLocaleString()}
                      </td>
                      <td className="py-2 pr-4 text-right">
                        <Badge variant={pos.pnl >= 0 ? "success" : "destructive"}>
                          {pos.pnl >= 0 ? "+" : ""}{(pos.pnl_pct * 100).toFixed(2)}%
                        </Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
