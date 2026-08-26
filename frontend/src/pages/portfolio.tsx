import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router";
import {
  Activity,
  Clock3,
  MessageSquare,
  RefreshCw,
  ShieldCheck,
  Wallet,
} from "lucide-react";
import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  getPortfolio,
  getSimulationAccounts,
  getSimulationSnapshots,
  openSimulationStream,
} from "@/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { MetricCard, PageHeader, PageShell } from "@/components/layout/Page";
import { assetTypeLabel } from "@/lib/assets";
import type { Portfolio, SimulationSnapshot } from "@/types";

const currency = (value: number, maximumFractionDigits = 2) =>
  `¥${value.toLocaleString(undefined, { maximumFractionDigits })}`;

const percent = (value: number) => `${(value * 100).toFixed(2)}%`;

const orderStatusLabel: Record<string, string> = {
  pending: "待成交",
  filled: "已成交",
  rejected: "已拒绝",
  cancelled: "已取消",
};

export function PortfolioPage() {
  const navigate = useNavigate();
  const { accountId: routeAccountId } = useParams();
  const accountId = routeAccountId || "default";
  const [accounts, setAccounts] = useState<Portfolio[]>([]);
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [snapshots, setSnapshots] = useState<SimulationSnapshot[]>([]);
  const [loading, setLoading] = useState(true);
  const [streamConnected, setStreamConnected] = useState(false);
  const [message, setMessage] = useState("");

  const refresh = async (showLoading = false) => {
    if (showLoading) setLoading(true);
    try {
      const [nextPortfolio, nextAccounts, nextSnapshots] = await Promise.all([
        getPortfolio(accountId),
        getSimulationAccounts(),
        getSimulationSnapshots(accountId),
      ]);
      setPortfolio(nextPortfolio);
      setAccounts(nextAccounts);
      setSnapshots(nextSnapshots);
      setMessage("");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "加载模拟组合失败");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh(true);
    const socket = openSimulationStream(
      accountId,
      (event) => {
        setStreamConnected(true);
        if (
          event.type.startsWith("order.") ||
          event.type === "daily.settled" ||
          event.type === "account.updated"
        ) {
          refresh();
        }
      },
      () => setStreamConnected(false),
    );
    socket.onopen = () => setStreamConnected(true);
    socket.onclose = () => setStreamConnected(false);
    const timer = window.setInterval(() => refresh(), 30_000);
    return () => {
      socket.close();
      window.clearInterval(timer);
    };
  }, [accountId]);

  if (loading || !portfolio) {
    return <div className="p-6 text-sm text-muted-foreground">加载模拟组合中...</div>;
  }

  const investedValue = Math.max(0, portfolio.total_value - portfolio.cash);
  const totalPositionPct =
    portfolio.total_value > 0 ? investedValue / portfolio.total_value : 0;
  const largestPositionPct =
    portfolio.total_value > 0
      ? Math.max(
          0,
          ...portfolio.positions.map(
            (item) => item.market_value / portfolio.total_value,
          ),
        )
      : 0;
  const maxTotalPosition = portfolio.config.max_total_position_pct;
  const maxSinglePosition = portfolio.config.max_single_position_pct;

  return (
    <PageShell>
      <PageHeader
        eyebrow="Paper Portfolio"
        title="模拟组合"
        description={
          <span className="flex flex-wrap items-center gap-2">
            只展示纸面账户、持仓、订单和风险边界。研究任务统一从 Chat Agent 发起。
            <Badge variant="outline">仅模拟交易</Badge>
            <span className="flex items-center gap-1 text-xs text-muted-foreground">
              <span
                className={`h-2 w-2 rounded-full ${streamConnected ? "bg-emerald-500" : "bg-amber-500"}`}
              />
              {streamConnected ? "实时同步" : "定时刷新"}
            </span>
          </span>
        }
        icon={Wallet}
        actions={
          <>
            <select
              className="field-surface w-auto min-w-48"
              value={accountId}
              onChange={(event) => navigate(`/portfolio/${event.target.value}`)}
            >
              {accounts.map((account) => (
                <option key={account.account_id} value={account.account_id}>
                  {account.name} · {account.account_id}
                </option>
              ))}
            </select>
            <Button variant="outline" onClick={() => refresh()}>
              <RefreshCw className="mr-2 h-4 w-4" />
              刷新
            </Button>
            <Button onClick={() => navigate("/chat")}>
              <MessageSquare className="mr-2 h-4 w-4" />
              和 Agent 对话
            </Button>
          </>
        }
      />

      {message && (
        <div className="rounded-md border bg-muted/40 px-3 py-2 text-sm">
          {message}
        </div>
      )}

      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="组合总资产"
          value={currency(portfolio.total_value)}
          detail={`可用现金 ${currency(portfolio.cash)}`}
          icon={Wallet}
        />
        <MetricCard
          label="累计收益"
          value={`${portfolio.total_return_pct >= 0 ? "+" : ""}${percent(portfolio.total_return_pct)}`}
          detail={`${portfolio.total_pnl >= 0 ? "+" : ""}${currency(portfolio.total_pnl)}`}
          icon={Activity}
          tone={portfolio.total_pnl < 0 ? "negative" : "positive"}
        />
        <MetricCard
          label="当前仓位"
          value={percent(totalPositionPct)}
          detail={`${portfolio.positions.length} 个持仓`}
          icon={ShieldCheck}
        />
        <MetricCard
          label="待处理订单"
          value={String(
            portfolio.orders.filter((order) => order.status === "pending").length,
          )}
          detail={`账户状态：${portfolio.status}`}
          icon={Clock3}
        />
      </section>

      <div className="grid gap-6 xl:grid-cols-[0.7fr_1.3fr]">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <ShieldCheck className="h-4 w-4" />
              风险边界
            </CardTitle>
            <CardDescription>纸面账户必须遵守的确定性约束。</CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            <div>
              <div className="mb-2 flex justify-between text-sm">
                <span>组合仓位</span>
                <span>
                  {percent(totalPositionPct)} / {percent(maxTotalPosition)}
                </span>
              </div>
              <Progress
                value={
                  maxTotalPosition
                    ? Math.min(100, (totalPositionPct / maxTotalPosition) * 100)
                    : 0
                }
              />
            </div>
            <div>
              <div className="mb-2 flex justify-between text-sm">
                <span>最大单一持仓</span>
                <span>
                  {percent(largestPositionPct)} / {percent(maxSinglePosition)}
                </span>
              </div>
              <Progress
                value={
                  maxSinglePosition
                    ? Math.min(
                        100,
                        (largestPositionPct / maxSinglePosition) * 100,
                      )
                    : 0
                }
              />
            </div>
            <div className="grid grid-cols-2 gap-3 rounded-lg bg-muted/40 p-3 text-sm">
              <div>
                <p className="text-xs text-muted-foreground">默认止损</p>
                <p className="mt-1 font-medium">
                  {percent(portfolio.config.default_stop_loss_pct)}
                </p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">最小交易单位</p>
                <p className="mt-1 font-medium">{portfolio.config.min_lot} 份</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">组合净值</CardTitle>
            <CardDescription>模拟账户的历史结算快照。</CardDescription>
          </CardHeader>
          <CardContent>
            {snapshots.length === 0 ? (
              <div className="flex h-56 items-center justify-center rounded-lg border border-dashed text-sm text-muted-foreground">
                暂无结算快照
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={[...snapshots].reverse()}>
                  <XAxis dataKey="date" fontSize={11} />
                  <YAxis fontSize={11} domain={["auto", "auto"]} />
                  <Tooltip formatter={(value) => currency(Number(value ?? 0))} />
                  <Line
                    type="monotone"
                    dataKey="total_value"
                    stroke="hsl(var(--chart-1))"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">当前持仓</CardTitle>
          <CardDescription>纸面账户当前持仓及浮动盈亏。</CardDescription>
        </CardHeader>
        <CardContent>
          {portfolio.positions.length === 0 ? (
            <p className="text-sm text-muted-foreground">暂无持仓</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-left text-muted-foreground">
                    <th className="pb-2 pr-4">标的</th>
                    <th className="pb-2 pr-4 text-right">持仓</th>
                    <th className="pb-2 pr-4 text-right">成本价</th>
                    <th className="pb-2 pr-4 text-right">现价</th>
                    <th className="pb-2 pr-4 text-right">市值</th>
                    <th className="pb-2 text-right">盈亏</th>
                  </tr>
                </thead>
                <tbody>
                  {portfolio.positions.map((position) => (
                    <tr key={position.ticker} className="border-b last:border-0">
                      <td className="py-3 pr-4">
                        <span className="font-medium">{position.ticker}</span>
                        <span className="ml-2 text-xs text-muted-foreground">
                          {assetTypeLabel(position.asset_type)}
                        </span>
                      </td>
                      <td className="py-3 pr-4 text-right">
                        {position.shares.toLocaleString()}
                      </td>
                      <td className="py-3 pr-4 text-right">
                        {currency(position.avg_cost, 3)}
                      </td>
                      <td className="py-3 pr-4 text-right">
                        {currency(position.current_price, 3)}
                      </td>
                      <td className="py-3 pr-4 text-right">
                        {currency(position.market_value)}
                      </td>
                      <td
                        className={`py-3 text-right ${position.pnl < 0 ? "text-destructive" : "text-emerald-600"}`}
                      >
                        {position.pnl >= 0 ? "+" : ""}
                        {currency(position.pnl)}
                        <div className="text-xs">
                          {position.pnl_pct >= 0 ? "+" : ""}
                          {percent(position.pnl_pct)}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid gap-6 xl:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">最近订单</CardTitle>
            <CardDescription>账户最近提交的纸面订单。</CardDescription>
          </CardHeader>
          <CardContent>
            {portfolio.orders.length === 0 ? (
              <p className="text-sm text-muted-foreground">暂无订单</p>
            ) : (
              <div className="space-y-2">
                {portfolio.orders.slice(0, 8).map((order) => (
                  <div
                    key={order.order_id}
                    className="flex items-center justify-between rounded-md border px-3 py-2 text-sm"
                  >
                    <div>
                      <p className="font-medium">
                        {order.ticker} · {order.side === "buy" ? "买入" : "卖出"}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {order.shares.toLocaleString()} 份 · {order.submitted_date}
                      </p>
                    </div>
                    <Badge variant={order.status === "filled" ? "success" : "secondary"}>
                      {orderStatusLabel[order.status] || order.status}
                    </Badge>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">最近成交</CardTitle>
            <CardDescription>纸面账户最近成交记录。</CardDescription>
          </CardHeader>
          <CardContent>
            {portfolio.trades.length === 0 ? (
              <p className="text-sm text-muted-foreground">暂无成交</p>
            ) : (
              <div className="space-y-2">
                {portfolio.trades.slice(0, 8).map((trade, index) => (
                  <div
                    key={`${trade.date}-${trade.ticker}-${index}`}
                    className="flex items-center justify-between rounded-md border px-3 py-2 text-sm"
                  >
                    <div>
                      <p className="font-medium">
                        {trade.ticker} · {trade.action === "buy" ? "买入" : "卖出"}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {trade.shares.toLocaleString()} 份 · {trade.date}
                      </p>
                    </div>
                    <span className="font-medium">{currency(trade.price, 3)}</span>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </PageShell>
  );
}
