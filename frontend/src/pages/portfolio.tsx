import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router";
import {
  ArrowRight,
  Bot,
  CheckCircle2,
  CirclePause,
  Clock3,
  MessageSquare,
  RefreshCw,
  ShieldCheck,
  Wallet,
  XCircle,
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
  confirmAutomationDecision,
  getAutomationDecisions,
  getAutomationRuns,
  getAutomationTask,
  getPortfolio,
  getSimulationAccounts,
  getSimulationSnapshots,
  getStrategyDeployments,
  openSimulationStream,
  rejectAutomationDecision,
  setStrategyDeploymentStatus,
  updateAutomationTask,
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
import type {
  AgentDecisionAudit,
  AgentRunSummary,
  AutomationTask,
  Portfolio,
  SimulationSnapshot,
  StrategyDeployment,
} from "@/types";

const currency = (value: number, maximumFractionDigits = 2) =>
  `¥${value.toLocaleString(undefined, { maximumFractionDigits })}`;

const percent = (value: number) => `${(value * 100).toFixed(2)}%`;

const assetLabel = (value?: string) => {
  if (value === "etf") return "ETF";
  if (value === "lof") return "LOF";
  return "股票";
};

const decisionLabel = (value: string) => {
  if (value === "buy") return "买入";
  if (value === "sell") return "卖出";
  return "持有";
};

const runStatusLabel = (value: string) => {
  const labels: Record<string, string> = {
    queued: "排队中",
    running: "运行中",
    completed: "已完成",
    failed: "失败",
    cancelled: "已取消",
    skipped: "已跳过",
  };
  return labels[value] || value;
};

const modeLabel = (value?: string) => {
  if (value === "auto") return "自动执行";
  if (value === "confirm") return "人工确认";
  return "仅观察";
};

function decisionState(item: AgentDecisionAudit) {
  if (item.confirmation_status === "rejected") return "用户已拒绝";
  if (item.confirmation_status === "expired") return "提案已过期";
  if (item.risk_status === "rejected") return "风控拦截";
  if (item.order_id) return "已生成订单";
  if (item.confirmation_status === "pending") return "待确认";
  if (item.decision.decision === "hold") return "继续持有";
  return "已记录";
}

export function PortfolioPage() {
  const navigate = useNavigate();
  const { accountId: routeAccountId } = useParams();
  const accountId = routeAccountId || "default";
  const [accounts, setAccounts] = useState<Portfolio[]>([]);
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [task, setTask] = useState<AutomationTask | null>(null);
  const [deployment, setDeployment] = useState<StrategyDeployment | null>(null);
  const [runs, setRuns] = useState<AgentRunSummary[]>([]);
  const [decisions, setDecisions] = useState<AgentDecisionAudit[]>([]);
  const [snapshots, setSnapshots] = useState<SimulationSnapshot[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [streamConnected, setStreamConnected] = useState(false);
  const [message, setMessage] = useState("");

  const refresh = async (showLoading = false) => {
    if (showLoading) setLoading(true);
    try {
      const [
        nextPortfolio,
        nextTask,
        nextRuns,
        nextDecisions,
        nextAccounts,
        nextSnapshots,
        deployments,
      ] = await Promise.all([
        getPortfolio(accountId),
        getAutomationTask(accountId),
        getAutomationRuns(accountId),
        getAutomationDecisions(accountId),
        getSimulationAccounts(),
        getSimulationSnapshots(accountId),
        getStrategyDeployments(accountId),
      ]);
      setPortfolio(nextPortfolio);
      setTask(nextTask);
      setRuns(nextRuns);
      setDecisions(nextDecisions);
      setAccounts(nextAccounts);
      setSnapshots(nextSnapshots);
      setDeployment(
        deployments.find((item) => item.status !== "archived") ||
          deployments[0] ||
          null,
      );
    } catch (error) {
      setMessage(
        error instanceof Error ? error.message : "加载 Agent 组合失败",
      );
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
          event.type.startsWith("agent.") ||
          event.type.startsWith("order.") ||
          event.type === "daily.settled" ||
          event.type === "automation.updated" ||
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

  const pendingDecisions = useMemo(
    () =>
      decisions.filter(
        (item) =>
          item.confirmation_status === "pending" &&
          item.risk_status !== "rejected" &&
          !item.order_id,
      ),
    [decisions],
  );

  if (loading || !portfolio || !task) {
    return (
      <div className="p-6 text-sm text-muted-foreground">
        加载 Agent 组合中...
      </div>
    );
  }

  const latestRun = runs[0];
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
  const universe = deployment?.universe || task.config.universe || [];
  const agentActive = Boolean(
    portfolio.status === "active" &&
    task.config.enabled &&
    (!deployment || deployment.status === "active"),
  );

  const handleDecision = async (
    decisionId: string,
    action: "confirm" | "reject",
  ) => {
    if (
      action === "reject" &&
      !window.confirm("拒绝后 Agent 不会为这项提案生成订单，确认继续？")
    )
      return;
    setBusy(true);
    setMessage("");
    try {
      if (action === "confirm") {
        await confirmAutomationDecision(accountId, decisionId);
        setMessage("提案已批准，Agent 已将其提交到模拟账户");
      } else {
        await rejectAutomationDecision(accountId, decisionId);
        setMessage("提案已拒绝并记录到决策审计");
      }
      await refresh();
    } catch (error) {
      setMessage(
        error instanceof Error ? error.message : "处理 Agent 提案失败",
      );
    } finally {
      setBusy(false);
    }
  };

  const toggleAgent = async () => {
    setBusy(true);
    setMessage("");
    try {
      if (deployment && deployment.status !== "archived") {
        await setStrategyDeploymentStatus(
          deployment.deployment_id,
          agentActive ? "paused" : "active",
        );
      } else {
        await updateAutomationTask(accountId, {
          ...task.config,
          enabled: !agentActive,
        });
      }
      setMessage(
        agentActive ? "Agent 已暂停，不会继续生成新订单" : "Agent 已恢复运行",
      );
      await refresh();
    } catch (error) {
      setMessage(
        error instanceof Error ? error.message : "更新 Agent 状态失败",
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-6 p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="mb-2 flex items-center gap-2">
            <Badge variant="outline">Agent 管理</Badge>
            <span className="flex items-center gap-1 text-xs text-muted-foreground">
              <span
                className={`h-2 w-2 rounded-full ${streamConnected ? "bg-emerald-500" : "bg-amber-500"}`}
              />
              {streamConnected ? "实时同步" : "定时刷新"}
            </span>
          </div>
          <h1 className="text-2xl font-bold">Agent 组合</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Agent 负责研究和模拟执行，您负责目标、风险边界与关键决策审批。
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <select
            className="h-10 rounded-md border bg-background px-3 text-sm"
            value={accountId}
            onChange={(event) => navigate(`/portfolio/${event.target.value}`)}
          >
            {accounts.map((account) => (
              <option key={account.account_id} value={account.account_id}>
                {account.name} · {account.account_id}
              </option>
            ))}
          </select>
          <Button variant="outline" onClick={() => refresh()} disabled={busy}>
            <RefreshCw className="mr-2 h-4 w-4" />
            刷新
          </Button>
          <Button onClick={() => navigate("/chat")}>
            <MessageSquare className="mr-2 h-4 w-4" />和 Agent 对话
          </Button>
        </div>
      </div>

      {message && (
        <div className="rounded-md border bg-muted/40 px-3 py-2 text-sm">
          {message}
        </div>
      )}

      <Card className="overflow-hidden border-blue-200 bg-gradient-to-r from-blue-50/80 via-background to-violet-50/60">
        <CardContent className="flex flex-wrap items-center justify-between gap-5 p-6">
          <div className="flex min-w-0 items-start gap-4">
            <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-blue-600 text-white shadow-lg shadow-blue-500/20">
              <Bot className="h-6 w-6" />
            </div>
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="text-lg font-semibold">
                  {deployment?.strategy_name ||
                    task.config.strategy_name ||
                    "默认研究 Agent"}
                </h2>
                <Badge variant={agentActive ? "success" : "warning"}>
                  {agentActive ? "运行中" : "已暂停"}
                </Badge>
                <Badge variant="secondary">模拟执行</Badge>
              </div>
              <p className="mt-1 text-sm text-muted-foreground">
                {modeLabel(task.config.mode)} · {task.config.schedule_time} 运行
                · {assetLabel(task.config.asset_type)} · 覆盖 {universe.length}{" "}
                个标的
              </p>
              <p className="mt-2 text-xs text-muted-foreground">
                {latestRun
                  ? `最近运行 ${latestRun.run_date}，处理 ${latestRun.symbols_processed}/${latestRun.symbols_total} 个标的，生成 ${latestRun.decisions_count} 项决策。`
                  : "尚无运行记录，Agent 会在配置的交易日按计划检查组合。"}
              </p>
            </div>
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              onClick={() => navigate(`/automation/${accountId}`)}
            >
              运行设置
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
            <Button
              variant={agentActive ? "outline" : "default"}
              onClick={toggleAgent}
              disabled={busy || deployment?.status === "archived"}
            >
              {agentActive ? (
                <CirclePause className="mr-2 h-4 w-4" />
              ) : (
                <Bot className="mr-2 h-4 w-4" />
              )}
              {agentActive ? "暂停 Agent" : "恢复 Agent"}
            </Button>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">组合总资产</CardTitle>
            <Wallet className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {currency(portfolio.total_value)}
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              可用现金 {currency(portfolio.cash)}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">累计表现</CardTitle>
          </CardHeader>
          <CardContent>
            <div
              className={`text-2xl font-bold ${portfolio.total_pnl < 0 ? "text-destructive" : "text-emerald-600"}`}
            >
              {portfolio.total_pnl >= 0 ? "+" : ""}
              {currency(portfolio.total_pnl)}
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              累计收益 {portfolio.total_return_pct >= 0 ? "+" : ""}
              {percent(portfolio.total_return_pct)}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">当前仓位</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {percent(totalPositionPct)}
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              {portfolio.positions.length} 个持仓 · 上限{" "}
              {percent(maxTotalPosition)}
            </p>
          </CardContent>
        </Card>
        <Card
          className={
            pendingDecisions.length ? "border-amber-300 bg-amber-50/40" : ""
          }
        >
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">等待您确认</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{pendingDecisions.length}</div>
            <p className="mt-1 text-xs text-muted-foreground">
              {pendingDecisions.length
                ? "Agent 已准备好交易提案"
                : "当前无需人工处理"}
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 xl:grid-cols-[1.35fr_0.65fr]">
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between gap-3">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <Clock3 className="h-4 w-4" />
                  待确认决策
                </CardTitle>
                <CardDescription>
                  只有通过 Agent 与风控检查的提案才会出现在这里。
                </CardDescription>
              </div>
              <Badge variant={pendingDecisions.length ? "warning" : "success"}>
                {pendingDecisions.length
                  ? `${pendingDecisions.length} 项待处理`
                  : "全部处理完毕"}
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            {pendingDecisions.length === 0 ? (
              <div className="flex min-h-40 flex-col items-center justify-center rounded-lg border border-dashed text-center">
                <CheckCircle2 className="mb-3 h-8 w-8 text-emerald-500" />
                <p className="font-medium">当前没有需要确认的提案</p>
                <p className="mt-1 text-sm text-muted-foreground">
                  Agent 会继续按照策略和风险边界观察组合。
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                {pendingDecisions.map((item) => {
                  const proposal = item.proposed_order || {};
                  const side = String(proposal.side || item.decision.decision);
                  const shares = Number(proposal.shares || 0);
                  const price = Number(
                    proposal.price || item.current_price || 0,
                  );
                  return (
                    <div
                      key={item.decision_id}
                      className="rounded-lg border bg-background p-4"
                    >
                      <div className="flex flex-wrap items-start justify-between gap-3">
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="text-lg font-semibold">
                              {item.ticker}
                            </span>
                            <Badge
                              variant={side === "sell" ? "warning" : "success"}
                            >
                              {decisionLabel(side)}
                            </Badge>
                            <span className="text-xs text-muted-foreground">
                              {assetLabel(item.decision.asset_type)}
                            </span>
                          </div>
                          <p className="mt-2 text-sm">
                            {item.decision.reasoning ||
                              "Agent 已根据当前策略生成组合调整提案。"}
                          </p>
                          <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-xs text-muted-foreground">
                            <span>参考价 {currency(price)}</span>
                            {shares > 0 && (
                              <span>数量 {shares.toLocaleString()}</span>
                            )}
                            {shares > 0 && price > 0 && (
                              <span>预计金额 {currency(shares * price)}</span>
                            )}
                            <span>{item.risk_reason || "风险检查通过"}</span>
                          </div>
                        </div>
                        <div className="flex gap-2">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() =>
                              handleDecision(item.decision_id, "reject")
                            }
                            disabled={busy}
                          >
                            <XCircle className="mr-1 h-4 w-4" />
                            拒绝
                          </Button>
                          <Button
                            size="sm"
                            onClick={() =>
                              handleDecision(item.decision_id, "confirm")
                            }
                            disabled={busy}
                          >
                            <CheckCircle2 className="mr-1 h-4 w-4" />
                            批准
                          </Button>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <ShieldCheck className="h-4 w-4" />
              风险边界
            </CardTitle>
            <CardDescription>
              只读展示 Agent 当前必须遵守的组合约束。
            </CardDescription>
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
                <p className="text-xs text-muted-foreground">单日亏损熔断</p>
                <p className="mt-1 font-medium">
                  {percent(task.config.daily_loss_limit_pct)}
                </p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">成交规则</p>
                <p className="mt-1 font-medium">
                  {task.config.fill_time === "next_open"
                    ? "下一交易日开盘"
                    : task.config.fill_time === "same_close"
                      ? "当日收盘"
                      : "确认后成交"}
                </p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">执行范围</p>
                <p className="mt-1 font-medium">仅模拟账户</p>
              </div>
            </div>
            <p className="text-xs text-muted-foreground">
              风险边界需要调整时，请前往运行设置；修改不会直接产生订单。
            </p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">组合净值</CardTitle>
          <CardDescription>
            模拟账户的日结算快照，用于观察策略表现与回撤。
          </CardDescription>
        </CardHeader>
        <CardContent>
          {snapshots.length === 0 ? (
            <div className="flex h-56 items-center justify-center rounded-lg border border-dashed text-sm text-muted-foreground">
              Agent 完成首次日结算后将在这里生成净值曲线
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

      <div className="grid gap-6 xl:grid-cols-[1.25fr_0.75fr]">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">当前持仓</CardTitle>
            <CardDescription>
              持仓由 Agent 的模拟订单更新，本页面不提供手工改仓。
            </CardDescription>
          </CardHeader>
          <CardContent>
            {portfolio.positions.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                暂无持仓，Agent 会在出现符合条件的机会后提交提案。
              </p>
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
                      <tr
                        key={position.ticker}
                        className="border-b last:border-0"
                      >
                        <td className="py-3 pr-4">
                          <span className="font-medium">{position.ticker}</span>
                          <span className="ml-2 text-xs text-muted-foreground">
                            {assetLabel(position.asset_type)}
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

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Agent 运行记录</CardTitle>
            <CardDescription>最近的研究、决策和订单数量。</CardDescription>
          </CardHeader>
          <CardContent>
            {runs.length === 0 ? (
              <p className="text-sm text-muted-foreground">暂无运行记录</p>
            ) : (
              <div className="space-y-2">
                {runs.slice(0, 6).map((run) => (
                  <div
                    key={run.run_id}
                    className="flex items-center justify-between rounded-md border px-3 py-2 text-sm"
                  >
                    <div>
                      <p className="font-medium">{run.run_date}</p>
                      <p className="text-xs text-muted-foreground">
                        {run.decisions_count} 项决策 · {run.orders_count} 个订单
                      </p>
                    </div>
                    <Badge
                      variant={
                        run.status === "completed"
                          ? "success"
                          : run.status === "failed"
                            ? "destructive"
                            : "secondary"
                      }
                    >
                      {runStatusLabel(run.status)}
                    </Badge>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">决策与订单审计</CardTitle>
          <CardDescription>
            保留 Agent 观点、风控结论和订单关联，便于复盘每一次动作。
          </CardDescription>
        </CardHeader>
        <CardContent>
          {decisions.length === 0 ? (
            <p className="text-sm text-muted-foreground">暂无 Agent 决策</p>
          ) : (
            <div className="space-y-2">
              {decisions.slice(0, 10).map((item) => (
                <div
                  key={item.decision_id}
                  className="flex flex-wrap items-center justify-between gap-3 rounded-md border px-3 py-3 text-sm"
                >
                  <div>
                    <span className="font-medium">{item.ticker}</span>
                    <span className="ml-3">
                      {decisionLabel(item.decision.decision)}
                    </span>
                    <div className="mt-1 text-xs text-muted-foreground">
                      参考价 {currency(item.current_price, 3)} ·{" "}
                      {item.risk_reason || "风控通过"}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge
                      variant={
                        item.risk_status === "rejected" ||
                        item.confirmation_status === "rejected"
                          ? "destructive"
                          : item.confirmation_status === "pending"
                            ? "warning"
                            : item.order_id
                              ? "success"
                              : "secondary"
                      }
                    >
                      {decisionState(item)}
                    </Badge>
                    {item.order_id && (
                      <span className="text-[10px] text-muted-foreground">
                        {item.order_id}
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
