import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router";
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
import { Badge } from "@/components/ui/badge";
import {
  getAutomationDecisions,
  getAutomationRuns,
  getAutomationTask,
  getPortfolio,
  getSimulationAccounts,
  getStrategyDeployments,
  openSimulationStream,
  confirmAutomationDecision,
  confirmAutomationRun,
  runAutomation,
  settleAutomation,
  setStrategyDeploymentStatus,
  updateAutomationTask,
} from "@/api";
import type {
  AgentDecisionAudit,
  AgentRunSummary,
  AssetType,
  AutomationTask,
  AutomationTaskConfig,
  Portfolio,
  StrategyDeployment,
} from "@/types";
import {
  Bot,
  CalendarClock,
  CheckCircle2,
  Play,
  RefreshCw,
  ShieldAlert,
} from "lucide-react";
import { PageHeader, PageShell } from "@/components/layout/Page";

const DEFAULT_CONFIG: AutomationTaskConfig = {
  enabled: false,
  mode: "observe",
  execution_mode: "paper",
  live_armed: false,
  schedule_time: "15:10",
  weekdays: [0, 1, 2, 3, 4],
  universe: [],
  asset_type: "stock",
  strategy_name: null,
  deployment_id: null,
  max_symbols_per_run: 50,
  max_orders_per_run: 20,
  daily_loss_limit_pct: 0.03,
  data_max_age_seconds: 86400,
  fill_time: "next_open",
  simulation_only: true,
};

const WEEKDAYS = ["周一", "周二", "周三", "周四", "周五"];

function statusLabel(status: string) {
  const labels: Record<string, string> = {
    completed: "已完成",
    running: "运行中",
    failed: "失败",
    skipped: "已跳过",
    queued: "排队中",
  };
  return labels[status] || status;
}

export function AutomationPage() {
  const navigate = useNavigate();
  const { accountId: routeAccountId } = useParams();
  const accountId = routeAccountId || "default";
  const [accounts, setAccounts] = useState<Portfolio[]>([]);
  const [deployment, setDeployment] = useState<StrategyDeployment | null>(null);
  const [task, setTask] = useState<AutomationTask | null>(null);
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [runs, setRuns] = useState<AgentRunSummary[]>([]);
  const [decisions, setDecisions] = useState<AgentDecisionAudit[]>([]);
  const [config, setConfig] = useState<AutomationTaskConfig>(DEFAULT_CONFIG);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");

  const refresh = async () => {
    const [
      nextTask,
      nextPortfolio,
      nextRuns,
      nextDecisions,
      nextAccounts,
      deployments,
    ] = await Promise.all([
      getAutomationTask(accountId),
      getPortfolio(accountId),
      getAutomationRuns(accountId),
      getAutomationDecisions(accountId),
      getSimulationAccounts(),
      getStrategyDeployments(accountId),
    ]);
    setTask(nextTask);
    setPortfolio(nextPortfolio);
    setConfig(nextTask.config);
    setRuns(nextRuns);
    setDecisions(nextDecisions);
    setAccounts(nextAccounts);
    setDeployment(deployments[0] || null);
  };

  useEffect(() => {
    refresh().catch((error) => setMessage(error.message));
    const socket = openSimulationStream(accountId, (event) => {
      if (
        event.type.startsWith("agent.") ||
        event.type === "daily.settled" ||
        event.type === "automation.updated"
      ) {
        refresh().catch(() => {});
      }
    });
    const timer = window.setInterval(() => refresh().catch(() => {}), 30_000);
    return () => {
      socket.close();
      window.clearInterval(timer);
    };
  }, [accountId]);

  const latestRun = runs[0];
  const autoModeWarning = useMemo(
    () => config.mode === "auto" && config.enabled,
    [config.enabled, config.mode],
  );

  const save = async () => {
    setBusy(true);
    try {
      await updateAutomationTask(accountId, config);
      setMessage("自动化配置已保存");
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "保存失败");
    } finally {
      setBusy(false);
    }
  };

  const runNow = async () => {
    setBusy(true);
    try {
      const result = await runAutomation(accountId);
      setMessage(`Agent 运行已结束：${statusLabel(result.status)}`);
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "运行失败");
    } finally {
      setBusy(false);
    }
  };

  const settle = async () => {
    setBusy(true);
    try {
      await settleAutomation(accountId);
      setMessage("已完成当前模拟日结算和盯市");
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "结算失败");
    } finally {
      setBusy(false);
    }
  };

  const confirm = async (decisionId: string) => {
    setBusy(true);
    try {
      await confirmAutomationDecision(accountId, decisionId);
      setMessage("决策已确认并提交到模拟账户");
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "确认失败");
    } finally {
      setBusy(false);
    }
  };

  const confirmLatestRun = async () => {
    if (
      !latestRun ||
      !window.confirm("确认按卖出优先顺序提交本次运行的全部待确认模拟订单？")
    )
      return;
    setBusy(true);
    try {
      const result = await confirmAutomationRun(accountId, latestRun.run_id);
      setMessage(
        `已确认 ${result.confirmed.length} 笔模拟订单${result.failures.length ? `，${result.failures.length} 笔失败` : ""}`,
      );
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "批量确认失败");
    } finally {
      setBusy(false);
    }
  };

  const updateDeploymentStatus = async (
    status: "active" | "paused" | "archived",
  ) => {
    if (!deployment) return;
    if (
      status === "archived" &&
      !window.confirm("归档后该模拟盘将停止运行，确认继续？")
    )
      return;
    setBusy(true);
    try {
      await setStrategyDeploymentStatus(deployment.deployment_id, status);
      setMessage(
        status === "active"
          ? "策略部署已启用"
          : status === "paused"
            ? "策略部署已暂停"
            : "策略部署已归档",
      );
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "更新部署状态失败");
    } finally {
      setBusy(false);
    }
  };

  return (
    <PageShell>
      <PageHeader
        eyebrow="Automation Control"
        title="Agent 自动化"
        description="按交易日运行研究任务、记录决策并管理模拟订单；关键交易仍可保留人工确认。"
        icon={Bot}
        actions={
          <>
            <select
              className="h-10 rounded-md border bg-background px-3 text-sm"
              value={accountId}
              onChange={(event) =>
                navigate(`/automation/${event.target.value}`)
              }
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
            <Button onClick={runNow} disabled={busy}>
              <Play className="mr-2 h-4 w-4" />
              立即运行 Agent
            </Button>
          </>
        }
      />

      {message && (
        <div className="rounded-md border bg-muted/40 px-3 py-2 text-sm">
          {message}
        </div>
      )}

      {deployment && (
        <Card>
          <CardContent className="flex flex-wrap items-center justify-between gap-3 pt-6">
            <div>
              <p className="font-medium">
                {deployment.strategy_name} · {deployment.status}
              </p>
              <p className="text-xs text-muted-foreground">
                {deployment.deployment_id} · SHA256{" "}
                {deployment.strategy_sha256.slice(0, 12)}…
              </p>
            </div>
            <div className="flex gap-2">
              {deployment.status !== "active" && (
                <Button
                  size="sm"
                  onClick={() => updateDeploymentStatus("active")}
                  disabled={busy}
                >
                  启用
                </Button>
              )}
              {deployment.status === "active" && (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => updateDeploymentStatus("paused")}
                  disabled={busy}
                >
                  暂停
                </Button>
              )}
              <Button
                size="sm"
                variant="outline"
                onClick={() => updateDeploymentStatus("archived")}
                disabled={busy || deployment.status === "archived"}
              >
                归档
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">模拟总资产</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              ¥
              {(portfolio?.total_value || 0).toLocaleString(undefined, {
                maximumFractionDigits: 2,
              })}
            </div>
            <p className="text-xs text-muted-foreground">
              {portfolio?.current_date || "尚未结算"}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">累计收益</CardTitle>
          </CardHeader>
          <CardContent>
            <div
              className={
                portfolio && portfolio.total_pnl < 0
                  ? "text-2xl font-bold text-destructive"
                  : "text-2xl font-bold text-emerald-600"
              }
            >
              ¥{(portfolio?.total_pnl || 0).toFixed(2)}
            </div>
            <p className="text-xs text-muted-foreground">
              {((portfolio?.total_return_pct || 0) * 100).toFixed(2)}%
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">任务状态</CardTitle>
          </CardHeader>
          <CardContent>
            <Badge variant={task?.config.enabled ? "success" : "secondary"}>
              {task?.config.enabled ? "已启用" : "未启用"}
            </Badge>
            <p className="mt-2 text-xs text-muted-foreground">
              {task?.config.mode || "observe"} ·{" "}
              {task?.config.schedule_time || "15:10"}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">最新运行</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-lg font-semibold">
              {latestRun ? statusLabel(latestRun.status) : "暂无"}
            </div>
            <p className="text-xs text-muted-foreground">
              {latestRun
                ? `${latestRun.decisions_count} 个决策 / ${latestRun.orders_count} 个订单`
                : "先运行一次 Agent"}
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-[1fr_1.3fr]">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CalendarClock className="h-4 w-4" />
              无人值守配置
            </CardTitle>
            <CardDescription>
              {accountId} · 所有订单均为模拟订单
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={config.enabled}
                onChange={(e) =>
                  setConfig({ ...config, enabled: e.target.checked })
                }
              />
              启用每日自动任务
            </label>
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-2">
                <Label>运行模式</Label>
                <select
                  className="h-10 w-full rounded-md border bg-background px-3 text-sm"
                  value={config.mode}
                  onChange={(e) =>
                    setConfig({
                      ...config,
                      mode: e.target.value as AutomationTaskConfig["mode"],
                    })
                  }
                >
                  <option value="observe">observe · 只观察</option>
                  <option value="confirm">confirm · 记录待确认</option>
                  <option value="auto">auto · 自动下单</option>
                </select>
              </div>
              <div className="space-y-2">
                <Label>执行账户</Label>
                <select
                  disabled={Boolean(config.deployment_id)}
                  className="h-10 w-full rounded-md border bg-background px-3 text-sm disabled:opacity-60"
                  value={config.execution_mode}
                  onChange={(e) =>
                    setConfig({
                      ...config,
                      execution_mode: e.target
                        .value as AutomationTaskConfig["execution_mode"],
                      simulation_only: e.target.value !== "live",
                      live_armed: false,
                    })
                  }
                >
                  <option value="paper">paper · 模拟盘</option>
                  <option value="live">live · 实盘 Adapter</option>
                </select>
              </div>
              <div className="space-y-2">
                <Label>每日运行时间</Label>
                <Input
                  type="time"
                  value={config.schedule_time}
                  onChange={(e) =>
                    setConfig({ ...config, schedule_time: e.target.value })
                  }
                />
              </div>
            </div>
            {config.execution_mode === "live" && (
              <label className="flex items-center gap-2 rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-900">
                <input
                  type="checkbox"
                  checked={config.live_armed}
                  onChange={(e) =>
                    setConfig({
                      ...config,
                      live_armed: e.target.checked,
                      simulation_only: false,
                    })
                  }
                />
                确认已配置并审核实盘 Adapter，允许任务提交真实订单
              </label>
            )}
            <div className="space-y-2">
              <Label>运行日</Label>
              <div className="flex flex-wrap gap-2">
                {WEEKDAYS.map((label, index) => (
                  <Button
                    key={label}
                    type="button"
                    size="sm"
                    variant={
                      config.weekdays.includes(index) ? "default" : "outline"
                    }
                    onClick={() =>
                      setConfig({
                        ...config,
                        weekdays: config.weekdays.includes(index)
                          ? config.weekdays.filter((day) => day !== index)
                          : [...config.weekdays, index].sort(),
                      })
                    }
                  >
                    {label}
                  </Button>
                ))}
              </div>
            </div>
            {config.deployment_id && (
              <p className="rounded-md border bg-muted/40 p-3 text-xs text-muted-foreground">
                部署 ID：{config.deployment_id}
                。策略版本、标的池、资产类型和成交参数已锁定；可调整运行时间、模式与风控限额。
              </p>
            )}
            <div className="space-y-2">
              <Label>资产类型</Label>
              <select
                disabled={Boolean(config.deployment_id)}
                className="h-10 w-full rounded-md border bg-background px-3 text-sm disabled:opacity-60"
                value={config.asset_type}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    asset_type: e.target.value as AssetType,
                  })
                }
              >
                <option value="stock">股票</option>
                <option value="etf">ETF</option>
                <option value="lof">LOF</option>
              </select>
            </div>
            <div className="space-y-2">
              <Label>交易标的池（逗号分隔）</Label>
              <Input
                disabled={Boolean(config.deployment_id)}
                placeholder={
                  config.asset_type === "stock"
                    ? "000001,600519"
                    : config.asset_type === "etf"
                      ? "510300,159915"
                      : "166009"
                }
                value={config.universe.join(",")}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    universe: e.target.value
                      .split(",")
                      .map((s) => s.trim())
                      .filter(Boolean),
                  })
                }
              />
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-2">
                <Label>策略名称（可选）</Label>
                <Input
                  disabled={Boolean(config.deployment_id)}
                  placeholder="默认路由策略"
                  value={config.strategy_name || ""}
                  onChange={(e) =>
                    setConfig({
                      ...config,
                      strategy_name: e.target.value || null,
                    })
                  }
                />
              </div>
              <div className="space-y-2">
                <Label>信号成交</Label>
                <select
                  disabled={Boolean(config.deployment_id)}
                  className="h-10 w-full rounded-md border bg-background px-3 text-sm disabled:opacity-60"
                  value={config.fill_time}
                  onChange={(e) =>
                    setConfig({
                      ...config,
                      fill_time: e.target
                        .value as AutomationTaskConfig["fill_time"],
                    })
                  }
                >
                  <option value="next_open">下一交易日开盘</option>
                  <option value="same_close">当日收盘价</option>
                  <option value="manual">手动确认</option>
                </select>
              </div>
            </div>
            <div className="grid gap-3 sm:grid-cols-3">
              <div className="space-y-2">
                <Label>单日亏损熔断比例</Label>
                <Input
                  type="number"
                  min="0"
                  max="1"
                  step="0.01"
                  value={config.daily_loss_limit_pct}
                  onChange={(e) =>
                    setConfig({
                      ...config,
                      daily_loss_limit_pct: Number(e.target.value),
                    })
                  }
                />
              </div>
              <div className="space-y-2">
                <Label>每次最大股票数</Label>
                <Input
                  type="number"
                  min="1"
                  max="500"
                  value={config.max_symbols_per_run}
                  onChange={(e) =>
                    setConfig({
                      ...config,
                      max_symbols_per_run: Number(e.target.value),
                    })
                  }
                />
              </div>
              <div className="space-y-2">
                <Label>每次最大订单数</Label>
                <Input
                  type="number"
                  min="1"
                  max="200"
                  value={config.max_orders_per_run}
                  onChange={(e) =>
                    setConfig({
                      ...config,
                      max_orders_per_run: Number(e.target.value),
                    })
                  }
                />
              </div>
            </div>
            {autoModeWarning && (
              <div className="flex gap-2 rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">
                <ShieldAlert className="h-4 w-4 shrink-0" />
                {config.execution_mode === "paper"
                  ? "当前任务只会进入本地模拟账户。"
                  : "实盘模式还需要服务端 LIVE_TRADING_ENABLED、账户 Adapter 和 live armed 同时满足。"}
              </div>
            )}
            <Button onClick={save} disabled={busy} className="w-full">
              保存自动化配置
            </Button>
          </CardContent>
        </Card>

        <div className="space-y-6">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between gap-3">
                <CardTitle className="flex items-center gap-2">
                  <Bot className="h-4 w-4" />
                  Agent 运行记录
                </CardTitle>
                {latestRun &&
                  decisions.some(
                    (item) =>
                      item.run_id === latestRun.run_id &&
                      item.confirmation_status === "pending",
                  ) && (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={confirmLatestRun}
                      disabled={busy}
                    >
                      确认本次全部提案
                    </Button>
                  )}
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {runs.length === 0 ? (
                  <p className="text-sm text-muted-foreground">暂无运行记录</p>
                ) : (
                  runs.slice(0, 8).map((run) => (
                    <div
                      key={run.run_id}
                      className="flex items-center justify-between rounded-md border px-3 py-2 text-sm"
                    >
                      <div>
                        <div className="font-medium">
                          {run.run_date} · {run.trigger}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {run.symbols_processed}/{run.symbols_total} 标的 ·{" "}
                          {run.decisions_count} 决策 · {run.orders_count} 订单
                        </div>
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
                        {statusLabel(run.status)}
                      </Badge>
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>最近决策审计</CardTitle>
              <CardDescription>
                每个决策都保留价格、风控和订单关联
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {decisions.length === 0 ? (
                  <p className="text-sm text-muted-foreground">暂无决策</p>
                ) : (
                  decisions.slice(0, 10).map((item) => (
                    <div
                      key={item.decision_id}
                      className="flex items-center justify-between rounded-md border px-3 py-2 text-sm"
                    >
                      <div>
                        <span className="font-medium">{item.ticker}</span>
                        <span className="ml-3">
                          {item.decision.decision === "buy"
                            ? "买入"
                            : item.decision.decision === "sell"
                              ? "卖出"
                              : "持有"}
                        </span>
                        <div className="text-xs text-muted-foreground">
                          价格 ¥{item.current_price.toFixed(2)} ·{" "}
                          {item.risk_reason || "风控通过"}
                        </div>
                      </div>
                      <div className="flex items-center gap-2 text-right">
                        <div>
                          <Badge
                            variant={
                              item.risk_status === "rejected"
                                ? "destructive"
                                : item.decision.decision === "hold"
                                  ? "secondary"
                                  : "success"
                            }
                          >
                            {item.risk_status === "rejected"
                              ? "拦截"
                              : item.order_id
                                ? "已下单"
                                : "已记录"}
                          </Badge>
                          {item.order_id && (
                            <div className="mt-1 text-[10px] text-muted-foreground">
                              {item.order_id}
                            </div>
                          )}
                        </div>
                        {!item.order_id &&
                          item.risk_status !== "rejected" &&
                          item.decision.decision !== "hold" && (
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => confirm(item.decision_id)}
                              disabled={busy}
                            >
                              确认
                            </Button>
                          )}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
          <div className="flex justify-end">
            <Button variant="outline" onClick={settle} disabled={busy}>
              <CheckCircle2 className="mr-2 h-4 w-4" />
              执行模拟日结算
            </Button>
          </div>
        </div>
      </div>
    </PageShell>
  );
}
