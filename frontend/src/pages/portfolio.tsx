import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  createSimulationOrder,
  getPortfolio,
  resetSimulationAccount,
  updateExternalSimulationConfig,
  updateLiveTradingConfig,
  updateSimulationConfig,
  validateSimulationBroker,
  syncSimulationBroker,
  openSimulationStream,
  getSimulationSnapshots,
} from "@/api";
import type { Portfolio, SimulationAccountConfig, SimulationEvent, SimulationSnapshot } from "@/types";
import { RefreshCw, RotateCcw } from "lucide-react";
import { LineChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const cloneConfig = (config: SimulationAccountConfig): SimulationAccountConfig => ({
  ...config,
  universe: [...config.universe],
  external: { ...config.external },
  live: { ...config.live },
});

export function PortfolioPage() {
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [configDraft, setConfigDraft] = useState<SimulationAccountConfig | null>(null);
  const [externalToken, setExternalToken] = useState("");
  const [liveToken, setLiveToken] = useState("");
  const [order, setOrder] = useState({ ticker: "", side: "buy" as "buy" | "sell", shares: "100", price: "" });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [validatingBroker, setValidatingBroker] = useState(false);
  const [streamConnected, setStreamConnected] = useState(false);
  const [events, setEvents] = useState<SimulationEvent[]>([]);
  const [message, setMessage] = useState<string | null>(null);
  const [snapshots, setSnapshots] = useState<SimulationSnapshot[]>([]);

  const fetchPortfolio = async () => {
    setLoading(true);
    try {
      const data = await getPortfolio();
      setPortfolio(data);
      setConfigDraft(cloneConfig(data.config));
      getSimulationSnapshots(data.account_id).then(setSnapshots).catch(() => setSnapshots([]));
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "加载模拟账户失败");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPortfolio();
    const timer = window.setInterval(fetchPortfolio, 30_000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!portfolio) return;
    const accountId = portfolio.account_id;
    const socket = openSimulationStream(
      accountId,
      (event) => {
        if (["connected", "pong", "heartbeat"].includes(event.type)) {
          setStreamConnected(true);
        }
        if (!["pong", "heartbeat"].includes(event.type)) {
          setEvents((current) => [event, ...current].slice(0, 8));
        }
        if (event.type !== "connected") {
          getPortfolio(accountId).then((data) => setPortfolio(data)).catch(() => undefined);
        }
      },
      () => setStreamConnected(false)
    );
    socket.onopen = () => setStreamConnected(true);
    socket.onclose = () => setStreamConnected(false);
    return () => socket.close();
  }, [portfolio?.account_id]);

  const handleSaveConfig = async () => {
    if (!portfolio || !configDraft) return;
    setSaving(true);
    setMessage(null);
    try {
      let updated = await updateSimulationConfig(portfolio.account_id, configDraft);
      updated = await updateExternalSimulationConfig(portfolio.account_id, {
        ...configDraft.external,
        ...(externalToken.trim() ? { token: externalToken.trim() } : {}),
      });
      setExternalToken("");
      if (liveToken.trim()) {
        updated = await updateLiveTradingConfig(portfolio.account_id, {
          ...configDraft.live,
          token: liveToken.trim(),
        });
        setLiveToken("");
      }
      setPortfolio(updated);
      setConfigDraft(cloneConfig(updated.config));
      setMessage("模拟账户配置已保存");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "保存配置失败");
    } finally {
      setSaving(false);
    }
  };

  const handleReset = async () => {
    if (!portfolio || !window.confirm("确定要清空该模拟账户的持仓、订单和快照吗？")) return;
    setSaving(true);
    try {
      const data = await resetSimulationAccount(portfolio.account_id);
      setPortfolio(data);
      setConfigDraft(cloneConfig(data.config));
      setMessage("模拟账户已重置");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "重置失败");
    } finally {
      setSaving(false);
    }
  };

  const handleOrder = async () => {
    if (!portfolio || !configDraft || !order.ticker.trim() || Number(order.shares) <= 0) return;
    const confirmed = window.confirm(
      `${order.side === "buy" ? "买入" : "卖出"} ${configDraft?.asset_type || "stock"} ${order.ticker.trim()} ${order.shares} 份，` +
      `${order.price ? `成交价 ¥${order.price}` : "使用实时价"}。仅提交到模拟账户，确认继续？`
    );
    if (!confirmed) return;
    setSaving(true);
    setMessage(null);
    try {
      const created = await createSimulationOrder(portfolio.account_id, {
        ticker: order.ticker.trim(),
        asset_type: configDraft.asset_type,
        side: order.side,
        shares: Number(order.shares),
        price: order.price ? Number(order.price) : undefined,
        fill_immediately: true,
        trade_date: portfolio.current_date || undefined,
      });
      setOrder({ ...order, ticker: "", price: "" });
      await fetchPortfolio();
      setMessage(
        portfolio.broker.provider === "eastmoney_file"
          ? "订单已写入东方财富文件单，等待终端成交回报"
          : created.status === "filled"
            ? "模拟订单已提交并成交"
            : "模拟订单已提交，等待日级撮合"
      );
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "模拟下单失败");
    } finally {
      setSaving(false);
    }
  };

  const handleValidateBroker = async () => {
    if (!portfolio) return;
    setValidatingBroker(true);
    setMessage(null);
    try {
      const broker = await validateSimulationBroker(portfolio.account_id);
      setPortfolio((current) => current ? { ...current, broker } : current);
      setMessage(broker.message);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "验证东方财富 EMT 配置失败");
    } finally {
      setValidatingBroker(false);
    }
  };

  const handleSyncBroker = async () => {
    if (!portfolio) return;
    setValidatingBroker(true);
    setMessage(null);
    try {
      const updated = await syncSimulationBroker(portfolio.account_id);
      setPortfolio(updated);
      setConfigDraft(cloneConfig(updated.config));
      setMessage("东方财富文件单已同步");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "同步东方财富文件单失败");
    } finally {
      setValidatingBroker(false);
    }
  };

  if (loading || !portfolio || !configDraft) {
    return <div className="p-6"><p className="text-sm text-muted-foreground">加载中...</p></div>;
  }

  const setConfig = (patch: Partial<SimulationAccountConfig>) => {
    setConfigDraft({ ...configDraft, ...patch });
  };

  const buyCount = portfolio.trades.filter((trade) => trade.action === "buy").length;
  const sellCount = portfolio.trades.filter((trade) => trade.action === "sell").length;
  const totalFees = portfolio.trades.reduce((sum, trade) => sum + (trade.commission || 0) + (trade.tax || 0), 0);

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Portfolio</h1>
          <p className="text-sm text-muted-foreground">
            {portfolio.name} · 日级 Agent 模拟账户 · {portfolio.account_id}
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={fetchPortfolio} disabled={saving}>
            <RefreshCw className="h-4 w-4" /> 刷新
          </Button>
          <Button variant="outline" size="sm" onClick={handleReset} disabled={saving}>
            <RotateCcw className="h-4 w-4" /> 重置账户
          </Button>
        </div>
      </div>

      {message && <p className="rounded-md border px-3 py-2 text-sm text-muted-foreground">{message}</p>}

      <div className="grid gap-4 md:grid-cols-4">
        <Card><CardHeader className="pb-2"><CardTitle className="text-sm font-medium">总资产</CardTitle></CardHeader>
          <CardContent><p className="text-2xl font-bold">¥{portfolio.total_value.toLocaleString()}</p></CardContent></Card>
        <Card><CardHeader className="pb-2"><CardTitle className="text-sm font-medium">可用现金</CardTitle></CardHeader>
          <CardContent><p className="text-2xl font-bold">¥{portfolio.cash.toLocaleString()}</p></CardContent></Card>
        <Card><CardHeader className="pb-2"><CardTitle className="text-sm font-medium">累计盈亏</CardTitle></CardHeader>
          <CardContent><p className={`text-2xl font-bold ${portfolio.total_pnl >= 0 ? "text-chart-1" : "text-destructive"}`}>
            {portfolio.total_pnl >= 0 ? "+" : ""}¥{portfolio.total_pnl.toLocaleString()}
          </p></CardContent></Card>
        <Card><CardHeader className="pb-2"><CardTitle className="text-sm font-medium">账户状态</CardTitle></CardHeader>
          <CardContent><Badge variant={portfolio.status === "active" ? "success" : "warning"}>
            {portfolio.status === "active" ? "运行中" : "已暂停"}
          </Badge><p className="mt-2 text-xs text-muted-foreground">当前日期：{portfolio.current_date || "尚未运行"}</p></CardContent></Card>
      </div>

      <Card>
        <CardHeader><CardTitle className="text-base">交易复盘</CardTitle><CardDescription>基于模拟账户已成交记录的基础统计，不代表真实交易绩效。</CardDescription></CardHeader>
        <CardContent className="grid gap-3 text-sm md:grid-cols-4">
          <div><Label className="text-xs text-muted-foreground">成交笔数</Label><p className="text-lg font-semibold">{portfolio.trades.length}</p></div>
          <div><Label className="text-xs text-muted-foreground">买入 / 卖出</Label><p className="text-lg font-semibold">{buyCount} / {sellCount}</p></div>
          <div><Label className="text-xs text-muted-foreground">累计费用</Label><p className="text-lg font-semibold">¥{totalFees.toFixed(2)}</p></div>
          <div><Label className="text-xs text-muted-foreground">复盘提示</Label><p>{portfolio.trades.length ? "结合每笔决策和当时数据复盘" : "完成首笔模拟交易后生成"}</p></div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle className="text-base">模拟净值曲线</CardTitle><CardDescription>日结算快照；没有快照时说明尚未执行日结算。</CardDescription></CardHeader>
        <CardContent>{snapshots.length === 0 ? <p className="text-sm text-muted-foreground">暂无日级净值数据</p> : <ResponsiveContainer width="100%" height={240}><LineChart data={[...snapshots].reverse()}><XAxis dataKey="date" fontSize={11} /><YAxis fontSize={11} domain={["auto", "auto"]} /><Tooltip formatter={(value: number) => `¥${value.toLocaleString()}`} /><Line type="monotone" dataKey="total_value" stroke="hsl(var(--chart-1))" strokeWidth={2} dot={false} /></LineChart></ResponsiveContainer>}</CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle className="text-base">模拟账户配置</CardTitle>
          <CardDescription>当前默认使用 FastAPI Web-native 日级模拟盘；也可以切换到东方财富文件单仿真账户。</CardDescription></CardHeader>
        <CardContent className="space-y-5">
          <div className="grid gap-4 md:grid-cols-3">
            <div className="space-y-2"><Label>账户名称</Label><Input value={configDraft.name} onChange={(e) => setConfig({ name: e.target.value })} /></div>
            <div className="space-y-2"><Label>初始资金</Label><Input type="number" value={configDraft.initial_cash} onChange={(e) => setConfig({ initial_cash: Number(e.target.value) })} /></div>
            <div className="space-y-2"><Label>资产类型</Label><select className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 text-sm" value={configDraft.asset_type} onChange={(e) => setConfig({ asset_type: e.target.value as SimulationAccountConfig["asset_type"] })}><option value="stock">股票</option><option value="etf">ETF</option><option value="lof">LOF</option></select></div>
            <div className="space-y-2"><Label>基准指数</Label><Input value={configDraft.benchmark} onChange={(e) => setConfig({ benchmark: e.target.value })} /></div>
            <div className="space-y-2"><Label>成交时点</Label>
              <select className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 text-sm" value={configDraft.fill_time} onChange={(e) => setConfig({ fill_time: e.target.value as SimulationAccountConfig["fill_time"] })}>
                <option value="next_open">次日开盘</option><option value="same_close">当日收盘</option><option value="manual">手动成交</option>
              </select>
            </div>
            <div className="space-y-2"><Label>滑点 (bps)</Label><Input type="number" value={configDraft.slippage_bps} onChange={(e) => setConfig({ slippage_bps: Number(e.target.value) })} /></div>
            <div className="space-y-2"><Label>最小交易单位</Label><Input type="number" value={configDraft.min_lot} onChange={(e) => setConfig({ min_lot: Number(e.target.value) })} /></div>
            <div className="space-y-2"><Label>单股最大仓位 (%)</Label><Input type="number" min="0" max="100" value={configDraft.max_single_position_pct * 100} onChange={(e) => setConfig({ max_single_position_pct: Number(e.target.value) / 100 })} /></div>
            <div className="space-y-2"><Label>组合最大仓位 (%)</Label><Input type="number" min="0" max="100" value={configDraft.max_total_position_pct * 100} onChange={(e) => setConfig({ max_total_position_pct: Number(e.target.value) / 100 })} /></div>
            <div className="space-y-2"><Label>默认止损 (%)</Label><Input type="number" min="0" max="100" value={configDraft.default_stop_loss_pct * 100} onChange={(e) => setConfig({ default_stop_loss_pct: Number(e.target.value) / 100 })} /></div>
          </div>

          <div className="rounded-md border p-4">
            <p className="mb-3 text-sm font-medium">外部模拟平台</p>
            <div className="grid gap-4 md:grid-cols-3">
              <div className="space-y-2"><Label>Provider</Label>
                <select className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 text-sm" value={configDraft.external.provider} onChange={(e) => setConfig({ external: { ...configDraft.external, provider: e.target.value as SimulationAccountConfig["external"]["provider"] } })}>
                  <option value="internal">FastAPI Web 日级模拟</option><option value="eastmoney_file">东方财富文件单仿真</option><option value="eastmoney_emt">东方财富 EMT（预留）</option><option value="juejin">掘金（预留）</option><option value="joinquant">聚宽（预留）</option><option value="ricequant">米筐（预留）</option><option value="custom">custom（预留）</option>
                </select>
              </div>
              <div className="space-y-2"><Label>{configDraft.external.provider === "eastmoney_file" ? "文件单输入目录" : "Endpoint"}</Label><Input value={configDraft.external.provider === "eastmoney_file" ? configDraft.external.input_dir : configDraft.external.endpoint} onChange={(e) => setConfig({ external: { ...configDraft.external, ...(configDraft.external.provider === "eastmoney_file" ? { input_dir: e.target.value } : { endpoint: e.target.value }) } })} placeholder={configDraft.external.provider === "eastmoney_file" ? "例如 C:\\eastmoney\\scan_order" : ""} /></div>
              <div className="space-y-2"><Label>外部账户 ID</Label><Input value={configDraft.external.account_id} onChange={(e) => setConfig({ external: { ...configDraft.external, account_id: e.target.value } })} /></div>
              {configDraft.external.provider === "eastmoney_file" && <div className="space-y-2"><Label>文件单输出目录</Label><Input value={configDraft.external.output_dir} onChange={(e) => setConfig({ external: { ...configDraft.external, output_dir: e.target.value } })} placeholder="例如 C:\\eastmoney\\push" /></div>}
              <div className="space-y-2"><Label>Token</Label><Input type="password" placeholder={configDraft.external.token_set ? `已配置：${configDraft.external.token_masked}` : "仅在接入时填写"} value={externalToken} onChange={(e) => setExternalToken(e.target.value)} /></div>
              <div className="flex items-end gap-2"><input id="external-enabled" type="checkbox" checked={configDraft.external.enabled} onChange={(e) => setConfig({ external: { ...configDraft.external, enabled: e.target.checked } })} /><Label htmlFor="external-enabled">启用外部 Adapter 配置</Label></div>
              <div className="flex items-end gap-2"><input id="simulation-only" type="checkbox" checked={configDraft.external.simulation_only} onChange={(e) => setConfig({ external: { ...configDraft.external, simulation_only: e.target.checked } })} /><Label htmlFor="simulation-only">仅允许模拟账户</Label></div>
            </div>
            {configDraft.external.provider === "internal" && <p className="mt-3 text-xs text-muted-foreground">Web-native 模拟盘直接运行在当前 FastAPI 服务中，订单、成交、账户快照都会持久化到 SQLite。</p>}
            {configDraft.external.provider === "eastmoney_file" && <p className="mt-3 text-xs text-muted-foreground">东方财富量化终端需要开启文件单输入/输出，并连接仿真账户。项目下单写入输入目录，点击“同步”或定时任务时读取输出目录。</p>}
            {configDraft.external.provider === "eastmoney_emt" && <p className="mt-3 text-xs text-muted-foreground">东方财富 EMT 仍作为预留外部适配器；当前已实现的是 CSV 文件单仿真。</p>}
          </div>
          <div className="rounded-md border border-red-200 bg-red-50/40 p-4">
            <p className="mb-3 text-sm font-medium text-red-900">实盘 Adapter（默认关闭）</p>
            <div className="grid gap-4 md:grid-cols-3">
              <div className="space-y-2"><Label>Provider</Label><Input value={configDraft.live.provider} onChange={(e) => setConfig({ live: { ...configDraft.live, provider: e.target.value } })} placeholder="custom_http" /></div>
              <div className="space-y-2"><Label>Endpoint</Label><Input value={configDraft.live.endpoint} onChange={(e) => setConfig({ live: { ...configDraft.live, endpoint: e.target.value } })} placeholder="http://broker-sidecar:9000" /></div>
              <div className="space-y-2"><Label>实盘账户 ID</Label><Input value={configDraft.live.account_id} onChange={(e) => setConfig({ live: { ...configDraft.live, account_id: e.target.value } })} /></div>
              <div className="space-y-2"><Label>Token</Label><Input type="password" placeholder={configDraft.live.token_set ? `已配置：${configDraft.live.token_masked}` : "仅在接入时填写"} value={liveToken} onChange={(e) => setLiveToken(e.target.value)} /></div>
              <div className="space-y-2"><Label>单笔金额上限</Label><Input type="number" min="0" value={configDraft.live.max_order_value} onChange={(e) => setConfig({ live: { ...configDraft.live, max_order_value: Number(e.target.value) } })} /></div>
              <div className="flex items-end gap-4"><label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={configDraft.live.enabled} onChange={(e) => setConfig({ live: { ...configDraft.live, enabled: e.target.checked } })} />启用 Adapter</label><label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={configDraft.live.require_manual_approval} onChange={(e) => setConfig({ live: { ...configDraft.live, require_manual_approval: e.target.checked } })} />要求人工确认</label></div>
            </div>
            <p className="mt-3 text-xs text-red-800">当前仓库只定义 sidecar 协议；服务端还需开启 LIVE_TRADING_ENABLED，且自动化任务必须明确选择 live 并 armed。</p>
          </div>
          <div className="flex justify-end"><Button onClick={handleSaveConfig} disabled={saving}>{saving ? "保存中..." : "保存配置"}</Button></div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between gap-3">
            <div><CardTitle className="text-base">模拟连接状态</CardTitle><CardDescription>显示当前账户实际使用的模拟撮合后端。</CardDescription></div>
            <Badge variant={portfolio.broker.state === "connected" || portfolio.broker.state === "ready" ? "success" : portfolio.broker.state === "blocked_live_mode" ? "destructive" : "warning"}>
              {portfolio.broker.state === "connected" ? "已连接" : portfolio.broker.state === "ready" ? "配置就绪" : portfolio.broker.state === "not_installed" ? "未安装" : portfolio.broker.state === "unsupported_platform" ? "环境不支持" : portfolio.broker.state === "disabled" ? "未启用" : portfolio.broker.state}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid gap-3 text-sm md:grid-cols-4">
            <div><Label className="text-xs text-muted-foreground">后端</Label><p>{portfolio.broker.label}</p></div>
            <div><Label className="text-xs text-muted-foreground">账户</Label><p>{portfolio.broker.account_id || "—"}</p></div>
            <div><Label className="text-xs text-muted-foreground">可发模拟订单</Label><p>{portfolio.broker.can_submit_orders ? "是" : "否"}</p></div>
            <div><Label className="text-xs text-muted-foreground">运行环境</Label><p>{portfolio.broker.runtime || "—"}</p></div>
          </div>
          <p className="rounded-md border px-3 py-2 text-sm text-muted-foreground">{portfolio.broker.message}</p>
          <div className="flex items-center justify-between rounded-md border px-3 py-2 text-sm">
            <span>WebSocket 实时推送</span>
            <Badge variant={streamConnected ? "success" : "warning"}>{streamConnected ? "已连接" : "未连接，使用 REST 刷新"}</Badge>
          </div>
          {(portfolio.broker.provider === "eastmoney_file" || portfolio.broker.provider === "eastmoney_emt") && <div className="flex items-center justify-between gap-3 rounded-md bg-muted/40 p-3 text-xs text-muted-foreground"><span>{portfolio.broker.provider === "eastmoney_file" ? "项目订单会写入东方财富文件单输入目录；同步时读取账户资金、持仓、委托和成交回报。" : "东方财富网页模拟组合不等同于 EMT API 账户；当前 EMT 适配器仍为预留。"}</span><div className="flex gap-2"><Button variant="outline" size="sm" onClick={handleValidateBroker} disabled={validatingBroker}>{validatingBroker ? "处理中..." : "验证配置"}</Button>{portfolio.broker.provider === "eastmoney_file" && <Button variant="outline" size="sm" onClick={handleSyncBroker} disabled={validatingBroker}>{validatingBroker ? "同步中..." : "同步文件单"}</Button>}</div></div>}
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle className="text-base">手动模拟下单</CardTitle><CardDescription>订单会经过当前账户的交易规则校验；不连接真实资金。</CardDescription></CardHeader>
        <CardContent><div className="grid items-end gap-4 md:grid-cols-5">
          <div className="space-y-2"><Label>标的代码</Label><Input placeholder={configDraft?.asset_type === "stock" ? "000001" : configDraft?.asset_type === "etf" ? "510300" : "166009"} value={order.ticker} onChange={(e) => setOrder({ ...order, ticker: e.target.value })} /></div>
          <div className="space-y-2"><Label>方向</Label><select className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 text-sm" value={order.side} onChange={(e) => setOrder({ ...order, side: e.target.value as "buy" | "sell" })}><option value="buy">买入</option><option value="sell">卖出</option></select></div>
          <div className="space-y-2"><Label>数量</Label><Input type="number" value={order.shares} onChange={(e) => setOrder({ ...order, shares: e.target.value })} /></div>
          <div className="space-y-2"><Label>成交价（可选）</Label><Input type="number" placeholder="留空取实时价" value={order.price} onChange={(e) => setOrder({ ...order, price: e.target.value })} /></div>
          <Button onClick={handleOrder} disabled={saving || !order.ticker.trim()}>提交模拟订单</Button>
        </div></CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle className="text-base">持仓明细</CardTitle><CardDescription>当前持仓、可卖数量和盈亏；资产类型：{configDraft?.asset_type || "stock"}。</CardDescription></CardHeader>
        <CardContent>{portfolio.positions.length === 0 ? <p className="text-sm text-muted-foreground">暂无持仓</p> : <div className="overflow-x-auto"><table className="w-full text-sm"><thead><tr className="border-b text-left text-muted-foreground"><th className="pb-2 pr-4">股票代码</th><th className="pb-2 pr-4 text-right">持仓</th><th className="pb-2 pr-4 text-right">可卖</th><th className="pb-2 pr-4 text-right">成本价</th><th className="pb-2 pr-4 text-right">现价</th><th className="pb-2 pr-4 text-right">市值</th><th className="pb-2 pr-4 text-right">盈亏</th><th className="pb-2 pr-4 text-right">收益率</th></tr></thead><tbody>{portfolio.positions.map((pos) => <tr key={pos.ticker} className="border-b"><td className="py-2 pr-4 font-medium">{pos.ticker}</td><td className="py-2 pr-4 text-right">{pos.shares.toLocaleString()}</td><td className="py-2 pr-4 text-right">{pos.available_shares.toLocaleString()}</td><td className="py-2 pr-4 text-right">¥{pos.avg_cost.toFixed(2)}</td><td className="py-2 pr-4 text-right">¥{pos.current_price.toFixed(2)}</td><td className="py-2 pr-4 text-right">¥{pos.market_value.toLocaleString()}</td><td className={`py-2 pr-4 text-right ${pos.pnl >= 0 ? "text-chart-1" : "text-destructive"}`}>{pos.pnl >= 0 ? "+" : ""}¥{pos.pnl.toLocaleString()}</td><td className="py-2 pr-4 text-right"><Badge variant={pos.pnl >= 0 ? "success" : "destructive"}>{pos.pnl >= 0 ? "+" : ""}{(pos.pnl_pct * 100).toFixed(2)}%</Badge></td></tr>)}</tbody></table></div>}</CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle className="text-base">最近订单</CardTitle></CardHeader>
        <CardContent>{portfolio.orders.length === 0 ? <p className="text-sm text-muted-foreground">暂无订单</p> : <div className="space-y-2">{portfolio.orders.slice(0, 10).map((item) => <div key={item.order_id} className="flex items-center justify-between rounded-md border px-3 py-2 text-sm"><div><span>{item.ticker} · {item.side === "buy" ? "买入" : "卖出"} {item.shares} 股</span><div className="text-xs text-muted-foreground">来源：{item.source === "agent" ? "Agent" : item.source === "backtest" ? "回测" : "手动"} · {item.fill_policy === "next_open" ? "次日开盘" : item.fill_policy === "same_close" ? "当日收盘" : "手动成交"}</div></div><Badge variant={item.status === "filled" ? "success" : item.status === "rejected" ? "destructive" : "secondary"}>{item.status}</Badge></div>)}</div>}</CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle className="text-base">Web 模拟事件</CardTitle><CardDescription>订单、成交、Agent 执行和每日快照通过 WebSocket 推送。</CardDescription></CardHeader>
        <CardContent>{events.length === 0 ? <p className="text-sm text-muted-foreground">等待模拟事件...</p> : <div className="space-y-2">{events.map((event, index) => <div key={`${event.timestamp}-${index}`} className="flex items-center justify-between rounded-md border px-3 py-2 text-sm"><span>{event.type}</span><span className="text-xs text-muted-foreground">{event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : ""}</span></div>)}</div>}</CardContent>
      </Card>
    </div>
  );
}
