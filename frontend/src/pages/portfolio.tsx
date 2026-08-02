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
  updateSimulationConfig,
} from "@/api";
import type { Portfolio, SimulationAccountConfig } from "@/types";
import { RefreshCw, RotateCcw } from "lucide-react";

const cloneConfig = (config: SimulationAccountConfig): SimulationAccountConfig => ({
  ...config,
  universe: [...config.universe],
  external: { ...config.external },
});

export function PortfolioPage() {
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [configDraft, setConfigDraft] = useState<SimulationAccountConfig | null>(null);
  const [externalToken, setExternalToken] = useState("");
  const [order, setOrder] = useState({ ticker: "", side: "buy" as "buy" | "sell", shares: "100", price: "" });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const fetchPortfolio = async () => {
    setLoading(true);
    try {
      const data = await getPortfolio();
      setPortfolio(data);
      setConfigDraft(cloneConfig(data.config));
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "加载模拟账户失败");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPortfolio();
  }, []);

  const handleSaveConfig = async () => {
    if (!portfolio || !configDraft) return;
    setSaving(true);
    setMessage(null);
    try {
      let updated = await updateSimulationConfig(portfolio.account_id, configDraft);
      if (externalToken.trim()) {
        updated = await updateExternalSimulationConfig(portfolio.account_id, {
          ...configDraft.external,
          token: externalToken.trim(),
        });
        setExternalToken("");
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
    if (!portfolio || !order.ticker.trim() || Number(order.shares) <= 0) return;
    setSaving(true);
    setMessage(null);
    try {
      await createSimulationOrder(portfolio.account_id, {
        ticker: order.ticker.trim(),
        side: order.side,
        shares: Number(order.shares),
        price: order.price ? Number(order.price) : undefined,
        fill_immediately: true,
        trade_date: portfolio.current_date || undefined,
      });
      setOrder({ ...order, ticker: "", price: "" });
      await fetchPortfolio();
      setMessage("模拟订单已提交并成交");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "模拟下单失败");
    } finally {
      setSaving(false);
    }
  };

  if (loading || !portfolio || !configDraft) {
    return <div className="p-6"><p className="text-sm text-muted-foreground">加载中...</p></div>;
  }

  const setConfig = (patch: Partial<SimulationAccountConfig>) => {
    setConfigDraft({ ...configDraft, ...patch });
  };

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
        <CardHeader><CardTitle className="text-base">模拟账户配置</CardTitle>
          <CardDescription>配置只作用于内部模拟账户；外部平台配置仅保存，不会自动连接。</CardDescription></CardHeader>
        <CardContent className="space-y-5">
          <div className="grid gap-4 md:grid-cols-3">
            <div className="space-y-2"><Label>账户名称</Label><Input value={configDraft.name} onChange={(e) => setConfig({ name: e.target.value })} /></div>
            <div className="space-y-2"><Label>初始资金</Label><Input type="number" value={configDraft.initial_cash} onChange={(e) => setConfig({ initial_cash: Number(e.target.value) })} /></div>
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
            <p className="mb-3 text-sm font-medium">外部模拟平台（预留）</p>
            <div className="grid gap-4 md:grid-cols-3">
              <div className="space-y-2"><Label>Provider</Label>
                <select className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 text-sm" value={configDraft.external.provider} onChange={(e) => setConfig({ external: { ...configDraft.external, provider: e.target.value as SimulationAccountConfig["external"]["provider"] } })}>
                  <option value="internal">internal</option><option value="juejin">掘金</option><option value="joinquant">聚宽</option><option value="ricequant">米筐</option><option value="custom">custom</option>
                </select>
              </div>
              <div className="space-y-2"><Label>Endpoint</Label><Input value={configDraft.external.endpoint} onChange={(e) => setConfig({ external: { ...configDraft.external, endpoint: e.target.value } })} /></div>
              <div className="space-y-2"><Label>外部账户 ID</Label><Input value={configDraft.external.account_id} onChange={(e) => setConfig({ external: { ...configDraft.external, account_id: e.target.value } })} /></div>
              <div className="space-y-2"><Label>Token</Label><Input type="password" placeholder={configDraft.external.token_set ? `已配置：${configDraft.external.token_masked}` : "仅在接入时填写"} value={externalToken} onChange={(e) => setExternalToken(e.target.value)} /></div>
              <div className="flex items-end gap-2"><input id="external-enabled" type="checkbox" checked={configDraft.external.enabled} onChange={(e) => setConfig({ external: { ...configDraft.external, enabled: e.target.checked } })} /><Label htmlFor="external-enabled">启用外部 Adapter 配置</Label></div>
            </div>
          </div>
          <div className="flex justify-end"><Button onClick={handleSaveConfig} disabled={saving}>{saving ? "保存中..." : "保存配置"}</Button></div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle className="text-base">手动模拟下单</CardTitle><CardDescription>订单会经过当前账户的交易规则校验；不连接真实资金。</CardDescription></CardHeader>
        <CardContent><div className="grid items-end gap-4 md:grid-cols-5">
          <div className="space-y-2"><Label>股票代码</Label><Input placeholder="000001" value={order.ticker} onChange={(e) => setOrder({ ...order, ticker: e.target.value })} /></div>
          <div className="space-y-2"><Label>方向</Label><select className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 text-sm" value={order.side} onChange={(e) => setOrder({ ...order, side: e.target.value as "buy" | "sell" })}><option value="buy">买入</option><option value="sell">卖出</option></select></div>
          <div className="space-y-2"><Label>数量</Label><Input type="number" value={order.shares} onChange={(e) => setOrder({ ...order, shares: e.target.value })} /></div>
          <div className="space-y-2"><Label>成交价（可选）</Label><Input type="number" placeholder="留空取实时价" value={order.price} onChange={(e) => setOrder({ ...order, price: e.target.value })} /></div>
          <Button onClick={handleOrder} disabled={saving || !order.ticker.trim()}>提交模拟订单</Button>
        </div></CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle className="text-base">持仓明细</CardTitle><CardDescription>当前持仓、可卖数量和盈亏。</CardDescription></CardHeader>
        <CardContent>{portfolio.positions.length === 0 ? <p className="text-sm text-muted-foreground">暂无持仓</p> : <div className="overflow-x-auto"><table className="w-full text-sm"><thead><tr className="border-b text-left text-muted-foreground"><th className="pb-2 pr-4">股票代码</th><th className="pb-2 pr-4 text-right">持仓</th><th className="pb-2 pr-4 text-right">可卖</th><th className="pb-2 pr-4 text-right">成本价</th><th className="pb-2 pr-4 text-right">现价</th><th className="pb-2 pr-4 text-right">市值</th><th className="pb-2 pr-4 text-right">盈亏</th><th className="pb-2 pr-4 text-right">收益率</th></tr></thead><tbody>{portfolio.positions.map((pos) => <tr key={pos.ticker} className="border-b"><td className="py-2 pr-4 font-medium">{pos.ticker}</td><td className="py-2 pr-4 text-right">{pos.shares.toLocaleString()}</td><td className="py-2 pr-4 text-right">{pos.available_shares.toLocaleString()}</td><td className="py-2 pr-4 text-right">¥{pos.avg_cost.toFixed(2)}</td><td className="py-2 pr-4 text-right">¥{pos.current_price.toFixed(2)}</td><td className="py-2 pr-4 text-right">¥{pos.market_value.toLocaleString()}</td><td className={`py-2 pr-4 text-right ${pos.pnl >= 0 ? "text-chart-1" : "text-destructive"}`}>{pos.pnl >= 0 ? "+" : ""}¥{pos.pnl.toLocaleString()}</td><td className="py-2 pr-4 text-right"><Badge variant={pos.pnl >= 0 ? "success" : "destructive"}>{pos.pnl >= 0 ? "+" : ""}{(pos.pnl_pct * 100).toFixed(2)}%</Badge></td></tr>)}</tbody></table></div>}</CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle className="text-base">最近订单</CardTitle></CardHeader>
        <CardContent>{portfolio.orders.length === 0 ? <p className="text-sm text-muted-foreground">暂无订单</p> : <div className="space-y-2">{portfolio.orders.slice(0, 10).map((item) => <div key={item.order_id} className="flex items-center justify-between rounded-md border px-3 py-2 text-sm"><span>{item.ticker} · {item.side === "buy" ? "买入" : "卖出"} {item.shares} 股</span><Badge variant={item.status === "filled" ? "success" : item.status === "rejected" ? "destructive" : "secondary"}>{item.status}</Badge></div>)}</div>}</CardContent>
      </Card>
    </div>
  );
}
