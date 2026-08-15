export type AssetType = "stock" | "etf" | "lof";

export interface AnalysisResult {
  ticker: string;
  asset_type: AssetType;
  decision: "buy" | "sell" | "hold";
  confidence: number;
  target_price: number | null;
  stop_loss: number | null;
  position_size: number | null;
  reasoning: string;
  agent_reports: Record<string, string>;
  dashboard?: DashboardData | null;
  data_status?: Record<string, unknown>;
}

export interface BacktestResult {
  ticker: string;
  tickers?: string[];
  start_date: string;
  end_date: string;
  initial_capital: number;
  final_value: number;
  total_return: number;
  max_drawdown: number;
  sharpe_ratio: number | null;
  win_rate: number;
  realized_pnl?: number;
  total_fees?: number;
  total_trades: number;
  equity_curve: { date: string; value: number }[];
  trades: TradeRecord[];
}

export interface TradeRecord {
  date: string;
  action: "buy" | "sell";
  ticker: string;
  asset_type?: AssetType;
  shares: number;
  price: number;
  amount: number;
  commission?: number;
  tax?: number;
}

export interface Position {
  ticker: string;
  asset_type?: AssetType;
  shares: number;
  avg_cost: number;
  current_price: number;
  available_shares: number;
  frozen_shares: number;
  market_value: number;
  pnl: number;
  pnl_pct: number;
}

export interface Portfolio {
  account_id: string;
  name: string;
  status: "active" | "paused";
  current_date: string;
  initial_capital: number;
  total_pnl: number;
  total_return_pct: number;
  cash: number;
  total_value: number;
  positions: Position[];
  trades: TradeRecord[];
  orders: SimulationOrder[];
  config: SimulationAccountConfig;
  daily_pnl: number;
  broker: SimulationBrokerStatus;
  live_broker: LiveBrokerStatus;
}

export interface SimulationSnapshot {
  account_id: string;
  date: string;
  cash: number;
  total_value: number;
  total_pnl: number;
  total_return_pct: number;
  positions: Position[];
}

export interface SimulationBrokerStatus {
  provider: string;
  label: string;
  enabled: boolean;
  simulation_only: boolean;
  account_id?: string;
  state: string;
  connected: boolean;
  can_submit_orders: boolean;
  installed: boolean;
  supported_platform: boolean;
  runtime: string;
  endpoint_configured: boolean;
  message: string;
  requirements: string[];
}

export interface LiveBrokerStatus {
  provider: string;
  enabled: boolean;
  configured: boolean;
  can_submit_orders: boolean;
  state: string;
  message: string;
}

export interface SimulationEvent {
  type: string;
  account_id: string;
  timestamp?: string;
  data: Record<string, unknown>;
}

export interface ExternalSimulationConfig {
  provider: "internal" | "eastmoney_emt" | "juejin" | "joinquant" | "ricequant" | "custom";
  enabled: boolean;
  simulation_only: boolean;
  endpoint: string;
  account_id: string;
  token_set: boolean;
  token_masked: string;
  options: Record<string, unknown>;
}

export interface SimulationAccountConfig {
  name: string;
  initial_cash: number;
  execution_frequency: "1d";
  signal_time: "after_close";
  fill_time: "next_open" | "same_close" | "manual";
  slippage_bps: number;
  buy_commission_rate: number;
  sell_commission_rate: number;
  minimum_commission: number;
  stamp_tax_rate: number;
  transfer_fee_rate: number;
  min_lot: number;
  max_single_position_pct: number;
  max_total_position_pct: number;
  default_stop_loss_pct: number;
  benchmark: string;
  asset_type: AssetType;
  universe: string[];
  external: ExternalSimulationConfig;
  live: LiveTradingConfig;
}

export interface LiveTradingConfig {
  provider: string;
  enabled: boolean;
  account_id: string;
  endpoint: string;
  token_set: boolean;
  token_masked: string;
  require_manual_approval: boolean;
  max_order_value: number;
}

export interface SimulationOrder {
  order_id: string;
  account_id: string;
  ticker: string;
  asset_type?: AssetType;
  side: "buy" | "sell";
  shares: number;
  order_type: "market" | "limit";
  limit_price: number | null;
  status: "pending" | "filled" | "rejected" | "cancelled";
  submitted_date: string;
  fill_date: string | null;
  fill_price: number | null;
  reject_reason: string | null;
  source?: "manual" | "agent" | "backtest" | "system";
  run_id?: string | null;
  fill_policy?: "next_open" | "same_close" | "manual";
}

export type AutomationMode = "observe" | "confirm" | "auto";

export interface AutomationTaskConfig {
  enabled: boolean;
  mode: AutomationMode;
  execution_mode: "paper" | "live";
  live_armed: boolean;
  schedule_time: string;
  weekdays: number[];
  universe: string[];
  asset_type: AssetType;
  strategy_name: string | null;
  max_symbols_per_run: number;
  max_orders_per_run: number;
  daily_loss_limit_pct: number;
  data_max_age_seconds: number;
  fill_time: "next_open" | "same_close" | "manual";
  simulation_only: boolean;
}

export interface AutomationTask {
  account_id: string;
  config: AutomationTaskConfig;
  status: string;
  last_run_id: string | null;
  last_run_date: string | null;
  last_error: string | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface AgentRunSummary {
  run_id: string;
  account_id: string;
  run_date: string;
  trigger: "schedule" | "manual" | "retry" | "settlement";
  status: "queued" | "running" | "completed" | "failed" | "cancelled" | "skipped";
  mode: AutomationMode;
  strategy_name: string | null;
  symbols_total: number;
  symbols_processed: number;
  decisions_count: number;
  orders_count: number;
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
  idempotency_key: string;
}

export interface AgentDecisionAudit {
  decision_id: string;
  run_id: string;
  account_id: string;
  ticker: string;
  decision: AnalysisResult;
  current_price: number;
  risk_status: "pending" | "approved" | "rejected";
  risk_reason: string | null;
  order_id: string | null;
  created_at: string;
}

export interface SSEProgress {
  stage: string;
  message: string;
}

// --- Chat / A2UI types ---

export interface DashboardData {
  core_conclusion: {
    signal: string;
    confidence: number;
    one_line_summary: string;
    position_advice: string;
  };
  data_perspective: {
    trend_status: string;
    price_position: string;
    volume_analysis: string;
    chip_structure: string;
  };
  intelligence: {
    latest_news: string[];
    risk_alerts: string[];
    positive_catalysts: string[];
    earnings_outlook: string;
  };
  battle_plan: {
    entry_price: number | null;
    stop_loss: number | null;
    take_profit: number | null;
    position_strategy: string;
    action_items: string[];
  };
  phase_decision: {
    pre_market: string;
    intraday: string;
    post_market: string;
  };
  signal_attribution: {
    technical_score: number;
    sentiment_score: number;
    fundamental_score: number;
    market_regime_score: number;
  };
}

export interface ChatWidgetEvent {
  type: string;
  html: string;
}

export interface ChatTextEvent {
  text: string;
}
