export type AssetType = "stock" | "etf" | "lof";

export interface AnalysisResult {
  ticker: string;
  asset_type: AssetType;
  decision: "buy" | "sell" | "hold";
  confidence: number;
  entry_price: number | null;
  target_price: number | null;
  stop_loss: number | null;
  take_profit: number | null;
  position_size: number | null;
  plan?: TradePlan;
  reasoning: string;
  agent_reports: Record<string, string>;
  dashboard?: DashboardData | null;
  data_status?: Record<string, unknown>;
  artifacts?: Artifact[];
}

export interface PriceEvidence {
  metric: string;
  value?: unknown;
  source: string;
  as_of: string;
  calculation: string;
}

export interface TradePlan {
  entry_price: number | null;
  stop_loss: number | null;
  take_profit: number | null;
  position_size: number | null;
  position_strategy: string;
  action_items: string[];
  entry_explanation: string;
  stop_loss_explanation: string;
  take_profit_explanation: string;
  price_evidence: PriceEvidence[];
}

export interface Artifact {
  artifact_id: string;
  name: string;
  artifact_type: string;
  mime_type: string;
  size_bytes: number;
  sha256: string;
  ticker?: string | null;
  asset_type?: AssetType | null;
  source?: string;
  conversation_id?: string | null;
  task_id?: string | null;
  metadata?: Record<string, unknown>;
  created_at: string;
  preview_url: string;
  download_url: string;
}

export type BacktestMode = "auto" | "single" | "pool" | "portfolio";

export interface PortfolioSpec {
  allocation_method: "equal_weight";
  rebalance_frequency: "daily" | "weekly" | "monthly" | "manual";
  max_position_weight: number;
  max_positions: number;
  cash_reserve: number;
}

export interface BacktestResult {
  ticker: string;
  mode?: "single" | "pool" | "portfolio";
  tickers?: string[];
  asset_type: AssetType;
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
  strategy_spec?: Record<string, unknown> | null;
  portfolio_spec?: PortfolioSpec | null;
  portfolio_history?: Array<Record<string, unknown>>;
  target_weights_history?: Array<Record<string, unknown>>;
  symbol_metrics?: Array<Record<string, unknown>>;
  data_snapshot?: Record<string, unknown> | null;
  data_snapshots?: Record<string, unknown>[];
  data_rejections?: Array<Record<string, unknown>>;
  buy_hold_return?: number;
  error?: string | null;
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
  external_id?: string | null;
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
  status: "active" | "paused" | "archived";
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
  provider:
    | "internal"
    | "eastmoney_emt"
    | "eastmoney_file"
    | "juejin"
    | "joinquant"
    | "ricequant"
    | "custom";
  enabled: boolean;
  simulation_only: boolean;
  endpoint: string;
  account_id: string;
  input_dir: string;
  output_dir: string;
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
  deployment_id?: string | null;
  decision_id?: string | null;
  fill_policy?: "next_open" | "same_close" | "manual";
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
    entry_explanation: string;
    stop_loss_explanation: string;
    take_profit_explanation: string;
    price_evidence: PriceEvidence[];
  };
  strategy_plan: {
    name: string;
    thesis: string;
    entry_conditions: string[];
    exit_conditions: string[];
    indicators_used: string[];
    data_basis: PriceEvidence[];
    spec?: {
      name: string;
      version: string;
      asset_types: AssetType[];
      indicators: string[];
      entry_conditions: Array<Record<string, unknown>>;
      exit_conditions: Array<Record<string, unknown>>;
      stop_loss_pct?: number | null;
      take_profit_pct?: number | null;
      position_size_pct: number;
      rebalance_frequency: "daily" | "weekly" | "manual";
      source: "yaml" | "llm" | "user";
    } | null;
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
