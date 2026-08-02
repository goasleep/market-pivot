export interface AnalysisResult {
  ticker: string;
  decision: "buy" | "sell" | "hold";
  confidence: number;
  target_price: number | null;
  stop_loss: number | null;
  position_size: number | null;
  reasoning: string;
  agent_reports: Record<string, string>;
  dashboard?: DashboardData | null;
}

export interface BacktestResult {
  ticker: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  final_value: number;
  total_return: number;
  max_drawdown: number;
  sharpe_ratio: number | null;
  win_rate: number;
  total_trades: number;
  equity_curve: { date: string; value: number }[];
  trades: TradeRecord[];
}

export interface TradeRecord {
  date: string;
  action: "buy" | "sell";
  ticker: string;
  shares: number;
  price: number;
  amount: number;
}

export interface Position {
  ticker: string;
  shares: number;
  avg_cost: number;
  current_price: number;
  market_value: number;
  pnl: number;
  pnl_pct: number;
}

export interface Portfolio {
  cash: number;
  total_value: number;
  positions: Position[];
  daily_pnl: number;
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
