import type {
  AnalysisResult,
  BacktestResult,
  Portfolio,
  SSEProgress,
  SimulationAccountConfig,
  SimulationOrder,
} from "@/types";

const BASE_URL = "/api";

export async function runAnalysis(
  ticker: string,
  strategy?: string
): Promise<AnalysisResult> {
  const res = await fetch(`${BASE_URL}/analysis/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ticker, show_reasoning: true, strategy }),
  });
  if (!res.ok) throw new Error(`Analysis failed: ${res.statusText}`);
  return res.json();
}

export function streamAnalysis(
  ticker: string,
  onProgress: (data: SSEProgress) => void,
  onComplete: (data: AnalysisResult) => void,
  onError: (err: EventSource) => void
): EventSource {
  // SSE doesn't support POST, so we use a GET with query params for streaming
  const url = `${BASE_URL}/analysis/stream?ticker=${encodeURIComponent(ticker)}`;
  const es = new EventSource(url);

  es.addEventListener("progress", (e) => {
    onProgress(JSON.parse(e.data));
  });
  es.addEventListener("complete", (e) => {
    onComplete(JSON.parse(e.data));
    es.close();
  });
  es.onerror = () => {
    onError(es);
    es.close();
  };

  return es;
}

export async function runBacktest(params: {
  ticker: string;
  start_date: string;
  end_date: string;
  initial_capital?: number;
  decision_interval?: number;
  fill_time?: "next_open" | "same_close";
  strategy?: string;
}): Promise<BacktestResult> {
  const res = await fetch(`${BASE_URL}/backtest/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ticker: params.ticker,
      start_date: params.start_date,
      end_date: params.end_date,
      initial_capital: params.initial_capital ?? 1_000_000,
      decision_interval: params.decision_interval ?? 1,
      fill_time: params.fill_time ?? "next_open",
      strategy: params.strategy,
    }),
  });
  if (!res.ok) throw new Error(`Backtest failed: ${res.statusText}`);
  return res.json();
}

export async function getPortfolio(): Promise<Portfolio> {
  const res = await fetch(`${BASE_URL}/portfolio/`);
  if (!res.ok) throw new Error(`Failed to fetch portfolio: ${res.statusText}`);
  return res.json();
}

export async function getSimulationAccounts(): Promise<Portfolio[]> {
  const res = await fetch(`${BASE_URL}/portfolio/accounts`);
  if (!res.ok) throw new Error(`Failed to fetch simulation accounts: ${res.statusText}`);
  const data = await res.json();
  return data.accounts;
}

export async function updateSimulationConfig(
  accountId: string,
  config: SimulationAccountConfig
): Promise<Portfolio> {
  const res = await fetch(`${BASE_URL}/portfolio/accounts/${encodeURIComponent(accountId)}/config`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || `Failed to update simulation config: ${res.statusText}`);
  }
  return res.json();
}

export async function resetSimulationAccount(accountId = "default"): Promise<Portfolio> {
  const res = await fetch(`${BASE_URL}/portfolio/accounts/${encodeURIComponent(accountId)}/reset`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(`Failed to reset simulation account: ${res.statusText}`);
  return res.json();
}

export async function updateExternalSimulationConfig(
  accountId: string,
  external: Record<string, unknown>
): Promise<Portfolio> {
  const res = await fetch(`${BASE_URL}/portfolio/accounts/${encodeURIComponent(accountId)}/external`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(external),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || `Failed to update external simulation config: ${res.statusText}`);
  }
  return res.json();
}

export async function createSimulationOrder(
  accountId: string,
  order: {
    ticker: string;
    side: "buy" | "sell";
    shares: number;
    price?: number;
    fill_immediately?: boolean;
    trade_date?: string;
  }
): Promise<SimulationOrder> {
  const res = await fetch(`${BASE_URL}/portfolio/accounts/${encodeURIComponent(accountId)}/orders`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(order),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || `Failed to create simulation order: ${res.statusText}`);
  }
  return res.json();
}

export async function resetPortfolio(): Promise<{ status: string; message: string }> {
  const res = await fetch(`${BASE_URL}/portfolio/reset`, { method: "POST" });
  if (!res.ok) throw new Error(`Failed to reset: ${res.statusText}`);
  return res.json();
}

// --- Strategy APIs ---

export interface StrategyInfo {
  name: string;
  display_name: string;
  description: string;
  category: string;
  default_active: boolean;
  default_router: boolean;
  priority: number;
  market_regimes: string[];
  aliases: string[];
}

export async function getStrategies(): Promise<StrategyInfo[]> {
  const res = await fetch(`${BASE_URL}/strategies`);
  if (!res.ok) throw new Error(`Failed to fetch strategies: ${res.statusText}`);
  const data = await res.json();
  return data.strategies;
}

// --- System Status API ---

export interface BreakerStatus {
  [key: string]: string;
}

export async function getSystemStatus(): Promise<BreakerStatus> {
  const res = await fetch(`${BASE_URL}/system/status`);
  if (!res.ok) throw new Error(`Failed to fetch status: ${res.statusText}`);
  const data = await res.json();
  return data.circuit_breakers;
}

// --- LLM Config APIs ---

export interface LLMModelInfo {
  description: string;
  max_tokens: number;
}

export interface LLMConfig {
  api_key_masked: string;
  api_key_set: boolean;
  base_url: string;
  model: string;
  temperature: number;
  max_tokens: number;
  available_models: Record<string, LLMModelInfo>;
}

export interface LLMConfigUpdate {
  api_key?: string;
  base_url?: string;
  model?: string;
  temperature?: number;
  max_tokens?: number;
}

export async function getLLMConfig(): Promise<LLMConfig> {
  const res = await fetch(`${BASE_URL}/config/llm`);
  if (!res.ok) throw new Error(`Failed to fetch LLM config: ${res.statusText}`);
  return res.json();
}

export async function updateLLMConfig(update: LLMConfigUpdate): Promise<LLMConfig> {
  const res = await fetch(`${BASE_URL}/config/llm`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(update),
  });
  if (!res.ok) throw new Error(`Failed to update LLM config: ${res.statusText}`);
  return res.json();
}
