import type {
  Portfolio,
  SimulationAccountConfig,
  SimulationOrder,
  SimulationBrokerStatus,
  LiveBrokerStatus,
  SimulationEvent,
  AutomationTask,
  AutomationTaskConfig,
  AgentRunSummary,
  AgentDecisionAudit,
  SimulationSnapshot,
  AssetType,
  Artifact,
  BacktestMode,
  BacktestResult,
  BacktestExperimentResult,
  PortfolioSpec,
} from "@/types";

const BASE_URL = "/api";

export async function getArtifacts(limit = 100): Promise<Artifact[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  const res = await fetch(`${BASE_URL}/artifacts?${params.toString()}`);
  if (!res.ok) throw new Error(`Failed to fetch artifacts: ${res.statusText}`);
  const data = (await res.json()) as { artifacts?: Artifact[] };
  return data.artifacts || [];
}

export async function runBacktest(params: {
  mode?: BacktestMode;
  ticker?: string;
  tickers?: string[];
  start_date: string;
  end_date: string;
  initial_capital?: number;
  decision_interval?: number;
  fill_time?: "next_open" | "same_close";
  strategy?: string;
  strategy_spec?: Record<string, unknown>;
  portfolio_spec?: PortfolioSpec;
  asset_type?: AssetType;
}): Promise<BacktestResult> {
  const res = await fetch(`${BASE_URL}/backtest/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...params,
      mode: params.mode ?? "auto",
      initial_capital: params.initial_capital ?? 1_000_000,
      decision_interval: params.decision_interval ?? 1,
      fill_time: params.fill_time ?? "next_open",
      asset_type: params.asset_type ?? "stock",
    }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || `Backtest failed: ${res.statusText}`);
  }
  return res.json();
}

export async function runBacktestExperiment(params: {
  objective: string;
  mode?: BacktestMode;
  ticker?: string;
  tickers?: string[];
  start_date: string;
  end_date: string;
  initial_capital?: number;
  decision_interval?: number;
  fill_time?: "next_open" | "same_close";
  strategy_name?: string;
  strategy_spec?: Record<string, unknown>;
  portfolio_spec?: PortfolioSpec;
  asset_type?: AssetType;
}): Promise<BacktestExperimentResult> {
  const res = await fetch(`${BASE_URL}/backtest/experiments`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...params,
      mode: params.mode ?? "auto",
      initial_capital: params.initial_capital ?? 1_000_000,
      decision_interval: params.decision_interval ?? 1,
      fill_time: params.fill_time ?? "next_open",
      asset_type: params.asset_type ?? "stock",
    }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || `Backtest experiment failed: ${res.statusText}`);
  }
  return res.json();
}

export async function getPortfolio(accountId = "default"): Promise<Portfolio> {
  const params = new URLSearchParams({ account_id: accountId });
  const res = await fetch(`${BASE_URL}/portfolio/?${params.toString()}`);
  if (!res.ok) throw new Error(`Failed to fetch portfolio: ${res.statusText}`);
  return res.json();
}

export async function getMarketQuote(ticker: string, assetType: AssetType = "stock"): Promise<{ ticker: string; asset_type: AssetType; quote: Record<string, unknown>; available: boolean }> {
  const params = new URLSearchParams({ ticker, asset_type: assetType });
  const res = await fetch(`${BASE_URL}/market/quote?${params.toString()}`);
  if (!res.ok) throw new Error(`Failed to fetch quote: ${res.statusText}`);
  return res.json();
}

export async function getSimulationAccounts(): Promise<Portfolio[]> {
  const res = await fetch(`${BASE_URL}/portfolio/accounts`);
  if (!res.ok) throw new Error(`Failed to fetch simulation accounts: ${res.statusText}`);
  const data = await res.json();
  return data.accounts;
}

export async function getSimulationSnapshots(accountId = "default"): Promise<SimulationSnapshot[]> {
  const res = await fetch(`${BASE_URL}/portfolio/accounts/${encodeURIComponent(accountId)}/snapshots?limit=500`);
  if (!res.ok) throw new Error(`Failed to fetch simulation snapshots: ${res.statusText}`);
  const data = await res.json();
  return data.snapshots;
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

export async function getSimulationBrokerStatus(accountId = "default"): Promise<SimulationBrokerStatus> {
  const res = await fetch(`${BASE_URL}/portfolio/accounts/${encodeURIComponent(accountId)}/broker`);
  if (!res.ok) throw new Error(`Failed to fetch broker status: ${res.statusText}`);
  return res.json();
}

export async function validateSimulationBroker(accountId = "default"): Promise<SimulationBrokerStatus> {
  const res = await fetch(`${BASE_URL}/portfolio/accounts/${encodeURIComponent(accountId)}/broker/validate`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(`Failed to validate broker: ${res.statusText}`);
  return res.json();
}

export async function syncSimulationBroker(accountId = "default"): Promise<Portfolio> {
  const res = await fetch(`${BASE_URL}/portfolio/accounts/${encodeURIComponent(accountId)}/broker/sync`, {
    method: "POST",
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || `Failed to sync simulation broker: ${res.statusText}`);
  }
  return res.json();
}

export async function updateLiveTradingConfig(
  accountId: string,
  config: Record<string, unknown>
): Promise<Portfolio> {
  const res = await fetch(`${BASE_URL}/portfolio/accounts/${encodeURIComponent(accountId)}/live`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || `Failed to update live trading config: ${res.statusText}`);
  }
  return res.json();
}

export async function getLiveBrokerStatus(accountId = "default"): Promise<LiveBrokerStatus> {
  const portfolio = await getPortfolio(accountId);
  return portfolio.live_broker;
}

export function openSimulationStream(
  accountId: string,
  onEvent: (event: SimulationEvent) => void,
  onError?: () => void
): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(
    `${protocol}//${window.location.host}${BASE_URL}/portfolio/accounts/${encodeURIComponent(accountId)}/stream`
  );
  socket.onmessage = (message) => {
    try {
      onEvent(JSON.parse(message.data) as SimulationEvent);
    } catch {
      onError?.();
    }
  };
  socket.onerror = () => onError?.();
  return socket;
}

export async function createSimulationOrder(
  accountId: string,
  order: {
    ticker: string;
    asset_type?: AssetType;
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

async function automationRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}/automation${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    ...init,
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || `Automation request failed: ${res.statusText}`);
  }
  return res.json();
}

export function getAutomationTask(accountId = "default"): Promise<AutomationTask> {
  return automationRequest(`/accounts/${encodeURIComponent(accountId)}`);
}

export function updateAutomationTask(
  accountId: string,
  config: AutomationTaskConfig
): Promise<AutomationTask> {
  return automationRequest(`/accounts/${encodeURIComponent(accountId)}`, {
    method: "PUT",
    body: JSON.stringify(config),
  });
}

export function runAutomation(
  accountId = "default",
  runDate?: string
): Promise<AgentRunSummary> {
  return automationRequest(`/accounts/${encodeURIComponent(accountId)}/run`, {
    method: "POST",
    body: JSON.stringify({ run_date: runDate }),
  });
}

export function settleAutomation(
  accountId = "default",
  payload: { settlement_date?: string; prices?: Record<string, number>; open_prices?: Record<string, number> } = {}
): Promise<Record<string, unknown>> {
  return automationRequest(`/accounts/${encodeURIComponent(accountId)}/settle`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function syncLiveAccount(accountId = "default"): Promise<Record<string, unknown>> {
  return automationRequest(`/accounts/${encodeURIComponent(accountId)}/live-sync`, {
    method: "POST",
  });
}

export async function getAutomationRuns(accountId = "default"): Promise<AgentRunSummary[]> {
  const data = await automationRequest<{ runs: AgentRunSummary[] }>(
    `/accounts/${encodeURIComponent(accountId)}/runs`
  );
  return data.runs;
}

export async function getAutomationDecisions(accountId = "default"): Promise<AgentDecisionAudit[]> {
  const data = await automationRequest<{ decisions: AgentDecisionAudit[] }>(
    `/accounts/${encodeURIComponent(accountId)}/decisions`
  );
  return data.decisions;
}

export function confirmAutomationDecision(
  accountId: string,
  decisionId: string,
  price?: number
): Promise<AgentDecisionAudit> {
  const query = price ? `?price=${encodeURIComponent(price)}` : "";
  return automationRequest(`/accounts/${encodeURIComponent(accountId)}/decisions/${encodeURIComponent(decisionId)}/confirm${query}`, { method: "POST" });
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
