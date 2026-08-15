"""Stock-focused conversational agent and task router.

The StockAgent keeps the chat surface focused on stock research tasks while
delegating analysis to the existing multi-agent LangGraph workflow.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Sequence

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import StructuredTool

from agents.chat_tools import build_chat_tools
from data.market_context import build_market_context
from graph.agent_loop import stream_agent_loop
from graph.workflow import workflow
from models.schemas import AssetType
from observability import build_trace_config


class StockIntent(str, Enum):
    ANALYZE = "analyze"
    QUOTE = "quote"
    HISTORY = "history"
    NEWS = "news"
    STRATEGIES = "strategies"
    PORTFOLIO = "portfolio"
    BACKTEST = "backtest"
    COMPARE = "compare"
    HELP = "help"


@dataclass(frozen=True)
class StockAgentRequest:
    message: str
    history: list[dict[str, str]]
    intent: StockIntent
    tickers: tuple[str, ...]
    asset_type: AssetType = AssetType.STOCK
    strategy: str | None = None
    conversation_id: str | None = None

    @property
    def ticker(self) -> str | None:
        return self.tickers[0] if self.tickers else None


class StockAgent:
    """Route conversational requests to stock data and analysis capabilities."""

    _ticker_pattern = re.compile(r"(?<!\d)(?:(?:sh|sz|bj)\s*)?(\d{6})(?!\d)", re.IGNORECASE)

    _keyword_groups = {
        StockIntent.BACKTEST: ("回测", "回测一下", "策略测试", "历史测试", "backtest"),
        StockIntent.COMPARE: ("对比", "比较", "compare", "vs", " versus "),
        StockIntent.NEWS: ("新闻", "消息", "舆情", "资讯", "news"),
        StockIntent.HISTORY: ("历史", "k线", "走势", "行情走势", "历史价格", "history", "chart"),
        StockIntent.QUOTE: ("实时", "现价", "报价", "行情", "价格", "quote"),
        StockIntent.STRATEGIES: ("策略", "选股", "交易规则", "strategies"),
        StockIntent.PORTFOLIO: ("持仓", "组合", "仓位", "账户", "portfolio"),
        StockIntent.ANALYZE: ("分析", "估值", "基本面", "技术面", "财报", "趋势", "买入", "卖出", "建议", "analy"),
    }

    @classmethod
    def extract_tickers(cls, *texts: str) -> tuple[str, ...]:
        """Extract normalized six-digit A-share codes in first-seen order."""
        found: list[str] = []
        for text in texts:
            for match in cls._ticker_pattern.finditer(text or ""):
                ticker = match.group(1)
                if ticker not in found:
                    found.append(ticker)
        return tuple(found)

    def resolve(
        self,
        message: str,
        history: Sequence[dict[str, str]] | None = None,
        strategy: str | None = None,
        conversation_id: str | None = None,
        asset_type: AssetType | str | None = None,
    ) -> StockAgentRequest:
        """Resolve intent and reuse the last ticker for conversational follow-ups."""
        history_items = list(history or [])
        current_tickers = self.extract_tickers(message)
        history_tickers = self.extract_tickers(*(item.get("content", "") for item in reversed(history_items)))
        tickers = current_tickers or history_tickers[:1]
        intent = self._infer_intent(message, len(current_tickers))
        asset_type = AssetType(asset_type) if asset_type else self._infer_asset_type(message, history_items)
        return StockAgentRequest(
            message=message,
            history=history_items,
            intent=intent,
            tickers=tickers,
            asset_type=asset_type,
            strategy=strategy,
            conversation_id=conversation_id,
        )

    def _infer_intent(self, message: str, current_ticker_count: int) -> StockIntent:
        text = f" {message.lower()} "
        if not message.strip():
            return StockIntent.HELP
        if any(keyword in text for keyword in self._keyword_groups[StockIntent.BACKTEST]):
            return StockIntent.BACKTEST
        if current_ticker_count > 1 or any(keyword in text for keyword in self._keyword_groups[StockIntent.COMPARE]):
            return StockIntent.COMPARE
        for intent in (
            StockIntent.NEWS,
            StockIntent.HISTORY,
            StockIntent.QUOTE,
            StockIntent.STRATEGIES,
            StockIntent.PORTFOLIO,
            StockIntent.ANALYZE,
        ):
            if any(keyword in text for keyword in self._keyword_groups[intent]):
                return intent
        follow_up_keywords = ("为什么", "依据", "解释", "怎么看", "如果", "还能", "风险", "止损")
        if (
            current_ticker_count
            or self.extract_tickers(message)
            or any(keyword in text for keyword in follow_up_keywords)
        ):
            return StockIntent.ANALYZE
        return StockIntent.HELP

    @staticmethod
    def _infer_asset_type(message: str, history: Sequence[dict[str, str]]) -> AssetType:
        text = " ".join([message, *(item.get("content", "") for item in history[-6:])]).lower()
        if any(token in text for token in ("lof", "场内基金", "lof基金")):
            return AssetType.LOF
        if any(token in text for token in ("etf", "交易型基金", "指数基金")):
            return AssetType.ETF
        return AssetType.STOCK

    def prepare(
        self,
        message: str,
        history: Sequence[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> StockAgentRequest:
        """Extract only safe context; intent is selected by the LLM, not keywords."""
        history_items = list(history or [])
        current_tickers = self.extract_tickers(message)
        history_tickers = self.extract_tickers(*(item.get("content", "") for item in reversed(history_items)))
        return StockAgentRequest(
            message=message,
            history=history_items,
            intent=StockIntent.ANALYZE,
            tickers=current_tickers or history_tickers[:1],
            asset_type=(
                AssetType(kwargs["asset_type"])
                if kwargs.get("asset_type")
                else self._infer_asset_type(message, history_items)
            ),
            strategy=kwargs.get("strategy"),
            conversation_id=kwargs.get("conversation_id"),
        )

    def _analysis_tool(self) -> StructuredTool:
        async def run_analysis(
            ticker: str,
            config: RunnableConfig,
            asset_type: str = "stock",
            strategy: str | None = None,
        ) -> str:
            """运行综合研究分析，适合用户要求趋势、买卖、风险或交易建议时使用。"""
            request = self.prepare(
                f"分析 {ticker}",
                strategy=strategy,
                asset_type=asset_type,
            )
            _, result = await self.analyze(request, config=config)
            decision = result.get("final_decision")
            return decision.model_dump_json() if decision is not None else "{}"

        return StructuredTool.from_function(
            coroutine=run_analysis,
            name="run_fund_or_stock_analysis",
            description="运行短中期股票、ETF或LOF研究分析。只有用户明确需要分析、判断、策略或风险建议时调用。",
        )

    async def chat(self, request: StockAgentRequest) -> AsyncIterator[dict[str, Any]]:
        """Run the LangGraph LLM/tool loop and stream its trace to the UI."""
        tools = build_chat_tools(self._analysis_tool())
        system = (
            "你是 A-Share Agent 的对话入口。你必须自行判断用户意图，并在需要事实数据时调用工具；"
            "禁止根据记忆编造行情、历史价格或新闻。行情、历史、新闻、对比和策略都必须通过工具获取。"
            "如果用户要综合分析，调用 run_fund_or_stock_analysis。完成工具调用后，用中文简洁回答，"
            "明确数据日期、来源和数据缺失。产品只服务于小散户的短中期基金交易研究和模拟交易，不承诺收益，"
            "股票分析不能冒充基金建议。若只是闲聊或询问能力，可以直接回答。"
            "系统支持查询模拟盘账户、持仓和订单；只有用户明确要求时才可创建或取消模拟盘订单，"
            "所有模拟盘操作都必须明确说明是纸面交易，禁止声称已经进行实盘交易。"
        )
        messages: list[Any] = [
            {"role": "system", "content": system},
            *request.history[-12:],
            {"role": "user", "content": request.message},
        ]
        final_response = ""
        chat_config = build_trace_config(
            "stock-agent-chat",
            tags=["stock-agent", "chat", request.intent.value],
            metadata={"intent": request.intent.value},
            session_id=request.conversation_id,
        )
        async for update in stream_agent_loop(messages, tools, max_steps=100, config=chat_config):
            for node_update in update.values():
                if not isinstance(node_update, dict):
                    continue
                for event in node_update.get("tool_events", []):
                    yield {
                        "type": "tool",
                        "name": event.get("name", "unknown"),
                        "status": event.get("status", "completed"),
                        "result": event.get("result", ""),
                    }
                if node_update.get("final_response"):
                    final_response = node_update["final_response"]
        if final_response:
            yield {"type": "text", "text": final_response}

    async def analyze(
        self,
        request: StockAgentRequest,
        *,
        config: RunnableConfig | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run the multi-agent stock analysis workflow with tracing metadata."""
        if not request.ticker:
            raise ValueError("A stock code is required for analysis")

        ticker = request.ticker
        context = await build_market_context(ticker, asset_type=request.asset_type)
        state: dict[str, Any] = {
            "ticker": ticker,
            "asset_type": request.asset_type.value,
            "current_price": context.current_price,
            "market_context": context,
            "conversation_history": request.history[-12:],
            "progress": [],
            "user_message": request.message,
        }
        if request.strategy:
            state["strategy_name"] = request.strategy

        if config is None:
            run_config = build_trace_config(
                "stock-agent-analysis",
                tags=["stock-agent", "chat", request.intent.value],
                metadata={
                    "ticker": ticker,
                    "intent": request.intent.value,
                    "strategy": request.strategy or "auto",
                    "conversation_id": request.conversation_id or "",
                },
                session_id=request.conversation_id,
            )
        else:
            # Nested analysis is already inside the chat/tool trace. Reusing
            # its config preserves the Langfuse parent observation instead of
            # opening a second root trace with a fresh callback handler.
            run_config = {
                **config,
                "run_name": "stock-agent-analysis",
                "tags": [*config.get("tags", []), "stock-agent", "analysis"],
                "metadata": {
                    **config.get("metadata", {}),
                    "ticker": ticker,
                    "intent": request.intent.value,
                    "strategy": request.strategy or "auto",
                    "conversation_id": request.conversation_id or "",
                },
            }
        result = await workflow.ainvoke(state, config=run_config)
        return context.realtime, result

    async def analyze_stream(self, request: StockAgentRequest) -> AsyncIterator[dict[str, Any]]:
        """Stream actual LangGraph node updates for the chat UI."""
        if not request.ticker:
            raise ValueError("A stock code is required for analysis")

        ticker = request.ticker
        context = await build_market_context(ticker, asset_type=request.asset_type)
        state: dict[str, Any] = {
            "ticker": ticker,
            "asset_type": request.asset_type.value,
            "current_price": context.current_price,
            "market_context": context,
            "conversation_history": request.history[-12:],
            "progress": [],
            "user_message": request.message,
        }
        if request.strategy:
            state["strategy_name"] = request.strategy

        run_config = build_trace_config(
            "stock-agent-analysis",
            tags=["stock-agent", "chat", request.intent.value],
            metadata={
                "ticker": ticker,
                "intent": request.intent.value,
                "strategy": request.strategy or "auto",
                "conversation_id": request.conversation_id or "",
            },
            session_id=request.conversation_id,
        )
        accumulated: dict[str, Any] = dict(state)
        async for update in workflow.astream(state, config=run_config, stream_mode="updates"):
            node_name, node_update = next(iter(update.items()))
            if isinstance(node_update, dict):
                accumulated.update(node_update)
            yield {
                "node": node_name,
                "update": node_update,
                "state": accumulated,
                "realtime": context.realtime,
                "data_status": context.data_status,
            }


stock_agent = StockAgent()


def capabilities_text() -> str:
    """Return the user-facing capability summary."""
    return (
        "我是股票 Agent，专注于 A 股研究与交易辅助。你可以让我：\n\n"
        "- 分析单只股票：`分析 000737`\n"
        "- 查询实时行情：`查询 600519 行情`\n"
        "- 查看历史走势：`查看 000858 K 线`\n"
        "- 查看个股新闻与舆情：`看看 000001 新闻`\n"
        "- 查看可用策略：`有哪些选股策略？`\n"
        "- 继续追问上一只股票：例如 `为什么这样判断？`\n\n"
        "回测和组合管理仍可在对应页面使用；我不会处理与股票无关的通用闲聊。"
    )
