"""Stock-focused conversational agent and task router.

The StockAgent keeps the chat surface focused on stock research tasks while
delegating analysis to the existing multi-agent LangGraph workflow.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from data.akshare_provider import get_stock_realtime
from graph.workflow import workflow


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
    ) -> StockAgentRequest:
        """Resolve intent and reuse the last ticker for conversational follow-ups."""
        history_items = list(history or [])
        current_tickers = self.extract_tickers(message)
        history_tickers = self.extract_tickers(*(item.get("content", "") for item in reversed(history_items)))
        tickers = current_tickers or history_tickers[:1]
        intent = self._infer_intent(message, len(current_tickers))
        return StockAgentRequest(
            message=message,
            history=history_items,
            intent=intent,
            tickers=tickers,
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
        return StockIntent.ANALYZE if current_ticker_count or self.extract_tickers(message) else StockIntent.HELP

    async def analyze(self, request: StockAgentRequest) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run the multi-agent stock analysis workflow with tracing metadata."""
        if not request.ticker:
            raise ValueError("A stock code is required for analysis")

        ticker = request.ticker
        realtime = get_stock_realtime(ticker)
        state: dict[str, Any] = {
            "ticker": ticker,
            "current_price": realtime.get("price", 0.0) if realtime else 0.0,
            "progress": [],
            "user_message": request.message,
        }
        if request.strategy:
            state["strategy_name"] = request.strategy

        run_config = {
            "run_name": "stock-agent-analysis",
            "tags": ["stock-agent", "chat", request.intent.value],
            "metadata": {
                "ticker": ticker,
                "intent": request.intent.value,
                "strategy": request.strategy or "auto",
                "conversation_id": request.conversation_id or "",
            },
        }
        result = await workflow.ainvoke(state, config=run_config)
        return realtime, result


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
