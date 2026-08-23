"""Intent resolution and request models for asset chat."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import date
from enum import Enum
from typing import Any, Sequence

from models.schemas import AssetType


class AssetIntent(str, Enum):
    ANALYZE = "analyze"
    QUOTE = "quote"
    HISTORY = "history"
    NEWS = "news"
    STRATEGIES = "strategies"
    PORTFOLIO = "portfolio"
    BACKTEST = "backtest"
    COMPARE = "compare"
    HELP = "help"


class RequestMode(str, Enum):
    """Coarse safety gate; financial semantics are planned after this gate."""

    FINANCIAL_RESEARCH = "financial_research"
    SIMULATION_MUTATION = "simulation_mutation"
    HELP = "help"


@dataclass(frozen=True)
class AssetAgentRequest:
    message: str
    history: list[dict[str, str]]
    intent: AssetIntent
    tickers: tuple[str, ...]
    mode: RequestMode = RequestMode.FINANCIAL_RESEARCH
    asset_type: AssetType = AssetType.STOCK
    start_date: str | None = None
    end_date: str | None = None
    strategy: str | None = None
    conversation_id: str | None = None
    task_id: str | None = None
    allow_mutating_tools: bool = False
    intent_confirmed: bool = False
    llm_profile_id: str | None = None
    llm_model: str | None = None
    llm_auto: bool = False

    @property
    def ticker(self) -> str | None:
        return self.tickers[0] if self.tickers else None

    def with_intent(self, intent: AssetIntent) -> "AssetAgentRequest":
        return replace(self, intent=intent, intent_confirmed=True)

    def with_mode(self, mode: RequestMode) -> "AssetAgentRequest":
        return replace(self, mode=mode)


class AssetRequestResolver:
    """Route conversational requests to common asset research capabilities."""

    _ticker_pattern = re.compile(r"(?<!\d)(?:(?:sh|sz|bj)\s*)?(\d{6})(?!\d)", re.IGNORECASE)
    _date_pattern = re.compile(
        r"(?<!\d)(?:(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})日?|(\d{8}))(?!\d)"
    )

    _keyword_groups = {
        AssetIntent.BACKTEST: ("回测", "回测一下", "策略测试", "历史测试", "backtest"),
        AssetIntent.COMPARE: ("对比", "比较", "compare", "vs", " versus "),
        AssetIntent.NEWS: ("新闻", "消息", "舆情", "资讯", "news"),
        AssetIntent.HISTORY: ("历史", "k线", "走势", "行情走势", "历史价格", "history", "chart"),
        AssetIntent.QUOTE: ("实时", "现价", "报价", "行情", "价格", "quote"),
        AssetIntent.STRATEGIES: ("策略", "选股", "交易规则", "strategies"),
        AssetIntent.PORTFOLIO: ("持仓", "组合", "仓位", "账户", "portfolio"),
        AssetIntent.ANALYZE: ("分析", "估值", "基本面", "技术面", "财报", "趋势", "买入", "卖出", "建议", "analy"),
    }
    _research_intents = {
        AssetIntent.QUOTE,
        AssetIntent.HISTORY,
        AssetIntent.NEWS,
        AssetIntent.STRATEGIES,
        AssetIntent.ANALYZE,
        AssetIntent.COMPARE,
        AssetIntent.BACKTEST,
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

    @classmethod
    def extract_date_range(cls, *texts: str) -> tuple[str | None, str | None]:
        """Extract the first explicit start/end pair from conversational text."""
        for text in texts:
            found: list[str] = []
            for match in cls._date_pattern.finditer(text or ""):
                compact = match.group(4)
                try:
                    parsed = (
                        date(int(compact[:4]), int(compact[4:6]), int(compact[6:]))
                        if compact
                        else date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
                    )
                except ValueError:
                    continue
                found.append(parsed.isoformat())
                if len(found) == 2:
                    return found[0], found[1]
        return None, None

    def resolve(
        self,
        message: str,
        history: Sequence[dict[str, str]] | None = None,
        strategy: str | None = None,
        conversation_id: str | None = None,
        asset_type: AssetType | str | None = None,
    ) -> AssetAgentRequest:
        """Resolve intent and reuse the last ticker for conversational follow-ups."""
        history_items = list(history or [])
        current_tickers = self.extract_tickers(message)
        history_tickers = self.extract_tickers(*(item.get("content", "") for item in reversed(history_items)))
        tickers = current_tickers or history_tickers[:1]
        start_date, end_date = self.extract_date_range(
            message,
            *(item.get("content", "") for item in reversed(history_items)),
        )
        intent = self._infer_intent(message, len(current_tickers))
        asset_type = AssetType(asset_type) if asset_type else self._infer_asset_type(message, history_items)
        return AssetAgentRequest(
            message=message,
            history=history_items,
            intent=intent,
            tickers=tickers,
            asset_type=asset_type,
            start_date=start_date,
            end_date=end_date,
            strategy=strategy,
            conversation_id=conversation_id,
        )

    def _infer_intent(self, message: str, current_ticker_count: int) -> AssetIntent:
        text = f" {message.lower()} "
        if not message.strip():
            return AssetIntent.HELP
        if any(keyword in text for keyword in self._keyword_groups[AssetIntent.BACKTEST]):
            return AssetIntent.BACKTEST
        if current_ticker_count > 1 or any(keyword in text for keyword in self._keyword_groups[AssetIntent.COMPARE]):
            return AssetIntent.COMPARE
        for intent in (
            AssetIntent.NEWS,
            AssetIntent.HISTORY,
            AssetIntent.QUOTE,
            AssetIntent.STRATEGIES,
            AssetIntent.PORTFOLIO,
            AssetIntent.ANALYZE,
        ):
            if any(keyword in text for keyword in self._keyword_groups[intent]):
                return intent
        follow_up_keywords = ("为什么", "依据", "解释", "怎么看", "如果", "还能", "风险", "止损")
        if (
            current_ticker_count
            or self.extract_tickers(message)
            or any(keyword in text for keyword in follow_up_keywords)
        ):
            return AssetIntent.ANALYZE
        return AssetIntent.HELP

    def _matched_intents(self, message: str) -> list[AssetIntent]:
        text = f" {message.lower()} "
        matches: list[AssetIntent] = []
        for intent in (
            AssetIntent.BACKTEST,
            AssetIntent.COMPARE,
            AssetIntent.NEWS,
            AssetIntent.HISTORY,
            AssetIntent.QUOTE,
            AssetIntent.STRATEGIES,
            AssetIntent.PORTFOLIO,
            AssetIntent.ANALYZE,
        ):
            if any(keyword in text for keyword in self._keyword_groups[intent]):
                matches.append(intent)
        return matches

    @staticmethod
    def _explicitly_requests_mutation(message: str) -> bool:
        return bool(
            re.search(
                r"(?:下单|提交(?:模拟)?订单|创建(?:模拟)?(?:订单|账户|盘)|取消订单|"
                r"买入(?:模拟)?单|卖出(?:模拟)?单|部署(?:到)?模拟盘|启用部署|暂停部署|归档部署)",
                message or "",
                flags=re.IGNORECASE,
            )
        )

    def resolve_intent(self, request: AssetAgentRequest) -> tuple[AssetAgentRequest, dict[str, Any] | None]:
        """Apply only safety-level routing; the research planner owns task semantics."""
        text = request.message.strip().lower()
        if request.allow_mutating_tools or self._explicitly_requests_mutation(request.message):
            return request.with_mode(RequestMode.SIMULATION_MUTATION), None
        if not text or text in {"帮助", "help", "你能做什么", "有什么功能", "使用说明"}:
            return replace(request, intent=AssetIntent.HELP, mode=RequestMode.HELP, intent_confirmed=True), None
        if request.intent_confirmed:
            return request.with_mode(RequestMode.FINANCIAL_RESEARCH), None
        # Preserve legacy intent metadata where it is unambiguous, but never use
        # it to decide whether a financial question may enter the research graph.
        inferred = self._infer_intent(request.message, len(request.tickers))
        if inferred == AssetIntent.HELP:
            inferred = AssetIntent.ANALYZE
        return replace(
            request,
            intent=inferred,
            mode=RequestMode.FINANCIAL_RESEARCH,
            intent_confirmed=True,
        ), None

    @staticmethod
    def _infer_asset_type(message: str, history: Sequence[dict[str, str]]) -> AssetType:
        text = " ".join([message, *(item.get("content", "") for item in history[-6:])]).lower()
        if any(token in text for token in ("lof", "场内基金", "lof基金")):
            return AssetType.LOF
        if any(token in text for token in ("etf", "交易型基金", "指数基金")):
            return AssetType.ETF
        if any(token in text for token in ("股票", "个股", "a股")):
            return AssetType.STOCK
        tickers = AssetRequestResolver.extract_tickers(text)
        if any(ticker.startswith(("15", "51", "56", "58")) for ticker in tickers):
            return AssetType.ETF
        if any(ticker.startswith(("16", "50")) for ticker in tickers):
            return AssetType.LOF
        return AssetType.STOCK

    def prepare(
        self,
        message: str,
        history: Sequence[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> AssetAgentRequest:
        """Extract only safe context; intent is selected by the LLM, not keywords."""
        history_items = list(history or [])
        current_tickers = self.extract_tickers(message)
        history_tickers = self.extract_tickers(*(item.get("content", "") for item in reversed(history_items)))
        start_date, end_date = self.extract_date_range(
            message,
            *(item.get("content", "") for item in reversed(history_items)),
        )
        return AssetAgentRequest(
            message=message,
            history=history_items,
            intent=AssetIntent.ANALYZE,
            tickers=current_tickers or history_tickers[:1],
            asset_type=(
                AssetType(kwargs["asset_type"])
                if kwargs.get("asset_type")
                else self._infer_asset_type(message, history_items)
            ),
            start_date=start_date,
            end_date=end_date,
            strategy=kwargs.get("strategy"),
            conversation_id=kwargs.get("conversation_id"),
            task_id=kwargs.get("task_id"),
            allow_mutating_tools=self._explicitly_requests_mutation(message),
            llm_profile_id=kwargs.get("llm_profile_id"),
            llm_model=kwargs.get("llm_model"),
            llm_auto=bool(kwargs.get("llm_auto", False)),
        )

    @staticmethod
    def request_payload(request: AssetAgentRequest) -> dict[str, Any]:
        return {
            "message": request.message,
            "history": request.history,
            "intent": request.intent.value,
            "mode": request.mode.value,
            "tickers": list(request.tickers),
            "asset_type": request.asset_type.value,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "strategy": request.strategy,
            "conversation_id": request.conversation_id,
            "task_id": request.task_id,
            "allow_mutating_tools": request.allow_mutating_tools,
            "intent_confirmed": request.intent_confirmed,
            "llm_profile_id": request.llm_profile_id,
            "llm_model": request.llm_model,
            "llm_auto": request.llm_auto,
        }

    @classmethod
    def research_request_payload(cls, request: AssetAgentRequest) -> dict[str, Any]:
        """Return the Research Plan input without unused cross-turn history."""
        payload = cls.request_payload(request)
        payload.pop("history", None)
        return payload

    @staticmethod
    def request_from_payload(payload: dict[str, Any]) -> AssetAgentRequest:
        return AssetAgentRequest(
            message=str(payload.get("message", "")),
            history=list(payload.get("history") or []),
            intent=AssetIntent(str(payload.get("intent", AssetIntent.ANALYZE.value))),
            tickers=tuple(str(item) for item in payload.get("tickers", []) if item),
            mode=RequestMode(str(payload.get("mode", RequestMode.FINANCIAL_RESEARCH.value))),
            asset_type=AssetType(str(payload.get("asset_type", AssetType.STOCK.value))),
            start_date=payload.get("start_date"),
            end_date=payload.get("end_date"),
            strategy=payload.get("strategy"),
            conversation_id=payload.get("conversation_id"),
            task_id=payload.get("task_id"),
            allow_mutating_tools=bool(payload.get("allow_mutating_tools", False)),
            intent_confirmed=bool(payload.get("intent_confirmed", False)),
            llm_profile_id=payload.get("llm_profile_id"),
            llm_model=payload.get("llm_model"),
            llm_auto=bool(payload.get("llm_auto", False)),
        )
