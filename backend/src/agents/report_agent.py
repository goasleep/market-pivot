"""Agent that turns research facts into a standalone HTML report."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from llm.service import get_llm_service
from models.schemas import TradeDecision

REPORT_AGENT_SYSTEM_PROMPT = """你是 A 股短中期研究报告 Agent，负责根据研究事实直接生成一份独立可预览的 HTML 报告。

你的输出必须是完整 HTML 文档，只能输出 HTML，不要输出 Markdown 代码块、解释文字或 JSON。

工作原则：
1. 只能使用用户输入的研究事实；不得补造价格、指数、资金、新闻、日期、来源或任何数值。
2. 根据实际数据情况自行决定报告结构。可以生成标题、摘要、目录、表格、卡片、风险提示、新闻、来源和图表；
   没有数据的部分应省略或明确写“暂无足够数据”。
3. 图表只能使用输入中的数据。需要图表时可以使用 ECharts CDN，并在 HTML 内用真实数据初始化图表；
   没有足够数据时不要伪造图表。
4. 关键数据旁边标注来源和数据时间；无法确认的来源要明确标记为“待核实”或“非官方来源”。
5. 报告面向基金短中期研究和模拟交易。只有股票数据时，必须说明这是底层股票研究，不能冒充基金专项分析。
6. 不承诺收益，不构成投资建议，不声称执行了真实交易，并在报告末尾加入研究用途免责声明。
7. 页面应是响应式、简洁、可读的 standalone HTML，包含 charset 和 viewport；不要依赖本地文件。
"""


@dataclass(frozen=True)
class ReportArtifact:
    """Raw report output returned by ReportAgent before persistence."""

    name: str
    html: str
    metadata: dict[str, Any]


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _bounded_context(market_context: Any | None) -> dict[str, Any]:
    if market_context is None:
        return {}
    payload = _jsonable(market_context)
    if not isinstance(payload, dict):
        return {"value": payload}

    # Keep the report prompt useful for long histories while retaining enough
    # points for the Agent to decide whether a trend chart is warranted.
    for key, limit in (("history", 240), ("news", 20), ("web_results", 20)):
        value = payload.get(key)
        if isinstance(value, list):
            payload[key] = value[-limit:]
    return payload


def _strip_code_fence(raw: str) -> str:
    text = raw.strip()
    match = re.fullmatch(r"```(?:html|xhtml)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text


def _extract_html(raw: str) -> str:
    text = _strip_code_fence(raw)
    start = re.search(r"<!doctype\s+html|<html\b", text, flags=re.IGNORECASE)
    if start:
        text = text[start.start() :]
    if not re.search(r"<html\b", text, flags=re.IGNORECASE) or not re.search(
        r"</html>\s*$", text, flags=re.IGNORECASE
    ):
        raise ValueError("ReportAgent 未返回完整 HTML 文档")
    return text


def _report_facts(
    decision: TradeDecision,
    market_context: Any | None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    return {
        "report_request": "根据可用数据生成最合适的研究报告；优先展示事实和数据，按数据情况组织章节。",
        "generated_at": generated_at,
        "decision": _jsonable(decision),
        "market_context": _bounded_context(market_context),
    }


class ReportAgent:
    """Dedicated report-writing Agent; it owns report structure and HTML output."""

    def __init__(self, llm_service: Any | None = None):
        self.llm_service = llm_service or get_llm_service()

    def _prompt(
        self,
        decision: TradeDecision,
        market_context: Any | None,
        generated_at: str | None = None,
    ) -> str:
        return json.dumps(
            _report_facts(decision, market_context, generated_at),
            ensure_ascii=False,
            default=str,
        )

    def generate(
        self,
        decision: TradeDecision,
        market_context: Any | None = None,
        *,
        generated_at: str | None = None,
    ) -> ReportArtifact:
        raw = self.llm_service.chat_sync(
            self._prompt(decision, market_context, generated_at),
            system=REPORT_AGENT_SYSTEM_PROMPT,
        )
        html = _extract_html(raw)
        web_results = getattr(market_context, "web_results", []) or []
        return ReportArtifact(
            name=f"{decision.ticker}-研究分析报告.html",
            html=html,
            metadata={
                "generated_by": "report_agent",
                "report_version": "3.0",
                "web_search_count": len(web_results),
                "generated_at": generated_at,
            },
        )
