"""Deep Agent research synthesis subgraph."""

from __future__ import annotations

import json

from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import StructuredTool

from agents.deep_agent_runtime import build_deep_agent, invoke_structured
from models.schemas import AgentReport, AssetType


def _report_reader(name: str, report: AgentReport) -> StructuredTool:
    def read_report() -> str:
        """Read the specialist report assigned to this researcher."""
        return json.dumps(
            {"specialist": name, "report": report.model_dump(mode="json")},
            ensure_ascii=False,
        )

    return StructuredTool.from_function(
        func=read_report,
        name=f"read_{name}_report",
        description=f"读取 {name} 专家的结构化研究报告。",
    )


async def synthesize_reports(
    ticker: str,
    reports: dict[str, AgentReport],
    *,
    asset_type: AssetType | str,
) -> AgentReport:
    """Delegate specialist review and synthesize one bounded debate report."""
    technical = reports.get("technical", AgentReport(agent_name="technical"))
    fundamentals = reports.get("fundamentals", AgentReport(agent_name="fundamentals"))
    sentiment = reports.get("sentiment", AgentReport(agent_name="sentiment"))

    subagents = [
        {
            "name": "technical-reviewer",
            "description": "复核技术分析报告，指出趋势、动量和数据限制。",
            "system_prompt": "你只复核技术报告，不补造行情；输出简短证据和风险。",
            "tools": [_report_reader("technical", technical)],
        },
        {
            "name": "fundamentals-reviewer",
            "description": "复核基本面分析报告，指出估值、财务和数据时效风险。",
            "system_prompt": "你只复核基本面报告，不补造财务数据；输出简短证据和风险。",
            "tools": [_report_reader("fundamentals", fundamentals)],
        },
        {
            "name": "sentiment-reviewer",
            "description": "复核情绪与资讯分析报告，指出来源和时效风险。",
            "system_prompt": "你只复核情绪报告，不补造新闻；输出简短证据和风险。",
            "tools": [_report_reader("sentiment", sentiment)],
        },
    ]
    agent = build_deep_agent(
        system_prompt=(
            "你是 A 股短中期研究的辩论协调 Agent。"
            "先用 write_todos 规划复核步骤，再分别委派给 technical-reviewer、"
            "fundamentals-reviewer、sentiment-reviewer，最后合并为一个结构化结论。"
            "只能依据子 Agent 读取到的报告，不得凭记忆补造价格、财务或新闻。"
            "结论必须说明数据缺失、来源和不确定性；这是研究与模拟交易辅助，不是实盘指令。"
        ),
        subagents=subagents,
        response_format=ToolStrategy(AgentReport),
        name="research-debate-coordinator",
    )
    prompt = json.dumps(
        {
            "ticker": ticker,
            "asset_type": AssetType(asset_type).value,
            "task": "复核三份专家报告并给出买卖方向、置信度和可审计理由。",
        },
        ensure_ascii=False,
    )
    result = await invoke_structured(agent, prompt, AgentReport)
    return result.model_copy(update={"agent_name": "deep_research_debate"})
