"""Small generic tool surface for catalog discovery and structured data queries."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool, tool
from loguru import logger

from application.market_data_query import market_data_query_service
from artifacts.service import artifact_service
from data.market_data_catalog import market_data_catalog
from models.financial_task import FinancialTaskSpec


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


@tool
async def search_market_data_catalog(query: str, asset_type: str | None = None, limit: int = 5) -> str:
    """搜索可查询的结构化股票/基金数据集及其字段、单位和供应商；不执行网页搜索。"""
    matches = market_data_catalog.search(query, asset_type=asset_type, limit=max(1, min(limit, 10)))
    return _dump(
        {
            "data_type": "market_data_catalog",
            "query": query,
            "matches": [item.model_dump(mode="json") for item in matches],
            "available": bool(matches),
        }
    )


def build_market_data_tools(
    *, conversation_id: str | None = None, task_id: str | None = None
) -> list[StructuredTool]:
    async def query_market_data(task_spec: dict[str, Any]) -> str:
        """查询一个已解析的数据集，并执行受限变换与验收。"""
        spec = FinancialTaskSpec.model_validate(task_spec)
        result, csv_content = await market_data_query_service.execute(spec)
        payload = result.model_dump(mode="json")
        if csv_content:
            years = spec.periods
            name = (
                f"A股连续分红筛选-{years[0]}-{years[-1]}.csv"
                if years
                else "市场数据查询完整结果.csv"
            )
            try:
                artifacts = await artifact_service.create_user_artifacts(
                    [
                        {
                            "name": name,
                            "format": "csv",
                            "content": csv_content,
                            "description": "结构化市场数据查询的完整确定性结果",
                            "artifact_type": "data",
                            "asset_type": spec.asset_type.value if spec.asset_type.value != "fund" else None,
                            "metadata": {
                                "dataset_id": spec.primary_dataset_id,
                                "periods": spec.periods,
                                "acceptance_status": result.acceptance.status,
                            },
                        }
                    ],
                    source="market_data_query",
                    conversation_id=conversation_id,
                    task_id=task_id,
                    execution_key=(
                        f"{task_id or 'adhoc'}:{spec.primary_dataset_id}:{','.join(map(str, spec.periods))}"
                    ),
                )
                payload["artifacts"] = artifacts
            except Exception as exc:
                logger.warning("Market-data CSV artifact persistence failed: {}", exc)
                payload["artifact_error"] = "完整 CSV 暂时无法保存；表格预览与验收结果仍然有效"
        return _dump(payload)

    query_tool = StructuredTool.from_function(
        coroutine=query_market_data,
        name="query_market_data",
        description=(
            "按 FinancialTaskSpec 查询结构化股票/基金数据集。工具内部执行固定 Polars 变换、覆盖率检查、"
            "任务验收并在需要时生成完整 CSV。默认不要传 Python；只有声明式 DSL 无法表达时，"
            "才可使用隔离、无网络、无文件权限的数据分析沙盒回退。"
        ),
    )
    return [search_market_data_catalog, query_tool]


TOOLS = [search_market_data_catalog]
