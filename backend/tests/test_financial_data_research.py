import json
from datetime import date

import pytest
import pytest_asyncio

from agents.asset_requests import AssetRequestResolver, RequestMode
from application.chat_service import ChatStore
from application.data_analysis_sandbox import run_data_analysis, validate_analysis_source
from application.financial_task_planner import compile_financial_task_spec
from application.market_data_query import MarketDataQueryError, MarketDataQueryService
from application.research_sandbox import SandboxError
from data import dividend_provider
from models.financial_task import ResearchAssetType
from tools.market_data import build_market_data_tools

PROMPT = "根据A股市场近6年分红数据，筛选每年均有分红的股票，按每股分红从大到小排序，输出排序结果表格。"


@pytest_asyncio.fixture
async def store(tmp_path):
    chat_store = ChatStore(tmp_path / "financial-chat.db")
    await chat_store.init()
    yield chat_store
    await chat_store.close()


def _provider_rows(year: int) -> list[dict]:
    rows = [
        {
            "代码": "600001",
            "名称": "测试甲",
            "现金分红-现金分红比例": 1.0,
            "方案进度": "实施方案",
        },
        {
            "代码": "600002",
            "名称": "测试乙",
            "现金分红-现金分红比例": 2.0,
            "方案进度": "实施方案",
        },
        {
            "代码": "600003",
            "名称": "测试丙",
            "现金分红-现金分红比例": 9.0,
            "方案进度": "董事会预案",
        },
    ]
    if year == 2022:
        rows.append(
            {
                "代码": "600001",
                "名称": "测试甲",
                "现金分红-现金分红比例": 0.5,
                "方案进度": "已实施",
            }
        )
    if year == 2025:
        rows = [item for item in rows if item["代码"] != "600002"]
    return rows


def test_financial_request_uses_coarse_research_mode_without_ticker():
    agent = AssetRequestResolver()
    prepared = agent.prepare(PROMPT)
    resolved, clarification = agent.resolve_intent(prepared)

    assert clarification is None
    assert resolved.mode == RequestMode.FINANCIAL_RESEARCH
    assert resolved.tickers == ()
    assert resolved.intent.value != "help"


def test_dividend_task_compiles_to_a_machine_checkable_six_year_contract():
    spec = compile_financial_task_spec(
        {"message": PROMPT, "asset_type": "stock"},
        as_of=date(2026, 8, 23),
    )

    assert spec is not None
    assert spec.primary_dataset_id == "cn_a_share_cash_dividends"
    assert spec.periods == [2020, 2021, 2022, 2023, 2024, 2025]
    assert spec.output.preview_limit == 30
    assert {item.criterion for item in spec.acceptance} >= {"period_coverage", "sorted", "non_empty"}


def test_query_market_data_exposes_the_complete_financial_task_schema():
    tool = next(item for item in build_market_data_tools() if item.name == "query_market_data")
    schema = tool.args_schema.model_json_schema()

    task_schema = schema["$defs"]["FinancialTaskSpec"]
    assert {"objective", "operation", "asset_type", "dataset_requirements"} <= set(task_schema["required"])
    assert set(schema["$defs"]["FinancialOperation"]["enum"]) == {
        "screen",
        "rank",
        "aggregate",
        "time_series",
        "compare",
        "analyze",
        "backtest",
    }
    assert "dataset_id" in schema["$defs"]["DatasetRequirement"]["required"]


@pytest.mark.asyncio
async def test_query_market_data_returns_safe_structured_validation_error():
    tool = next(item for item in build_market_data_tools() if item.name == "query_market_data")

    payload = json.loads(await tool.ainvoke({"task_spec": {"asset_type": "etf"}}))

    assert payload["ok"] is False
    assert payload["error"]["code"] == "invalid_task_spec"
    assert "validation errors" not in payload["error"]["message"]
    assert "pydantic.dev" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_market_data_query_rejects_an_incompatible_asset_before_provider_access(monkeypatch):
    spec = compile_financial_task_spec(
        {"message": PROMPT, "asset_type": "stock"},
        as_of=date(2026, 8, 23),
    )
    assert spec is not None
    incompatible = spec.model_copy(update={"asset_type": ResearchAssetType.ETF})

    async def forbidden_fetch(_periods):
        raise AssertionError("an incompatible dataset must not reach the provider")

    monkeypatch.setattr("application.market_data_query.fetch_cash_dividends", forbidden_fetch)
    with pytest.raises(MarketDataQueryError) as error:
        await MarketDataQueryService().execute(incompatible)

    assert error.value.code == "dataset_asset_mismatch"


@pytest.mark.asyncio
async def test_dividend_query_filters_proposals_aggregates_plans_and_checks_coverage(monkeypatch):
    spec = compile_financial_task_spec(
        {"message": PROMPT, "asset_type": "stock"},
        as_of=date(2026, 8, 23),
    )
    assert spec is not None

    async def fake_fetch(periods):
        normalized = []
        counts = {}
        for year in periods:
            rows = dividend_provider.normalize_dividend_records(_provider_rows(year), year)
            counts[year] = len(rows)
            normalized.extend(rows)
        return normalized, counts

    monkeypatch.setattr("application.market_data_query.fetch_cash_dividends", fake_fetch)
    result, csv_content = await MarketDataQueryService().execute(spec)

    assert result.acceptance.status == "satisfied"
    assert result.coverage.status == "complete"
    assert result.coverage.result_rows == 1
    assert result.preview[0]["ticker"] == "600001"
    assert result.preview[0]["2022"] == pytest.approx(0.15)
    assert result.preview[0]["six_year_total"] == pytest.approx(0.65)
    assert "600003" not in (csv_content or "")
    assert "600002" not in (csv_content or "")


def test_data_analysis_sandbox_rejects_external_io():
    with pytest.raises(SandboxError, match="I/O|名称"):
        validate_analysis_source(
            "def analyze_data(frame):\n    return open('/tmp/leak').read()"
        )


@pytest.mark.asyncio
async def test_data_analysis_sandbox_runs_deterministic_dataframe_transform():
    rows = await run_data_analysis(
        "def analyze_data(frame):\n    return frame.assign(double=frame['value'] * 2)",
        [{"value": 2}, {"value": 3}],
    )

    assert rows == [{"value": 2, "double": 4}, {"value": 3, "double": 6}]
