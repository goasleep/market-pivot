from agents.asset_requests import AssetAgentRequest, AssetIntent, RequestMode
from harness.bootstrap import build_default_catalog, load_default_skills
from harness.compiler import harness_task_compiler
from harness.planner import harness_planner, skill_selector
from models.schemas import AssetType
from models.supervisor import ExecutionMode, TaskRoutingDecision


def _routing(mode: ExecutionMode = ExecutionMode.EVIDENCE_RESEARCH) -> TaskRoutingDecision:
    return TaskRoutingDecision(mode=mode, requires_tools=True, allow_research_plan=True)


def _request(message: str, *, asset_type: AssetType, intent: AssetIntent) -> AssetAgentRequest:
    return AssetAgentRequest(
        message=message,
        history=[],
        intent=intent,
        tickers=("510300",),
        asset_type=asset_type,
        intent_confirmed=True,
    )


def test_etf_analysis_contract_forbids_stock_comprehensive_graph():
    contract = harness_task_compiler.compile(
        _request("分析 510300 ETF 的趋势和风险", asset_type=AssetType.ETF, intent=AssetIntent.ANALYZE),
        _routing(),
    )

    assert "stock.comprehensive_analysis" in contract.forbidden_capabilities
    assert "stock.comprehensive_analysis" not in contract.allowed_capabilities
    assert contract.required_capabilities == ("exchange_fund.comprehensive_analysis",)


def test_stock_analysis_can_use_comprehensive_skill():
    contract = harness_task_compiler.compile(
        _request("分析 600519", asset_type=AssetType.STOCK, intent=AssetIntent.ANALYZE),
        _routing(),
    )
    assert "stock.comprehensive_analysis" in contract.required_capabilities


def test_structured_stock_universe_query_uses_market_dataset_skill():
    contract = harness_task_compiler.compile(
        _request(
            "筛选连续5年都有现金分红的全部A股，并按累计分红排序",
            asset_type=AssetType.STOCK,
            intent=AssetIntent.ANALYZE,
        ),
        _routing(),
    )

    assert contract.required_capabilities == ("market.dataset",)
    registry = load_default_skills(catalog=build_default_catalog())
    skills = skill_selector.select(contract, registry)
    assert [skill.id for skill in skills] == ["market.dataset"]
    assert skills[0].tools == ("search_market_data_catalog", "query_market_data")


def test_system_strategy_catalog_request_uses_strategy_list_skill():
    contract = harness_task_compiler.compile(
        _request("系统目前支持哪些交易策略？", asset_type=AssetType.STOCK, intent=AssetIntent.STRATEGIES),
        _routing(),
    )

    assert contract.required_capabilities == ("strategy.list",)
    registry = load_default_skills(catalog=build_default_catalog())
    skills = skill_selector.select(contract, registry)
    assert [skill.id for skill in skills] == ["strategy.list"]
    assert skills[0].tools == ("list_trading_strategies",)


def test_general_strategy_question_keeps_methodology_skill():
    contract = harness_task_compiler.compile(
        _request("解释趋势跟踪策略的方法论", asset_type=AssetType.STOCK, intent=AssetIntent.STRATEGIES),
        _routing(),
    )

    assert contract.required_capabilities == ("methodology.search",)


def test_etf_universe_research_uses_screen_dependency_closure_only():
    contract = harness_task_compiler.compile(
        _request(
            "请你帮我筛选一下白酒相关的etf，哪一个适合我？我更偏向于做短线",
            asset_type=AssetType.ETF,
            intent=AssetIntent.ANALYZE,
        ),
        _routing(),
    )

    assert contract.required_capabilities == ("exchange_fund.screen_compare",)
    assert "stock.comprehensive_analysis" in contract.forbidden_capabilities
    registry = load_default_skills(catalog=build_default_catalog())
    skills = skill_selector.select(contract, registry)
    selected = {skill.id for skill in skills}
    tools = {tool_name for skill in skills for tool_name in skill.tools}
    assert selected == {
        "market.quote",
        "market.history",
        "exchange_fund.profile",
        "exchange_fund.liquidity_cost",
        "exchange_fund.relative_strength",
        "exchange_fund.screen_compare",
    }
    assert "discover_exchange_fund_candidates" in tools
    assert {"compute_technical_indicators", "calculate_risk_metrics", "get_exchange_fund_nav_history"}.isdisjoint(tools)


def test_direct_explanation_does_not_load_financial_tools():
    contract = harness_task_compiler.compile(
        _request("解释 ETF 是什么", asset_type=AssetType.ETF, intent=AssetIntent.ANALYZE),
        TaskRoutingDecision(mode=ExecutionMode.DIRECT_RESPONSE, requires_tools=False),
    )
    assert contract.required_capabilities == ()
    assert contract.allowed_capabilities == ()


def test_explicit_code_backtest_selects_only_sandbox_execution_tool():
    contract = harness_task_compiler.compile(
        _request(
            "用 Python 生成代码策略并回测 510300",
            asset_type=AssetType.ETF,
            intent=AssetIntent.BACKTEST,
        ),
        _routing(ExecutionMode.BACKTEST_EXECUTION),
    )

    assert contract.required_capabilities == ("market.history", "strategy.sandbox_research")
    registry = load_default_skills(catalog=build_default_catalog())
    skills = skill_selector.select(contract, registry)
    tools = {tool_name for skill in skills for tool_name in skill.tools}
    assert "design_and_run_sandbox_strategy" in tools
    assert {
        "run_backtest",
        "design_and_run_backtest",
        "compare_strategy_backtests",
    }.isdisjoint(tools)


def test_plain_backtest_excludes_sandbox_execution_tool():
    contract = harness_task_compiler.compile(
        _request("回测 510300", asset_type=AssetType.ETF, intent=AssetIntent.BACKTEST),
        _routing(ExecutionMode.BACKTEST_EXECUTION),
    )

    assert contract.required_capabilities == ("market.history", "backtest.execute")
    registry = load_default_skills(catalog=build_default_catalog())
    skills = skill_selector.select(contract, registry)
    tools = {tool_name for skill in skills for tool_name in skill.tools}
    assert "design_and_run_sandbox_strategy" not in tools
    assert "run_backtest" in tools


def test_sandbox_execution_does_not_depend_on_legacy_backtest_intent():
    contract = harness_task_compiler.compile(
        _request(
            "用代码实现 RSI 策略并在沙箱运行",
            asset_type=AssetType.ETF,
            intent=AssetIntent.STRATEGIES,
        ),
        _routing(ExecutionMode.BACKTEST_EXECUTION),
    )

    assert contract.required_capabilities == ("market.history", "strategy.sandbox_research")
    assert contract.budget_profile == "deep"


def test_code_backtest_explanation_does_not_execute_sandbox():
    contract = harness_task_compiler.compile(
        _request(
            "只解释这段 Python 回测代码，不要执行",
            asset_type=AssetType.ETF,
            intent=AssetIntent.BACKTEST,
        ),
        TaskRoutingDecision(mode=ExecutionMode.DIRECT_RESPONSE, requires_tools=False),
    )

    assert contract.required_capabilities == ()
    assert contract.allowed_capabilities == ()


def test_simulation_write_requires_explicit_mutation():
    request = _request("查看模拟盘持仓", asset_type=AssetType.ETF, intent=AssetIntent.PORTFOLIO)
    contract = harness_task_compiler.compile(request, _routing())
    assert "simulation.write" in contract.forbidden_capabilities

    mutation = AssetAgentRequest(
        **{
            **request.__dict__,
            "message": "提交模拟盘买入订单",
            "mode": RequestMode.SIMULATION_MUTATION,
            "allow_mutating_tools": True,
        }
    )
    mutation_contract = harness_task_compiler.compile(mutation, _routing(ExecutionMode.MIXED_WORKFLOW))
    assert "simulation.write" in mutation_contract.required_capabilities
    assert mutation_contract.allow_mutations is True


def test_selector_and_planner_build_minimal_dependency_dag():
    contract = harness_task_compiler.compile(
        _request("分析 510300 ETF 的趋势和风险", asset_type=AssetType.ETF, intent=AssetIntent.ANALYZE),
        _routing(),
    )
    registry = load_default_skills(catalog=build_default_catalog())
    skills = skill_selector.select(contract, registry)
    plan = harness_planner.deterministic_plan(contract, skills)

    assert "stock.comprehensive_analysis" not in plan.selected_skills
    assert set(plan.selected_skills) == {
        "market.quote",
        "market.history",
        "exchange_fund.nav",
        "technical.indicators",
        "risk.metrics",
        "exchange_fund.profile",
        "exchange_fund.liquidity_cost",
        "exchange_fund.relative_strength",
        "exchange_fund.comprehensive_analysis",
    }
    assert len(plan.steps) <= contract.budget.max_steps
