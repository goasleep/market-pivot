"""Strategy research and comparison A2UI surfaces."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from widgets.a2ui import _number_label, _percent_label, _ref, _surface, _text


def render_sandbox_strategy_candidate(
    payload: dict[str, Any],
    surface_id: str | None = None,
) -> list[dict[str, Any]]:
    """Render generated code, sandbox checks, and its trusted-engine backtest."""
    surface_id = surface_id or f"sandbox-candidate-{uuid4().hex}"
    result = dict(payload.get("result") or {})
    backtest = dict(result.get("backtest") or {})
    strategy = dict(payload.get("strategy_spec") or {})
    validation = dict(payload.get("validation") or {})
    source_code = str(payload.get("source_code") or "# 未返回脚本源码")
    promotion_eligible = result.get("promotion_eligible") is True
    validation_passed = validation.get("passed") is True

    status = str(payload.get("status") or "draft")
    status_label = {
        "draft": "草稿",
        "validated": "已验证",
        "approved": "已审核",
        "rejected": "已拒绝",
        "deployed": "已部署",
    }.get(status, status)
    validation_label = "沙盒验证通过" if validation_passed else "沙盒验证未通过"
    eligibility_label = "可提交人工审核" if promotion_eligible else "仅限研究"

    check_labels = {
        "ast_parse": "Python 语法可解析",
        "source_size": "源码大小符合限制",
        "allowed_imports": "仅使用允许的依赖",
        "denied_names": "未使用危险内置函数",
        "denied_attributes": "未调用危险属性",
        "function_contract": "目标仓位函数契约正确",
        "output_length": "输出长度与行情一致",
        "bounded_exposure": "目标仓位位于 0–95%",
        "binary_positions": "兼容旧式离散仓位",
        "deterministic_output": "重复执行结果一致",
        "causal_output": "未使用未来数据",
        "dsl_signal_equivalent": "代码信号与 StrategySpec 一致",
        "minimum_history_5y": "有效历史区间不少于 5 年",
        "strategy_spec_valid": "结构化 StrategySpec 可执行",
    }
    validation_rows = []
    for category, checks in (
        ("静态检查", validation.get("static_checks") or {}),
        ("输出检查", validation.get("output_checks") or {}),
    ):
        for name, passed in checks.items():
            validation_rows.append(
                {
                    "category": category,
                    "check": check_labels.get(str(name), str(name)),
                    "status": "通过" if passed else "未通过",
                }
            )
    validation_rows.extend(
        [
            {
                "category": "行为检查",
                "check": "重复执行结果一致",
                "status": "通过" if validation.get("deterministic") is True else "未通过",
            },
            {
                "category": "行为检查",
                "check": "未使用未来数据",
                "status": "通过" if validation.get("causal") is True else "未通过",
            },
        ]
    )
    errors = [str(item) for item in validation.get("errors") or []]

    rules = []
    for direction, conditions in (
        ("入场", strategy.get("entry_conditions") or []),
        ("退出", strategy.get("exit_conditions") or []),
    ):
        for condition in conditions:
            if not isinstance(condition, dict):
                continue
            value = condition.get("value")
            rules.append(
                {
                    "direction": direction,
                    "indicator": condition.get("indicator", ""),
                    "operator": condition.get("operator", ""),
                    "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value,
                    "window": condition.get("window") or "—",
                }
            )

    points = [
        {"label": str(item.get("date", "")), "value": float(item.get("value", 0) or 0)}
        for item in backtest.get("equity_curve") or []
        if isinstance(item, dict) and item.get("date") is not None
    ]
    exposure_points = [
        {"label": str(item.get("date", "")), "value": float(item.get("target_exposure", 0) or 0)}
        for item in backtest.get("equity_curve") or []
        if isinstance(item, dict) and item.get("date") is not None
    ]
    trades = [
        {
            "date": trade.get("date", ""),
            "action": "买入" if str(trade.get("action", "")).lower() == "buy" else "卖出",
            "shares": trade.get("shares", 0),
            "price": trade.get("price", 0),
            "amount": trade.get("amount", 0),
        }
        for trade in (backtest.get("trades") or [])[-20:]
        if isinstance(trade, dict)
    ]

    root_children = ["header", "meta", "description", "validation-summary"]
    components: list[dict[str, Any]] = [
        {"id": "root", "component": "Card", "children": root_children},
        {"id": "header", "component": "Row", "children": ["title", "status", "eligibility"]},
        _text("title", f"代码策略候选 · {payload.get('name') or strategy.get('name', '未命名策略')}", "h3"),
        {
            "id": "status",
            "component": "Badge",
            "text": f"状态：{status_label}",
            "tone": "buy" if validation_passed else "sell",
        },
        {
            "id": "eligibility",
            "component": "Badge",
            "text": eligibility_label,
            "tone": "buy" if promotion_eligible else "hold",
        },
        _text(
            "meta",
            (
                f"{payload.get('candidate_id', '—')} · {payload.get('ticker', '—')} · "
                f"{str(payload.get('asset_type', '')).upper()} · v{payload.get('version', '—')}"
            ),
            "caption",
        ),
        _text("description", strategy.get("description") or "Agent 未提供策略说明。", "body"),
        _text(
            "validation-summary",
            f"{validation_label} · {eligibility_label}",
            "body",
            tone="positive" if validation_passed else "negative",
        ),
    ]

    if backtest.get("final_value") is not None:
        root_children.append("performance")
        components.extend(
            [
                {
                    "id": "performance",
                    "component": "Section",
                    "title": "可信回测引擎结果",
                    "children": ["performance-row"],
                },
                {
                    "id": "performance-row",
                    "component": "Row",
                    "children": ["final", "return", "benchmark", "drawdown", "sharpe", "trade-count"],
                },
                _text("final", f"最终市值 ¥{_number_label(backtest.get('final_value'))}", "caption"),
                _text("return", f"策略收益 {_percent_label(backtest.get('total_return'))}", "caption"),
                _text("benchmark", f"买入持有 {_percent_label(backtest.get('buy_hold_return'))}", "caption"),
                _text(
                    "drawdown",
                    f"最大回撤 {_percent_label(backtest.get('max_drawdown'), signed=False)}",
                    "caption",
                    tone="negative",
                ),
                _text("sharpe", f"Sharpe {_number_label(backtest.get('sharpe_ratio'))}", "caption"),
                _text("trade-count", f"交易 {backtest.get('total_trades', len(trades))} 次", "caption"),
            ]
        )
    elif not validation_passed:
        root_children.append("backtest-unavailable")
        components.append(
            _text("backtest-unavailable", "脚本未通过验证，因此没有执行绩效回测。", "body", tone="negative")
        )

    if points:
        root_children.append("chart")
        components.extend(
            [
                {"id": "chart", "component": "LineChart", "points": _ref("/points"), "ariaLabel": "代码策略资金曲线"},
                {
                    "id": "exposure-chart",
                    "component": "Collapsible",
                    "title": "目标仓位曲线",
                    "defaultExpanded": False,
                    "children": ["exposure-line"],
                },
                {
                    "id": "exposure-line",
                    "component": "LineChart",
                    "points": _ref("/exposurePoints"),
                    "ariaLabel": "代码策略目标仓位曲线",
                },
            ]
        )
        root_children.append("exposure-chart")
    if trades:
        root_children.append("trades")
        components.extend(
            [
                {
                    "id": "trades",
                    "component": "Collapsible",
                    "title": "最近交易记录",
                    "defaultExpanded": False,
                    "children": ["trade-table"],
                },
                {
                    "id": "trade-table",
                    "component": "DataTable",
                    "columns": [
                        {"key": "date", "label": "日期"},
                        {"key": "action", "label": "方向"},
                        {"key": "shares", "label": "数量"},
                        {"key": "price", "label": "价格"},
                        {"key": "amount", "label": "金额"},
                    ],
                    "rows": _ref("/trades"),
                },
            ]
        )
    if rules:
        root_children.append("rules")
        components.extend(
            [
                {
                    "id": "rules",
                    "component": "Collapsible",
                    "title": "结构化交易规则",
                    "defaultExpanded": False,
                    "children": ["rule-table"],
                },
                {
                    "id": "rule-table",
                    "component": "DataTable",
                    "columns": [
                        {"key": "direction", "label": "阶段"},
                        {"key": "indicator", "label": "指标"},
                        {"key": "operator", "label": "关系"},
                        {"key": "value", "label": "阈值"},
                        {"key": "window", "label": "窗口"},
                    ],
                    "rows": _ref("/rules"),
                },
            ]
        )

    validation_children = ["validation-table"]
    if errors:
        validation_children.append("validation-errors")
    root_children.append("validation")
    components.extend(
        [
            {
                "id": "validation",
                "component": "Collapsible",
                "title": (
                    f"沙盒验证明细（{sum(row['status'] == '通过' for row in validation_rows)}"
                    f"/{len(validation_rows)}）"
                ),
                "defaultExpanded": not validation_passed,
                "children": validation_children,
            },
            {
                "id": "validation-table",
                "component": "DataTable",
                "columns": [
                    {"key": "category", "label": "类别"},
                    {"key": "check", "label": "检查项"},
                    {"key": "status", "label": "结果"},
                ],
                "rows": _ref("/validationRows"),
            },
        ]
    )
    if errors:
        components.append(
            {"id": "validation-errors", "component": "List", "title": "失败原因", "items": _ref("/errors")}
        )

    root_children.extend(["code", "notice"])
    components.extend(
        [
            {
                "id": "code",
                "component": "Collapsible",
                "title": "Agent 生成的 Python 信号脚本",
                "defaultExpanded": False,
                "children": ["source-hash", "source-code"],
            },
            _text("source-hash", f"SHA-256：{payload.get('source_sha256', '—')}", "caption"),
            {"id": "source-code", "component": "CodeBlock", "language": "python", "code": _ref("/sourceCode")},
            _text(
                "notice",
                "脚本仅在受限沙盒中生成目标仓位；成交、费用和绩效统一由可信回测引擎计算。仅用于研究和模拟盘。",
                "caption",
            ),
        ]
    )
    return _surface(
        surface_id,
        components,
        {
            "points": points,
            "exposurePoints": exposure_points,
            "trades": trades,
            "rules": rules,
            "validationRows": validation_rows,
            "errors": errors,
            "sourceCode": source_code,
        },
    )


def _normalised_curve(item: dict[str, Any]) -> list[dict[str, Any]]:
    points = [point for point in item.get("equity_curve", []) if isinstance(point, dict)]
    first = next((float(point.get("value")) for point in points if point.get("value") not in (None, 0)), None)
    if first is None:
        return []
    return [
        {"label": str(point.get("date", "")), "value": round(float(point.get("value", 0)) / first, 6)}
        for point in points
        if point.get("date") is not None and point.get("value") is not None
    ]


def _comparison_winner_rows(conclusion: dict[str, Any]) -> list[dict[str, Any]]:
    labels = (
        ("绝对收益", "absolute_return_winner"),
        ("风险收益", "risk_adjusted_winner"),
        ("回撤控制", "drawdown_winner"),
        ("样本外", "out_of_sample_winner"),
        ("稳健性", "robustness_winner"),
    )
    rows = []
    for label, key in labels:
        item = conclusion.get(key) or {}
        value = item.get("value")
        metric = str(item.get("metric") or "")
        display = _number_label(value)
        if "return" in metric or "drawdown" in metric:
            display = _percent_label(value, signed="drawdown" not in metric)
        rows.append(
            {
                "category": label,
                "strategy": item.get("display_name") or item.get("strategy_name") or "暂无正式结论",
                "metric": metric or "—",
                "value": display,
            }
        )
    return rows


def render_strategy_comparison(
    payload: dict[str, Any],
    surface_id: str | None = None,
) -> list[dict[str, Any]]:
    """Render a complete, auditable multi-strategy research surface."""
    comparisons = [item for item in payload.get("comparisons", []) if isinstance(item, dict)]
    rows = []
    for item in comparisons:
        diagnostics = item.get("diagnostics") or {}
        out_of_sample = diagnostics.get("out_of_sample") or {}
        rows.append(
            {
                "strategy": item.get("display_name") or item.get("strategy_name", ""),
                "return": _percent_label(item.get("total_return")),
                "annualized": _percent_label(item.get("annualized_return")),
                "drawdown": _percent_label(item.get("max_drawdown"), signed=False),
                "sharpe": _number_label(item.get("sharpe_ratio")),
                "calmar": _number_label(item.get("calmar_ratio")),
                "oos": _percent_label(out_of_sample.get("out_of_sample_return")),
                "exposure": _percent_label(item.get("exposure")),
                "turnover": _number_label(item.get("turnover")),
                "fees": f"¥{_number_label(item.get('total_fees'))}",
                "profit": "¥"
                + _number_label(
                    float(item.get("final_value") or 0) - float(payload.get("initial_capital") or 0)
                ),
            }
        )

    by_name = {item.get("strategy_name"): item for item in comparisons}
    ranked_names = [payload.get("benchmark"), *(payload.get("ranking") or [])]
    chart_items = []
    for name in ranked_names:
        if name in by_name and by_name[name] not in chart_items:
            chart_items.append(by_name[name])
        if len(chart_items) >= 4:
            break
    equity_series = [
        {
            "name": item.get("display_name") or item.get("strategy_name"),
            "points": _normalised_curve(item),
        }
        for item in chart_items
    ]
    drawdown_series = [
        {
            "name": item.get("display_name") or item.get("strategy_name"),
            "points": [
                {"label": str(point.get("date", "")), "value": point.get("value")}
                for point in item.get("drawdown_curve", [])
                if isinstance(point, dict)
            ],
        }
        for item in chart_items
    ]
    exposure_series = [
        {
            "name": item.get("display_name") or item.get("strategy_name"),
            "points": [
                {"label": str(point.get("date", "")), "value": point.get("target_exposure")}
                for point in item.get("signal_curve", [])
                if isinstance(point, dict) and point.get("target_exposure") is not None
            ],
        }
        for item in comparisons
        if ((item.get("strategy_spec") or {}).get("position_model") or {}).get("type")
        in {"volatility_target", "trend_volatility_target"}
    ]

    conclusion = dict(payload.get("conclusion") or {})
    validation = dict(payload.get("data_validation") or {})
    acceptance = dict(payload.get("acceptance") or {})
    acceptance_rows = [
        {"check": key, "status": "通过" if value else "未通过"}
        for key, value in (acceptance.get("checks") or {}).items()
    ]
    cost_rows = [
        {
            "scenario": scenario,
            "strategy": row.get("strategy_name"),
            "return": _percent_label(row.get("total_return")),
            "drawdown": _percent_label(row.get("max_drawdown"), signed=False),
            "fees": f"¥{_number_label(row.get('total_fees'))}",
        }
        for scenario, scenario_rows in (payload.get("cost_scenarios") or {}).items()
        for row in scenario_rows
    ]
    stability_rows = []
    sensitivity = payload.get("parameter_sensitivity") or {}
    for item in comparisons:
        diagnostics = item.get("diagnostics") or {}
        out_of_sample = diagnostics.get("out_of_sample") or {}
        rolling = diagnostics.get("rolling") or []
        if isinstance(rolling, list):
            rolling_returns = [
                float(window["total_return"])
                for window in rolling
                if isinstance(window, dict) and window.get("total_return") is not None
            ]
            rolling_positive_ratio = (
                sum(value > 0 for value in rolling_returns) / len(rolling_returns) if rolling_returns else None
            )
        elif isinstance(rolling, dict):
            # Keep compatibility with older persisted comparison payloads.
            rolling_positive_ratio = rolling.get("positive_ratio")
        else:
            rolling_positive_ratio = None
        stable = sensitivity.get(item.get("strategy_name"), {})
        stability_rows.append(
            {
                "strategy": item.get("display_name") or item.get("strategy_name"),
                "oos": _percent_label(out_of_sample.get("out_of_sample_return")),
                "rollingPositive": _percent_label(rolling_positive_ratio),
                "sensitivity": stable.get("status", "—"),
                "worst": _percent_label(stable.get("worst_return")),
            }
        )
    source_rows = [
        {
            "source": item.get("source_name") or item.get("source_id"),
            "selected": "是" if item.get("selected") else "否",
            "score": _number_label(item.get("quality_score")),
            "period": f"{item.get('actual_start_date') or '—'} 至 {item.get('actual_end_date') or '—'}",
            "rows": item.get("row_count", 0),
            "hash": str(item.get("sha256") or "")[:12],
        }
        for item in validation.get("candidates", [])
        if isinstance(item, dict)
    ]
    strategy_rows = [
        {
            "strategy": item.get("display_name") or item.get("strategy_name"),
            "entry": json.dumps(item.get("entry_rules") or [], ensure_ascii=False),
            "exit": json.dumps(item.get("exit_rules") or [], ensure_ascii=False),
            "position": json.dumps(
                (item.get("strategy_spec") or {}).get("position_model")
                or {"type": "fixed", "max_exposure": (item.get("strategy_spec") or {}).get("position_size_pct")},
                ensure_ascii=False,
            ),
        }
        for item in comparisons
    ]
    artifact_rows = [
        {
            "name": artifact.get("name"),
            "mimeType": artifact.get("mime_type"),
            "size": artifact.get("size_bytes"),
            "previewUrl": artifact.get("preview_url"),
            "downloadUrl": artifact.get("download_url"),
        }
        for artifact in payload.get("artifacts", [])
        if isinstance(artifact, dict)
    ]

    surface_id = surface_id or f"strategy-comparison-{uuid4().hex}"
    acceptance_label = "验收通过" if acceptance.get("satisfied") else "验收未通过"
    official_label = "正式结论" if conclusion.get("official") else "探索性结果"
    components: list[dict[str, Any]] = [
        {
            "id": "root",
            "component": "Card",
            "children": [
                "title",
                "meta",
                "status-row",
                "winners",
                "table",
                "equity",
                "drawdown",
                *(["exposure"] if exposure_series else []),
                "costs",
                "stability",
                "sources",
                "contract",
                "definitions",
                "conclusion",
                "artifacts",
                "notice",
            ],
        },
        _text("title", f"{payload.get('ticker', '')} 多策略回测研究", "h3"),
        _text(
            "meta",
            (
                f"评价期 {payload.get('evaluation_start_date', '—')} 至 {payload.get('evaluation_end_date', '—')} · "
                f"热身 {payload.get('warmup_bars', 0)} 个交易日 · {len(comparisons)} 个策略 · "
                f"选定来源 {validation.get('selected_source', '—')}"
            ),
            "caption",
        ),
        {"id": "status-row", "component": "Row", "children": ["verification", "acceptance", "official"]},
        {
            "id": "verification",
            "component": "Badge",
            "text": f"数据核验：{validation.get('status', 'unknown')}",
            "tone": "buy" if validation.get("status") == "verified" else "hold",
        },
        {
            "id": "acceptance",
            "component": "Badge",
            "text": acceptance_label,
            "tone": "buy" if acceptance.get("satisfied") else "sell",
        },
        {
            "id": "official",
            "component": "Badge",
            "text": official_label,
            "tone": "buy" if conclusion.get("official") else "hold",
        },
        {
            "id": "winners",
            "component": "Section",
            "title": "五类优胜者",
            "children": ["winner-table"],
        },
        {
            "id": "winner-table",
            "component": "DataTable",
            "columns": [
                {"key": "category", "label": "维度"},
                {"key": "strategy", "label": "策略"},
                {"key": "metric", "label": "指标"},
                {"key": "value", "label": "结果"},
            ],
            "rows": _ref("/winners"),
        },
        {
            "id": "table",
            "component": "DataTable",
            "columns": [
                {"key": "strategy", "label": "策略"},
                {"key": "return", "label": "收益"},
                {"key": "annualized", "label": "年化"},
                {"key": "drawdown", "label": "回撤"},
                {"key": "sharpe", "label": "夏普"},
                {"key": "calmar", "label": "Calmar"},
                {"key": "oos", "label": "样本外"},
                {"key": "exposure", "label": "平均仓位"},
                {"key": "turnover", "label": "换手率"},
                {"key": "fees", "label": "费用"},
                {"key": "profit", "label": "期末盈利"},
            ],
            "rows": _ref("/rows"),
        },
        {
            "id": "equity",
            "component": "Collapsible",
            "title": "买入持有与排名前三：归一化净值",
            "defaultExpanded": True,
            "children": ["equity-chart"],
        },
        {
            "id": "equity-chart",
            "component": "MultiLineChart",
            "series": _ref("/equitySeries"),
            "ariaLabel": "多策略归一化净值",
        },
        {
            "id": "drawdown",
            "component": "Collapsible",
            "title": "买入持有与排名前三：回撤",
            "defaultExpanded": False,
            "children": ["drawdown-chart"],
        },
        {
            "id": "drawdown-chart",
            "component": "MultiLineChart",
            "series": _ref("/drawdownSeries"),
            "ariaLabel": "多策略回撤",
        },
        *(
            [
                {
                    "id": "exposure",
                    "component": "Collapsible",
                    "title": "动态策略目标仓位",
                    "defaultExpanded": False,
                    "children": ["exposure-chart"],
                },
                {
                    "id": "exposure-chart",
                    "component": "MultiLineChart",
                    "series": _ref("/exposureSeries"),
                    "ariaLabel": "动态策略目标仓位",
                },
            ]
            if exposure_series
            else []
        ),
        {
            "id": "costs",
            "component": "Collapsible",
            "title": "成本低 / 基准 / 压力情景",
            "defaultExpanded": False,
            "children": ["cost-table"],
        },
        {
            "id": "cost-table",
            "component": "DataTable",
            "columns": [
                {"key": "scenario", "label": "情景"},
                {"key": "strategy", "label": "策略"},
                {"key": "return", "label": "收益"},
                {"key": "drawdown", "label": "回撤"},
                {"key": "fees", "label": "费用"},
            ],
            "rows": _ref("/costRows"),
        },
        {
            "id": "stability",
            "component": "Collapsible",
            "title": "样本外、滚动表现与参数敏感性",
            "defaultExpanded": False,
            "children": ["stability-table"],
        },
        {
            "id": "stability-table",
            "component": "DataTable",
            "columns": [
                {"key": "strategy", "label": "策略"},
                {"key": "oos", "label": "样本外收益"},
                {"key": "rollingPositive", "label": "滚动正收益率"},
                {"key": "sensitivity", "label": "敏感性"},
                {"key": "worst", "label": "最差参数收益"},
            ],
            "rows": _ref("/stabilityRows"),
        },
        {
            "id": "sources",
            "component": "Collapsible",
            "title": "多数据源差异与自动选择理由",
            "defaultExpanded": validation.get("status") != "verified",
            "children": ["source-reason", "source-table"],
        },
        _text("source-reason", str(validation.get("selection_reason") or "未提供选择理由"), "caption"),
        {
            "id": "source-table",
            "component": "DataTable",
            "columns": [
                {"key": "source", "label": "来源"},
                {"key": "selected", "label": "选中"},
                {"key": "score", "label": "质量分"},
                {"key": "period", "label": "实际区间"},
                {"key": "rows", "label": "行数"},
                {"key": "hash", "label": "SHA-256"},
            ],
            "rows": _ref("/sourceRows"),
        },
        {
            "id": "contract",
            "component": "Collapsible",
            "title": "任务契约与验收检查",
            "defaultExpanded": not acceptance.get("satisfied", False),
            "children": ["contract-table"],
        },
        {
            "id": "contract-table",
            "component": "DataTable",
            "columns": [{"key": "check", "label": "检查项"}, {"key": "status", "label": "结果"}],
            "rows": _ref("/acceptanceRows"),
        },
        {
            "id": "definitions",
            "component": "Collapsible",
            "title": "策略入场、退出与仓位模型",
            "defaultExpanded": False,
            "children": ["definition-table"],
        },
        {
            "id": "definition-table",
            "component": "DataTable",
            "columns": [
                {"key": "strategy", "label": "策略"},
                {"key": "entry", "label": "入场"},
                {"key": "exit", "label": "退出"},
                {"key": "position", "label": "仓位模型"},
            ],
            "rows": _ref("/strategyRows"),
        },
        {
            "id": "conclusion",
            "component": "Collapsible",
            "title": "结论、权衡与局限",
            "defaultExpanded": True,
            "children": ["tradeoffs", "warnings", "limitations"],
        },
        {"id": "tradeoffs", "component": "List", "title": "权衡", "items": _ref("/tradeoffs")},
        {"id": "warnings", "component": "List", "title": "数据警告", "items": _ref("/warnings")},
        {"id": "limitations", "component": "List", "title": "局限", "items": _ref("/limitations")},
        {
            "id": "artifacts",
            "component": "Collapsible",
            "title": f"完整可审计成果包（{len(artifact_rows)} 个文件）",
            "defaultExpanded": True,
            "children": ["artifact-list"],
        },
        {
            "id": "artifact-list",
            "component": "List",
            "items": _ref("/artifacts"),
            "itemTemplate": "artifact-item",
        },
        {
            "id": "artifact-item",
            "component": "ArtifactLink",
            "name": _ref("name"),
            "mimeType": _ref("mimeType"),
            "size": _ref("size"),
            "previewUrl": _ref("previewUrl"),
            "downloadUrl": _ref("downloadUrl"),
        },
        _text("notice", "回测仅用于研究与模拟盘，不代表未来表现，也不构成直接交易建议。", "caption"),
    ]
    model = {
        "rows": rows,
        "winners": _comparison_winner_rows(conclusion),
        "equitySeries": equity_series,
        "drawdownSeries": drawdown_series,
        "exposureSeries": exposure_series,
        "costRows": cost_rows,
        "stabilityRows": stability_rows,
        "sourceRows": source_rows,
        "acceptanceRows": acceptance_rows,
        "strategyRows": strategy_rows,
        "artifacts": artifact_rows,
        "tradeoffs": [*(conclusion.get("tradeoffs") or []), *(conclusion.get("interpretations") or [])],
        "warnings": conclusion.get("data_warnings") or ["未发现额外数据警告"],
        "limitations": conclusion.get("limitations") or [],
    }
    return _surface(surface_id, components, model)
