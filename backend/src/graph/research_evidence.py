"""Evidence normalization and deterministic research summaries."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from models.research_plan import EvidenceRef


def _compact(value: Any, *, depth: int = 0) -> Any:
    if depth == 0 and isinstance(value, dict) and value.get("data_type") == "strategy_backtest_comparison":
        return _compact_strategy_comparison(value)
    if depth >= 5:
        return str(value)[:500]
    if isinstance(value, dict):
        return {str(key): _compact(item, depth=depth + 1) for key, item in list(value.items())[:30]}
    if isinstance(value, list):
        return [_compact(item, depth=depth + 1) for item in value[:20]]
    if isinstance(value, str):
        return value[:4000]
    return value


def _sample_curve(points: Any, maximum: int = 240) -> list[dict[str, Any]]:
    if not isinstance(points, list):
        return []
    rows = [item for item in points if isinstance(item, dict)]
    if len(rows) <= maximum:
        return rows
    step = max(1, (len(rows) - 1) // (maximum - 1) + 1)
    sampled = rows[::step]
    if sampled[-1] is not rows[-1]:
        sampled.append(rows[-1])
    return sampled[: maximum - 1] + [rows[-1]] if len(sampled) > maximum else sampled


def _compact_strategy_comparison(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep A2UI/audit metadata while bounding checkpoint and SSE payload size."""
    comparison_fields = {
        "strategy_name",
        "display_name",
        "description",
        "strategy_spec",
        "entry_rules",
        "exit_rules",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "win_rate",
        "profit_factor",
        "exposure",
        "turnover",
        "total_fees",
        "excess_return",
        "final_value",
        "total_trades",
        "metrics",
        "diagnostics",
        "error",
    }
    comparisons = []
    for row in payload.get("comparisons", []):
        if not isinstance(row, dict):
            continue
        compact_row = {key: value for key, value in row.items() if key in comparison_fields}
        compact_row["equity_curve"] = _sample_curve(row.get("equity_curve"))
        compact_row["drawdown_curve"] = _sample_curve(row.get("drawdown_curve"))
        compact_row["signal_curve"] = _sample_curve(row.get("signal_curve"))
        comparisons.append(compact_row)
    keep = {
        "comparison_id",
        "data_type",
        "_tool_name",
        "available",
        "data_status",
        "message",
        "error",
        "ticker",
        "asset_type",
        "start_date",
        "end_date",
        "requested_start_date",
        "warmup_start_date",
        "evaluation_start_date",
        "evaluation_end_date",
        "warmup_bars",
        "actual_start_date",
        "actual_end_date",
        "history_years",
        "initial_capital",
        "strategy_count",
        "benchmark",
        "ranking_metric",
        "ranking_label",
        "task_contract",
        "data_snapshot",
        "execution",
        "ranking",
        "cost_scenarios",
        "cost_consistency",
        "parameter_sensitivity",
        "market_benchmark",
        "acceptance",
        "conclusion",
        "artifacts",
        "artifact_error",
        "provenance",
        "decision_interval",
    }
    compact = {key: value for key, value in payload.items() if key in keep}
    compact["comparisons"] = comparisons
    validation = dict(payload.get("data_validation") or {})
    validation["differences"] = (validation.get("differences") or [])[:20]
    compact["data_validation"] = validation
    market_benchmark = dict(compact.get("market_benchmark") or {})
    market_benchmark["comparisons"] = [
        {
            key: row.get(key)
            for key in (
                "strategy_name",
                "display_name",
                "asset_total_return",
                "market_total_return",
                "excess_return",
                "asset_max_drawdown",
                "market_max_drawdown",
                "drawdown_improvement",
                "asset_sharpe_ratio",
                "market_sharpe_ratio",
                "market_error",
            )
        }
        for row in (market_benchmark.get("comparisons") or [])
        if isinstance(row, dict)
    ]
    compact["market_benchmark"] = market_benchmark
    return compact


def _evidence(payload: dict[str, Any], kind: str) -> list[EvidenceRef]:
    now = datetime.now(timezone.utc).isoformat()

    def collect_provenance(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [row for item in value for row in collect_provenance(item)]
        if not isinstance(value, dict):
            return []
        found: list[dict[str, Any]] = []
        provenance = value.get("provenance")
        if isinstance(provenance, list):
            found.extend(item for item in provenance if isinstance(item, dict))
        elif isinstance(provenance, dict):
            found.append(provenance)
        for key, child in value.items():
            if key != "provenance":
                found.extend(collect_provenance(child))
        return found

    rows = collect_provenance(payload)
    source_type = (
        "web"
        if kind == "news"
        else "methodology"
        if kind == "methodology"
        else "backtest"
        if kind == "backtest"
        else "derived"
        if kind in {"risk", "synthesis"}
        else "market_data"
    )
    result = []
    for row in rows or [{}]:
        result.append(
            EvidenceRef(
                source=str(
                    row.get("name")
                    or row.get("source")
                    or row.get("source_id")
                    or row.get("provider")
                    or payload.get("data_type")
                    or kind
                ),
                source_type=source_type,
                as_of=str(
                    row.get("as_of")
                    or payload.get("searched_at")
                    or (row.get("fetched_at") if row.get("freshness") in {"realtime", "latest_available"} else "")
                    or ""
                )
                or None,
                retrieved_at=str(row.get("fetched_at") or now),
                data_status=str(row.get("status") or "available"),
                url=row.get("url"),
                content_hash=hashlib.sha256(
                    json.dumps(_compact(payload), sort_keys=True, default=str).encode()
                ).hexdigest(),
            )
        )
    return result


def _result_summaries(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "step_id": item.get("step_id"),
            "status": item.get("status"),
            "summary": item.get("summary"),
            "evidence_status": item.get("evidence_status", "not_assessed"),
            "evidence_issues": item.get("evidence_issues", []),
            "output": _synthesis_output(item.get("output", {})),
        }
        for item in results.values()
    ]


def _synthesis_request(request: dict[str, Any]) -> dict[str, Any]:
    """Keep only current-task fields; cross-turn history is not synthesis evidence."""
    keys = (
        "message",
        "intent",
        "tickers",
        "asset_type",
        "strategy",
        "as_of_date",
    )
    return {key: request.get(key) for key in keys if request.get(key) is not None}


def _deterministic_synthesis_fallback(evidence: list[dict[str, Any]]) -> str:
    completed = [
        (
            f"- {item.get('step_id')}: {item.get('summary') or '已完成'}"
            + (
                f"（证据{item.get('evidence_status')}：{'；'.join(item.get('evidence_issues') or [])}）"
                if item.get("evidence_status") in {"limited", "unavailable"}
                else ""
            )
        )
        for item in evidence
        if item.get("status") == "completed"
    ]
    failed = [str(item.get("step_id")) for item in evidence if item.get("status") in {"failed", "skipped"}]
    lines = ["研究证据已完成汇总。", *(completed[:12] or ["- 暂无足够的已完成证据步骤。"])]
    if failed:
        lines.append("数据不足或未完成步骤：" + "、".join(failed[:8]) + "。")
    lines.append("请结合证据日期、数据缺失和风险约束审慎判断；以上仅用于短中期研究与模拟交易，不承诺收益。")
    return "\n".join(lines)


def _synthesis_output(output: Any) -> Any:
    """Bound the evidence sent to synthesis without weakening the public A2UI payload."""
    if not isinstance(output, dict) or output.get("data_type") != "strategy_backtest_comparison":
        return _compact(output)
    metric_fields = (
        "strategy_name",
        "display_name",
        "total_return",
        "sharpe_ratio",
        "max_drawdown",
        "calmar_ratio",
        "total_fees",
        "final_value",
    )
    return {
        "data_type": output.get("data_type"),
        "available": output.get("available", True),
        "data_status": output.get("data_status"),
        "message": output.get("message"),
        "error": output.get("error"),
        "ticker": output.get("ticker"),
        "evaluation_start_date": output.get("evaluation_start_date"),
        "evaluation_end_date": output.get("evaluation_end_date"),
        "warmup_bars": output.get("warmup_bars"),
        "strategy_count": output.get("strategy_count"),
        "comparisons": [
            {key: row.get(key) for key in metric_fields}
            for row in (output.get("comparisons") or [])
            if isinstance(row, dict)
        ],
        "conclusion": output.get("conclusion"),
        "data_validation": {
            key: (output.get("data_validation") or {}).get(key)
            for key in ("status", "selected_source", "selection_reason", "rule_version")
        },
        "execution": output.get("execution"),
        "acceptance": output.get("acceptance"),
        "market_benchmark": {
            **{
                key: (output.get("market_benchmark") or {}).get(key)
                for key in (
                    "status",
                    "ticker",
                    "name",
                    "evaluation_start_date",
                    "evaluation_end_date",
                    "coverage_ratio",
                    "error",
                    "simulation_note",
                )
            },
            "comparisons": [
                {
                    key: row.get(key)
                    for key in (
                        "strategy_name",
                        "display_name",
                        "asset_total_return",
                        "market_total_return",
                        "excess_return",
                        "asset_max_drawdown",
                        "market_max_drawdown",
                        "drawdown_improvement",
                    )
                }
                for row in ((output.get("market_benchmark") or {}).get("comparisons") or [])
                if isinstance(row, dict)
            ],
        },
        "artifacts": [
            {key: artifact.get(key) for key in ("name", "mime_type", "size_bytes")}
            for artifact in (output.get("artifacts") or [])
            if isinstance(artifact, dict)
        ],
    }


def _format_winner_metric(metric: str, value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "指标缺失"
    if metric in {"total_return", "max_drawdown", "out_of_sample_return", "stress_total_return"}:
        sign = "" if metric == "max_drawdown" else "+"
        return f"{sign}{number:.2%}"
    return f"{number:.3f}"


def _comparison_synthesis_text(payload: dict[str, Any]) -> str:
    """Render frozen comparison facts without asking an LLM to re-rank strategies."""
    conclusion = payload.get("conclusion") or {}
    if payload.get("available") is False:
        error = payload.get("error")
        message = error.get("message") if isinstance(error, dict) else error
        warnings = [str(item) for item in conclusion.get("data_warnings", []) if str(item)]
        reason = str(message or payload.get("message") or (warnings[0] if warnings else "没有可用历史数据"))
        limitations = [str(item) for item in conclusion.get("limitations", []) if str(item)]
        lines = [
            f"{payload.get('ticker') or '该标的'} 的多策略回测工具已正常返回，但没有产生回测结果。",
            f"数据说明：{reason}",
            "由于没有可比较的策略结果，本步骤不形成优胜策略；应结合其他已取得的证据继续判断。",
        ]
        if limitations:
            lines.append("局限：" + "；".join(limitations))
        lines.append("以上仅用于研究与模拟盘，不构成收益承诺或直接交易建议。")
        return "\n".join(lines)
    validation = payload.get("data_validation") or {}
    execution = payload.get("execution") or {}
    snapshot = payload.get("data_snapshot") or {}
    lines = [
        f"{payload.get('ticker') or '该标的'} 多策略回测已完成。统一评价区间为 "
        f"{payload.get('evaluation_start_date', '—')} 至 "
        f"{payload.get('evaluation_end_date', '—')}，热身期 {payload.get('warmup_bars', '—')} 个交易日，"
        f"共比较 {payload.get('strategy_count') or len(payload.get('comparisons') or [])} 个策略。",
        (
            f"行情采用 {snapshot.get('adjustment') or '前复权/数据源声明方式'}；基准成交成本为买入佣金 "
            f"{float(execution.get('buy_commission_rate') or 0):.3%}、卖出佣金 "
            f"{float(execution.get('sell_commission_rate') or 0):.3%}、滑点 "
            f"{float(execution.get('slippage_bps') or 0):g} bps，并按下一交易日开盘执行。"
        ),
        "",
        "程序确定的五类优胜者：",
    ]
    winner_labels = (
        ("absolute_return_winner", "绝对收益"),
        ("risk_adjusted_winner", "风险收益"),
        ("drawdown_winner", "回撤控制"),
        ("out_of_sample_winner", "样本外"),
        ("robustness_winner", "稳健性"),
    )
    winner_count = 0
    for key, label in winner_labels:
        winner = conclusion.get(key)
        if not isinstance(winner, dict):
            continue
        winner_count += 1
        name = winner.get("display_name") or winner.get("strategy_name") or "—"
        lines.append(
            f"- {label}：{name}（{winner.get('metric', 'metric')} "
            f"{_format_winner_metric(str(winner.get('metric') or ''), winner.get('value'))}）"
        )
    if not winner_count:
        lines.append("- 当前数据覆盖或核验状态不足以形成正式优胜者，结果仅作探索性比较。")
    lines.extend(["", "为什么不存在唯一“最好策略”："])
    lines.extend(f"- {item}" for item in (conclusion.get("tradeoffs") or ["不同评价维度对应不同交易取舍。"]))
    lines.extend(
        [
            "",
            f"数据核验状态：{validation.get('status', 'unknown')}；选定数据源："
            f"{validation.get('selected_source') or '未记录'}。{validation.get('selection_reason') or ''}",
        ]
    )
    warnings = conclusion.get("data_warnings") or []
    limitations = conclusion.get("limitations") or []
    if warnings:
        lines.append("数据警告：" + "；".join(str(item) for item in warnings))
    if limitations:
        lines.append("局限：" + "；".join(str(item) for item in limitations))
    market_benchmark = payload.get("market_benchmark") or {}
    market_comparisons = [
        item
        for item in (market_benchmark.get("comparisons") or [])
        if isinstance(item, dict) and item.get("excess_return") is not None
    ]
    if market_benchmark.get("status") in {"available", "partial"} and market_comparisons:
        leading_relative = max(market_comparisons, key=lambda item: float(item["excess_return"]))
        lines.append(
            f"同期大盘对比：以 {market_benchmark.get('name') or market_benchmark.get('ticker')} 为基准，"
            f"同策略相对表现最高的是 "
            f"{leading_relative.get('display_name') or leading_relative.get('strategy_name')}，"
            f"当前标的相对大盘同策略超额为 {float(leading_relative['excess_return']):+.2%}。"
        )
    elif market_benchmark.get("status") == "unavailable":
        lines.append(f"同期大盘对比暂不可用：{market_benchmark.get('error') or '未取得指数行情'}。")
    artifacts = payload.get("artifacts") or []
    if artifacts:
        lines.append(
            f"完整可审计成果包已生成，共 {len(artifacts)} 个文件，包含 HTML、XLSX、JSON 与 CSV；"
            "可在上方成果区预览或下载。"
        )
    else:
        lines.append("本次未生成可下载的回测成果文件。")
    lines.append("以上仅用于研究与模拟盘，不构成收益承诺或直接交易建议。")
    return "\n".join(lines)


def _find_price(results: dict[str, dict[str, Any]]) -> float | None:
    def walk(value: Any) -> float | None:
        if isinstance(value, dict):
            quote = value.get("quote")
            if isinstance(quote, dict):
                try:
                    price = float(quote.get("price") or quote.get("最新价") or 0)
                    if price > 0:
                        return price
                except (TypeError, ValueError):
                    pass
            for child in value.values():
                found = walk(child)
                if found:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = walk(child)
                if found:
                    return found
        return None

    return walk(results)


def _classify_failure(error: str) -> str:
    """Classify a public tool error before deciding whether reflection is useful."""
    text = error.lower()
    if any(
        token in text
        for token in (
            "user_denied",
            "unauthorized",
            "forbidden",
            "permission",
            "用户拒绝",
            "没有权限",
            "权限不足",
            "工具不可用",
            "不支持的研究步骤",
        )
    ):
        return "terminal"
    if any(
        token in text
        for token in (
            "timeout",
            "timed out",
            "rate limit",
            "429",
            "connection",
            "temporarily",
            "service unavailable",
            "超时",
            "限流",
            "网络",
            "连接失败",
            "暂时不可用",
            "服务繁忙",
        )
    ):
        return "transient"
    if any(
        token in text
        for token in (
            "invalid",
            "unsupported",
            "validation",
            "schema",
            "required",
            "missing",
            "argument",
            "parameter",
            "校验",
            "验证",
            "参数",
            "缺少",
            "不能为空",
            "不受支持",
            "指标",
            "格式",
        )
    ):
        return "correctable"
    return "unknown"
