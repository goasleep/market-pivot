"""Auditable deliverables for formal multi-strategy comparisons."""

from __future__ import annotations

import base64
import csv
import html
import io
import json
from typing import Any

import xlsxwriter
from xlsxwriter.utility import xl_col_to_name

from artifacts.service import artifact_service

_GATE_LABELS = {
    "task_acceptance": "任务验收通过",
    "official_sample": "样本长度达到正式标准",
    "data_cross_validation": "行情交叉核验通过",
    "out_of_sample_positive": "样本外收益为正",
    "rolling_positive_majority": "多数滚动窗口为正",
    "parameters_stable": "相邻参数表现稳定",
    "stress_cost_positive": "压力成本下收益为正",
    "trade_sample_sufficient": "闭合交易样本充足",
    "profit_not_over_concentrated": "盈利不过度集中",
}
_GATE_STATUS_LABELS = {
    "research_only": "仅限研究与模拟验证",
    "eligible_for_manual_review": "可提交人工审核",
}
_ROBUSTNESS_LABELS = {"strong": "强", "moderate": "中", "weak": "弱", "unknown": "未知"}
_PARAMETER_STATUS_LABELS = {
    "stable": "稳定",
    "sensitive": "敏感",
    "unstable": "不稳定",
    "not_applicable": "不适用",
    "unknown": "未知",
}
_VALIDATION_STATUS_LABELS = {
    "verified": "已核验",
    "degraded": "降级核验",
    "unverified": "未核验",
    "conflict": "来源冲突",
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows({field: row.get(field) for field in fields} for row in rows)
    return buffer.getvalue()


def _comparison_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    fields = (
        "strategy_name",
        "display_name",
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
    )
    return [{field: row.get(field) for field in fields} for row in payload.get("comparisons", [])]


def _daily_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for strategy in payload.get("comparisons", []):
        name = str(strategy.get("strategy_name", ""))
        signals = {str(item.get("date")): item for item in strategy.get("signal_curve", [])}
        for point in strategy.get("equity_curve", []):
            day = str(point.get("date"))
            signal = signals.get(day, {})
            row = rows.setdefault(day, {"date": day})
            row[f"{name}_value"] = point.get("value")
            row[f"{name}_target_position"] = signal.get("target_position")
            row[f"{name}_target_exposure"] = signal.get("target_exposure")
            row[f"{name}_actual_exposure"] = signal.get("actual_exposure", point.get("exposure"))
    return [rows[key] for key in sorted(rows)]


def _trade_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for strategy in payload.get("comparisons", []):
        for trade in strategy.get("trades", []):
            output.append({"strategy_name": strategy.get("strategy_name"), **trade})
    return output


def _validation_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    report = payload.get("data_validation") or {}
    rows = []
    for candidate in report.get("candidates", []):
        rows.append({"record_type": "candidate", **candidate, "issues": "；".join(candidate.get("issues", []))})
    for item in report.get("differences", []):
        rows.append({"record_type": "difference", **item})
    return rows


def _regime_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "strategy_name": strategy.get("strategy_name"),
            "display_name": strategy.get("display_name"),
            **regime,
        }
        for strategy in (payload.get("market_regime_attribution", {}).get("strategies") or [])
        if isinstance(strategy, dict)
        for regime in (strategy.get("regimes") or [])
        if isinstance(regime, dict)
    ]


def _trade_attribution_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    excluded = {"matched_trades", "best_trades", "worst_trades"}
    return [
        {key: value for key, value in item.items() if key not in excluded}
        for item in (payload.get("trade_attribution", {}).get("strategies") or [])
        if isinstance(item, dict)
    ]


def _matched_trade_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "strategy_name": item.get("strategy_name"),
            "display_name": item.get("display_name"),
            **trade,
        }
        for item in (payload.get("trade_attribution", {}).get("strategies") or [])
        if isinstance(item, dict)
        for trade in (item.get("matched_trades") or [])
        if isinstance(trade, dict)
    ]


def _robustness_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in payload.get("robustness_assessments", []) if isinstance(item, dict)]


def _experiment_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in (payload.get("research_decision", {}).get("next_experiments") or [])
        if isinstance(item, dict)
    ]


def _winner_label(item: dict[str, Any] | None) -> str:
    if not item:
        return "该维度缺少可比数据"
    metric = str(item.get("metric") or "")
    if "return" in metric:
        value = _percent(item.get("value"))
    elif "drawdown" in metric:
        value = _percent(item.get("value"), signed=False)
    else:
        value = _number(item.get("value"))
    return f"{item.get('display_name') or item.get('strategy_name')}（{value}）"


def _market_comparison_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in row.items()
            if key not in {"asset_equity_curve", "market_equity_curve"}
        }
        for row in ((payload.get("market_benchmark") or {}).get("comparisons") or [])
        if isinstance(row, dict)
    ]


def _percent(value: Any, *, signed: bool = True) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:+.2%}" if signed else f"{number:.2%}"


def _number(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _money(value: Any) -> str:
    try:
        return f"¥{float(value):,.2f}"
    except (TypeError, ValueError):
        return "—"


def _chart_svg(payload: dict[str, Any], *, drawdown: bool = False) -> str:
    by_name = {
        str(item.get("strategy_name")): item
        for item in payload.get("comparisons", [])
        if isinstance(item, dict)
    }
    ordered_names = [str(payload.get("benchmark") or ""), *map(str, payload.get("ranking") or [])]
    selected = []
    for name in ordered_names:
        if name in by_name and by_name[name] not in selected:
            selected.append(by_name[name])
        if len(selected) == 4:
            break
    series = []
    for item in selected:
        curve = item.get("drawdown_curve" if drawdown else "equity_curve") or []
        values = [float(point.get("value")) for point in curve if point.get("value") is not None]
        if not values:
            continue
        if not drawdown:
            first = values[0]
            values = [value / first for value in values] if first else values
        step = max(1, len(values) // 180)
        sampled = values[::step]
        if sampled[-1] != values[-1]:
            sampled.append(values[-1])
        series.append((str(item.get("display_name") or item.get("strategy_name")), sampled))
    if not series:
        return '<div class="empty">没有可绘制的数据</div>'
    width, height = 1040, 320
    left, right, top, bottom = 72, 24, 24, 54
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_values = [value for _, values in series for value in values]
    low, high = min(all_values), max(all_values)
    if high == low:
        high = low + 1
    padding = (high - low) * 0.08
    low -= padding
    high += padding
    colors = ("#2563eb", "#7c3aed", "#0f9f6e", "#e8790c")
    grid = []
    for index in range(5):
        ratio = index / 4
        y = top + plot_height * ratio
        value = high - (high - low) * ratio
        label = f"{value:.1%}" if drawdown else f"{value:.2f}"
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#dbe5f2"/>'
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" fill="#64748b" font-size="12">{label}</text>'
        )
    paths = []
    legends = []
    for index, (name, values) in enumerate(series):
        points = []
        denominator = max(len(values) - 1, 1)
        for point_index, value in enumerate(values):
            x = left + plot_width * point_index / denominator
            y = top + plot_height * (high - value) / (high - low)
            points.append(f"{x:.1f},{y:.1f}")
        color = colors[index % len(colors)]
        paths.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.4"/>'
        )
        legend_x = left + index * 235
        legends.append(
            f'<line x1="{legend_x}" y1="{height - 20}" x2="{legend_x + 24}" y2="{height - 20}" '
            f'stroke="{color}" stroke-width="3"/><text x="{legend_x + 31}" y="{height - 16}" '
            f'fill="#334155" font-size="12">{html.escape(name[:24])}</text>'
        )
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{"回撤曲线" if drawdown else "归一化净值曲线"}">'
        + "".join(grid)
        + "".join(paths)
        + "".join(legends)
        + "</svg>"
    )


def _html_report(payload: dict[str, Any]) -> str:
    conclusion = payload.get("conclusion") or {}
    winners = [
        ("绝对收益", conclusion.get("absolute_return_winner")),
        ("风险调整", conclusion.get("risk_adjusted_winner")),
        ("回撤控制", conclusion.get("drawdown_winner")),
        ("样本外", conclusion.get("out_of_sample_winner")),
        ("稳健性", conclusion.get("robustness_winner")),
    ]
    headers = ["排名", "策略", "总收益", "年化", "最大回撤", "夏普", "Calmar", "样本外", "换手", "费用"]
    ranking = {str(name): index for index, name in enumerate(payload.get("ranking") or [], 1)}
    table_rows = []
    for row in payload.get("comparisons", []):
        oos = row.get("diagnostics", {}).get("out_of_sample", {}).get("out_of_sample_return")
        values = [
            ranking.get(str(row.get("strategy_name")), "—"),
            row.get("display_name") or row.get("strategy_name"),
            _percent(row.get("total_return")),
            _percent(row.get("annualized_return")),
            _percent(row.get("max_drawdown"), signed=False),
            _number(row.get("sharpe_ratio")),
            _number(row.get("calmar_ratio")),
            _percent(oos),
            _number(row.get("turnover"), 2),
            _money(row.get("total_fees")),
        ]
        table_rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in values) + "</tr>")
    winner_cards = "".join(
        f"<section><strong>{html.escape(label)}</strong><p>{html.escape(_winner_label(value))}</p></section>"
        for label, value in winners
    )
    warnings = "".join(f"<li>{html.escape(str(item))}</li>" for item in conclusion.get("data_warnings", []))
    limitations = "".join(f"<li>{html.escape(str(item))}</li>" for item in conclusion.get("limitations", []))
    recommendations = "".join(
        f"<li>{html.escape(str(item))}</li>" for item in conclusion.get("recommendations", [])
    ) or "<li>当前没有生成 Agent 建议，请结合逐策略诊断判断。</li>"
    assessment_cards = []
    for item in payload.get("strategy_assessments", []):
        strengths = "".join(f"<li>{html.escape(str(value))}</li>" for value in item.get("strengths", []))
        weaknesses = "".join(f"<li>{html.escape(str(value))}</li>" for value in item.get("weaknesses", []))
        assessment_cards.append(
            f"<article class=\"diagnosis\"><div class=\"diagnosis-head\"><span class=\"rank\">"
            f"#{html.escape(str(item.get('rank') or '—'))}</span><h3>"
            f"{html.escape(str(item.get('display_name') or item.get('strategy_name')))}</h3>"
            f"<span class=\"verdict\">{html.escape(str(item.get('verdict') or '待判断'))}</span></div>"
            f"<p><strong>怎么交易：</strong>{html.escape(str(item.get('mechanism') or '未记录'))}</p>"
            f"<div class=\"two-cols\"><div><h4>本期为什么好</h4><ul>{strengths}</ul></div>"
            f"<div><h4>本期为什么不好</h4><ul>{weaknesses}</ul></div></div>"
            f"<p><strong>适合：</strong>{html.escape(str(item.get('suitable_market') or '—'))}</p>"
            f"<p><strong>容易失效：</strong>{html.escape(str(item.get('failure_mode') or '—'))}</p></article>"
        )
    ticker = html.escape(str(payload.get("ticker")))
    header_html = "".join(f"<th>{item}</th>" for item in headers)
    market_benchmark = payload.get("market_benchmark") or {}
    market_headers = ["同一策略", "当前标的收益", "大盘收益", "同策略超额", "标的回撤", "大盘回撤"]
    market_table_rows = []
    for row in _market_comparison_rows(payload):
        values = [
            row.get("display_name") or row.get("strategy_name"),
            _percent(row.get("asset_total_return")),
            _percent(row.get("market_total_return")),
            _percent(row.get("excess_return")),
            _percent(row.get("asset_max_drawdown"), signed=False),
            _percent(row.get("market_max_drawdown"), signed=False),
        ]
        market_table_rows.append(
            "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in values) + "</tr>"
        )
    market_html = ""
    if market_benchmark:
        market_header_html = "".join(f"<th>{item}</th>" for item in market_headers)
        market_note = html.escape(
            str(market_benchmark.get("simulation_note") or market_benchmark.get("error") or "")
        )
        market_html = (
            f"<h2>同一策略：当前标的 vs "
            f"{html.escape(str(market_benchmark.get('name') or market_benchmark.get('ticker')))}</h2>"
            f"<p class=\"meta\">{market_note}</p>"
            f"<table><thead><tr>{market_header_html}</tr></thead>"
            f"<tbody>{''.join(market_table_rows)}</tbody></table>"
        )
    display_by_name = {
        str(row.get("strategy_name")): str(row.get("display_name") or row.get("strategy_name"))
        for row in payload.get("comparisons", [])
    }
    scenario_labels = {"low": "低成本", "base": "基准", "stress": "压力"}
    cost_rows = []
    for scenario, rows in (payload.get("cost_scenarios") or {}).items():
        for row in rows:
            strategy_name = str(row.get("strategy_name"))
            cost_rows.append(
                "<tr>"
                f"<td>{html.escape(scenario_labels.get(str(scenario), str(scenario)))}</td>"
                f"<td>{html.escape(display_by_name.get(strategy_name, strategy_name))}</td>"
                f"<td>{_percent(row.get('total_return'))}</td>"
                f"<td>{_percent(row.get('max_drawdown'), signed=False)}</td>"
                f"<td>{_money(row.get('total_fees'))}</td></tr>"
            )
    robustness_rows = []
    for row in _robustness_rows(payload):
        grade = {"strong": "强", "moderate": "中", "weak": "弱"}.get(
            str(row.get("grade")), str(row.get("grade") or "—")
        )
        robustness_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('display_name') or row.get('strategy_name')))}</td>"
            f"<td>{grade}</td><td>{_percent(row.get('out_of_sample_return'))}</td>"
            f"<td>{_percent(row.get('rolling_positive_ratio'))}</td>"
            f"<td>{_percent(row.get('rolling_worst_return'))}</td>"
            f"<td>{html.escape(_PARAMETER_STATUS_LABELS.get(str(row.get('parameter_status')), '未知'))}</td>"
            f"<td>{_percent(row.get('stress_total_return'))}</td>"
            f"<td>{_percent(row.get('cost_degradation'), signed=False)}</td></tr>"
        )
    sensitivity = payload.get("parameter_sensitivity") or {}
    parameter_rows = []
    for strategy_name, result in sensitivity.items():
        for variant in (result or {}).get("variants", []):
            value = variant.get("total_return")
            css_class = "positive" if value is not None and float(value) >= 0 else "negative"
            parameter_rows.append(
                "<tr>"
                f"<td>{html.escape(display_by_name.get(str(strategy_name), str(strategy_name)))}</td>"
                f"<td>{html.escape(str(variant.get('variant') or '—'))}</td>"
                f"<td class=\"{css_class}\">{_percent(value)}</td>"
                f"<td>{_number(variant.get('sharpe_ratio'))}</td></tr>"
            )
    regime_rows = []
    for row in _regime_rows(payload):
        regime_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('display_name') or row.get('strategy_name')))}</td>"
            f"<td>{html.escape(str(row.get('label') or row.get('regime')))}</td>"
            f"<td>{row.get('days', 0)}</td><td>{_percent(row.get('strategy_return'))}</td>"
            f"<td>{_percent(row.get('benchmark_return'))}</td><td>{_percent(row.get('excess_return'))}</td>"
            f"<td>{_percent(row.get('average_exposure'))}</td><td>{_percent(row.get('worst_day'))}</td></tr>"
        )
    trade_summary_rows = []
    for row in _trade_attribution_rows(payload):
        trade_summary_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('display_name') or row.get('strategy_name')))}</td>"
            f"<td>{row.get('closed_trade_segments', 0)}</td><td>{_money(row.get('realized_pnl'))}</td>"
            f"<td>{_percent(row.get('win_rate'))}</td><td>{_number(row.get('payoff_ratio'), 2)}</td>"
            f"<td>{_number(row.get('average_holding_days'), 1)}</td>"
            f"<td>{row.get('max_consecutive_losses', 0)}</td>"
            f"<td>{_percent(row.get('top3_profit_concentration'))}</td>"
            f"<td>{row.get('open_shares', 0)}</td></tr>"
        )
    matched_trade_rows = []
    for row in _matched_trade_rows(payload):
        matched_trade_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('display_name') or row.get('strategy_name')))}</td>"
            f"<td>{html.escape(str(row.get('buy_date') or '—'))}</td>"
            f"<td>{html.escape(str(row.get('sell_date') or '—'))}</td>"
            f"<td>{row.get('shares', 0)}</td><td>{row.get('holding_days', 0)}</td>"
            f"<td>{_money(row.get('pnl'))}</td><td>{_percent(row.get('return_pct'))}</td></tr>"
        )
    validation = payload.get("data_validation") or {}
    source_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item.get('source_name') or item.get('source_id')))}</td>"
        f"<td>{'是' if item.get('selected') else '否'}</td>"
        f"<td>{_number(item.get('quality_score'), 1)}</td>"
        f"<td>{html.escape(str(item.get('actual_start_date') or '—'))} 至 "
        f"{html.escape(str(item.get('actual_end_date') or '—'))}</td>"
        f"<td><code>{html.escape(str(item.get('sha256') or '')[:16])}</code></td></tr>"
        for item in validation.get("candidates", [])
    )
    execution_rows = "".join(
        f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>"
        for key, value in (payload.get("execution") or {}).items()
    )
    acceptance_rows = "".join(
        f"<tr><td>{html.escape(str(key))}</td><td>{'通过' if value else '未通过'}</td></tr>"
        for key, value in (payload.get("acceptance", {}).get("checks") or {}).items()
    )
    decision = payload.get("research_decision") or {}
    falsification_html = "".join(
        f"<li>{html.escape(str(item))}</li>" for item in decision.get("falsification_risks", [])
    ) or "<li>尚未生成可证伪风险。</li>"
    experiment_html = "".join(
        "<tr>"
        f"<td>{html.escape(str(item.get('question') or '—'))}</td>"
        f"<td>{html.escape(str(item.get('method') or '—'))}</td>"
        f"<td>{html.escape(str(item.get('success_criteria') or '—'))}</td></tr>"
        for item in _experiment_rows(payload)
    )
    gate = decision.get("deployment_gate") or {}
    gate_rows = "".join(
        f"<tr><td>{html.escape(_GATE_LABELS.get(str(key), str(key)))}</td>"
        f"<td>{'通过' if value else '未通过'}</td></tr>"
        for key, value in (gate.get("checks") or {}).items()
    )
    gate_status = _GATE_STATUS_LABELS.get(str(gate.get("status")), str(gate.get("status") or "仅限研究"))
    robustness_label = _ROBUSTNESS_LABELS.get(
        str(decision.get("robustness_grade")), str(decision.get("robustness_grade") or "未知")
    )
    validation_status = _VALIDATION_STATUS_LABELS.get(
        str(payload.get("data_validation", {}).get("status")),
        str(payload.get("data_validation", {}).get("status") or "未知"),
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{ticker} 多策略回测</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:Inter,"PingFang SC",system-ui;background:#edf3fa;color:#172033;margin:0;line-height:1.65}}
main{{max-width:1240px;margin:28px auto;background:white;padding:42px;border-radius:22px;
box-shadow:0 20px 60px #1e3a5f18}}
h1{{font-size:32px;margin:0 0 8px}}h2{{margin:42px 0 14px;padding-bottom:8px;border-bottom:2px solid #d9e7f5}}
h3,h4{{margin:0}}.meta{{color:#64748b}}.lead{{font-size:17px;color:#334155}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}}
section,.panel,.diagnosis{{border:1px solid #dbe5f2;border-radius:14px;padding:16px;background:#fbfdff}}
.advice{{background:#eef6ff;border-left:5px solid #2563eb;padding:18px 22px;border-radius:12px}}
.chart{{width:100%;height:auto;background:#fbfdff;border:1px solid #dbe5f2;border-radius:14px}}
.table-wrap{{overflow:auto;border:1px solid #dbe5f2;border-radius:12px}}
table{{width:100%;border-collapse:collapse;min-width:820px}}th{{background:#eef4fb;color:#334155;position:sticky;top:0}}
th,td{{padding:10px 12px;border-bottom:1px solid #e5edf6;text-align:right;white-space:nowrap}}
th:nth-child(2),td:nth-child(2),th:first-child,td:first-child{{text-align:left}}
.diagnoses{{display:grid;gap:16px}}.diagnosis-head{{display:flex;gap:12px;align-items:center;margin-bottom:8px}}
.rank{{font-weight:700;color:#2563eb}}
.verdict{{margin-left:auto;background:#e7f0ff;color:#174ea6;padding:3px 10px;border-radius:999px}}
.two-cols{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px}}
.two-cols>*{{min-width:0}}.two-cols h4{{color:#334155}}code{{font-size:12px}}
.positive{{color:#087f5b;font-weight:700}}.negative{{color:#c2413b;font-weight:700}}
.narrative-table td{{white-space:normal;min-width:240px;vertical-align:top}}
.gate-status{{font-size:18px;font-weight:700;color:#174ea6}}
details{{margin-top:14px}}summary{{cursor:pointer;font-weight:700;color:#1d4ed8}}.empty{{padding:24px;color:#64748b}}
@media(max-width:700px){{main{{margin:0;padding:20px;border-radius:0}}.two-cols{{grid-template-columns:1fr}}}}
@media print{{body{{background:white}}main{{box-shadow:none;margin:0;max-width:none}}details{{display:block}}}}
</style></head><body><main>
<h1>{ticker} 多策略量化回测</h1>
<p class="meta">评价期 {payload.get('evaluation_start_date')} 至 {payload.get('evaluation_end_date')} ·
热身 {payload.get('warmup_bars')} 个交易日 · {payload.get('strategy_count')} 个策略 ·
数据核验 {validation_status} ·
数据源 {payload.get('data_validation', {}).get('selected_source')}</p>
<p class="lead">本报告先说明策略如何交易及本期为什么表现好或不好，再给出净值、回撤、样本外、成本压力和数据审计证据。</p>
<h2>Agent 建议</h2><div class="advice"><ul>{recommendations}</ul></div>
<h2>研究结论与下一步</h2>
<p class="gate-status">当前首选：{html.escape(str(decision.get('preferred_display_name') or '—'))} ·
稳健性：{html.escape(robustness_label)} ·
准入：{html.escape(gate_status)}</p>
<p>{html.escape(str(gate.get('message') or '当前仅限研究与模拟验证。'))}</p>
<div class="two-cols"><div class="panel"><h3>最可能推翻当前建议的证据</h3><ul>{falsification_html}</ul></div>
<div class="panel"><h3>部署准入检查</h3><div class="table-wrap"><table><tbody>
{gate_rows}</tbody></table></div></div></div>
<h3>下一轮实验</h3><div class="table-wrap"><table class="narrative-table"><thead><tr>
<th>要回答什么</th><th>怎么验证</th><th>通过标准</th></tr></thead><tbody>{experiment_html}</tbody></table></div>
<h2>分维度领先策略</h2>
<div class="cards">{winner_cards}</div>
<h2>净值与回撤路径</h2>
<h3>归一化净值</h3>{_chart_svg(payload)}
<h3>回撤</h3>{_chart_svg(payload, drawdown=True)}
<h2>完整绩效对比</h2><div class="table-wrap"><table><thead><tr>{header_html}</tr></thead>
<tbody>{''.join(table_rows)}</tbody></table></div>
<h2>逐策略诊断</h2><div class="diagnoses">
{''.join(assessment_cards) or '<div class="empty">暂无逐策略诊断</div>'}</div>
<h2>市场阶段归因</h2><p class="meta">上涨、下跌、横盘为互斥阶段；高波动为可与三类趋势阶段重叠的风险切片。</p>
<div class="table-wrap"><table><thead><tr><th>策略</th><th>市场阶段</th><th>天数</th><th>策略收益</th>
<th>标的收益</th><th>相对超额</th><th>平均仓位</th><th>最差单日</th></tr></thead>
<tbody>{''.join(regime_rows)}</tbody></table></div>
<h2>交易级归因</h2><p class="meta">按 FIFO 配对实际成交，计入成交记录中的佣金、税费和过户费。</p>
<div class="table-wrap"><table><thead><tr><th>策略</th><th>闭合交易段</th><th>已实现盈亏</th><th>胜率</th>
<th>平均盈亏比</th><th>平均持有天数</th><th>最长连亏</th><th>前三笔盈利占比</th><th>未闭合股数</th></tr></thead>
<tbody>{''.join(trade_summary_rows)}</tbody></table></div>
<details><summary>查看所有 FIFO 闭合交易段</summary><div class="table-wrap"><table><thead><tr>
<th>策略</th><th>买入日</th><th>卖出日</th><th>数量</th><th>持有天数</th><th>盈亏</th><th>收益率</th>
</tr></thead><tbody>{''.join(matched_trade_rows)}</tbody></table></div></details>
<h2>稳健性证据</h2><div class="table-wrap"><table><thead><tr><th>策略</th><th>等级</th>
<th>样本外收益</th><th>滚动正收益率</th><th>最差滚动收益</th><th>参数敏感性</th><th>压力成本收益</th>
<th>成本拖累</th></tr></thead><tbody>{''.join(robustness_rows)}</tbody></table></div>
<details open><summary>相邻参数结果</summary><div class="table-wrap"><table><thead><tr>
<th>策略</th><th>参数版本</th><th>总收益</th><th>夏普</th></tr></thead><tbody>{''.join(parameter_rows)}</tbody>
</table></div></details>
<h2>交易成本压力测试</h2><details open><summary>展开低成本、基准和压力情景</summary>
<div class="table-wrap"><table><thead><tr><th>情景</th><th>策略</th><th>收益</th><th>回撤</th>
<th>费用</th></tr></thead><tbody>{''.join(cost_rows)}</tbody></table></div></details>
{market_html}
<h2>数据与执行审计</h2>
<p class="meta">选择理由：{html.escape(str(validation.get('selection_reason') or '未记录'))} ·
数据快照 SHA-256：<code>
{html.escape(str(payload.get('data_snapshot', {}).get('sha256') or '未记录'))}</code></p>
<div class="table-wrap"><table><thead><tr><th>数据源</th><th>选中</th><th>质量分</th>
<th>实际区间</th><th>哈希</th></tr></thead><tbody>{source_rows}</tbody></table></div>
<div class="two-cols"><div><h3>成交与费用假设</h3><div class="table-wrap"><table>
<tbody>{execution_rows}</tbody></table></div></div>
<div><h3>验收检查</h3><div class="table-wrap"><table><tbody>{acceptance_rows}</tbody></table></div></div></div>
<h2>数据警告与局限</h2><div class="two-cols"><div class="panel"><h3>数据警告</h3>
<ul>{warnings or '<li>未发现额外数据警告</li>'}</ul></div>
<div class="panel"><h3>模型局限</h3><ul>{limitations}</ul></div></div>
<p class="meta">以上结论仅适用于本报告记录的固定数据、参数、评价区间和成交假设，用于短中期研究与模拟盘。</p>
</main></body></html>"""


def _write_table(
    workbook: Any,
    name: str,
    rows: list[dict[str, Any]],
    *,
    widths: dict[str, int] | None = None,
) -> Any:
    sheet = workbook.add_worksheet(name[:31])
    sheet.freeze_panes(1, 1)
    if not rows:
        sheet.write(0, 0, "暂无数据")
        return sheet
    headers = list(dict.fromkeys(key for row in rows for key in row))
    header_format = workbook.add_format({"bold": True, "bg_color": "#DCE8F8", "font_color": "#15345E"})
    sheet.write_row(0, 0, headers, header_format)
    for row_index, row in enumerate(rows, 1):
        for column_index, key in enumerate(headers):
            value = row.get(key)
            if isinstance(value, (list, dict)):
                value = _json(value)
            if key == "formula_check" and isinstance(value, str) and value.startswith("="):
                sheet.write_formula(row_index, column_index, value)
            else:
                sheet.write(row_index, column_index, value)
    sheet.autofilter(0, 0, len(rows), len(headers) - 1)
    for index, header in enumerate(headers):
        width = (widths or {}).get(header, min(max(len(header) + 2, 12), 28))
        sheet.set_column(index, index, width)
    return sheet


def _workbook(payload: dict[str, Any], selected_rows: list[dict[str, Any]]) -> bytes:
    output = io.BytesIO()
    workbook = xlsxwriter.Workbook(output, {"in_memory": True, "strings_to_formulas": False})
    summary = workbook.add_worksheet("结果总览")
    title = workbook.add_format({"bold": True, "font_size": 20, "font_color": "#15345E"})
    label = workbook.add_format({"bold": True, "bg_color": "#EAF1FB"})
    summary.set_column("A:A", 20)
    summary.set_column("B:B", 42)
    summary.write("A1", f"{payload.get('ticker')} 多策略量化回测", title)
    meta = [
        ("评价期", f"{payload.get('evaluation_start_date')} 至 {payload.get('evaluation_end_date')}"),
        ("热身交易日", payload.get("warmup_bars")),
        (
            "数据核验",
            _VALIDATION_STATUS_LABELS.get(
                str(payload.get("data_validation", {}).get("status")),
                payload.get("data_validation", {}).get("status"),
            ),
        ),
        ("选定来源", payload.get("data_validation", {}).get("selected_source")),
        ("策略数量", payload.get("strategy_count")),
        ("验收结果", payload.get("acceptance", {}).get("satisfied")),
    ]
    for row, (key, value) in enumerate(meta, 2):
        summary.write(row, 0, key, label)
        summary.write(row, 1, value)
    conclusion = payload.get("conclusion") or {}
    for row, (key, field) in enumerate(
        (
            ("绝对收益最高", "absolute_return_winner"),
            ("风险调整最佳", "risk_adjusted_winner"),
            ("回撤控制最佳", "drawdown_winner"),
            ("样本外最佳", "out_of_sample_winner"),
            ("稳健性最佳", "robustness_winner"),
        ),
        9,
    ):
        summary.write(row, 0, key, label)
        summary.write(row, 1, _winner_label(conclusion.get(field)))
    recommendation_row = 15
    summary.write(recommendation_row, 0, "Agent 建议", label)
    recommendations = conclusion.get("recommendations") or []
    summary.write(recommendation_row, 1, "；".join(str(item) for item in recommendations) or "未生成")
    decision = payload.get("research_decision") or {}
    gate = decision.get("deployment_gate") or {}
    decision_meta = [
        ("研究首选", decision.get("preferred_display_name") or "—"),
        (
            "稳健性等级",
            _ROBUSTNESS_LABELS.get(
                str(decision.get("robustness_grade")), decision.get("robustness_grade") or "未知"
            ),
        ),
        ("部署准入", _GATE_STATUS_LABELS.get(str(gate.get("status")), gate.get("status") or "仅限研究")),
        ("准入说明", gate.get("message") or "当前仅限研究与模拟验证"),
        ("可证伪风险", "；".join(str(item) for item in decision.get("falsification_risks") or []) or "未生成"),
    ]
    for row, (key, value) in enumerate(decision_meta, recommendation_row + 1):
        summary.write(row, 0, key, label)
        summary.write(row, 1, value)

    execution = [{"key": key, "value": value} for key, value in (payload.get("execution") or {}).items()]
    _write_table(workbook, "假设与成交规则", execution)
    _write_table(workbook, "数据源与交叉核验", _validation_rows(payload))
    _write_table(
        workbook,
        "策略定义",
        [
            {
                "strategy_name": row.get("strategy_name"),
                "description": row.get("description"),
                "entry_rules": row.get("entry_rules"),
                "exit_rules": row.get("exit_rules"),
                "position_policy": row.get("strategy_spec", {}).get("position_policy"),
            }
            for row in payload.get("comparisons", [])
        ],
        widths={"description": 45, "entry_rules": 50, "exit_rules": 50, "position_policy": 50},
    )
    _write_table(
        workbook,
        "策略诊断",
        [
            {
                "rank": item.get("rank"),
                "strategy_name": item.get("strategy_name"),
                "display_name": item.get("display_name"),
                "mechanism": item.get("mechanism"),
                "why_good": item.get("why_good"),
                "why_bad": item.get("why_bad"),
                "suitable_market": item.get("suitable_market"),
                "failure_mode": item.get("failure_mode"),
                "verdict": item.get("verdict"),
            }
            for item in payload.get("strategy_assessments", [])
        ],
        widths={
            "mechanism": 48,
            "why_good": 60,
            "why_bad": 60,
            "suitable_market": 48,
            "failure_mode": 48,
        },
    )
    _write_table(workbook, "绩效对比", _comparison_rows(payload))
    _write_table(workbook, "市场阶段归因", _regime_rows(payload))
    _write_table(workbook, "交易归因", _trade_attribution_rows(payload))
    _write_table(workbook, "FIFO闭合交易", _matched_trade_rows(payload))
    _write_table(workbook, "稳健性评估", _robustness_rows(payload))
    _write_table(
        workbook,
        "下一轮实验",
        _experiment_rows(payload),
        widths={"question": 42, "method": 62, "success_criteria": 62},
    )
    _write_table(
        workbook,
        "部署准入",
        [
            {"check": _GATE_LABELS.get(str(key), str(key)), "passed": value}
            for key, value in (gate.get("checks") or {}).items()
        ],
    )
    _write_table(workbook, "同期大盘同策略", _market_comparison_rows(payload))
    cost_rows = [
        {"scenario": scenario, **row}
        for scenario, rows in (payload.get("cost_scenarios") or {}).items()
        for row in rows
    ]
    _write_table(workbook, "成本情景", cost_rows)
    diagnostic_rows = []
    for row in payload.get("comparisons", []):
        diagnostic_rows.append(
            {
                "strategy_name": row.get("strategy_name"),
                **row.get("diagnostics", {}).get("out_of_sample", {}),
                "parameter_sensitivity": payload.get("parameter_sensitivity", {}).get(row.get("strategy_name")),
            }
        )
    _write_table(workbook, "样本外与参数敏感性", diagnostic_rows, widths={"parameter_sensitivity": 80})
    parameter_variant_rows = [
        {
            "strategy_name": strategy_name,
            "status": (result or {}).get("status"),
            "variant": variant.get("variant"),
            "total_return": variant.get("total_return"),
            "sharpe_ratio": variant.get("sharpe_ratio"),
        }
        for strategy_name, result in (payload.get("parameter_sensitivity") or {}).items()
        for variant in (result or {}).get("variants", [])
    ]
    _write_table(workbook, "参数敏感性明细", parameter_variant_rows)
    daily = _daily_rows(payload)
    daily_sheet = _write_table(workbook, "每日净值", daily)
    if daily:
        chart = workbook.add_chart({"type": "line"})
        value_columns = [index for index, key in enumerate(daily[0]) if key.endswith("_value")][:4]
        for column in value_columns:
            chart.add_series(
                {
                    "name": ["每日净值", 0, column],
                    "categories": ["每日净值", 1, 0, len(daily), 0],
                    "values": ["每日净值", 1, column, len(daily), column],
                    "line": {"width": 1.5},
                }
            )
        chart.set_title({"name": "策略净值对比（引擎预计算）"})
        chart.set_legend({"position": "bottom"})
        daily_sheet.insert_chart("S2", chart, {"x_scale": 1.4, "y_scale": 1.2})
    _write_table(workbook, "信号与仓位", daily)
    _write_table(workbook, "成交记录", _trade_rows(payload))
    checks = [
        {"check": key, "engine_value": value, "formula_check": value}
        for key, value in (payload.get("acceptance", {}).get("checks") or {}).items()
    ]
    comparison_count = len(payload.get("comparisons", []))
    fees = sum(float(row.get("total_fees") or 0) for row in payload.get("comparisons", []))
    checks.extend(
        [
            {
                "check": "策略行数核对",
                "engine_value": comparison_count,
                "formula_check": f"=COUNTA('绩效对比'!A2:A{comparison_count + 1})={comparison_count}",
            },
            {
                "check": "每日净值行数核对",
                "engine_value": len(daily),
                "formula_check": f"=COUNTA('每日净值'!A2:A{len(daily) + 1})={len(daily)}",
            },
            {
                "check": "行情行数核对",
                "engine_value": len(selected_rows),
                "formula_check": (
                    f"=COUNTA('选定行情数据'!A2:A{len(selected_rows) + 1})={len(selected_rows)}"
                ),
            },
            {
                "check": "费用合计核对",
                "engine_value": fees,
                "formula_check": f"=ABS(SUM('绩效对比'!N2:N{comparison_count + 1})-{fees})<0.01",
            },
        ]
    )
    for index, row in enumerate(payload.get("comparisons", [])):
        excel_row = index + 2
        value_column = xl_col_to_name(1 + 4 * index)
        checks.append(
            {
                "check": f"{row.get('strategy_name')} 期末净值核对",
                "engine_value": row.get("final_value"),
                "formula_check": (
                    f"=ABS('绩效对比'!P{excel_row}-LOOKUP(2,1/('每日净值'!{value_column}:"
                    f"{value_column}<>\"\"),'每日净值'!{value_column}:{value_column}))<0.01"
                ),
            }
        )
        checks.append(
            {
                "check": f"{row.get('strategy_name')} 收益恒等式",
                "engine_value": row.get("total_return"),
                "formula_check": (
                    f"=ABS('绩效对比'!C{excel_row}-"
                    f"('绩效对比'!P{excel_row}/{float(payload.get('initial_capital') or 1)}-1))<0.000001"
                ),
            }
        )
    _write_table(workbook, "模型检查", checks)
    _write_table(workbook, "选定行情数据", selected_rows)
    workbook.close()
    return output.getvalue()


async def create_comparison_artifacts(
    payload: dict[str, Any],
    *,
    selected_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    comparison_id = str(payload["comparison_id"])
    daily = _daily_rows(payload)
    trades = _trade_rows(payload)
    validation = _validation_rows(payload)
    replay_payload = {key: value for key, value in payload.items() if key != "artifacts"}
    workbook = _workbook(replay_payload, selected_rows)
    artifacts = [
        {"name": f"{comparison_id}-回测报告.html", "format": "html", "content": _html_report(payload)},
        {
            "name": f"{comparison_id}-审计工作簿.xlsx",
            "format": "xlsx",
            "content_base64": base64.b64encode(workbook).decode(),
        },
        {"name": f"{comparison_id}-完整结果.json", "format": "json", "content": _json(replay_payload)},
        {"name": f"{comparison_id}-选定行情.csv", "format": "csv", "content": _csv(selected_rows)},
        {"name": f"{comparison_id}-每日策略.csv", "format": "csv", "content": _csv(daily)},
        {"name": f"{comparison_id}-成交记录.csv", "format": "csv", "content": _csv(trades) or "strategy_name\n"},
        {"name": f"{comparison_id}-数据核验.csv", "format": "csv", "content": _csv(validation)},
    ]
    return await artifact_service.create_user_artifacts(
        artifacts,
        source="strategy_comparison",
        task_id=comparison_id,
        ticker=str(payload.get("ticker")),
        asset_type=str(payload.get("asset_type")),
        metadata={"comparison_id": comparison_id, "generated_by": "strategy_comparison"},
        execution_key=comparison_id,
    )
