"""Auditable deliverables for formal multi-strategy comparisons."""

from __future__ import annotations

import base64
import csv
import html
import io
import json
from typing import Any

from artifacts.service import artifact_service


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


def _winner_label(item: dict[str, Any] | None) -> str:
    if not item:
        return "暂无正式结论"
    return f"{item.get('display_name') or item.get('strategy_name')}（{item.get('value')}）"


def _html_report(payload: dict[str, Any]) -> str:
    conclusion = payload.get("conclusion") or {}
    winners = [
        ("绝对收益", conclusion.get("absolute_return_winner")),
        ("风险调整", conclusion.get("risk_adjusted_winner")),
        ("回撤控制", conclusion.get("drawdown_winner")),
        ("样本外", conclusion.get("out_of_sample_winner")),
        ("稳健性", conclusion.get("robustness_winner")),
    ]
    headers = ["策略", "总收益", "年化", "最大回撤", "夏普", "Calmar", "样本外", "费用"]
    table_rows = []
    for row in payload.get("comparisons", []):
        oos = row.get("diagnostics", {}).get("out_of_sample", {}).get("out_of_sample_return")
        values = [
            row.get("display_name") or row.get("strategy_name"),
            row.get("total_return"),
            row.get("annualized_return"),
            row.get("max_drawdown"),
            row.get("sharpe_ratio"),
            row.get("calmar_ratio"),
            oos,
            row.get("total_fees"),
        ]
        table_rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in values) + "</tr>")
    winner_cards = "".join(
        f"<section><strong>{html.escape(label)}</strong><p>{html.escape(_winner_label(value))}</p></section>"
        for label, value in winners
    )
    warnings = "".join(f"<li>{html.escape(str(item))}</li>" for item in conclusion.get("data_warnings", []))
    ticker = html.escape(str(payload.get("ticker")))
    header_html = "".join(f"<th>{item}</th>" for item in headers)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{ticker} 多策略回测</title>
<style>
body{{font-family:system-ui;background:#f4f7fb;color:#172033;margin:0}}
main{{max-width:1180px;margin:28px auto;background:white;padding:32px;border-radius:18px}}
.meta{{color:#64748b}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}
section{{border:1px solid #dbe5f2;border-radius:12px;padding:14px}}
table{{width:100%;border-collapse:collapse;margin-top:20px}}
th,td{{padding:9px;border-bottom:1px solid #e5e7eb;text-align:right}}
th:first-child,td:first-child{{text-align:left}}
@media(max-width:700px){{main{{margin:0;padding:18px;overflow:auto}}}}
</style></head><body><main>
<h1>{ticker} 多策略量化回测</h1>
<p class="meta">评价期 {payload.get('evaluation_start_date')} 至 {payload.get('evaluation_end_date')} ·
热身 {payload.get('warmup_bars')} 个交易日 · 数据核验 {payload.get('data_validation', {}).get('status')}</p>
<div class="cards">{winner_cards}</div>
<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<h2>数据与限制</h2><ul>{warnings}</ul>
<p>历史研究与模拟交易结果不代表未来表现，不构成投资建议。</p>
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
    import xlsxwriter
    from xlsxwriter.utility import xl_col_to_name

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
        ("数据核验", payload.get("data_validation", {}).get("status")),
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
                "position_model": row.get("strategy_spec", {}).get("position_model"),
            }
            for row in payload.get("comparisons", [])
        ],
        widths={"description": 45, "entry_rules": 50, "exit_rules": 50, "position_model": 50},
    )
    _write_table(workbook, "绩效对比", _comparison_rows(payload))
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
