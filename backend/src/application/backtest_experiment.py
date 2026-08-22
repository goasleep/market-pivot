"""Agent-designed, reproducible backtest experiments."""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from application.backtest_service import run_backtest, run_pool_backtest
from artifacts.service import artifact_service
from charts.echarts import ECHARTS_CDN, line_option, render_chart_container
from config import settings
from data.db_models import BacktestExperimentRecord
from data.tortoise_db import init_database
from llm.service import get_llm_service
from models.schemas import AssetType, PortfolioSpec, StrategySpec
from strategies.compiler import available_indicators, strategy_from_mapping
from strategies.skill_manager import register_strategy_spec

STRATEGY_DESIGN_SYSTEM = """你是一个严格的量化策略设计 Agent。请为 A 股股票或场内基金设计可执行的日线策略。
只允许使用用户提供的目标、日线 OHLCV 数据字段和下方受控指标，不能引用未来数据、实时新闻或当前才知道的财务数据。
必须返回 JSON 对象，不能输出 Markdown。JSON 必须包含：name、version、description、asset_types、indicators、
indicator_specs、entry_conditions、exit_conditions、entry_condition_logic、exit_condition_logic、
stop_loss_pct、take_profit_pct、position_size_pct、
rebalance_frequency、source。每个条件必须使用受控指标名称、gt/gte/lt/lte/eq/between 操作符和数值。

格式约束：
1. indicators 必须是字符串数组，例如 ["ma", "rsi"]。
2. indicator_specs 必须是数组，不能是以指标名为 key 的对象；每项格式为
   {"name":"ma","alias":"ma20","source":"close","window":20,"role":"filter","params":{}}。
3. stop_loss_pct、take_profit_pct、position_size_pct 必须使用 0 到 1 之间的小数，
   例如 0.05 表示 5%，不能填写 5 或 5%。
4. asset_types 必须是数组，只能包含 stock、etf、lof；source 必须为 llm。
5. entry_condition_logic 和 exit_condition_logic 只能是 all 或 any；“并且”使用 all，“或者”使用 any。
   多个入场过滤条件通常使用 all，多个独立退出触发条件通常使用 any。
"""

PORTFOLIO_DESIGN_SYSTEM = """你是一个严格的组合配置 Agent。请为给定的同一资产类型标的池设计可复现的组合规则。
只能返回 JSON 对象，字段必须符合：allocation_method=equal_weight、rebalance_frequency 为 daily/weekly/monthly/manual、
max_position_weight 为 0 到 1 的小数、max_positions 为正整数、cash_reserve 为 0 到 1 的小数。
第一版只允许 equal_weight，不得使用未来数据、主观择时或未提供的数据。
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class BacktestExperimentStore:
    """Durable metadata for replaying and auditing a backtest experiment."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_url = (
            f"sqlite://{Path(db_path).expanduser().resolve()}" if db_path is not None else settings.chat_database_url
        )

    async def save(self, experiment_id: str, status: str, payload: dict[str, Any]) -> None:
        timestamp = _now()
        await init_database(db_url=self.db_url)
        await BacktestExperimentRecord.update_or_create(
            experiment_id=experiment_id,
            defaults={
                "status": status,
                "payload_json": _json(payload),
                "created_at": timestamp,
                "updated_at": timestamp,
            },
        )

    async def get(self, experiment_id: str) -> dict[str, Any] | None:
        await init_database(db_url=self.db_url)
        row = await BacktestExperimentRecord.get_or_none(experiment_id=experiment_id)
        if row is None:
            return None
        return {
            "experiment_id": experiment_id,
            "status": row.status,
            **json.loads(row.payload_json),
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }

    async def list(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        await init_database(db_url=self.db_url)
        rows = await BacktestExperimentRecord.all().order_by("-created_at").offset(max(0, offset)).limit(
            max(1, min(limit, 200))
        )
        return [
            {
                "experiment_id": row.experiment_id,
                "status": row.status,
                **json.loads(row.payload_json),
                "created_at": row.created_at,
                "updated_at": row.updated_at,
            }
            for row in rows
        ]


backtest_experiments = BacktestExperimentStore()


async def design_strategy(
    *,
    objective: str,
    asset_type: AssetType,
    ticker: str | None = None,
    strategy_name: str | None = None,
) -> StrategySpec:
    """Use the configured LLM to design and validate a bounded strategy."""
    indicator_contract = json.dumps(available_indicators(), ensure_ascii=False)
    prompt = _json(
        {
            "objective": objective,
            "asset_type": asset_type.value,
            "ticker": ticker,
            "requested_name": strategy_name,
            "available_indicators": json.loads(indicator_contract),
        }
    )
    raw = await get_llm_service().chat_json(prompt, system=STRATEGY_DESIGN_SYSTEM)
    if not isinstance(raw, dict):
        raise ValueError("Agent 策略设计结果不是 JSON 对象")
    raw["source"] = "llm"
    raw.setdefault("asset_types", [asset_type.value])
    raw.setdefault("name", strategy_name or "agent_backtest_strategy")
    spec = strategy_from_mapping(raw, source="llm")
    if asset_type not in spec.asset_types:
        raise ValueError(f"Agent 策略不支持资产类型 {asset_type.value}")
    register_strategy_spec(spec)
    return spec


async def design_portfolio(
    *,
    objective: str,
    asset_type: AssetType,
    tickers: list[str],
) -> PortfolioSpec:
    """Use the configured LLM to design bounded, reproducible portfolio rules."""
    prompt = _json({"objective": objective, "asset_type": asset_type.value, "tickers": tickers})
    raw = await get_llm_service().chat_json(prompt, system=PORTFOLIO_DESIGN_SYSTEM)
    if not isinstance(raw, dict):
        raise ValueError("Agent 组合配置结果不是 JSON 对象")
    return PortfolioSpec.model_validate(raw)


def _slug(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    return safe.strip("_") or "backtest"


def _format_pct(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "暂无"


def _build_report_markdown(
    *,
    experiment_id: str,
    objective: str,
    spec: StrategySpec,
    result: dict[str, Any],
) -> str:
    snapshots = result.get("data_snapshots") or ([result.get("data_snapshot")] if result.get("data_snapshot") else [])
    portfolio = result.get("portfolio_spec")
    lines = [
        f"# Agent 回测实验报告：{spec.name}",
        "",
        f"- 实验 ID：`{experiment_id}`",
        f"- 实验目标：{objective}",
        f"- 标的：{result.get('ticker') or ', '.join(result.get('tickers', []))}",
        f"- 资产类型：{result.get('asset_type', 'unknown')}",
        f"- 回测模式：{result.get('mode', 'single')}",
        f"- 回测区间：{result.get('start_date')} 至 {result.get('end_date')}",
        "- 用途：历史研究与纸面交易准备，不构成投资建议，不执行真实交易。",
        "",
        "## 一、策略定义",
        "",
        f"- 名称：{spec.name} v{spec.version}",
        f"- 假设：{spec.description or 'Agent 未提供额外策略假设。'}",
        f"- 入场条件：{len(spec.entry_conditions)} 条",
        f"- 出场条件：{len(spec.exit_conditions)} 条",
        f"- 入场条件关系：{'全部满足' if spec.entry_condition_logic == 'all' else '任一满足'}",
        f"- 出场条件关系：{'全部满足' if spec.exit_condition_logic == 'all' else '任一满足'}",
        f"- 止损：{_format_pct(spec.stop_loss_pct)}",
        f"- 止盈：{_format_pct(spec.take_profit_pct)}",
        f"- 单次仓位：{_format_pct(spec.position_size_pct)}",
        f"- 指标：{', '.join(spec.indicators) or '由条件隐式使用'}",
        *[
            (
                f"- 指标定义：{item.alias or item.name}（{item.name}，"
                f"窗口 {item.window or item.params.get('window') or '默认'}，用途 {item.role}）"
            )
            for item in spec.indicator_specs
        ],
        "- 入场条件：",
        *[
            f"  - {condition.indicator} {condition.operator} {condition.value}"
            f"（窗口 {condition.window or '默认'}）"
            for condition in spec.entry_conditions
        ],
        "- 出场条件：",
        *[
            f"  - {condition.indicator} {condition.operator} {condition.value}"
            f"（窗口 {condition.window or '默认'}）"
            for condition in spec.exit_conditions
        ],
        "",
        "## 二、回测结果",
        "",
        f"- 初始资金：{result.get('initial_capital', 0):,.2f}",
        f"- 最终资产：{result.get('final_value', 0):,.2f}",
        f"- 总收益率：{_format_pct(result.get('total_return'))}",
        f"- 同标的买入持有参考：{_format_pct(result.get('buy_hold_return'))}"
        if result.get("buy_hold_return") is not None
        else "- 同标的买入持有参考：暂无",
        f"- 最大回撤：{_format_pct(result.get('max_drawdown'))}",
        f"- 夏普比率：{result.get('sharpe_ratio') if result.get('sharpe_ratio') is not None else '暂无'}",
        f"- 胜率：{_format_pct(result.get('win_rate'))}",
        f"- 已实现盈亏：{result.get('realized_pnl', 0):,.2f}",
        f"- 总费用：{result.get('total_fees', 0):,.2f}",
        f"- 成交笔数：{result.get('total_trades', 0)}",
        *(
            [
                "",
                "## 三、组合规则",
                "",
                f"- 分配方式：{portfolio.get('allocation_method')}",
                f"- 调仓频率：{portfolio.get('rebalance_frequency')}",
                f"- 最大单标的仓位：{_format_pct(portfolio.get('max_position_weight'))}",
                f"- 最大持仓数量：{portfolio.get('max_positions')}",
                f"- 现金保留：{_format_pct(portfolio.get('cash_reserve'))}",
            ]
            if portfolio
            else []
        ),
        "",
        "## 四、成交规则",
        "",
        *[
            f"- {key}：{value}"
            for key, value in (result.get("execution") or {}).items()
        ],
        "",
        "## 五、数据可追溯性",
        "",
    ]
    for snapshot in snapshots:
        if not snapshot:
            continue
        lines.extend(
            [
                (
                    f"- {snapshot.get('ticker')}: {snapshot.get('source')}，"
                    f"{snapshot.get('actual_start_date')} 至 {snapshot.get('actual_end_date')}，"
                    f"{snapshot.get('row_count')} 行"
                ),
                f"- 数据摘要 SHA-256：`{snapshot.get('sha256')}`",
                f"- 数据质量：{snapshot.get('quality', {}).get('status', 'unknown')}",
            ]
        )
    for rejection in result.get("data_rejections", []):
        lines.append(f"- 数据未纳入：{rejection.get('ticker')}（{rejection.get('reason')}）")
    lines.extend(
        [
            "",
            "## 六、限制和后续验证",
            "",
            "- 回测使用日线参考价格，不模拟盘口排队、部分成交和真实流动性。",
            "- 本报告不代表未来收益，仍需在未参与策略设计的时间区间进行验证。",
            "- 结束日的未平仓头寸按最后可用收盘价估值，未强制平仓。",
        ]
    )
    return "\n".join(lines) + "\n"


def _experiment_equity_points(result: dict[str, Any]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for item in result.get("equity_curve") or []:
        if not isinstance(item, dict):
            continue
        try:
            value = float(item["value"])
        except (KeyError, TypeError, ValueError):
            continue
        label = str(item.get("date") or len(points))
        points.append({"label": label, "value": value})
    return points[-250:]


def _experiment_drawdown_points(result: dict[str, Any]) -> list[dict[str, Any]]:
    points = _experiment_equity_points(result)
    drawdowns: list[dict[str, Any]] = []
    peak = 0.0
    for point in points:
        peak = max(peak, float(point["value"]))
        drawdown = ((float(point["value"]) - peak) / peak * 100) if peak else 0.0
        drawdowns.append({"label": point["label"], "value": round(drawdown, 4)})
    return drawdowns


def _experiment_equity_chart(result: dict[str, Any]) -> str:
    points = _experiment_equity_points(result)
    if len(points) < 2:
        return '<p class="chart-empty">暂无足够的资产曲线数据</p>'
    option = line_option("回测资产曲线", points)
    labels = {str(point["label"]): point["value"] for point in points}
    trade_markers = []
    for trade in result.get("trades") or []:
        if not isinstance(trade, dict):
            continue
        date = str(trade.get("date") or "")
        if date not in labels:
            continue
        raw_action = trade.get("action")
        action = str(getattr(raw_action, "value", raw_action) or "").lower()
        is_buy = action == "buy"
        trade_markers.append(
            {
                "name": "买入" if is_buy else "卖出",
                "coord": [date, labels[date]],
                "value": "买" if is_buy else "卖",
                "itemStyle": {"color": "#16a34a" if is_buy else "#dc2626"},
            }
        )
    if trade_markers:
        option["series"][0]["markPoint"] = {
            "symbol": "pin",
            "symbolSize": 42,
            "data": trade_markers,
            "label": {"color": "#fff", "fontSize": 11},
        }
    return render_chart_container(
        "experiment-equity-chart",
        option,
        aria_label="回测资产曲线",
        height=340,
    )


def _experiment_drawdown_chart(result: dict[str, Any]) -> str:
    points = _experiment_drawdown_points(result)
    if len(points) < 2:
        return '<p class="chart-empty">暂无足够的回撤数据</p>'
    option = line_option("回测回撤曲线", points)
    option["yAxis"] = {"type": "value", "max": 0, "axisLabel": {"formatter": "{value}%"}}
    option["series"][0].update(
        {
            "lineStyle": {"width": 2, "color": "#dc2626"},
            "itemStyle": {"color": "#dc2626"},
            "areaStyle": {"color": "#dc2626", "opacity": 0.1},
        }
    )
    return render_chart_container(
        "experiment-drawdown-chart",
        option,
        aria_label="回测回撤曲线",
        height=280,
    )


def _render_report_html(markdown: str, title: str, result: dict[str, Any] | None = None) -> str:
    chart_tokens = ""
    if result is not None:
        chart_tokens = (
            "\n\n[[EXPERIMENT_EQUITY_CHART]]\n\n"
            "[[EXPERIMENT_DRAWDOWN_CHART]]\n"
        )
        markdown = markdown.replace("## 二、回测结果", f"## 二、回测结果{chart_tokens}", 1)
    rendered: list[str] = []
    for raw_line in markdown.splitlines():
        line = html.escape(raw_line)
        if line in {"[[EXPERIMENT_EQUITY_CHART]]", "[[EXPERIMENT_DRAWDOWN_CHART]]"}:
            rendered.append(line)
        elif line.startswith("# "):
            rendered.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            rendered.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("- "):
            rendered.append(f"<li>{line[2:]}</li>")
        elif line:
            rendered.append(f"<p>{line}</p>")
    body = "\n".join(rendered)
    if result is not None:
        body = body.replace("[[EXPERIMENT_EQUITY_CHART]]", _experiment_equity_chart(result))
        body = body.replace("[[EXPERIMENT_DRAWDOWN_CHART]]", _experiment_drawdown_chart(result))
    # The report remains intentionally simple and self-contained; the source
    # Markdown is also persisted for machine-readable reproducibility.
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><script src="{ECHARTS_CDN}"></script><style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
background:#f5f7fb;color:#172033;line-height:1.7;margin:0}}
main{{max-width:920px;margin:32px auto;padding:36px 44px;background:#fff;box-shadow:0 8px 32px #17203312}}
h1{{color:#102a56;margin-top:0}} h2{{color:#1d4d91;border-bottom:1px solid #dce4f2;padding-bottom:6px;margin-top:28px}}
.echarts-chart{{margin:16px 0 12px;padding:8px;border:1px solid #dce4f2;border-radius:14px;background:#fbfcff}}
.chart-empty{{color:#64748b;font-size:13px}}
@media(max-width:640px){{main{{margin:0;padding:24px}}}}
</style></head><body><main>{body}</main></body></html>"""


def _trades_csv(trades: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    fields = ["date", "action", "ticker", "asset_type", "shares", "price", "amount", "commission", "tax"]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows({field: trade.get(field) for field in fields} for trade in trades)
    return buffer.getvalue()


def _data_csv(rows: Any) -> str:
    """Serialize the exact normalized rows consumed by the backtest."""
    if isinstance(rows, dict):
        flattened = []
        for ticker, records in rows.items():
            flattened.extend({"ticker": ticker, **record} for record in records)
        rows = flattened
    if not isinstance(rows, list) or not rows:
        return ""
    fields = sorted({key for row in rows if isinstance(row, dict) for key in row})
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(
        {field: row.get(field) for field in fields}
        for row in rows
        if isinstance(row, dict)
    )
    return buffer.getvalue()


def _portfolio_history_csv(history: list[dict[str, Any]]) -> str:
    rows: list[dict[str, Any]] = []
    for snapshot in history:
        base = {
            "date": snapshot.get("date"),
            "cash": snapshot.get("cash"),
            "total_value": snapshot.get("total_value"),
            "cash_weight": snapshot.get("cash_weight"),
        }
        positions = snapshot.get("positions") or []
        rows.extend(({**base, **position} for position in positions))
        if not positions:
            rows.append(base)
    return _data_csv(rows)


def _target_weights_csv(history: list[dict[str, Any]]) -> str:
    rows = [
        {"date": item.get("date"), "ticker": ticker, "target_weight": weight}
        for item in history
        for ticker, weight in (item.get("weights") or {}).items()
    ]
    return _data_csv(rows)


def _symbol_metrics_csv(metrics: list[dict[str, Any]]) -> str:
    return _data_csv(metrics)


async def run_backtest_experiment(
    *,
    objective: str,
    start_date: str,
    end_date: str,
    asset_type: AssetType | str = AssetType.STOCK,
    ticker: str | None = None,
    tickers: list[str] | None = None,
    initial_capital: float = 1_000_000,
    decision_interval: int = 1,
    fill_time: str = "next_open",
    strategy_spec: dict[str, Any] | None = None,
    strategy_name: str | None = None,
    mode: str = "auto",
    portfolio_spec: dict[str, Any] | PortfolioSpec | None = None,
) -> dict[str, Any]:
    """Design (unless supplied), run, persist, and publish one experiment."""
    kind = AssetType(asset_type)
    symbols = tickers or ([ticker] if ticker else [])
    if not symbols:
        raise ValueError("ticker 或 tickers 至少提供一个标的")
    actual_mode = mode if mode != "auto" else ("single" if len(symbols) == 1 else "pool")
    if actual_mode == "single" and len(symbols) != 1:
        raise ValueError("single 模式只能提供一个标的")
    if actual_mode in {"pool", "portfolio"} and len(symbols) < 2:
        raise ValueError(f"{actual_mode} 模式至少需要两个标的")
    experiment_id = f"bt-exp-{uuid4().hex[:16]}"
    if strategy_spec:
        spec = strategy_from_mapping(strategy_spec, source=strategy_spec.get("source") or "user")
    else:
        spec = await design_strategy(
            objective=objective,
            asset_type=kind,
            ticker=symbols[0] if len(symbols) == 1 else None,
            strategy_name=strategy_name,
        )

    executable_portfolio = None
    if actual_mode == "portfolio":
        executable_portfolio = (
            portfolio_spec
            if isinstance(portfolio_spec, PortfolioSpec)
            else PortfolioSpec.model_validate(portfolio_spec)
            if portfolio_spec is not None
            else await design_portfolio(objective=objective, asset_type=kind, tickers=symbols)
        )

    backtest_kwargs = {
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": initial_capital,
        "decision_interval": decision_interval,
        "fill_time": fill_time,
        "asset_type": kind,
        "strategy_spec": spec.model_dump(mode="json"),
        "capture_data": True,
    }
    result = (
        await run_pool_backtest(
            tickers=symbols,
            portfolio_spec=executable_portfolio.model_dump(mode="json") if executable_portfolio else None,
            **backtest_kwargs,
        )
        if len(symbols) > 1
        else await run_backtest(ticker=symbols[0], **backtest_kwargs)
    )
    data_rows = result.pop("_data_snapshot_rows", None)
    report_markdown = _build_report_markdown(
        experiment_id=experiment_id,
        objective=objective,
        spec=spec,
        result=result,
    )
    report_html = _render_report_html(
        report_markdown,
        f"Agent 回测实验报告：{spec.name}",
        result=result,
    )
    run_json = {
        "experiment_id": experiment_id,
        "objective": objective,
        "mode": actual_mode,
        "strategy_spec": spec.model_dump(mode="json"),
        "portfolio_spec": executable_portfolio.model_dump(mode="json") if executable_portfolio else None,
        "result": result,
    }
    artifact_inputs = [
            {
                "name": f"{_slug(spec.name)}-{experiment_id}-回测报告.html",
                "format": "html",
                "content": report_html,
                "description": "Agent 回测实验报告",
            },
            {
                "name": f"{_slug(spec.name)}-{experiment_id}-实验结果.json",
                "format": "json",
                "content": _json(run_json),
                "description": "可重放的策略、数据摘要和回测结果",
            },
            {
                "name": f"{_slug(spec.name)}-{experiment_id}-交易记录.csv",
                "format": "csv",
                "content": _trades_csv(result.get("trades", [])),
                "description": "回测成交记录",
            },
            {
                "name": f"{_slug(spec.name)}-{experiment_id}-报告源.md",
                "format": "md",
                "content": report_markdown,
                "description": "报告的可审计 Markdown 源文件",
            },
        ]
    data_csv = _data_csv(data_rows)
    if data_csv:
        artifact_inputs.append(
            {
                "name": f"{_slug(spec.name)}-{experiment_id}-历史数据快照.csv",
                "format": "csv",
                "content": data_csv,
                "description": "回测实际消费的标准化历史数据快照",
            }
        )
    if executable_portfolio:
        portfolio_artifacts = [
            (
                "持仓快照.csv",
                _portfolio_history_csv(result.get("portfolio_history", [])),
                "组合每日持仓、现金和权重",
            ),
            (
                "目标权重.csv",
                _target_weights_csv(result.get("target_weights_history", [])),
                "组合调仓目标权重",
            ),
            (
                "标的归因.csv",
                _symbol_metrics_csv(result.get("symbol_metrics", [])),
                "组合标的表现和交易归因",
            ),
        ]
        for suffix, content, description in portfolio_artifacts:
            if content:
                artifact_inputs.append(
                    {
                        "name": f"{_slug(spec.name)}-{experiment_id}-{suffix}",
                        "format": "csv",
                        "content": content,
                        "description": description,
                    }
                )
    artifacts = await artifact_service.create_user_artifacts(
        artifact_inputs,
        source="backtest",
        task_id=experiment_id,
        ticker=symbols[0] if len(symbols) == 1 else None,
        asset_type=kind.value,
        metadata={
            "experiment_id": experiment_id,
            "strategy_name": spec.name,
            "generated_by": "backtest_experiment",
            "strategy_sha256": hashlib.sha256(_json(spec.model_dump(mode="json")).encode()).hexdigest(),
        },
    )
    payload = {
        "experiment_id": experiment_id,
        "status": "completed",
        "objective": objective,
        "mode": actual_mode,
        "strategy_spec": spec.model_dump(mode="json"),
        "portfolio_spec": executable_portfolio.model_dump(mode="json") if executable_portfolio else None,
        "result": result,
        "artifacts": artifacts,
    }
    await backtest_experiments.save(experiment_id, "completed", payload)
    return payload
