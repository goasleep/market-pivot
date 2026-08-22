"""Restricted subprocess sandbox for LLM-authored research signals.

This is deliberately not a general Python runner. Candidate code can only
transform an immutable OHLCV frame into a binary target-position series. The
trusted trading engine remains the only component allowed to create fills and
performance metrics.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from engine.backtester import _calc_metrics, _execution_manifest
from engine.trading_engine import TimeAwareTradingEngine, decision_shares
from models.schemas import AssetType, Decision, SimulationAccountConfig, TradeDecision, TradePlan
from models.strategy_research import SandboxPolicy, SandboxValidation


class SandboxError(ValueError):
    """Raised when candidate code violates the research sandbox contract."""


_DENIED_NAMES = {
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "vars",
    "__import__",
}
_DENIED_MODULES = {
    "builtins",
    "ctypes",
    "importlib",
    "multiprocessing",
    "os",
    "pathlib",
    "resource",
    "shutil",
    "signal",
    "socket",
    "subprocess",
    "sys",
    "threading",
}
_DENIED_ATTRIBUTES = {
    "ctypeslib",
    "eval",
    "fromfile",
    "load",
    "memmap",
    "popen",
    "query",
    "request",
    "save",
    "savez",
    "savez_compressed",
    "system",
    "to_clipboard",
    "to_csv",
    "to_excel",
    "to_feather",
    "to_file",
    "to_hdf",
    "to_html",
    "to_json",
    "to_parquet",
    "to_pickle",
    "to_sql",
    "to_stata",
    "tofile",
    "urlopen",
}


def validate_source(source: str, policy: SandboxPolicy | None = None) -> dict[str, bool]:
    """Reject filesystem, network, process, reflection, and dynamic-code access."""
    policy = policy or SandboxPolicy()
    if len(source.encode("utf-8")) > policy.max_source_bytes:
        raise SandboxError("候选代码超过大小限制")
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise SandboxError(f"候选代码语法错误: {exc.msg}") from exc
    functions = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if policy.function_name not in functions:
        raise SandboxError(f"候选代码必须定义 {policy.function_name}(frame)")
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.ClassDef, ast.Lambda, ast.Global, ast.Nonlocal)):
            raise SandboxError(f"不允许的语法: {type(node).__name__}")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name.split(".", 1)[0] for alias in node.names]
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module.split(".", 1)[0])
            if any(name not in policy.allowed_imports for name in names):
                raise SandboxError("候选代码只能导入 numpy 或 pandas")
        if isinstance(node, ast.Name) and node.id in _DENIED_NAMES | _DENIED_MODULES:
            raise SandboxError(f"不允许访问名称: {node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise SandboxError("不允许访问双下划线属性")
            if node.attr in _DENIED_ATTRIBUTES or node.attr.startswith("read_"):
                raise SandboxError(f"不允许访问可能产生外部 I/O 的属性: {node.attr}")
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in _DENIED_MODULES:
                raise SandboxError(f"不允许访问模块: {root.id}")
    return {
        "syntax_valid": True,
        "entrypoint_present": True,
        "imports_allowlisted": True,
        "dangerous_names_absent": True,
    }


_CHILD_RUNNER = r'''
import json
import sys
import numpy as np
import pandas as pd

payload = json.loads(sys.stdin.read())
allowed_imports = set(payload["allowed_imports"])
def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".", 1)[0]
    if root not in allowed_imports:
        raise ImportError("module is not allowlisted")
    return __import__(name, globals, locals, fromlist, level)

safe_builtins = {
    "__import__": safe_import,
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "float": float, "int": int, "len": len,
    "list": list, "max": max, "min": min, "range": range, "round": round,
    "set": set, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    "Exception": Exception, "ValueError": ValueError,
}
namespace = {"__builtins__": safe_builtins, "np": np, "pd": pd}
exec(compile(payload["source"], "<candidate>", "exec"), namespace, namespace)
frame = pd.DataFrame(payload["records"])
result = namespace[payload["function_name"]](frame.copy(deep=True))
if isinstance(result, pd.Series):
    result = result.tolist()
elif isinstance(result, np.ndarray):
    result = result.tolist()
print(json.dumps({"positions": result}, allow_nan=False))
'''


def _limit_child(policy: SandboxPolicy) -> Any:
    def apply_limits() -> None:
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (policy.timeout_seconds, policy.timeout_seconds + 1))
            resource.setrlimit(resource.RLIMIT_FSIZE, (1_000_000, 1_000_000))
            if sys.platform.startswith("linux"):
                memory = policy.memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        except (ImportError, OSError, ValueError):
            return

    return apply_limits


async def _run_child(source: str, frame: pd.DataFrame, policy: SandboxPolicy) -> list[int]:
    records = frame.where(pd.notna(frame), None).to_dict(orient="records")
    payload = json.dumps(
        {
            "source": source,
            "records": records,
            "function_name": policy.function_name,
            "allowed_imports": list(policy.allowed_imports),
        },
        ensure_ascii=False,
        default=str,
    ).encode()
    with tempfile.TemporaryDirectory(prefix="a-share-sandbox-") as scratch:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-c",
            _CHILD_RUNNER,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=Path(scratch),
            env={"PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1", "PATH": os.defpath},
            preexec_fn=_limit_child(policy) if os.name == "posix" else None,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(payload), timeout=policy.timeout_seconds + 2)
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            raise SandboxError("候选代码执行超时") from exc
    if process.returncode != 0:
        message = stderr.decode("utf-8", errors="replace")[-1000:]
        raise SandboxError(f"候选代码执行失败: {message}")
    try:
        positions = json.loads(stdout)["positions"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SandboxError("候选代码没有返回可解析的目标仓位") from exc
    if not isinstance(positions, list) or len(positions) != len(frame):
        raise SandboxError("目标仓位长度必须与输入数据完全一致")
    normalized: list[int] = []
    for value in positions:
        if isinstance(value, bool):
            normalized.append(int(value))
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise SandboxError("目标仓位必须全部为 0 或 1") from exc
        if not np.isfinite(number) or number not in {0.0, 1.0}:
            raise SandboxError("MVP 沙盒仅接受 0/1 二元目标仓位")
        normalized.append(int(number))
    return normalized


async def validate_and_run_signals(
    source: str,
    frame: pd.DataFrame,
    policy: SandboxPolicy | None = None,
) -> tuple[list[int], SandboxValidation]:
    """Run twice and on historical prefixes to detect nondeterminism/leakage."""
    policy = policy or SandboxPolicy()
    static = validate_source(source, policy)
    errors: list[str] = []
    try:
        first = await _run_child(source, frame, policy)
        second = await _run_child(source, frame, policy)
        deterministic = first == second
        causal = True
        for ratio in (0.5, 0.75):
            end = max(2, int(len(frame) * ratio))
            prefix = await _run_child(source, frame.iloc[:end], policy)
            if prefix != first[:end]:
                causal = False
                break
        if not deterministic:
            errors.append("相同输入重复执行得到不同信号")
        if not causal:
            errors.append("前缀不变性失败，候选信号可能使用未来数据")
        validation = SandboxValidation(
            passed=deterministic and causal,
            static_checks=static,
            output_checks={
                "length_matches": len(first) == len(frame),
                "binary_positions": all(value in {0, 1} for value in first),
                "finite": True,
            },
            deterministic=deterministic,
            causal=causal,
            errors=errors,
        )
        return first, validation
    except SandboxError as exc:
        return [], SandboxValidation(
            passed=False,
            static_checks=static,
            output_checks={},
            deterministic=False,
            causal=False,
            errors=[str(exc)],
        )


def replay_target_positions(
    *,
    ticker: str,
    asset_type: AssetType | str,
    frame: pd.DataFrame,
    positions: list[int],
    initial_capital: float = 1_000_000,
    account_config: SimulationAccountConfig | None = None,
) -> dict[str, Any]:
    """Replay after-close targets with next-open fills in the trusted engine."""
    kind = AssetType(asset_type)
    if len(frame) != len(positions):
        raise ValueError("目标仓位长度与行情数据不一致")
    rules = account_config or SimulationAccountConfig(
        initial_cash=initial_capital,
        asset_type=kind,
        fill_time="next_open",
        max_single_position_pct=0.95,
        max_total_position_pct=0.95,
    )
    engine = TimeAwareTradingEngine(initial_capital=initial_capital, rules=rules)
    dates = frame["date"].astype(str).tolist()
    engine.set_available_dates(dates)
    pending: int | None = None
    curve: list[dict[str, Any]] = []
    for index, row in frame.reset_index(drop=True).iterrows():
        current_date = str(row["date"])
        engine.advance_to_date(current_date)
        if pending is not None:
            current = engine._find_position(ticker)
            if pending == 1 and current is None:
                decision = TradeDecision(
                    ticker=ticker,
                    asset_type=kind,
                    decision=Decision.BUY,
                    plan=TradePlan(position_size=0.95),
                )
                shares = decision_shares(engine.portfolio, rules, decision, float(row["open"]))
                engine.buy(ticker, shares, float(row["open"]), current_date)
            elif pending == 0 and current is not None:
                engine.sell(ticker, current.available_shares, float(row["open"]), current_date)
        engine.update_prices({ticker: float(row["close"])}, trigger_exits=False)
        total = engine.portfolio.total_value
        market_value = total - engine.portfolio.cash
        curve.append(
            {
                "date": current_date,
                "value": round(total, 2),
                "cash": round(engine.portfolio.cash, 2),
                "market_value": round(market_value, 2),
                "exposure": round(market_value / total, 8) if total else 0.0,
            }
        )
        pending = positions[index]
    metrics = _calc_metrics(curve, engine.portfolio.trades, initial_capital)
    return {
        **metrics,
        "ticker": ticker,
        "asset_type": kind.value,
        "initial_capital": initial_capital,
        "final_value": round(engine.portfolio.total_value, 2),
        "total_trades": len(engine.portfolio.trades),
        "equity_curve": curve,
        "trades": [item.model_dump(mode="json") for item in engine.portfolio.trades],
        "execution": _execution_manifest(engine, "next_open"),
    }


def source_sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()
