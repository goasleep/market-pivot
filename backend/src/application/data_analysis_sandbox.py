"""Optional isolated fallback for transforms that the declarative DSL cannot express."""

from __future__ import annotations

import ast
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from application.research_sandbox import (
    _DENIED_ATTRIBUTES,
    _DENIED_MODULES,
    _DENIED_NAMES,
    SandboxError,
)


class DataAnalysisSandboxPolicy(BaseModel):
    timeout_seconds: int = Field(default=10, ge=1, le=30)
    memory_mb: int = Field(default=512, ge=128, le=2048)
    max_source_bytes: int = Field(default=20_000, ge=1_000, le=100_000)
    max_input_rows: int = Field(default=50_000, ge=1, le=200_000)
    max_output_rows: int = Field(default=10_000, ge=1, le=100_000)
    allowed_imports: tuple[str, ...] = ("numpy", "pandas")
    function_name: str = "analyze_data"


def validate_analysis_source(source: str, policy: DataAnalysisSandboxPolicy | None = None) -> None:
    """Allow dataframe computation while rejecting I/O, reflection and process access."""
    policy = policy or DataAnalysisSandboxPolicy()
    if len(source.encode()) > policy.max_source_bytes:
        raise SandboxError("分析代码超过大小限制")
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise SandboxError(f"分析代码语法错误: {exc.msg}") from exc
    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    if policy.function_name not in functions:
        raise SandboxError(f"分析代码必须定义 {policy.function_name}(frame)")
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.ClassDef, ast.Lambda, ast.Global, ast.Nonlocal)):
            raise SandboxError(f"不允许的语法: {type(node).__name__}")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            roots = [alias.name.split(".", 1)[0] for alias in node.names]
            if isinstance(node, ast.ImportFrom) and node.module:
                roots.append(node.module.split(".", 1)[0])
            if any(root not in policy.allowed_imports for root in roots):
                raise SandboxError("分析代码只能导入 numpy 或 pandas")
        if isinstance(node, ast.Name) and node.id in _DENIED_NAMES | _DENIED_MODULES:
            raise SandboxError(f"不允许访问名称: {node.id}")
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("__") or node.attr in _DENIED_ATTRIBUTES or node.attr.startswith("read_")
        ):
            raise SandboxError(f"不允许访问可能产生外部 I/O 的属性: {node.attr}")


_ANALYSIS_RUNNER = r"""
import json
import sys
import numpy as np
import pandas as pd

payload = json.loads(sys.stdin.read())
allowed_imports = set(payload["allowed_imports"])
def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".", 1)[0] not in allowed_imports:
        raise ImportError("module is not allowlisted")
    return __import__(name, globals, locals, fromlist, level)
safe_builtins = {
    "__import__": safe_import, "abs": abs, "all": all, "any": any, "bool": bool,
    "dict": dict, "enumerate": enumerate, "float": float, "int": int, "len": len,
    "list": list, "max": max, "min": min, "range": range, "round": round,
    "set": set, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    "Exception": Exception, "ValueError": ValueError,
}
namespace = {"__builtins__": safe_builtins, "np": np, "pd": pd}
exec(compile(payload["source"], "<analysis-candidate>", "exec"), namespace, namespace)
frame = pd.DataFrame(payload["records"])
result = namespace[payload["function_name"]](frame.copy(deep=True))
if isinstance(result, pd.DataFrame):
    result = result.where(pd.notna(result), None).to_dict(orient="records")
elif isinstance(result, pd.Series):
    result = result.where(pd.notna(result), None).to_frame("value").to_dict(orient="records")
print(json.dumps({"rows": result}, allow_nan=False, default=str))
"""


def _limit_child(policy: DataAnalysisSandboxPolicy) -> Any:
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


async def run_data_analysis(
    source: str,
    records: list[dict[str, Any]],
    policy: DataAnalysisSandboxPolicy | None = None,
) -> list[dict[str, Any]]:
    """Run twice in an isolated process and reject nondeterministic or oversized output."""
    policy = policy or DataAnalysisSandboxPolicy()
    validate_analysis_source(source, policy)
    if len(records) > policy.max_input_rows:
        raise SandboxError("分析输入超过行数限制")
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

    async def execute() -> list[dict[str, Any]]:
        with tempfile.TemporaryDirectory(prefix="financial-analysis-sandbox-") as scratch:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                "-c",
                _ANALYSIS_RUNNER,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path(scratch),
                env={"PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1", "PATH": os.defpath},
                preexec_fn=_limit_child(policy) if os.name == "posix" else None,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(payload), timeout=policy.timeout_seconds + 2
                )
            except TimeoutError as exc:
                process.kill()
                await process.wait()
                raise SandboxError("分析代码执行超时") from exc
        if process.returncode != 0:
            raise SandboxError(f"分析代码执行失败: {stderr.decode(errors='replace')[-1000:]}")
        try:
            rows = json.loads(stdout)["rows"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise SandboxError("分析代码没有返回可解析的表格行") from exc
        if not isinstance(rows, list) or any(not isinstance(item, dict) for item in rows):
            raise SandboxError("分析代码必须返回 DataFrame、Series 或对象行列表")
        if len(rows) > policy.max_output_rows:
            raise SandboxError("分析输出超过行数限制")
        return rows

    first = await execute()
    second = await execute()
    if first != second:
        raise SandboxError("分析代码未通过确定性复跑")
    return first
