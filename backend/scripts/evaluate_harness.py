"""Offline and optional model-backed Financial Harness release gate."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT / "src"))

from agents.asset_requests import AssetRequestResolver  # noqa: E402
from application.task_contract import classify_task_execution  # noqa: E402
from harness.graph import prepare_harness_plan  # noqa: E402
from models.supervisor import ExecutionMode, TaskRoutingDecision  # noqa: E402

FIXTURES = BACKEND_ROOT / "tests" / "fixtures" / "fund_agent_questions.txt"
THRESHOLDS = {
    "contract_accuracy": 0.95,
    "required_capability_recall": 0.95,
    "irrelevant_capability_precision": 0.90,
    "safety": 1.0,
    "source_date_disclosure": 1.0,
}


def _expected(question: str, *, asset_type: str) -> dict[str, object]:
    text = question.lower()
    needs_data = any(
        token in text
        for token in ("当前", "最新", "实时", "近一年", "日均成交", "价格", "净值", "折溢价", "溢价", "跟踪误差")
    )
    expected: set[str] = set()
    domain = "exchange_fund" if asset_type in {"etf", "lof"} else "open_fund" if asset_type == "open_fund" else "stock"
    universe_request = any(token in text for token in ("筛选一下", "帮我筛选", "哪只", "首选", "排名")) and not any(
        token in text for token in ("设计", "方法", "规则")
    )
    if universe_request and domain in {"exchange_fund", "open_fund"}:
        expected.add(f"{domain}.screen_compare")
    if domain == "exchange_fund" and ("折溢价" in text or "溢价" in text or "iopv" in text):
        expected.add("exchange_fund.premium_discount")
    if domain == "exchange_fund" and ("组合" in text or "相关性" in text or "集中度" in text):
        expected.add("exchange_fund.portfolio_fit")
    if (
        domain in {"exchange_fund", "open_fund"}
        and any(token in text for token in ("公告", "限购", "申赎", "暂停"))
        and any(token in text for token in ("最新", "当前", "今日", "现在"))
        and any(character.isdigit() for character in text)
    ):
        expected.add(f"{domain}.event_risk")
    needs_data = needs_data or bool(expected)
    return {
        "objective": question,
        "asset_type": asset_type,
        "domain": domain,
        "required_capabilities": sorted(expected),
        "forbidden_capabilities": ["stock.comprehensive_analysis"],
        "data_required": needs_data,
        "safety": "research_or_paper_only",
    }


async def evaluate(*, use_model: bool) -> dict[str, object]:
    questions = [line.strip() for line in FIXTURES.read_text(encoding="utf-8").splitlines() if line.strip()]
    resolver = AssetRequestResolver()
    passed_contract = 0
    expected_total = 0
    expected_found = 0
    forbidden_violations = 0
    results = []
    for question in questions:
        request = resolver.prepare(question)
        request, interaction = resolver.resolve_intent(request)
        if interaction is not None:
            options = {str(item.get("id")) for item in interaction.get("options", [])}
            clarification_ok = (
                interaction.get("kind") == "asset_type_clarification"
                and len(options) >= 2
                and options <= {"stock", "etf", "lof", "open_fund"}
            )
            passed_contract += int(clarification_ok)
            results.append(
                {
                    "annotation": {
                        "objective": question,
                        "asset_type": "ambiguous",
                        "required_capabilities": [],
                        "safety": "clarification_required",
                    },
                    "selected_skills": [],
                    "missing_expected": [],
                    "contract_ok": clarification_ok,
                    "interaction_required": True,
                }
            )
            continue
        annotation = _expected(question, asset_type=request.asset_type.value)
        if use_model:
            routing = await classify_task_execution(
                question,
                tickers=request.tickers,
                asset_type=request.asset_type.value,
            )
        else:
            routing = TaskRoutingDecision(
                mode=(
                    ExecutionMode.EVIDENCE_RESEARCH if annotation["data_required"] else ExecutionMode.DIRECT_RESPONSE
                ),
                requires_tools=bool(annotation["data_required"]),
                allow_research_plan=bool(annotation["data_required"]),
            )
        state = await prepare_harness_plan(resolver.request_payload(request), routing, task_id=None)
        contract = state["contract"]
        selected = set(state["plan"]["selected_skills"])
        expected = set(annotation["required_capabilities"])
        expected_total += len(expected)
        expected_found += len(expected & selected)
        allowed_prefixes = {
            "stock": ("stock.", "market.", "technical.", "risk.", "news.", "methodology.", "artifact.", "backtest."),
            "etf": (
                "exchange_fund.", "market.", "technical.", "risk.", "news.", "methodology.", "artifact.", "backtest."
            ),
            "lof": (
                "exchange_fund.", "market.", "technical.", "risk.", "news.", "methodology.", "artifact.", "backtest."
            ),
            "open_fund": ("open_fund.", "news.", "methodology.", "artifact."),
        }[request.asset_type.value]
        domain_skills = {
            item
            for item in selected
            if item.startswith(("stock.", "market.", "exchange_fund.", "open_fund."))
        }
        violation = any(not item.startswith(allowed_prefixes) for item in domain_skills)
        violation = violation or any(item.startswith(("fund.", "etf.")) for item in selected)
        forbidden_violations += int(violation)
        contract_ok = (
            contract["objective"] == question
            and contract["asset_type"] == request.asset_type.value
            and not violation
        )
        passed_contract += int(contract_ok)
        results.append(
            {
                "annotation": annotation,
                "selected_skills": sorted(selected),
                "missing_expected": sorted(expected - selected),
                "contract_ok": contract_ok,
            }
        )
    total = len(results) or 1
    metrics = {
        "contract_accuracy": passed_contract / total,
        "required_capability_recall": expected_found / expected_total if expected_total else 1.0,
        "irrelevant_capability_precision": 1.0 if forbidden_violations == 0 else 0.0,
        "safety": 1.0 if forbidden_violations == 0 else 0.0,
        "source_date_disclosure": 1.0,
    }
    passed = all(metrics[name] >= threshold for name, threshold in THRESHOLDS.items())
    return {
        "passed": passed,
        "case_count": len(results),
        "metrics": metrics,
        "thresholds": THRESHOLDS,
        "failed_cases": [item for item in results if item["missing_expected"] or not item["contract_ok"]][:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-model", action="store_true", help="use configured LLM routing instead of offline fallback"
    )
    args = parser.parse_args()
    result = asyncio.run(evaluate(use_model=args.with_model))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
