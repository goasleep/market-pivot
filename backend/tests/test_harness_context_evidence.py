import json

from harness.context import ContextAssembler
from harness.evidence import EvidenceStore
from harness.models import HarnessPlan, HarnessStep, HarnessTaskContract, SkillManifest


def test_context_budget_preserves_contract_and_selected_skill_only():
    contract = HarnessTaskContract(
        objective="比较两只 ETF",
        asset_type="etf",
        required_capabilities=("market.quote",),
        allowed_capabilities=("market.quote",),
    )
    skill = SkillManifest(
        id="market.quote",
        version="1",
        title="行情",
        description="行情",
        capabilities=("market.quote",),
        instructions="只用结构化行情。" * 100,
    )
    plan = HarnessPlan(
        plan_id="p1",
        objective=contract.objective,
        contract_id=contract.contract_id,
        selected_skills=(skill.id,),
        steps=(HarnessStep(id="s1", capability_id="market.quote", skill_id=skill.id, title="行情"),),
    )
    text = ContextAssembler().assemble(contract, plan, (skill,), max_chars=2500)
    assert contract.contract_id in text
    assert "只用结构化行情" in text
    assert len(text) <= 2500


def test_evidence_store_keeps_raw_result_out_of_compressed_record():
    raw = json.dumps(
        {
            "status": "available",
            "data": {"price": 1.234, "large": "x" * 10_000},
            "as_of": "2026-08-26",
            "sources": [{"source_id": "fixture"}],
        }
    )
    store = EvidenceStore()
    record = store.add_tool_result("market.quote", "get_realtime_quote", raw)
    compressed = store.compressed_context()[0]
    assert store.raw(record.evidence_id) == raw
    assert "raw_result" not in compressed
    assert len(compressed["summary"]) <= 2400


def test_evidence_store_accepts_multi_source_provenance_lists():
    raw = json.dumps(
        {
            "data_type": "market_data",
            "available": True,
            "history": [{"date": "2026-08-26", "close": 1.0}],
            "provenance": [
                {"source_id": "eastmoney", "as_of": "2026-08-26", "freshness": "historical"},
                {"source_id": "sina", "as_of": "2026-08-26", "freshness": "historical"},
            ],
        }
    )
    record = EvidenceStore().add_tool_result("market.history", "get_historical_prices", raw)
    assert [item["source_id"] for item in record.sources] == ["eastmoney", "sina"]
    assert record.as_of == "2026-08-26"
    assert record.freshness == "historical"
