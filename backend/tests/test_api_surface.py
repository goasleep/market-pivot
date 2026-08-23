import pytest

from api.main import app
from api.routers.backtest import _strategy_payload
from models.schemas import AssetType


def test_agent_entry_is_chat_sse_only():
    paths = set(app.openapi()["paths"])

    assert "/api/chat/tasks/{task_id}/stream" in paths
    assert not any(path.startswith("/api/automation") for path in paths)
    assert not any(path.startswith("/api/backtest/experiments") for path in paths)
    assert not any(path.startswith("/api/deployments/experiments") for path in paths)
    assert "/api/deployments/candidates/generate" not in paths


def test_backtest_resolves_named_strategy_without_agent():
    payload = _strategy_payload("bull_trend", None, AssetType.ETF)

    assert payload["name"] == "bull_trend"
    assert payload["source"] == "yaml"


def test_backtest_rejects_unknown_named_strategy():
    with pytest.raises(ValueError, match="策略不存在或不可执行"):
        _strategy_payload("missing-strategy", None, AssetType.ETF)
