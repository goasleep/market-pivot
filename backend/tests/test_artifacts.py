import base64
from types import SimpleNamespace

from artifacts.service import ArtifactService, render_analysis_html, render_analysis_markdown
from models.schemas import Decision, TradeDecision


class MemoryArtifactStorage:
    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def put(self, object_key: str, content: bytes, content_type: str) -> None:
        del content_type
        self.objects[object_key] = content

    def get(self, object_key: str) -> bytes:
        return self.objects[object_key]


def test_analysis_artifacts_are_written_and_retrievable(tmp_path):
    storage = MemoryArtifactStorage()
    service = ArtifactService(db_path=tmp_path / "artifacts.db", storage=storage)
    decision = TradeDecision(
        ticker="510300",
        decision=Decision.HOLD,
        confidence=0.72,
        reasoning="趋势仍需观察，等待更明确的入场信号。",
    )

    artifacts = service.create_analysis_artifacts(decision, source="test")

    assert [item["mime_type"] for item in artifacts] == ["text/html"]
    for artifact in artifacts:
        saved = service.get(artifact["artifact_id"])
        assert saved is not None
        assert saved["object_key"] in storage.objects
        assert saved["size_bytes"] > 0
        assert artifact["preview_url"].endswith(f"/{artifact['artifact_id']}/preview")
    assert "研究分析报告" in storage.objects[artifacts[0]["object_key"]].decode()
    assert len(service.list()) == 1


def test_user_artifacts_support_multiple_text_and_binary_files(tmp_path):
    storage = MemoryArtifactStorage()
    service = ArtifactService(db_path=tmp_path / "artifacts.db", storage=storage)

    artifacts = service.create_user_artifacts(
        [
            {"name": "持仓概览", "format": "md", "content": "# 国家队持仓\n\n暂无完整披露。"},
            {"name": "来源", "format": "json", "content": '{"source": "公开披露"}'},
            {
                "name": "图表",
                "format": "png",
                "content_base64": base64.b64encode(b"fake-png").decode(),
            },
        ],
        source="chat",
        conversation_id="conversation-test",
        task_id="task-test",
    )

    assert len(artifacts) == 3
    assert {item["artifact_type"] for item in artifacts} == {"document", "data", "image"}
    assert {item["mime_type"] for item in artifacts} == {
        "text/markdown",
        "application/json",
        "image/png",
    }
    assert len(service.list()) == 3

    retried = service.create_user_artifacts(
        [{"name": "持仓概览.md", "format": "md", "content": "# 国家队持仓\n\n暂无完整披露。"}],
        source="chat",
        conversation_id="conversation-test",
        task_id="task-test",
    )
    assert retried[0]["artifact_id"] == artifacts[0]["artifact_id"]
    assert len(service.list()) == 3


def test_report_renders_source_urls_as_hidden_links():
    url = "https://example.com/report?id=1"
    decision = TradeDecision(
        ticker="512480",
        decision=Decision.HOLD,
        confidence=0.72,
        reasoning=f"趋势参考：{url}",
    )
    context = SimpleNamespace(
        web_results=[
            {
                "title": "半导体行业新闻",
                "snippet": "行业数据摘要",
                "link": url,
            }
        ],
    )

    report = render_analysis_html(
        render_analysis_markdown(decision, context),
        decision,
        "2026-08-16T00:00:00+00:00",
    )

    assert f'href="{url}"' in report
    assert ">查看来源</a>" in report
    assert ">查看链接</a>" in report
    assert f">{url}</a>" not in report


def test_report_uses_llm_copy_without_replacing_structured_sections():
    decision = TradeDecision(
        ticker="600000",
        asset_type="stock",
        decision="hold",
        confidence=0.65,
        target_price=12.5,
        stop_loss=9.8,
        position_size=0.2,
        reasoning="结构化决策依据",
    )
    report = render_analysis_markdown(
        decision,
        report_copy={
            "executive_summary": "LLM 生成的统一风格结论。",
            "decision_basis": "LLM 生成的决策解释。",
            "risk_summary": "LLM 生成的风险提示。",
        },
    )

    assert "LLM 生成的统一风格结论。" in report
    assert "LLM 生成的决策解释。" in report
    assert "LLM 生成的风险提示。" in report
    assert "决策：**hold**" in report
    assert "目标价：12.5" in report


def test_report_embeds_history_trend_chart_when_history_is_available():
    decision = TradeDecision(
        ticker="510300",
        decision=Decision.HOLD,
        confidence=0.7,
        reasoning="趋势观察",
    )
    context = SimpleNamespace(
        current_price=4.2,
        history=[
            {"date": "2026-08-12", "close": 4.0},
            {"date": "2026-08-13", "close": 4.1},
            {"date": "2026-08-14", "close": 4.2},
        ],
    )

    report = render_analysis_html(
        render_analysis_markdown(decision, context),
        decision,
        "2026-08-16T00:00:00+00:00",
        context,
    )

    assert "历史收盘价趋势" in report
    assert "echarts.min.js" in report
    assert 'renderer: "canvas"' in report
    assert "<svg" not in report
    assert "2026-08-12" in report
    assert "2026-08-14" in report
    assert "[[PRICE_TREND_CHART]]" not in report


def test_report_embeds_signal_attribution_chart_when_dashboard_is_available():
    decision = TradeDecision(
        ticker="510300",
        decision=Decision.HOLD,
        confidence=0.7,
        reasoning="趋势观察",
        dashboard={
            "signal_attribution": {
                "technical_score": 70,
                "sentiment_score": -20,
                "fundamental_score": 35,
                "market_regime_score": 10,
            }
        },
    )

    report = render_analysis_html(
        render_analysis_markdown(decision),
        decision,
        "2026-08-16T00:00:00+00:00",
    )

    assert "信号归因" in report
    assert "signal-attribution-chart" in report
    assert '"type":"bar"' in report
    assert "<svg" not in report
    assert "[[SIGNAL_ATTRIBUTION_CHART]]" not in report
