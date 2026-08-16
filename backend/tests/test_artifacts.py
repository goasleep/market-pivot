from artifacts.service import ArtifactService
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
