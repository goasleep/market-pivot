from data import anysearch_provider


def test_anysearch_provider_posts_documented_request_and_normalises_response(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "code": 0,
                "data": {
                    "results": [
                        {
                            "title": "AnySearch result",
                            "url": "https://example.com/result",
                            "snippet": "A useful summary",
                        }
                    ]
                },
            }

    class FakeClient:
        def __init__(self, **kwargs):
            captured["timeout"] = kwargs["timeout"]

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, endpoint, *, headers, json):
            captured.update({"endpoint": endpoint, "headers": headers, "json": json})
            return FakeResponse()

    monkeypatch.setattr(anysearch_provider.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(anysearch_provider.settings, "anysearch_api_key", "test-key")
    monkeypatch.setattr(anysearch_provider.settings, "anysearch_base_url", "https://api.example")
    monkeypatch.setattr(anysearch_provider.settings, "anysearch_zone", "cn")
    monkeypatch.setattr(anysearch_provider.settings, "anysearch_language", "zh-CN")
    monkeypatch.setattr(anysearch_provider._cache, "get", lambda *args, **kwargs: None)
    monkeypatch.setattr(anysearch_provider._cache, "set", lambda *args, **kwargs: None)

    result = anysearch_provider.search_web_anysearch("510300 最新公告", num_results=5)

    assert result["available"] is True
    assert result["source"] == "AnySearch"
    assert result["results"] == [
        {
            "position": 1,
            "title": "AnySearch result",
            "link": "https://example.com/result",
            "snippet": "A useful summary",
            "date": "",
            "source": "AnySearch",
        }
    ]
    assert captured == {
        "timeout": 12.0,
        "endpoint": "https://api.example/v1/search",
        "headers": {"Content-Type": "application/json", "Authorization": "Bearer test-key"},
        "json": {
            "query": "510300 最新公告",
            "max_results": 5,
            "format": "json",
            "zone": "cn",
            "language": "zh-CN",
        },
    }
