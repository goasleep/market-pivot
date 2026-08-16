from data.source_registry import data_sources, provenance, provenance_for_labels


def test_data_source_registry_resolves_capability_and_aliases():
    source = data_sources.resolve("market.quote")
    assert source.source_id == "akshare"
    assert data_sources.get("Serper / Google Search").source_id == "serper"


def test_provenance_is_uniform_and_preserves_source_dates():
    records = provenance("akshare", as_of="2026-08-15", freshness="historical")
    assert records[0]["source_id"] == "akshare"
    assert records[0]["as_of"] == "2026-08-15"
    assert records[0]["fetched_at"]

    web_records = provenance_for_labels(["Serper / Google Search", "DDGS metasearch"])
    assert [item["source_id"] for item in web_records] == ["serper", "ddgs"]
