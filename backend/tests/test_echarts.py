import pytest

from charts.echarts import line_option, render_chart_container, render_chart_document
from tools.artifacts import _render_line_chart_html


def test_echarts_line_chart_uses_canvas_and_not_hand_built_svg():
    option = line_option(
        "510300 历史走势",
        [
            {"label": "2026-08-12", "value": 4.0},
            {"label": "2026-08-13", "value": 4.2},
        ],
    )
    chart = render_chart_container("test-chart", option, aria_label="走势")
    document = render_chart_document("走势", chart)

    assert "echarts.init" in document
    assert 'renderer: "canvas"' in document
    assert "<svg" not in document
    assert "2026-08-12" in document


def test_chat_chart_artifact_is_echarts_html():
    artifact = _render_line_chart_html(
        "走势图",
        [
            {"label": "D1", "value": 10},
            {"label": "D2", "value": 12},
        ],
    )

    assert artifact.startswith("<!doctype html>")
    assert "echarts.min.js" in artifact
    assert '"type":"line"' in artifact
    assert "<svg" not in artifact


def test_echarts_line_chart_requires_two_points():
    with pytest.raises(ValueError, match="至少需要两个"):
        line_option("走势", [{"label": "D1", "value": 10}])
