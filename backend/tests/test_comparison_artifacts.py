import io
import zipfile

from application.comparison_artifacts import _html_report, _workbook


def test_audit_workbook_has_fixed_sheets_chart_and_formula_checks():
    payload = {
        "ticker": "510300",
        "initial_capital": 1_000_000,
        "evaluation_start_date": "2020-01-02",
        "evaluation_end_date": "2026-08-21",
        "warmup_bars": 252,
        "strategy_count": 1,
        "acceptance": {"satisfied": True, "checks": {"fair_evaluation_period": True}},
        "data_validation": {"status": "verified", "selected_source": "eastmoney", "candidates": []},
        "conclusion": {"recommendations": ["建议优先验证买入持有。"]},
        "comparisons": [
            {
                "strategy_name": "buy_hold",
                "display_name": "买入持有",
                "total_return": 0.1,
                "final_value": 1_100_000,
                "total_fees": 20,
                "equity_curve": [
                    {"date": "2020-01-02", "value": 1_000_000},
                    {"date": "2026-08-21", "value": 1_100_000},
                ],
                "drawdown_curve": [
                    {"date": "2020-01-02", "value": 0},
                    {"date": "2026-08-21", "value": -0.02},
                ],
                "signal_curve": [
                    {"date": "2020-01-02", "target_exposure": 0.95, "actual_exposure": 0},
                    {"date": "2026-08-21", "target_exposure": 0.95, "actual_exposure": 0.95},
                ],
                "trades": [],
                "diagnostics": {"out_of_sample": {"out_of_sample_return": 0.02}},
                "strategy_spec": {},
            }
        ],
        "cost_scenarios": {},
        "parameter_sensitivity": {},
        "execution": {"fill_time": "next_open"},
        "benchmark": "buy_hold",
        "ranking": ["buy_hold"],
        "data_snapshot": {"sha256": "a" * 64},
        "strategy_assessments": [
            {
                "rank": 1,
                "strategy_name": "buy_hold",
                "display_name": "买入持有",
                "mechanism": "首个交易日买入并持有",
                "strengths": ["总收益排名第一"],
                "weaknesses": ["完整承受市场回撤"],
                "why_good": "总收益排名第一",
                "why_bad": "完整承受市场回撤",
                "suitable_market": "持续上行行情",
                "failure_mode": "下跌行情",
                "verdict": "有条件候选",
            }
        ],
        "market_regime_attribution": {
            "strategies": [
                {
                    "strategy_name": "buy_hold",
                    "display_name": "买入持有",
                    "regimes": [{"label": "上涨趋势", "days": 100, "strategy_return": 0.1}],
                }
            ]
        },
        "trade_attribution": {
            "strategies": [
                {
                    "strategy_name": "buy_hold",
                    "display_name": "买入持有",
                    "closed_trade_segments": 0,
                    "open_shares": 100,
                    "matched_trades": [],
                }
            ]
        },
        "robustness_assessments": [
            {"strategy_name": "buy_hold", "display_name": "买入持有", "grade": "moderate"}
        ],
        "research_decision": {
            "preferred_display_name": "买入持有",
            "robustness_grade": "moderate",
            "falsification_risks": ["优势可能由少数行情阶段贡献。"],
            "next_experiments": [
                {"question": "弱势阶段能否改善", "method": "滚动回测", "success_criteria": "多数窗口为正"}
            ],
            "deployment_gate": {
                "status": "research_only",
                "message": "继续模拟验证。",
                "checks": {"out_of_sample_positive": True},
            },
        },
        "market_benchmark": {
            "status": "available",
            "ticker": "000300",
            "name": "沪深300",
            "comparisons": [
                {
                    "strategy_name": "buy_hold",
                    "asset_total_return": 0.1,
                    "market_total_return": 0.06,
                    "excess_return": 0.04,
                }
            ],
        },
    }
    selected = [
        {"date": "2020-01-02", "open": 10, "high": 10, "low": 10, "close": 10, "volume": 1}
    ]

    content = _workbook(payload, selected)

    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        workbook_xml = archive.read("xl/workbook.xml").decode()
        worksheet_xml = "".join(
            archive.read(name).decode() for name in archive.namelist() if name.startswith("xl/worksheets/sheet")
        )
        assert "结果总览" in workbook_xml
        assert "模型检查" in workbook_xml
        assert "策略诊断" in workbook_xml
        assert "市场阶段归因" in workbook_xml
        assert "交易归因" in workbook_xml
        assert "稳健性评估" in workbook_xml
        assert "下一轮实验" in workbook_xml
        assert "部署准入" in workbook_xml
        assert "同期大盘同策略" in workbook_xml
        assert "选定行情数据" in workbook_xml
        assert "COUNTA" in worksheet_xml
        assert "drawings/drawing" in " ".join(archive.namelist())


def test_html_report_contains_diagnosis_charts_and_audit_sections():
    payload = {
        "ticker": "510300",
        "strategy_count": 1,
        "benchmark": "buy_hold",
        "ranking": ["buy_hold"],
        "evaluation_start_date": "2020-01-02",
        "evaluation_end_date": "2026-08-21",
        "warmup_bars": 252,
        "data_snapshot": {"sha256": "a" * 64},
        "data_validation": {"status": "verified", "selected_source": "eastmoney"},
        "acceptance": {"checks": {"required_metrics": True}},
        "execution": {"fill_time": "next_open"},
        "conclusion": {"recommendations": ["建议优先验证买入持有。"]},
        "comparisons": [
            {
                "strategy_name": "buy_hold",
                "display_name": "买入持有",
                "total_return": 0.1,
                "max_drawdown": 0.05,
                "equity_curve": [
                    {"date": "2020-01-02", "value": 1_000_000},
                    {"date": "2026-08-21", "value": 1_100_000},
                ],
                "drawdown_curve": [
                    {"date": "2020-01-02", "value": 0},
                    {"date": "2026-08-21", "value": -0.05},
                ],
                "diagnostics": {"out_of_sample": {"out_of_sample_return": 0.03}},
            }
        ],
        "strategy_assessments": [
            {
                "rank": 1,
                "strategy_name": "buy_hold",
                "display_name": "买入持有",
                "mechanism": "首个交易日买入并持有",
                "strengths": ["总收益排名第一"],
                "weaknesses": ["完整承受市场回撤"],
                "suitable_market": "持续上行行情",
                "failure_mode": "下跌行情",
                "verdict": "有条件候选",
            }
        ],
        "market_regime_attribution": {
            "strategies": [
                {
                    "strategy_name": "buy_hold",
                    "display_name": "买入持有",
                    "regimes": [
                        {
                            "label": "上涨趋势",
                            "days": 100,
                            "strategy_return": 0.1,
                            "benchmark_return": 0.08,
                            "excess_return": 0.02,
                        }
                    ],
                }
            ]
        },
        "trade_attribution": {
            "strategies": [
                {
                    "strategy_name": "buy_hold",
                    "display_name": "买入持有",
                    "closed_trade_segments": 0,
                    "open_shares": 100,
                    "matched_trades": [],
                }
            ]
        },
        "robustness_assessments": [
            {"strategy_name": "buy_hold", "display_name": "买入持有", "grade": "moderate"}
        ],
        "research_decision": {
            "preferred_display_name": "买入持有",
            "robustness_grade": "moderate",
            "falsification_risks": ["优势可能由少数行情阶段贡献。"],
            "next_experiments": [
                {"question": "弱势阶段能否改善", "method": "滚动回测", "success_criteria": "多数窗口为正"}
            ],
            "deployment_gate": {
                "status": "research_only",
                "message": "继续模拟验证。",
                "checks": {"out_of_sample_positive": True},
            },
        },
    }

    report = _html_report(payload)

    assert "逐策略诊断" in report
    assert "本期为什么好" in report
    assert "本期为什么不好" in report
    assert "净值与回撤路径" in report
    assert "<svg" in report
    assert "数据与执行审计" in report
    assert "研究结论与下一步" in report
    assert "市场阶段归因" in report
    assert "交易级归因" in report
    assert "稳健性证据" in report
    assert "最可能推翻当前建议的证据" in report
    assert "下一轮实验" in report
    assert "部署准入检查" in report
    assert len(report) > 5_000
