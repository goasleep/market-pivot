import io
import zipfile

from application.comparison_artifacts import _workbook


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
        "conclusion": {},
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
        assert "选定行情数据" in workbook_xml
        assert "COUNTA" in worksheet_xml
        assert "drawings/drawing" in " ".join(archive.namelist())
