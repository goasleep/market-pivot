from typing import Any


def compare_expression(
    indicator: str,
    operator: str,
    value: float | list[float],
    window: int | None = None,
) -> dict[str, Any]:
    left: dict[str, Any] = {"type": "indicator", "indicator": indicator}
    if window is not None:
        left["window"] = window
    return {
        "type": "compare",
        "left": left,
        "operator": operator,
        "right": {"type": "constant", "value": value},
    }


def strategy_mapping(
    name: str,
    *,
    asset_type: str = "etf",
    entry: dict[str, Any] | None = None,
    exit: dict[str, Any] | None = None,
    max_exposure: float = 0.95,
    **fields: Any,
) -> dict[str, Any]:
    entry = entry or compare_expression("close", "gt", 0)
    components = [
        {
            "id": "entry",
            "type": "dsl",
            "expression": entry,
            "score_when_true": 1,
            "score_when_false": 0 if exit else -1,
        }
    ]
    if exit is not None:
        components.append(
            {
                "id": "exit",
                "type": "dsl",
                "expression": exit,
                "score_when_true": -1,
                "score_when_false": 0,
            }
        )
    payload = {
        "name": name,
        "asset_types": [asset_type],
        "components": components,
        "fusion": {
            "type": "priority",
            "entry_threshold": 0.25,
            "exit_threshold": -0.25 if exit else 0.05,
            "conflict_policy": "exit",
        },
        "position_policy": {"mode": "continuous", "max_exposure": max_exposure},
        **fields,
    }
    return payload
