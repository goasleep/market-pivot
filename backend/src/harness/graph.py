"""LangGraph kernel for contract compilation, Skill selection and plan validation."""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, StateGraph

from agents.asset_requests import AssetRequestResolver
from graph.checkpointing import checkpoint_manager
from harness.compiler import harness_task_compiler
from harness.planner import harness_planner, skill_selector
from harness.runtime import get_harness_registry
from models.supervisor import TaskRoutingDecision


class HarnessKernelState(TypedDict, total=False):
    request: dict[str, Any]
    routing: dict[str, Any]
    contract: dict[str, Any]
    selected_skills: list[dict[str, Any]]
    plan: dict[str, Any]
    lifecycle: list[str]
    dynamic_planning: bool


def compile_contract(state: HarnessKernelState) -> dict[str, Any]:
    request = AssetRequestResolver.request_from_payload(state["request"])
    routing = TaskRoutingDecision.model_validate(state["routing"])
    contract = harness_task_compiler.compile(request, routing)
    return {"contract": contract.model_dump(mode="json"), "lifecycle": ["compile"]}


def select_skills(state: HarnessKernelState) -> dict[str, Any]:
    from harness.models import HarnessTaskContract

    contract = HarnessTaskContract.model_validate(state["contract"])
    skills = skill_selector.select(contract, get_harness_registry())
    return {
        "selected_skills": [skill.model_dump(mode="json") for skill in skills],
        "lifecycle": [*state.get("lifecycle", []), "select_skills"],
    }


async def plan_task(state: HarnessKernelState) -> dict[str, Any]:
    from harness.models import HarnessTaskContract, SkillManifest

    contract = HarnessTaskContract.model_validate(state["contract"])
    skills = tuple(SkillManifest.model_validate(item) for item in state.get("selected_skills", []))
    plan = (
        await harness_planner.constrained_plan(contract, skills)
        if state.get("dynamic_planning")
        else harness_planner.deterministic_plan(contract, skills)
    )
    return {"plan": plan.model_dump(mode="json"), "lifecycle": [*state.get("lifecycle", []), "plan"]}


def _stage(name: str):
    def node(state: HarnessKernelState) -> dict[str, Any]:
        return {"lifecycle": [*state.get("lifecycle", []), name]}

    return node


def build_harness_graph(checkpointer: Any | None = None):
    graph = StateGraph(HarnessKernelState)
    graph.add_node("compile", compile_contract)
    graph.add_node("select_skills", select_skills)
    graph.add_node("plan", plan_task)
    for name in ("dispatch", "execute", "verify", "replan", "synthesize", "judge"):
        graph.add_node(name, _stage(name))
    graph.set_entry_point("compile")
    graph.add_edge("compile", "select_skills")
    graph.add_edge("select_skills", "plan")
    graph.add_edge("plan", "dispatch")
    graph.add_edge("dispatch", "execute")
    graph.add_edge("execute", "verify")
    graph.add_edge("verify", "replan")
    graph.add_edge("replan", "synthesize")
    graph.add_edge("synthesize", "judge")
    graph.add_edge("judge", END)
    return graph.compile(checkpointer=checkpointer)


harness_graph = build_harness_graph()


def configure_harness_graph(checkpointer: Any | None) -> None:
    global harness_graph
    harness_graph = build_harness_graph(checkpointer)


async def prepare_harness_plan(
    request_payload: dict[str, Any],
    routing: TaskRoutingDecision,
    *,
    task_id: str | None,
    dynamic_planning: bool = False,
) -> HarnessKernelState:
    config: RunnableConfig = {}
    if task_id and checkpoint_manager.saver is not None:
        config = checkpoint_manager.graph_config(f"{task_id}:kernel", {})
    return await harness_graph.ainvoke(
        {
            "request": request_payload,
            "routing": routing.model_dump(mode="json"),
            "dynamic_planning": dynamic_planning,
        },
        config=config,
    )
