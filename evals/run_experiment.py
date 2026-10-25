"""Run the golden dataset through the real agent and record an experiment in Langfuse.

    python -m evals.run_experiment            # needs the MCP server running + API keys
    python -m evals.run_experiment --min-score 0.8   # exit 1 below the bar (CI gate)

Every case becomes a trace in Langfuse, scores attach to it, and the run appears under
Experiments so runs (e.g. before and after a prompt change) can be compared side by side.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from typing import Any, cast

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langfuse.experiment import TaskFunction

from eq_analyst.agent.checkpoint import memory_checkpointer
from eq_analyst.bootstrap import build_agent_runtime
from eq_analyst.core.config import get_settings
from eq_analyst.core.log import configure_logging
from eq_analyst.observability.tracing import init_langfuse, langchain_handler
from evals.dataset import EVAL_CASES
from evals.evaluators import ITEM_EVALUATORS, average_scores

GATED_SCORES = ("avg_intent_correct", "avg_symbol_correct", "avg_policy_compliant")


async def run_case(*, item: dict[str, Any], **_: Any) -> dict[str, Any]:
    """The task under test: play every turn of a conversation, return the last result."""
    settings = get_settings()
    # Built per case: each case runs on a fresh in-memory conversation.
    runtime = await build_agent_runtime(settings, memory_checkpointer(), init_langfuse(settings))
    config: RunnableConfig = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "callbacks": [langchain_handler(settings)],
    }
    state: dict[str, Any] = {}
    for turn in item["input"]["turns"]:
        state = dict(
            await runtime.graph.ainvoke(
                {"messages": [HumanMessage(turn)]}, config, context=runtime.context
            )
        )

    ticker = state.get("ticker")
    report = state.get("report") if state.get("response_kind") == "report" else None
    return {
        "intent": state["intent"].kind,
        "symbol": ticker.symbol if ticker else None,
        "horizon": str(state.get("horizon")),
        "reply": state["messages"][-1].text,
        "report": report.model_dump(mode="json") if report else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-score", type=float, default=None, help="fail below this average")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging("WARNING", settings.log_json)
    langfuse = init_langfuse(settings)
    if langfuse is None:
        print("Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to run experiments.")
        return 2

    result = langfuse.run_experiment(
        name="equity-analyst-golden-set",
        description="Intent routing, ticker resolution, policy compliance and faithfulness.",
        data=EVAL_CASES,
        task=cast(TaskFunction, run_case),
        evaluators=ITEM_EVALUATORS,
        run_evaluators=[average_scores],
        max_concurrency=2,  # stay under Gemini and Yahoo rate limits
        metadata={"model": settings.gemini_model},
    )
    print(result.format())
    langfuse.flush()

    if args.min_score is not None:
        averages = {e.name: e.value for e in result.run_evaluations}
        failing = {
            name: value
            for name, value in averages.items()
            if name in GATED_SCORES and isinstance(value, float) and value < args.min_score
        }
        if failing:
            print(f"FAILED quality gate (< {args.min_score}): {failing}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
