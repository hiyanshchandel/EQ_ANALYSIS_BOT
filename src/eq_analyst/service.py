"""Application service: one method per thing a user can do.

The Telegram bot only talks to this class, never to LangGraph directly. That keeps the
bot thin (presentation only) and makes it easy to add another front end later. This
layer also owns cross-cutting concerns: tracing, online scores and user feedback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse, LangfuseAgent, propagate_attributes
from langfuse.api import ScoreDataType

from eq_analyst.bootstrap import AgentRuntime
from eq_analyst.core.config import Settings
from eq_analyst.core.log import get_logger
from eq_analyst.domain.models import AnalysisReport
from eq_analyst.observability.tracing import langchain_handler

log = get_logger(__name__)

TRACE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


@dataclass(frozen=True)
class TurnResult:
    text: str
    report: AnalysisReport | None  # set when this turn produced a new analysis
    trace_id: str | None  # Langfuse trace id, used to attach user feedback later


class AnalystService:
    def __init__(
        self, runtime: AgentRuntime, settings: Settings, langfuse: Langfuse | None
    ) -> None:
        self._runtime = runtime
        self._settings = settings
        self._langfuse = langfuse

    async def handle_message(self, *, chat_id: int, user_id: int, text: str) -> TurnResult:
        config: RunnableConfig = {
            # thread_id selects which saved conversation to load and continue.
            "configurable": {"thread_id": str(chat_id)},
            "run_name": "equity-analyst",
        }
        if self._langfuse is None:
            state = await self._run_graph(text, config)
            return self._to_result(state, trace_id=None)

        # One Langfuse trace per Telegram message. Everything the graph does nests inside.
        with (
            self._langfuse.start_as_current_observation(
                name="telegram-message", as_type="agent", input={"text": text}
            ) as root,
            # session = chat, so Langfuse groups a whole conversation together.
            propagate_attributes(user_id=str(user_id), session_id=str(chat_id), tags=["telegram"]),
        ):
            config["callbacks"] = [langchain_handler(self._settings)]
            try:
                state = await self._run_graph(text, config)
            except Exception as exc:
                root.update(level="ERROR", status_message=str(exc))
                raise
            result = self._to_result(state, trace_id=self._langfuse.get_current_trace_id())
            root.update(output={"text": result.text, "response_kind": state.get("response_kind")})
            self._score_online(root, state)
        return result

    async def reset_conversation(self, chat_id: int) -> None:
        checkpointer = self._runtime.graph.checkpointer
        if checkpointer is not None and not isinstance(checkpointer, bool):
            await checkpointer.adelete_thread(str(chat_id))

    def record_feedback(self, *, trace_id: str, positive: bool, user_id: int) -> None:
        """Thumbs up/down from Telegram becomes a score on that message's trace."""
        if self._langfuse is None or not TRACE_ID_PATTERN.match(trace_id):
            return
        self._langfuse.create_score(
            name="user_feedback",
            value=1 if positive else 0,
            data_type="BOOLEAN",
            trace_id=trace_id,
            comment=f"telegram user {user_id}",
        )

    # --- internals ---------------------------------------------------------------

    async def _run_graph(self, text: str, config: RunnableConfig) -> dict[str, Any]:
        state = await self._runtime.graph.ainvoke(
            {"messages": [HumanMessage(text)]}, config, context=self._runtime.context
        )
        return dict(state)

    @staticmethod
    def _to_result(state: dict[str, Any], trace_id: str | None) -> TurnResult:
        report = state.get("report") if state.get("response_kind") == "report" else None
        return TurnResult(text=state["messages"][-1].text, report=report, trace_id=trace_id)

    @staticmethod
    def _score_online(root: LangfuseAgent, state: dict[str, Any]) -> None:
        """Cheap, deterministic scores on every production trace (online evaluation)."""
        root.score_trace(
            name="response_kind",
            value=str(state.get("response_kind")),
            data_type=ScoreDataType.CATEGORICAL,
        )
        report: AnalysisReport | None = state.get("report")
        if state.get("response_kind") == "report" and report is not None:
            root.score_trace(
                name="risk_level",
                value=report.risk.level.value,
                data_type=ScoreDataType.CATEGORICAL,
            )
            root.score_trace(
                name="guardrail_triggered",
                value=1.0 if report.guardrail_notes else 0.0,
                data_type=ScoreDataType.BOOLEAN,
            )
            root.score_trace(
                name="fundamentals_available",
                value=1.0 if report.fundamentals is not None else 0.0,
                data_type=ScoreDataType.BOOLEAN,
            )
