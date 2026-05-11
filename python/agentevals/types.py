from collections.abc import Callable
from typing import Any
from typing import Literal

from openevals.types import ChatCompletionMessage
from openevals.types import EvaluatorResult
from openevals.types import FewShotExample
from openevals.types import SimpleAsyncEvaluator
from openevals.types import SimpleEvaluator
from typing_extensions import TypedDict


# Trajectory extracted from agent
class GraphTrajectory(TypedDict):
    inputs: list[dict] | None
    results: list[dict]
    steps: list[list[str]]


# Trajectory extracted from a LangGraph thread
class ExtractedLangGraphThreadTrajectory(TypedDict):
    inputs: list
    outputs: GraphTrajectory


ToolArgsMatchMode = Literal["exact", "ignore", "subset", "superset"]

ToolArgsMatchOverrides = dict[str, ToolArgsMatchMode | list[str] | Callable[[dict, dict], bool]]


ATIFSource = Literal["user", "agent", "system"]


class ATIFToolCall(TypedDict):
    tool_call_id: str
    function_name: str
    arguments: dict[str, Any]


class ATIFContentPart(TypedDict, total=False):
    type: str
    text: str
    source: dict[str, Any]


ATIFMessage = str | list[ATIFContentPart]


class ATIFObservationResult(TypedDict, total=False):
    source_call_id: str
    content: ATIFMessage


class ATIFObservation(TypedDict):
    results: list[ATIFObservationResult]


class ATIFStep(TypedDict, total=False):
    step_id: int
    source: ATIFSource
    message: ATIFMessage
    tool_calls: list[ATIFToolCall]
    observation: ATIFObservation


class ATIFAgent(TypedDict, total=False):
    name: str
    version: str
    model_name: str


class ATIFTrajectory(TypedDict, total=False):
    schema_version: str
    session_id: str
    agent: ATIFAgent
    steps: list[ATIFStep]

__all__ = [
    "GraphTrajectory",
    "ATIFAgent",
    "ATIFContentPart",
    "ATIFMessage",
    "ATIFObservation",
    "ATIFObservationResult",
    "ATIFSource",
    "ATIFStep",
    "ATIFToolCall",
    "ATIFTrajectory",
    "ChatCompletionMessage",
    "EvaluatorResult",
    "SimpleEvaluator",
    "SimpleAsyncEvaluator",
    "FewShotExample",
    "ToolArgsMatchMode",
    "ToolArgsMatchOverrides",
]
