import json
from collections.abc import Mapping
from typing import Any

from agentevals.types import ChatCompletionMessage


def _json_dumps(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def _tool_arguments_to_json(value: Any) -> str:
    if isinstance(value, str):
        try:
            json.loads(value)
        except json.JSONDecodeError:
            return json.dumps(value)
        return value
    return json.dumps(value, sort_keys=True)


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_content_to_text(item) for item in content)
    if isinstance(content, Mapping):
        text = content.get("text")
        if isinstance(text, str):
            return text
        nested_content = content.get("content")
        if nested_content is not None:
            return _content_to_text(nested_content)
    return _json_dumps(content)


def _tool_call_to_openai(
    tool_call: Mapping[str, Any], fallback_index: int
) -> dict[str, Any] | None:
    name = tool_call.get("function_name") or tool_call.get("name")
    if not name:
        return None

    tool_call_id = (
        tool_call.get("tool_call_id") or tool_call.get("id") or f"call_{fallback_index}"
    )
    return {
        "id": str(tool_call_id),
        "type": "function",
        "function": {
            "name": str(name),
            "arguments": _tool_arguments_to_json(
                tool_call.get("arguments", tool_call.get("args", {}))
            ),
        },
    }


def _observation_results(step: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    observation = step.get("observation")
    if not isinstance(observation, Mapping):
        return []
    results = observation.get("results")
    if not isinstance(results, list):
        return []
    return [result for result in results if isinstance(result, Mapping)]


def atif_to_openai_messages(atif_trajectory: Mapping[str, Any]) -> list[ChatCompletionMessage]:
    """Convert an ATIF trajectory into OpenAI-compatible chat messages.

    This adapter lets ATIF-producing agent runtimes reuse the existing trajectory
    match and LLM-as-judge evaluators without depending on NAT-specific code.
    """
    trajectory = atif_trajectory.get("trajectory", atif_trajectory)
    if not isinstance(trajectory, Mapping):
        raise ValueError("ATIF trajectory must be a mapping")

    steps = trajectory.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("ATIF trajectory must contain a list of steps")

    messages: list[ChatCompletionMessage] = []
    for step_index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            continue

        source = step.get("source")
        content = _content_to_text(step.get("message"))
        openai_tool_calls = []
        for tool_call_index, tool_call in enumerate(step.get("tool_calls", [])):
            if not isinstance(tool_call, Mapping):
                continue
            openai_tool_call = _tool_call_to_openai(
                tool_call, step_index + tool_call_index
            )
            if openai_tool_call is not None:
                openai_tool_calls.append(openai_tool_call)

        if source == "user":
            messages.append(ChatCompletionMessage(role="user", content=content))
        elif source == "system":
            if content:
                messages.append(ChatCompletionMessage(role="system", content=content))
        elif source == "agent":
            if content or openai_tool_calls:
                message = ChatCompletionMessage(role="assistant", content=content)
                if openai_tool_calls:
                    message["tool_calls"] = openai_tool_calls  # type: ignore
                messages.append(message)
        elif content:
            messages.append(ChatCompletionMessage(role="system", content=content))

        for result in _observation_results(step):
            messages.append(
                ChatCompletionMessage(
                    role="tool",
                    tool_call_id=str(
                        result.get("source_call_id") or result.get("tool_call_id") or ""
                    ),
                    content=_content_to_text(result.get("content")),
                )
            )

    return messages


__all__ = ["atif_to_openai_messages"]
