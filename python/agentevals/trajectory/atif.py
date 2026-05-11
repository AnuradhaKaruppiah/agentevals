import json
import re
import warnings
from collections.abc import Mapping
from typing import Any

from agentevals.types import ATIFTrajectory
from agentevals.types import ChatCompletionMessage

_MAX_SUPPORTED_MINOR = 7
_SCHEMA_VERSION_PATTERN = re.compile(r"^ATIF-v(?P<major>\d+)\.(?P<minor>\d+)$")
_VALID_SOURCES = {"user", "agent", "system"}


def _raise_validation_errors(errors: list[str]) -> None:
    if errors:
        raise ValueError("Invalid ATIF trajectory:\n" + "\n".join(f"  - {e}" for e in errors))


def _validate_schema_version(schema_version: Any, errors: list[str]) -> None:
    if not isinstance(schema_version, str):
        errors.append("schema_version must be a string in the format 'ATIF-vX.Y'")
        return

    match = _SCHEMA_VERSION_PATTERN.match(schema_version)
    if match is None:
        errors.append(f"Invalid schema_version '{schema_version}': expected format 'ATIF-vX.Y'")
        return

    major = int(match.group("major"))
    minor = int(match.group("minor"))
    if major != 1:
        errors.append(
            f"Unsupported ATIF major version {major} in '{schema_version}': only ATIF v1.x is supported"
        )
    elif minor > _MAX_SUPPORTED_MINOR:
        warnings.warn(
            f"ATIF minor version {minor} in '{schema_version}' is newer than "
            f"the latest supported version (v1.{_MAX_SUPPORTED_MINOR}); "
            "some fields may not be validated or converted",
            stacklevel=3,
        )


def _validate_atif_trajectory(trajectory: Mapping[str, Any]) -> None:
    errors: list[str] = []
    for field in ("schema_version", "session_id", "agent", "steps"):
        if field not in trajectory:
            errors.append(f"Missing required root field: '{field}'")
    _raise_validation_errors(errors)

    _validate_schema_version(trajectory["schema_version"], errors)

    if not isinstance(trajectory["session_id"], str) or not trajectory["session_id"].strip():
        errors.append("session_id must be a non-empty string")

    agent = trajectory["agent"]
    if not isinstance(agent, Mapping):
        errors.append("agent must be a mapping")
    else:
        for field in ("name", "version"):
            value = agent.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"agent.{field} must be a non-empty string")

    steps = trajectory["steps"]
    if not isinstance(steps, list) or not steps:
        errors.append("steps must be a non-empty list")
        _raise_validation_errors(errors)

    for step_index, step in enumerate(steps):
        prefix = f"steps[{step_index}]"
        if not isinstance(step, Mapping):
            errors.append(f"{prefix} must be a mapping")
            continue

        step_id = step.get("step_id")
        expected_step_id = step_index + 1
        if step_id != expected_step_id:
            errors.append(f"{prefix}.step_id must be {expected_step_id}")

        source = step.get("source")
        if source not in _VALID_SOURCES:
            errors.append(f"{prefix}.source must be one of {sorted(_VALID_SOURCES)}")

        message = step.get("message")
        if source in ("user", "system") and message is None:
            errors.append(f"{prefix}.message is required for {source} steps")
        elif message is not None and not isinstance(message, str | list):
            errors.append(f"{prefix}.message must be a string or content part list")

        tool_call_ids = _validate_tool_calls(step, prefix, errors)
        _validate_observation(step, prefix, tool_call_ids, errors)

    _raise_validation_errors(errors)


def _validate_tool_calls(
    step: Mapping[str, Any], prefix: str, errors: list[str]
) -> set[str]:
    tool_call_ids: set[str] = set()
    if "tool_calls" not in step:
        return tool_call_ids

    tool_calls = step["tool_calls"]
    if not isinstance(tool_calls, list):
        errors.append(f"{prefix}.tool_calls must be a list")
        return tool_call_ids

    for tool_call_index, tool_call in enumerate(tool_calls):
        tool_call_prefix = f"{prefix}.tool_calls[{tool_call_index}]"
        if not isinstance(tool_call, Mapping):
            errors.append(f"{tool_call_prefix} must be a mapping")
            continue

        tool_call_id = tool_call.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            errors.append(f"{tool_call_prefix}.tool_call_id must be a non-empty string")
        else:
            tool_call_ids.add(tool_call_id)

        function_name = tool_call.get("function_name")
        if not isinstance(function_name, str) or not function_name:
            errors.append(f"{tool_call_prefix}.function_name must be a non-empty string")

        if "arguments" not in tool_call:
            errors.append(f"{tool_call_prefix}.arguments is required")

    return tool_call_ids


def _validate_observation(
    step: Mapping[str, Any], prefix: str, tool_call_ids: set[str], errors: list[str]
) -> None:
    if "observation" not in step:
        return

    observation = step["observation"]
    if not isinstance(observation, Mapping):
        errors.append(f"{prefix}.observation must be a mapping")
        return

    results = observation.get("results")
    if not isinstance(results, list):
        errors.append(f"{prefix}.observation.results must be a list")
        return

    for result_index, result in enumerate(results):
        result_prefix = f"{prefix}.observation.results[{result_index}]"
        if not isinstance(result, Mapping):
            errors.append(f"{result_prefix} must be a mapping")
            continue

        source_call_id = result.get("source_call_id")
        if isinstance(source_call_id, str) and tool_call_ids and source_call_id not in tool_call_ids:
            errors.append(
                f"{result_prefix}.source_call_id '{source_call_id}' does not match a tool_call_id in this step"
            )


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
    name = tool_call.get("function_name")
    if not name:
        return None

    tool_call_id = tool_call.get("tool_call_id") or f"call_{fallback_index}"
    return {
        "id": str(tool_call_id),
        "type": "function",
        "function": {
            "name": str(name),
            "arguments": _tool_arguments_to_json(tool_call.get("arguments", {})),
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


def atif_to_openai_messages(
    atif_trajectory: ATIFTrajectory | Mapping[str, Any],
) -> list[ChatCompletionMessage]:
    """Convert an ATIF trajectory into OpenAI-compatible chat messages.

    This adapter lets ATIF-producing agent runtimes reuse the existing trajectory
    match and LLM-as-judge evaluators without depending on NAT-specific code.
    """
    trajectory = atif_trajectory.get("trajectory", atif_trajectory)
    if not isinstance(trajectory, Mapping):
        raise ValueError("ATIF trajectory must be a mapping")

    _validate_atif_trajectory(trajectory)
    steps = trajectory["steps"]

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
