import json

import pytest
from agentevals.trajectory import atif_to_openai_messages
from agentevals.trajectory import create_trajectory_match_evaluator
from agentevals.types import EvaluatorResult


def _atif_trajectory(steps):
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": "session-1",
        "agent": {"name": "test-agent", "version": "1.0.0"},
        "steps": steps,
    }


def test_atif_to_openai_messages_supports_trajectory_match_evaluator():
    atif_trajectory = _atif_trajectory(
        [
            {"step_id": 1, "source": "user", "message": "What is the weather in SF?"},
            {
                "step_id": 2,
                "source": "agent",
                "message": "(tool use)",
                "tool_calls": [
                    {
                        "tool_call_id": "call_weather",
                        "function_name": "get_weather",
                        "arguments": {"city": "San Francisco"},
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "call_weather",
                            "content": "It's 80 degrees and sunny in SF.",
                        }
                    ]
                },
            },
            {
                "step_id": 3,
                "source": "agent",
                "message": "The weather in SF is 80 degrees and sunny.",
            },
        ]
    )

    messages = atif_to_openai_messages(atif_trajectory)

    assert messages == [
        {"role": "user", "content": "What is the weather in SF?"},
        {
            "role": "assistant",
            "content": "(tool use)",
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "San Francisco"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_weather",
            "content": "It's 80 degrees and sunny in SF.",
        },
        {
            "role": "assistant",
            "content": "The weather in SF is 80 degrees and sunny.",
        },
    ]

    evaluator = create_trajectory_match_evaluator(trajectory_match_mode="strict")
    assert evaluator(outputs=messages, reference_outputs=messages) == EvaluatorResult(
        key="trajectory_strict_match",
        score=True,
        comment=None,
        metadata=None,
    )


def test_atif_to_openai_messages_handles_content_parts_and_missing_observation_ids():
    atif_trajectory = _atif_trajectory(
        [
            {
                "step_id": 1,
                "source": "system",
                "message": [{"type": "text", "text": "You are concise."}],
            },
            {
                "step_id": 2,
                "source": "agent",
                "tool_calls": [
                    {
                        "tool_call_id": "call_search",
                        "function_name": "search",
                        "arguments": {"query": "ATIF"},
                    }
                ],
                "observation": {"results": [{"content": {"answer": "found"}}]},
            },
        ]
    )

    messages = atif_to_openai_messages(atif_trajectory)

    assert messages[0] == {"role": "system", "content": "You are concise."}
    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_calls"][0]["id"] == "call_search"
    assert messages[1]["tool_calls"][0]["function"] == {
        "name": "search",
        "arguments": json.dumps({"query": "ATIF"}),
    }
    assert messages[2] == {
        "role": "tool",
        "tool_call_id": "",
        "content": json.dumps({"answer": "found"}, sort_keys=True),
    }


def test_atif_to_openai_messages_json_encodes_plain_string_tool_arguments():
    messages = atif_to_openai_messages(
        _atif_trajectory(
            [
                {
                    "step_id": 1,
                    "source": "agent",
                    "tool_calls": [
                        {
                            "tool_call_id": "call_lookup",
                            "function_name": "lookup",
                            "arguments": "ATIF",
                        }
                    ],
                }
            ]
        )
    )

    assert messages[0]["tool_calls"][0]["function"] == {
        "name": "lookup",
        "arguments": json.dumps("ATIF"),
    }


def test_atif_to_openai_messages_rejects_unsupported_major_version():
    atif_trajectory = _atif_trajectory(
        [{"step_id": 1, "source": "agent", "message": "done"}]
    )
    atif_trajectory["schema_version"] = "ATIF-v2.0"

    with pytest.raises(ValueError, match="Unsupported ATIF major version"):
        atif_to_openai_messages(atif_trajectory)


def test_atif_to_openai_messages_warns_on_newer_minor_version():
    atif_trajectory = _atif_trajectory(
        [{"step_id": 1, "source": "agent", "message": "done"}]
    )
    atif_trajectory["schema_version"] = "ATIF-v1.8"

    with pytest.warns(UserWarning, match="newer than the latest supported version"):
        assert atif_to_openai_messages(atif_trajectory) == [
            {"role": "assistant", "content": "done"}
        ]


def test_atif_to_openai_messages_rejects_mismatched_observation_ids():
    atif_trajectory = _atif_trajectory(
        [
            {
                "step_id": 1,
                "source": "agent",
                "tool_calls": [
                    {
                        "tool_call_id": "call_lookup",
                        "function_name": "lookup",
                        "arguments": {},
                    }
                ],
                "observation": {
                    "results": [
                        {"source_call_id": "call_other", "content": "result"},
                    ]
                },
            }
        ]
    )

    with pytest.raises(ValueError, match="does not match a tool_call_id"):
        atif_to_openai_messages(atif_trajectory)
