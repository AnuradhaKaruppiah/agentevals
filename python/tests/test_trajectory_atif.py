import json

from agentevals.trajectory import atif_to_openai_messages
from agentevals.trajectory import create_trajectory_match_evaluator
from agentevals.types import EvaluatorResult


def test_atif_to_openai_messages_supports_trajectory_match_evaluator():
    atif_trajectory = {
        "schema_version": "ATIF-v1.7",
        "steps": [
            {"source": "user", "message": "What is the weather in SF?"},
            {
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
                "source": "agent",
                "message": "The weather in SF is 80 degrees and sunny.",
            },
        ],
    }

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
    atif_trajectory = {
        "schema_version": "ATIF-v1.7",
        "steps": [
            {
                "source": "system",
                "message": [{"type": "text", "text": "You are concise."}],
            },
            {
                "source": "agent",
                "tool_calls": [
                    {
                        "function_name": "search",
                        "args": {"query": "ATIF"},
                    }
                ],
                "observation": {"results": [{"content": {"answer": "found"}}]},
            },
        ],
    }

    messages = atif_to_openai_messages(atif_trajectory)

    assert messages[0] == {"role": "system", "content": "You are concise."}
    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_calls"][0]["id"] == "call_1"
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
        {
            "steps": [
                {
                    "source": "agent",
                    "tool_calls": [
                        {
                            "function_name": "lookup",
                            "arguments": "ATIF",
                        }
                    ],
                }
            ],
        }
    )

    assert messages[0]["tool_calls"][0]["function"] == {
        "name": "lookup",
        "arguments": json.dumps("ATIF"),
    }
