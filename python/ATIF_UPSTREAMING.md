# ATIF Trajectory Support for AgentEvals

ATIF, the
[Agent Trajectory Interchange Format](https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md),
is a JSON schema for recording an agent run as ordered user, agent, tool-call,
and observation steps.

This is a reference implementation for letting AgentEvals consume ATIF
trajectories directly.

## Why

Agent runtimes increasingly emit structured trajectory artifacts, but every
evaluator currently needs its own adapter. ATIF gives those runtimes a common
interchange format for agent steps, tool calls, and observations. Supporting it
in AgentEvals lets users bring ATIF traces directly to the existing trajectory
match and LLM-as-judge evaluators.

In Harbor, where the base ATIF reference is maintained, agent harnesses are the
natural ATIF emission point: after a run, the harness can normalize its native
event stream into `agent/trajectory.json`. That makes ATIF a unified trajectory
schema across harnesses such as OpenCode, Hermes, Claude Code, Codex CLI, and
NeMo Agent, independent of the dataset or benchmark being executed.

This keeps evaluation portable:

- Agent harnesses can emit ATIF once instead of maintaining per-evaluator
  conversions.
- AgentEvals users can evaluate traces from non-LangChain runtimes without
  first standing up a separate toolkit.
- AgentEvals can be plugged in as a Harbor verifier for quick trajectory health
  checks across benchmarks and harnesses.
- Deterministic match and LLM judge workflows both work after conversion to the
  OpenAI-style messages AgentEvals already accepts.

Other ecosystem tools are also moving in this direction. For example,
[Phoenix](https://github.com/Arize-ai/phoenix) already includes ATIF export
support, with follow-up work in review to align it with
[ATIF v1.7](https://github.com/Arize-ai/phoenix/pull/13107).

## What This POC Adds

- `atif_to_openai_messages(...)`, a lightweight ATIF-to-OpenAI-message adapter.
- Minimal ATIF `TypedDict` contracts in AgentEvals style.
- Local validation for schema version, required root fields, steps, tool calls,
  and observation/tool-call relationships.
- README examples for ATIF with both trajectory match and reference-free
  LLM-as-judge evaluation.
- Focused unit tests for conversion, validation, and trajectory match
  compatibility.

## Example Conversion

A compact ATIF trajectory:

```json
{
  "schema_version": "ATIF-v1.7",
  "session_id": "weather-session",
  "agent": {
    "name": "weather-agent",
    "version": "1.0.0"
  },
  "steps": [
    {
      "step_id": 1,
      "source": "user",
      "message": "What is the weather in SF, and should I bring a jacket?"
    },
    {
      "step_id": 2,
      "source": "agent",
      "message": "I will check the current weather and rain forecast.",
      "tool_calls": [
        {
          "tool_call_id": "call_weather",
          "function_name": "get_weather",
          "arguments": {
            "city": "San Francisco"
          }
        },
        {
          "tool_call_id": "call_precipitation",
          "function_name": "get_precipitation",
          "arguments": {
            "city": "San Francisco"
          }
        }
      ],
      "observation": {
        "results": [
          {
            "source_call_id": "call_weather",
            "content": "It is 60 degrees in San Francisco."
          },
          {
            "source_call_id": "call_precipitation",
            "content": "There is a 10% chance of rain today."
          }
        ]
      }
    },
    {
      "step_id": 3,
      "source": "agent",
      "message": "I will check whether it is windy.",
      "tool_calls": [
        {
          "tool_call_id": "call_wind",
          "function_name": "get_wind_conditions",
          "arguments": {
            "city": "San Francisco"
          }
        }
      ],
      "observation": {
        "results": [
          {
            "source_call_id": "call_wind",
            "content": "It is windy today."
          }
        ]
      }
    },
    {
      "step_id": 4,
      "source": "agent",
      "message": "It is 60 degrees and windy with little chance of rain, so bring a light jacket."
    }
  ]
}
```

converts to the OpenAI-style messages AgentEvals already accepts:

```json
[
  {
    "role": "user",
    "content": "What is the weather in SF, and should I bring a jacket?"
  },
  {
    "role": "assistant",
    "content": "I will check the current weather and rain forecast.",
    "tool_calls": [
      {
        "id": "call_weather",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"city\": \"San Francisco\"}"
        }
      },
      {
        "id": "call_precipitation",
        "type": "function",
        "function": {
          "name": "get_precipitation",
          "arguments": "{\"city\": \"San Francisco\"}"
        }
      }
    ]
  },
  {
    "role": "tool",
    "tool_call_id": "call_weather",
    "content": "It is 60 degrees in San Francisco."
  },
  {
    "role": "tool",
    "tool_call_id": "call_precipitation",
    "content": "There is a 10% chance of rain today."
  },
  {
    "role": "assistant",
    "content": "I will check whether it is windy.",
    "tool_calls": [
      {
        "id": "call_wind",
        "type": "function",
        "function": {
          "name": "get_wind_conditions",
          "arguments": "{\"city\": \"San Francisco\"}"
        }
      }
    ]
  },
  {
    "role": "tool",
    "tool_call_id": "call_wind",
    "content": "It is windy today."
  },
  {
    "role": "assistant",
    "content": "It is 60 degrees and windy with little chance of rain, so bring a light jacket."
  }
]
```

## Design Notes

The adapter intentionally avoids a dependency on Pydantic or any specific agent
runtime. It accepts plain ATIF JSON-like mappings and converts them to the
message shape AgentEvals already evaluates. That keeps the feature small,
optional, and easy to review.

The goal is not to make AgentEvals own the ATIF schema. The goal is to provide a
stable ingestion path for ATIF trajectories while preserving AgentEvals'
existing evaluator APIs.
