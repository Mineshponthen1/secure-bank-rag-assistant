"""Make autoevals work with local Ollama judges.

autoevals expects the judge to reply with a structured "tool call". Local models
served by Ollama often write the JSON as plain text instead, sometimes inside a
markdown code block, and sometimes as a bare list without its field name.
This patch reads the JSON from the text, adds the missing field name when the
form has exactly one list field, and turns it into a proper tool call.
"""
import json

import autoevals.ragas as ragas

_original_run = ragas.run_cached_request


def _extract_json(text):
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise ValueError(f"judge replied without any JSON: {text[:150]!r}")
    end = max(text.rfind("}"), text.rfind("]"))
    return json.loads(text[min(starts):end + 1])


def _wrap_for_tool(data, tools):
    """If the judge sent a bare list, put it under the form's single list field."""
    if isinstance(data, dict):
        return data
    params = tools[0]["function"]["parameters"]
    list_fields = [k for k, v in params.get("properties", {}).items() if v.get("type") == "array"]
    return {list_fields[0]: data} if len(list_fields) == 1 else data


def run_cached_request_with_repair(*args, **kwargs):
    response = _original_run(*args, **kwargs)
    message = response["choices"][0]["message"]
    tools = kwargs.get("tools")
    if message.get("tool_calls") or not tools:
        return response
    data = _wrap_for_tool(_extract_json(message.get("content") or ""), tools)
    message["tool_calls"] = [{
        "type": "function",
        "function": {"name": tools[0]["function"]["name"], "arguments": json.dumps(data)},
    }]
    return response


ragas.run_cached_request = run_cached_request_with_repair
