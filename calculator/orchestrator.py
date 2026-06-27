import json
import os
import sys

from openai import OpenAI

from tools import validate_expression, parse_expression

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "validate_expression",
            "description": "Validates a mathematical expression for allowed characters and balanced parentheses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "The mathematical expression to validate"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "parse_expression",
            "description": "Parses a mathematical expression into an operation tree showing the structure and order of operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "The mathematical expression to parse"},
                },
                "required": ["expression"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a math expression orchestrator.

When given an expression:
1. Call validate_expression first. Pass the expression EXACTLY as received — do not correct, modify, or complete it in any way.
2. If invalid, report the error and stop.
3. If valid, call parse_expression to get the operation tree.
4. Show the operation tree to the user.

Do not compute the result yourself. Your job is validation and parsing only."""


def dispatch_tool(name: str, args: dict) -> dict:
    if name == "validate_expression":
        return validate_expression(args["expression"])
    elif name == "parse_expression":
        return parse_expression(args["expression"])
    return {"error": f"Unknown tool: {name}"}


def run(expression: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Process this expression: {expression}"},
    ]

    for _ in range(10):
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return msg.content or "No output."

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            print(f"  [tool call] {tc.function.name}({json.dumps(args)})", file=sys.stderr)
            result = dispatch_tool(tc.function.name, args)
            print(f"  [tool result] {json.dumps(result)}", file=sys.stderr)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

    return "Orchestrator reached iteration limit."
