import json
import os
import sys

from openai import OpenAI

from tools import prefilter_syntax, parse_expression, executable_operations
import sub_agent

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "prefilter_syntax",
            "description": "Pre-filters a mathematical expression for allowed characters and balanced parentheses. Does not guarantee structural validity — call parse_expression afterwards to confirm.",
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
    {
        "type": "function",
        "function": {
            "name": "executable_operations",
            "description": "Given the operation tree and a map of already-completed node results, returns the operations whose inputs are fully resolved and can be evaluated now. Call with an empty completed dict on the first call, then with growing completed after each wave. Returns an empty list when all operations are done. For a bare-number expression, returns [{\"id\": \"root\", \"result\": <number>}] on the first call — add it directly to completed without spawning a sub-agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tree": {
                        "description": "The operation tree returned by parse_expression (nested object or number).",
                    },
                    "completed": {
                        "type": "object",
                        "description": "Map of node id to computed result for all operations evaluated so far.",
                        "additionalProperties": {"type": "number"},
                    },
                },
                "required": ["tree", "completed"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spawn_evaluator_agent",
            "description": "Delegates a single arithmetic operation to a sub-agent for evaluation. The sub-agent computes the result and returns it. Use this for every operation — never compute arithmetic yourself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "The node id from executable_operations (e.g. 'root.left')."},
                    "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                    "left": {"type": "number"},
                    "right": {"type": "number"},
                },
                "required": ["id", "operation", "left", "right"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a math expression evaluator.

When given an expression:
1. Call prefilter_syntax first. Pass the expression EXACTLY as received — do not correct, modify, or complete it in any way.
2. If invalid, report the error and stop.
3. If valid, call parse_expression to get the operation tree.
4. Call executable_operations with the tree and an empty completed dict ({}).
5. For each entry returned by executable_operations:
   - If it has an "operation" key, call spawn_evaluator_agent to delegate it to a sub-agent.
   - If it has a "result" key but no "operation" key, it is pre-resolved — add it directly to completed as {id: result} without spawning.
6. Collect results into completed: add each {"id": ..., "result": ...} entry as {id: result}.
7. Call executable_operations again with the updated completed dict.
8. Repeat steps 5-7 until executable_operations returns an empty list.
9. Report the final result: the value at key "root" in completed.

Never compute arithmetic yourself. Use spawn_evaluator_agent for every operation."""


def dispatch_tool(name: str, args: dict, original_expression: str) -> dict:
    if name in ("prefilter_syntax", "parse_expression"):
        if args["expression"] != original_expression:
            return {
                "error": (
                    f"Expression was modified before {name}. "
                    f"Received: {args['expression']!r}, expected: {original_expression!r}. "
                    "Pass the expression exactly as the user provided it."
                )
            }
    if name == "prefilter_syntax":
        return prefilter_syntax(args["expression"])
    elif name == "parse_expression":
        return parse_expression(args["expression"])
    elif name == "executable_operations":
        return executable_operations(args["tree"], args["completed"])
    elif name == "spawn_evaluator_agent":
        return sub_agent.run(args["id"], args["operation"], args["left"], args["right"])
    return {"error": f"Unknown tool: {name}"}


def run(expression: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Process this expression: {expression}"},
    ]

    for _ in range(20):
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
            try:
                result = dispatch_tool(tc.function.name, args, expression)
            except (KeyError, TypeError) as e:
                result = {"error": f"Malformed tool call arguments: {e}"}
            print(f"  [tool result] {json.dumps(result)}", file=sys.stderr)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

    return "Orchestrator reached iteration limit."
