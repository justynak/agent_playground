import json
import os
import sys

from openai import OpenAI

from tools import evaluate_operation

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "evaluate_operation",
            "description": "Evaluates a single arithmetic operation and returns the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "The node id (e.g. 'root.left')."},
                    "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                    "left": {"type": "number"},
                    "right": {"type": "number"},
                },
                "required": ["id", "operation", "left", "right"],
            },
        },
    }
]

SYSTEM_PROMPT = "You are a math operation evaluator. You will be given one arithmetic operation. Call evaluate_operation with the provided inputs. Do not compute arithmetic yourself."


def run(id: str, operation: str, left: float, right: float) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Evaluate: {operation}({left}, {right}), node id: {id}"},
    ]

    for _ in range(5):
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return {"error": f"Sub-agent did not call evaluate_operation for {id}"}

        tc = msg.tool_calls[0]
        args = json.loads(tc.function.arguments)
        print(f"    [sub-agent {id}] evaluate_operation({json.dumps(args)})", file=sys.stderr)
        result = evaluate_operation(args["id"], args["operation"], args["left"], args["right"])
        print(f"    [sub-agent {id}] result: {json.dumps(result)}", file=sys.stderr)
        return result

    return {"error": f"Sub-agent for {id} reached iteration limit"}
