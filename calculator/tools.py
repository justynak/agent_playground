import ast
import re


def prefilter_syntax(expression: str) -> dict:
    if not expression.strip():
        return {"valid": False, "reason": "Expression is empty"}

    if not re.match(r'^[\d\s\+\-\*\/\(\)\.]+$', expression):
        return {"valid": False, "reason": "Invalid characters. Allowed: digits, +, -, *, /, (, ), ."}

    depth = 0
    for char in expression:
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        if depth < 0:
            return {"valid": False, "reason": "Unbalanced parentheses: closing bracket without opening"}

    if depth != 0:
        return {"valid": False, "reason": "Unbalanced parentheses: unclosed opening bracket"}

    return {"valid": True, "reason": None}


def parse_expression(expression: str) -> dict:
    try:
        tree = ast.parse(expression.strip(), mode='eval')
        return {"success": True, "operation_tree": _node_to_dict(tree.body)}
    except (SyntaxError, ValueError) as e:
        return {"success": False, "error": str(e)}


def _node_to_dict(node):
    if isinstance(node, ast.BinOp):
        op_map = {
            ast.Add: "add",
            ast.Sub: "subtract",
            ast.Mult: "multiply",
            ast.Div: "divide",
        }
        op = op_map.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return {
            "operation": op,
            "left": _node_to_dict(node.left),
            "right": _node_to_dict(node.right),
        }
    elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _node_to_dict(node.operand)
        if isinstance(operand, (int, float)):
            return -operand
        return {"operation": "subtract", "left": 0, "right": operand}
    else:
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def evaluate_operation(id: str, operation: str, left: float, right: float) -> dict:
    if operation == "add":
        result = left + right
    elif operation == "subtract":
        result = left - right
    elif operation == "multiply":
        result = left * right
    elif operation == "divide":
        if right == 0:
            return {"error": "Division by zero"}
        result = left / right
    else:
        return {"error": f"Unknown operation: {operation}"}
    return {"id": id, "result": result}


def executable_operations(tree: dict | int | float, completed: dict) -> list:
    result = []
    _collect_executable(tree, "root", completed, result)
    return result


def _collect_executable(node, node_id: str, completed: dict, result: list) -> None:
    if not isinstance(node, dict):
        return
    if node_id in completed:
        return

    left, right = node["left"], node["right"]
    left_id, right_id = f"{node_id}.left", f"{node_id}.right"

    left_resolved = not isinstance(left, dict) or left_id in completed
    right_resolved = not isinstance(right, dict) or right_id in completed

    if left_resolved and right_resolved:
        result.append({
            "id": node_id,
            "operation": node["operation"],
            "left": completed[left_id] if isinstance(left, dict) else left,
            "right": completed[right_id] if isinstance(right, dict) else right,
        })
    else:
        if not left_resolved:
            _collect_executable(left, left_id, completed, result)
        if not right_resolved:
            _collect_executable(right, right_id, completed, result)
