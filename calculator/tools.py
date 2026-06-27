import ast
import re


def validate_expression(expression: str) -> dict:
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
        return -_node_to_dict(node.operand)
    else:
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")
