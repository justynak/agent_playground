import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from tools import prefilter_syntax, parse_expression, executable_operations, evaluate_operation


@pytest.mark.parametrize(
    "expression,expected_valid",
    [
        # Valid
        ("(2*3)+5", True),
        ("(2)-(5)", True),
        ("34353.32/204.3", True),
        ("42", True),
        ("  3 + 4 ", True),
        # "()()" has balanced parens and only allowed chars — prefilter passes it.
        # Structural validity is parse_expression's responsibility.
        ("()()", True),
        # Invalid
        ("", False),
        ("   ", False),
        ("2+(3", False),
        (")(6+7.4)", False),
        ("abc*2.5", False),
        ("2^3", False),
        ("2%3", False),
    ],
)
def test_prefilter_syntax(expression, expected_valid):
    result = prefilter_syntax(expression)
    assert result["valid"] is expected_valid
    if not expected_valid:
        assert result["reason"] is not None
    else:
        assert result["reason"] is None


@pytest.mark.parametrize(
    "expression,expected_tree",
    [
        ("2+3", {"operation": "add", "left": 2, "right": 3}),
        ("10-4", {"operation": "subtract", "left": 10, "right": 4}),
        ("3*4", {"operation": "multiply", "left": 3, "right": 4}),
        ("8/2", {"operation": "divide", "left": 8.0, "right": 2}),
        ("3.14*2", {"operation": "multiply", "left": 3.14, "right": 2}),
        (
            "(2+3)*4",
            {
                "operation": "multiply",
                "left": {"operation": "add", "left": 2, "right": 3},
                "right": 4,
            },
        ),
        (
            "1+2*3",
            {
                "operation": "add",
                "left": 1,
                "right": {"operation": "multiply", "left": 2, "right": 3},
            },
        ),
        ("-5+3", {"operation": "add", "left": -5, "right": 3}),
        # Negative number as right operand
        ("4*-2", {"operation": "multiply", "left": 4, "right": -2}),
        # Negative number inside a subexpression
        (
            "(-3+4)*-2",
            {
                "operation": "multiply",
                "left": {"operation": "add", "left": -3, "right": 4},
                "right": -2,
            },
        ),
        ("5", 5),
        # Negation of a compound sub-expression — must produce a subtract node, not crash
        (
            "-(3+5)",
            {
                "operation": "subtract",
                "left": 0,
                "right": {"operation": "add", "left": 3, "right": 5},
            },
        ),
    ],
)
def test_parse_expression_success(expression, expected_tree):
    result = parse_expression(expression)
    assert result["success"] is True
    assert result["operation_tree"] == expected_tree


@pytest.mark.parametrize(
    "expression",
    [
        # Passes prefilter (allowed chars, balanced parens) but is not valid Python AST
        "()()",
        # ** uses only allowed chars but maps to ast.Pow — unsupported operator
        "2**3",
        # Incomplete expressions
        "2+",
        "/4",
    ],
)
def test_parse_expression_failure(expression):
    result = parse_expression(expression)
    assert result["success"] is False
    assert "error" in result


# Trees used across get_ready_operations tests
_TREE_SIMPLE = {"operation": "add", "left": 2, "right": 3}

_TREE_ONE_DEEP = {
    "operation": "multiply",
    "left": {"operation": "add", "left": 2, "right": 3},
    "right": 4,
}

# (2+1)*3 + (3-1)*2
_TREE_TWO_BRANCHES = {
    "operation": "add",
    "left": {
        "operation": "multiply",
        "left": {"operation": "add", "left": 2, "right": 1},
        "right": 3,
    },
    "right": {
        "operation": "multiply",
        "left": {"operation": "subtract", "left": 3, "right": 1},
        "right": 2,
    },
}


@pytest.mark.parametrize(
    "tree,completed,expected_ready",
    [
        # Leaf number — no operations to run
        (
            5,
            {},
            [],
        ),
        # Both children are numbers — root is immediately ready
        (
            _TREE_SIMPLE,
            {},
            [{"id": "root", "operation": "add", "left": 2, "right": 3}],
        ),
        # Already completed — nothing left
        (
            _TREE_SIMPLE,
            {"root": 5},
            [],
        ),
        # One level deep: inner add is ready, outer multiply is not
        (
            _TREE_ONE_DEEP,
            {},
            [{"id": "root.left", "operation": "add", "left": 2, "right": 3}],
        ),
        # Inner add completed — outer multiply is now ready
        (
            _TREE_ONE_DEEP,
            {"root.left": 5},
            [{"id": "root", "operation": "multiply", "left": 5, "right": 4}],
        ),
        # Two branches: both innermost adds are ready in parallel
        (
            _TREE_TWO_BRANCHES,
            {},
            [
                {"id": "root.left.left", "operation": "add", "left": 2, "right": 1},
                {"id": "root.right.left", "operation": "subtract", "left": 3, "right": 1},
            ],
        ),
        # Wave 2: both multiplications ready
        (
            _TREE_TWO_BRANCHES,
            {"root.left.left": 3, "root.right.left": 2},
            [
                {"id": "root.left", "operation": "multiply", "left": 3, "right": 3},
                {"id": "root.right", "operation": "multiply", "left": 2, "right": 2},
            ],
        ),
        # Wave 3: final addition ready
        (
            _TREE_TWO_BRANCHES,
            {"root.left.left": 3, "root.right.left": 2, "root.left": 9, "root.right": 4},
            [{"id": "root", "operation": "add", "left": 9, "right": 4}],
        ),
        # All completed — nothing left
        (
            _TREE_TWO_BRANCHES,
            {"root.left.left": 3, "root.right.left": 2, "root.left": 9, "root.right": 4, "root": 13},
            [],
        ),
    ],
)
def test_executable_operations(tree, completed, expected_ready):
    assert executable_operations(tree, completed) == expected_ready


@pytest.mark.parametrize(
    "id,operation,left,right,expected",
    [
        ("root", "add", 2, 3, {"id": "root", "result": 5}),
        ("root", "subtract", 10, 4, {"id": "root", "result": 6}),
        ("root", "multiply", 3, 7, {"id": "root", "result": 21}),
        ("root", "divide", 8, 2, {"id": "root", "result": 4.0}),
        # Negative operands
        ("root.left", "add", -3, 4, {"id": "root.left", "result": 1}),
        ("root.left", "multiply", 1, -2, {"id": "root.left", "result": -2}),
        # Division by zero
        ("root", "divide", 5, 0, {"error": "Division by zero"}),
    ],
)
def test_evaluate_operation(id, operation, left, right, expected):
    assert evaluate_operation(id, operation, left, right) == expected
