import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from tools import prefilter_syntax, parse_expression


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
        ("5", 5),
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
