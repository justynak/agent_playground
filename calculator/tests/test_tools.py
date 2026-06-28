import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import prefilter_syntax


class TestPrefilterSyntaxValid:
    def test_nested_parens_with_multiple_ops(self):
        result = prefilter_syntax("(2*3)+5")
        assert result["valid"] is True

    def test_parenthesized_operands(self):
        result = prefilter_syntax("(2)-(5)")
        assert result["valid"] is True

    def test_floating_point_division(self):
        result = prefilter_syntax("34353.32/204.3")
        assert result["valid"] is True

    def test_empty_parens_pairs_pass_prefilter(self):
        # "()()" has balanced parens and allowed chars — prefilter_syntax returns valid.
        # Structural validity is parse_expression's responsibility.
        result = prefilter_syntax("()()")
        assert result["valid"] is True


class TestPrefilterSyntaxInvalid:
    def test_unclosed_paren(self):
        result = prefilter_syntax("2+(3")
        assert result["valid"] is False

    def test_closing_before_opening(self):
        result = prefilter_syntax(")(6+7.4)")
        assert result["valid"] is False

    def test_invalid_characters(self):
        result = prefilter_syntax("abc*2.5")
        assert result["valid"] is False

    def test_empty_string(self):
        result = prefilter_syntax("")
        assert result["valid"] is False
