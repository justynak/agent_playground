import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("DEEPSEEK_API_KEY", "test")  # prevent import-time crash; real key required at runtime

import pytest
from orchestrator import run


@pytest.mark.integration
def test_happy_path():
    result = run("(33*34)-50*2")
    assert "1022" in result
