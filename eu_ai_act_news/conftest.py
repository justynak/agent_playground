"""Pytest bootstrap: make the project root importable and stub required env vars.

`config` reads its environment at import time, so the secrets must be present (to
non-empty placeholders) before any test module imports the `news_agent` package.
Real values are only required for the `integration` tests, which are run separately
with `--env-file .env`.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

os.environ.setdefault("DEEPSEEK_API_KEY", "test")
os.environ.setdefault("GITHUB_TOKEN", "test")
os.environ.setdefault("GITHUB_REPOSITORY", "test/test")
