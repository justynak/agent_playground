"""Pytest bootstrap: stub the env vars `config` requires at import time.

`news_agent.config` reads its environment eagerly, so the secrets must be present
(non-empty placeholders) before any test module imports the package. Real values are
only required for the `integration` tests, which are run separately with `--env-file .env`.

The project root is put on `sys.path` by `pythonpath = ["."]` in `pyproject.toml`,
so this file needs no path manipulation.
"""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test")
os.environ.setdefault("GITHUB_TOKEN", "test")
os.environ.setdefault("GITHUB_REPOSITORY", "test/test")
