"""EU AI Act daily news agent.

Split into focused modules:

- ``config``        static configuration (env, feeds, keywords, schemas, tunables)
- ``collection``    RSS feed collection + relevance filtering + dedup
- ``fetching``      allowlisted, SSRF-safe article fetching + text extraction
- ``summarization`` LLM summarization + independent verification
- ``publishing``    digest rendering + GitHub Issue publishing
- ``main``          orchestration entrypoint

Import from the submodule you need (e.g. ``from news_agent import fetching``);
this package has no re-exports of its own.
"""
