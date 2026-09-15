"""EU AI Act daily news agent.

Split into focused modules:

- ``config``        static configuration (env, feeds, keywords, schemas, tunables)
- ``collection``    RSS feed collection + relevance filtering + dedup
- ``fetching``      allowlisted, SSRF-safe article fetching + text extraction
- ``summarization`` LLM summarization + independent verification
- ``publishing``    digest rendering + GitHub Issue publishing
- ``main``          orchestration entrypoint

Names are re-exported here for convenience, but code that monkeypatches in tests
should target the submodule where the name is actually *used* (e.g.
``news_agent.fetching.requests``), because that is where it is looked up.
"""

from .collection import (
    collect_relevant_articles,
    dedup_articles,
    fetch_rss_articles,
    is_relevant,
)
from .config import (
    AI_ACT_KEYWORDS,
    ALLOWED_DOMAINS,
    CATEGORIES,
    KEYWORD_PATTERN,
    MAX_ARTICLE_CHARS,
    MAX_REDIRECTS,
    MAX_SUMMARY_ATTEMPTS,
    RSS_FEEDS,
    SUMMARIZE_TOOL,
    VERIFY_TOOL,
)
from .fetching import (
    fetch_article_content,
    fetch_following_safe_redirects,
    is_allowed_domain,
)
from .publishing import create_github_issue, issue_already_exists_today, render_digest
from .summarization import (
    client,
    generate_summary_payload,
    summarize_article,
    summarize_articles,
    validate_summary_payload,
    validate_verification_payload,
    verify_summary,
)

__all__ = [
    "AI_ACT_KEYWORDS",
    "ALLOWED_DOMAINS",
    "CATEGORIES",
    "KEYWORD_PATTERN",
    "MAX_ARTICLE_CHARS",
    "MAX_REDIRECTS",
    "MAX_SUMMARY_ATTEMPTS",
    "RSS_FEEDS",
    "SUMMARIZE_TOOL",
    "VERIFY_TOOL",
    "client",
    "collect_relevant_articles",
    "create_github_issue",
    "dedup_articles",
    "fetch_article_content",
    "fetch_following_safe_redirects",
    "fetch_rss_articles",
    "generate_summary_payload",
    "is_allowed_domain",
    "is_relevant",
    "issue_already_exists_today",
    "render_digest",
    "summarize_article",
    "summarize_articles",
    "validate_summary_payload",
    "validate_verification_payload",
    "verify_summary",
]
