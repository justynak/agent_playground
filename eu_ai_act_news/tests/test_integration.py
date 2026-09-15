import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("DEEPSEEK_API_KEY", "test")  # prevent import-time crash; real key required at runtime
os.environ.setdefault("GITHUB_TOKEN", "test")
os.environ.setdefault("GITHUB_REPOSITORY", "test/test")

import pytest

from news_agent import (
    CATEGORIES,
    collect_relevant_articles,
    render_digest,
    summarize_articles,
)

# Bounds cost/runtime of a real run (each article can cost up to MAX_SUMMARY_ATTEMPTS
# generate+verify LLM round trips) while still exercising the full pipeline for real.
MAX_ARTICLES_TO_VERIFY = 2

# Wider than the daily digest's 24h window: this test is triggered manually and on-demand
# (not on a schedule), so it needs a good chance of finding *something* to verify on any
# given day rather than depending on EU AI Act news having broken in the last 24 hours.
INTEGRATION_TEST_DAYS_BACK = 7


@pytest.mark.integration
def test_pipeline_end_to_end():
    """Runs feed collection, real LLM summarization+verification, and rendering together.

    Does not call create_github_issue — this only proves the pipeline produces a sane,
    verified digest from live feeds and a live model, it doesn't publish anything.
    """
    articles = collect_relevant_articles(days_back=INTEGRATION_TEST_DAYS_BACK)
    if not articles:
        pytest.skip(f"No relevant EU AI Act articles in the last {INTEGRATION_TEST_DAYS_BACK} days to verify against.")

    summaries = summarize_articles(articles[:MAX_ARTICLES_TO_VERIFY])
    digest = render_digest(summaries)

    if not summaries:
        pytest.skip("All sampled articles were rejected by verification or failed to fetch — nothing to render.")

    assert digest is not None
    sampled_urls = {a["url"] for a in articles[:MAX_ARTICLES_TO_VERIFY]}
    for item in summaries:
        assert item["url"] in sampled_urls
        assert item["category"] in CATEGORIES
        assert item["summary"].strip()
        assert f"## {item['category']}" in digest
