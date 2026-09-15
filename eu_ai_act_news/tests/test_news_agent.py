import json
import os
import sys
from unittest.mock import MagicMock, patch

# news_agent reads env vars at module level — stub them before import
os.environ.setdefault("DEEPSEEK_API_KEY", "test")
os.environ.setdefault("GITHUB_TOKEN", "test")
os.environ.setdefault("GITHUB_REPOSITORY", "test/test")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import news_agent
from news_agent import (
    CATEGORIES,
    _is_allowed_domain,
    _is_relevant,
    _validate_summary_payload,
    collect_relevant_articles,
    dedup_articles,
    render_digest,
    summarize_article,
    summarize_articles,
)


class TestIsRelevant:
    def test_keyword_in_title(self):
        assert _is_relevant("EU AI Act enters into force", "") is True

    def test_keyword_in_summary(self):
        assert _is_relevant("New rules published", "GPAI models face stricter requirements") is True

    def test_no_keyword_match(self):
        assert _is_relevant("Football match results", "Arsenal won 3-0") is False

    def test_empty_strings(self):
        assert _is_relevant("", "") is False

    def test_case_insensitive(self):
        assert _is_relevant("AI REGULATION UPDATE", "") is True


class TestIsAllowedDomain:
    def test_exact_match(self):
        assert _is_allowed_domain("https://euractiv.com/article") is True

    def test_www_prefix_stripped(self):
        assert _is_allowed_domain("https://www.euractiv.com/article") is True

    def test_subdomain_allowed(self):
        assert _is_allowed_domain("https://sub.bbc.co.uk/news") is True

    def test_blocked_domain(self):
        assert _is_allowed_domain("https://evil.com/article") is False

    def test_domain_spoofing_attempt(self):
        # euractiv.com.evil.com should not pass
        assert _is_allowed_domain("https://euractiv.com.evil.com/article") is False


def _llm_response_with_arguments(arguments: dict) -> MagicMock:
    response = MagicMock()
    tool_call = MagicMock()
    tool_call.function.arguments = json.dumps(arguments)
    response.choices[0].message.tool_calls = [tool_call]
    return response


class TestDedupArticles:
    def test_removes_duplicate_urls(self):
        articles = [{"url": "a"}, {"url": "b"}, {"url": "a"}]
        assert dedup_articles(articles) == [{"url": "a"}, {"url": "b"}]

    def test_preserves_first_occurrence_order(self):
        articles = [{"url": "b"}, {"url": "a"}, {"url": "b"}]
        assert [a["url"] for a in dedup_articles(articles)] == ["b", "a"]

    def test_skips_articles_without_url(self):
        articles = [{"url": ""}, {"title": "no url key"}, {"url": "a"}]
        assert dedup_articles(articles) == [{"url": "a"}]

    def test_empty_list(self):
        assert dedup_articles([]) == []


class TestValidateSummaryPayload:
    def test_valid_payload_is_normalized(self):
        payload = {"category": CATEGORIES[0], "summary": "  A summary.  "}
        assert _validate_summary_payload(payload) == {"category": CATEGORIES[0], "summary": "A summary."}

    def test_rejects_unknown_category(self):
        assert _validate_summary_payload({"category": "Not A Real Category", "summary": "text"}) is None

    def test_rejects_missing_category(self):
        assert _validate_summary_payload({"summary": "text"}) is None

    def test_rejects_blank_summary(self):
        assert _validate_summary_payload({"category": CATEGORIES[0], "summary": "   "}) is None

    def test_rejects_non_string_summary(self):
        assert _validate_summary_payload({"category": CATEGORIES[0], "summary": 123}) is None


class TestRenderDigest:
    def test_empty_summaries_returns_none(self):
        assert render_digest([]) is None

    def test_groups_sections_in_fixed_category_order(self):
        summaries = [
            {"title": "B", "url": "https://x/b", "category": CATEGORIES[2], "summary": "sum b"},
            {"title": "A", "url": "https://x/a", "category": CATEGORIES[0], "summary": "sum a"},
        ]
        digest = render_digest(summaries)
        assert digest.index(f"## {CATEGORIES[0]}") < digest.index(f"## {CATEGORIES[2]}")
        assert "sum a" in digest and "sum b" in digest

    def test_omits_categories_with_no_items(self):
        summaries = [{"title": "A", "url": "https://x/a", "category": CATEGORIES[0], "summary": "sum"}]
        digest = render_digest(summaries)
        for category in CATEGORIES[1:]:
            assert f"## {category}" not in digest

    def test_sources_section_lists_every_item(self):
        summaries = [
            {"title": "A", "url": "https://x/a", "category": CATEGORIES[0], "summary": "sum a"},
            {"title": "B", "url": "https://x/b", "category": CATEGORIES[1], "summary": "sum b"},
        ]
        digest = render_digest(summaries)
        assert "## Sources" in digest
        assert "[A](https://x/a)" in digest
        assert "[B](https://x/b)" in digest


class TestCollectRelevantArticles:
    def test_dedupes_across_feeds_and_queries_every_feed(self):
        shared = {"url": "https://example.com/a", "title": "Shared"}
        with patch("news_agent.fetch_rss_articles", return_value={"relevant_articles": [shared]}) as mock_fetch:
            articles = collect_relevant_articles()
        assert articles == [shared]
        assert mock_fetch.call_count == len(news_agent.RSS_FEEDS)

    def test_no_relevant_articles_returns_empty_list(self):
        with patch("news_agent.fetch_rss_articles", return_value={"relevant_articles": []}):
            assert collect_relevant_articles() == []


class TestSummarizeArticle:
    def test_skips_article_when_content_fetch_fails(self):
        article = {"title": "T", "url": "https://example.com/a"}
        with patch("news_agent.fetch_article_content", return_value={"error": "boom"}):
            assert summarize_article(article) is None

    def test_returns_structured_summary_on_success(self):
        article = {"title": "T", "url": "https://example.com/a"}
        response = _llm_response_with_arguments({"category": CATEGORIES[0], "summary": "A summary."})
        with patch("news_agent.fetch_article_content", return_value={"content": "full text", "url": article["url"]}), \
                patch("news_agent.client") as mock_client:
            mock_client.chat.completions.create.return_value = response
            result = summarize_article(article)
        assert result == {"title": "T", "url": article["url"], "category": CATEGORIES[0], "summary": "A summary."}

    def test_returns_none_when_model_picks_invalid_category(self):
        article = {"title": "T", "url": "https://example.com/a"}
        response = _llm_response_with_arguments({"category": "Not Real", "summary": "A summary."})
        with patch("news_agent.fetch_article_content", return_value={"content": "full text", "url": article["url"]}), \
                patch("news_agent.client") as mock_client:
            mock_client.chat.completions.create.return_value = response
            assert summarize_article(article) is None

    def test_returns_none_when_no_tool_call_is_made(self):
        article = {"title": "T", "url": "https://example.com/a"}
        response = MagicMock()
        response.choices[0].message.tool_calls = []
        with patch("news_agent.fetch_article_content", return_value={"content": "full text", "url": article["url"]}), \
                patch("news_agent.client") as mock_client:
            mock_client.chat.completions.create.return_value = response
            assert summarize_article(article) is None

    def test_returns_none_on_api_error(self):
        article = {"title": "T", "url": "https://example.com/a"}
        with patch("news_agent.fetch_article_content", return_value={"content": "full text", "url": article["url"]}), \
                patch("news_agent.client") as mock_client:
            mock_client.chat.completions.create.side_effect = RuntimeError("network down")
            assert summarize_article(article) is None


class TestSummarizeArticles:
    def test_keeps_successes_and_skips_failures(self):
        articles = [
            {"title": "Good", "url": "https://example.com/good"},
            {"title": "Bad", "url": "https://example.com/bad"},
        ]

        def fake_summarize(article):
            if article["title"] != "Good":
                return None
            return {"title": "Good", "url": article["url"], "category": CATEGORIES[0], "summary": "ok"}

        with patch("news_agent.summarize_article", side_effect=fake_summarize):
            result = summarize_articles(articles)

        assert result == [{"title": "Good", "url": "https://example.com/good", "category": CATEGORIES[0], "summary": "ok"}]

    def test_empty_input_returns_empty_list(self):
        assert summarize_articles([]) == []
