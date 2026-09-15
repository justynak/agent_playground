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
    _generate_summary_payload,
    _is_allowed_domain,
    _is_relevant,
    _validate_summary_payload,
    _validate_verification_payload,
    collect_relevant_articles,
    dedup_articles,
    render_digest,
    summarize_article,
    summarize_articles,
    verify_summary,
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


class TestGenerateSummaryPayload:
    def test_returns_validated_payload_on_success(self):
        article = {"title": "T", "url": "https://example.com/a"}
        response = _llm_response_with_arguments({"category": CATEGORIES[0], "summary": "A summary."})
        with patch("news_agent.client") as mock_client:
            mock_client.chat.completions.create.return_value = response
            result = _generate_summary_payload(article, "full text")
        assert result == {"category": CATEGORIES[0], "summary": "A summary."}

    def test_returns_none_when_model_picks_invalid_category(self):
        article = {"title": "T", "url": "https://example.com/a"}
        response = _llm_response_with_arguments({"category": "Not Real", "summary": "A summary."})
        with patch("news_agent.client") as mock_client:
            mock_client.chat.completions.create.return_value = response
            assert _generate_summary_payload(article, "full text") is None

    def test_returns_none_when_no_tool_call_is_made(self):
        article = {"title": "T", "url": "https://example.com/a"}
        response = MagicMock()
        response.choices[0].message.tool_calls = []
        with patch("news_agent.client") as mock_client:
            mock_client.chat.completions.create.return_value = response
            assert _generate_summary_payload(article, "full text") is None

    def test_returns_none_on_api_error(self):
        article = {"title": "T", "url": "https://example.com/a"}
        with patch("news_agent.client") as mock_client:
            mock_client.chat.completions.create.side_effect = RuntimeError("network down")
            assert _generate_summary_payload(article, "full text") is None

    def test_feedback_is_included_in_the_prompt(self):
        article = {"title": "T", "url": "https://example.com/a"}
        response = _llm_response_with_arguments({"category": CATEGORIES[0], "summary": "Corrected."})
        with patch("news_agent.client") as mock_client:
            mock_client.chat.completions.create.return_value = response
            _generate_summary_payload(article, "full text", feedback=["the deadline is Q3 2026"])
        _, kwargs = mock_client.chat.completions.create.call_args
        user_message = kwargs["messages"][1]["content"]
        assert "the deadline is Q3 2026" in user_message


class TestValidateVerificationPayload:
    def test_valid_faithful_payload(self):
        payload = {"faithful": True, "unsupported_claims": []}
        assert _validate_verification_payload(payload) == payload

    def test_valid_unfaithful_payload_with_claims(self):
        payload = {"faithful": False, "unsupported_claims": ["wrong date"]}
        assert _validate_verification_payload(payload) == payload

    def test_rejects_non_bool_faithful(self):
        assert _validate_verification_payload({"faithful": "yes", "unsupported_claims": []}) is None

    def test_rejects_non_list_claims(self):
        assert _validate_verification_payload({"faithful": False, "unsupported_claims": "wrong date"}) is None

    def test_rejects_claims_with_non_string_items(self):
        assert _validate_verification_payload({"faithful": False, "unsupported_claims": [1, 2]}) is None


class TestVerifySummary:
    def test_returns_faithful_verdict(self):
        response = _llm_response_with_arguments({"faithful": True, "unsupported_claims": []})
        with patch("news_agent.client") as mock_client:
            mock_client.chat.completions.create.return_value = response
            assert verify_summary("full text", "a summary") == {"faithful": True, "unsupported_claims": []}

    def test_returns_unfaithful_verdict_with_claims(self):
        response = _llm_response_with_arguments({"faithful": False, "unsupported_claims": ["wrong date"]})
        with patch("news_agent.client") as mock_client:
            mock_client.chat.completions.create.return_value = response
            result = verify_summary("full text", "a summary")
        assert result == {"faithful": False, "unsupported_claims": ["wrong date"]}

    def test_returns_none_on_malformed_response(self):
        response = MagicMock()
        response.choices[0].message.tool_calls = []
        with patch("news_agent.client") as mock_client:
            mock_client.chat.completions.create.return_value = response
            assert verify_summary("full text", "a summary") is None

    def test_returns_none_on_api_error(self):
        with patch("news_agent.client") as mock_client:
            mock_client.chat.completions.create.side_effect = RuntimeError("network down")
            assert verify_summary("full text", "a summary") is None


class TestSummarizeArticle:
    def test_skips_article_when_content_fetch_fails(self):
        article = {"title": "T", "url": "https://example.com/a"}
        with patch("news_agent.fetch_article_content", return_value={"error": "boom"}):
            assert summarize_article(article) is None

    def test_accepts_summary_that_passes_verification_on_first_attempt(self):
        article = {"title": "T", "url": "https://example.com/a"}
        generated = {"category": CATEGORIES[0], "summary": "A faithful summary."}
        with patch("news_agent.fetch_article_content", return_value={"content": "full text", "url": article["url"]}), \
                patch("news_agent._generate_summary_payload", return_value=generated) as mock_generate, \
                patch("news_agent.verify_summary", return_value={"faithful": True, "unsupported_claims": []}) as mock_verify:
            result = summarize_article(article)
        assert result == {"title": "T", "url": article["url"], **generated}
        assert mock_generate.call_count == 1
        assert mock_verify.call_count == 1

    def test_retries_with_feedback_after_failed_verification_then_succeeds(self):
        article = {"title": "T", "url": "https://example.com/a"}
        bad = {"category": CATEGORIES[0], "summary": "A summary with a made-up date."}
        good = {"category": CATEGORIES[0], "summary": "A corrected summary."}

        with patch("news_agent.fetch_article_content", return_value={"content": "full text", "url": article["url"]}), \
                patch("news_agent._generate_summary_payload", side_effect=[bad, good]) as mock_generate, \
                patch(
                    "news_agent.verify_summary",
                    side_effect=[
                        {"faithful": False, "unsupported_claims": ["the deadline is Q3 2026"]},
                        {"faithful": True, "unsupported_claims": []},
                    ],
                ):
            result = summarize_article(article)

        assert result == {"title": "T", "url": article["url"], **good}
        assert mock_generate.call_count == 2
        # second generation attempt must have been given the first attempt's unsupported claims
        assert mock_generate.call_args_list[1].args[2] == ["the deadline is Q3 2026"]

    def test_rejects_article_after_exhausting_max_attempts(self):
        article = {"title": "T", "url": "https://example.com/a"}
        generated = {"category": CATEGORIES[0], "summary": "Always wrong."}

        with patch("news_agent.fetch_article_content", return_value={"content": "full text", "url": article["url"]}), \
                patch("news_agent._generate_summary_payload", return_value=generated) as mock_generate, \
                patch(
                    "news_agent.verify_summary",
                    return_value={"faithful": False, "unsupported_claims": ["still wrong"]},
                ) as mock_verify:
            result = summarize_article(article, max_attempts=3)

        assert result is None
        assert mock_generate.call_count == 3
        assert mock_verify.call_count == 3

    def test_failed_verification_call_counts_as_a_failed_attempt(self):
        article = {"title": "T", "url": "https://example.com/a"}
        generated = {"category": CATEGORIES[0], "summary": "Some summary."}

        with patch("news_agent.fetch_article_content", return_value={"content": "full text", "url": article["url"]}), \
                patch("news_agent._generate_summary_payload", return_value=generated), \
                patch("news_agent.verify_summary", return_value=None) as mock_verify:
            result = summarize_article(article, max_attempts=2)

        assert result is None
        assert mock_verify.call_count == 2

    def test_failed_generation_attempt_does_not_stop_retries(self):
        article = {"title": "T", "url": "https://example.com/a"}
        good = {"category": CATEGORIES[0], "summary": "Finally faithful."}

        with patch("news_agent.fetch_article_content", return_value={"content": "full text", "url": article["url"]}), \
                patch("news_agent._generate_summary_payload", side_effect=[None, good]) as mock_generate, \
                patch("news_agent.verify_summary", return_value={"faithful": True, "unsupported_claims": []}):
            result = summarize_article(article, max_attempts=3)

        assert result == {"title": "T", "url": article["url"], **good}
        assert mock_generate.call_count == 2


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
