import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# news_agent reads env vars at module level — stub them before import
os.environ.setdefault("DEEPSEEK_API_KEY", "test")
os.environ.setdefault("GITHUB_TOKEN", "test")
os.environ.setdefault("GITHUB_REPOSITORY", "test/test")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import news_agent
from news_agent import (
    CATEGORIES,
    _fetch_following_safe_redirects,
    _generate_summary_payload,
    _is_allowed_domain,
    _is_relevant,
    _validate_summary_payload,
    _validate_verification_payload,
    collect_relevant_articles,
    dedup_articles,
    fetch_article_content,
    issue_already_exists_today,
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

    def test_substring_false_positive_ai_act_in_thai_activist(self):
        # "Thai activist" contains the substring "ai act" but is not about the AI Act
        assert _is_relevant("Thai activist wins journalism award", "") is False

    def test_substring_false_positive_ai_action_plan(self):
        # "AI action plan" contains "ai act" as a prefix of "action" — word boundaries reject it
        assert _is_relevant("Government unveils AI action plan", "") is False

    def test_hyphenated_variant_matches(self):
        assert _is_relevant("New ai-act guidance published", "") is True

    def test_multi_word_keyword_tolerates_spacing(self):
        assert _is_relevant("general   purpose  ai   rules", "") is True

    def test_keyword_boundary_at_punctuation(self):
        assert _is_relevant("The AI Act, explained", "") is True


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


def _response(status_code=200, headers=None, text="", is_redirect=False):
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.text = text
    resp.is_redirect = is_redirect
    resp.raise_for_status = MagicMock()
    return resp


class TestSafeRedirects:
    def test_direct_response_returned_without_redirect(self):
        resp = _response()
        with patch("news_agent.requests.get", return_value=resp) as mock_get:
            result = _fetch_following_safe_redirects("https://www.euractiv.com/article")
        assert result is resp
        assert mock_get.call_count == 1
        # auto-redirects must be disabled so every hop is validated by us
        assert mock_get.call_args.kwargs["allow_redirects"] is False

    def test_follows_redirect_to_allowed_domain(self):
        first = _response(status_code=302, headers={"Location": "https://www.politico.eu/real"}, is_redirect=True)
        second = _response()
        with patch("news_agent.requests.get", side_effect=[first, second]) as mock_get:
            result = _fetch_following_safe_redirects("https://www.euractiv.com/article")
        assert result is second
        assert mock_get.call_count == 2
        assert mock_get.call_args_list[1].args[0] == "https://www.politico.eu/real"

    def test_relative_redirect_location_is_resolved(self):
        first = _response(status_code=302, headers={"Location": "/moved"}, is_redirect=True)
        second = _response()
        with patch("news_agent.requests.get", side_effect=[first, second]) as mock_get:
            _fetch_following_safe_redirects("https://www.euractiv.com/a/b")
        assert mock_get.call_args_list[1].args[0] == "https://www.euractiv.com/moved"

    def test_redirect_to_blocked_domain_is_rejected(self):
        first = _response(status_code=302, headers={"Location": "https://evil.com/x"}, is_redirect=True)
        with patch("news_agent.requests.get", return_value=first) as mock_get, \
                pytest.raises(ValueError, match="allowlist"):
            _fetch_following_safe_redirects("https://www.euractiv.com/article")
        # the off-allowlist host must never be requested
        assert mock_get.call_count == 1

    def test_exceeding_max_redirects_raises(self):
        def always_redirect(*args, **kwargs):
            return _response(status_code=302, headers={"Location": "https://www.euractiv.com/loop"}, is_redirect=True)

        with patch("news_agent.requests.get", side_effect=always_redirect), \
                pytest.raises(ValueError, match="Too many redirects"):
            _fetch_following_safe_redirects("https://www.euractiv.com/article")

    def test_redirect_without_location_header_stops(self):
        resp = _response(status_code=302, headers={}, is_redirect=True)
        with patch("news_agent.requests.get", return_value=resp):
            assert _fetch_following_safe_redirects("https://www.euractiv.com/article") is resp


class TestFetchArticleContentAllowlist:
    def test_off_allowlist_url_rejected_before_any_request(self):
        with patch("news_agent.requests.get") as mock_get:
            result = fetch_article_content("https://evil.com/article")
        assert "error" in result
        mock_get.assert_not_called()

    def test_redirect_escape_is_reported_as_error_not_crash(self):
        first = _response(status_code=302, headers={"Location": "https://evil.com/x"}, is_redirect=True)
        with patch("news_agent.requests.get", return_value=first):
            result = fetch_article_content("https://www.euractiv.com/article")
        assert "error" in result
        assert "allowlist" in result["error"]

    def test_truncated_flag_set_when_over_limit(self):
        long_text = "<p>" + ("word " * 5000) + "</p>"
        resp = _response(text=f"<article>{long_text}</article>")
        with patch("news_agent.requests.get", return_value=resp):
            result = fetch_article_content("https://www.euractiv.com/article")
        assert result["truncated"] is True
        assert len(result["content"]) == news_agent.MAX_ARTICLE_CHARS


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

    def test_truncation_notice_absent_by_default(self):
        article = {"title": "T", "url": "https://example.com/a"}
        response = _llm_response_with_arguments({"category": CATEGORIES[0], "summary": "A summary."})
        with patch("news_agent.client") as mock_client:
            mock_client.chat.completions.create.return_value = response
            _generate_summary_payload(article, "full text")
        _, kwargs = mock_client.chat.completions.create.call_args
        user_message = kwargs["messages"][1]["content"]
        assert "cut off" not in user_message

    def test_truncation_notice_included_when_truncated(self):
        article = {"title": "T", "url": "https://example.com/a"}
        response = _llm_response_with_arguments({"category": CATEGORIES[0], "summary": "A summary."})
        with patch("news_agent.client") as mock_client:
            mock_client.chat.completions.create.return_value = response
            _generate_summary_payload(article, "full text", truncated=True)
        _, kwargs = mock_client.chat.completions.create.call_args
        user_message = kwargs["messages"][1]["content"]
        assert "cut off" in user_message


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

    def test_truncation_notice_absent_by_default(self):
        response = _llm_response_with_arguments({"faithful": True, "unsupported_claims": []})
        with patch("news_agent.client") as mock_client:
            mock_client.chat.completions.create.return_value = response
            verify_summary("full text", "a summary")
        _, kwargs = mock_client.chat.completions.create.call_args
        user_message = kwargs["messages"][1]["content"]
        assert "cut off" not in user_message

    def test_truncation_notice_included_when_truncated(self):
        response = _llm_response_with_arguments({"faithful": True, "unsupported_claims": []})
        with patch("news_agent.client") as mock_client:
            mock_client.chat.completions.create.return_value = response
            verify_summary("full text", "a summary", truncated=True)
        _, kwargs = mock_client.chat.completions.create.call_args
        user_message = kwargs["messages"][1]["content"]
        assert "cut off" in user_message


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

    def test_truncation_flag_is_propagated_to_generate_and_verify(self):
        article = {"title": "T", "url": "https://example.com/a"}
        generated = {"category": CATEGORIES[0], "summary": "A summary."}
        content_result = {"content": "full text", "url": article["url"], "truncated": True}

        with patch("news_agent.fetch_article_content", return_value=content_result), \
                patch("news_agent._generate_summary_payload", return_value=generated) as mock_generate, \
                patch("news_agent.verify_summary", return_value={"faithful": True, "unsupported_claims": []}) as mock_verify:
            result = summarize_article(article)

        assert result == {"title": "T", "url": article["url"], **generated}
        assert mock_generate.call_args.kwargs["truncated"] is True
        assert mock_verify.call_args.kwargs["truncated"] is True

    def test_non_truncated_article_passes_truncated_false(self):
        article = {"title": "T", "url": "https://example.com/a"}
        generated = {"category": CATEGORIES[0], "summary": "A summary."}
        content_result = {"content": "full text", "url": article["url"], "truncated": False}

        with patch("news_agent.fetch_article_content", return_value=content_result), \
                patch("news_agent._generate_summary_payload", return_value=generated) as mock_generate, \
                patch("news_agent.verify_summary", return_value={"faithful": True, "unsupported_claims": []}) as mock_verify:
            summarize_article(article)

        assert mock_generate.call_args.kwargs["truncated"] is False
        assert mock_verify.call_args.kwargs["truncated"] is False

    def test_missing_truncated_key_defaults_to_false(self):
        # fetch_article_content results built by older/other callers may omit the key
        article = {"title": "T", "url": "https://example.com/a"}
        generated = {"category": CATEGORIES[0], "summary": "A summary."}
        content_result = {"content": "full text", "url": article["url"]}

        with patch("news_agent.fetch_article_content", return_value=content_result), \
                patch("news_agent._generate_summary_payload", return_value=generated) as mock_generate, \
                patch("news_agent.verify_summary", return_value={"faithful": True, "unsupported_claims": []}):
            summarize_article(article)

        assert mock_generate.call_args.kwargs["truncated"] is False


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


def _issues_response(issues: list[dict], status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = issues
    return resp


def _issues_titles(count: int, prefix: str = "Some issue") -> list[dict]:
    return [{"title": f"{prefix} #{i}"} for i in range(count)]


class TestIssueAlreadyExistsToday:
    def test_returns_true_when_today_is_on_first_page(self):
        issues = [{"title": "EU AI Act News Digest — 2026-01-01"}, {"title": "Other"}]
        with patch("news_agent.requests.get", return_value=_issues_response(issues)):
            assert issue_already_exists_today("2026-01-01") is True

    def test_returns_false_when_today_is_absent(self):
        issues = [{"title": "Some unrelated issue"}]
        with patch("news_agent.requests.get", return_value=_issues_response(issues)):
            assert issue_already_exists_today("2026-01-01") is False

    def test_short_first_page_stops_after_one_request(self):
        # fewer than a full page means there are no further pages
        issues = _issues_titles(3)
        with patch("news_agent.requests.get", return_value=_issues_response(issues)) as mock_get:
            assert issue_already_exists_today("2026-01-01") is False
        assert mock_get.call_count == 1

    def test_finds_today_on_a_later_page(self):
        # first page full (no match), second page short but contains today's digest
        first_page = _issues_titles(news_agent.ISSUES_PER_PAGE)
        second_page = [{"title": "EU AI Act News Digest — 2026-01-01"}]
        with patch(
            "news_agent.requests.get",
            side_effect=[_issues_response(first_page), _issues_response(second_page)],
        ) as mock_get:
            assert issue_already_exists_today("2026-01-01") is True
        assert mock_get.call_count == 2
        # the second request must ask for page 2
        assert mock_get.call_args_list[1].kwargs["params"]["page"] == 2

    def test_pagination_stops_when_no_more_pages(self):
        # every page is full but none contains today; must terminate via MAX_ISSUE_PAGES
        full_page = _issues_titles(news_agent.ISSUES_PER_PAGE)
        with patch("news_agent.requests.get", return_value=_issues_response(full_page)) as mock_get:
            assert issue_already_exists_today("2026-01-01") is False
        assert mock_get.call_count == news_agent.MAX_ISSUE_PAGES

    def test_returns_false_on_api_error(self):
        with patch("news_agent.requests.get", return_value=_issues_response([], status_code=500)):
            assert issue_already_exists_today("2026-01-01") is False

    def test_requests_only_open_issues_with_per_page(self):
        with patch("news_agent.requests.get", return_value=_issues_response([])) as mock_get:
            issue_already_exists_today("2026-01-01")
        params = mock_get.call_args.kwargs["params"]
        assert params["state"] == "open"
        assert params["per_page"] == news_agent.ISSUES_PER_PAGE
