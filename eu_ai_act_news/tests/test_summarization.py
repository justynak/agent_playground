import json
from unittest.mock import MagicMock, patch

from news_agent import config, summarization


def _llm_response_with_arguments(arguments: dict) -> MagicMock:
    response = MagicMock()
    tool_call = MagicMock()
    tool_call.function.arguments = json.dumps(arguments)
    response.choices[0].message.tool_calls = [tool_call]
    return response


class TestValidateSummaryPayload:
    def test_valid_payload_is_normalized(self):
        payload = {"category": config.CATEGORIES[0], "summary": "  A summary.  "}
        assert summarization.validate_summary_payload(payload) == {
            "category": config.CATEGORIES[0],
            "summary": "A summary.",
        }

    def test_rejects_unknown_category(self):
        assert summarization.validate_summary_payload({"category": "Not A Real Category", "summary": "text"}) is None

    def test_rejects_missing_category(self):
        assert summarization.validate_summary_payload({"summary": "text"}) is None

    def test_rejects_blank_summary(self):
        assert summarization.validate_summary_payload({"category": config.CATEGORIES[0], "summary": "   "}) is None

    def test_rejects_non_string_summary(self):
        assert summarization.validate_summary_payload({"category": config.CATEGORIES[0], "summary": 123}) is None


class TestValidateVerificationPayload:
    def test_valid_faithful_payload(self):
        payload = {"faithful": True, "unsupported_claims": []}
        assert summarization.validate_verification_payload(payload) == payload

    def test_valid_unfaithful_payload_with_claims(self):
        payload = {"faithful": False, "unsupported_claims": ["wrong date"]}
        assert summarization.validate_verification_payload(payload) == payload

    def test_rejects_non_bool_faithful(self):
        assert summarization.validate_verification_payload({"faithful": "yes", "unsupported_claims": []}) is None

    def test_rejects_non_list_claims(self):
        assert summarization.validate_verification_payload({"faithful": False, "unsupported_claims": "wrong date"}) is None

    def test_rejects_claims_with_non_string_items(self):
        assert summarization.validate_verification_payload({"faithful": False, "unsupported_claims": [1, 2]}) is None


class TestGenerateSummaryPayload:
    def test_returns_validated_payload_on_success(self):
        article = {"title": "T", "url": "https://example.com/a"}
        response = _llm_response_with_arguments({"category": config.CATEGORIES[0], "summary": "A summary."})
        with patch("news_agent.summarization.client") as mock_client:
            mock_client.chat.completions.create.return_value = response
            result = summarization.generate_summary_payload(article, "full text")
        assert result == {"category": config.CATEGORIES[0], "summary": "A summary."}

    def test_returns_none_when_model_picks_invalid_category(self):
        article = {"title": "T", "url": "https://example.com/a"}
        response = _llm_response_with_arguments({"category": "Not Real", "summary": "A summary."})
        with patch("news_agent.summarization.client") as mock_client:
            mock_client.chat.completions.create.return_value = response
            assert summarization.generate_summary_payload(article, "full text") is None

    def test_returns_none_when_no_tool_call_is_made(self):
        article = {"title": "T", "url": "https://example.com/a"}
        response = MagicMock()
        response.choices[0].message.tool_calls = []
        with patch("news_agent.summarization.client") as mock_client:
            mock_client.chat.completions.create.return_value = response
            assert summarization.generate_summary_payload(article, "full text") is None

    def test_returns_none_on_api_error(self):
        article = {"title": "T", "url": "https://example.com/a"}
        with patch("news_agent.summarization.client") as mock_client:
            mock_client.chat.completions.create.side_effect = RuntimeError("network down")
            assert summarization.generate_summary_payload(article, "full text") is None

    def test_feedback_is_included_in_the_prompt(self):
        article = {"title": "T", "url": "https://example.com/a"}
        response = _llm_response_with_arguments({"category": config.CATEGORIES[0], "summary": "Corrected."})
        with patch("news_agent.summarization.client") as mock_client:
            mock_client.chat.completions.create.return_value = response
            summarization.generate_summary_payload(article, "full text", feedback=["the deadline is Q3 2026"])
        _, kwargs = mock_client.chat.completions.create.call_args
        user_message = kwargs["messages"][1]["content"]
        assert "the deadline is Q3 2026" in user_message

    def test_truncation_notice_absent_by_default(self):
        article = {"title": "T", "url": "https://example.com/a"}
        response = _llm_response_with_arguments({"category": config.CATEGORIES[0], "summary": "A summary."})
        with patch("news_agent.summarization.client") as mock_client:
            mock_client.chat.completions.create.return_value = response
            summarization.generate_summary_payload(article, "full text")
        _, kwargs = mock_client.chat.completions.create.call_args
        user_message = kwargs["messages"][1]["content"]
        assert "cut off" not in user_message

    def test_truncation_notice_included_when_truncated(self):
        article = {"title": "T", "url": "https://example.com/a"}
        response = _llm_response_with_arguments({"category": config.CATEGORIES[0], "summary": "A summary."})
        with patch("news_agent.summarization.client") as mock_client:
            mock_client.chat.completions.create.return_value = response
            summarization.generate_summary_payload(article, "full text", truncated=True)
        _, kwargs = mock_client.chat.completions.create.call_args
        user_message = kwargs["messages"][1]["content"]
        assert "cut off" in user_message


class TestVerifySummary:
    def test_returns_faithful_verdict(self):
        response = _llm_response_with_arguments({"faithful": True, "unsupported_claims": []})
        with patch("news_agent.summarization.client") as mock_client:
            mock_client.chat.completions.create.return_value = response
            assert summarization.verify_summary("full text", "a summary") == {"faithful": True, "unsupported_claims": []}

    def test_returns_unfaithful_verdict_with_claims(self):
        response = _llm_response_with_arguments({"faithful": False, "unsupported_claims": ["wrong date"]})
        with patch("news_agent.summarization.client") as mock_client:
            mock_client.chat.completions.create.return_value = response
            result = summarization.verify_summary("full text", "a summary")
        assert result == {"faithful": False, "unsupported_claims": ["wrong date"]}

    def test_returns_none_on_malformed_response(self):
        response = MagicMock()
        response.choices[0].message.tool_calls = []
        with patch("news_agent.summarization.client") as mock_client:
            mock_client.chat.completions.create.return_value = response
            assert summarization.verify_summary("full text", "a summary") is None

    def test_returns_none_on_api_error(self):
        with patch("news_agent.summarization.client") as mock_client:
            mock_client.chat.completions.create.side_effect = RuntimeError("network down")
            assert summarization.verify_summary("full text", "a summary") is None

    def test_truncation_notice_absent_by_default(self):
        response = _llm_response_with_arguments({"faithful": True, "unsupported_claims": []})
        with patch("news_agent.summarization.client") as mock_client:
            mock_client.chat.completions.create.return_value = response
            summarization.verify_summary("full text", "a summary")
        _, kwargs = mock_client.chat.completions.create.call_args
        user_message = kwargs["messages"][1]["content"]
        assert "cut off" not in user_message

    def test_truncation_notice_included_when_truncated(self):
        response = _llm_response_with_arguments({"faithful": True, "unsupported_claims": []})
        with patch("news_agent.summarization.client") as mock_client:
            mock_client.chat.completions.create.return_value = response
            summarization.verify_summary("full text", "a summary", truncated=True)
        _, kwargs = mock_client.chat.completions.create.call_args
        user_message = kwargs["messages"][1]["content"]
        assert "cut off" in user_message


class TestSummarizeArticle:
    def test_skips_article_when_content_fetch_fails(self):
        article = {"title": "T", "url": "https://example.com/a"}
        with patch("news_agent.summarization.fetch_article_content", return_value={"error": "boom"}):
            assert summarization.summarize_article(article) is None

    def test_accepts_summary_that_passes_verification_on_first_attempt(self):
        article = {"title": "T", "url": "https://example.com/a"}
        generated = {"category": config.CATEGORIES[0], "summary": "A faithful summary."}
        with patch("news_agent.summarization.fetch_article_content", return_value={"content": "full text", "url": article["url"]}), \
                patch("news_agent.summarization.generate_summary_payload", return_value=generated) as mock_generate, \
                patch("news_agent.summarization.verify_summary", return_value={"faithful": True, "unsupported_claims": []}) as mock_verify:
            result = summarization.summarize_article(article)
        assert result == {"title": "T", "url": article["url"], **generated}
        assert mock_generate.call_count == 1
        assert mock_verify.call_count == 1

    def test_retries_with_feedback_after_failed_verification_then_succeeds(self):
        article = {"title": "T", "url": "https://example.com/a"}
        bad = {"category": config.CATEGORIES[0], "summary": "A summary with a made-up date."}
        good = {"category": config.CATEGORIES[0], "summary": "A corrected summary."}

        with patch("news_agent.summarization.fetch_article_content", return_value={"content": "full text", "url": article["url"]}), \
                patch("news_agent.summarization.generate_summary_payload", side_effect=[bad, good]) as mock_generate, \
                patch(
                    "news_agent.summarization.verify_summary",
                    side_effect=[
                        {"faithful": False, "unsupported_claims": ["the deadline is Q3 2026"]},
                        {"faithful": True, "unsupported_claims": []},
                    ],
                ):
            result = summarization.summarize_article(article)

        assert result == {"title": "T", "url": article["url"], **good}
        assert mock_generate.call_count == 2
        # second generation attempt must have been given the first attempt's unsupported claims
        assert mock_generate.call_args_list[1].args[2] == ["the deadline is Q3 2026"]

    def test_rejects_article_after_exhausting_max_attempts(self):
        article = {"title": "T", "url": "https://example.com/a"}
        generated = {"category": config.CATEGORIES[0], "summary": "Always wrong."}

        with patch("news_agent.summarization.fetch_article_content", return_value={"content": "full text", "url": article["url"]}), \
                patch("news_agent.summarization.generate_summary_payload", return_value=generated) as mock_generate, \
                patch(
                    "news_agent.summarization.verify_summary",
                    return_value={"faithful": False, "unsupported_claims": ["still wrong"]},
                ) as mock_verify:
            result = summarization.summarize_article(article, max_attempts=3)

        assert result is None
        assert mock_generate.call_count == 3
        assert mock_verify.call_count == 3

    def test_failed_verification_call_counts_as_a_failed_attempt(self):
        article = {"title": "T", "url": "https://example.com/a"}
        generated = {"category": config.CATEGORIES[0], "summary": "Some summary."}

        with patch("news_agent.summarization.fetch_article_content", return_value={"content": "full text", "url": article["url"]}), \
                patch("news_agent.summarization.generate_summary_payload", return_value=generated), \
                patch("news_agent.summarization.verify_summary", return_value=None) as mock_verify:
            result = summarization.summarize_article(article, max_attempts=2)

        assert result is None
        assert mock_verify.call_count == 2

    def test_failed_generation_attempt_does_not_stop_retries(self):
        article = {"title": "T", "url": "https://example.com/a"}
        good = {"category": config.CATEGORIES[0], "summary": "Finally faithful."}

        with patch("news_agent.summarization.fetch_article_content", return_value={"content": "full text", "url": article["url"]}), \
                patch("news_agent.summarization.generate_summary_payload", side_effect=[None, good]) as mock_generate, \
                patch("news_agent.summarization.verify_summary", return_value={"faithful": True, "unsupported_claims": []}):
            result = summarization.summarize_article(article, max_attempts=3)

        assert result == {"title": "T", "url": article["url"], **good}
        assert mock_generate.call_count == 2

    def test_truncation_flag_is_propagated_to_generate_and_verify(self):
        article = {"title": "T", "url": "https://example.com/a"}
        generated = {"category": config.CATEGORIES[0], "summary": "A summary."}
        content_result = {"content": "full text", "url": article["url"], "truncated": True}

        with patch("news_agent.summarization.fetch_article_content", return_value=content_result), \
                patch("news_agent.summarization.generate_summary_payload", return_value=generated) as mock_generate, \
                patch("news_agent.summarization.verify_summary", return_value={"faithful": True, "unsupported_claims": []}) as mock_verify:
            result = summarization.summarize_article(article)

        assert result == {"title": "T", "url": article["url"], **generated}
        assert mock_generate.call_args.kwargs["truncated"] is True
        assert mock_verify.call_args.kwargs["truncated"] is True

    def test_non_truncated_article_passes_truncated_false(self):
        article = {"title": "T", "url": "https://example.com/a"}
        generated = {"category": config.CATEGORIES[0], "summary": "A summary."}
        content_result = {"content": "full text", "url": article["url"], "truncated": False}

        with patch("news_agent.summarization.fetch_article_content", return_value=content_result), \
                patch("news_agent.summarization.generate_summary_payload", return_value=generated) as mock_generate, \
                patch("news_agent.summarization.verify_summary", return_value={"faithful": True, "unsupported_claims": []}) as mock_verify:
            summarization.summarize_article(article)

        assert mock_generate.call_args.kwargs["truncated"] is False
        assert mock_verify.call_args.kwargs["truncated"] is False

    def test_missing_truncated_key_defaults_to_false(self):
        # fetch_article_content results built by older/other callers may omit the key
        article = {"title": "T", "url": "https://example.com/a"}
        generated = {"category": config.CATEGORIES[0], "summary": "A summary."}
        content_result = {"content": "full text", "url": article["url"]}

        with patch("news_agent.summarization.fetch_article_content", return_value=content_result), \
                patch("news_agent.summarization.generate_summary_payload", return_value=generated) as mock_generate, \
                patch("news_agent.summarization.verify_summary", return_value={"faithful": True, "unsupported_claims": []}):
            summarization.summarize_article(article)

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
            return {"title": "Good", "url": article["url"], "category": config.CATEGORIES[0], "summary": "ok"}

        with patch("news_agent.summarization.summarize_article", side_effect=fake_summarize):
            result = summarization.summarize_articles(articles)

        assert result == [
            {"title": "Good", "url": "https://example.com/good", "category": config.CATEGORIES[0], "summary": "ok"}
        ]

    def test_empty_input_returns_empty_list(self):
        assert summarization.summarize_articles([]) == []
