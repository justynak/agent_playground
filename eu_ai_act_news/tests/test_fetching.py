from unittest.mock import MagicMock, patch

import pytest

from news_agent import config, fetching


class TestIsAllowedDomain:
    def test_exact_match(self):
        assert fetching.is_allowed_domain("https://euractiv.com/article") is True

    def test_www_prefix_stripped(self):
        assert fetching.is_allowed_domain("https://www.euractiv.com/article") is True

    def test_subdomain_allowed(self):
        assert fetching.is_allowed_domain("https://sub.bbc.co.uk/news") is True

    def test_blocked_domain(self):
        assert fetching.is_allowed_domain("https://evil.com/article") is False

    def test_domain_spoofing_attempt(self):
        # euractiv.com.evil.com should not pass
        assert fetching.is_allowed_domain("https://euractiv.com.evil.com/article") is False


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
        with patch("news_agent.fetching.requests.get", return_value=resp) as mock_get:
            result = fetching.fetch_following_safe_redirects("https://www.euractiv.com/article")
        assert result is resp
        assert mock_get.call_count == 1
        # auto-redirects must be disabled so every hop is validated by us
        assert mock_get.call_args.kwargs["allow_redirects"] is False

    def test_follows_redirect_to_allowed_domain(self):
        first = _response(status_code=302, headers={"Location": "https://www.politico.eu/real"}, is_redirect=True)
        second = _response()
        with patch("news_agent.fetching.requests.get", side_effect=[first, second]) as mock_get:
            result = fetching.fetch_following_safe_redirects("https://www.euractiv.com/article")
        assert result is second
        assert mock_get.call_count == 2
        assert mock_get.call_args_list[1].args[0] == "https://www.politico.eu/real"

    def test_relative_redirect_location_is_resolved(self):
        first = _response(status_code=302, headers={"Location": "/moved"}, is_redirect=True)
        second = _response()
        with patch("news_agent.fetching.requests.get", side_effect=[first, second]) as mock_get:
            fetching.fetch_following_safe_redirects("https://www.euractiv.com/a/b")
        assert mock_get.call_args_list[1].args[0] == "https://www.euractiv.com/moved"

    def test_redirect_to_blocked_domain_is_rejected(self):
        first = _response(status_code=302, headers={"Location": "https://evil.com/x"}, is_redirect=True)
        with patch("news_agent.fetching.requests.get", return_value=first) as mock_get, \
                pytest.raises(ValueError, match="allowlist"):
            fetching.fetch_following_safe_redirects("https://www.euractiv.com/article")
        # the off-allowlist host must never be requested
        assert mock_get.call_count == 1

    def test_exceeding_max_redirects_raises(self):
        def always_redirect(*args, **kwargs):
            return _response(status_code=302, headers={"Location": "https://www.euractiv.com/loop"}, is_redirect=True)

        with patch("news_agent.fetching.requests.get", side_effect=always_redirect), \
                pytest.raises(ValueError, match="Too many redirects"):
            fetching.fetch_following_safe_redirects("https://www.euractiv.com/article")

    def test_redirect_without_location_header_stops(self):
        resp = _response(status_code=302, headers={}, is_redirect=True)
        with patch("news_agent.fetching.requests.get", return_value=resp):
            assert fetching.fetch_following_safe_redirects("https://www.euractiv.com/article") is resp


class TestFetchArticleContentAllowlist:
    def test_off_allowlist_url_rejected_before_any_request(self):
        with patch("news_agent.fetching.requests.get") as mock_get:
            result = fetching.fetch_article_content("https://evil.com/article")
        assert "error" in result
        mock_get.assert_not_called()

    def test_redirect_escape_is_reported_as_error_not_crash(self):
        first = _response(status_code=302, headers={"Location": "https://evil.com/x"}, is_redirect=True)
        with patch("news_agent.fetching.requests.get", return_value=first):
            result = fetching.fetch_article_content("https://www.euractiv.com/article")
        assert "error" in result
        assert "allowlist" in result["error"]

    def test_truncated_flag_set_when_over_limit(self):
        long_text = "<p>" + ("word " * 5000) + "</p>"
        resp = _response(text=f"<article>{long_text}</article>")
        with patch("news_agent.fetching.requests.get", return_value=resp):
            result = fetching.fetch_article_content("https://www.euractiv.com/article")
        assert result["truncated"] is True
        assert len(result["content"]) == config.MAX_ARTICLE_CHARS

    def test_short_article_is_not_truncated(self):
        resp = _response(text="<article><p>Short body.</p></article>")
        with patch("news_agent.fetching.requests.get", return_value=resp):
            result = fetching.fetch_article_content("https://www.euractiv.com/article")
        assert result["truncated"] is False
        assert "Short body." in result["content"]
