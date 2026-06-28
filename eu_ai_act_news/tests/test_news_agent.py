import os
import sys

# news_agent reads env vars at module level — stub them before import
os.environ.setdefault("DEEPSEEK_API_KEY", "test")
os.environ.setdefault("GITHUB_TOKEN", "test")
os.environ.setdefault("GITHUB_REPOSITORY", "test/test")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from news_agent import _is_allowed_domain, _is_relevant


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
