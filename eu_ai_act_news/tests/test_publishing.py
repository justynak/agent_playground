from unittest.mock import MagicMock, patch

from news_agent import config, publishing


class TestRenderDigest:
    def test_empty_summaries_returns_none(self):
        assert publishing.render_digest([]) is None

    def test_groups_sections_in_fixed_category_order(self):
        summaries = [
            {"title": "B", "url": "https://x/b", "category": config.CATEGORIES[2], "summary": "sum b"},
            {"title": "A", "url": "https://x/a", "category": config.CATEGORIES[0], "summary": "sum a"},
        ]
        digest = publishing.render_digest(summaries)
        assert digest.index(f"## {config.CATEGORIES[0]}") < digest.index(f"## {config.CATEGORIES[2]}")
        assert "sum a" in digest and "sum b" in digest

    def test_omits_categories_with_no_items(self):
        summaries = [{"title": "A", "url": "https://x/a", "category": config.CATEGORIES[0], "summary": "sum"}]
        digest = publishing.render_digest(summaries)
        for category in config.CATEGORIES[1:]:
            assert f"## {category}" not in digest

    def test_sources_section_lists_every_item(self):
        summaries = [
            {"title": "A", "url": "https://x/a", "category": config.CATEGORIES[0], "summary": "sum a"},
            {"title": "B", "url": "https://x/b", "category": config.CATEGORIES[1], "summary": "sum b"},
        ]
        digest = publishing.render_digest(summaries)
        assert "## Sources" in digest
        assert "[A](https://x/a)" in digest
        assert "[B](https://x/b)" in digest


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
        with patch("news_agent.publishing.requests.get", return_value=_issues_response(issues)):
            assert publishing.issue_already_exists_today("2026-01-01") is True

    def test_returns_false_when_today_is_absent(self):
        issues = [{"title": "Some unrelated issue"}]
        with patch("news_agent.publishing.requests.get", return_value=_issues_response(issues)):
            assert publishing.issue_already_exists_today("2026-01-01") is False

    def test_short_first_page_stops_after_one_request(self):
        # fewer than a full page means there are no further pages
        issues = _issues_titles(3)
        with patch("news_agent.publishing.requests.get", return_value=_issues_response(issues)) as mock_get:
            assert publishing.issue_already_exists_today("2026-01-01") is False
        assert mock_get.call_count == 1

    def test_finds_today_on_a_later_page(self):
        # first page full (no match), second page short but contains today's digest
        first_page = _issues_titles(config.ISSUES_PER_PAGE)
        second_page = [{"title": "EU AI Act News Digest — 2026-01-01"}]
        with patch(
            "news_agent.publishing.requests.get",
            side_effect=[_issues_response(first_page), _issues_response(second_page)],
        ) as mock_get:
            assert publishing.issue_already_exists_today("2026-01-01") is True
        assert mock_get.call_count == 2
        # the second request must ask for page 2
        assert mock_get.call_args_list[1].kwargs["params"]["page"] == 2

    def test_pagination_stops_when_no_more_pages(self):
        # every page is full but none contains today; must terminate via MAX_ISSUE_PAGES
        full_page = _issues_titles(config.ISSUES_PER_PAGE)
        with patch("news_agent.publishing.requests.get", return_value=_issues_response(full_page)) as mock_get:
            assert publishing.issue_already_exists_today("2026-01-01") is False
        assert mock_get.call_count == config.MAX_ISSUE_PAGES

    def test_returns_false_on_api_error(self):
        with patch("news_agent.publishing.requests.get", return_value=_issues_response([], status_code=500)):
            assert publishing.issue_already_exists_today("2026-01-01") is False

    def test_requests_only_open_issues_with_per_page(self):
        with patch("news_agent.publishing.requests.get", return_value=_issues_response([])) as mock_get:
            publishing.issue_already_exists_today("2026-01-01")
        params = mock_get.call_args.kwargs["params"]
        assert params["state"] == "open"
        assert params["per_page"] == config.ISSUES_PER_PAGE
