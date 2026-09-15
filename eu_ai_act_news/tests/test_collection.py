from unittest.mock import patch

from news_agent import collection


class TestIsRelevant:
    def test_keyword_in_title(self):
        assert collection.is_relevant("EU AI Act enters into force", "") is True

    def test_keyword_in_summary(self):
        assert collection.is_relevant("New rules published", "GPAI models face stricter requirements") is True

    def test_no_keyword_match(self):
        assert collection.is_relevant("Football match results", "Arsenal won 3-0") is False

    def test_empty_strings(self):
        assert collection.is_relevant("", "") is False

    def test_case_insensitive(self):
        assert collection.is_relevant("AI REGULATION UPDATE", "") is True

    def test_substring_false_positive_ai_act_in_thai_activist(self):
        # "Thai activist" contains the substring "ai act" but is not about the AI Act
        assert collection.is_relevant("Thai activist wins journalism award", "") is False

    def test_substring_false_positive_ai_action_plan(self):
        # "AI action plan" contains "ai act" as a prefix of "action" — word boundaries reject it
        assert collection.is_relevant("Government unveils AI action plan", "") is False

    def test_hyphenated_variant_matches(self):
        assert collection.is_relevant("New ai-act guidance published", "") is True

    def test_multi_word_keyword_tolerates_spacing(self):
        assert collection.is_relevant("general   purpose  ai   rules", "") is True

    def test_keyword_boundary_at_punctuation(self):
        assert collection.is_relevant("The AI Act, explained", "") is True


class TestDedupArticles:
    def test_removes_duplicate_urls(self):
        articles = [{"url": "a"}, {"url": "b"}, {"url": "a"}]
        assert collection.dedup_articles(articles) == [{"url": "a"}, {"url": "b"}]

    def test_preserves_first_occurrence_order(self):
        articles = [{"url": "b"}, {"url": "a"}, {"url": "b"}]
        assert [a["url"] for a in collection.dedup_articles(articles)] == ["b", "a"]

    def test_skips_articles_without_url(self):
        articles = [{"url": ""}, {"title": "no url key"}, {"url": "a"}]
        assert collection.dedup_articles(articles) == [{"url": "a"}]

    def test_empty_list(self):
        assert collection.dedup_articles([]) == []


class TestCollectRelevantArticles:
    def test_dedupes_across_feeds_and_queries_every_feed(self):
        shared = {"url": "https://example.com/a", "title": "Shared"}
        with patch("news_agent.collection.fetch_rss_articles", return_value={"relevant_articles": [shared]}) as mock_fetch:
            articles = collection.collect_relevant_articles()
        assert articles == [shared]
        assert mock_fetch.call_count == len(collection.RSS_FEEDS)

    def test_no_relevant_articles_returns_empty_list(self):
        with patch("news_agent.collection.fetch_rss_articles", return_value={"relevant_articles": []}):
            assert collection.collect_relevant_articles() == []

    def test_skips_feed_entries_when_fetch_returns_no_relevant_key(self):
        # a feed whose fetch failed returns {"error": ...} with no "relevant_articles"
        with patch("news_agent.collection.fetch_rss_articles", return_value={"error": "boom"}):
            assert collection.collect_relevant_articles() == []


def _dummy_feed():
    return [{"name": "Fake", "url": "https://fake.example/feed"}]


class TestFetchRssArticles:
    def test_marks_articles_by_relevance_and_filters_the_rest(self):
        entry_relevant = _FakeEntry("EU AI Act update", "https://fake.example/a", "about the AI Act")
        entry_irrelevant = _FakeEntry("Football results", "https://fake.example/b", "sports")
        feed = _FakeFeed(entries=[entry_relevant, entry_irrelevant])

        with patch("news_agent.collection.feedparser.parse", return_value=feed):
            result = collection.fetch_rss_articles("Fake", "https://fake.example/feed", days_back=7)

        assert result["total_recent_articles"] == 2
        assert result["relevant_count"] == 1
        assert result["relevant_articles"][0]["title"] == "EU AI Act update"

    def test_reports_error_when_feed_fails_and_has_no_entries(self):
        feed = _FakeFeed(entries=[], bozo=True, bozo_exception="bad xml")
        with patch("news_agent.collection.feedparser.parse", return_value=feed):
            result = collection.fetch_rss_articles("Fake", "https://fake.example/feed")
        assert "error" in result
        assert result["feed_name"] == "Fake"

    def test_reports_error_when_parsing_raises(self):
        with patch("news_agent.collection.feedparser.parse", side_effect=RuntimeError("network down")):
            result = collection.fetch_rss_articles("Fake", "https://fake.example/feed")
        assert result["error"] == "network down"


class _FakeEntry:
    def __init__(self, title, link, summary=""):
        self.title = title
        self.link = link
        self.summary = summary


class _FakeFeed:
    def __init__(self, entries, bozo=False, bozo_exception=None):
        self.entries = entries
        self.bozo = bozo
        self.bozo_exception = bozo_exception
