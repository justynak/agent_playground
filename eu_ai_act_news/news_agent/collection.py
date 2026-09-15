"""RSS feed collection: relevance filtering, parsing and cross-feed dedup."""

import sys
from datetime import UTC, datetime, timedelta
from typing import Any

import feedparser

from .config import KEYWORD_PATTERN, RSS_FEEDS


def is_relevant(title: str, summary: str) -> bool:
    return bool(KEYWORD_PATTERN.search(f"{title} {summary}"))


def fetch_rss_articles(feed_name: str, feed_url: str, days_back: int = 1) -> dict[str, Any]:
    try:
        cutoff = datetime.now(UTC) - timedelta(days=days_back)
        feed = feedparser.parse(
            feed_url,
            request_headers={"User-Agent": "Mozilla/5.0 (compatible; NewsResearchBot/1.0)"},
        )

        if feed.bozo and not feed.entries:
            return {"error": f"Failed to parse feed: {feed.bozo_exception}", "feed_name": feed_name}

        articles = []
        for entry in feed.entries:
            published = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                import calendar
                published = datetime.fromtimestamp(
                    calendar.timegm(entry.published_parsed), tz=UTC
                )

            if published and published < cutoff:
                continue

            title = getattr(entry, "title", "")
            summary = getattr(entry, "summary", "")
            link = getattr(entry, "link", "")

            articles.append({
                "title": title,
                "url": link,
                "published": published.isoformat() if published else "unknown",
                "summary": summary[:500] if summary else "",
                "potentially_relevant": is_relevant(title, summary),
            })

        relevant = [a for a in articles if a["potentially_relevant"]]
        return {
            "feed_name": feed_name,
            "total_recent_articles": len(articles),
            "relevant_count": len(relevant),
            "relevant_articles": relevant,
        }
    except Exception as e:
        return {"error": str(e), "feed_name": feed_name}


def dedup_articles(articles: list[dict]) -> list[dict]:
    """Dedup articles by URL (an article can appear in more than one feed), preserving order."""
    seen = set()
    result = []
    for article in articles:
        url = article.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        result.append(article)
    return result


def collect_relevant_articles(days_back: int = 1) -> list[dict]:
    """Fetch every configured feed and return the deduped set of potentially relevant articles."""
    all_articles = []
    for feed in RSS_FEEDS:
        result = fetch_rss_articles(feed["name"], feed["url"], days_back)
        count = len(result.get("relevant_articles", []))
        if count:
            print(f"  [feeds] {feed['name']}: {count} relevant article(s)", file=sys.stderr)
        all_articles.extend(result.get("relevant_articles", []))
    return dedup_articles(all_articles)
