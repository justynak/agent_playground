"""EU AI Act daily news agent using DeepSeek API with tool calling."""

import json
import os
import sys
from datetime import UTC, datetime, timedelta
from typing import Any

import feedparser
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
GITHUB_REPOSITORY = os.environ["GITHUB_REPOSITORY"]

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)

RSS_FEEDS = [
    {"name": "Euractiv", "url": "https://www.euractiv.com/feed/"},
    {"name": "European Parliament", "url": "https://www.europarl.europa.eu/rss/doc/latest-news/en.xml"},
    {"name": "Politico Europe", "url": "https://www.politico.eu/feed/"},
    {"name": "Reuters Technology", "url": "https://feeds.reuters.com/reuters/technologyNews"},
    {"name": "TechCrunch", "url": "https://techcrunch.com/feed/"},
    {"name": "The Verge", "url": "https://www.theverge.com/rss/index.xml"},
    {"name": "Wired", "url": "https://www.wired.com/feed/rss"},
    {"name": "BBC Technology", "url": "https://feeds.bbci.co.uk/news/technology/rss.xml"},
    {"name": "VentureBeat AI", "url": "https://venturebeat.com/category/ai/feed/"},
]

ALLOWED_DOMAINS = {
    "euractiv.com",
    "europarl.europa.eu",
    "politico.eu",
    "reuters.com",
    "techcrunch.com",
    "theverge.com",
    "wired.com",
    "bbc.co.uk",
    "bbc.com",
    "venturebeat.com",
    "ec.europa.eu",
    "consilium.europa.eu",
    "eur-lex.europa.eu",
}

AI_ACT_KEYWORDS = [
    "eu ai act", "ai act", "artificial intelligence act",
    "ai regulation", "eu ai regulation", "general purpose ai",
    "gpai", "ai liability", "ai office", "eu ai office",
    "eu ai", "europe ai", "european ai",
]

CATEGORIES = [
    "Key Developments",
    "Legislative & Regulatory Updates",
    "Industry & Compliance",
    "Research & Expert Opinion",
]

SUMMARIZE_TOOL = {
    "type": "function",
    "function": {
        "name": "record_summary",
        "description": "Records a structured summary for a single article.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "The single best-fitting category for this article.",
                    "enum": CATEGORIES,
                },
                "summary": {
                    "type": "string",
                    "description": "2-4 sentence summary of the article, grounded only in the provided article text.",
                },
            },
            "required": ["category", "summary"],
        },
    },
}

VERIFY_TOOL = {
    "type": "function",
    "function": {
        "name": "record_verification",
        "description": "Records whether a summary is fully supported by the article text it claims to summarize.",
        "parameters": {
            "type": "object",
            "properties": {
                "faithful": {
                    "type": "boolean",
                    "description": "True only if every factual claim in the summary is directly supported by the article text.",
                },
                "unsupported_claims": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Claims in the summary not supported by the article text. Empty if faithful is true.",
                },
            },
            "required": ["faithful", "unsupported_claims"],
        },
    },
}

MAX_SUMMARY_ATTEMPTS = 3


def _is_relevant(title: str, summary: str) -> bool:
    text = f"{title} {summary}".lower()
    return any(kw in text for kw in AI_ACT_KEYWORDS)


def _is_allowed_domain(url: str) -> bool:
    from urllib.parse import urlparse
    domain = urlparse(url).netloc.removeprefix("www.")
    return any(domain == d or domain.endswith("." + d) for d in ALLOWED_DOMAINS)


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

            is_relevant = _is_relevant(title, summary)

            articles.append({
                "title": title,
                "url": link,
                "published": published.isoformat() if published else "unknown",
                "summary": summary[:500] if summary else "",
                "potentially_relevant": is_relevant,
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


def fetch_article_content(url: str) -> dict[str, Any]:
    if not _is_allowed_domain(url):
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.removeprefix("www.")
        return {"error": f"Domain not in allowlist: {domain}", "url": url}

    try:
        resp = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        )
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        main_content = (
            soup.find("article")
            or soup.find("main")
            or soup.find(attrs={"class": lambda c: c and any(
                x in " ".join(c) for x in ["article-body", "article__body", "story-body", "post-content", "entry-content"]
            )})
            or soup.body
        )

        text = (main_content or soup).get_text(separator=" ", strip=True)
        text = " ".join(text.split())
        return {"url": url, "content": text[:4000], "truncated": len(text) > 4000}
    except Exception as e:
        return {"error": str(e), "url": url}


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


def _validate_summary_payload(payload: dict) -> dict[str, str] | None:
    category = payload.get("category")
    summary = payload.get("summary")
    if not isinstance(category, str) or category not in CATEGORIES:
        return None
    if not isinstance(summary, str) or not summary.strip():
        return None
    return {"category": category, "summary": summary.strip()}


def _generate_summary_payload(article: dict, content: str, feedback: list[str] | None = None) -> dict[str, str] | None:
    """Ask the model for one structured, categorized summary of the given article text.

    If `feedback` (unsupported claims from a prior failed verification) is given, the
    model is asked to correct them rather than starting from scratch.
    """
    system_prompt = (
        "You are an EU AI Act policy analyst. You will be given the full text of one news article.\n"
        "Summarize it in 2-4 sentences, using only information present in the article text below — "
        "do not add outside knowledge or speculation.\n"
        f"Then call record_summary with your summary and the single best-fitting category from: "
        f"{', '.join(CATEGORIES)}."
    )
    user_content = f"Article title: {article['title']}\n\nArticle text:\n{content}"
    if feedback:
        claims = "\n".join(f"- {claim}" for claim in feedback)
        user_content += (
            "\n\nYour previous summary made claims not supported by the article text above:\n"
            f"{claims}\n"
            "Write a corrected summary that avoids these unsupported claims."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=[SUMMARIZE_TOOL],
            tool_choice={"type": "function", "function": {"name": "record_summary"}},
        )
        payload = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    except Exception as e:
        print(f"  [summarize] generation call failed: {e}", file=sys.stderr)
        return None

    return _validate_summary_payload(payload)


def _validate_verification_payload(payload: dict) -> dict[str, Any] | None:
    faithful = payload.get("faithful")
    claims = payload.get("unsupported_claims")
    if not isinstance(faithful, bool):
        return None
    if not isinstance(claims, list) or not all(isinstance(c, str) for c in claims):
        return None
    return {"faithful": faithful, "unsupported_claims": claims}


def verify_summary(content: str, summary: str) -> dict[str, Any] | None:
    """Judge whether `summary` is fully supported by the article text `content`.

    Returns None if the judge call itself fails or returns something unparseable —
    callers must treat that as "could not verify" (i.e. not faithful), not as a pass.
    """
    system_prompt = (
        "You are a strict fact-checker. You will be given an article's full text and a summary of it.\n"
        "Check whether every factual claim in the summary — dates, numbers, names, actions, outcomes — "
        "is directly supported by the article text. Assume the summary is wrong until you find clear "
        "support for each claim; do not give it the benefit of the doubt.\n"
        "Call record_verification with your verdict."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Article text:\n{content}\n\nSummary to check:\n{summary}"},
    ]

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=[VERIFY_TOOL],
            tool_choice={"type": "function", "function": {"name": "record_verification"}},
        )
        payload = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    except Exception as e:
        print(f"  [verify] verification call failed: {e}", file=sys.stderr)
        return None

    return _validate_verification_payload(payload)


def summarize_article(article: dict, max_attempts: int = MAX_SUMMARY_ATTEMPTS) -> dict[str, str] | None:
    """Summarize a single article, verifying the summary against the article text before accepting it.

    Regenerates the summary (with feedback on what was unsupported) up to `max_attempts`
    times. If no attempt passes verification, the article is rejected and None is returned
    — callers are expected to skip such articles rather than publish an unverified summary.
    """
    content_result = fetch_article_content(article["url"])
    if "error" in content_result:
        print(f"  [summarize] skipping {article['url']}: {content_result['error']}", file=sys.stderr)
        return None
    content = content_result["content"]

    feedback: list[str] | None = None
    for attempt in range(1, max_attempts + 1):
        generated = _generate_summary_payload(article, content, feedback)
        if generated is None:
            print(f"  [summarize] {article['url']} attempt {attempt}/{max_attempts}: generation failed", file=sys.stderr)
            feedback = None
            continue

        verification = verify_summary(content, generated["summary"])
        if verification is not None and verification["faithful"]:
            return {"title": article["title"], "url": article["url"], **generated}

        unsupported = verification["unsupported_claims"] if verification else []
        reason = unsupported or "verification call failed"
        print(
            f"  [summarize] {article['url']} attempt {attempt}/{max_attempts} failed verification: {reason}",
            file=sys.stderr,
        )
        feedback = unsupported

    print(f"  [summarize] rejected {article['url']} after {max_attempts} failed attempt(s)", file=sys.stderr)
    return None


def summarize_articles(articles: list[dict]) -> list[dict]:
    summaries = []
    for article in articles:
        summary = summarize_article(article)
        if summary:
            summaries.append(summary)
    return summaries


def render_digest(summaries: list[dict]) -> str | None:
    """Deterministically render validated per-article summaries into the Markdown digest.

    Returns None if there is nothing to render.
    """
    if not summaries:
        return None

    sections = []
    for category in CATEGORIES:
        items = [s for s in summaries if s["category"] == category]
        if not items:
            continue
        lines = [f"## {category}", ""]
        lines.extend(f"- **[{item['title']}]({item['url']})** — {item['summary']}" for item in items)
        sections.append("\n".join(lines))

    sources_lines = ["## Sources", ""]
    sources_lines.extend(f"- [{item['title']}]({item['url']})" for item in summaries)
    sections.append("\n".join(sources_lines))

    return "\n\n".join(sections)


def issue_already_exists_today(today: str) -> bool:
    owner, repo = GITHUB_REPOSITORY.split("/", 1)
    resp = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/issues",
        headers={
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
        },
        params={"state": "open", "per_page": 20},
    )
    if resp.status_code != 200:
        return False
    return any(today in issue["title"] for issue in resp.json())


def create_github_issue(title: str, body: str) -> str:
    owner, repo = GITHUB_REPOSITORY.split("/", 1)
    api_url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # Try with labels first; fall back to no labels if they don't exist yet
    for payload in [
        {"title": title, "body": body, "labels": ["eu-ai-act", "automated-digest"]},
        {"title": title, "body": body},
    ]:
        resp = requests.post(api_url, headers=headers, json=payload)
        if resp.status_code == 201:
            return resp.json()["html_url"]
        if resp.status_code != 422:
            resp.raise_for_status()

    raise RuntimeError(f"Failed to create issue: {resp.status_code} {resp.text}")


def main():
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    print(f"Starting EU AI Act news agent for {today}", file=sys.stderr)

    if issue_already_exists_today(today):
        print(f"Digest for {today} already exists — skipping.", file=sys.stderr)
        sys.exit(0)

    articles = collect_relevant_articles()
    if not articles:
        print(f"No relevant EU AI Act articles found for {today} — skipping issue.", file=sys.stderr)
        sys.exit(0)

    print(f"Found {len(articles)} relevant article(s) — summarizing.", file=sys.stderr)
    summaries = summarize_articles(articles)

    digest = render_digest(summaries)
    if digest is None:
        print(f"No summaries could be produced for {today} — skipping issue.", file=sys.stderr)
        sys.exit(0)

    skipped = len(articles) - len(summaries)
    if skipped:
        print(f"{skipped} article(s) could not be summarized and were excluded.", file=sys.stderr)

    issue_title = f"EU AI Act News Digest — {today}"
    issue_body = (
        f"{digest}\n\n"
        f"---\n"
        f"*Automated daily digest. Sources: {', '.join(f['name'] for f in RSS_FEEDS)}. "
        f"Powered by DeepSeek AI via the EU AI Act News Agent.*"
    )

    issue_url = create_github_issue(issue_title, issue_body)
    print(f"Issue created: {issue_url}", file=sys.stderr)


if __name__ == "__main__":
    main()
