#!/usr/bin/env python3
"""EU AI Act daily news agent using DeepSeek API with tool calling."""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

import feedparser
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

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


def fetch_rss_articles(feed_name: str, feed_url: str, days_back: int = 1) -> dict[str, Any]:
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
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
                    calendar.timegm(entry.published_parsed), tz=timezone.utc
                )

            if published and published < cutoff:
                continue

            title = getattr(entry, "title", "")
            summary = getattr(entry, "summary", "")
            link = getattr(entry, "link", "")

            text_to_check = f"{title} {summary}".lower()
            is_relevant = any(kw in text_to_check for kw in AI_ACT_KEYWORDS)

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
    from urllib.parse import urlparse
    domain = urlparse(url).netloc.removeprefix("www.")
    if not any(domain == d or domain.endswith("." + d) for d in ALLOWED_DOMAINS):
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


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_rss_articles",
            "description": (
                "Fetch recent articles from a predefined RSS news feed. "
                "Returns articles from the last N days, flagging those potentially related to the EU AI Act."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "feed_name": {
                        "type": "string",
                        "description": "Name of the RSS feed to fetch",
                        "enum": [f["name"] for f in RSS_FEEDS],
                    },
                    "days_back": {
                        "type": "integer",
                        "description": "How many days back to search (default: 1)",
                        "default": 1,
                    },
                },
                "required": ["feed_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_article_content",
            "description": "Fetch and extract the full text of a specific article by URL. Use this to read a relevant article in detail.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL of the article to read",
                    }
                },
                "required": ["url"],
            },
        },
    },
]


def dispatch_tool(name: str, args: dict) -> Any:
    if name == "fetch_rss_articles":
        feed_name = args["feed_name"]
        feed_url = next((f["url"] for f in RSS_FEEDS if f["name"] == feed_name), None)
        if not feed_url:
            return {"error": f"Unknown feed: {feed_name}"}
        return fetch_rss_articles(feed_name, feed_url, args.get("days_back", 1))
    elif name == "fetch_article_content":
        return fetch_article_content(args["url"])
    return {"error": f"Unknown tool: {name}"}


def run_agent(today: str) -> str:
    feed_list = "\n".join(f"- {f['name']}" for f in RSS_FEEDS)
    system_prompt = f"""You are an expert EU AI policy research agent. Today is {today}.

Your task: Find and summarize all significant EU AI Act news from the last 24 hours.

Available RSS feeds to check:
{feed_list}

Workflow:
1. Call fetch_rss_articles for EVERY feed listed above (use days_back=1)
2. For each article marked potentially_relevant=true, call fetch_article_content to read it in full
3. After checking all feeds and reading relevant articles, write a comprehensive digest

Your final digest must be formatted in Markdown with these sections:
## Key Developments
## Legislative & Regulatory Updates
## Industry & Compliance
## Research & Expert Opinion
## Sources

If no relevant news was found in any source, say so clearly — do not fabricate news.
Always include direct URLs for every item you mention."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Research EU AI Act news for {today} and write the daily digest."},
    ]

    for iteration in range(25):
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return msg.content or "Agent produced no output."

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            print(f"  [tool] {tc.function.name}({json.dumps(args)})", file=sys.stderr)
            result = dispatch_tool(tc.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

    return "Agent reached the iteration limit without producing a final summary."


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
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"Starting EU AI Act news agent for {today}", file=sys.stderr)

    if issue_already_exists_today(today):
        print(f"Digest for {today} already exists — skipping.", file=sys.stderr)
        sys.exit(0)

    summary = run_agent(today)

    issue_title = f"EU AI Act News Digest — {today}"
    issue_body = (
        f"{summary}\n\n"
        f"---\n"
        f"*Automated daily digest. Sources: {', '.join(f['name'] for f in RSS_FEEDS)}. "
        f"Powered by DeepSeek AI via the EU AI Act News Agent.*"
    )

    issue_url = create_github_issue(issue_title, issue_body)
    print(f"Issue created: {issue_url}", file=sys.stderr)


if __name__ == "__main__":
    main()
