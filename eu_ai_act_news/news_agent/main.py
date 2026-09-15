"""Entrypoint: wires collection, summarization and publishing into a daily run."""

import sys
from datetime import UTC, datetime

from .collection import collect_relevant_articles
from .config import RSS_FEEDS
from .publishing import create_github_issue, issue_already_exists_today, render_digest
from .summarization import summarize_articles


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
