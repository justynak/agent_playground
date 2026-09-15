"""Entrypoint: wires collection, summarization and publishing into a daily run."""

import logging
import sys
from datetime import UTC, datetime

from opentelemetry import trace

from .collection import collect_relevant_articles
from .config import RSS_FEEDS
from .publishing import create_github_issue, issue_already_exists_today, render_digest
from .summarization import summarize_articles
from .telemetry import digests_published_total, setup_telemetry, shutdown_telemetry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


def run(today: str) -> None:
    """The pipeline itself, as one root span covering the whole run."""
    span = trace.get_current_span()
    span.set_attribute("run.date", today)

    if issue_already_exists_today(today):
        logger.info("Digest for %s already exists — skipping.", today)
        span.set_attribute("run.outcome", "skipped_duplicate")
        digests_published_total.add(1, {"outcome": "skipped_duplicate"})
        return

    articles = collect_relevant_articles()
    span.set_attribute("run.articles_collected", len(articles))
    if not articles:
        logger.info("No relevant EU AI Act articles found for %s — skipping issue.", today)
        span.set_attribute("run.outcome", "skipped_no_articles")
        digests_published_total.add(1, {"outcome": "skipped_no_articles"})
        return

    logger.info("Found %d relevant article(s) — summarizing.", len(articles))
    summaries = summarize_articles(articles)
    span.set_attribute("run.summaries_produced", len(summaries))

    digest = render_digest(summaries)
    if digest is None:
        logger.info("No summaries could be produced for %s — skipping issue.", today)
        span.set_attribute("run.outcome", "skipped_no_summaries")
        digests_published_total.add(1, {"outcome": "skipped_no_summaries"})
        return

    skipped = len(articles) - len(summaries)
    if skipped:
        logger.info("%d article(s) could not be summarized and were excluded.", skipped)
    span.set_attribute("run.articles_skipped", skipped)

    issue_title = f"EU AI Act News Digest — {today}"
    issue_body = (
        f"{digest}\n\n"
        f"---\n"
        f"*Automated daily digest. Sources: {', '.join(f['name'] for f in RSS_FEEDS)}. "
        f"Powered by DeepSeek AI via the EU AI Act News Agent.*"
    )

    issue_url = create_github_issue(issue_title, issue_body)
    span.set_attribute("run.outcome", "published")
    span.set_attribute("run.issue_url", issue_url)
    digests_published_total.add(1, {"outcome": "published"})


def main():
    setup_telemetry()
    try:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        logger.info("Starting EU AI Act news agent for %s", today)
        with tracer.start_as_current_span("news_agent.run"):
            run(today)
    finally:
        shutdown_telemetry()


if __name__ == "__main__":
    main()
