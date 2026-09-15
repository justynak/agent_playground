"""Digest rendering and GitHub Issue publishing (incl. idempotency check)."""

import logging

import requests
from opentelemetry import trace

from .config import (
    CATEGORIES,
    GITHUB_REPOSITORY,
    GITHUB_TOKEN,
    ISSUES_PER_PAGE,
    MAX_ISSUE_PAGES,
)

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


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
    """Return True if an open issue whose title contains `today` already exists.

    Paginates through all open issues rather than only the first page, so a digest
    buried beyond the first page is still detected (otherwise a duplicate could be
    created for the same day).
    """
    with tracer.start_as_current_span("publishing.issue_already_exists_today") as span:
        span.set_attribute("digest.date", today)
        owner, repo = GITHUB_REPOSITORY.split("/", 1)
        api_url = f"https://api.github.com/repos/{owner}/{repo}/issues"
        headers = {
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
        }

        for page in range(1, MAX_ISSUE_PAGES + 1):
            resp = requests.get(
                api_url,
                headers=headers,
                params={"state": "open", "per_page": ISSUES_PER_PAGE, "page": page},
            )
            if resp.status_code != 200:
                span.set_attribute("github.error_status", resp.status_code)
                span.set_status(trace.StatusCode.ERROR, f"GitHub API returned {resp.status_code}")
                span.set_attribute("digest.exists", False)
                return False

            issues = resp.json()
            if any(today in issue["title"] for issue in issues):
                span.set_attribute("digest.exists", True)
                return True

            # A short page means there are no further pages to fetch.
            if len(issues) < ISSUES_PER_PAGE:
                span.set_attribute("digest.exists", False)
                return False
        span.set_attribute("digest.exists", False)
        return False


def create_github_issue(title: str, body: str) -> str:
    with tracer.start_as_current_span("publishing.create_github_issue") as span:
        span.set_attribute("issue.title", title)
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
                issue_url = resp.json()["html_url"]
                span.set_attribute("issue.url", issue_url)
                logger.info("Issue created: %s", issue_url)
                return issue_url
            if resp.status_code != 422:
                span.set_status(trace.StatusCode.ERROR, f"GitHub API returned {resp.status_code}")
                resp.raise_for_status()

        span.set_status(trace.StatusCode.ERROR, f"Failed to create issue: {resp.status_code}")
        raise RuntimeError(f"Failed to create issue: {resp.status_code} {resp.text}")
