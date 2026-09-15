"""Digest rendering and GitHub Issue publishing (incl. idempotency check)."""

import requests

from .config import (
    CATEGORIES,
    GITHUB_REPOSITORY,
    GITHUB_TOKEN,
    ISSUES_PER_PAGE,
    MAX_ISSUE_PAGES,
)


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
            return False

        issues = resp.json()
        if any(today in issue["title"] for issue in issues):
            return True

        # A short page means there are no further pages to fetch.
        if len(issues) < ISSUES_PER_PAGE:
            return False
    return False


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
