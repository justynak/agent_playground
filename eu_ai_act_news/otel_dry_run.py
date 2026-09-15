"""Manual, no-network smoke test for the OpenTelemetry wiring.

Not part of the test suite — run it by hand after `docker compose -f
observability/docker-compose.yml up -d` to see a real trace/logs/metrics show up in
Grafana without spending a real DeepSeek call or creating a real GitHub issue.

    uv run python otel_dry_run.py
"""

import os
from unittest.mock import MagicMock, patch

os.environ.setdefault("DEEPSEEK_API_KEY", "dry-run")
os.environ.setdefault("GITHUB_TOKEN", "dry-run")
os.environ.setdefault("GITHUB_REPOSITORY", "acme/repo")

from news_agent import main, summarization


def make_tool_call_response(args_json: str):
    tool_call = MagicMock()
    tool_call.function.arguments = args_json
    message = MagicMock()
    message.tool_calls = [tool_call]
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


fake_article = {
    "title": "Commission publishes new EU AI Act guidance",
    "url": "https://example.com/ai-act-guidance",
    "published": "2026-09-15T08:00:00+00:00",
    "summary": "The European Commission published guidance on GPAI obligations.",
    "potentially_relevant": True,
}

generate_response = make_tool_call_response(
    '{"category": "Legislative & Regulatory Updates", '
    '"summary": "The European Commission published new guidance clarifying GPAI obligations under the EU AI Act."}'
)
verify_response = make_tool_call_response('{"faithful": true, "unsupported_claims": []}')

with (
    patch("news_agent.main.issue_already_exists_today", return_value=False),
    patch("news_agent.main.collect_relevant_articles", return_value=[fake_article]),
    patch(
        "news_agent.summarization.fetch_article_content",
        return_value={"url": fake_article["url"], "content": "Full article text here.", "truncated": False},
    ),
    patch.object(
        summarization.client.chat.completions,
        "create",
        side_effect=[generate_response, verify_response],
    ),
    patch("news_agent.main.create_github_issue", return_value="https://github.com/acme/repo/issues/1"),
):
    main.main()

print("Dry run complete — check Grafana for a news_agent.run trace, correlated logs, and metrics.")
