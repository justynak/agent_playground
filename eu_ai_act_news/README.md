# EU AI Act News Agent

A daily AI agent that monitors major news sources for EU AI Act developments and publishes a structured digest as a GitHub Issue.

Runs automatically every morning via GitHub Actions. If nothing relevant happened, no issue is created.

## Why

The EU AI Act is the world's first comprehensive AI regulation — and it's moving fast. Compliance deadlines, implementing acts, guidance from the EU AI Office, enforcement decisions: staying current manually is a part-time job. This agent does it automatically and keeps a searchable archive in Issues.

## How it works

1. Scans 9 RSS feeds for articles mentioning the EU AI Act published in the last 24 hours, deduping articles that appear in more than one feed
2. Fetches the full text of each relevant article and asks DeepSeek to produce a structured, categorized summary for that single article (title, category, summary) — one model call per article, not one call for the whole digest
3. A second, independent DeepSeek call fact-checks the summary against the article's full text ("LLM-as-judge"), flagging any claim not directly supported by the source
4. If verification fails, the summary is regenerated with the specific unsupported claims fed back to the model, up to 3 attempts total. An article that never passes verification is rejected and excluded from the digest rather than published unverified
5. Articles that fail to fetch, or whose summary can't be parsed into a valid category, are skipped and logged rather than corrupting the whole run
6. Deterministically renders the verified per-article summaries into a Markdown digest grouped by: Key Developments, Legislative & Regulatory Updates, Industry & Compliance, Research & Expert Opinion, plus a Sources section
7. Publishes the digest as a GitHub Issue — skips creation if nothing relevant was found or no summaries could be produced

The agent uses an allowlisted set of domains to prevent prompt injection via fetched article content.

## Sources monitored

| Source | Focus |
|--------|-------|
| Euractiv | EU policy and legislation |
| European Parliament | Official EP news |
| Politico Europe | EU politics and regulation |
| Reuters Technology | Breaking tech news |
| TechCrunch | AI industry |
| The Verge | Tech policy |
| Wired | Technology and society |
| BBC Technology | Mainstream tech coverage |
| VentureBeat AI | AI industry developments |

## Setup

1. Fork this repository
2. Add `DEEPSEEK_API_KEY` as a repository secret (Settings → Secrets → Actions)
3. The `GITHUB_TOKEN` secret is provided automatically by GitHub Actions

The workflow runs daily at 8:00 AM UTC. You can also trigger it manually from the Actions tab.

## Tech stack

- **Agent framework**: OpenAI-compatible tool calling (DeepSeek `deepseek-chat`)
- **Scheduling**: GitHub Actions cron
- **Output**: GitHub Issues with labels `eu-ai-act`, `automated-digest`
- **Dependencies**: `feedparser`, `beautifulsoup4`, `requests`, `openai`
