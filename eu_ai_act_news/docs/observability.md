# Observability

The agent is instrumented with [OpenTelemetry](https://opentelemetry.io/): traces, logs
and metrics, exported over OTLP/gRPC. `news_agent/telemetry.py` owns setup/shutdown;
every other module just acquires a tracer/logger the normal OTel way
(`trace.get_tracer(__name__)`, `logging.getLogger(__name__)`) and is a no-op until
telemetry is configured — so tests and any other import stay unaffected.

Because this is a batch job (one run, then exit) rather than a server, `main()` calls
`setup_telemetry()` before doing anything and `shutdown_telemetry()` in a `finally`, to
force-flush the last run's data before the process exits.

## What's traced

One root span per run (`news_agent.run`), with a span per pipeline stage nested inside it,
so a single trace shows the whole run's decision path:

- `collection.collect_relevant_articles` → `collection.fetch_feed` (per RSS feed: article
  counts, parse errors)
- `summarization.summarize_article` (per article: fetch outcome, generate/verify attempts
  as span events, final accept/reject decision and why) → `summarization.generate` /
  `summarization.verify` (the two DeepSeek calls) → `fetching.fetch_article_content`
- `publishing.issue_already_exists_today`, `publishing.create_github_issue`

Span attributes carry the *why*: rejection reasons, unsupported claims from the verifier,
retry counts, categories chosen, truncation flags. This is what makes a run explainable
after the fact — you can open one trace and see exactly which articles were rejected, at
which attempt, and for what reason.

Structured logs (the same `logger.info`/`logger.warning` calls used for console output)
are exported alongside the traces and automatically tagged with the active trace/span ID,
so a log line and its span can be cross-referenced in Grafana.

Metrics (`news_agent.articles.collected`, `.articles.processed` by outcome,
`.verification.attempts` by result, `.llm.call.duration`, `.digests.published` by reason)
support dashboards and alerts across runs, where a single trace only tells you about one.

## Running locally

```bash
docker compose -f observability/docker-compose.yml up -d
```

This starts an OTel Collector (OTLP receiver on `localhost:4317`/`4318`), Tempo (traces),
Loki (logs), Prometheus (metrics, via its remote-write receiver) and Grafana
(`localhost:3000`, anonymous admin access) with all three datasources pre-provisioned.

Then just run the agent — no env vars needed, the OTLP exporters default to
`http://localhost:4317`:

```bash
make run
```

Open Grafana → Explore:
- **Tempo**: search by service name `eu-ai-act-news-agent` for the run's trace.
- **Loki**: query `{service_name="eu-ai-act-news-agent"}` for the run's logs.
- **Prometheus**: query `news_agent_articles_processed_total` etc.

The Tempo↔Loki correlation and the exact metric/label names come from a fairly standard
OTel Collector → Tempo/Loki/Prometheus wiring, but this compose stack hasn't been run
end-to-end in this environment (no Docker available here) — if a datasource link doesn't
resolve, check the Grafana datasource config in
`observability/grafana/provisioning/datasources/` against the version of
Tempo/Loki/Grafana that actually got pulled.

## Pointing at a real backend (e.g. Grafana Cloud)

Nothing in the code is backend-specific. Set the standard OTel env vars before running,
and the exporters pick them up automatically:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="https://otlp-gateway-<region>.grafana.net/otlp"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic <base64 instance_id:api_key>"
```

(Or point `OTEL_EXPORTER_OTLP_ENDPOINT` at any other OTLP-compatible collector.) In
GitHub Actions, add these as repository secrets and pass them through as env vars in the
workflow step that runs the agent.
