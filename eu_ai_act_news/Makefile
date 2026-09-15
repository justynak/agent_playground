.PHONY: install
install:
	uv sync --group dev

.PHONY: lint
lint:
	uv run ruff check .

.PHONY: test
test:
	uv run pytest tests/ -v -m "not integration"

.PHONY: coverage
coverage:
	uv run pytest tests/ -v -m "not integration" --cov --cov-report=term-missing --cov-report=html

.PHONY: integration_tests
integration_tests:
	uv run $(if $(wildcard .env),--env-file .env,) pytest tests/ -v -m integration

.PHONY: run
run:
	uv run python -m news_agent.main

.PHONY: check
check: lint test
