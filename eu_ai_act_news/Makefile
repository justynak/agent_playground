.PHONY: install
install:
	uv sync --group dev

.PHONY: lint
lint:
	uv run ruff check .

.PHONY: test
test:
	uv run pytest tests/ -v

.PHONY: run
run:
	uv run python news_agent.py

.PHONY: check
check: lint test
