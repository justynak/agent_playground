.PHONY: install
install:
	uv sync --group dev

.PHONY: lint
lint:
	uv run ruff check .

.PHONY: test
test:
	uv run pytest tests/ -v -m "not integration"

.PHONY: check
check: lint test
