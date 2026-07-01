.PHONY: install
install:
	uv sync --group dev

.PHONY: lint
lint:
	uv run ruff check .

.PHONY: test
test:
	uv run pytest tests/ -v -m "not integration"

.PHONY: integration_tests
integration_tests:
	uv run $(if $(wildcard .env),--env-file .env,) pytest tests/ -v -m integration

.PHONY: run
export EXPR
run:
ifndef EXPR
	$(error Usage: make run EXPR="(2+3)*4")
endif
	uv run python main.py "$$EXPR"

.PHONY: check
check: lint test
