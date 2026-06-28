CALCULATOR_VENV := calculator/.venv
CALCULATOR_PYTHON := $(CALCULATOR_VENV)/bin/python

# ── calculator ────────────────────────────────────────────────────────────────

.PHONY: calculator-install
calculator-install:
	python3 -m venv $(CALCULATOR_VENV)
	$(CALCULATOR_PYTHON) -m pip install -q -r calculator/requirements.txt

.PHONY: calculator-lint
calculator-lint:
	$(CALCULATOR_PYTHON) -m ruff check calculator/

.PHONY: calculator-test
calculator-test:
	$(CALCULATOR_PYTHON) -m pytest calculator/tests/ -v

.PHONY: calculator-check
calculator-check: calculator-lint calculator-test

# ── top-level shortcuts ────────────────────────────────────────────────────────

.PHONY: install
install: calculator-install

.PHONY: check
check: calculator-check
