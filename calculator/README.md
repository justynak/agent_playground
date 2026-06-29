# Calculator

A multi-agent calculator that evaluates arithmetic expressions by delegating every operation to an LLM-powered sub-agent. Built as a learning project for agentic AI architecture.

## What it does

Takes an arithmetic expression like `(2+3)*4` and produces the correct numeric result. Supported operators: `+`, `-`, `*`, `/`, with parentheses and decimal numbers. It deliberately never computes arithmetic directly — every single operation is routed through a sub-agent.

## How to use

```bash
cp .env.example .env        # add your DEEPSEEK_API_KEY
pip install -r requirements.txt
python main.py "(2+3)*4"
python main.py "100/4+7*3"
```

Output goes to stdout; tool call traces go to stderr.

## Architecture

```
main.py
  └── orchestrator.run(expression)
        ├── prefilter_syntax       — character whitelist + balanced-parens check
        ├── parse_expression       — builds an operation tree via Python's ast module
        ├── executable_operations  — returns the next wave of operations whose inputs are resolved
        └── spawn_evaluator_agent  ──► sub_agent.run(id, op, left, right)
                                           └── evaluate_operation  — does the arithmetic
```

### Orchestrator

The orchestrator is an LLM agent (DeepSeek, via OpenAI-compatible API) that drives the full evaluation loop. It calls tools in sequence:

1. Validate the expression with `prefilter_syntax`.
2. Parse it into an operation tree with `parse_expression`.
3. Call `executable_operations` to find which operations are ready (both inputs resolved).
4. Spawn a sub-agent for each ready operation via `spawn_evaluator_agent`.
5. Collect results and repeat from step 3 until the tree is fully evaluated.
6. Report the final result.

Operations at the same depth of the tree are independent and returned together by `executable_operations`, so the orchestrator can dispatch them as a wave. The LLM decides the pacing; the tools enforce the structure.

### Sub-agent

Each `spawn_evaluator_agent` call starts a fresh LLM agent whose only job is to call `evaluate_operation` with the given operands and return the result. The sub-agent does not share context with the orchestrator.

### Tools (pure Python, no LLM)

| Tool | What it does |
|---|---|
| `prefilter_syntax` | Rejects invalid characters and unbalanced parentheses |
| `parse_expression` | Converts the expression string to a nested operation tree |
| `executable_operations` | Given the tree and completed results, returns the next ready operations |
| `evaluate_operation` | Performs one arithmetic operation (`+`, `-`, `*`, `/`) |

### Safety limits

The orchestrator loop is capped at `MAX_LLM_TURNS = 100` turns. This is a safety net against a runaway LLM — legitimate evaluation of any reasonable expression completes well within this limit (worst case is roughly `2N + 4` turns for an expression with `N` operations). If 3 consecutive tool calls return errors the orchestrator bails early with an `Aborted:` message.

## Running tests

```bash
pytest tests/
```

Tests cover all pure-Python tools and the orchestrator's bail-on-consecutive-errors logic (LLM calls are mocked).
