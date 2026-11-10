.PHONY: check lint test sync

sync:
	uv sync

lint:
	uv run ruff check .

test:
	uv run pytest

check: lint test
