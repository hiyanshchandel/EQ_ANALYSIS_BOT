.PHONY: install lint format typecheck test check mcp bot eval prompts up down

install:      ## Install all dependencies (including dev tools)
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy src evals scripts

test:
	uv run pytest

check: lint typecheck test   ## Everything CI runs

mcp:          ## Run the MCP market-data server
	uv run python -m eq_analyst.mcp_server

bot:          ## Run the Telegram bot (start `make mcp` first)
	uv run python -m eq_analyst.bot

eval:         ## Run the golden-set experiment in Langfuse
	uv run python -m evals.run_experiment

prompts:      ## Push prompts to Langfuse prompt management
	uv run python scripts/sync_prompts.py

up:           ## Run everything in Docker
	docker compose up --build -d

down:
	docker compose down
