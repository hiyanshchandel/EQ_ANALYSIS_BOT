# syntax=docker/dockerfile:1
# One image, two services: the MCP server and the bot pick their command in docker-compose.

# ---- build stage: resolve and install dependencies with uv ----------------------
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependencies first, project code second: code changes don't invalidate the
# (slow) dependency layer, so rebuilds take seconds.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev

# ---- runtime stage: no compilers, no uv, non-root user --------------------------
FROM python:3.12-slim
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/data && chown app:app /app/data
WORKDIR /app
COPY --from=builder --chown=app:app /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER app
CMD ["python", "-m", "eq_analyst.bot"]
