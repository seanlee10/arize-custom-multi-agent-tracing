FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
COPY common ./common
COPY services ./services
RUN uv sync --locked --no-dev --package decision-agent
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8002
CMD ["uvicorn", "decision_agent.app:create_app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8002"]
