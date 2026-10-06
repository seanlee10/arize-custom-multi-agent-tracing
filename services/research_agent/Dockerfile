FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
COPY common ./common
COPY services ./services
RUN uv sync --locked --no-dev --package research-agent
# OpenSSL (via cryptography, pulled in by mcp) can die with SIGILL probing ARM SME/SVE2
# features inside VMs on Apple M4 hosts (e.g. Colima). Disable the probing; costs nothing here.
ENV PATH="/app/.venv/bin:$PATH" OPENSSL_armcap=0
EXPOSE 8001
CMD ["uvicorn", "research_agent.app:create_app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8001"]
