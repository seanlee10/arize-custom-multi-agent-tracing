FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
COPY common ./common
COPY services ./services
RUN uv sync --locked --no-dev --package websearch-mcp
# OpenSSL (via cryptography, pulled in by mcp) can die with SIGILL probing ARM SME/SVE2
# features inside VMs on Apple M4 hosts (e.g. Colima). Disable the probing; costs nothing here.
ENV PATH="/app/.venv/bin:$PATH" OPENSSL_armcap=0 PORT=8000
EXPOSE 8000
CMD ["websearch-mcp"]
