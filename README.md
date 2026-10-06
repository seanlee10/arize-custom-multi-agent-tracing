# Multi-agent financial analysis with manual Arize tracing

**If context is propagated correctly, spans reported separately by many services become one trace,
whatever language or framework each service uses.**

In this sample, three containers turn a ticker into a **buy / hold / sell** decision. No collector
or shared process stitches their spans together. Each container exports its own spans straight to
Arize AX, and Arize assembles them into one trace from two IDs on every span: `trace_id` and
`parent_id`.

![One trace assembled from three containers](images/arize-trace.png)

```
client ─POST /invoke─► research-agent ──MCP (traceparent+baggage in _meta)──► websearch-mcp ─► Tavily
                            │
                            └──POST /invoke (traceparent+baggage headers)──► decision-agent
                                                                              ├─ force_index
                                                                              ├─ bollinger_bands
                                                                              └─ insider_trades (yfinance)
```

## Why it works: the contract between services

At every hop, the caller sends its current context and the callee continues from it. That
contract is made of open standards, not anything specific to Python, these frameworks, or Arize:

| What crosses the boundary | Standard | What it gives you |
|---|---|---|
| `traceparent` | [W3C Trace Context](https://www.w3.org/TR/trace-context/) | Same `trace_id`; the callee's span is parented to the caller's span |
| `baggage` | [W3C Baggage](https://www.w3.org/TR/baggage/) | Request-level values (`session.id`, `user.id`, ticker) that every service copies onto its spans |
| Span data | [OTLP](https://opentelemetry.io/docs/specs/otlp/) + [OpenInference](https://github.com/Arize-ai/openinference) attributes | Any SDK can export spans, and Arize renders them as AGENT / LLM / TOOL spans |
| Resource attribute `openinference.project.name` | OpenInference | All services report into the same Arize project |

The carrier can be anything that reaches the other side. Here it's HTTP headers for the
agent-to-agent call, and the request's `_meta` field for MCP, which has no per-call headers.

Every service in this repo happens to be Python and the agents use no framework. A service in
TypeScript, Java or Go, or one built on LangGraph or another agent framework, joins the same trace
by doing the same three things:
1. Read `traceparent` and `baggage` from the incoming request.
2. Create its spans as children of that context, with the OpenInference attributes.
3. Export them over OTLP to the same Arize project.

Two things break the single trace in practice, and this repo handles both:
- **Request attributes are lost at the boundary.** OpenInference's `using_attributes` (session,
  user, metadata, tags) lives only in the current process, and `traceparent` doesn't carry it. This
  sample also sends those values as W3C baggage, and each receiver rebuilds them
  (`fin_tracing/propagation.py`).
- **Library spans take over the tree.** FastAPI and the MCP SDK emit their own spans once a tracer
  provider exists, and those can become the trace root. This sample records only its own spans
  (`FinTracingOnlyProvider`); see the Code map below.

## Run

```bash
cp .env.example .env        # fill in ANTHROPIC_API_KEY, TAVILY_API_KEY, ARIZE_SPACE_ID, ARIZE_API_KEY
docker compose up --build   # or `docker-compose up --build` if Compose is a standalone binary
uv run scripts/analyze.py AAPL
```

The response includes a `trace_id`; search for it in your Arize project (`ARIZE_PROJECT_NAME`,
default `financial-multi-agent`). Without Arize keys, spans are printed to each container's logs.

Every run makes real, billed Claude and Tavily calls (an analysis takes roughly a minute).

More runs and housekeeping:

```bash
uv run scripts/analyze.py MSFT --session demo-1   # same --session groups runs into one Arize session
uv run scripts/analyze.py NVDA --url http://localhost:8001
docker compose logs -f research-agent             # follow one service
docker compose down                               # stop everything
```

### Apple Silicon (M4) + Colima note

The research-agent and websearch-mcp images set `OPENSSL_armcap=0`. Without it, those containers
exit immediately with code 132 (SIGILL) on Apple M4 hosts running a VM such as Colima: OpenSSL,
loaded via `cryptography` (a dependency of `mcp`), probes ARM SME/SVE2 CPU features the VM exposes
but cannot execute. The setting only disables that probing and has no practical cost here.

## What to look for in Arize

- One trace: `research_agent` (AGENT) → LLM / `web_search` (client) → `web_search` (server, other container)
  → `invoke_decision_agent` (CHAIN) → `decision_agent` (AGENT) → LLM / indicator TOOL spans → `submit_decision`.
- Every span carries the same `session.id`, `user.id`, `metadata`, `tag.tags` and `finagent.ticker`,
  even across containers — baggage carries them and each service rebuilds `using_attributes`.
- LLM spans: input/output messages including tool calls, token counts, prompt template + version.
- Filter on `finagent.decision`, `finagent.bollinger.signal`, `finagent.force_index.trend`, ...

In the trace at the top (one AAPL run), each of the research agent's four `web_search` calls
contains the MCP server's own span, and the decision agent, from a different container, sits under
`invoke_decision_agent`. Arize also builds an Agent Graph from the same spans:

![Agent graph in Arize](images/arize-agent-graph.png)

## Code map

- `common/fin_tracing` — all tracing code: provider/exporter, span helpers, Anthropic→OpenInference
  mapping, cross-process propagation.
- `services/websearch_mcp` — MCP server (`MCPServer`, streamable HTTP) with the `web_search` tool.
- `services/research_agent` — entry point; Claude tool loop over MCP tools; calls the decision agent.
- `services/decision_agent` — Claude tool loop over local indicator tools; `submit_decision` ends it.

FastAPI and the MCP Python SDK emit their own OpenTelemetry spans whenever a tracer provider is
configured. To keep the trace purely manual, `fin_tracing` records only its own tracer and hands
every other library a pass-through no-op tracer (see `FinTracingOnlyProvider`).

## Tests

```bash
uv sync
uv run pytest
```

`tests/test_end_to_end.py` runs all three services in-process and asserts the single-trace shape.
