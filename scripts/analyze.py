"""Usage: uv run scripts/analyze.py AAPL [--url http://localhost:8001] [--session my-session]"""

import argparse
import json
import sys

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the research agent for a buy/hold/sell decision.")
    parser.add_argument("ticker")
    parser.add_argument("--url", default="http://localhost:8001")
    parser.add_argument("--session", default=None, help="Arize session id (groups runs together)")
    args = parser.parse_args()

    response = httpx.post(
        f"{args.url}/invoke",
        json={"ticker": args.ticker, "session_id": args.session},
        timeout=300,
    )
    print(json.dumps(response.json(), indent=2))
    return 0 if response.is_success else 1


if __name__ == "__main__":
    sys.exit(main())
