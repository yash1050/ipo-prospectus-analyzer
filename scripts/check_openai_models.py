"""
List OpenAI models visible to an API key (GET https://api.openai.com/v1/models).

Usage:
  python scripts/check_openai_models.py
  python scripts/check_openai_models.py --api-key sk-...
  set OPENAI_API_KEY=... && python scripts/check_openai_models.py

Loads .env from the project root when present.
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_URL = "https://api.openai.com/v1/models"


def fetch_models(api_key: str) -> list[str]:
    req = urllib.request.Request(
        MODELS_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="GET",
    )
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
        body = json.loads(resp.read().decode())
    data = body.get("data") or []
    return sorted({item["id"] for item in data if isinstance(item, dict) and "id" in item})


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="List OpenAI model IDs available to the given API key."
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=os.getenv("OPENAI_API_KEY", "").strip() or None,
        help="OpenAI API key (defaults to OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print model ids as JSON array",
    )
    args = parser.parse_args()

    if not args.api_key:
        print(
            "Error: No API key. Set OPENAI_API_KEY or pass --api-key.",
            file=sys.stderr,
        )
        return 1

    try:
        ids = fetch_models(args.api_key)
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode()
        except OSError:
            detail = str(e)
        print(f"HTTP {e.code}: {detail}", file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"Request failed: {e.reason}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(ids, indent=2))
    else:
        print(f"Models available to this key ({len(ids)}):\n")
        for mid in ids:
            print(mid)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
