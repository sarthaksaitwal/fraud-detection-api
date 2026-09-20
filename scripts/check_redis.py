"""Check that Redis answers, and show what it is holding (Step 5.1).

    python -m scripts.check_redis
    python -m scripts.check_redis --url redis://127.0.0.1:6379/0

Exits 1 if no server answers, so it can gate a script.
"""

from __future__ import annotations

import argparse
import sys

from redis.exceptions import RedisError

from src.config import settings
from src.features.redis_client import create_redis


def describe(client) -> str:
    """Version, memory in use, keys, and what Redis does when memory runs out."""
    info = client.info("server") | client.info("memory")
    policy = client.config_get("maxmemory-policy").get("maxmemory-policy", "unknown")
    return (
        f"redis {info['redis_version']}, {client.dbsize()} key(s), "
        f"{info['used_memory_human']} in use, maxmemory-policy {policy}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.check_redis",
        description="Check that Redis answers and show what it is holding.",
    )
    parser.add_argument("--url", default=settings.redis_url, help="default: REDIS_URL from .env")
    args = parser.parse_args(argv)

    client = create_redis(args.url)
    try:
        summary = describe(client)
    except RedisError as exc:
        print(f"no Redis answering at {args.url}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()

    print(f"Redis at {args.url}: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
