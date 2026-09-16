"""Check that Kafka answers and the project's topics exist (Step 4.1).

    python -m scripts.check_kafka
    python -m scripts.check_kafka --bootstrap-servers 127.0.0.1:9092

Exits 1 if no broker answers or a topic is missing, so it can gate a script.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from aiokafka.errors import KafkaError

from src.config import settings
from src.streaming.kafka import MissingTopicsError, check_topics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.check_kafka",
        description="Check that Kafka answers and the project's topics exist.",
    )
    parser.add_argument(
        "--bootstrap-servers",
        default=settings.kafka_bootstrap_servers,
        help="default: KAFKA_BOOTSTRAP_SERVERS from .env",
    )
    args = parser.parse_args(argv)

    try:
        counts = asyncio.run(check_topics(bootstrap_servers=args.bootstrap_servers))
    except MissingTopicsError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KafkaError as exc:
        print(f"no Kafka answering at {args.bootstrap_servers}: {exc}", file=sys.stderr)
        return 1

    topics = ", ".join(
        f"{topic} ({count} partition{'s' if count != 1 else ''})" for topic, count in counts.items()
    )
    print(f"Kafka at {args.bootstrap_servers}: {topics}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
