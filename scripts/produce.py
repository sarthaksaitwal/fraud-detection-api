"""Stream test-set transactions into Kafka (Step 4.3).

    python -m scripts.produce                         # every transaction, PRODUCER_RATE_PER_SEC/s
    python -m scripts.produce --limit 100 --rate 50
    python -m scripts.produce --rate 0                # as fast as Kafka accepts them
    python -m scripts.produce --start 1000            # resume where an earlier run stopped

Ctrl+C stops cleanly: messages already queued are delivered, and the summary
says which --start resumes after the last confirmed one.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

from aiokafka.errors import KafkaError

from src.config import settings
from src.streaming.kafka import MissingTopicsError
from src.streaming.producer import ProduceReport, load_transactions, run


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.produce",
        description="Stream test-set transactions into Kafka, in time order.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=settings.producer_rate_per_sec,
        help="transactions per second; 0 = as fast as possible (default: PRODUCER_RATE_PER_SEC)",
    )
    parser.add_argument("--limit", type=int, help="send at most this many (default: all)")
    parser.add_argument("--start", type=int, default=0, help="position to start from (default: 0)")
    parser.add_argument("--topic", default=settings.kafka_topic)
    parser.add_argument("--bootstrap-servers", default=settings.kafka_bootstrap_servers)
    args = parser.parse_args(argv)
    if args.rate < 0:
        parser.error("--rate must be 0 or more")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.start < 0:
        parser.error("--start must be 0 or more")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    try:
        transactions = load_transactions(args.start, args.limit)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    async def produce_until_interrupted() -> ProduceReport:
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        # Ctrl+C sets `stop` instead of killing the program mid-send, so queued
        # messages are still delivered and the report is accurate.
        previous = signal.signal(signal.SIGINT, lambda *_: loop.call_soon_threadsafe(stop.set))
        try:
            return await run(
                start=args.start,
                rate=args.rate,
                topic=args.topic,
                bootstrap_servers=args.bootstrap_servers,
                stop=stop,
                transactions=transactions,
            )
        finally:
            signal.signal(signal.SIGINT, previous)

    try:
        report = asyncio.run(produce_until_interrupted())
    except MissingTopicsError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KafkaError as exc:
        print(f"Kafka error at {args.bootstrap_servers}: {exc}", file=sys.stderr)
        return 1

    print(report.summary(args.topic))
    return 0


if __name__ == "__main__":
    sys.exit(main())
