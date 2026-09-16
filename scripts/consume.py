"""Score transactions from Kafka and record the decisions in Postgres (Step 4.4).

    python -m scripts.consume                          # run until Ctrl+C
    python -m scripts.consume --until-idle 10          # stop once the topic is drained
    python -m scripts.consume --group candidate-model  # read the whole topic again, separately

Needs the model (python -m src.ml.train), Kafka with its topics (make kafka)
and Postgres (docker compose up -d postgres).

If Postgres goes down, the consumer pauses and waits for it rather than
stopping, and no decision is lost.

Ctrl+C stops after the batch in hand: its decisions are recorded and its
offsets committed, so a restart continues exactly where this run stopped. During
a database outage the batch in hand is not committed, and is read again on restart.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

from aiokafka.errors import KafkaError

from src.config import settings
from src.streaming.consumer import ConsumeStats, run
from src.streaming.kafka import MissingTopicsError


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.consume",
        description="Score transactions from Kafka and record the decisions.",
    )
    parser.add_argument(
        "--until-idle",
        type=float,
        metavar="SECONDS",
        help="stop after this long with no new messages (default: run until stopped)",
    )
    parser.add_argument("--group", default=settings.kafka_consumer_group)
    parser.add_argument("--max-batch", type=int, default=settings.consumer_max_batch)
    parser.add_argument("--topic", default=settings.kafka_topic)
    parser.add_argument("--dlq-topic", default=settings.kafka_dlq_topic)
    parser.add_argument("--bootstrap-servers", default=settings.kafka_bootstrap_servers)
    args = parser.parse_args(argv)
    if args.until_idle is not None and args.until_idle <= 0:
        parser.error("--until-idle must be more than 0")
    if args.max_batch < 1:
        parser.error("--max-batch must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    async def consume_until_stopped() -> ConsumeStats:
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        # Ctrl+C (SIGINT) and `docker stop` (SIGTERM) finish the current batch
        # instead of abandoning it halfway.
        handled = [signal.SIGINT, signal.SIGTERM]
        previous = {
            signum: signal.signal(signum, lambda *_: loop.call_soon_threadsafe(stop.set))
            for signum in handled
        }
        try:
            return await run(
                stop=stop,
                until_idle=args.until_idle,
                group=args.group,
                topic=args.topic,
                dlq_topic=args.dlq_topic,
                bootstrap_servers=args.bootstrap_servers,
                max_batch=args.max_batch,
            )
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)

    try:
        stats = asyncio.run(consume_until_stopped())
    except MissingTopicsError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KafkaError as exc:
        print(f"Kafka error at {args.bootstrap_servers}: {exc}", file=sys.stderr)
        return 1

    print(stats.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
