"""Create the decisions table and report what it holds (Step 3.5).

    python -m scripts.init_db                          # the database in DATABASE_URL
    python -m scripts.init_db --url sqlite:///local.db

The API creates the table itself on first use, so this is never required. It
is a quick way to check that DATABASE_URL points at a database that answers,
and to create the schema up front where the service should not be allowed to.

It only creates tables that are missing; it never alters or drops one. A
changed column needs a migration tool such as Alembic.
"""

from __future__ import annotations

import argparse
import sys

from sqlalchemy.exc import SQLAlchemyError

from src.storage.db import create_db_engine, create_schema, create_session_factory, session_scope
from src.storage.repository import count_decisions
from src.storage.store import failure_reason


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.init_db",
        description="Create the decisions table if it is missing and report its contents.",
    )
    parser.add_argument("--url", help="database URL (default: DATABASE_URL from .env)")
    args = parser.parse_args(argv)

    engine = create_db_engine(args.url)
    # Never print a password, even the demo one.
    where = engine.url.render_as_string(hide_password=True)
    try:
        create_schema(engine)
        with session_scope(create_session_factory(engine)) as session:
            counts = count_decisions(session)
    except SQLAlchemyError as exc:
        print(f"could not set up {where}: {failure_reason(exc)}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()

    breakdown = ", ".join(f"{count} {decision.value}" for decision, count in counts.items())
    print(f"decisions table ready at {where}: {sum(counts.values())} row(s) ({breakdown})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
