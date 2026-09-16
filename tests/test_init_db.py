"""Step 3.5 guard rails: the schema script."""

from scripts import init_db
from src.api.schemas import Decision, RiskResult
from src.storage.db import create_db_engine, create_session_factory, session_scope
from src.storage.repository import save_decisions


def sqlite_url(path):
    return f"sqlite:///{path.as_posix()}"


def test_creates_the_table_and_reports_it_empty(tmp_path, capsys):
    assert init_db.main(["--url", sqlite_url(tmp_path / "new.db")]) == 0
    assert "ready at sqlite:///" in capsys.readouterr().out
    assert (tmp_path / "new.db").exists()


def test_running_again_keeps_the_data_and_counts_it(tmp_path, capsys):
    url = sqlite_url(tmp_path / "decisions.db")
    init_db.main(["--url", url])

    engine = create_db_engine(url)
    block = RiskResult(
        transaction_id="tx-1",
        risk_score=0.99,
        decision=Decision.BLOCK,
        review_threshold=0.24,
        block_threshold=0.95,
        model_version="v-test",
    )
    with session_scope(create_session_factory(engine)) as session:
        save_decisions(session, [block])
    engine.dispose()

    capsys.readouterr()
    assert init_db.main(["--url", url]) == 0
    assert "1 row(s) (0 approve, 0 review, 1 block)" in capsys.readouterr().out


def test_an_unreachable_database_fails_without_printing_the_password(capsys):
    # Port 1 on this machine: nothing listens there.
    url = "postgresql+psycopg://fraud:s3cret-password@127.0.0.1:1/fraud"
    assert init_db.main(["--url", url]) == 1
    error = capsys.readouterr().err
    assert "could not set up postgresql+psycopg://fraud:***@127.0.0.1:1/fraud" in error
    assert "s3cret-password" not in error
