"""The FastAPI application (Steps 2.3-2.5, 3.2).

    uvicorn src.api.main:app --reload --no-access-log   # development, from the repo root
    python -m src.api.main                              # uses API_HOST and API_PORT from .env

The model is loaded once, at startup. If it cannot be loaded (missing file,
SHA-256 mismatch, bad thresholds) the server refuses to start, rather than
running without a model and failing every request. The decision store is
opened at startup too, but a database that is down only produces a warning:
see src/storage/store.py.

Every request is logged once, with its status, latency and request id:

    INFO src.api POST /score 200 9.8ms request_id=3f2a...
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from src.api.routes import health, score, transactions
from src.config import settings
from src.scoring import Scorer
from src.storage.store import DecisionStore, failure_reason, open_configured_store

log = logging.getLogger("src.api")

VALIDATION_ERROR_KEYS = ("type", "loc", "msg")
REQUEST_ID_HEADER = "X-Request-ID"
# A client-supplied id is copied into the logs, so only short, plain ids are kept.
SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


async def log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Log each request once with its status and latency, and tag the response with an id."""
    request_id = request.headers.get(REQUEST_ID_HEADER, "")
    if not SAFE_REQUEST_ID.fullmatch(request_id):
        request_id = uuid4().hex
    started = time.perf_counter()
    status = 500  # stays 500 if the route raises
    try:
        response = await call_next(request)
        status = response.status_code
    finally:
        log.info(
            "%s %s %d %.1fms request_id=%s",
            request.method,
            request.url.path,
            status,
            (time.perf_counter() - started) * 1000,
            request_id,
        )
    response.headers[REQUEST_ID_HEADER] = request_id
    return response


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 listing which fields are wrong and why, without echoing the values sent.

    FastAPI's default handler includes each bad input in the response. A NaN
    cannot be written as JSON, so that turned a bad request into a 500 error.
    """
    detail = [{key: error[key] for key in VALIDATION_ERROR_KEYS} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": detail})


async def store_unavailable_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    """503 when a read from the decision store fails. Scoring routes never get here."""
    log.error("decision store error: %s", failure_reason(exc))
    return JSONResponse(status_code=503, content={"detail": "decision store unavailable"})


async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """500 with a generic JSON body. The traceback goes to the server log, not the caller."""
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


def create_app(
    load_scorer: Callable[[], Scorer] = Scorer.load,
    open_store: Callable[[], DecisionStore | None] = open_configured_store,
) -> FastAPI:
    """Build the app. Tests pass their own loaders to avoid the real model and database."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logging.basicConfig(
            level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
        )
        app.state.scorer = load_scorer()
        app.state.store = open_store()
        try:
            yield
        finally:
            if app.state.store is not None:
                app.state.store.close()

    app = FastAPI(
        title=settings.app_name,
        description="Real-time fraud risk scoring: approve, review or block each transaction.",
        lifespan=lifespan,
    )
    app.middleware("http")(log_requests)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(SQLAlchemyError, store_unavailable_handler)
    app.add_exception_handler(Exception, unexpected_error_handler)
    app.include_router(health.router)
    app.include_router(score.router)
    app.include_router(transactions.router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    # access_log=False: log_requests already logs every request, with its latency.
    uvicorn.run(app, host=settings.api_host, port=settings.api_port, access_log=False)
