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
from redis import Redis
from sqlalchemy.exc import SQLAlchemyError

from src.api.routes import health, metrics, score, transactions
from src.config import settings
from src.features.redis_client import open_configured_redis
from src.features.velocity import open_configured_velocity
from src.observability.metrics import model_loaded, request_duration_seconds, requests_total
from src.scoring import Scorer
from src.storage.store import DecisionStore, StoreUnavailableError, open_configured_store

log = logging.getLogger("src.api")

VALIDATION_ERROR_KEYS = ("type", "loc", "msg")
METRICS_PATH = "/metrics"
REQUEST_ID_HEADER = "X-Request-ID"
# A client-supplied id is copied into the logs, so only short, plain ids are kept.
SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


def route_of(request: Request) -> str:
    """The route template this request matched, e.g. "/transactions/{transaction_id}".

    Never request.url.path: that would make every transaction id its own metric
    series, and the id is the caller's to choose. Requests that match no route
    share one label, so a scan for URLs cannot create series either.
    """
    route = request.scope.get("route")
    return getattr(route, "path", "unmatched")


async def observe_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Log and measure each request, and tag the response with an id."""
    request_id = request.headers.get(REQUEST_ID_HEADER, "")
    if not SAFE_REQUEST_ID.fullmatch(request_id):
        request_id = uuid4().hex
    started = time.perf_counter()
    status = 500  # stays 500 if the route raises
    try:
        response = await call_next(request)
        status = response.status_code
    finally:
        elapsed = time.perf_counter() - started
        route = route_of(request)
        log.info(
            "%s %s %d %.1fms request_id=%s",
            request.method,
            request.url.path,
            status,
            elapsed * 1000,
            request_id,
        )
        # A scrape measuring itself tells nobody anything, and it runs on a timer,
        # so it would drown the rates it is meant to report.
        if route != METRICS_PATH:
            requests_total.labels(method=request.method, route=route, status=status).inc()
            request_duration_seconds.labels(method=request.method, route=route).observe(elapsed)
    response.headers[REQUEST_ID_HEADER] = request_id
    return response


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 listing which fields are wrong and why, without echoing the values sent.

    FastAPI's default handler includes each bad input in the response. A NaN
    cannot be written as JSON, so that turned a bad request into a 500 error.
    """
    detail = [{key: error[key] for key in VALIDATION_ERROR_KEYS} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": detail})


async def store_unavailable_handler(request: Request, exc: Exception) -> JSONResponse:
    """503 when a read from the decision store fails. Scoring routes never get here.

    The store has already logged why (src/storage/store.py).
    """
    return JSONResponse(status_code=503, content={"detail": "decision store unavailable"})


async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """500 with a generic JSON body. The traceback goes to the server log, not the caller."""
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


def create_app(
    load_scorer: Callable[[], Scorer] = Scorer.load,
    open_store: Callable[[], DecisionStore | None] = open_configured_store,
    open_redis: Callable[[], Redis | None] = open_configured_redis,
) -> FastAPI:
    """Build the app. Tests pass their own loaders to avoid the real model and database."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logging.basicConfig(
            level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
        )
        app.state.scorer = load_scorer()
        model_loaded.labels(model_version=app.state.scorer.model_version).set(1)
        app.state.store = open_store()
        app.state.redis = open_redis()
        app.state.velocity = open_configured_velocity(app.state.redis)
        try:
            yield
        finally:
            if app.state.store is not None:
                app.state.store.close()
            if app.state.redis is not None:
                app.state.redis.close()

    app = FastAPI(
        title=settings.app_name,
        description="Real-time fraud risk scoring: approve, review or block each transaction.",
        lifespan=lifespan,
    )
    app.middleware("http")(observe_requests)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(SQLAlchemyError, store_unavailable_handler)
    app.add_exception_handler(StoreUnavailableError, store_unavailable_handler)
    app.add_exception_handler(Exception, unexpected_error_handler)
    app.include_router(health.router)
    app.include_router(metrics.router)
    app.include_router(score.router)
    app.include_router(transactions.router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    # access_log=False: observe_requests already logs every request, with its latency.
    uvicorn.run(app, host=settings.api_host, port=settings.api_port, access_log=False)
