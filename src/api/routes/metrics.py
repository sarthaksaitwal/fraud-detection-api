"""GET /metrics: what a Prometheus server scrapes (Step 6.1)."""

from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest

router = APIRouter(tags=["metrics"])


# include_in_schema=False: /metrics is for a scraper, not for API clients, and its
# text format is not JSON, so it would only clutter the OpenAPI docs.
@router.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Every metric this process holds, in Prometheus' text format.

    Plain `def`: generate_latest walks the registry and formats text, which is
    CPU work, so FastAPI runs it in a worker thread rather than on the event
    loop where a slow scrape would delay scoring.
    """
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
