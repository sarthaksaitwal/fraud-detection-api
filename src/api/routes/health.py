"""GET /health: is the service up, which model is it serving, is it recording?"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from src.api.dependencies import get_scorer, get_store, get_velocity
from src.api.schemas import HealthResponse
from src.features.velocity import VelocityStore
from src.scoring import Scorer
from src.storage.store import DecisionStore

router = APIRouter(tags=["health"])


def store_status(store: DecisionStore | None) -> str:
    if store is None:
        return "disabled"
    return "ok" if store.ping() else "unavailable"


def velocity_status(velocity: VelocityStore | None) -> str:
    """Asks the store, not Redis: while resting after a failure it answers instantly."""
    if velocity is None:
        return "disabled"
    return "ok" if velocity.ping() else "unavailable"


# The status stays "ok" when the database or Redis is down: scoring still works,
# and a container orchestrator restarting the API would not bring either back.
@router.get("/health", response_model=HealthResponse)
def health(
    scorer: Annotated[Scorer, Depends(get_scorer)],
    store: Annotated[DecisionStore | None, Depends(get_store)],
    velocity: Annotated[VelocityStore | None, Depends(get_velocity)],
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_version=scorer.model_version,
        review_threshold=scorer.review_threshold,
        block_threshold=scorer.block_threshold,
        decision_store=store_status(store),
        velocity=velocity_status(velocity),
    )
