"""Objects the routes receive from FastAPI instead of importing globals (Steps 2.3, 3.2)."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request

from src.scoring import Scorer
from src.storage.store import DecisionStore


def get_scorer(request: Request) -> Scorer:
    """The Scorer loaded once at startup (see src/api/main.py)."""
    return request.app.state.scorer


def get_store(request: Request) -> DecisionStore | None:
    """The decision store opened at startup, or None when recording is switched off."""
    return request.app.state.store


def require_store(store: Annotated[DecisionStore | None, Depends(get_store)]) -> DecisionStore:
    """For routes that only read the store, and so mean nothing without one."""
    if store is None:
        raise HTTPException(status_code=503, detail="decisions are not being recorded")
    return store
