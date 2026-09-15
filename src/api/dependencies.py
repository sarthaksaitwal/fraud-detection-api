"""Objects the routes receive from FastAPI instead of importing globals (Step 2.3)."""

from __future__ import annotations

from fastapi import Request

from src.scoring import Scorer


def get_scorer(request: Request) -> Scorer:
    """The Scorer loaded once at startup (see src/api/main.py)."""
    return request.app.state.scorer
