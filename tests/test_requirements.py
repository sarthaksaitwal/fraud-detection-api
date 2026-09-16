"""Step 3.3 guard rails: the serving image installs what the model was trained with."""

import re

from src.config import PROJECT_ROOT
from src.ml.artifact import TRACKED_LIBRARIES

REQUIREMENTS = PROJECT_ROOT / "requirements-api.txt"
# name[extras]==version
PIN = re.compile(r"(?P<name>[A-Za-z0-9_.-]+)(\[[^\]]*\])?==(?P<version>\S+)")
# Same Python package, packaged without the CUDA libraries.
DISTRIBUTION_ALIASES = {"xgboost-cpu": "xgboost"}


def requirement_lines():
    lines = REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def pins():
    found = {}
    for match in map(PIN.fullmatch, requirement_lines()):
        if match:
            name = match["name"].lower()
            found[DISTRIBUTION_ALIASES.get(name, name)] = match["version"]
    return found


def test_every_serving_requirement_is_pinned_exactly():
    unpinned = [line for line in requirement_lines() if not PIN.fullmatch(line)]
    assert unpinned == []


def test_model_libraries_match_the_training_environment():
    """Retrain with upgraded libraries and forget the image, and this fails."""
    installed = {name: module.__version__ for name, module in TRACKED_LIBRARIES.items()}
    pinned = {name: pins().get(name) for name in TRACKED_LIBRARIES}
    assert pinned == installed


def test_the_database_driver_is_installed():
    # SQLAlchemy only imports the driver when it first connects, so a missing one
    # would not show up until the container was already running.
    assert "psycopg" in pins()
