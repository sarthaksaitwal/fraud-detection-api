"""Steps 3.3 and 4.6 guard rails: the images install what the model was trained with."""

import re

import pytest

from src.config import PROJECT_ROOT
from src.ml.artifact import TRACKED_LIBRARIES

SERVICE_REQUIREMENTS = PROJECT_ROOT / "requirements-api.txt"
PRODUCER_REQUIREMENTS = PROJECT_ROOT / "requirements-producer.txt"
# name[extras]==version
PIN = re.compile(r"(?P<name>[A-Za-z0-9_.-]+)(\[[^\]]*\])?==(?P<version>\S+)")
# `-r other-file.txt`: include another requirements file.
INCLUDE = re.compile(r"-r\s+(?P<path>\S+)")
# Same Python package, packaged without the CUDA libraries.
DISTRIBUTION_ALIASES = {"xgboost-cpu": "xgboost"}


def requirement_lines(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def pins(path=SERVICE_REQUIREMENTS):
    """Every pinned package, following `-r` includes."""
    found = {}
    for line in requirement_lines(path):
        if include := INCLUDE.fullmatch(line):
            found |= pins(path.parent / include["path"])
        elif match := PIN.fullmatch(line):
            name = match["name"].lower()
            found[DISTRIBUTION_ALIASES.get(name, name)] = match["version"]
    return found


@pytest.mark.parametrize(
    "path", [SERVICE_REQUIREMENTS, PRODUCER_REQUIREMENTS], ids=lambda p: p.name
)
def test_every_image_requirement_is_pinned_exactly(path):
    loose = [
        line
        for line in requirement_lines(path)
        if not PIN.fullmatch(line) and not INCLUDE.fullmatch(line)
    ]
    assert loose == []


def test_model_libraries_match_the_training_environment():
    """Retrain with upgraded libraries and forget the image, and this fails."""
    installed = {name: module.__version__ for name, module in TRACKED_LIBRARIES.items()}
    pinned = {name: pins().get(name) for name in TRACKED_LIBRARIES}
    assert pinned == installed


def test_the_database_driver_and_the_kafka_and_redis_clients_are_installed():
    # SQLAlchemy only imports the driver when it first connects, so a missing one
    # would not show up until the container was already running.
    assert {"psycopg", "aiokafka", "redis"} <= pins().keys()


def test_the_producer_image_has_everything_the_service_image_has_plus_parquet():
    service, producer = pins(SERVICE_REQUIREMENTS), pins(PRODUCER_REQUIREMENTS)
    assert service.items() <= producer.items()
    assert "pyarrow" in producer
    assert "pyarrow" not in service
