# The API and Kafka consumer image, and the producer image (Steps 3.3, 4.6).
#
#   docker build -t fraud-detection-api .                                # API and consumer
#   docker build --target producer -t fraud-detection-producer .         # producer
#   docker run --rm -p 8000:8000 -e PERSIST_DECISIONS=false fraud-detection-api
#
# Needs models/fraud_model.joblib and models/model_metadata.json: run
# `python -m src.ml.train` first. The model is baked into the image, so an image
# tag always identifies exactly one model.

# ---------------------------------------------------------------- build stages
# Install the dependencies into a virtual environment that is copied into a final
# image on its own, leaving pip's caches and build files behind.
FROM python:3.10-slim-bookworm AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copied before the code, so editing src/ reuses this cached layer instead of
# reinstalling every library.
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# The producer adds a parquet reader on top of the same libraries.
FROM build AS build-producer
COPY requirements-producer.txt .
RUN pip install -r requirements-producer.txt

# ---------------------------------------------------------------- shared base
FROM python:3.10-slim-bookworm AS base

# libgomp1: the OpenMP runtime XGBoost uses for parallel scoring. Not in slim.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# A fixed, unprivileged user. If a service is ever compromised, the attacker is not root.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=docker

WORKDIR /app

# ------------------------------------------------------------------- producer
# Streams the test split into Kafka. The split is not in the image: docker-compose.yml
# mounts data/processed read-only. Arguments go straight to scripts/produce.py.
FROM base AS producer
COPY --from=build-producer /opt/venv /opt/venv
COPY src/ src/
COPY scripts/ scripts/
USER app
ENTRYPOINT ["python", "-m", "scripts.produce"]

# ------------------------------------------------------------- API / consumer
# The default target, so a plain `docker build .` still builds the API.
FROM base AS service
COPY --from=build /opt/venv /opt/venv
# Owned by root and only readable by `app`: a service cannot modify its own
# code or swap its model.
COPY src/ src/
COPY scripts/ scripts/
COPY models/fraud_model.joblib models/model_metadata.json models/

USER app

EXPOSE 8000

# Docker marks the container unhealthy if /health stops answering. Python
# instead of curl, which the slim image does not include. The 10s timeout allows
# for /health's database probe, which takes ~4s when the database has vanished:
# a slow answer still proves the API is alive. The consumer, which runs this image
# without a web server, switches it off in docker-compose.yml.
HEALTHCHECK --interval=15s --timeout=10s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=9)"]

# Exec form: uvicorn is PID 1 and receives `docker stop`'s SIGTERM directly, so
# it finishes in-flight requests and closes the database pool before exiting.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
