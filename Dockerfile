# The scoring API (Step 3.3).
#
#   docker build -t fraud-detection-api .
#   docker run --rm -p 8000:8000 -e PERSIST_DECISIONS=false fraud-detection-api
#
# Needs models/fraud_model.joblib and models/model_metadata.json: run
# `python -m src.ml.train` first. The model is baked into the image, so an image
# tag always identifies exactly one model.

# ---------------------------------------------------------------- build stage
# Installs the dependencies into a virtual environment that is copied into the
# final image on its own, leaving pip's caches and build files behind.
FROM python:3.10-slim-bookworm AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copied before the code, so editing src/ reuses this cached layer instead of
# reinstalling every library.
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# -------------------------------------------------------------- runtime stage
FROM python:3.10-slim-bookworm

# libgomp1: the OpenMP runtime XGBoost uses for parallel scoring. Not in slim.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# A fixed, unprivileged user. If the API is ever compromised, the attacker is not root.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=docker

COPY --from=build /opt/venv /opt/venv

WORKDIR /app
# Owned by root and only readable by `app`: the service cannot modify its own
# code or swap its model.
COPY src/ src/
COPY models/fraud_model.joblib models/model_metadata.json models/

USER app

EXPOSE 8000

# Docker marks the container unhealthy if /health stops answering. Python
# instead of curl, which the slim image does not include.
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

# Exec form: uvicorn is PID 1 and receives `docker stop`'s SIGTERM directly, so
# it finishes in-flight requests and closes the database pool before exiting.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
