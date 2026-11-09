# syntax=docker/dockerfile:1
# One image, two entrypoints: the REST API (default) and the Streamlit dashboard.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RE_DATA_DIR=/app/data \
    RE_ARTIFACTS_DIR=/app/artifacts

WORKDIR /app

# Install dependencies first so code changes don't bust the dependency layer.
COPY pyproject.toml README.md ./
COPY src/retention_engine/__init__.py src/retention_engine/__init__.py
RUN pip install ".[all]" && pip uninstall -y retention-engine

COPY src ./src
COPY app ./app
COPY .streamlit ./.streamlit
RUN pip install --no-deps .

# Bake a ready-to-serve dataset + model into the image so containers start instantly.
# (Mount volumes over /app/data and /app/artifacts to use your own.)
RUN retention-engine run --provider template \
    && useradd --create-home --uid 1000 app \
    && chown -R app:app /app
USER app

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "retention_engine.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
