# syntax=docker/dockerfile:1.7
# Slim runtime image for the v2 Python package (pure ONNX).
FROM python:3.13-slim AS runtime

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY packages/nakdimon ./packages/nakdimon
COPY spec ./spec
RUN uv pip install --system ./packages/nakdimon

ENTRYPOINT ["diacritize"]
CMD ["-"]
