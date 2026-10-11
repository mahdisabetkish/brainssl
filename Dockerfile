# syntax=docker/dockerfile:1.7
#
# Training and evaluation image. One Dockerfile, two builds:
#   docker build -t brainssl:cpu .
#   docker build --build-arg TORCH_VARIANT=cu126 -t brainssl:gpu .
#
# The image carries the environment and the package; data and runs are
# mounted from the host (see compose.yaml), so nothing large is baked in.

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim-bookworm AS base

ARG TORCH_VARIANT=cpu
ARG TORCH_VERSION=2.7.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

RUN apt-get update \
 && apt-get install -y --no-install-recommends tini procps \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so code changes do not invalidate this layer.
COPY requirements-lock.txt ./
RUN pip install "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}" \
 && pip install -r requirements-lock.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

COPY scripts ./scripts
COPY assets ./assets
COPY bin ./bin
COPY docker/entrypoint.sh /usr/local/bin/entrypoint
RUN chmod +x /usr/local/bin/entrypoint \
 && useradd --create-home --uid 1000 brainssl \
 && mkdir -p /app/data /app/runs \
 && chown -R brainssl /app

USER brainssl
ENTRYPOINT ["tini", "--", "entrypoint"]
CMD ["info"]

LABEL org.opencontainers.image.title="brainssl" \
      org.opencontainers.image.description="Label-efficient learning on brain MRI and fMRI" \
      org.opencontainers.image.source="https://github.com/mahdisabetkish/brainssl" \
      org.opencontainers.image.licenses="MIT"
