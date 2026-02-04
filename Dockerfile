# ==============================================================================
# Build Arguments
# ==============================================================================
ARG CUDA_VERSION=12.9.1
ARG PYTHON_VERSION=3.10
ARG BUILDER_IMAGE=nvcr.io/nvidia/ai-workbench/python-cuda129:1.0.1
ARG RUNTIME_IMAGE=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04
ARG WORK_DIR=/app

# ==============================================================================
# Builder Stage
# ==============================================================================
FROM ${BUILDER_IMAGE} AS builder

ARG WORK_DIR

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_SYSTEM_PYTHON=true \
    UV_PYTHON_PREFERENCE="only-system" \
    UV_COMPILE_BYTECODE=1

WORKDIR ${WORK_DIR}

# Install dependencies first (cache-friendly)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Copy source and install project
COPY mini_tgi ./mini_tgi
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ==============================================================================
# Runtime Stage
# ==============================================================================
FROM ${RUNTIME_IMAGE} AS runtime

ARG PYTHON_VERSION
ARG WORK_DIR
ARG USERNAME=appuser
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

# Install Python runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user first
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home ${USERNAME} \
    && mkdir -p ${WORK_DIR}

# Copy virtual environment and source code
COPY --from=builder --chown=${USER_UID}:${USER_GID} ${WORK_DIR}/.venv ${WORK_DIR}/.venv
COPY --from=builder --chown=${USER_UID}:${USER_GID} ${WORK_DIR}/mini_tgi ${WORK_DIR}/mini_tgi

# Environment configuration
ENV VIRTUAL_ENV=${WORK_DIR}/.venv \
    PATH="${WORK_DIR}/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

USER ${USERNAME}
WORKDIR ${WORK_DIR}

CMD ["mini-tgi", "serve"]
