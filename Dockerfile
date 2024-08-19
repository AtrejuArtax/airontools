# docker build -t airontools-linux .
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base
ENV TZ="UTC"
RUN apt update && \
    apt install --no-install-recommends -y curl ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    update-ca-certificates

FROM base as build_base
# Install dependencies.
ENV PYTHONUNBUFFERED True
ENV PATH="/root/.local/bin:$PATH"
RUN apt update && \
    apt install --no-install-recommends -y \
    curl \
    python3 \
    python3-dev \
    python-is-python3 \
    gcc \
    pkg-config \
    libhdf5-dev \
    pipx \
    && rm -rf /var/lib/apt/lists/* \
    && pipx install poetry

FROM build_base as build_airontools
# Copy the code to the container image
WORKDIR /app
COPY airontools /app/airontools
COPY setup.py /app/setup.py
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md
COPY LICENSE /app/LICENSE
COPY .pypirc /app/.pypirc
# Install packages, build the wheel and publish it
RUN poetry config virtualenvs.in-project true && \
    poetry install && \
    poetry export --without-hashes --format=requirements.txt > requirements.txt && \
    poetry run python setup.py bdist_wheel && \
    poetry run python -m twine upload dist/* --config-file .pypirc