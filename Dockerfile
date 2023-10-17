FROM python:3.9.18-slim-bullseye as base
ENV TZ="UTC"
RUN apt update && \
    apt install --no-install-recommends -y curl ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    update-ca-certificates

FROM base as build_base
# Install dependencies.
ENV PYTHONUNBUFFERED True
ENV PATH="/root/.local/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && apt-get update  \
    && apt-get install -y build-essential \
    && apt-get -y install gcc mono-mcs \
    && rm -rf /var/lib/apt/lists/*

FROM build_base as build_airontools
# Copy the code to the container image
WORKDIR /app
COPY . ./
# Mount credentials to install dependencies from private packages
RUN poetry install