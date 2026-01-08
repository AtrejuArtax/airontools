# docker build -t airontools .
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ="UTC"
ENV PYTHONUNBUFFERED=True
ENV PATH="/root/.local/bin:$PATH"
RUN apt update && \
    apt upgrade -y
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install dependencies.
FROM base AS build_base
RUN apt install --no-install-recommends -y \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update
RUN apt install --no-install-recommends -y \
    curl \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    build-essential \
    gcc \
    pkg-config \
    libhdf5-dev \
    default-jdk ca-certificates \
    git \
    mono-mcs \
    && update-ca-certificates
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3 && \
    python3 -m pip install poetry==2.2.1 && \
    pip install --upgrade twine && \
	echo "$$(python3 -m site --user-base)/bin" >> $$GITHUB_PATH
RUN rm -rf /usr/lib/python3.*/ensurepip

# Copy the code to the container image
FROM build_base AS build_airontools
WORKDIR /app
COPY airontools /app/airontools
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md
COPY LICENSE /app/LICENSE
COPY .pypirc /app/.pypirc

# Install packages, build the wheel and publish it
RUN poetry config virtualenvs.in-project true && \
    poetry env use 3.11 && \
    poetry self add poetry-plugin-export && \
    poetry install && \
    poetry build && \
    python -m twine upload dist/*.whl --config-file .pypirc --verbose