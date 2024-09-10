# Base image with Python 3.8 and CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python3 as the default Python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# PyTorch
RUN pip install --upgrade pip
RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

# Working directory
WORKDIR /imm
COPY pyproject.toml /imm/
COPY . /imm/

# Install
RUN pip install --ignore-installed --no-cache-dir .

