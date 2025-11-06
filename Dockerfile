# Use Python 3.11 (latest stable, similar to Colab)
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory
WORKDIR /workspace

# Copy project files
COPY pyproject.toml requirements.txt ./
COPY revdeq/ ./revdeq/
COPY train.py inference.py ./
COPY configs/ ./configs/

# Install dependencies using uv
RUN uv pip install --system -r requirements.txt

# Install package in development mode
RUN uv pip install --system -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=""

# Default command
CMD ["/bin/bash"]

