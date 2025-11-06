# Use Python 3.11 (latest stable, similar to Colab)
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package installer)
# uv requires cargo/rust, so we'll use pip as fallback or install rust toolchain
# Option 1: Try to install uv via pip (simpler, but slower)
RUN pip install --upgrade pip && \
    pip install uv || \
    (echo "Installing rust toolchain for uv..." && \
     curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
     export PATH="/root/.cargo/bin:$PATH" && \
     curl -LsSf https://astral.sh/uv/install.sh | sh)

ENV PATH="/root/.cargo/bin:/root/.local/bin:$PATH"

# Set working directory
WORKDIR /workspace

# Copy project files
COPY pyproject.toml requirements.txt ./
COPY revdeq/ ./revdeq/
COPY train.py inference.py ./
COPY configs/ ./configs/
COPY tests/ ./tests/
COPY run_tests.sh ./

# Install dependencies (try uv first, fallback to pip)
RUN if command -v uv >/dev/null 2>&1; then \
        uv pip install --system -r requirements.txt; \
    else \
        echo "Using pip instead of uv"; \
        pip install -r requirements.txt; \
    fi

# Note: We don't need to install the package with pip install -e .
# Because docker-compose.yml mounts the project directory as a volume,
# we can just set PYTHONPATH to /workspace in the environment variables
# This allows 'from revdeq import ...' to work without installation

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=""
# Add workspace to Python path so we can import revdeq without pip install
ENV PYTHONPATH="/workspace:${PYTHONPATH}"

# Default command
CMD ["/bin/bash"]

