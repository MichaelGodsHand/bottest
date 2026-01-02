# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Enable bytecode compilation for better performance
ENV UV_COMPILE_BYTECODE=1

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies using uv (with cache mount for faster builds)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# Copy application code
COPY bot.py ./

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Expose port (for FastAPI server mode)
EXPOSE 8080

# Run the bot using uv
CMD ["uv", "run", "bot.py"]
