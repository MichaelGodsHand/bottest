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
COPY pyproject.toml ./

# Install dependencies using uv (with cache mount for faster builds)
# Note: --locked flag removed to work without uv.lock file
# If uv.lock exists, you can add it back and use --locked for reproducible builds
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-dev

# Copy application code
COPY bot.py ./

# Note: service-account.json is NOT copied to the image for security
# In Cloud Run, use one of these options:
# 1. Set GOOGLE_APPLICATION_CREDENTIALS as environment variable with JSON content
# 2. Use Google Cloud Secret Manager and mount as file
# 3. Use Cloud Run's built-in service account (if configured)

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Expose port (for FastAPI server mode)
EXPOSE 8080

# Run the bot using uv
CMD ["uv", "run", "bot.py"]
