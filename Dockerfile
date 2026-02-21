FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY agentvault/ agentvault/

# Install the package
RUN pip install --no-cache-dir -e ".[dev]"

# Create data directory
RUN mkdir -p /data

# Expose API port
EXPOSE 8000

# Default command: run the API server
CMD ["python", "-m", "uvicorn", "agentvault.api.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
