# Python 3.11 — best compatibility with all packages (sentence-transformers, grpcio, etc.)
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (Docker layer caching)
COPY agents/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformers model during build
# (avoids downloading at runtime which delays first task)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy proto files and agent code
COPY proto/ /proto/
COPY agents/ .

# Regenerate gRPC stubs from proto after source copy so they match the
# installed grpcio-tools version (avoids gencode/runtime version mismatch).
RUN mkdir -p /app/base/generated && \
    python -m grpc_tools.protoc \
        -I /proto \
        --python_out=/app/base/generated \
        --grpc_python_out=/app/base/generated \
        /proto/soil.proto /proto/gardener.proto && \
    touch /app/base/generated/__init__.py

# Data and tasks directories (mounted via Docker volumes)
RUN mkdir -p /data /tasks

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["python", "run_agent.py"]
