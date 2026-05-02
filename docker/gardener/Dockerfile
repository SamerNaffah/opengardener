# syntax=docker/dockerfile:1.6
#
# Multi-stage build for the OpenGardener Rust orchestrator.
#  - Stage 1 caches dependencies separately from source so a code change doesn't
#    invalidate the heavy crate compilation.
#  - Stage 2 is a distroless-style debian-slim image carrying only the binary,
#    runtime libs, and the netbase package required by reqwest's DNS lookups.
#
# Build:    docker build -f docker/gardener/Dockerfile .
# Run:      ports 50051 (Soil gRPC), 50052 (Gardener gRPC), 8080 (metrics HTTP).
#
FROM rust:1.82-slim AS builder

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        protobuf-compiler \
        pkg-config \
 && rm -rf /var/lib/apt/lists/*

# ── Dependency-only build to leverage Docker layer caching ─────────────────
COPY proto/ /proto/
COPY gardener/Cargo.toml gardener/build.rs ./
RUN mkdir -p src && \
    echo 'fn main() { println!("dependency-prebuild stub"); }' > src/main.rs && \
    cargo build --release || true && \
    rm -rf src

# ── Real build ─────────────────────────────────────────────────────────────
COPY gardener/ .
RUN cargo build --release

# ── Runtime ────────────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

# `iproute2` provides `ss` for the healthcheck. `ca-certificates` is required
# for HTTPS to ChromaDB / external resources. We deliberately drop libssl3
# because the rustls feature flag means we no longer link OpenSSL.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        iproute2 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 10001 --no-create-home gardener

WORKDIR /app
COPY --from=builder /app/target/release/gardener /app/gardener

USER gardener

EXPOSE 50051 50052 8080

# Verify the Soil gRPC port is listening.
HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
    CMD ss -tnlp | grep -q ':50051 ' || exit 1

CMD ["/app/gardener"]
