# Stage 1: Build
FROM rust:1.82-slim AS builder

WORKDIR /app

# Install protobuf compiler (required by tonic-build)
RUN apt-get update && apt-get install -y protobuf-compiler pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*

# Copy proto files first (build dependency)
COPY proto/ /proto/

# Copy the Rust crate
COPY gardener/ .

# Build in release mode
RUN cargo build --release

# Stage 2: Minimal runtime image
FROM debian:bookworm-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y ca-certificates libssl3 && rm -rf /var/lib/apt/lists/*

# Copy only the compiled binary
COPY --from=builder /app/target/release/gardener /app/gardener

# Expose gRPC ports
EXPOSE 50051 50052

# Health check — verify the Soil gRPC port is listening
HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
    CMD sh -c 'ss -tnlp | grep -q 50051 || exit 1'

CMD ["/app/gardener"]
