.PHONY: up down logs seed observe dashboard demo clean build proto test test-rust test-python lint

# Copy .env if it doesn't exist
.env:
	cp .env.example .env
	@echo "Created .env from .env.example — edit it if needed."

# Build all Docker images
build:
	docker compose build

# Start the full stack
up: .env
	docker compose up -d
	@echo "Stack is up. Run 'make logs' to watch output."

# Stop everything
down:
	docker compose down

# Tail logs (all services or specific: make logs SERVICE=gardener)
logs:
	docker compose logs -f $(SERVICE)

# Run the task seeder (bootstraps soil with initial tasks)
seed: .env
	docker compose --profile tools run --rm seed-runner

# Query the soil and display trail stats
observe: .env
	docker compose --profile tools run --rm observer

# Live terminal dashboard (auto-refreshes every 3s)
dashboard: .env
	docker compose --profile tools run --rm dashboard

# Full demo: bring up, seed, wait 30s, observe
demo: up
	@echo "Waiting 30s for agents to process initial tasks..."
	sleep 30
	$(MAKE) seed
	@echo "Waiting 60s for pheromone trails to accumulate..."
	sleep 60
	$(MAKE) observe

# Remove all containers, networks, volumes
clean:
	docker compose down -v --remove-orphans

# Regenerate Python gRPC stubs from proto files
proto:
	python3 -m grpc_tools.protoc \
		-I proto \
		--python_out=agents/base/generated \
		--grpc_python_out=agents/base/generated \
		proto/soil.proto proto/gardener.proto
	@echo "Proto stubs generated in agents/base/generated/"

# Watch gardener logs specifically
logs-gardener:
	docker compose logs -f gardener

# Watch all agent logs
logs-agents:
	docker compose logs -f agent-data agent-code agent-api

# ─── Tests / Lint ─────────────────────────────────────────────────────────
# Rust unit tests (cargo test under gardener/)
test-rust:
	cd gardener && cargo test --all-features

# Python unit tests (regenerates stubs first so tests don't depend on prior `make proto`)
test-python:
	mkdir -p agents/base/generated
	python3 -m grpc_tools.protoc -I proto \
		--python_out=agents/base/generated \
		--grpc_python_out=agents/base/generated \
		proto/soil.proto proto/gardener.proto
	touch agents/base/generated/__init__.py
	PYTHONPATH=agents pytest -q tests/

# Run both
test: test-rust test-python

# Lint everything
lint:
	cd gardener && cargo fmt --all -- --check && cargo clippy --all-targets -- -D warnings
	@echo "Tip: run 'pip install ruff && ruff check agents/' for Python linting."
