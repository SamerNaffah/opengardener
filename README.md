<p align="center">
  <img src="artwork/OpenGardener logo with title.png" alt="OpenGardener" width="480"/>
</p>

<p align="center">
  <strong>Biologically-inspired multi-agent orchestration - where agents grow, specialize, and are pruned like a garden.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/rust-orchestrator-orange?logo=rust" />
  <img src="https://img.shields.io/badge/python-agents-blue?logo=python" />
  <img src="https://img.shields.io/badge/llm-ollama%20%7C%20claude%20%7C%20openai-purple" />
  <img src="https://img.shields.io/badge/vector%20db-chromadb-green" />
  <img src="https://img.shields.io/badge/transport-gRPC-lightgrey?logo=grpc" />
  <img src="https://img.shields.io/badge/deploy-docker--compose-2496ED?logo=docker" />
</p>

---

## What is OpenGardener?

OpenGardener is a research prototype for a **stigmergic multi-agent system** - agents coordinate by modifying a shared environment (the *Soil*), not by sending messages to each other. This mirrors how ant colonies and other biological systems achieve complex, adaptive behavior without central planning.

The system is inspired by the metaphor of a garden:

| Garden | OpenGardener |
|--------|-------------|
| Soil | Shared vector memory (ChromaDB) |
| Pheromone trails | Embedded task outcomes stored as trails |
| Plants specializing by environment | Agents self-specializing by domain |
| Gardener pruning weak plants | Orchestrator retiring underperforming agents |
| Seeds carrying genetic potential | New agents initialized with genome defaults |

> Full theoretical foundation in [`researchpaper.md`](researchpaper.md).

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Docker Network                      │
│                                                         │
│  ┌──────────────────────────────────────┐               │
│  │          Gardener (Rust)             │               │
│  │  ┌─────────────┐  ┌───────────────┐ │  ┌──────────┐ │
│  │  │ Soil gRPC   │  │Gardener gRPC  │ │  │ ChromaDB │ │
│  │  │  :50051     │  │   :50052      │ │◄─┤  :8000   │ │
│  │  └─────────────┘  └───────────────┘ │  │  (Soil)  │ │
│  │  ┌──────────────────────────────┐   │  └──────────┘ │
│  │  │  Metrics HTTP  :8080         │   │               │
│  │  └──────────────────────────────┘   │               │
│  └──────────────────────────────────────┘               │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │DataCleaner│  │CodeGen   │  │ApiTester │  │ Ollama │  │
│  │(×3)      │  │(×2)      │  │(×2)      │  │:11434  │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────┘  │
└─────────────────────────────────────────────────────────┘
```

**Rust** runs the orchestration core (Gardener + Soil gRPC servers). It owns lifecycle management, pheromone trail storage, agent health tracking, and auto-pruning.

**Python** runs the agents. Each agent embeds tasks via `sentence-transformers`, queries the Soil for past approaches, executes with exploit-or-explore logic, and deposits pheromone trails after every task.

**ChromaDB** is the Soil - a persistent vector database where all agent experience lives.

**Ollama** provides a fully local, free LLM (default: `llama3.2`) for approach generation. Claude and OpenAI are also supported.

---

## How Agents Work

```
          ┌─────────────────────────────────────────┐
          │              EmergentAgent               │
          │                                         │
  Task ──►│  1. embed(task)                         │
          │  2. soil.query_similar() → past trails  │
          │                                         │
          │  ┌─────────────┐   ┌──────────────────┐ │
          │  │  EXPLOIT    │   │    EXPLORE        │ │
          │  │ adopt best  │   │  LLM-guided or   │ │
          │  │ known trail │   │  rule mutation   │ │
          │  └─────────────┘   └──────────────────┘ │
          │                                         │
          │  3. execute strategy                    │
          │  4. soil.leave_trail() or mark_failure()│
          │  5. report_health() → Gardener          │
          └─────────────────────────────────────────┘
```

Agents **never communicate directly**. Coordination is entirely via the Soil.

---

## Auto-Pruning

When an agent's `failure_rate > 0.7` across `>20 tasks`, or it becomes stagnant (24h old, fewer than 5 tasks completed), the Gardener queues it for pruning.

On the agent's next health report, `HealthAck.should_terminate = true` is returned. The agent:
1. Logs the prune signal
2. Sends `RequestTermination` to the Gardener (confirming graceful shutdown)
3. Exits - its **pheromone trails remain in the Soil** as collective memory for future agents

No agent knowledge is destroyed on pruning.

---

## Quickstart

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) + [Docker Compose](https://docs.docker.com/compose/)
- ~4 GB disk space (Ollama model download on first run)

### 1. Configure

```bash
cp .env.example .env
# Default config uses Ollama (free, local). No API key needed.
# To use Claude: set LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY=...
```

### 2. Build & Start

```bash
make build   # compile Rust binary + build Python image (~5 min first time)
make up      # start ChromaDB, Ollama, Gardener, and 7 agents
```

### 3. Seed the Soil

```bash
make seed    # inject initial tasks to bootstrap pheromone trails
```

### 4. Watch the Ecosystem

```bash
make dashboard      # live terminal dashboard (agents, trails, health)
make logs           # raw log stream from all services
make logs-gardener  # Gardener observer output (pruning decisions, niche detection)
make observe        # one-shot soil trail stats
```

### 5. Stop

```bash
make down    # stop stack (preserves soil data)
make clean   # stop + delete all volumes (full reset)
```

---

## LLM Configuration

| Provider | `LLM_PROVIDER` | Key needed | Notes |
|----------|---------------|-----------|-------|
| **Ollama** (default) | `ollama` | No | Local, free, ~2GB model pulled on first use |
| **Anthropic Claude** | `anthropic` | `ANTHROPIC_API_KEY` | Recommended: `claude-haiku-4-5-20251001` |
| **OpenAI** | `openai` | `OPENAI_API_KEY` | Recommended: `gpt-4o-mini` |

Set `LLM_MODEL` to override the model name.

---

## Specialist Agents

| Agent | Domain | Default Strategy | Fallback |
|-------|--------|-----------------|---------|
| `DataCleanerAgent` | `data_cleaning` | `pandas_dropna` | `regex_cleaning` |
| `CodeGeneratorAgent` | `code_generation` | `llm_direct` | `template_based` |
| `ApiTesterAgent` | `api_testing` | `basic_get` | `health_check` |

Agents self-select strategies via Soil queries - the genome defaults are only used on first run before any trails exist.

---

## Project Structure

```
opengardener/
├── gardener/               # Rust orchestrator
│   └── src/
│       ├── main.rs         # Entry point (gRPC + metrics HTTP servers)
│       ├── gardener/       # Observer loop, registry, resource manager
│       ├── soil/           # ChromaDB client, pheromone trail logic
│       ├── grpc/           # tonic gRPC service implementations
│       └── metrics.rs      # axum HTTP /metrics endpoint
│
├── agents/                 # Python agents
│   ├── base/
│   │   ├── agent.py        # EmergentAgent base class
│   │   ├── soil_client.py  # Soil gRPC client
│   │   └── llm_client.py   # Ollama / Anthropic / OpenAI wrapper
│   ├── specialists/
│   │   ├── data_cleaner.py
│   │   ├── code_generator.py
│   │   └── api_tester.py
│   └── run_agent.py        # Docker entry point
│
├── proto/
│   ├── soil.proto          # Soil gRPC interface
│   └── gardener.proto      # Gardener gRPC interface
│
├── scripts/
│   ├── dashboard.py        # Live terminal dashboard (rich)
│   ├── seed_tasks.py       # Soil bootstrap
│   └── observe_soil.py     # One-shot trail inspector
│
├── docker-compose.yml
├── Makefile
└── .env.example
```

---

## gRPC API

### Soil Service (`:50051`)

| RPC | Request | Response | Purpose |
|-----|---------|----------|---------|
| `QuerySimilar` | embedding + domain | list of `TrailResult` | Find past approaches for a task |
| `LeaveTrail` | outcome + approach + resources | `TrailAck` | Deposit pheromone after success |
| `MarkFailure` | embedding + approach + error | `TrailAck` | Mark a failed approach to avoid |
| `GetReputation` | `agent_id` | `ReputationResult` | Query agent's success history |

### Gardener Service (`:50052`)

| RPC | Request | Response | Purpose |
|-----|---------|----------|---------|
| `ReportHealth` | metrics snapshot | `HealthAck` | Periodic health check; receives prune signal if queued |
| `RequestResources` | cpu + memory ask | `ResourceAllocation` | Request more compute |
| `RequestTermination` | agent_id + reason | `Empty` | Graceful shutdown confirmation |

### Metrics HTTP (`:8080`)

| Endpoint | Response |
|----------|---------|
| `GET /metrics` | JSON: all agents + soil stats |
| `GET /health` | `"ok"` |

---

## Makefile Reference

```
make build          Build all Docker images
make up             Start the full stack
make down           Stop the stack
make clean          Stop + wipe all volumes
make seed           Bootstrap Soil with initial tasks
make dashboard      Live terminal dashboard
make observe        One-shot trail stats
make logs           Tail all logs
make logs-gardener  Tail Gardener logs only
make logs-agents    Tail agent logs only
make proto          Regenerate Python gRPC stubs locally
```

---

## Roadmap

| Version | Focus |
|---------|-------|
| **V1** (now) | Proof of concept - stigmergic pipeline, auto-pruning, live dashboard |
| **V2** | Auto-spawning of agents into detected niches, pheromone decay, persistent agent state |
| **V3** | WebAssembly sandboxing (Wasmtime), capability-based security, multi-node distribution |
| **V4** | Prometheus/Grafana observability, mTLS, compliance-grade audit logging |

---

## Research

The full theoretical foundation — including the biological analogy, formal model, and simulation results — is in [`researchpaper.md`](researchpaper.md).

The system architecture diagram is in [`Diagram1.png`](Diagram1.png).

---

## Author

**Samer Naffah**

---

## License

MIT
