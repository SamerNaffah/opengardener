/// Lightweight HTTP metrics endpoint for the dashboard.
/// GET /metrics   → JSON snapshot of agents + soil stats.
/// GET /metrics/prom → Prometheus text exposition (for V4 prep).
/// GET /health    → "ok" liveness probe.
use std::collections::HashMap;
use std::sync::Arc;

use axum::{extract::State, response::Json, response::Response, routing::get, Router};
use serde::Serialize;
use tokio_util::sync::CancellationToken;
use tracing::info;

use crate::gardener::GardenerCore;
use crate::soil::Soil;

const KNOWN_DOMAINS: &[&str] = &["data_cleaning", "code_generation", "api_testing"];

#[derive(Serialize)]
pub struct MetricsSnapshot {
    pub timestamp: String,
    pub soil: SoilMetrics,
    pub gardener: GardenerMetrics,
    pub agents: Vec<AgentMetrics>,
}

#[derive(Serialize)]
pub struct SoilMetrics {
    pub total_trails: u64,
    pub active_agents: usize,
    pub trails_per_domain: HashMap<String, u64>,
}

#[derive(Serialize)]
pub struct GardenerMetrics {
    pub pending_prune: usize,
}

#[derive(Serialize)]
pub struct AgentMetrics {
    pub id: String,
    pub agent_type: String,
    pub domain: String,
    pub tasks_completed: i32,
    pub failure_rate: f32,
    pub status: String,
    pub age_secs: i64,
}

struct AppState {
    gardener: Arc<GardenerCore>,
    soil: Arc<Soil>,
}

async fn metrics_handler(State(state): State<Arc<AppState>>) -> Json<MetricsSnapshot> {
    let snapshot = build_snapshot(&state).await;
    Json(snapshot)
}

async fn prom_handler(State(state): State<Arc<AppState>>) -> Response<axum::body::Body> {
    let snap = build_snapshot(&state).await;
    let mut out = String::new();
    out.push_str("# HELP opengardener_soil_trails Total pheromone trails in soil.\n");
    out.push_str("# TYPE opengardener_soil_trails gauge\n");
    out.push_str(&format!("opengardener_soil_trails {}\n", snap.soil.total_trails));
    out.push_str("# HELP opengardener_active_agents Active agents.\n");
    out.push_str("# TYPE opengardener_active_agents gauge\n");
    out.push_str(&format!(
        "opengardener_active_agents {}\n",
        snap.soil.active_agents
    ));
    out.push_str("# HELP opengardener_pending_prune Agents queued for prune.\n");
    out.push_str("# TYPE opengardener_pending_prune gauge\n");
    out.push_str(&format!(
        "opengardener_pending_prune {}\n",
        snap.gardener.pending_prune
    ));
    out.push_str("# HELP opengardener_domain_trails Pheromone trails per domain.\n");
    out.push_str("# TYPE opengardener_domain_trails gauge\n");
    for (d, n) in &snap.soil.trails_per_domain {
        out.push_str(&format!(
            "opengardener_domain_trails{{domain=\"{}\"}} {}\n",
            d, n
        ));
    }
    out.push_str("# HELP opengardener_agent_failure_rate Per-agent failure rate.\n");
    out.push_str("# TYPE opengardener_agent_failure_rate gauge\n");
    for a in &snap.agents {
        out.push_str(&format!(
            "opengardener_agent_failure_rate{{agent_id=\"{}\",type=\"{}\"}} {}\n",
            a.id, a.agent_type, a.failure_rate
        ));
        out.push_str(&format!(
            "opengardener_agent_tasks_completed{{agent_id=\"{}\",type=\"{}\"}} {}\n",
            a.id, a.agent_type, a.tasks_completed
        ));
    }
    Response::builder()
        .header("content-type", "text/plain; version=0.0.4")
        .body(axum::body::Body::from(out))
        .unwrap()
}

async fn build_snapshot(state: &AppState) -> MetricsSnapshot {
    let soil_stats = state.soil.get_stats().await;
    let trails_per_domain = state.soil.count_per_domain(KNOWN_DOMAINS).await;
    let agents = state.gardener.agent_snapshot().await;
    let pending_prune = state.gardener.pending_prune_count().await;

    let agent_metrics: Vec<AgentMetrics> = agents
        .iter()
        .map(|a| {
            let domain = a
                .last_health_report
                .as_ref()
                .map(|r| r.current_domain.clone())
                .unwrap_or_else(|| a.agent_type.clone());

            let status = a
                .last_health_report
                .as_ref()
                .map(|r| r.status.to_string())
                .unwrap_or_else(|| "unknown".to_string());

            AgentMetrics {
                id: a.id.0.clone(),
                agent_type: a.agent_type.clone(),
                domain,
                tasks_completed: a.tasks_completed(),
                failure_rate: a.failure_rate(),
                status,
                age_secs: (chrono::Utc::now() - a.created_at).num_seconds(),
            }
        })
        .collect();

    MetricsSnapshot {
        timestamp: chrono::Utc::now().to_rfc3339(),
        soil: SoilMetrics {
            total_trails: soil_stats.total_trails,
            active_agents: soil_stats.agent_count,
            trails_per_domain,
        },
        gardener: GardenerMetrics { pending_prune },
        agents: agent_metrics,
    }
}

async fn health_handler() -> &'static str {
    "ok"
}

pub async fn serve(
    gardener: Arc<GardenerCore>,
    soil: Arc<Soil>,
    port: u16,
    cancel: CancellationToken,
) {
    let state = Arc::new(AppState { gardener, soil });

    let app = Router::new()
        .route("/metrics", get(metrics_handler))
        .route("/metrics/prom", get(prom_handler))
        .route("/health", get(health_handler))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    info!("Metrics HTTP server on {}", addr);

    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("Failed to bind metrics port {}: {}", port, e);
            return;
        }
    };

    let serve = axum::serve(listener, app).with_graceful_shutdown(async move {
        cancel.cancelled().await;
    });

    if let Err(e) = serve.await {
        tracing::warn!("Metrics server exited with error: {}", e);
    }
}
