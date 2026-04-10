/// Lightweight HTTP metrics endpoint for the dashboard.
/// GET /metrics → JSON snapshot of all agents + soil stats.
use std::sync::Arc;

use axum::{extract::State, response::Json, routing::get, Router};
use serde::Serialize;
use tracing::info;

use crate::gardener::GardenerCore;
use crate::soil::Soil;

#[derive(Serialize)]
pub struct MetricsSnapshot {
    pub timestamp: String,
    pub soil: SoilMetrics,
    pub agents: Vec<AgentMetrics>,
}

#[derive(Serialize)]
pub struct SoilMetrics {
    pub total_trails: u64,
    pub active_agents: usize,
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

async fn metrics_handler(
    State(state): State<Arc<AppState>>,
) -> Json<MetricsSnapshot> {
    let soil_stats = state.soil.get_stats().await;
    let agents = state.gardener.agent_snapshot().await;

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

    Json(MetricsSnapshot {
        timestamp: chrono::Utc::now().to_rfc3339(),
        soil: SoilMetrics {
            total_trails: soil_stats.total_trails,
            active_agents: soil_stats.agent_count,
        },
        agents: agent_metrics,
    })
}

async fn health_handler() -> &'static str {
    "ok"
}

pub async fn serve(gardener: Arc<GardenerCore>, soil: Arc<Soil>, port: u16) {
    let state = Arc::new(AppState { gardener, soil });

    let app = Router::new()
        .route("/metrics", get(metrics_handler))
        .route("/health", get(health_handler))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    info!("Metrics HTTP server on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("Failed to bind metrics port");

    axum::serve(listener, app)
        .await
        .expect("Metrics server failed");
}
