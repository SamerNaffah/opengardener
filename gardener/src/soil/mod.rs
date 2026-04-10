pub mod chroma;

use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::types::{AgentReputation, PheromoneTrail};
use chroma::{ChromaClient, ChromaTrailMetadata};

pub struct Soil {
    chroma: Arc<RwLock<ChromaClient>>,
    // In-memory reputation cache (agent_id -> reputation)
    reputations: Arc<RwLock<std::collections::HashMap<String, AgentReputation>>>,
}

impl Soil {
    pub async fn new(chroma_host: &str, chroma_port: u16) -> Result<Self, Box<dyn std::error::Error>> {
        let mut client = ChromaClient::new(chroma_host, chroma_port);
        client.init().await?;
        info!("Soil initialized with ChromaDB at {}:{}", chroma_host, chroma_port);

        Ok(Self {
            chroma: Arc::new(RwLock::new(client)),
            reputations: Arc::new(RwLock::new(std::collections::HashMap::new())),
        })
    }

    /// Query soil for trails similar to the given task embedding.
    /// Returns trails above the similarity threshold, sorted by success rate.
    pub async fn query_similar(
        &self,
        embedding: Vec<f32>,
        limit: usize,
        threshold: f32,
        domain: Option<&str>,
    ) -> Vec<SoilQueryResult> {
        let chroma = self.chroma.read().await;
        match chroma.query_similar(embedding, limit, domain).await {
            Ok(result) => {
                let mut results = Vec::new();
                if result.ids.is_empty() || result.ids[0].is_empty() {
                    return results;
                }

                for (i, trail_id) in result.ids[0].iter().enumerate() {
                    let distance = result.distances[0].get(i).copied().unwrap_or(1.0);
                    // ChromaDB cosine distance: 0 = identical, 2 = opposite
                    // Convert to similarity: 1 - (distance / 2)
                    let similarity = 1.0 - (distance / 2.0);

                    if similarity < threshold {
                        continue;
                    }

                    if let Some(meta) = result.metadatas[0].get(i) {
                        let approach = serde_json::from_str(&meta.approach)
                            .unwrap_or(serde_json::Value::Null);

                        results.push(SoilQueryResult {
                            trail_id: trail_id.clone(),
                            outcome: meta.outcome.clone(),
                            approach,
                            similarity,
                            agent_id: meta.agent_id.clone(),
                            task_domain: meta.task_domain.clone(),
                            hits: meta.hits as i32,
                            cpu_ms: meta.cpu_ms as f32,
                        });
                    }
                }

                // Sort by success + similarity combined score
                results.sort_by(|a, b| {
                    let score_a = if a.outcome == "success" { a.similarity } else { 0.0 };
                    let score_b = if b.outcome == "success" { b.similarity } else { 0.0 };
                    score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
                });

                results
            }
            Err(e) => {
                warn!("Soil query failed: {}", e);
                Vec::new()
            }
        }
    }

    /// Store a pheromone trail in the soil after task completion.
    pub async fn leave_trail(&self, trail: PheromoneTrail) {
        let metadata = ChromaTrailMetadata {
            outcome: trail.outcome.clone(),
            approach: trail.approach.to_string(),
            agent_id: trail.agent_id.clone(),
            task_domain: trail.task_domain.clone(),
            task_summary: trail.task_summary.clone(),
            timestamp: trail.timestamp.to_rfc3339(),
            hits: 0,
            cpu_ms: trail.resources.cpu_ms as f64,
            memory_mb: trail.resources.memory_mb as f64,
            severity: None,
            avoid_in_future: None,
        };

        let chroma = self.chroma.read().await;
        if let Err(e) = chroma.add_trail(&trail.trail_id, trail.task_embedding, metadata).await {
            warn!("Failed to leave trail {}: {}", trail.trail_id, e);
        } else {
            info!(
                "Trail left: agent={} domain={} outcome={}",
                trail.agent_id, trail.task_domain, trail.outcome
            );

            // Update in-memory reputation
            self.update_reputation(&trail.agent_id, &trail.outcome).await;
        }
    }

    /// Store a failure marker in the soil.
    pub async fn mark_failure(
        &self,
        embedding: Vec<f32>,
        failed_approach: &str,
        error: &str,
        agent_id: &str,
        severity: &str,
        domain: &str,
        avoid: bool,
    ) {
        let trail_id = format!("failure-{}", uuid::Uuid::new_v4());
        let metadata = ChromaTrailMetadata {
            outcome: "failure".to_string(),
            approach: failed_approach.to_string(),
            agent_id: agent_id.to_string(),
            task_domain: domain.to_string(),
            task_summary: format!("FAILURE: {}", error),
            timestamp: chrono::Utc::now().to_rfc3339(),
            hits: 0,
            cpu_ms: 0.0,
            memory_mb: 0.0,
            severity: Some(severity.to_string()),
            avoid_in_future: Some(avoid),
        };

        let chroma = self.chroma.read().await;
        if let Err(e) = chroma.add_trail(&trail_id, embedding, metadata).await {
            warn!("Failed to mark failure: {}", e);
        } else {
            info!(
                "Failure marked: agent={} severity={} domain={}",
                agent_id, severity, domain
            );
        }
    }

    /// Get agent reputation from in-memory cache.
    pub async fn get_reputation(&self, agent_id: &str) -> AgentReputation {
        let reps = self.reputations.read().await;
        reps.get(agent_id).cloned().unwrap_or_else(|| AgentReputation {
            agent_id: agent_id.to_string(),
            ..Default::default()
        })
    }

    async fn update_reputation(&self, agent_id: &str, outcome: &str) {
        let mut reps = self.reputations.write().await;
        let rep = reps.entry(agent_id.to_string()).or_insert_with(|| AgentReputation {
            agent_id: agent_id.to_string(),
            ..Default::default()
        });

        rep.tasks_completed += 1;
        let success = outcome == "success";
        // Exponential moving average for success rate
        let alpha = 0.1;
        let new_sample = if success { 1.0 } else { 0.0 };
        rep.success_rate = (1.0 - alpha) * rep.success_rate + alpha * new_sample;
        rep.last_updated = chrono::Utc::now();
    }

    /// Get stats about the soil (for the Gardener observer).
    pub async fn get_stats(&self) -> SoilStats {
        let chroma = self.chroma.read().await;
        let stats = chroma.get_collection_stats().await.unwrap_or_default();
        let total_trails = stats["count"].as_u64().unwrap_or(0);

        let reps = self.reputations.read().await;
        SoilStats {
            total_trails,
            agent_count: reps.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SoilQueryResult {
    pub trail_id: String,
    pub outcome: String,
    pub approach: serde_json::Value,
    pub similarity: f32,
    pub agent_id: String,
    pub task_domain: String,
    pub hits: i32,
    pub cpu_ms: f32,
}

#[derive(Debug, Clone)]
pub struct SoilStats {
    pub total_trails: u64,
    pub agent_count: usize,
}
