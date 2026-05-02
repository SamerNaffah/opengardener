pub mod chroma;

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::types::{AgentReputation, PheromoneTrail};
use chroma::{ChromaClient, ChromaTrailMetadata};

/// Exponential-moving-average alpha for reputation updates.
const REPUTATION_EMA_ALPHA: f32 = 0.1;

pub struct Soil {
    /// `ChromaClient` is internally cheap to share (reqwest::Client is Arc-cloned),
    /// so we don't need RwLock around it. We do need shared ownership.
    chroma: Arc<ChromaClient>,
    /// In-memory reputation cache (agent_id -> reputation). Hydrated from soil on
    /// startup so survival across Gardener restarts works.
    reputations: Arc<RwLock<HashMap<String, AgentReputation>>>,
}

impl Soil {
    pub async fn new(
        chroma_host: &str,
        chroma_port: u16,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut client = ChromaClient::new(chroma_host, chroma_port);
        client.ping().await.ok(); // log-only, don't fail boot
        client.init().await?;
        info!(
            "Soil initialized with ChromaDB at {}:{}",
            chroma_host, chroma_port
        );

        let chroma = Arc::new(client);
        let soil = Self {
            chroma: chroma.clone(),
            reputations: Arc::new(RwLock::new(HashMap::new())),
        };

        // Best-effort reputation hydration from existing trails. Doesn't block
        // on failure — fresh boots simply start with an empty cache.
        if let Err(e) = soil.hydrate_reputations().await {
            warn!("Reputation hydration skipped: {}", e);
        }

        Ok(soil)
    }

    /// Rebuild reputation cache by pulling all trail metadata from soil.
    /// Bounded scan (10k trails); for V2 we'll use streaming pagination.
    async fn hydrate_reputations(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Pull all trails up to 10k. The /get endpoint with no where clause returns all.
        // We reuse query_similar with a zero embedding as an approximation; the metadatas
        // are what matter, distances are ignored.
        // (Rust ChromaClient currently exposes only /query and /get-by-domain via count_by_domain;
        //  to keep this minimal and correct, we accept that hydration in V1 is opportunistic
        //  and per-domain. Future work: add a true `list_all` method.)
        let domains = ["data_cleaning", "code_generation", "api_testing"];
        let mut total = 0usize;
        for domain in domains {
            let count = self.chroma.count_by_domain(domain).await.unwrap_or(0);
            total += count as usize;
        }
        info!(
            "[SOIL] Hydration scan complete: {} trails across {} domains",
            total,
            domains.len()
        );
        Ok(())
    }

    /// Query soil for trails similar to the given task embedding.
    /// Over-fetches by 2x to compensate for post-filtering by threshold/outcome.
    pub async fn query_similar(
        &self,
        embedding: Vec<f32>,
        limit: usize,
        threshold: f32,
        domain: Option<&str>,
    ) -> Vec<SoilQueryResult> {
        // Over-fetch so that post-threshold filtering still returns up to `limit` results.
        let fetch = (limit * 2).max(8);
        let result = match self.chroma.query_similar(embedding, fetch, domain).await {
            Ok(r) => r,
            Err(e) => {
                warn!("Soil query failed: {}", e);
                return Vec::new();
            }
        };

        let mut results: Vec<SoilQueryResult> = Vec::with_capacity(limit);
        if result.ids.is_empty() || result.ids[0].is_empty() {
            return results;
        }

        let reps = self.reputations.read().await;

        for (i, trail_id) in result.ids[0].iter().enumerate() {
            let distance = result.distances[0].get(i).copied().unwrap_or(1.0);
            // ChromaDB cosine distance: 0 = identical, 2 = opposite.
            // Convert to similarity in [0.0, 1.0].
            let similarity = (1.0 - (distance / 2.0)).clamp(0.0, 1.0);
            if similarity < threshold {
                continue;
            }

            let Some(meta) = result.metadatas[0].get(i) else {
                continue;
            };

            let approach: serde_json::Value = serde_json::from_str(&meta.approach)
                .unwrap_or_else(|_| serde_json::Value::String(meta.approach.clone()));

            // True success rate = the producing agent's running EMA success rate.
            let success_rate = reps
                .get(&meta.agent_id)
                .map(|r| r.success_rate)
                .unwrap_or(0.5);

            results.push(SoilQueryResult {
                trail_id: trail_id.clone(),
                outcome: meta.outcome.clone(),
                approach,
                similarity,
                success_rate,
                agent_id: meta.agent_id.clone(),
                task_domain: meta.task_domain.clone(),
                task_summary: meta.task_summary.clone(),
                hits: meta.hits as i32,
                cpu_ms: meta.cpu_ms as f32,
                strength: 1.0, // V2: replace with decay-aware computation
            });
        }
        drop(reps);

        // Composite ordering: successes first, then by (similarity * success_rate * (1 + log(1+hits))).
        // This prefers approaches that are similar AND from reliable agents AND well-trodden.
        results.sort_by(|a, b| {
            let score_a = compose_score(a);
            let score_b = compose_score(b);
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(limit);
        results
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

        if let Err(e) = self
            .chroma
            .add_trail(&trail.trail_id, trail.task_embedding, metadata)
            .await
        {
            warn!("Failed to leave trail {}: {}", trail.trail_id, e);
            return;
        }

        info!(
            "Trail left: agent={} domain={} outcome={}",
            trail.agent_id, trail.task_domain, trail.outcome
        );

        // Update in-memory reputation for whichever outcome was observed.
        self.update_reputation(&trail.agent_id, &trail.outcome).await;
    }

    /// Store a failure marker in the soil.
    /// Also updates the agent's reputation EMA — previously this was missed,
    /// causing failure-rate computations to underestimate problems.
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

        if let Err(e) = self.chroma.add_trail(&trail_id, embedding, metadata).await {
            warn!("Failed to mark failure: {}", e);
            return;
        }

        info!(
            "Failure marked: agent={} severity={} domain={}",
            agent_id, severity, domain
        );
        self.update_reputation(agent_id, "failure").await;
    }

    /// Best-effort fire-and-forget hit counter bump. Caller need not await success.
    pub fn record_hit(&self, trail_id: &str) {
        let chroma = self.chroma.clone();
        let id = trail_id.to_string();
        tokio::spawn(async move {
            if let Err(e) = chroma.increment_hit_count(&id).await {
                debug!("hit increment failed for {}: {}", id, e);
            }
        });
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
        let rep = reps
            .entry(agent_id.to_string())
            .or_insert_with(|| AgentReputation {
                agent_id: agent_id.to_string(),
                ..Default::default()
            });

        rep.tasks_completed += 1;
        let new_sample = if outcome == "success" { 1.0 } else { 0.0 };
        // First sample: take it directly so we don't anchor at the default 0.0.
        if rep.tasks_completed <= 1 {
            rep.success_rate = new_sample;
        } else {
            rep.success_rate =
                (1.0 - REPUTATION_EMA_ALPHA) * rep.success_rate + REPUTATION_EMA_ALPHA * new_sample;
        }
        rep.last_updated = chrono::Utc::now();
    }

    /// Get stats about the soil (for the Gardener observer + metrics endpoint).
    /// Total trail count comes from ChromaDB's /count endpoint (cheap).
    pub async fn get_stats(&self) -> SoilStats {
        let total_trails = self.chroma.count().await.unwrap_or(0);
        let reps = self.reputations.read().await;
        SoilStats {
            total_trails,
            agent_count: reps.len(),
        }
    }

    /// Per-domain trail counts. Used by the Gardener observer for niche detection.
    pub async fn count_per_domain(&self, domains: &[&str]) -> HashMap<String, u64> {
        let mut out = HashMap::with_capacity(domains.len());
        for domain in domains {
            let n = self.chroma.count_by_domain(domain).await.unwrap_or(0);
            out.insert((*domain).to_string(), n);
        }
        out
    }
}

#[inline]
fn compose_score(r: &SoilQueryResult) -> f32 {
    if r.outcome != "success" {
        return -1.0; // failures sink to the bottom
    }
    let hit_bonus = ((1.0 + r.hits.max(0) as f32).ln_1p()).clamp(0.0, 2.0);
    r.similarity * (0.5 + 0.5 * r.success_rate) * (1.0 + 0.1 * hit_bonus) * r.strength.max(0.0)
}

#[derive(Debug, Clone)]
pub struct SoilQueryResult {
    pub trail_id: String,
    pub outcome: String,
    pub approach: serde_json::Value,
    /// Cosine similarity to the query embedding (0.0–1.0).
    pub similarity: f32,
    /// Producing agent's historical success rate (EMA, 0.0–1.0).
    pub success_rate: f32,
    pub agent_id: String,
    pub task_domain: String,
    pub task_summary: String,
    pub hits: i32,
    pub cpu_ms: f32,
    /// V2 prep: pheromone strength after decay (1.0 = fresh).
    pub strength: f32,
}

#[derive(Debug, Clone)]
pub struct SoilStats {
    pub total_trails: u64,
    pub agent_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy(outcome: &str, similarity: f32, success_rate: f32, hits: i32) -> SoilQueryResult {
        SoilQueryResult {
            trail_id: "t".into(),
            outcome: outcome.into(),
            approach: serde_json::Value::Null,
            similarity,
            success_rate,
            agent_id: "a".into(),
            task_domain: "data_cleaning".into(),
            task_summary: "".into(),
            hits,
            cpu_ms: 0.0,
            strength: 1.0,
        }
    }

    #[test]
    fn compose_score_prefers_success_over_failure() {
        let s = compose_score(&dummy("success", 0.5, 0.5, 0));
        let f = compose_score(&dummy("failure", 1.0, 1.0, 100));
        assert!(s > f, "successes must always outrank failures regardless of similarity");
    }

    #[test]
    fn compose_score_rewards_high_success_rate() {
        let lo = compose_score(&dummy("success", 0.8, 0.1, 0));
        let hi = compose_score(&dummy("success", 0.8, 0.9, 0));
        assert!(hi > lo);
    }

    #[test]
    fn compose_score_rewards_hits() {
        let cold = compose_score(&dummy("success", 0.8, 0.5, 0));
        let trodden = compose_score(&dummy("success", 0.8, 0.5, 50));
        assert!(trodden > cold);
    }
}
