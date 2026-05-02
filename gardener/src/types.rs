use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentId(pub String);

impl AgentId {
    pub fn new() -> Self {
        AgentId(uuid::Uuid::new_v4().to_string())
    }
}

impl Default for AgentId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Stored alongside the trail metadata. The wire format is the lowercase string
/// (`"success"` | `"failure"`) — keep this enum for callers that want type safety
/// when constructing trails programmatically.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Outcome {
    Success,
    Failure,
}

impl Outcome {
    pub fn as_str(&self) -> &'static str {
        match self {
            Outcome::Success => "success",
            Outcome::Failure => "failure",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureType {
    PermissionDenied,
    Timeout,
    InvalidInput,
    ExternalServiceError,
    Unknown(String),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceMetrics {
    pub cpu_ms: f32,
    pub memory_mb: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentReputation {
    pub agent_id: String,
    pub success_rate: f32,
    pub tasks_completed: i32,
    pub specialization: Option<String>,
    pub last_updated: DateTime<Utc>,
}

impl Default for AgentReputation {
    fn default() -> Self {
        Self {
            agent_id: String::new(),
            success_rate: 0.0,
            tasks_completed: 0,
            specialization: None,
            last_updated: Utc::now(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PheromoneTrail {
    pub trail_id: String,
    pub task_embedding: Vec<f32>,
    pub outcome: String,
    pub approach: serde_json::Value,
    pub resources: ResourceMetrics,
    pub agent_id: String,
    pub timestamp: DateTime<Utc>,
    pub task_domain: String,
    pub task_summary: String,
    pub hits: i32,
}

impl PheromoneTrail {
    pub fn new(
        embedding: Vec<f32>,
        outcome: &str,
        approach: serde_json::Value,
        agent_id: &str,
        domain: &str,
        summary: &str,
    ) -> Self {
        Self {
            trail_id: uuid::Uuid::new_v4().to_string(),
            task_embedding: embedding,
            outcome: outcome.to_string(),
            approach,
            resources: ResourceMetrics::default(),
            agent_id: agent_id.to_string(),
            timestamp: Utc::now(),
            task_domain: domain.to_string(),
            task_summary: summary.to_string(),
            hits: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentHealthReport {
    pub agent_id: String,
    pub failure_rate: f32,
    pub tasks_completed: i32,
    pub cpu_ms: f32,
    pub memory_mb: f32,
    pub current_domain: String,
    pub status: AgentStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AgentStatus {
    Active,
    Idle,
    Struggling,
}

impl std::fmt::Display for AgentStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentStatus::Active => write!(f, "active"),
            AgentStatus::Idle => write!(f, "idle"),
            AgentStatus::Struggling => write!(f, "struggling"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outcome_round_trips_to_string() {
        assert_eq!(Outcome::Success.as_str(), "success");
        assert_eq!(Outcome::Failure.as_str(), "failure");
    }

    #[test]
    fn pheromone_trail_has_unique_id() {
        let t1 = PheromoneTrail::new(vec![0.0; 4], "success", serde_json::Value::Null, "a", "d", "s");
        let t2 = PheromoneTrail::new(vec![0.0; 4], "success", serde_json::Value::Null, "a", "d", "s");
        assert_ne!(t1.trail_id, t2.trail_id);
    }
}
