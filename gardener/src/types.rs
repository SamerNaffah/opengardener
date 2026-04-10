use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentId(pub String);

impl AgentId {
    pub fn new() -> Self {
        AgentId(uuid::Uuid::new_v4().to_string())
    }
}

impl std::fmt::Display for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Outcome {
    Success(f32), // confidence score 0.0-1.0
    Failure(FailureType),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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
