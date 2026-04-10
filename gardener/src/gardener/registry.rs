use chrono::{DateTime, Utc};
use std::collections::HashMap;
use tokio::sync::RwLock;

use crate::types::{AgentId, AgentHealthReport, AgentStatus};

#[derive(Debug, Clone)]
pub struct AgentHandle {
    pub id: AgentId,
    pub agent_type: String,
    pub created_at: DateTime<Utc>,
    pub last_health_report: Option<AgentHealthReport>,
}

impl AgentHandle {
    pub fn new(id: AgentId, agent_type: &str) -> Self {
        Self {
            id,
            agent_type: agent_type.to_string(),
            created_at: Utc::now(),
            last_health_report: None,
        }
    }

    pub fn update_health(&mut self, report: AgentHealthReport) {
        self.last_health_report = Some(report);
    }

    pub fn failure_rate(&self) -> f32 {
        self.last_health_report
            .as_ref()
            .map(|r| r.failure_rate)
            .unwrap_or(0.0)
    }

    pub fn tasks_completed(&self) -> i32 {
        self.last_health_report
            .as_ref()
            .map(|r| r.tasks_completed)
            .unwrap_or(0)
    }

    pub fn is_stagnant(&self) -> bool {
        let age = Utc::now() - self.created_at;
        age.num_hours() > 24 && self.tasks_completed() < 5
    }

    pub fn is_struggling(&self) -> bool {
        self.failure_rate() > 0.7 && self.tasks_completed() > 20
    }
}

pub struct AgentRegistry {
    agents: RwLock<HashMap<String, AgentHandle>>,
}

impl AgentRegistry {
    pub fn new() -> Self {
        Self {
            agents: RwLock::new(HashMap::new()),
        }
    }

    pub async fn register(&self, handle: AgentHandle) {
        let mut agents = self.agents.write().await;
        agents.insert(handle.id.0.clone(), handle);
    }

    pub async fn update_health(&self, agent_id: &str, report: AgentHealthReport) {
        let mut agents = self.agents.write().await;
        if let Some(handle) = agents.get_mut(agent_id) {
            handle.update_health(report);
        } else {
            // Auto-register unknown agents that report health (Docker agents self-register on first report)
            let id = AgentId(agent_id.to_string());
            let domain = report.current_domain.clone();
            let mut handle = AgentHandle::new(id, &domain);
            handle.update_health(report);
            agents.insert(agent_id.to_string(), handle);
        }
    }

    pub async fn remove(&self, agent_id: &str) -> Option<AgentHandle> {
        let mut agents = self.agents.write().await;
        agents.remove(agent_id)
    }

    pub async fn snapshot(&self) -> Vec<AgentHandle> {
        let agents = self.agents.read().await;
        agents.values().cloned().collect()
    }

    pub async fn count(&self) -> usize {
        self.agents.read().await.len()
    }

    pub async fn count_by_domain(&self, domain: &str) -> usize {
        let agents = self.agents.read().await;
        agents
            .values()
            .filter(|h| {
                h.last_health_report
                    .as_ref()
                    .map(|r| r.current_domain == domain)
                    .unwrap_or(false)
            })
            .count()
    }
}
