use std::collections::HashMap;
use tokio::sync::RwLock;
use tracing::info;

// V1: Simple quota tracking per agent. No hard enforcement — logs warnings only.
const DEFAULT_MAX_CPU_MS: f32 = 10_000.0;
const DEFAULT_MAX_MEMORY_MB: f32 = 512.0;

#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    pub granted: bool,
    pub cpu_ms: f32,
    pub memory_mb: f32,
    pub message: String,
}

struct AgentQuota {
    cpu_ms_used: f32,
    memory_mb_used: f32,
}

pub struct ResourceManager {
    quotas: RwLock<HashMap<String, AgentQuota>>,
    max_cpu_ms: f32,
    max_memory_mb: f32,
}

impl ResourceManager {
    pub fn new() -> Self {
        Self {
            quotas: RwLock::new(HashMap::new()),
            max_cpu_ms: DEFAULT_MAX_CPU_MS,
            max_memory_mb: DEFAULT_MAX_MEMORY_MB,
        }
    }

    pub async fn allocate(&self, agent_id: &str, cpu_ms: f32, memory_mb: f32) -> ResourceAllocation {
        let mut quotas = self.quotas.write().await;
        let quota = quotas.entry(agent_id.to_string()).or_insert(AgentQuota {
            cpu_ms_used: 0.0,
            memory_mb_used: 0.0,
        });

        // V1: Always grant but log if over threshold
        quota.cpu_ms_used += cpu_ms;
        quota.memory_mb_used = memory_mb; // track current usage, not cumulative

        if quota.cpu_ms_used > self.max_cpu_ms {
            info!(
                "[GARDENER] Agent {} exceeding CPU quota ({:.0}ms used)",
                agent_id, quota.cpu_ms_used
            );
        }

        ResourceAllocation {
            granted: true,
            cpu_ms,
            memory_mb,
            message: "granted".to_string(),
        }
    }

    pub async fn release(&self, agent_id: &str) {
        let mut quotas = self.quotas.write().await;
        quotas.remove(agent_id);
    }
}
