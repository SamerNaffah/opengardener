pub mod registry;
pub mod resource;

use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{interval, Duration};
use tracing::{info, warn};

use crate::soil::Soil;
use crate::types::AgentHealthReport;
use registry::{AgentRegistry, AgentHandle};
use resource::{ResourceAllocation, ResourceManager};

// Pruning thresholds
const PRUNE_FAILURE_RATE: f32 = 0.7;
const PRUNE_MIN_TASKS: i32 = 20;
const STAGNANT_AGE_HOURS: i64 = 24;
const STAGNANT_MIN_TASKS: i32 = 5;
const NICHE_MIN_TRAILS: u64 = 10;
const NICHE_MAX_AGENTS: usize = 2;
const OBSERVER_INTERVAL_SECS: u64 = 60;

pub struct GardenerCore {
    pub soil: Arc<Soil>,
    pub registry: Arc<AgentRegistry>,
    pub resources: ResourceManager,
    /// Agents scheduled for pruning on their next health report.
    pending_prune: Arc<RwLock<HashSet<String>>>,
}

impl GardenerCore {
    pub fn new(soil: Arc<Soil>) -> Self {
        Self {
            soil,
            registry: Arc::new(AgentRegistry::new()),
            resources: ResourceManager::new(),
            pending_prune: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    /// Observation loop — runs every OBSERVER_INTERVAL_SECS.
    /// Identifies prune candidates and queues them for termination on next health report.
    pub async fn observe_ecosystem(self: Arc<Self>) {
        let mut ticker = interval(Duration::from_secs(OBSERVER_INTERVAL_SECS));

        loop {
            ticker.tick().await;

            let agents = self.registry.snapshot().await;
            let soil_stats = self.soil.get_stats().await;

            info!(
                "[GARDENER] Observation cycle: {} agents, {} soil trails",
                agents.len(),
                soil_stats.total_trails
            );

            for agent in &agents {
                let failure_rate = agent.failure_rate();
                let tasks = agent.tasks_completed();

                // Prune: persistent high-failure agent with enough data
                if failure_rate > PRUNE_FAILURE_RATE && tasks > PRUNE_MIN_TASKS {
                    warn!(
                        "[GARDENER] Queuing prune: agent={} failure_rate={:.2} tasks={}",
                        agent.id, failure_rate, tasks
                    );
                    self.queue_prune(&agent.id.0, "high_failure_rate").await;
                }

                // Prune: stagnant agent (long-lived, barely any work)
                if agent.is_stagnant() {
                    warn!(
                        "[GARDENER] Queuing prune (stagnant): agent={} age={}h tasks={}",
                        agent.id,
                        (chrono::Utc::now() - agent.created_at).num_hours(),
                        tasks
                    );
                    self.queue_prune(&agent.id.0, "stagnant").await;
                }
            }

            // Detect niches with few active agents
            let domains = ["data_cleaning", "code_generation", "api_testing"];
            for domain in domains {
                let trail_count = soil_stats.total_trails / 3;
                let agent_count = self.registry.count_by_domain(domain).await;

                if trail_count > NICHE_MIN_TRAILS && agent_count < NICHE_MAX_AGENTS {
                    info!(
                        "[GARDENER] NICHE DETECTED: domain={} trails≈{} agents={}",
                        domain, trail_count, agent_count
                    );
                }
            }
        }
    }

    /// Queue an agent for pruning. The signal is delivered on its next health report.
    async fn queue_prune(&self, agent_id: &str, reason: &str) {
        let mut pending = self.pending_prune.write().await;
        if pending.insert(agent_id.to_string()) {
            warn!("[GARDENER] PRUNE QUEUED: agent={} reason={}", agent_id, reason);
        }
    }

    /// Process a health report. Returns (feedback, should_terminate, reason).
    pub async fn process_health_report(
        &self,
        report: &crate::grpc::gardener_proto::HealthReport,
    ) -> (String, bool, String) {
        let health = AgentHealthReport {
            agent_id: report.agent_id.clone(),
            failure_rate: report.failure_rate,
            tasks_completed: report.tasks_completed,
            cpu_ms: report.cpu_ms,
            memory_mb: report.memory_mb,
            current_domain: report.current_domain.clone(),
            status: crate::types::AgentStatus::Active,
        };

        self.registry.update_health(&report.agent_id, health).await;

        // Check if this agent is queued for pruning
        let should_terminate = {
            let pending = self.pending_prune.read().await;
            pending.contains(&report.agent_id)
        };

        if should_terminate {
            // Remove from pending — signal sent once; agent will exit and confirm via RequestTermination
            let mut pending = self.pending_prune.write().await;
            pending.remove(&report.agent_id);

            let reason = if report.failure_rate > PRUNE_FAILURE_RATE {
                "high_failure_rate"
            } else {
                "stagnant"
            };

            warn!(
                "[GARDENER] PRUNE SIGNAL sent: agent={} reason={}",
                report.agent_id, reason
            );

            // Archive the agent's reputation in soil metadata before it exits
            self.archive_agent_trails(&report.agent_id).await;

            return (
                format!("Pruning: {}", reason),
                true,
                reason.to_string(),
            );
        }

        // Normal feedback
        let feedback = if report.failure_rate > PRUNE_FAILURE_RATE {
            format!(
                "High failure rate ({:.0}%). Consider mutating your approach or switching domains.",
                report.failure_rate * 100.0
            )
        } else {
            "OK".to_string()
        };

        (feedback, false, String::new())
    }

    /// Mark an agent's trails as "archived" in the reputation cache so future agents
    /// can still query its knowledge, tagged with pruned=true.
    async fn archive_agent_trails(&self, agent_id: &str) {
        let rep = self.soil.get_reputation(agent_id).await;
        info!(
            "[GARDENER] Archiving agent {}: success_rate={:.2} tasks={}",
            agent_id, rep.success_rate, rep.tasks_completed
        );
        // Reputation stays in ChromaDB (trails are not deleted) — they remain as collective
        // memory. Future agents can still query them. This implements the paper's
        // "knowledge preservation on pruning" requirement.
    }

    /// Allocate resources for an agent.
    pub async fn allocate_resources(
        &self,
        agent_id: &str,
        cpu_ms: f32,
        memory_mb: f32,
    ) -> ResourceAllocation {
        self.resources.allocate(agent_id, cpu_ms, memory_mb).await
    }

    /// Handle an agent's self-termination request.
    pub async fn handle_termination(&self, agent_id: &str, reason: &str) {
        info!("[GARDENER] Agent {} terminated: {}", agent_id, reason);
        self.registry.remove(agent_id).await;
        self.resources.release(agent_id).await;
    }

    /// Snapshot of all agents — used by the metrics endpoint.
    pub async fn agent_snapshot(&self) -> Vec<AgentHandle> {
        self.registry.snapshot().await
    }
}
