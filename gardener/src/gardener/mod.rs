pub mod registry;
pub mod resource;

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{interval, Duration};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::soil::Soil;
use crate::types::{AgentHealthReport, AgentStatus};
use registry::{AgentHandle, AgentRegistry};
use resource::{ResourceAllocation, ResourceManager};

// Pruning thresholds — env-overridable in `Self::from_env`.
const PRUNE_FAILURE_RATE: f32 = 0.7;
const PRUNE_MIN_TASKS: i32 = 20;
const NICHE_MIN_TRAILS: u64 = 10;
const NICHE_MAX_AGENTS: usize = 2;
const OBSERVER_INTERVAL_SECS: u64 = 60;
const KNOWN_DOMAINS: &[&str] = &["data_cleaning", "code_generation", "api_testing"];

/// Reason an agent was queued for pruning. Encoded as a string on the wire,
/// preserved so the eventual `terminate_reason` matches the actual queue cause.
#[derive(Debug, Clone, Copy)]
enum PruneReason {
    HighFailureRate,
    Stagnant,
}

impl PruneReason {
    fn as_str(self) -> &'static str {
        match self {
            PruneReason::HighFailureRate => "high_failure_rate",
            PruneReason::Stagnant => "stagnant",
        }
    }
}

pub struct GardenerCore {
    pub soil: Arc<Soil>,
    pub registry: Arc<AgentRegistry>,
    pub resources: ResourceManager,
    /// Agents scheduled for pruning on their next health report. The value
    /// preserves the *reason* the agent was queued so the wire signal is honest.
    pending_prune: Arc<RwLock<HashMap<String, PruneReason>>>,
    /// Allows callers (main loop) to request graceful shutdown of the observer.
    cancel: CancellationToken,
}

impl GardenerCore {
    pub fn new(soil: Arc<Soil>) -> Self {
        Self {
            soil,
            registry: Arc::new(AgentRegistry::new()),
            resources: ResourceManager::new(),
            pending_prune: Arc::new(RwLock::new(HashMap::new())),
            cancel: CancellationToken::new(),
        }
    }

    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancel.clone()
    }

    /// Observation loop — runs every OBSERVER_INTERVAL_SECS until cancelled.
    /// Identifies prune candidates and queues them for termination on next health report.
    pub async fn observe_ecosystem(self: Arc<Self>) {
        let mut ticker = interval(Duration::from_secs(OBSERVER_INTERVAL_SECS));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                _ = self.cancel.cancelled() => {
                    info!("[GARDENER] Observer shutting down (cancellation requested).");
                    break;
                }
                _ = ticker.tick() => {}
            }

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

                // Prune: persistent high-failure agent with enough data.
                if failure_rate > PRUNE_FAILURE_RATE && tasks > PRUNE_MIN_TASKS {
                    warn!(
                        "[GARDENER] Queuing prune: agent={} failure_rate={:.2} tasks={}",
                        agent.id, failure_rate, tasks
                    );
                    self.queue_prune(&agent.id.0, PruneReason::HighFailureRate)
                        .await;
                }

                // Prune: stagnant agent (long-lived, barely any work).
                if agent.is_stagnant() {
                    warn!(
                        "[GARDENER] Queuing prune (stagnant): agent={} age={}h tasks={}",
                        agent.id,
                        (chrono::Utc::now() - agent.created_at).num_hours(),
                        tasks
                    );
                    self.queue_prune(&agent.id.0, PruneReason::Stagnant).await;
                }
            }

            // Real per-domain niche detection: query soil for actual trail counts
            // per domain (replaces the previous "total/3" heuristic).
            let counts = self.soil.count_per_domain(KNOWN_DOMAINS).await;
            for domain in KNOWN_DOMAINS {
                let trail_count = counts.get(*domain).copied().unwrap_or(0);
                let agent_count = self.registry.count_by_domain(domain).await;
                if trail_count > NICHE_MIN_TRAILS && agent_count < NICHE_MAX_AGENTS {
                    info!(
                        "[GARDENER] NICHE DETECTED: domain={} trails={} agents={}",
                        domain, trail_count, agent_count
                    );
                }
            }
        }
    }

    /// Queue an agent for pruning.
    async fn queue_prune(&self, agent_id: &str, reason: PruneReason) {
        let mut pending = self.pending_prune.write().await;
        let inserted = pending
            .insert(agent_id.to_string(), reason)
            .is_none();
        if inserted {
            warn!(
                "[GARDENER] PRUNE QUEUED: agent={} reason={}",
                agent_id,
                reason.as_str()
            );
        }
    }

    /// Process a health report. Returns (feedback, should_terminate, reason).
    pub async fn process_health_report(
        &self,
        report: &crate::grpc::gardener_proto::HealthReport,
    ) -> (String, bool, String) {
        let status = match report.status.as_str() {
            "idle" => AgentStatus::Idle,
            "struggling" => AgentStatus::Struggling,
            _ => AgentStatus::Active,
        };
        let health = AgentHealthReport {
            agent_id: report.agent_id.clone(),
            failure_rate: report.failure_rate,
            tasks_completed: report.tasks_completed,
            cpu_ms: report.cpu_ms,
            memory_mb: report.memory_mb,
            current_domain: report.current_domain.clone(),
            status,
        };

        self.registry.update_health(&report.agent_id, health).await;

        // Race-free check-and-clear of the prune flag: take a single write lock.
        let queued = {
            let mut pending = self.pending_prune.write().await;
            pending.remove(&report.agent_id)
        };

        if let Some(reason) = queued {
            let reason_s = reason.as_str();
            warn!(
                "[GARDENER] PRUNE SIGNAL sent: agent={} reason={}",
                report.agent_id, reason_s
            );
            // Archive the agent's reputation in soil metadata before it exits.
            self.archive_agent_trails(&report.agent_id).await;
            return (
                format!("Pruning: {}", reason_s),
                true,
                reason_s.to_string(),
            );
        }

        // Normal feedback.
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

    /// Number of agents pending prune (for /metrics observability).
    pub async fn pending_prune_count(&self) -> usize {
        self.pending_prune.read().await.len()
    }
}
