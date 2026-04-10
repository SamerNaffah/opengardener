use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::debug;

use crate::soil::Soil;
use crate::types::PheromoneTrail;

// Include the generated protobuf code
pub mod soil_proto {
    tonic::include_proto!("soil");
}

use soil_proto::{
    soil_server::Soil as SoilService,
    AgentRequest, Empty, FailureMarker, QueryRequest, QueryResponse, Reputation, Trail, TrailResult,
};

pub struct SoilGrpcService {
    pub soil: Arc<Soil>,
}

#[tonic::async_trait]
impl SoilService for SoilGrpcService {
    async fn query_similar(
        &self,
        request: Request<QueryRequest>,
    ) -> Result<Response<QueryResponse>, Status> {
        let req = request.into_inner();

        if req.embedding.is_empty() {
            return Err(Status::invalid_argument("embedding must not be empty"));
        }

        let threshold = if req.threshold <= 0.0 { 0.7 } else { req.threshold };
        let limit = if req.limit <= 0 { 5 } else { req.limit as usize };
        let domain = if req.domain.is_empty() { None } else { Some(req.domain.as_str()) };

        debug!("QuerySimilar: limit={}, threshold={:.2}, domain={:?}", limit, threshold, domain);

        let results = self.soil.query_similar(req.embedding, limit, threshold, domain).await;

        let trail_results: Vec<TrailResult> = results
            .into_iter()
            .map(|r| TrailResult {
                trail_id: r.trail_id,
                outcome: r.outcome,
                approach: r.approach.to_string(),
                success_rate: r.similarity,
                resources: std::collections::HashMap::from([
                    ("cpu_ms".to_string(), r.cpu_ms),
                ]),
                agent_id: r.agent_id,
                task_domain: r.task_domain,
                hits: r.hits,
            })
            .collect();

        Ok(Response::new(QueryResponse { results: trail_results }))
    }

    async fn leave_trail(
        &self,
        request: Request<Trail>,
    ) -> Result<Response<Empty>, Status> {
        let trail_data = request.into_inner();

        if trail_data.embedding.is_empty() {
            return Err(Status::invalid_argument("embedding must not be empty"));
        }

        let approach: serde_json::Value = serde_json::from_str(&trail_data.approach)
            .unwrap_or(serde_json::Value::String(trail_data.approach.clone()));

        let trail = PheromoneTrail::new(
            trail_data.embedding,
            &trail_data.outcome,
            approach,
            &trail_data.agent_id,
            &trail_data.task_domain,
            &trail_data.task_summary,
        );

        self.soil.leave_trail(trail).await;
        Ok(Response::new(Empty {}))
    }

    async fn mark_failure(
        &self,
        request: Request<FailureMarker>,
    ) -> Result<Response<Empty>, Status> {
        let marker = request.into_inner();

        if marker.embedding.is_empty() {
            return Err(Status::invalid_argument("embedding must not be empty"));
        }

        self.soil
            .mark_failure(
                marker.embedding,
                &marker.failed_approach,
                &marker.error,
                &marker.agent_id,
                &marker.severity,
                &marker.task_domain,
                marker.avoid_in_future,
            )
            .await;

        Ok(Response::new(Empty {}))
    }

    async fn get_reputation(
        &self,
        request: Request<AgentRequest>,
    ) -> Result<Response<Reputation>, Status> {
        let req = request.into_inner();
        let rep = self.soil.get_reputation(&req.agent_id).await;

        Ok(Response::new(Reputation {
            agent_id: rep.agent_id,
            success_rate: rep.success_rate,
            tasks_completed: rep.tasks_completed,
            specialization: rep.specialization.unwrap_or_default(),
        }))
    }
}
