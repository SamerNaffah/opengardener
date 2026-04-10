use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::info;

use crate::gardener::GardenerCore;

pub mod gardener_proto {
    tonic::include_proto!("gardener");
}

use gardener_proto::{
    gardener_server::Gardener as GardenerService,
    Empty, HealthAck, HealthReport, ResourceAllocation, ResourceRequest, TerminationRequest,
};

pub struct GardenerGrpcService {
    pub gardener: Arc<GardenerCore>,
}

#[tonic::async_trait]
impl GardenerService for GardenerGrpcService {
    async fn request_resources(
        &self,
        request: Request<ResourceRequest>,
    ) -> Result<Response<ResourceAllocation>, Status> {
        let req = request.into_inner();
        info!(
            "ResourceRequest from agent={}: cpu={}ms, mem={}MB, reason={}",
            req.agent_id, req.cpu_ms, req.memory_mb, req.reason
        );

        let allocation = self.gardener.allocate_resources(&req.agent_id, req.cpu_ms, req.memory_mb).await;

        Ok(Response::new(ResourceAllocation {
            granted: allocation.granted,
            cpu_ms: allocation.cpu_ms,
            memory_mb: allocation.memory_mb,
            message: allocation.message,
        }))
    }

    async fn report_health(
        &self,
        request: Request<HealthReport>,
    ) -> Result<Response<HealthAck>, Status> {
        let report = request.into_inner();

        let (feedback, should_terminate, terminate_reason) =
            self.gardener.process_health_report(&report).await;

        Ok(Response::new(HealthAck {
            message: feedback,
            should_terminate,
            terminate_reason,
        }))
    }

    async fn request_termination(
        &self,
        request: Request<TerminationRequest>,
    ) -> Result<Response<Empty>, Status> {
        let req = request.into_inner();
        info!(
            "[GARDENER] Agent {} requesting termination: {}",
            req.agent_id, req.reason
        );

        self.gardener.handle_termination(&req.agent_id, &req.reason).await;
        Ok(Response::new(Empty {}))
    }
}
