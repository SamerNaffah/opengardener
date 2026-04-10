mod gardener;
mod grpc;
mod metrics;
mod soil;
mod types;

use std::env;
use std::sync::Arc;
use tracing::info;
use tracing_subscriber::{EnvFilter, fmt};

use grpc::gardener_server::{GardenerGrpcService, gardener_proto::gardener_server::GardenerServer};
use grpc::soil_server::{SoilGrpcService, soil_proto::soil_server::SoilServer};
use soil::Soil;
use gardener::GardenerCore;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    info!("OpenGardener V1 starting...");

    let chroma_host = env::var("CHROMA_HOST").unwrap_or_else(|_| "localhost".to_string());
    let chroma_port: u16 = env::var("CHROMA_PORT")
        .unwrap_or_else(|_| "8000".to_string())
        .parse()
        .unwrap_or(8000);

    let soil_port: u16 = env::var("SOIL_GRPC_PORT")
        .unwrap_or_else(|_| "50051".to_string())
        .parse()
        .unwrap_or(50051);

    let gardener_port: u16 = env::var("GARDENER_GRPC_PORT")
        .unwrap_or_else(|_| "50052".to_string())
        .parse()
        .unwrap_or(50052);

    let metrics_port: u16 = env::var("METRICS_PORT")
        .unwrap_or_else(|_| "8080".to_string())
        .parse()
        .unwrap_or(8080);

    info!("Connecting to ChromaDB at {}:{}", chroma_host, chroma_port);
    let soil = Arc::new(
        Soil::new(&chroma_host, chroma_port)
            .await
            .expect("Failed to initialize Soil — is ChromaDB running?"),
    );

    let gardener_core = Arc::new(GardenerCore::new(Arc::clone(&soil)));

    // Observation loop
    let gardener_observer = Arc::clone(&gardener_core);
    tokio::spawn(async move {
        gardener_observer.observe_ecosystem().await;
    });

    // Metrics HTTP server
    let gardener_metrics = Arc::clone(&gardener_core);
    let soil_metrics = Arc::clone(&soil);
    tokio::spawn(async move {
        metrics::serve(gardener_metrics, soil_metrics, metrics_port).await;
    });

    let soil_addr = format!("0.0.0.0:{}", soil_port).parse()?;
    let gardener_addr = format!("0.0.0.0:{}", gardener_port).parse()?;

    info!("Soil gRPC server listening on {}", soil_addr);
    info!("Gardener gRPC server listening on {}", gardener_addr);
    info!("Metrics HTTP server listening on 0.0.0.0:{}", metrics_port);

    let soil_service = SoilGrpcService { soil: Arc::clone(&soil) };
    let gardener_service = GardenerGrpcService { gardener: Arc::clone(&gardener_core) };

    tokio::try_join!(
        tonic::transport::Server::builder()
            .add_service(SoilServer::new(soil_service))
            .serve(soil_addr),
        tonic::transport::Server::builder()
            .add_service(GardenerServer::new(gardener_service))
            .serve(gardener_addr),
    )?;

    Ok(())
}
