mod gardener;
mod grpc;
mod metrics;
mod soil;
mod types;

use std::env;
use std::sync::Arc;
use tokio::signal;
use tracing::{info, warn};
use tracing_subscriber::{fmt, EnvFilter};

use gardener::GardenerCore;
use grpc::{
    gardener_server::{GardenerGrpcService, gardener_proto::gardener_server::GardenerServer},
    soil_server::{SoilGrpcService, soil_proto::soil_server::SoilServer},
};
use soil::Soil;

fn parse_env<T: std::str::FromStr>(key: &str, default: T) -> T {
    env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    info!("OpenGardener V1 starting...");

    let chroma_host = env::var("CHROMA_HOST").unwrap_or_else(|_| "localhost".to_string());
    let chroma_port: u16 = parse_env("CHROMA_PORT", 8000);
    let soil_port: u16 = parse_env("SOIL_GRPC_PORT", 50051);
    let gardener_port: u16 = parse_env("GARDENER_GRPC_PORT", 50052);
    let metrics_port: u16 = parse_env("METRICS_PORT", 8080);

    info!("Connecting to ChromaDB at {}:{}", chroma_host, chroma_port);
    let soil = Arc::new(
        Soil::new(&chroma_host, chroma_port)
            .await
            .map_err(|e| {
                let msg = format!("Failed to initialize Soil: {} — is ChromaDB reachable?", e);
                std::io::Error::new(std::io::ErrorKind::Other, msg)
            })?,
    );

    let gardener_core = Arc::new(GardenerCore::new(Arc::clone(&soil)));
    let cancel = gardener_core.cancellation_token();

    // Observation loop.
    let gardener_observer = Arc::clone(&gardener_core);
    let observer_handle = tokio::spawn(async move {
        gardener_observer.observe_ecosystem().await;
    });

    // Metrics HTTP server.
    let gardener_metrics = Arc::clone(&gardener_core);
    let soil_metrics = Arc::clone(&soil);
    let metrics_cancel = cancel.clone();
    let metrics_handle = tokio::spawn(async move {
        metrics::serve(gardener_metrics, soil_metrics, metrics_port, metrics_cancel).await;
    });

    let soil_addr = format!("0.0.0.0:{}", soil_port).parse()?;
    let gardener_addr = format!("0.0.0.0:{}", gardener_port).parse()?;

    info!("Soil gRPC server listening on {}", soil_addr);
    info!("Gardener gRPC server listening on {}", gardener_addr);
    info!("Metrics HTTP server listening on 0.0.0.0:{}", metrics_port);

    let soil_service = SoilGrpcService { soil: Arc::clone(&soil) };
    let gardener_service = GardenerGrpcService { gardener: Arc::clone(&gardener_core) };

    let soil_cancel = cancel.clone();
    let soil_server = tonic::transport::Server::builder()
        .add_service(SoilServer::new(soil_service))
        .serve_with_shutdown(soil_addr, async move { soil_cancel.cancelled().await });

    let gard_cancel = cancel.clone();
    let gardener_server = tonic::transport::Server::builder()
        .add_service(GardenerServer::new(gardener_service))
        .serve_with_shutdown(gardener_addr, async move { gard_cancel.cancelled().await });

    // Listen for SIGTERM (Docker stop) and SIGINT (Ctrl-C). Either triggers a
    // cooperative shutdown of the gRPC servers and the observer loop.
    let shutdown_cancel = cancel.clone();
    tokio::spawn(async move {
        let ctrl_c = async { let _ = signal::ctrl_c().await; };
        #[cfg(unix)]
        let term = async {
            use signal::unix::{signal, SignalKind};
            if let Ok(mut s) = signal(SignalKind::terminate()) {
                s.recv().await;
            } else {
                std::future::pending::<()>().await;
            }
        };
        #[cfg(not(unix))]
        let term = std::future::pending::<()>();

        tokio::select! {
            _ = ctrl_c => info!("SIGINT received, beginning graceful shutdown"),
            _ = term => info!("SIGTERM received, beginning graceful shutdown"),
        }
        shutdown_cancel.cancel();
    });

    // Run both gRPC servers concurrently. If either errors, propagate.
    if let Err(e) = tokio::try_join!(soil_server, gardener_server) {
        warn!("gRPC server error: {}", e);
    }

    // Stop the observer/metrics tasks. The cancel was already triggered by the
    // shutdown handler — this just awaits their exit so logs are flushed cleanly.
    let _ = observer_handle.await;
    let _ = metrics_handle.await;

    info!("OpenGardener shutdown complete.");
    Ok(())
}
