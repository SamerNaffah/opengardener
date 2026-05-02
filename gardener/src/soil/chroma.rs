//! Thin async client for ChromaDB v2 REST API (works with chromadb 0.5+ / 0.6.x).
//!
//! ChromaDB v2 paths are tenant- and database-scoped:
//!     /api/v2/tenants/{tenant}/databases/{database}/collections...
//!
//! For OpenGardener V1 we always use the default tenant / default database.
//! This client is internally cheap to clone (`reqwest::Client` shares its connection
//! pool via Arc), so callers wrap it in `Arc` rather than `Arc<RwLock<...>>`.
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;
use tracing::{debug, error, info, warn};

const COLLECTION_NAME: &str = "opengardener_soil";
const DEFAULT_TENANT: &str = "default_tenant";
const DEFAULT_DATABASE: &str = "default_database";

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChromaTrailMetadata {
    pub outcome: String,
    pub approach: String, // JSON string
    pub agent_id: String,
    pub task_domain: String,
    pub task_summary: String,
    pub timestamp: String,
    pub hits: i64,
    pub cpu_ms: f64,
    pub memory_mb: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub severity: Option<String>, // only for failure markers
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub avoid_in_future: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChromaQueryResult {
    pub ids: Vec<Vec<String>>,
    pub embeddings: Option<Vec<Vec<Vec<f32>>>>,
    pub metadatas: Vec<Vec<ChromaTrailMetadata>>,
    pub distances: Vec<Vec<f32>>,
}

pub struct ChromaClient {
    client: Client,
    base_url: String,
    tenant: String,
    database: String,
    collection_id: Option<String>,
}

impl ChromaClient {
    pub fn new(host: &str, port: u16) -> Self {
        // Tuned HTTP client: keepalives, pooled connections, modest timeouts.
        let client = Client::builder()
            .pool_idle_timeout(Some(Duration::from_secs(90)))
            .pool_max_idle_per_host(8)
            .tcp_keepalive(Some(Duration::from_secs(30)))
            .timeout(Duration::from_secs(15))
            .build()
            .expect("Failed to build reqwest client");

        Self {
            client,
            base_url: format!("http://{}:{}/api/v2", host, port),
            tenant: DEFAULT_TENANT.to_string(),
            database: DEFAULT_DATABASE.to_string(),
            collection_id: None,
        }
    }

    fn collections_root(&self) -> String {
        format!(
            "{}/tenants/{}/databases/{}/collections",
            self.base_url, self.tenant, self.database
        )
    }

    fn collection_url(&self, suffix: &str) -> String {
        format!(
            "{}/{}{}",
            self.collections_root(),
            self.collection_id.as_deref().unwrap_or(""),
            suffix
        )
    }

    /// Discover (or create) the soil collection. Idempotent.
    pub async fn init(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // List existing collections
        let url = self.collections_root();
        let res = self.client.get(&url).send().await?;

        if res.status().is_success() {
            let collections: Vec<Value> = res.json().await.unwrap_or_default();
            for col in &collections {
                if col["name"].as_str() == Some(COLLECTION_NAME) {
                    let id = col["id"].as_str().unwrap_or("").to_string();
                    info!("Found existing soil collection: {}", id);
                    self.collection_id = Some(id);
                    return Ok(());
                }
            }
        } else {
            warn!(
                "Listing collections returned HTTP {} — falling back to create-or-get",
                res.status()
            );
        }

        // Create. ChromaDB returns the existing collection if name matches and
        // get_or_create is true.
        let res = self
            .client
            .post(&url)
            .json(&json!({
                "name": COLLECTION_NAME,
                "metadata": {
                    "hnsw:space": "cosine",
                    "description": "OpenGardener pheromone trails"
                },
                "get_or_create": true
            }))
            .send()
            .await?;

        if !res.status().is_success() {
            let status = res.status();
            let body = res.text().await.unwrap_or_default();
            return Err(format!(
                "Failed to create soil collection (HTTP {}): {}",
                status, body
            )
            .into());
        }

        let col: Value = res.json().await?;
        let id = col["id"].as_str().unwrap_or("").to_string();
        if id.is_empty() {
            return Err("ChromaDB create_collection returned no id".into());
        }
        info!("Created/loaded soil collection: {}", id);
        self.collection_id = Some(id);
        Ok(())
    }

    pub async fn add_trail(
        &self,
        trail_id: &str,
        embedding: Vec<f32>,
        metadata: ChromaTrailMetadata,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let url = self.collection_url("/add");

        let body = json!({
            "ids":         [trail_id],
            "embeddings":  [embedding],
            "metadatas":   [serde_json::to_value(&metadata)?],
            "documents":   [metadata.task_summary]
        });

        let res = self.client.post(&url).json(&body).send().await?;

        if !res.status().is_success() {
            let err = res.text().await.unwrap_or_default();
            error!("ChromaDB add_trail failed: {}", err);
            return Err(err.into());
        }

        debug!("Trail {} added to soil", trail_id);
        Ok(())
    }

    pub async fn query_similar(
        &self,
        embedding: Vec<f32>,
        n_results: usize,
        domain_filter: Option<&str>,
    ) -> Result<ChromaQueryResult, Box<dyn std::error::Error>> {
        let url = self.collection_url("/query");

        let mut body = json!({
            "query_embeddings": [embedding],
            "n_results": n_results,
            "include": ["metadatas", "distances"]
        });

        if let Some(domain) = domain_filter {
            body["where"] = json!({ "task_domain": { "$eq": domain } });
        }

        let res = self.client.post(&url).json(&body).send().await?;

        if !res.status().is_success() {
            let status = res.status();
            let err = res.text().await.unwrap_or_default();
            warn!("ChromaDB query failed (HTTP {}): {}", status, err);
            // Return empty result on query failure rather than crashing the caller.
            return Ok(ChromaQueryResult {
                ids: vec![vec![]],
                embeddings: None,
                metadatas: vec![vec![]],
                distances: vec![vec![]],
            });
        }

        let result: ChromaQueryResult = res.json().await?;
        Ok(result)
    }

    /// Increment hit counter on a trail. Best-effort — Chroma has no atomic update,
    /// so we read the metadata, bump it, and call /update. Fails silently on race.
    pub async fn increment_hit_count(
        &self,
        trail_id: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Read the trail
        let get_url = self.collection_url("/get");
        let get_body = json!({
            "ids": [trail_id],
            "include": ["metadatas"]
        });
        let res = self.client.post(&get_url).json(&get_body).send().await?;
        if !res.status().is_success() {
            return Ok(());
        }
        let data: Value = res.json().await?;
        let metas = data["metadatas"].as_array().cloned().unwrap_or_default();
        let Some(meta_val) = metas.first() else {
            return Ok(());
        };
        let mut meta: ChromaTrailMetadata =
            serde_json::from_value(meta_val.clone()).unwrap_or_default();
        meta.hits += 1;

        let upd_url = self.collection_url("/update");
        let upd_body = json!({
            "ids": [trail_id],
            "metadatas": [serde_json::to_value(&meta)?]
        });
        let _ = self.client.post(&upd_url).json(&upd_body).send().await;
        debug!("Incremented hit count for trail {} to {}", trail_id, meta.hits);
        Ok(())
    }

    /// Total trail count via the dedicated `/count` endpoint.
    pub async fn count(&self) -> Result<u64, Box<dyn std::error::Error>> {
        let url = self.collection_url("/count");
        let res = self.client.get(&url).send().await?;
        if !res.status().is_success() {
            return Ok(0);
        }
        let v: Value = res.json().await?;
        // Some Chroma versions return a bare integer, others an object {"count": N}.
        Ok(v.as_u64()
            .or_else(|| v["count"].as_u64())
            .unwrap_or(0))
    }

    /// Per-domain trail count via `where` + `/count` (one call per domain).
    pub async fn count_by_domain(
        &self,
        domain: &str,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        // ChromaDB's /count doesn't accept where filters; fall back to /get with limit.
        let url = self.collection_url("/get");
        let body = json!({
            "where": { "task_domain": { "$eq": domain } },
            "limit": 10_000,
            "include": []
        });
        let res = self.client.post(&url).json(&body).send().await?;
        if !res.status().is_success() {
            return Ok(0);
        }
        let v: Value = res.json().await?;
        Ok(v["ids"].as_array().map(|a| a.len() as u64).unwrap_or(0))
    }

    /// Best-effort heartbeat for liveness checks.
    pub async fn ping(&self) -> Result<(), Box<dyn std::error::Error>> {
        let url = format!("{}/heartbeat", self.base_url);
        let res = self.client.get(&url).send().await?;
        if !res.status().is_success() {
            return Err(format!("ChromaDB heartbeat failed: HTTP {}", res.status()).into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collection_url_format_includes_tenant_and_database() {
        let mut c = ChromaClient::new("h", 8000);
        c.collection_id = Some("abc-123".into());
        let url = c.collection_url("/add");
        assert!(url.contains("/tenants/default_tenant/"));
        assert!(url.contains("/databases/default_database/"));
        assert!(url.ends_with("/collections/abc-123/add"));
    }

    #[test]
    fn metadata_serialises_round_trip() {
        let m = ChromaTrailMetadata {
            outcome: "success".into(),
            approach: "{}".into(),
            agent_id: "x".into(),
            task_domain: "data_cleaning".into(),
            task_summary: "s".into(),
            timestamp: "t".into(),
            hits: 1,
            cpu_ms: 1.0,
            memory_mb: 1.0,
            severity: None,
            avoid_in_future: None,
        };
        let s = serde_json::to_string(&m).unwrap();
        let m2: ChromaTrailMetadata = serde_json::from_str(&s).unwrap();
        assert_eq!(m2.outcome, "success");
        assert_eq!(m2.hits, 1);
    }
}
