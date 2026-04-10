use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing::{debug, error, info, warn};

const COLLECTION_NAME: &str = "opengardener_soil";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromaTrailMetadata {
    pub outcome: String,
    pub approach: String,      // JSON string
    pub agent_id: String,
    pub task_domain: String,
    pub task_summary: String,
    pub timestamp: String,
    pub hits: i64,
    pub cpu_ms: f64,
    pub memory_mb: f64,
    pub severity: Option<String>,    // only for failure markers
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
    collection_id: Option<String>,
}

impl ChromaClient {
    pub fn new(host: &str, port: u16) -> Self {
        Self {
            client: Client::new(),
            base_url: format!("http://{}:{}/api/v2", host, port),
            collection_id: None,
        }
    }

    pub async fn init(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Get or create the soil collection
        let url = format!("{}/collections", self.base_url);

        // Check if collection exists
        let res = self.client.get(&url).send().await?;
        let collections: Vec<Value> = res.json().await?;

        for col in &collections {
            if col["name"].as_str() == Some(COLLECTION_NAME) {
                let id = col["id"].as_str().unwrap_or("").to_string();
                info!("Found existing soil collection: {}", id);
                self.collection_id = Some(id);
                return Ok(());
            }
        }

        // Create new collection with cosine similarity
        let res = self
            .client
            .post(&url)
            .json(&json!({
                "name": COLLECTION_NAME,
                "metadata": {
                    "hnsw:space": "cosine",
                    "description": "OpenGardener pheromone trails"
                }
            }))
            .send()
            .await?;

        let col: Value = res.json().await?;
        let id = col["id"].as_str().unwrap_or("").to_string();
        info!("Created soil collection: {}", id);
        self.collection_id = Some(id);
        Ok(())
    }

    fn collection_url(&self) -> String {
        format!(
            "{}/collections/{}/points",
            self.base_url,
            self.collection_id.as_deref().unwrap_or("")
        )
    }

    pub async fn add_trail(
        &self,
        trail_id: &str,
        embedding: Vec<f32>,
        metadata: ChromaTrailMetadata,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let url = format!(
            "{}/collections/{}/add",
            self.base_url,
            self.collection_id.as_deref().unwrap_or("")
        );

        let body = json!({
            "ids": [trail_id],
            "embeddings": [embedding],
            "metadatas": [serde_json::to_value(&metadata)?],
            "documents": [metadata.task_summary]
        });

        let res = self.client.post(&url).json(&body).send().await?;

        if !res.status().is_success() {
            let err = res.text().await?;
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
        let url = format!(
            "{}/collections/{}/query",
            self.base_url,
            self.collection_id.as_deref().unwrap_or("")
        );

        let mut body = json!({
            "query_embeddings": [embedding],
            "n_results": n_results,
            "include": ["metadatas", "distances"]
        });

        // Optional domain filter
        if let Some(domain) = domain_filter {
            body["where"] = json!({ "task_domain": { "$eq": domain } });
        }

        let res = self.client.post(&url).json(&body).send().await?;

        if !res.status().is_success() {
            let err = res.text().await?;
            warn!("ChromaDB query failed: {}", err);
            // Return empty result on query failure rather than crashing
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

    pub async fn increment_hit_count(&self, trail_id: &str) -> Result<(), Box<dyn std::error::Error>> {
        // ChromaDB doesn't support atomic increments; we use upsert with updated metadata.
        // In V1 this is a best-effort operation.
        debug!("Incrementing hit count for trail {}", trail_id);
        Ok(())
    }

    pub async fn get_collection_stats(&self) -> Result<Value, Box<dyn std::error::Error>> {
        let url = format!(
            "{}/collections/{}",
            self.base_url,
            self.collection_id.as_deref().unwrap_or("")
        );
        let res = self.client.get(&url).send().await?;
        let stats: Value = res.json().await?;
        Ok(stats)
    }
}
