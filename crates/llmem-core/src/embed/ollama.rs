use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::Error;
use crate::embed::Embedder;

/// Ollama-backed embedder using the `/api/embed` endpoint.
pub struct OllamaEmbedder {
    host: String,
    model: String,
    agent: ureq::Agent,
    dimension: OnceLock<usize>,
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a str,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

impl OllamaEmbedder {
    /// Create a new Ollama embedder with explicit host and model.
    pub fn new(host: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            host: host.into(),
            model: model.into(),
            agent: ureq::Agent::new_with_config(
                ureq::config::Config::builder()
                    .timeout_global(Some(std::time::Duration::from_secs(60)))
                    .build(),
            ),
            dimension: OnceLock::new(),
        }
    }

    /// Create from environment variables.
    /// - `OLLAMA_HOST` (default: `http://localhost:11434`)
    /// - `OLLAMA_EMBED_MODEL` (default: `nomic-embed-text`)
    pub fn from_env() -> Self {
        let host =
            std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());
        let model =
            std::env::var("OLLAMA_EMBED_MODEL").unwrap_or_else(|_| "nomic-embed-text".to_string());
        Self::new(host, model)
    }

    /// The configured Ollama host URL.
    pub fn host(&self) -> &str {
        &self.host
    }

    fn embed_text(&self, text: &str) -> Result<Vec<f32>, Error> {
        let url = format!("{}/api/embed", self.host);
        let req = EmbedRequest {
            model: &self.model,
            input: text,
        };

        let resp: EmbedResponse = self
            .agent
            .post(&url)
            .send_json(&req)
            .map_err(|e| Error::Embedding(format!("ollama request failed: {e}")))?
            .body_mut()
            .read_json()
            .map_err(|e| Error::Embedding(format!("ollama response parse failed: {e}")))?;

        resp.embeddings
            .into_iter()
            .next()
            .ok_or_else(|| Error::Embedding("no embeddings returned".into()))
    }
}

impl Embedder for OllamaEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, Error> {
        let vec = self.embed_text(text)?;
        // Cache dimension on first call
        let _ = self.dimension.set(vec.len());
        Ok(vec)
    }

    fn dimension(&self) -> Result<usize, Error> {
        if let Some(&dim) = self.dimension.get() {
            return Ok(dim);
        }
        // Probe with a short text to discover dimension
        let vec = self.embed_text("hello")?;
        let dim = vec.len();
        let _ = self.dimension.set(dim);
        Ok(dim)
    }
}
