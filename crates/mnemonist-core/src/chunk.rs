/// A segment of content produced by a [`ChunkingStrategy`].
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Relative file path from project root.
    pub file: String,
    /// Start line (1-indexed).
    pub start_line: usize,
    /// End line (1-indexed).
    pub end_line: usize,
    /// The text content.
    pub content: String,
}

impl Chunk {
    /// Unique ID: `file:start:end`.
    pub fn id(&self) -> String {
        format!("{}:{}:{}", self.file, self.start_line, self.end_line)
    }
}

/// Strategy for splitting content into chunks for embedding and retrieval.
///
/// mnemonist is a formalized RAG + memory architecture — chunking is the
/// first stage of the pipeline. Implementations decide how to partition
/// raw text into embeddable units.
pub trait ChunkingStrategy: Send + Sync {
    /// Split `content` (from `file`) into chunks.
    fn chunk(&self, content: &str, file: &str) -> Vec<Chunk>;
}
