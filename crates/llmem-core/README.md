# llmem-core

Core types and operations for [llmem](https://github.com/urmzd/llmem) — the open ecosystem for tool-agnostic AI agent memory.

## What's included

- **Memory types** — `MemoryFile`, `Frontmatter`, `MemoryType` (user, feedback, project, reference)
- **Index** — `MemoryIndex` for managing `MEMORY.md` index files with add/upsert/remove/search
- **Embedder trait** — pluggable embedding interface with built-in `OllamaEmbedder`
- **Embedding store** — binary format for persisting embeddings alongside memory files
- **File backend** — `FileBackend` implementing `MemoryBackend` for file-based storage
- **Config** — `Config` struct for `~/.llmem/config.toml` management

## Usage

```rust
use llmem_core::{Config, MemoryIndex, MemoryFile, OllamaEmbedder, Embedder};

// Load config and resolve paths
let config = Config::load();
let project_dir = config.project_dir(std::path::Path::new("."));

// Load memory index
let index = MemoryIndex::load(&project_dir)?;
let results = index.search("rust");

// Embed with Ollama
let embedder = OllamaEmbedder::from_config(&config);
let vector = embedder.embed("some text")?;
```

## License

Apache-2.0
