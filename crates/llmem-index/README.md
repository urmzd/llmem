# llmem-index

ANN index implementations and code chunking for [llmem](https://github.com/urmzd/llmem) — the open ecosystem for tool-agnostic AI agent memory.

## What's included

- **AnnIndex trait** — pluggable approximate nearest neighbor interface
- **HNSW** — Hierarchical Navigable Small World graph (default, best recall)
- **IVF-Flat** — Inverted File Index with flat search (faster for large datasets)
- **CodeIndex** — tree-sitter semantic chunking for Rust, Python, JS/TS, Go
- **Distance functions** — cosine similarity, L2 distance, dot product

## Usage

```rust
use llmem_index::{AnnIndex, hnsw::{HnswIndex, HnswConfig}};

let mut index = HnswIndex::new(768, HnswConfig::default());
index.insert("doc-1", &embedding)?;

let hits = index.search(&query_vec, 10)?;
for hit in hits {
    println!("{}: {}", hit.id, hit.score);
}
```

## Features

Language support is feature-gated (all enabled by default):

- `lang-rust` — Rust via tree-sitter-rust
- `lang-python` — Python via tree-sitter-python
- `lang-javascript` — JavaScript/TypeScript via tree-sitter-javascript/typescript
- `lang-go` — Go via tree-sitter-go

## License

Apache-2.0
