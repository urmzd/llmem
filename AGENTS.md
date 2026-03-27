[llmem] Open standard for tool-agnostic AI agent memory

## Project Overview

A specification and Rust implementation defining a convention for storing AI agent memory as plain markdown files. Two levels: project (`~/.llmem/{project}/`) and global (`~/.llmem/global/`). Uses cognitive metaphors for its CLI: memorize, remember, note, learn, consolidate, reflect, forget.

## Architecture

Rust workspace with four crates:
- `llmem-core` — spec types, file parsing, index operations, inbox, config, embeddings
- `llmem-cli` — CLI binary (`llmem`) with cognitive commands
- `llmem-server` — RAG HTTP server for semantic search
- `llmem-index` — ANN indices (HNSW, IVF-Flat) and tree-sitter code indexing
- `llmem-quant` — TurboQuant vector quantization (1-4 bit)

Training pipeline in `training/` (Python): data generation, model distillation, ONNX export.

## Discovering Structure

Use `tree` and `ripgrep` to discover project layout. Do not rely on static listings.

## Commands

```bash
cargo build              # build all crates
cargo test               # run all tests
cargo run -p llmem-cli   # run the CLI
```

## Code Style

- Rust edition 2024
- Workspace dependencies centralized in root `Cargo.toml`
- Error handling: `thiserror` for library errors, `anyhow` for binaries
- Serialization: `serde` + `serde_yaml` for frontmatter

## Commit Guidelines

Conventional commits via `sr commit`:
- `feat(core):` / `feat(cli):` / `feat(server):` / `feat(quant):` — scoped by crate
- `docs(spec):` — specification changes
- `docs(readme):` — documentation changes

## Extension Guide

To add a new memory type:
1. Add variant to `MemoryType` enum in `crates/llmem-core/src/memory.rs`
2. Update `Display` and `FromStr` implementations
3. Add the type to Section 5 of `SPECIFICATION.md`
4. Update the README memory types table
