# llmem-cli

CLI tool for [llmem](https://github.com/urmzd/llmem) — the open ecosystem for tool-agnostic AI agent memory.

## Install

```bash
cargo install llmem-cli
```

## Commands

| Command | Description |
|---------|-------------|
| `llmem init [--global]` | Initialize memory directory |
| `llmem add <type> <name> -d <desc>` | Add a memory |
| `llmem learn [--stdin]` | Upsert a memory (JSON stdin or args) |
| `llmem recall --query <q>` | Retrieve relevant memories (pre-hook) |
| `llmem list [--all]` | List memories |
| `llmem search <query>` | Search by description |
| `llmem remove <file>` | Remove a memory |
| `llmem embed [--global]` | Sync embeddings via Ollama |
| `llmem code index` | Index source code with tree-sitter |
| `llmem code search <query>` | Search indexed code chunks |
| `llmem config show\|get\|set\|init\|path` | Manage configuration |
| `llmem ctx switch\|show` | Manage project context |

All commands output JSON to stdout (`{"ok": true, "data": {...}}`).

## Storage

- Project memory: `~/.llmem/{project}/`
- Global memory: `~/.llmem/global/`
- Config: `~/.llmem/config.toml`

## License

Apache-2.0
