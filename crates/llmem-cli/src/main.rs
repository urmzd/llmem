use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use llmem_core::{
    Embedder, Frontmatter, IndexEntry, MemoryFile, MemoryIndex, MemoryType, global_dir, project_dir,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::io::{self, Read as _};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "llmem", about = "Manage AI agent memory")]
struct Cli {
    #[command(subcommand)]
    command: Command,

    /// Force JSON output (default when stdout is not a TTY)
    #[arg(long, global = true)]
    json: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Initialize a memory directory
    Init {
        /// Initialize global memory (~/.config/llmem/) instead of project
        #[arg(long, short)]
        global: bool,

        /// Project root (defaults to current directory)
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },
    /// Add a new memory
    Add {
        /// Memory type: user, feedback, project, reference
        #[arg(value_parser = parse_memory_type)]
        memory_type: MemoryType,

        /// Short kebab-case name
        name: String,

        /// One-line description
        #[arg(long, short)]
        description: String,

        /// Memory body content
        #[arg(long, short)]
        body: Option<String>,

        /// Store in global memory instead of project
        #[arg(long, short)]
        global: bool,

        /// Project root (defaults to current directory)
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },
    /// List memories from the index
    List {
        /// Show global memories instead of project
        #[arg(long, short)]
        global: bool,

        /// Show both project and global memories
        #[arg(long, short)]
        all: bool,

        /// Project root (defaults to current directory)
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },
    /// Search memories by query
    Search {
        /// Search query
        query: String,

        /// Search level: project, global, or both
        #[arg(long, short, default_value = "both")]
        level: String,

        /// Project root (defaults to current directory)
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },
    /// Remove a memory by filename
    Remove {
        /// Filename to remove (e.g., feedback_standards.md)
        file: String,

        /// Remove from global memory instead of project
        #[arg(long, short)]
        global: bool,

        /// Project root (defaults to current directory)
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },
    /// Sync embeddings and rebuild the ANN index
    Embed {
        /// Sync global memory instead of project
        #[arg(long, short)]
        global: bool,

        /// Project root (defaults to current directory)
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },
    /// Context management
    Ctx {
        #[command(subcommand)]
        action: CtxAction,
    },
    /// Recall relevant memories (pre-hook)
    Recall {
        /// Search query (reads JSON from stdin if omitted)
        #[arg(long, short)]
        query: Option<String>,

        /// Max output chars (approximate token budget)
        #[arg(long, default_value = "2000")]
        budget: usize,

        /// Search level: project, global, or both
        #[arg(long, short, default_value = "both")]
        level: String,

        /// Project root (defaults to current directory)
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },
    /// Learn a new memory (post-hook, upsert semantics)
    Learn {
        /// Memory type: user, feedback, project, reference
        #[arg(value_parser = parse_memory_type)]
        memory_type: Option<MemoryType>,

        /// Short kebab-case name
        name: Option<String>,

        /// One-line description
        #[arg(long, short)]
        description: Option<String>,

        /// Memory body content
        #[arg(long, short)]
        body: Option<String>,

        /// Store in global memory instead of project
        #[arg(long, short)]
        global: bool,

        /// Read memory from JSON on stdin
        #[arg(long)]
        stdin: bool,

        /// Project root (defaults to current directory)
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },
    /// Code indexing and search
    Code {
        #[command(subcommand)]
        action: CodeAction,
    },
}

#[derive(Subcommand)]
enum CtxAction {
    /// Switch active project context
    Switch {
        /// Project root to switch to (defaults to current directory)
        #[arg(default_value = ".")]
        root: PathBuf,
    },
    /// Show the currently active context
    Show,
}

#[derive(Subcommand)]
enum CodeAction {
    /// Index source code using tree-sitter
    Index {
        /// Project root (defaults to current directory)
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },
    /// Search indexed code chunks
    Search {
        /// Search query
        query: String,

        /// Max results
        #[arg(long, short = 'k', default_value = "10")]
        top_k: usize,

        /// Project root (defaults to current directory)
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },
}

fn parse_memory_type(s: &str) -> Result<MemoryType, String> {
    s.parse()
}

fn resolve_dir(global: bool, root: &std::path::Path) -> Result<PathBuf> {
    if global {
        global_dir().context("could not determine config directory")
    } else {
        Ok(project_dir(root))
    }
}

/// Collect memory directories for a given level string.
fn level_dirs(level: &str, root: &std::path::Path) -> Result<Vec<PathBuf>> {
    match level {
        "project" => Ok(vec![project_dir(root)]),
        "global" => Ok(vec![
            global_dir().context("could not determine config directory")?,
        ]),
        _ => {
            let mut v = vec![project_dir(root)];
            if let Some(g) = global_dir() {
                v.push(g);
            }
            Ok(v)
        }
    }
}

// -- JSON output helpers --

fn output_ok(data: Value) {
    let out = json!({"ok": true, "data": data});
    println!("{}", serde_json::to_string(&out).unwrap());
}

fn output_err(msg: &str) -> ! {
    let out = json!({"ok": false, "error": msg});
    println!("{}", serde_json::to_string(&out).unwrap());
    std::process::exit(1);
}

/// Print to stderr for human UX (ignored when piped).
macro_rules! info {
    ($($arg:tt)*) => {
        eprintln!($($arg)*);
    };
}

// -- Stdin JSON types for learn --

#[derive(Deserialize)]
struct LearnInput {
    #[serde(rename = "type")]
    memory_type: String,
    name: String,
    description: String,
    #[serde(default)]
    body: String,
    #[serde(default = "default_level")]
    level: String,
}

fn default_level() -> String {
    "project".to_string()
}

#[derive(Deserialize)]
struct RecallInput {
    query: String,
}

// -- Serializable output types --

#[derive(Serialize)]
struct EntryOut {
    title: String,
    file: String,
    summary: String,
}

impl From<&IndexEntry> for EntryOut {
    fn from(e: &IndexEntry) -> Self {
        Self {
            title: e.title.clone(),
            file: e.file.clone(),
            summary: e.summary.clone(),
        }
    }
}

#[derive(Serialize)]
struct SearchResultOut {
    title: String,
    file: String,
    summary: String,
    level: String,
}

#[derive(Serialize)]
struct MemoryOut {
    file: String,
    #[serde(rename = "type")]
    memory_type: String,
    name: String,
    description: String,
    body: String,
}

fn main() {
    let cli = Cli::parse();

    if let Err(e) = run(cli) {
        output_err(&format!("{e:#}"));
    }
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Init { global, root } => {
            let dir = resolve_dir(global, &root)?;
            MemoryIndex::init(&dir)?;
            let level = if global { "global" } else { "project" };
            info!("initialized {level} memory at {}", dir.display());
            output_ok(json!({
                "level": level,
                "path": dir.display().to_string(),
            }));
        }

        Command::Add {
            memory_type,
            name,
            description,
            body,
            global,
            root,
        } => {
            let dir = resolve_dir(global, &root)?;
            let mut index = MemoryIndex::load(&dir)?;

            let mem = MemoryFile {
                frontmatter: Frontmatter {
                    name: name.clone(),
                    description: description.clone(),
                    memory_type,
                },
                body: body.unwrap_or_default(),
            };

            let filename = mem.filename();
            let entry = IndexEntry {
                title: name.replace('-', " "),
                file: filename.clone(),
                summary: description,
            };

            index.add(entry)?;
            mem.write(&dir.join(&filename))?;
            index.save()?;

            info!("added {filename}");
            output_ok(json!({
                "file": filename,
                "action": "created",
            }));
        }

        Command::List { global, all, root } => {
            let dirs = if all {
                let mut v = vec![project_dir(&root)];
                if let Some(g) = global_dir() {
                    v.push(g);
                }
                v
            } else {
                vec![resolve_dir(global, &root)?]
            };

            let mut entries = Vec::new();
            for dir in dirs {
                if let Ok(index) = MemoryIndex::load(&dir) {
                    for entry in &index.entries {
                        entries.push(EntryOut::from(entry));
                    }
                }
            }

            output_ok(json!({ "entries": entries }));
        }

        Command::Search { query, level, root } => {
            let dirs = level_dirs(&level, &root)?;
            let mut results = Vec::new();

            for dir in &dirs {
                let level_name = if dir == &project_dir(&root) {
                    "project"
                } else {
                    "global"
                };
                if let Ok(index) = MemoryIndex::load(dir) {
                    for entry in index.search(&query) {
                        results.push(SearchResultOut {
                            title: entry.title.clone(),
                            file: entry.file.clone(),
                            summary: entry.summary.clone(),
                            level: level_name.to_string(),
                        });
                    }
                }
            }

            output_ok(json!({ "results": results }));
        }

        Command::Remove { file, global, root } => {
            let dir = resolve_dir(global, &root)?;
            let mut index = MemoryIndex::load(&dir)?;

            if index.remove(&file) {
                let path = dir.join(&file);
                if path.exists() {
                    std::fs::remove_file(&path)?;
                }
                index.save()?;
                info!("removed {file}");
                output_ok(json!({
                    "file": file,
                    "action": "removed",
                }));
            } else {
                output_err(&format!("not found: {file}"));
            }
        }

        Command::Embed { global, root } => {
            let dir = resolve_dir(global, &root)?;
            let store_path = dir.join(".embeddings.bin");

            let embedder = llmem_core::OllamaEmbedder::from_env();
            info!("connecting to Ollama at {}", embedder.host());

            // Load existing or create new
            let mut store = if store_path.exists() {
                llmem_core::EmbeddingStore::load(&store_path)?
            } else {
                llmem_core::EmbeddingStore::new(embedder.dimension()?)
            };

            let synced = store.sync(&dir, &embedder)?;
            store.save(&store_path)?;

            info!(
                "synced {} embeddings ({} total, dim={})",
                synced,
                store.entries.len(),
                store.dimension
            );

            output_ok(json!({
                "synced": synced,
                "total": store.entries.len(),
                "dimension": store.dimension,
                "path": store_path.display().to_string(),
            }));
        }

        Command::Ctx { action } => match action {
            CtxAction::Switch { root } => {
                let root =
                    std::fs::canonicalize(&root).context("could not resolve project root")?;
                let ctx_dir = global_dir().context("could not determine config directory")?;
                std::fs::create_dir_all(&ctx_dir)?;
                let ctx_file = ctx_dir.join(".active-ctx");
                std::fs::write(&ctx_file, root.display().to_string())?;
                info!("switched context to {}", root.display());
                output_ok(json!({
                    "context": root.display().to_string(),
                }));
            }
            CtxAction::Show => {
                let ctx_dir = global_dir().context("could not determine config directory")?;
                let ctx_file = ctx_dir.join(".active-ctx");
                if ctx_file.exists() {
                    let ctx = std::fs::read_to_string(&ctx_file)?;
                    output_ok(json!({
                        "context": ctx.trim(),
                    }));
                } else {
                    output_ok(json!({
                        "context": null,
                    }));
                }
            }
        },

        Command::Recall {
            query,
            budget,
            level,
            root,
        } => {
            // Get query from arg or stdin
            let q = match query {
                Some(q) => q,
                None => {
                    let mut buf = String::new();
                    io::stdin()
                        .read_to_string(&mut buf)
                        .context("failed to read stdin")?;
                    let input: RecallInput = serde_json::from_str(&buf)
                        .context("stdin must be JSON with a \"query\" field")?;
                    input.query
                }
            };

            let dirs = level_dirs(&level, &root)?;
            let mut memories: Vec<MemoryOut> = Vec::new();
            let mut total_chars = 0usize;

            // Collect matching entries from all levels
            let mut matched: Vec<(PathBuf, IndexEntry)> = Vec::new();
            for dir in &dirs {
                if let Ok(index) = MemoryIndex::load(dir) {
                    for entry in index.search(&q) {
                        matched.push((dir.clone(), entry.clone()));
                    }
                }
            }

            // Sort by type priority: feedback > project > user > reference
            matched.sort_by_key(|(_, e)| type_priority_from_filename(&e.file));

            // Load memory files up to budget
            for (dir, entry) in &matched {
                if total_chars >= budget {
                    break;
                }
                let path = dir.join(&entry.file);
                if let Ok(mem) = MemoryFile::read(&path) {
                    let body_chars = mem.body.len();
                    if total_chars + body_chars > budget && !memories.is_empty() {
                        break;
                    }
                    total_chars += body_chars;
                    memories.push(MemoryOut {
                        file: entry.file.clone(),
                        memory_type: mem.frontmatter.memory_type.to_string(),
                        name: mem.frontmatter.name.clone(),
                        description: mem.frontmatter.description.clone(),
                        body: mem.body.clone(),
                    });
                }
            }

            output_ok(json!({
                "memories": memories,
                "token_estimate": total_chars / 4,
            }));
        }

        Command::Learn {
            memory_type,
            name,
            description,
            body,
            global,
            stdin,
            root,
        } => {
            let (mt, n, desc, b, is_global) = match memory_type {
                Some(mt) if !stdin => {
                    let n = name.context("name is required")?;
                    let desc = description.context("description is required (-d)")?;
                    (mt, n, desc, body.unwrap_or_default(), global)
                }
                _ => {
                    let mut buf = String::new();
                    io::stdin()
                        .read_to_string(&mut buf)
                        .context("failed to read stdin")?;
                    let input: LearnInput =
                        serde_json::from_str(&buf).context("invalid JSON on stdin")?;
                    let mt: MemoryType = input
                        .memory_type
                        .parse()
                        .map_err(|e: String| anyhow::anyhow!(e))?;
                    let is_global = input.level == "global";
                    (mt, input.name, input.description, input.body, is_global)
                }
            };

            let dir = resolve_dir(is_global, &root)?;
            let mut index = MemoryIndex::load(&dir)?;

            let mem = MemoryFile {
                frontmatter: Frontmatter {
                    name: n.clone(),
                    description: desc.clone(),
                    memory_type: mt,
                },
                body: b,
            };

            let filename = mem.filename();
            let entry = IndexEntry {
                title: n.replace('-', " "),
                file: filename.clone(),
                summary: desc,
            };

            let action = index.upsert(entry);
            mem.write(&dir.join(&filename))?;
            index.save()?;

            info!("{action} {filename}");
            output_ok(json!({
                "file": filename,
                "action": action,
            }));
        }

        Command::Code { action } => match action {
            CodeAction::Index { root } => {
                let root =
                    std::fs::canonicalize(&root).context("could not resolve project root")?;
                let mut code_index = llmem_index::code::CodeIndex::new(&root);
                let chunk_count = code_index.index()?;

                // Count unique files
                let file_count = code_index
                    .chunks()
                    .iter()
                    .map(|c| &c.file)
                    .collect::<std::collections::HashSet<_>>()
                    .len();

                // Save chunk manifest
                let mem_dir = project_dir(&root);
                std::fs::create_dir_all(&mem_dir)?;
                let manifest_path = mem_dir.join(".code-index.json");

                let manifest: Vec<Value> = code_index
                    .chunks()
                    .iter()
                    .map(|c: &llmem_index::code::CodeChunk| {
                        json!({
                            "id": c.id(),
                            "file": c.file,
                            "kind": c.kind,
                            "name": c.name,
                            "start_line": c.start_line,
                            "end_line": c.end_line,
                        })
                    })
                    .collect();

                let now = chrono::Utc::now().to_rfc3339();
                let manifest_doc = json!({
                    "root": root.display().to_string(),
                    "indexed_at": now,
                    "chunks": manifest,
                });
                std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest_doc)?)?;

                info!("indexed {} chunks from {} files", chunk_count, file_count);
                output_ok(json!({
                    "chunks": chunk_count,
                    "files": file_count,
                    "indexed_at": now,
                    "manifest": manifest_path.display().to_string(),
                }));
            }

            CodeAction::Search { query, top_k, root } => {
                let root =
                    std::fs::canonicalize(&root).context("could not resolve project root")?;
                let mem_dir = project_dir(&root);
                let manifest_path = mem_dir.join(".code-index.json");

                if !manifest_path.exists() {
                    output_err("no code index found — run `llmem code index` first");
                }

                let manifest_str = std::fs::read_to_string(&manifest_path)?;
                let manifest_doc: Value = serde_json::from_str(&manifest_str)?;
                let chunks = manifest_doc["chunks"]
                    .as_array()
                    .context("invalid manifest")?;

                let query_lower = query.to_lowercase();
                let mut hits: Vec<Value> = Vec::new();

                for chunk in chunks {
                    let name = chunk["name"].as_str().unwrap_or("");
                    let file = chunk["file"].as_str().unwrap_or("");
                    let kind = chunk["kind"].as_str().unwrap_or("");

                    let name_lower = name.to_lowercase();
                    let file_lower = file.to_lowercase();
                    let kind_lower = kind.to_lowercase();

                    // Score: exact name > partial name > file path > kind
                    let score = if name_lower == query_lower {
                        1.0
                    } else if name_lower.contains(&query_lower) {
                        0.8
                    } else if file_lower.contains(&query_lower) {
                        0.5
                    } else if kind_lower.contains(&query_lower) {
                        0.3
                    } else {
                        continue;
                    };

                    hits.push(json!({
                        "file": file,
                        "name": name,
                        "kind": kind,
                        "start_line": chunk["start_line"],
                        "end_line": chunk["end_line"],
                        "score": score,
                    }));
                }

                // Sort by score descending
                hits.sort_by(|a, b| {
                    b["score"]
                        .as_f64()
                        .unwrap_or(0.0)
                        .partial_cmp(&a["score"].as_f64().unwrap_or(0.0))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                hits.truncate(top_k);

                output_ok(json!({ "hits": hits }));
            }
        },
    }

    Ok(())
}

/// Map memory type prefix in filename to priority (lower = higher priority).
fn type_priority_from_filename(filename: &str) -> u8 {
    if filename.starts_with("feedback_") {
        0
    } else if filename.starts_with("project_") {
        1
    } else if filename.starts_with("user_") {
        2
    } else {
        3
    }
}
