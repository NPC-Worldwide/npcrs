# npcrs

Rust core for the NPC system. Port of [npcpy](https://github.com/NPC-Worldwide/npcpy).

Agent kernel, jinx executor, multi-provider LLM client, knowledge graph, and FFI layer for embedding in shells, mobile apps, and servers.

## Modules

| Module | Description |
|--------|-------------|
| `kernel` | Boot, process table, scheduling, IPC |
| `npc` | NPC agent definition, .npc file loading |
| `jinx` | Jinx (tool/workflow) definitions, Jinja2 rendering, step execution |
| `team` | Team loading, forenpc coordination, orchestration |
| `llm` | Multi-provider LLM client (OpenAI, Anthropic, Gemini, Ollama, llama.cpp, LM Studio, custom) |
| `memory` | SQLite conversation history, knowledge graph (facts, concepts, links) |
| `process` | NPC process lifecycle, resource limits, token budgets |
| `ffi` | C-ABI for Flutter/Dart/mobile integration |

## Usage

```toml
[dependencies]
npcrs = "0.1"
```

```rust
use npcrs::{Kernel, Npc, Team};

// Boot kernel with a team directory
let kernel = Kernel::boot("./npc_team", "history.db")?;

// Execute a command through an NPC process
let output = kernel.exec(0, "hello").await?;

// Delegate to a specific NPC
let output = kernel.delegate(0, "corca", "search for papers on LLMs").await?;
```

## FFI

Produces `cdylib` and `staticlib` for embedding in Flutter, Android, iOS, desktop apps.

## License

MIT
