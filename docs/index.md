# NPCrs — Rust Core for NPC

Rust implementation of the core NPC engine providing the agent kernel, Jinx executor, and LLM client.

## What is NPCrs?

NPCrs is the Rust core for the NPC (Next-level Prompting & Command) system. It provides:

- **NPC Compiler** — Parses `.npc` configuration files defining agent personas, behaviors, and capabilities
- **Jinx Executor** — Executes `.jinx` templates with a Jinja2-like engine
- **Agent Kernel** — Core runtime for managing agent state and execution
- **LLM Client** — Unified interface to Ollama, LM Studio, MLX, llama.cpp, and OpenAI-compatible endpoints

## Providers and Models

NPCrs supports multiple inference backends:

| Provider | Example Models | Notes |
|----------|---------------|-------|
| `ollama` | `llama3.2`, `gemma3:4b`, `qwen3:latest` | Local, free |
| `lmstudio` | `qwen2.5-7b`, `llama-3.1-8b` | OpenAI-compatible on port 1234 |
| `llamacpp` | path to GGUF file | Direct llama.cpp binding |
| `transformers` | HuggingFace model ID | Full HuggingFace transformers |
| `mlx` | model identifier | Apple Silicon MLX |
| `openai-compatible` | any | Custom OpenAI-compatible servers |

## Quick Start

Create an `.npc` file in your project directory:

```yaml
name: analyst
provider: ollama
model: llama3.2
system_prompt: |
  You are a data analyst. Provide clear, concise insights.
```

Or use a custom OpenAI-compatible endpoint (LM Studio, etc.):

```yaml
name: local-analyst
provider: openai-compatible
api_url: http://localhost:1234/v1/
model: qwen2.5-7b
system_prompt: |
  You are a data analyst.
```

Then from Rust:

```rust
use npcrs::NPC;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let npc = NPC::from_file("analyst.npc")?;
    let response = npc.get_response("What is 2+2?", None).await?;
    println!("{}", response);
    Ok(())
}
```

## Related Projects

- [npcpy](https://github.com/NPC-Worldwide/npcpy) — Core Python library for NPC data structures
- [npcsh](https://github.com/NPC-Worldwide/npcsh) — Multi-agent shell with Jinja execution
- [nql](https://github.com/NPC-Worldwide/nql) — NPC Query Language for AI-powered SQL

## License

MIT License — See [LICENSE](https://github.com/NPC-Worldwide/npcrs/blob/main/LICENSE).
