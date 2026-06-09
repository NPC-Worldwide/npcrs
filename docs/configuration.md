# NPC Configuration

NPC agents are configured using `.npc` files. These YAML files define the agent's identity, behavior, LLM provider, and optional custom endpoints.

## File Format

`.npc` files use YAML syntax:

```yaml
name: analyst
provider: ollama
model: llama3.2
system_prompt: |
  You are a data analyst. Provide clear, concise insights.
```

## Configuration Fields

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Unique identifier for the agent |
| `provider` | string | LLM provider identifier (see table below) |
| `model` | string | Model name/identifier for the provider |
| `system_prompt` | string | Base prompt defining agent behavior (Jinja2 template) |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `api_key` | string | API key for the provider. Use `${ENV_VAR}` for environment lookup |
| `api_url` | string | Custom API base URL for self-hosted or third-party endpoints |
| `temperature` | float | Sampling temperature (0.0 - 2.0, default: 0.7) |
| `max_tokens` | integer | Maximum tokens per response |
| `tools` | list | Tool names the agent is authorized to use |

## Provider Reference

| Provider ID | Description | Default API URL |
|-------------|-------------|-----------------|
| `ollama` | Ollama local models | `http://localhost:11434/api/` |
| `lmstudio` | LM Studio OpenAI-compatible API | `http://localhost:1234/v1/` |
| `llamacpp` | llama.cpp GGUF models | Local only |
| `mlx` | Apple Silicon MLX | Local only |
| `transformers` | HuggingFace transformers | Local only |
| `openai-compatible` | Any OpenAI-compatible endpoint | Requires `api_url` |
| `openai-like` | Alias for `openai-compatible` | Requires `api_url` |

Provider identifiers are normalized internally. `openai-compatible` and `openai-like` resolve to `openai` for the underlying LLM client.

## OpenAI-Compatible Endpoints

Use these providers with custom `api_url` for local or third-party LLM servers:

### LM Studio

```yaml
name: lm-analyst
provider: openai-compatible
api_url: http://localhost:1234/v1/
model: qwen2.5-7b
system_prompt: |
  You are a helpful assistant running through LM Studio.
```

### LocalAI

```yaml
name: localai-assistant
provider: openai-compatible
api_url: http://localhost:8080/v1/
model: llama2-13b
system_prompt: |
  You are a helpful local AI assistant.
```

### vLLM

```yaml
name: vllm-local
provider: openai-compatible
api_url: http://localhost:8000/v1/
model: meta-llama/Llama-2-70b-chat-hf
system_prompt: |
  You are a local vLLM instance.
```

### URL Normalization

The `api_url` is automatically normalized — a trailing slash is added if missing:

- `http://localhost:1234/v1` → `http://localhost:1234/v1/`

## Security Considerations

!!! warning "API Key Storage"

    `.npc` files support plaintext `api_key` values. Best practices:

    - **Never commit `.npc` files with real API keys**
    - Use environment variable substitution: `api_key: ${OPENAI_API_KEY}`
    - Set file permissions: `chmod 600 *.npc`

## Examples

### Local Ollama Agent

```yaml
name: coder
provider: ollama
model: codellama:7b-code
temperature: 0.1
system_prompt: |
  You are a coding assistant.
```

### MLX Agent on Apple Silicon

```yaml
name: mlx-analyst
provider: mlx
model: mlx-community/Qwen2.5-7B-Instruct
system_prompt: |
  You are a data analyst.
```

### Transformers Agent

```yaml
name: hf-coder
provider: transformers
model: Qwen/Qwen2.5-Coder-7B-Instruct
system_prompt: |
  You are a coding assistant.
```
