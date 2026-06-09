# NPCRS Coverage Layout

Complete API surface of NPCRS for parity tracking with NPCPY/NPCTS.

## Core Types

### NPC (src/npc_compiler.rs:13-126)
- 35 fields including name, primary_directive, model, provider, api_key, etc.
- Methods: from_file, new, system_prompt, resolved_model, get_response, etc.
- Memory methods: create_memory, read_memory, search_memories, etc.
- Planning: generate_todos, execute_planning_item, think_step_by_step
- Jinx: execute_jinx

### Jinx (src/npc_compiler.rs:1133-1158)
- 8 fields: name, description, steps[], engine, etc.
- Methods: from_file, execute, render_first_pass, to_tool_def

### Team (src/npc_compiler.rs:1977-2050)
- 10 fields: name, npcs[], jinxes[], context, etc.
- Methods: get_npc, orchestrate, update_context, save

### Agent (src/npc_compiler.rs:2388-2392)
- Fields: npc, messages, tool_registry
- Methods: new, run (async)

### ToolRegistry (src/tools/mod.rs:11-14)
- register, tool_defs, execute, has_tool

### KnowledgeGraph (src/memory/knowledge_graph.rs:7-12)
- add_entity, add_relation, neighbors, to_json, from_json
- Special: kg_sleep_process, kg_dream_process

## Total API Surface
- ~35 structs
- ~164 functions/methods

## Cross-Reference
| Feature | NPCRS | Coverage |
|---------|-------|----------|
| Core NPC | Full | Complete |
| Jinx Execution | Full | Multi-step, all engines |
| Team/Agents | Full | Lead delegation |
| Tool Registry | Full | With default tools |
| Knowledge Graph | Full | Most complete impl |
| Process Kernel | Full | Context switching |
| Memory | Full | Lifecycle |
| Python Bridge | Full | Daemon subprocess |

---
*Generated: 2026-06-09*