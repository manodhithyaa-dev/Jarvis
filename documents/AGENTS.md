
# Agent Registry

A model-agnostic registry for a local Ollama-based assistant. The router selects agents by task, and agents may call tools or external backends where a local language model is not sufficient.

## Architecture Principles

- **One responsibility per agent.** Keep each agent's scope clear and testable.
- **Route before loading.** Do not load every model simultaneously; select the smallest suitable model and reuse it when practical.
- **Models are not tools.** Use deterministic code for calculations, schema validation, retrieval, file operations, and other tasks that do not require generation.
- **Verify model tags.** Run `ollama show <model>` before relying on a tag, especially custom or uncertain tags.
- **Validate consequential output.** Validate generated code, SQL, tool arguments, and structured output before execution.
- **Use least privilege.** Terminal, browser, file, and database tools should have explicit permissions and safe execution limits.
- **Support fallbacks.** If a model or backend is unavailable, return a clear error or route to an approved alternative.

## Core Agent Registry

| # | Agent ID | Primary model / backend | Responsibility |
|---:|---|---|---|
| 1 | `chat` | `qwen3:8b` | General conversation, explanations, and question answering |
| 2 | `reasoning` | `deepseek-r1:7b` | Multi-step reasoning, mathematics explanations, and difficult problem solving |
| 3 | `planner` | `qwen3.8:27b` (verify tag first) | Break complex tasks into ordered steps and delegate subtasks |
| 4 | `classifier` | `llama3.2:3b` | Classify user intent and select an agent or workflow |
| 5 | `quick_response` | `qwen3:4b` | Low-latency responses and simple tasks |
| 6 | `coding_quick` | `qwen2.5-coder:7b` | Small code snippets, functions, and localized fixes |
| 7 | `coding_deep` | `qwen3-coder:latest` | Multi-file coding, refactoring, and repository-level work |
| 8 | `code_review` | `qwen3-coder:latest` | Review correctness, readability, maintainability, and conventions |
| 9 | `debugger` | `deepseek-r1:7b` | Diagnose errors, trace likely causes, and propose/test fixes |
| 10 | `test_generation` | `qwen2.5-coder:7b` | Generate unit, integration, regression, and edge-case tests |
| 11 | `sql` | `sqlcoder:7b` | Generate and explain SQL for supported relational databases |
| 12 | `data_analysis` | `duckdb-nsql:7b` + DuckDB runtime | Query and analyze datasets; use a real execution engine for results |
| 13 | `vision` | `qwen3-vl:8b` | Understand images, screenshots, and diagrams |
| 14 | `ocr_document` | `qwen2.5vl:7b` | Read text from document images and inspect visual page layouts |
| 15 | `image_generation` | Compatible image-generation backend | Generate images from text prompts |
| 16 | `image_editing` | Compatible image-editing backend | Edit existing images using a backend that supports image input and editing |
| 17 | `text_to_audio` | `legraphista/Orpheus:latest` | Generate speech from text; verify model input/output requirements |
| 18 | `speech_to_text` | Whisper via `whisper.cpp` or `faster-whisper` | Transcribe audio into text |
| 19 | `text_to_video` | Compatible video-generation backend | Generate video from text prompts |
| 20 | `video_understanding` | `qwen3-vl:8b` + frame extraction | Analyze selected video frames and describe visible events |
| 21 | `translation` | `translategemma:4b` | Translate between supported languages |
| 22 | `summarization` | `qwen3:4b` | Summarize conversations, documents, and agent results |
| 23 | `research` | `qwen3:8b` + web/search tools | Research questions, synthesize retrieved sources, and cite evidence |
| 24 | `rag_retrieval` | `qwen3-embedding:4b` + retrieval engine | Embed queries and retrieve relevant knowledge-base passages |
| 25 | `document_reader` | `reader-lm:1.5b` | Clean or convert extracted webpage/document text |
| 26 | `memory_manager` | `qwen3:4b` + memory store | Propose useful memory updates and conversation summaries |
| 27 | `security_audit` | `qwen3-coder:latest` + security tools | Inspect code for vulnerabilities |
| 28 | `validator` | Deterministic validators + optional `qwen3:8b` | Validate schemas, required fields, constraints, and task requirements |
| 29 | `critic` | `deepseek-r1:7b` | Challenge assumptions, find inconsistencies, and identify failure modes |
| 30 | `medical_research` | `medllama2:7b` | Assist with medical literature and terminology; not a diagnostic authority |

## Optional Tool Agents

These can be implemented as tool-backed workflows rather than separate LLMs.

| Agent ID | Implementation | Responsibility |
|---|---|---|
| `web_search` | Search API / browser tool | Find sources and return URLs/snippets |
| `file_manager` | Sandboxed filesystem tool | Read, create, and modify permitted files |
| `terminal` | Restricted command runner | Execute approved commands with timeouts and working-directory limits |
| `browser_automation` | Browser automation tool | Navigate and interact with approved websites |
| `database_admin` | Database driver + policy layer | Manage schemas and perform authorized database operations |
| `workflow_executor` | Application code / graph runtime | Execute approved multi-step plans and track state |
| `conversation_manager` | Token counter + summarizer + state store | Manage conversation history and context budgets |
| `fallback` | Router policy | Recover from timeouts, unavailable models, and invalid outputs |
| `robotics` | `qwen2.5-coder:7b` or `qwen3-coder:latest` + robotics tools | Assist with robotics code, control logic, and diagnostics |
| `math` | Python / numeric libraries | Perform exact calculations deterministically |

## Routing Policy

1. Classify the request with `classifier`.
2. For a simple request, route directly to `quick_response` or `chat`.
3. For a complex multi-step request, ask `planner` to produce a bounded plan and select the required agents.
4. Route code tasks to `coding_quick` or `coding_deep`. Use `debugger`, `test_generation`, and `code_review` when appropriate.
5. Route image inputs to `vision` or `ocr_document`; route audio and video to their dedicated processing pipelines.
6. Route knowledge questions through `rag_retrieval` and/or `research` when evidence is needed.
7. Validate structured outputs and tool arguments before using them. Require human confirmation for destructive or high-impact actions.
8. Use `critic` selectively for ambiguous, high-impact, or complex outputs—not on every trivial response.
9. If a route fails, use `fallback` to retry safely or choose an approved alternative. Avoid infinite retries.

## Runtime and Resource Guidelines

- Model file size is different from runtime RAM/VRAM use. Context length, KV cache, image resolution, quantization, and concurrency affect memory consumption.
- On a GPU with about 5.6 GB VRAM, do not assume 7B–9B models will fit entirely in VRAM. Use CPU offloading, suitable quantization, shorter contexts, or sequential execution as needed.
- Large models such as the 18 GB coder and 17 GB candidate may require substantial system RAM and/or CPU offloading.
- Configure timeouts, concurrency limits, cancellation, and per-agent context budgets.
- Embedding models generate vector representations; they are not general-purpose response generators. Keep the embedding model and index configuration consistent.
- Confirm model tags and capabilities using `ollama show <model>` and test each agent on representative tasks before production use.

## Output Contract

Each agent should return a predictable envelope where practical.

```json
{
  "agent": "chat",
  "status": "success",
  "content": "Agent response",
  "artifacts": [],
  "tool_calls": [],
  "errors": [],
  "metadata": {
    "model": "qwen3:8b",
    "duration_ms": 0
  }
}
```

Use the actual runtime's schema and serialization rules. Do not trust model-generated status, tool execution claims, or metadata without verifying them in application code.

## Implementation Checklist

- [ ] Define typed input/output schemas for each agent.
- [ ] Keep model names and backend settings in configuration, not scattered through code.
- [ ] Implement routing rules and explicit allowlists for tool access.
- [ ] Add timeout, cancellation, retry, and fallback behavior.
- [ ] Validate tool arguments and outputs before execution.
- [ ] Log latency, errors, model tags, token counts, and tool outcomes without logging secrets.
- [ ] Benchmark quality, latency, memory use, and tool-call reliability on representative tasks.
- [ ] Add regression tests for routing and critical workflows.
- [ ] Verify uncertain/custom model tags, especially `qwen3.8:27b` and `tev1:4b`, before assigning them production roles.
