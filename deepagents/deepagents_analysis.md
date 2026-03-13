# deepagents Analysis — What's Included (Code Deep Dive)

_Author: lilCel (AI Assistant)_
_Date: March 10, 2026_
_Source: https://github.com/langchain-ai/deepagents_

---

## Overview

**deepagents** is LangChain's open-source clone of Claude Code. The README explicitly acknowledges: _"This project was primarily inspired by Claude Code, and initially was largely an attempt to see what made Claude Code general purpose."_

It's an **agent harness** — an opinionated, ready-to-run agent with batteries included. Instead of wiring up prompts, tools, and context management yourself, you get a working agent immediately.

**Key Differentiator from Claude Code:**
- **Claude Code:** Anthropic-only (Claude models)
- **deepagents:** Provider-agnostic (OpenAI, Gemini, Anthropic, Qwen, Kilo, local models)

---

## Architecture Summary

```
deepagents = LangGraph (runtime) + LangChain (agent framework) + Middleware Stack
```

The core function is `create_deep_agent()` which returns a **compiled LangGraph StateGraph**.

---

## "What's Included" — Code-Level Analysis

### 1. Planning (`write_todos`)

**Implementation:** `TodoListMiddleware` from `langchain.agents.middleware`

```python
from langchain.agents.middleware import TodoListMiddleware

deepagent_middleware = [
    TodoListMiddleware(),  # <-- First in the middleware stack
    # ... other middleware
]
```

**Capabilities:**
- Track multiple tasks with statuses: `'pending'`, `'in_progress'`, `'completed'`
- Persisted in LangGraph state (survives across turns)
- Helps agent organize complex multi-step work
- Accessible via `write_todos` tool

**How it works:**
1. Agent calls `write_todos(todos=[...], completed=[...])` to update task list
2. Middleware persists todo state in the graph state
3. On each turn, current todos are injected into context
4. Agent can mark tasks complete as it works

**Code location:** `langchain/agents/middleware/todo_list.py` (LangChain core)

---

### 2. Filesystem (`read_file`, `write_file`, `edit_file`, `ls`, `glob`, `grep`)

**Implementation:** `FilesystemMiddleware` from `deepagents.middleware.filesystem`

```python
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.backends import StateBackend

deepagent_middleware = [
    TodoListMiddleware(),
    FilesystemMiddleware(backend=backend),  # <-- Pluggable backend
    # ...
]
```

**Tools exposed:**
| Tool | Description |
|------|-------------|
| `ls` | List files with metadata (size, modified time) |
| `read_file` | Read file contents with offset/limit pagination; supports images as multimodal blocks |
| `write_file` | Create new files |
| `edit_file` | Exact string replacements (like Claude's Edit tool) |
| `glob` | Find files matching patterns (`**/*.py`) |
| `grep` | Search file contents with context |

**Pluggable Backends:**
1. **StateBackend** (default) — Ephemeral, stored in LangGraph state (single thread only)
2. **FilesystemBackend** — Local disk access
3. **StoreBackend** — Durable store (Postgres, Redis, etc.)
4. **SandboxBackend** — Remote sandbox (Daytona, Modal, Runloop)
5. **CompositeBackend** — Route different paths to different backends

**Backend selection code:**
```python
backend = backend if backend is not None else StateBackend
```

**Image support:** `read_file` natively handles `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp` — returns multimodal content blocks.

---

### 3. Shell Access (`execute`)

**Implementation:** Only available with sandbox backends that implement `SandboxBackendProtocol`

```python
# From the docs:
# For non-sandbox backends, the `execute` tool will return an error message.
```

**Sandbox partners:**
- **Daytona** — `langchain-daytona`
- **Modal** — `langchain-modal`
- **Runloop** — `langchain-runloop`

**Security model:** "Trust the LLM" — enforce boundaries at the tool/sandbox level, not by expecting the model to self-police.

---

### 4. Sub-agents (`task` tool)

**Implementation:** `SubAgentMiddleware` from `deepagents.middleware.subagents`

```python
from deepagents.middleware.subagents import (
    GENERAL_PURPOSE_SUBAGENT,
    SubAgentMiddleware,
)

# General-purpose subagent is always included
general_purpose_spec = {
    **GENERAL_PURPOSE_SUBAGENT,
    "model": model,
    "tools": tools or [],
    "middleware": gp_middleware,  # Same middleware stack as main agent
}

# User can add custom specialized subagents
all_subagents = [general_purpose_spec, *processed_subagents]

deepagent_middleware.append(
    SubAgentMiddleware(
        backend=backend,
        subagents=all_subagents,
    )
)
```

**Why sub-agents:**
- **Context isolation** — Subagent's work doesn't clutter main agent's context
- **Parallel execution** — Multiple subagents can run concurrently
- **Specialization** — Subagents can have different tools/configurations
- **Token efficiency** — Large subtask context is compressed into a single result

**How it works:**
1. Main agent calls `task(name="general-purpose", task="Do X")`
2. Fresh agent instance spawned with its own context
3. Subagent executes autonomously until completion
4. Returns single final report to main agent
5. Subagent is stateless (single-shot, can't send multiple messages)

**Custom subagent definition:**
```python
agent = create_deep_agent(
    subagents=[
        {
            "name": "code-reviewer",
            "description": "Reviews code changes for quality",
            "system_prompt": "You are an expert code reviewer...",
            "tools": [my_lint_tool],
        }
    ]
)
```

---

### 5. Smart Defaults (System Prompts)

**Implementation:** `BASE_AGENT_PROMPT` constant in `graph.py`

```python
BASE_AGENT_PROMPT = """You are a Deep Agent, an AI assistant that helps users accomplish tasks using tools...

## Core Behavior
- Be concise and direct. Don't over-explain unless asked.
- NEVER add unnecessary preamble ("Sure!", "Great question!", "I'll now...").
- Don't say "I'll now do X" — just do it.

## Professional Objectivity
- Prioritize accuracy over validating the user's beliefs
- Disagree respectfully when the user is incorrect
...

## Doing Tasks
1. **Understand first** — read relevant files, check existing patterns
2. **Act** — implement the solution
3. **Verify** — check your work against what was asked

## Progress Updates
For longer tasks, provide brief progress updates at reasonable intervals
"""
```

**Prompt composition:**
```python
# User's custom prompt is PREPENDED, then base prompt appended
if system_prompt is None:
    final_system_prompt = BASE_AGENT_PROMPT
else:
    final_system_prompt = system_prompt + "\n\n" + BASE_AGENT_PROMPT
```

**Skills system:** `SkillsMiddleware` loads custom instructions from skill files
```python
if skills is not None:
    deepagent_middleware.append(SkillsMiddleware(backend=backend, sources=skills))
```

**Memory system:** `MemoryMiddleware` loads AGENTS.md files for persistent context
```python
if memory is not None:
    deepagent_middleware.append(MemoryMiddleware(backend=backend, sources=memory))
```

---

### 6. Context Management (Auto-Summarization)

**Implementation:** `create_summarization_middleware` from `deepagents.middleware.summarization`

```python
from deepagents.middleware.summarization import create_summarization_middleware

deepagent_middleware = [
    # ...
    create_summarization_middleware(model, backend),  # <-- Context management
    # ...
]
```

**How it works:**
1. Monitors conversation length/token count
2. When context gets too long → auto-summarizes older turns
3. Large tool outputs saved to virtual files instead of staying in context
4. Allows long-running tasks without context overflow

**Strategies:**
- Pagination hints in prompts: _"Start with `read_file(path, limit=100)` to scan structure"_
- Offset/limit for large files
- _"Only read full files when necessary for editing"_

---

## Full Middleware Stack (Order Matters!)

```python
deepagent_middleware = [
    TodoListMiddleware(),                    # 1. Planning
    MemoryMiddleware(...),                   # 2. Memory (if configured)
    SkillsMiddleware(...),                   # 3. Skills (if configured)
    FilesystemMiddleware(backend=backend),   # 4. File tools
    SubAgentMiddleware(...),                 # 5. Sub-agent spawning
    create_summarization_middleware(...),    # 6. Context management
    AnthropicPromptCachingMiddleware(...),   # 7. Prompt caching (Anthropic only)
    PatchToolCallsMiddleware(),              # 8. Tool call normalization
    HumanInTheLoopMiddleware(...),           # 9. Human approval (if configured)
]
```

---

## Orchestrator Assessment: How to Make lilCel Better

### Scenario 1: lilCel as OpenClaw bot = Claude Code / Deep Agent

**Current OpenClaw capabilities vs deepagents:**

| Capability | deepagents | OpenClaw/lilCel | Gap |
|------------|------------|-----------------|-----|
| **Planning (write_todos)** | ✅ TodoListMiddleware | ❌ Manual notes/memory | Add task tracking tool |
| **Filesystem** | ✅ read/write/edit/ls/glob/grep | ✅ exec + Read/Write/Edit | ✅ Equivalent |
| **Shell access** | ✅ execute (sandbox) | ✅ exec tool | ✅ Equivalent |
| **Sub-agents** | ✅ task tool (isolated context) | ⚠️ sessions_spawn (partial) | Configure ACP properly |
| **Smart prompts** | ✅ BASE_AGENT_PROMPT | ✅ SOUL.md + AGENTS.md | ✅ Equivalent |
| **Context management** | ✅ Auto-summarization | ⚠️ Compaction only | Add auto-summarize middleware |
| **Skills system** | ✅ SkillsMiddleware | ✅ SKILL.md pattern | ✅ Equivalent |
| **Memory** | ✅ MemoryMiddleware | ✅ MEMORY.md + memory/*.md | ✅ Equivalent |

**Verdict:** lilCel is **~85% equivalent** to deepagents. Missing pieces:
1. Formal `write_todos` task tracking (we use notes instead)
2. Auto-summarization (we rely on compaction)
3. Proper ACP sub-agent configuration

**Recommended additions for parity:**
1. Add a `write_todos` tool that persists to a file (e.g., `workspace/TODOS.md`)
2. Implement context-aware summarization before compaction triggers
3. Configure `acp.defaultAgent` to enable Claude Code-style sub-agents

---

### Scenario 2: lilCel spawns sub-agents = Claude Code / Deep Agents bots

**Current state:**
```python
# OpenClaw config
sessions_spawn(runtime="acp", agentId=..., task=..., mode="run")
```

**Problem:** ACP not configured — requires `agentId` or `acp.defaultAgent`.

**Solution — Build lilCel sub-agent skill:**

```yaml
# Proposed: /root/.openclaw/skills/spawn-subagent/SKILL.md
name: spawn-subagent
description: >
  Spawn isolated sub-agent for context-heavy tasks.
  Mimics deepagents `task` tool behavior.

workflow:
  1. Receive task description + optional tools/model override
  2. Call sessions_spawn(runtime="subagent", task=task, mode="run")
  3. Wait for completion (push-based notification)
  4. Receive compressed result
  5. Return summary to main session
```

**Key insight from deepagents:**
- Subagents are **ephemeral** — fresh context per task
- Subagents return **single final report** — compressed result
- Main agent doesn't see subagent's full conversation — just the output

**Implementation path:**
1. Add `acp` config section with `defaultAgent` or per-task agent definitions
2. Create skill that wraps `sessions_spawn` with proper result handling
3. Add orchestrator prompts teaching when to spawn vs handle directly

---

## Key Takeaways

1. **deepagents is LangGraph-native** — use checkpointers, streaming, Studio
2. **Middleware pattern** — each capability is a middleware layer (composable)
3. **Provider-agnostic** — `init_chat_model("openai:gpt-5")` or any other
4. **Sub-agents are context isolation** — not parallel workers, but context windows
5. **Trust the LLM model** — enforce security at tool/sandbox level

---

## References

- **Source:** https://github.com/langchain-ai/deepagents
- **Docs:** https://docs.langchain.com/oss/python/deepagents/overview
- **Core file:** `libs/deepagents/deepagents/graph.py`
- **Base prompt:** `libs/deepagents/deepagents/base_prompt.md`
