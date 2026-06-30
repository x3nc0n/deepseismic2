# Squad Decisions

## Active Decisions

# Decision: Atomic Thread-History Commit in FoundryAgent.chat() — Issue #25

**Date:** 2026-06-29  
**Author:** Lambert (AI Integration Specialist)  
**PR:** #27  
**Status:** Merged to `squad/25-chat-wedge-tool-calls`, pending review

---

## Context

`FoundryAgent.chat()` is a streaming generator that drives the Gradio and Streamlit UIs. It maintains a persistent per-thread message history (`self._threads[thread_id]`) shared across all requests on a container (the UI agent is a process-wide singleton).

The generator dispatches AOAI tool calls inside a `for tool_call in tool_calls` loop that contains `yield` statements (tool-trace markers emitted to the UI). The Gradio app has a 25s wall-clock guard that `break`s the outer `for chunk in agent.chat(...)` loop when the timeout fires.

---

## Problem

### Bug B (p0 — AOAI 400 wedge)

Prior code appended the assistant message (with `tool_calls`) to `history` **before** the matching tool result messages. When the 25s guard fired mid-round, `GeneratorExit` was raised at one of the `yield` statements inside the tool loop. Python's generator protocol guarantees `GeneratorExit` propagates immediately — the code after the `yield` (including the `history.append(tool_result)` calls) never ran.

Result: persistent `history` contained an assistant message with `tool_calls` but zero matching `tool` responses. AOAI validates this contract on every call. Every subsequent request from **any** user on that container replayed the corrupt history → HTTP 400 indefinitely. Only a container restart cleared the state.

### Bug A (UX — truncation mid-tool-round)

The same `break` also cut off tool-trace output to the user mid-round, showing a partial sequence in the chat panel.

---

## Decision

### Architectural principle (generalizes from this fix)

> **Agent thread-state must be committed atomically.** The assistant `tool_calls` message and all matching tool-result messages must be appended to thread history in a single atomic write. Writing the assistant entry first creates a window where any interruption (`GeneratorExit`, timeout, exception) produces permanently corrupt thread state.

### Implementation — three layers

**Layer 1 — Atomic `round_buffer`** (primary fix)  
Stage the assistant message and all tool results for the current round in a local `round_buffer: list[dict]`. Call `history.extend(round_buffer)` exactly once, after the round is complete. Persistent history is never visible in a partial state.

**Layer 2 — `try/finally` seal** (safety net)  
Wrap the tool-dispatch loop in `try/finally`. In the `finally` block, synthesize `{"error": "interrupted"}` tool responses for every `tool_call_id` in `round_buffer[0]["tool_calls"]` that is not in `answered_ids`. Then commit. This guarantees the committed history is AOAI contract-valid even if `GeneratorExit` fires at any yield inside the loop.

**Layer 3 — `_seal_dangling_tool_calls()` on entry** (self-heal)  
At the top of every `chat()` call, scan history for the last assistant message. If it has `tool_calls` with missing tool responses, append synthetic `{"error": "interrupted"}` tool messages. Self-heals any pre-existing corrupt state from before this fix was deployed.

### Bug A fix

Track `in_tool_round` using the `\n> 🔧` chunk prefix that tool-trace yields emit. Set `timed_out = True` when the clock fires; only `break` when `not in_tool_round`. This avoids cutting mid-tool-round. History integrity is guaranteed by Bug B's fix regardless of when the break occurs.

---

## Key Implementation Notes

- `_get_history(thread_id)` returns the **mutable list** stored in `self._threads` via `dict.setdefault`. `history.extend(round_buffer)` mutates in-place — rebinding the local `history` variable would silently lose the reference to the stored list.
- Streaming behavior (text chunks, tool-trace yields) is fully preserved — `yield` still happens inside `try`; only `history.extend` is deferred to `finally`.
- The 16-iteration tool-call cap is unchanged.

---

## Affected Files

| File | Change |
|---|---|
| `src/deepseismic/agent/agent.py` | Refactored `FoundryAgent.chat()` (round_buffer, try/finally, _seal_dangling_tool_calls) |
| `src/deepseismic/ui/gradio_app.py` | Round-boundary-aware 25s timeout guard |
| `src/tests/test_agent_atomic_commit.py` | 6 new CI-safe tests for the atomic-commit contract |

---

## Follow-up

Migrate `FoundryAgent.chat()` to `stream=True` (AOAI chunked SSE). This would:
- Eliminate the 25s truncation risk entirely (tokens stream to the user as they arrive)
- Remove the need for the `round_buffer` accumulation pattern
- Allow true incremental rendering in the Gradio / Streamlit UIs

---

# Decision: Catalog Index + Pending Manifest for HNS-Safe Prefix Resolution

**Date:** 2026-06-29T18:30:00-05:00  
**Author:** Parker (squad:parker)  
**Issue:** #26 — "Run lookup by short id-prefix 404s: _resolve_run_id catalog list-scan fails on ADLS/HNS"  
**PR:** https://github.com/x3nc0n/deepseismic2/pull/28  
**Status:** Implemented

---

## Context

`_resolve_run_id()` in `src/deepseismic/api/routes/interpretation.py` resolves a
short run-id prefix (e.g. `abcd1234`) to the full UUID needed for blob lookups.  
Step 3 (the only prefix→full path) called `list_blobs('catalog', 'interpretation/')`
to enumerate blobs and find matches.

On ADLS Gen2 / hierarchical-namespace `catalog` containers, `ContainerClient.list_blobs()`
returns nothing or raises. A bare `except Exception: pass` silently swallowed the
failure. Result: every short-prefix lookup 404-ed even for runs that persisted
correctly. Full UUIDs worked fine (exact `download_blob` — unaffected by HNS).

---

## Decision

**Do not change the storage tier or switch to DataLake FileSystemClient.** Fix is
app-side only — the run is intact and the exact-download path works.

### 1. Catalog index.json
Maintain `catalog/interpretation/index.json` — a JSON list of all full run ids.

- Appended at job submit time via `_catalog_index_append()` (read-modify-write;
  tolerates a missing index; write failure is logged but never blocks submission).
- `_resolve_run_id` step 3 reads the index via `download_blob` (exact, HNS-safe)
  before attempting `list_blobs`, so prefix resolution does not depend on blob
  enumeration on the hot path.

### 2. Pending manifest at submit time
Write `catalog/interpretation/{run_id}/status.json` with `status: pending` in the
`run_fault_detection` route handler **before** `background_tasks.add_task` fires.

- Makes the full run id resolvable via step 2 (exact download) cross-replica
  immediately after submission — not only after inference completes.
- Combined with the index, this means prefix lookups work cross-replica from the
  moment the POST /fault-detection response is returned.

### 3. Logged WARNING on list_blobs failure
Replaced the bare `except Exception: pass` with a `logger.warning(..., exc_info=True)`
so HNS listing failures are visible in production logs. `list_blobs` is retained as
a fallback for pre-index runs (only attempted when index scan yields no matches).

---

## Rationale

- `download_blob` (exact path) works on HNS — used for both step 2 (status.json)
  and step 3a (index.json). No DataLake SDK, no infra changes, no extra dependency.
- Index is append-only, tolerant of concurrent writes (PoC scale — single replica).
  At higher scale a separate indexing service or Cosmos/Table Storage would be
  appropriate, but for the current PoC the blob-based index is sufficient.
- Pending manifest provides defense-in-depth: even if the index write fails,
  the full UUID resolves immediately via step 2.

---

## Files Changed

| File | Change |
|---|---|
| `src/deepseismic/api/routes/interpretation.py` | `_CATALOG_INDEX_BLOB`, `_catalog_index_append()`, updated `_resolve_run_id()` step 3, pending manifest in `run_fault_detection()` |
| `src/tests/test_api/test_resolve_run_id.py` | 12 new focused tests |

---

## Follow-ups

- **Lambert:** Surface / echo the full run id in the UI panel and chat agent
  response so users can bookmark or copy the full UUID (out of scope for this PR).
- **At scale:** Replace the blob-based index with Azure Table Storage or Cosmos DB
  (atomic appends, consistent reads) if multi-replica contention becomes a concern.

# Triage Decision — Issues #25 and #26

**Date:** 2026-06-29T17:46:41-05:00  
**Author:** Ripley (Lead)

---

## Issue #25 — Chat wedges after truncated tool turn (AOAI 400)

**Title:** "Chat wedges after a truncated tool turn: dangling tool_calls corrupts shared thread (AOAI 400) + 25s truncation"

| Field | Value |
|-------|-------|
| Owner | squad:lambert |
| Type | type:bug |
| Priority | priority:p0 |
| Release | release:v0.4.0 |

### Root cause summary

Two bugs, one blocker:

**Bug B (blocker):** `FoundryAgent.chat()` (`src/deepseismic/agent/agent.py` ~L402/437) appends the assistant message WITH `tool_calls` to the persistent thread history before the matching tool-result messages are committed. When the UI's 25s guard (`gradio_app.py` L318-323) fires `GeneratorExit`, the thread is left with an unmatched `tool_calls` entry. Because the thread is reused per session and the UI agent is a process-wide singleton, every subsequent request from any user on the container replays the corrupt history → AOAI 400. Container restart is the only recovery path.

**Bug A (contributing):** The 25s wall-clock guard abandons the generator mid-round because the agent makes blocking (non-streaming) completion calls. Fix requires real streaming and/or turn-boundary-aware truncation.

### Ownership rationale

Thread-state management and streaming completion are LLM integration code owned by Lambert. Bug A's truncation guard is UI-side (Parker territory) but the real fix — streaming — lives in the agent. Lambert leads; Lambert/Parker coordinate on the UI guard cleanup.

### Priority rationale

p0: the bug permanently wedges the hosted demo for all users until an operator manually restarts the container. No user-facing workaround exists.

---

## Issue #26 — Run lookup by short id-prefix 404s on ADLS/HNS

**Title:** "Run lookup by short id-prefix 404s: _resolve_run_id catalog list-scan fails on ADLS/HNS (full UUID works)"

| Field | Value |
|-------|-------|
| Owner | squad:parker |
| Type | type:bug |
| Priority | priority:p1 |
| Release | release:v0.5.0 |

### Root cause summary

`_resolve_run_id()` (`src/deepseismic/api/routes/interpretation.py` L48-105) uses `ContainerClient.list_blobs(name_starts_with=...)` for prefix resolution against the ADLS Gen2 / hierarchical-namespace `catalog` container. This API returns nothing (or raises) on HNS containers where flat-blob enumeration is not available. A bare `except Exception: pass` at L84 silently swallows the failure, making the scan appear to return zero matches rather than surfacing an error. The caller sees a 404 even though the run persisted correctly.

Exact `download_blob` (used when a full UUID is supplied) works correctly.

### Ownership rationale

Pure backend/API + Azure storage-client bug. No ML or LLM surface. Parker owns.

### Priority rationale

p1: a clean workaround exists (supply the full UUID). No data loss. The run is intact; only the short-prefix UX is broken. Independent of #25 — no shared code surface.

### Suggested fix path

1. Replace `list_blobs` with `DataLake FileSystemClient.get_paths(path="catalog/interpretation/", recursive=False)` — the same client the ADLS browser uses, OR  
2. Write a `catalog/interpretation/index.json` manifest atomically at submit time; prefix scan reads the index instead of enumerating blobs.

Either way: replace the bare `except Exception: pass` at L84 with a logged error so future failures surface diagnostically.

---

## Sequencing

**#25 must land before #26.** #25 is a p0 that blocks all users; #26 is a p1 with a workaround. Both are independent bugs with no shared code surface.

---

## Architectural note (general, team-wide)

**Agent thread-state must be committed atomically.** The assistant `tool_calls` message and all matching tool-result messages must be appended to thread history in a single atomic write. Writing the assistant entry first creates a window where any interruption (`GeneratorExit`, timeout, exception) produces permanently corrupt thread state. This principle applies to any component that reuses a persistent conversation thread across requests.

---

