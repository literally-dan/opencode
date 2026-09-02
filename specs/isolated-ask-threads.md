# Isolated Ask Threads

Status: proposed, implementation requires approval

## Goal

Turn `/ask` into a persistent, browsable side conversation with follow-ups and visible read-only tool activity. Ask threads must never enter the normal Session message flow or affect Session execution.

## Required Invariants

- A normal prompt never receives Ask questions, answers, tool calls, or tool results.
- Ask does not create a Session, child Session, Session message, part, input, status, summary, snapshot, compaction, continuation, share, export, or replay record.
- Ask does not update Session timestamps or ordering.
- Ask reads a snapshot of current compacted Session context but cannot mutate that context.
- Starting or continuing an Ask does not pause, interrupt, steer, queue behind, or otherwise affect normal Session execution.
- Each Ask thread belongs to one Session and uses that Session's project and location boundary.
- Deleting a Session deletes its Ask threads.

## User Experience

- `/ask <question>` starts a new Ask thread.
- The Ask dialog becomes a two-pane panel: browsable threads on the left and the selected conversation on the right.
- A composer in the panel adds a follow-up to the selected thread.
- Starting another `/ask` creates another thread instead of appending to the selected thread.
- The panel shows user questions, assistant answers, and expandable tool activity.
- Tool names and running/completed/error states update while the request runs. Final bounded tool output remains visible after reload.
- Ask-specific permission requests render inside the panel and use modal key bindings.
- Closing an active Ask aborts that exact HTTP request only. The embedded RPC bridge propagates the abort to provider and tool work. Normal Session execution continues.

## Persistence

Add separate Core tables. Do not reuse Session messages or hidden Sessions.

### `session_ask_thread`

- `id`: `ask_...` primary key
- `session_id`: Session foreign key with cascade delete
- `title`: bounded title derived from the first question
- `created_at`
- `updated_at`
- index on `(session_id, updated_at, id)`

### `session_ask_turn`

- `id`: `atn_...` primary key
- `thread_id`: Ask thread foreign key with cascade delete
- `question`
- `answer`
- `tool_activity`: bounded JSON array
- `created_at`
- index on `(thread_id, id)`

Insert a thread and its first turn atomically only after a successful answer. Insert a follow-up turn and update its thread atomically. Failed or cancelled requests leave no rows. No backfill is required because current asks are not persisted.

Tool activity contains call ID, source-owned tool name, status, bounded input, bounded text output, and error text. Reuse normal tool-output truncation. Do not persist attachments or unbounded output.

## Service Boundary

Add `packages/opencode/src/session/ask.ts` with a `SessionAsk` service:

- `ask(input)`: start a thread or append to an authenticated thread
- `threads(input)`: paginate threads for a Session
- `turns(input)`: paginate turns for an authenticated thread
- `cancel(input)`: interrupt active Ask work without cancelling Session execution

`SessionPrompt.ask` remains a compatibility facade. `SessionPrompt.cancel` delegates Ask cancellation to `SessionAsk` where current compatibility requires it.

`SessionAsk` must not call `SessionProcessor`, `SessionPrompt.loop`, `SessionStatus`, or Session message/part update functions.

Allow one active request per Ask thread. This exclusion is process-local until clustered Session execution has a durable ownership design. Different threads can run concurrently. Session-level cancellation can cancel active Ask requests for that Session, but Ask cancellation cannot cancel the normal Session runner.

## Model Context

For each turn:

1. Read the current compacted normal Session view without writing it.
2. Read recent completed turns from the selected Ask thread only.
3. Add an Ask-specific system instruction with this explicit meaning: "You are answering in an isolated Ask thread. Nothing in the main Session was paused or interrupted. You are seeing a read-only snapshot that can become stale while the main Session continues. Treat current Session state as authoritative. Do not claim that Ask text entered normal Session history or changed its execution."
4. Append prior Ask questions and answers as in-memory user/assistant messages.
5. Append the current question.
6. Run an Ask-local provider/tool loop.

Bound Ask history independently by completed-turn count and model context budget. Drop oldest Ask turns when needed. Do not summarize or compact Ask rows.

## Read-Only Tools

Add a source-owned Ask allowlist in `ToolRegistry`. Capture built-in tool definitions before plugin/custom/MCP registration so a plugin cannot replace an allowed ID.

Initial allowlist:

- `read`
- `glob`
- `grep`
- `webfetch`
- available read-only web search
- read-only LSP operations when their existing feature flag permits them

Exclude shell, edit, write, patch, Task, question, todo, execute/plan, compaction/history mutation, custom tools, plugin tools, and MCP tools. Disable plugin execution hooks for Ask tool calls so an allowed built-in cannot gain plugin-defined side effects.

Use the selected agent and model, but enforce the Ask allowlist after all agent tool settings. A small hard provider-turn limit stops tool-only loops.

Reject Ask requests for GitLab workflow models. Their cached language-model objects contain request-mutated Session and tool state, so concurrent Ask execution cannot satisfy the isolation invariant. Making that provider request-local is separate work.

## Permissions

Use the real Session ID and normal merged agent/Session permission rules. An `always` approval therefore remains shared with normal Session tools.

Do not attach a permission request to a nonexistent Session Part. Add Ask metadata with Ask thread, turn, tool, and input identifiers. The TUI permission component reads this metadata when no normal Part exists.

The Ask panel renders Ask-tagged pending permissions. Permission cleanup must publish the normal reply event so cancelled Ask requests do not leave stale UI state.

## Tool Activity Events

Publish transient Ask-specific tool activity events with Session, client request, thread, turn, call, tool, status, bounded input, and bounded result/error fields. These events use the live server event stream only; they do not enter EventV2 durable replay or Session history.

Each POST accepts an optional bounded `requestID`. The server generates one when absent, returns it in the response, and includes it in every live activity event. The TUI generates a fresh request ID before POST and accepts live events only for that exact ID. This prevents concurrent new Asks in one Session from claiming each other's pre-thread events. The server rejects concurrent duplicate request IDs within one Session. Request IDs are correlation metadata, not cancellation capabilities.

Persist the final bounded activity list with the completed Ask turn. A reopened panel therefore has final activity even though live events are transient.

Token streaming is excluded from the first implementation. The panel shows a working state, live tool activity, permission prompts, and the final answer.

## HTTP API

Keep:

- `POST /session/:sessionID/ask`

Extend its body with optional `threadID` and `requestID`. Return the completed turn with top-level `text` retained for compatibility, plus request ID, thread ID, turn ID, tool activity, and timestamps.

Add:

- `GET /session/:sessionID/ask` for paginated thread summaries
- `GET /session/:sessionID/ask/:threadID` for paginated turns
- `POST /session/:sessionID/ask/:threadID/cancel` for Ask-only cancellation

Authenticate every thread lookup against the path Session ID.

This remains on the legacy Server API. Session V2 Protocol/Server integration is excluded. Regenerate the legacy JavaScript SDK and OpenAPI document.

## TUI

- Replace the one-answer `DialogAsk` with one mounted Ask panel.
- Load thread summaries when it opens.
- Select the new thread after `/ask` completes.
- Support keyboard and mouse thread selection, scrolling, expandable tool results, follow-up submission, cancellation, retry after failure, and Ask-tagged permissions.
- Keep all Ask state outside the normal prompt queue and Session transcript renderer.
- Do not use `dialog.replace()` while a request is active because replacement invokes close/abort behavior.
- Preserve desktop/mobile-neutral API behavior; this plan changes only the current TUI client.

## Main Files

- `packages/core/src/session/sql.ts`
- `packages/core/src/database/migration/`
- `packages/core/schema.json`
- generated Core database files
- `packages/opencode/src/id/id.ts`
- new `packages/opencode/src/session/ask.ts`
- `packages/opencode/src/session/prompt.ts`
- `packages/opencode/src/tool/registry.ts`
- `packages/opencode/src/permission/index.ts`
- Session HTTP API group and handler
- `packages/tui/src/component/dialog-ask.tsx`
- `packages/tui/src/component/prompt/index.tsx`
- `packages/tui/src/routes/session/permission.tsx`
- generated legacy SDK and OpenAPI files

## Acceptance Checks

1. `/ask` creates a new browsable thread; panel input adds a follow-up to that thread.
2. A follow-up model request contains prior turns from only the selected Ask thread.
3. A later normal prompt contains no Ask question, answer, tool call, or result.
4. Ask creates no normal Session message, part, input, status, child, summary, snapshot, compaction, continuation, share, export, or durable replay record.
5. Starting Ask while the normal Session is busy does not interrupt, pause, steer, or queue its runner; the Ask model request receives the explicit snapshot instruction.
6. Threads survive process/instance reload, paginate deterministically, and cascade-delete with the Session.
7. Failed and cancelled asks create no thread or turn.
8. Allowed built-in read tools execute, visible activity updates, and tool results reach the next provider turn.
9. Mutating, Task, custom, plugin, and MCP tools are absent; a plugin named `read` cannot replace the built-in.
10. Permission allow/ask/always/deny/reject and cancellation use normal routing without stale modal state.
11. The provider-turn limit stops a tool-only loop.
12. Concurrent different threads persist independently; concurrent turns in one thread are rejected.
13. Reopening the panel shows prior threads, turns, and bounded tool activity.
14. Existing one-turn clients can still read the top-level `text` response.
15. GitLab workflow models return a clear unsupported-model error without starting or persisting Ask work.
16. Concurrent new Asks in one Session correlate live activity by client request ID before thread/turn IDs are known; exact concurrent duplicates are rejected.
17. Initial Ask cancellation propagates the HTTP abort through embedded RPC and cannot cancel a different Ask or normal Session execution.

## Verification

- Core migration/schema generation and database tests
- SessionAsk isolation, context, concurrency, failure, cancellation, tools, and permissions tests
- Session HTTP API and exercise tests
- ToolRegistry allowlist and plugin-replacement tests
- TUI Ask panel, permission, history, follow-up, and cancellation tests
- affected package type-checks
- legacy JavaScript SDK/OpenAPI generation
- local opencode build and smoke test
- final scoped implementation review

## Excluded Work

- Session V2 API/client integration
- Ask history in normal Session transcript, compaction, summaries, sharing, export, import, fork cloning, or cloud sync
- mutating tools, Task/subagents, custom/plugin/MCP tools
- image/editor attachments
- token streaming
- pending Ask recovery after process restart
- clustered or distributed Ask execution ownership
- GitLab workflow model support until its request state becomes request-local
- Ask history summarization
- per-turn deletion and retention policy
