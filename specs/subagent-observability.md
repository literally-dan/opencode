# Subagent Observability

## Goal

Give every Session a canonical, ownership-scoped view of its Task descendants so the model and clients can distinguish active, blocked, completed, undelivered, and detached subagents. Keep execution process-local and do not add automatic restart recovery.

## Current Problem

- Durable Session rows preserve `parentID` and authenticated `taskParentID` lineage.
- `BackgroundJob` and `SessionStatus` are process-local and disappear after restart.
- Missing `SessionStatus` reads as idle.
- `task_control list` only scans running process-local jobs.
- The TUI reconstructs a separate partial tree from Session and status snapshots.
- Task settlement and completion-notification delivery are not durable facts.

After a restart, a durable unfinished Task can therefore look idle and absent. While the process is alive, a completed child whose result is still being delivered can also look the same as one that returned successfully.

## Scope

- Add durable Task lifecycle state keyed by the child Session ID.
- Keep Session `parentID` plus `taskParentID` plus a matching `projectID` as the ownership source. Location is not compared. Notification targets and `task_id` resume still require the same location. A durable unfinished Task in another location lists as `unknown`, and `task_control stop` reports it as unavailable instead of cancelling it.
- Build one Task tree service that joins durable lifecycle state with process-local jobs, Session status, pending permissions and questions, and recent Session message or tool activity.
- Return execution state, phase, current tool, activity time, quiet duration, settlement time, delivery state, and children.
- Treat a durable unfinished Task with no local job as detached.
- Extend `task_control list` to return the canonical tree, including nested descendants and connector ancestors.
- Keep `task_control stop` descendant-only. Cancel local jobs and durably cancel detached unfinished Tasks.
- Add per-spawn ancestor access with separate transcript and status capabilities.
- Preserve existing Task history recovery by default for compatibility. A blind Task can explicitly disable ancestor access.
- Add focused service and tool tests for ownership, nesting, blockers, activity, delivery, restart detachment, cancellation, and visibility.

## Excluded Work

- Restarting or replaying detached provider work.
- Cluster-wide job discovery, leases, or remote cancellation.
- Declaring a Task frozen from elapsed time alone.
- Letting a child stop, prompt, or mutate an ancestor.
- Exposing sibling or unrelated Session state.
- Replacing existing Session or BackgroundJob execution coordinators.

## Durable State

Add a Task lifecycle table keyed by `session_id` with:

- ancestor access policy
- monotonically increasing lifecycle generation
- lifecycle status: running, completed, error, or cancelled
- created, updated, and completed timestamps
- terminal error text when present
- completion-notification target Session and delivery timestamp

Creating a fresh BackgroundJob generation increments the lifecycle generation, resets its start time, and clears prior settlement and delivery fields. Extending a running BackgroundJob keeps its current generation. Settlement and delivery updates require the matching generation so stale finalizers cannot overwrite resumed work. Settling the BackgroundJob records the terminal state before notification routing. Successfully admitting a completion notification records its target and delivery time.

Existing Task Sessions without a lifecycle row remain readable. Their state is inferred from live jobs and durable messages where possible and otherwise reported as unknown. Existing Sessions retain ancestor history access.

## Query Contract

Each Task node contains:

- `taskID`, `parentID`, `agent`, and `title`
- `execution`: running, completed, error, cancelled, detached, or unknown
- `phase`: starting, model, tool, retry, permission, question, delivering, idle, or unknown
- optional current tool name and title
- `startedAt`, `lastActivityAt`, `quietForMs`, and optional `completedAt`
- `delivery`: pending, delivered, or unknown, with an optional target and timestamp
- nested `children`

`quietForMs` is evidence only. Long-running tools can be healthy while quiet.

## Visibility

Default descendant access is unchanged:

- A Session can list and stop authenticated Task descendants.
- A Session cannot list or stop siblings or unrelated Tasks.
- Ancestor status is hidden unless the current Task was spawned with that read-only capability.
- Ancestor transcript recovery is independently controlled.
- A Task can delegate only ancestor capabilities that its own Session has. A blind Task cannot amplify access by spawning a child with `all`.

The Task spawn parameter supports `ancestor_access`:

- `history` is the compatibility default. It allows ancestor transcript recovery but not ancestor status.
- `none` is blind. It allows neither ancestor transcript nor ancestor status.
- `status` allows read-only ancestor status but no ancestor transcript.
- `all` allows both.

`task_control` supports descendant listing by default and ancestor listing only when the current Task has status access. Stop always remains descendant-only.

## Boundaries

- Schema and migration definitions remain in Core.
- The legacy OpenCode Session runtime owns Task lifecycle transitions and tree composition.
- `task_control` formats and exposes the service result but does not duplicate ownership logic.
- History tools ask the lifecycle service whether cross-Session ancestor reads are allowed.
- No public Server `HttpApi` change is required for the model-facing first slice. Existing clients continue to build their current Session UI from Session events.

## Acceptance Checks

- A parent lists direct and nested Task descendants with stable structured state.
- A completed intermediate Task remains in the tree when it connects an active grandchild.
- A child cannot see a sibling or unrelated Task.
- A default child cannot inspect ancestor status.
- A child with status access can inspect its ancestor chain but cannot stop it.
- A blind child cannot read ancestor transcript history.
- A process restart simulation reports an unfinished durable Task as detached.
- Pending permission and question requests identify the blocking phase.
- Retry and active tool phases are visible.
- Last activity and quiet duration advance from durable Session or tool state.
- Terminal execution is durable before result routing.
- Successful notification admission records delivery target and time.
- Failed or skipped notification routing remains pending.
- Stopping a detached Task records cancellation without claiming successful execution.
- Existing Task Sessions without lifecycle state remain safe and queryable as unknown or inferred state.
- Focused tests, package typechecks, migration checks, formatting, and generated-schema checks pass.
