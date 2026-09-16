# Subagent Job Control

## Goal

Make running nested subagents visible and cancellable from every owning Task ancestor, even after an intermediate Task completes. Remove foreground-promotion surfaces that cannot work now that every Task is asynchronous.

## Scope

- Add authenticated Task ancestry to process-local Task job metadata.
- Use that ancestry in `SessionRunState.cancel` and `task_control` list/stop operations.
- Keep a direct/all-status metadata traversal fallback for jobs created before this change.
- Remove the experimental session-background HTTP endpoint.
- Remove BackgroundJob promotion methods and callbacks that have no remaining caller.
- Remove foreground-only TUI and run-mode commands, shortcuts, and hints.
- Regenerate affected clients after the Server `HttpApi` change.

## Constraints

- Preserve process-local BackgroundJob execution and durable Session ownership boundaries.
- Validate ancestry through `taskParentID`, `parentID`, and matching Session location.
- Do not expose or cancel jobs from sibling or unrelated Sessions.
- Keep the persisted `SubtaskPart.background` field for existing Session data.
- Keep the `backgroundSubagents` capability because clients still support asynchronous subagents.
- Do not change completion notification behavior.

## Interfaces

- Task job metadata gains `ancestorSessionIds`, ordered from the direct parent toward the root Task owner.
- `task_control` still exposes `list` and `stop`; output and parameters do not change.
- `SessionRunState.cancel(sessionID)` still cancels that Session and its Task descendants.
- `POST /experimental/session/:sessionID/background` is removed.

## Acceptance Checks

- A root can list and stop a running grandchild after the intermediate Task job completes.
- Session abort cancels the same grandchild in that state.
- A child cannot list or stop a sibling Task.
- Existing direct-child cancellation still works.
- No foreground-promotion reference remains outside historical persisted data.
- Core, opencode, client generation, SDK generation, and TUI checks pass for changed areas.

## Excluded Work

- Durable recovery of process-local jobs after a process restart.
- Cluster-wide job placement or cancellation.
- Changes to Task completion notifications.
