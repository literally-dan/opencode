# Final Review Race Plan

Plan for closing the final reliability and ownership gaps before adopting the
28-commit `final-history` branch.

## Goal

Preserve durable Task completion and immutable Task ownership across provider
disconnects, cancellation, session removal, retries, and compaction. Keep the
existing public wire contracts and the 28-commit history.

## Constraints

- Keep prompt admission durable and separate from model continuation.
- Trust a Task ancestry edge only when `taskParentID === parentID` and both
  sessions have the same project, workspace, directory, and compatible path.
- Keep Task execution and Session runner ownership process-local. Clustered
  removal and recovery are excluded.
- Do not let completion callbacks block on a full provider turn.
- Run every lease and scope finalizer exactly once, including cancellation and
  foreground completion claims.
- Preserve classic `opencode run`, public Session creation, metadata stamping,
  model selection, `/target`, `/ask`, and `PermissionConfig.list` behavior.
- Fold changes into their existing owning commits and keep 28 commits.

## Workstreams

### 1. Task Provenance Consumers

- Update app permission auto-response and request trees, TUI request trees, and
  mini-run descendant discovery to accept only authenticated Task edges with a
  matching location.
- Keep ordinary `parentID` relationships visible as Session history, but never
  use them for Task permission inheritance or routing.
- Clear imported `taskParentID` before inserting a new Session. Do not change an
  existing Session owner during conflict handling.
- Add parent-only, valid Task, cross-location, and imported-owner regressions.

### 2. Runner And Session Removal Admission

- Keep Runner finalization serialized. Do not expose `Idle` until `onIdle`
  finishes. A wake during finalization must queue or join work, not start work
  that a stale idle callback can remove.
- Add a process-local Session lifecycle admission gate. Mark a Session as
  removing before cancellation and keep the tombstone until Instance disposal.
- Serialize checkpoint creation, prompt persistence, runner/shell enqueue, and
  Task job admission against this gate.
- Validate both the parent checkpoint and child lifecycle when admitting a Task
  job. After removal, reject new process-local runner, checkpoint, and job state.
- Add blocked-`onIdle`, HTTP prompt/removal, post-removal checkpoint, and removed
  child Task admission regressions.

### 3. BackgroundJob Finalization And Task Delivery

- Make normal and cancelled generation finalization uninterruptible only for
  the bounded state/scope cleanup. Keep waits on an older generation
  interruptible.
- Keep a generation non-replaceable until callback ownership, scope closure,
  terminal publication, and `done` completion finish.
- Add BackgroundJob-owned generation and extension finalizers. An extension
  finalizer must cover both the serialized tail wait and its work.
- Split Task notification into durable prompt admission and provider
  continuation. Root completion awaits admission, then forks continuation.
- For nested routing, transfer an idempotent release owner to the accepted
  extension and wait only for its admission acknowledgement. Cancellation must
  resolve the acknowledgement and release every retained checkpoint once.
- Add cancel/restart, interrupted finalization, interruptible waiter, root and
  nested admission ordering, queued cancellation, and foreground claim tests.

### 4. Retry Attempt Accounting

- Track retry-attempt `step-finish` and patch part IDs with the other attempt
  parts.
- Snapshot assistant finish, cost, and token accounting at attempt start and
  restore it before a retry. Reverse any Session aggregate usage applied by the
  discarded attempt through the existing Session update/event boundary.
- Keep billed failed-attempt accounting separate if the provider contract
  requires it; do not preserve orphan terminal parts.
- Add a retry-after-`step-finish` regression with nonzero usage and verify one
  step boundary, no stale patch, and exact assistant/Session totals.

### 5. Superseded And Native Compaction

- Run the full superseded read-modify-write operation inside the existing
  per-Session compaction lock, including the initial message read.
- Await superseded collapse before prompt completion. Fork optional native
  pruning only after collapse releases the lock.
- Add deterministic lock-overlap and next-turn visibility regressions.

## Integration Order

1. Fix Runner finalization and add the Session lifecycle gate.
2. Fix BackgroundJob cleanup/finalizers and split notification admission from
   continuation.
3. Update Task routing, lease transfer, and all provenance consumers.
4. Fix retry accounting and compaction serialization.
5. Run focused tests after each workstream, then package typechecks and broad
   regression suites.
6. Regenerate public clients only if a public Protocol or `HttpApi` changes.
7. Create targeted fixup commits, autosquash, and confirm a clean 28-commit
   branch with the same tested tree.
8. Request a second independent completion review before moving local `dev`.

## Acceptance Checks

- A provider disconnect during a running tool still continues with the exact
  tool result and no duplicate provider/tool call.
- A blocked idle callback cannot remove or overlap newly woken work.
- Removal wins atomically against prompt, runner, checkpoint, and Task job
  admission.
- A Task job cannot become terminal before its completion prompt is durably
  admitted when a valid route exists.
- Same-ID restart cannot overlap prior generation cleanup, and waiting starts
  remain interruptible.
- All Task checkpoint leases release exactly once on success, cancellation,
  queued cancellation, and foreground claim.
- Parent-only and cross-location Sessions receive no Task permission trust.
- Imported Session data cannot create a Task ownership edge.
- Retries leave one terminal step boundary with exact usage totals.
- Superseded and native/manual compaction metadata cannot overwrite each other,
  and the next turn observes completed collapse.
- Required typechecks, focused tests, generators, and independent review pass.

## Excluded Work

- Cluster-wide or durable Session-removal tombstones.
- Post-crash retry of in-flight provider work.
- Changes to classic run ownership lookup or public Session creation.
- Unrelated EventManifest expectation drift and machine-contention test timing.
