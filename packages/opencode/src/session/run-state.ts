import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { InstanceRef } from "@/effect/instance-ref"
import { InstanceState } from "@/effect/instance-state"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { Runner } from "@/effect/runner"
import { BackgroundJob } from "@/background/job"
import { Deferred, Effect, Fiber, Latch, Layer, Scope, Context, Option, Schema, SynchronizedRef } from "effect"
import { Session } from "./session"
import { SessionID } from "./schema"
import { SessionStatus } from "./status"
import { EventV2Bridge } from "@/event-v2-bridge"

type Checkpoint = {
  generation: number
  leases: number
}

type Lifecycle = {
  removing: Set<SessionID>
  // Sessions whose turn a revert stopped. No turn starts for them until the revert finishes.
  stopped: Set<SessionID>
  // Sessions whose turn a turn-only interrupt stopped. Background Task results do not start turns for them until the
  // next prompt, a cancel, or a revert stop.
  paused: Set<SessionID>
}

type State = {
  runners: Map<SessionID, Runner.Runner<SessionV1.WithParts>>
  scope: Scope.Scope
  checkpoints: SynchronizedRef.SynchronizedRef<Map<SessionID, Checkpoint>>
  lifecycle: SynchronizedRef.SynchronizedRef<Lifecycle>
  generation: number
  // Waits of Sessions with a revert in progress or staged. A revert change completes and removes the wait.
  revertChanges: Map<SessionID, Deferred.Deferred<void>>
}

export type Admission = {
  sessionID: SessionID
  checkpoint?: number
}

export interface Interface {
  readonly assertNotBusy: (sessionID: SessionID) => Effect.Effect<void, Session.BusyError>
  readonly cancel: (sessionID: SessionID) => Effect.Effect<void>
  /**
   * Stops the running turn of a Session and the turns queued behind it, like `cancel`. Unlike `cancel`, background
   * Tasks keep running, and the checkpoint and leases they deliver with stay current. Prompts sent while the turn
   * stops are queued and run after it.
   * Background Task results of the Session then do not start turns until `resumeNotificationTurns`, `cancel`, or
   * `stopTurn`, so the model does not answer them right after the user stopped it.
   */
  readonly interruptTurn: (sessionID: SessionID) => Effect.Effect<void>
  /** Returns whether a turn-only interrupt keeps background Task results of the Session from starting turns. */
  readonly notificationTurnsPaused: (sessionID: SessionID) => Effect.Effect<boolean>
  /** Lets background Task results start turns of the Session again. Call it before a prompt starts its turn. */
  readonly resumeNotificationTurns: (sessionID: SessionID) => Effect.Effect<void>
  /**
   * Stops the running turn of a Session, and keeps other turns of the Session from starting until the scope closes.
   * Unlike `cancel`, background Tasks keep running, and the checkpoint and leases they deliver with stay current.
   * Fails with BusyError while another stop of the Session is open, because two reverts cannot restore one snapshot.
   * Only revert and unrevert use it. Other turn stops use `interruptTurn`, so later prompts are not blocked.
   */
  readonly stopTurn: (sessionID: SessionID) => Effect.Effect<void, Session.BusyError, Scope.Scope>
  /**
   * Returns a wait that ends at the next revert change of the Session when a revert of the Session is in progress
   * or `staged` reports a staged revert, and none otherwise. Both are read again after the wait is registered, so a
   * revert that finishes meanwhile cannot leave the wait open. A cancel or removal of the Session ends its waits, and
   * instance disposal ends every wait.
   */
  readonly awaitRevert: <E, R>(
    sessionID: SessionID,
    staged: Effect.Effect<boolean, E, R>,
  ) => Effect.Effect<Option.Option<Effect.Effect<void>>, E, R>
  /** Ends the revert waits of a Session. Call it after a staged revert is committed or cleared. */
  readonly revertChanged: (sessionID: SessionID) => Effect.Effect<void>
  /**
   * Cancels the background Tasks that the given messages of a Session started or resumed, and their descendants.
   * `taskIDs` names the Tasks that task calls in the messages started or resumed, because a job records only the
   * message that started it.
   * Their results are discarded, not delivered.
   */
  readonly cancelTasks: (
    sessionID: SessionID,
    messageIDs: ReadonlySet<string>,
    taskIDs?: ReadonlySet<string>,
  ) => Effect.Effect<void>
  readonly checkpoint: (sessionID: SessionID) => Effect.Effect<number>
  readonly isCurrent: (sessionID: SessionID, checkpoint: number) => Effect.Effect<boolean>
  readonly retain: (sessionID: SessionID, checkpoint: number) => Effect.Effect<Option.Option<Effect.Effect<void>>>
  readonly admit: <A, E, R>(
    requirements: readonly Admission[],
    effect: Effect.Effect<A, E, R>,
  ) => Effect.Effect<Option.Option<A>, E, R>
  readonly admitIfCurrent: <A, E, R>(
    sessionID: SessionID,
    checkpoint: number,
    effect: Effect.Effect<A, E, R>,
  ) => Effect.Effect<Option.Option<A>, E, R>
  readonly ensureRunning: (
    sessionID: SessionID,
    onInterrupt: Effect.Effect<SessionV1.WithParts>,
    work: Effect.Effect<SessionV1.WithParts>,
  ) => Effect.Effect<SessionV1.WithParts>
  readonly startShell: (
    sessionID: SessionID,
    onInterrupt: Effect.Effect<SessionV1.WithParts>,
    work: Effect.Effect<SessionV1.WithParts>,
    ready?: Latch.Latch,
  ) => Effect.Effect<SessionV1.WithParts, Session.BusyError>
}

export class Service extends Context.Service<Service, Interface>()("@opencode/SessionRunState") {}

const layer = Layer.effect(
  Service,
  Effect.gen(function* () {
    const background = yield* BackgroundJob.Service
    const status = yield* SessionStatus.Service
    const events = yield* EventV2Bridge.Service

    const state = yield* InstanceState.make<State>(
      Effect.fn("SessionRunState.state")(function* () {
        const scope = yield* Scope.Scope
        const runners = new Map<SessionID, Runner.Runner<SessionV1.WithParts>>()
        const checkpoints = yield* SynchronizedRef.make(new Map<SessionID, Checkpoint>())
        const lifecycle = yield* SynchronizedRef.make<Lifecycle>({
          removing: new Set(),
          stopped: new Set(),
          paused: new Set(),
        })
        const revertChanges = new Map<SessionID, Deferred.Deferred<void>>()
        yield* Effect.addFinalizer(
          Effect.fnUntraced(function* () {
            // End revert waits first, so work that waits on this instance ends even when a runner does not stop.
            yield* Effect.forEach(revertChanges.values(), (change) => Deferred.succeed(change, undefined), {
              discard: true,
            })
            revertChanges.clear()
            yield* Effect.forEach(runners.values(), (runner) => runner.cancel, {
              concurrency: "unbounded",
              discard: true,
            })
            runners.clear()
          }),
        )
        return { runners, scope, checkpoints, lifecycle, generation: 0, revertChanges }
      }),
    )

    // A Session is idle only when its runner is finished and no background Task lease is held.
    // The last lease release publishes idle from a separate fiber because releases can run
    // inside work that a lifecycle holder awaits.
    const publishIdleAfterRelease = Effect.fn("SessionRunState.publishIdleAfterRelease")(function* (
      data: State,
      sessionID: SessionID,
    ) {
      if (!(yield* InstanceRef)) return
      yield* SynchronizedRef.modifyEffect(
        data.lifecycle,
        Effect.fnUntraced(function* (admission) {
          const leases = SynchronizedRef.getUnsafe(data.checkpoints).get(sessionID)?.leases ?? 0
          if (data.runners.has(sessionID) || leases > 0) return [undefined, admission] as const
          if ((yield* status.get(sessionID)).type !== "idle") yield* status.set(sessionID, { type: "idle" })
          return [undefined, admission] as const
        }),
      ).pipe(Effect.forkIn(data.scope, { startImmediately: true }))
    })

    const runner = Effect.fn("SessionRunState.runner")(function* (
      sessionID: SessionID,
      onInterrupt: Effect.Effect<SessionV1.WithParts>,
    ) {
      return yield* InstanceState.useEffect(
        state,
        Effect.fnUntraced(function* (data) {
          const existing = data.runners.get(sessionID)
          if (existing) return existing
          const next = Runner.make<SessionV1.WithParts>(data.scope, {
            onIdle: SynchronizedRef.modifyEffect(
              data.lifecycle,
              Effect.fnUntraced(function* (admission) {
                const terminal =
                  next.state._tag === "Finalizing" ||
                  next.state._tag === "ShellFinalizing" ||
                  next.state._tag === "Cancelling"
                if (data.runners.get(sessionID) !== next || !terminal) return [undefined, admission] as const
                // Remove the runner in the same checkpoint update that lease releases observe.
                const settled = yield* SynchronizedRef.modify(data.checkpoints, (checkpoints) => {
                  data.runners.delete(sessionID)
                  const checkpoint = checkpoints.get(sessionID)
                  if (checkpoint && checkpoint.leases > 0) return [false, checkpoints] as const
                  if (!checkpoint) return [true, checkpoints] as const
                  const remaining = new Map(checkpoints)
                  remaining.delete(sessionID)
                  return [true, remaining] as const
                })
                if (settled) yield* status.set(sessionID, { type: "idle" })
                return [undefined, admission] as const
              }),
            ),
            onBusy: status.set(sessionID, { type: "busy" }),
            onInterrupt,
          })
          data.runners.set(sessionID, next)
          return next
        }),
      )
    })

    const assertNotBusy = Effect.fn("SessionRunState.assertNotBusy")(function* (sessionID: SessionID) {
      return yield* InstanceState.useEffect(
        state,
        Effect.fnUntraced(function* (data) {
          const existing = data.runners.get(sessionID)
          if (existing?.busy) yield* busyError(sessionID)
        }),
      )
    })

    const cancel = Effect.fn("SessionRunState.cancel")(function* (sessionID: SessionID) {
      return yield* InstanceState.useEffect(
        state,
        Effect.fnUntraced(function* (data) {
          yield* SynchronizedRef.modifyEffect(
            data.lifecycle,
            Effect.fnUntraced(function* (admission) {
              yield* SynchronizedRef.update(data.checkpoints, (checkpoints) => {
                if (!checkpoints.has(sessionID)) return checkpoints
                const next = new Map(checkpoints)
                next.delete(sessionID)
                return next
              })
              return [undefined, resume(admission, sessionID)] as const
            }),
          )
          // Task results held for a revert see the stale checkpoint when they wake, and drop.
          yield* publishRevertChange(data, sessionID)
          const cancelled = yield* cancelBackgroundJobs(background, sessionID)
          const cancellation = yield* SynchronizedRef.modifyEffect(
            data.lifecycle,
            Effect.fnUntraced(function* (admission) {
              const existing = data.runners.get(sessionID)
              if (existing) return [Option.some(yield* existing.enqueueCancel), admission] as const
              yield* status.set(sessionID, { type: "idle" })
              return [Option.none(), admission] as const
            }),
          )
          if (Option.isSome(cancellation)) yield* cancellation.value
          yield* stopTaskRunners(data, cancelled)
        }),
      )
    })

    const interruptTurn = Effect.fn("SessionRunState.interruptTurn")(function* (sessionID: SessionID) {
      return yield* InstanceState.useEffect(
        state,
        Effect.fnUntraced(function* (data) {
          const cancellation = yield* SynchronizedRef.modifyEffect(
            data.lifecycle,
            Effect.fnUntraced(function* (admission) {
              const paused = { ...admission, paused: new Set(admission.paused).add(sessionID) }
              const existing = data.runners.get(sessionID)
              if (existing) return [Option.some(yield* existing.enqueueCancel), paused] as const
              return [Option.none(), paused] as const
            }),
          )
          // Only the enqueued cancellation moves the runner out of Cancelling, so it must finish when the caller
          // is interrupted.
          if (Option.isSome(cancellation)) yield* Effect.uninterruptible(cancellation.value)
        }),
      )
    })

    const notificationTurnsPaused = Effect.fn("SessionRunState.notificationTurnsPaused")(function* (
      sessionID: SessionID,
    ) {
      return yield* InstanceState.useEffect(state, (data) =>
        SynchronizedRef.get(data.lifecycle).pipe(Effect.map((admission) => admission.paused.has(sessionID))),
      )
    })

    const resumeNotificationTurns = Effect.fn("SessionRunState.resumeNotificationTurns")(function* (
      sessionID: SessionID,
    ) {
      return yield* InstanceState.useEffect(state, (data) =>
        SynchronizedRef.update(data.lifecycle, (admission) => resume(admission, sessionID)),
      )
    })

    const stopTurn = Effect.fn("SessionRunState.stopTurn")(function* (sessionID: SessionID) {
      return yield* InstanceState.useEffect(
        state,
        Effect.fnUntraced(function* (data) {
          const cancellation = yield* Effect.acquireRelease(
            SynchronizedRef.modifyEffect(
              data.lifecycle,
              Effect.fnUntraced(function* (admission) {
                if (admission.stopped.has(sessionID)) return [Option.none(), admission] as const
                const existing = data.runners.get(sessionID)
                const stopped = new Set(admission.stopped).add(sessionID)
                return [
                  Option.some(existing ? yield* existing.enqueueCancel : Effect.void),
                  { ...resume(admission, sessionID), stopped },
                ] as const
              }),
            ).pipe(
              // Start the runner cancellation in the acquire step and run it in its own fiber, so a revert that
              // is interrupted while a turn does not stop closes its stop without leaving the runner half cancelled.
              Effect.flatMap((cancel) =>
                Option.isNone(cancel)
                  ? Effect.succeed(Option.none())
                  : cancel.value.pipe(
                      Effect.uninterruptible,
                      Effect.forkIn(data.scope, { startImmediately: true }),
                      Effect.map(Option.some),
                    ),
              ),
            ),
            (cancellation) =>
              Option.isNone(cancellation)
                ? Effect.void
                : SynchronizedRef.update(data.lifecycle, (admission) => {
                    const stopped = new Set(admission.stopped)
                    stopped.delete(sessionID)
                    return { ...admission, stopped }
                  }).pipe(Effect.andThen(publishRevertChange(data, sessionID))),
          )
          if (Option.isNone(cancellation)) return yield* busyError(sessionID)
          // The runner publishes idle when it stops and no background Task lease is held.
          yield* Fiber.join(cancellation.value)
        }),
      )
    })

    const awaitRevert: Interface["awaitRevert"] = (sessionID, staged) =>
      InstanceState.useEffect(
        state,
        Effect.fnUntraced(function* (data) {
          const reverting = staged.pipe(
            Effect.map((value) => value || SynchronizedRef.getUnsafe(data.lifecycle).stopped.has(sessionID)),
          )
          if (!(yield* reverting)) return Option.none()
          const change = data.revertChanges.get(sessionID) ?? Deferred.makeUnsafe<void>()
          data.revertChanges.set(sessionID, change)
          // A revert that finished before the registration did not complete this wait, so read again.
          if (!(yield* reverting)) return Option.none()
          return Option.some(Deferred.await(change))
        }),
      )

    const revertChanged = Effect.fn("SessionRunState.revertChanged")(function* (sessionID: SessionID) {
      return yield* InstanceState.useEffect(state, (data) => publishRevertChange(data, sessionID))
    })

    const markRemoving = Effect.fn("SessionRunState.markRemoving")(function* (sessionID: SessionID) {
      return yield* InstanceState.useEffect(state, (data) =>
        SynchronizedRef.modifyEffect(
          data.lifecycle,
          Effect.fnUntraced(function* (admission) {
            if (admission.removing.has(sessionID)) return [undefined, admission] as const
            yield* SynchronizedRef.update(data.checkpoints, (checkpoints) => {
              if (!checkpoints.has(sessionID)) return checkpoints
              const next = new Map(checkpoints)
              next.delete(sessionID)
              return next
            })
            const removing = new Set(admission.removing)
            removing.add(sessionID)
            return [undefined, { ...admission, removing }] as const
          }),
        ),
      )
    })

    const remove = Effect.fn("SessionRunState.remove")(function* (sessionID: SessionID) {
      yield* markRemoving(sessionID)
      yield* cancel(sessionID)
    })

    const cancelTasks = Effect.fn("SessionRunState.cancelTasks")(function* (
      sessionID: SessionID,
      messageIDs: ReadonlySet<string>,
      taskIDs?: ReadonlySet<string>,
    ) {
      const cancelled = yield* cancelBackgroundJobs(background, sessionID, messageIDs, taskIDs)
      yield* InstanceState.useEffect(state, (data) => stopTaskRunners(data, cancelled))
    })

    const checkpoint = Effect.fn("SessionRunState.checkpoint")(function* (sessionID: SessionID) {
      return yield* InstanceState.useEffect(state, (data) =>
        SynchronizedRef.modifyEffect(
          data.lifecycle,
          Effect.fnUntraced(function* (admission) {
            if (admission.removing.has(sessionID)) return yield* removingError(sessionID)
            const generation = yield* SynchronizedRef.modify(data.checkpoints, (checkpoints) => {
              const current = checkpoints.get(sessionID)
              if (current) return [current.generation, checkpoints] as const
              const generation = data.generation + 1
              data.generation = generation
              return [generation, new Map(checkpoints).set(sessionID, { generation, leases: 0 })] as const
            })
            return [generation, admission] as const
          }),
        ),
      )
    })

    const isCurrent = Effect.fn("SessionRunState.isCurrent")(function* (sessionID: SessionID, expected: number) {
      return yield* InstanceState.useEffect(state, (data) =>
        SynchronizedRef.modifyEffect(
          data.lifecycle,
          Effect.fnUntraced(function* (admission) {
            const current = admission.removing.has(sessionID)
              ? false
              : (yield* SynchronizedRef.get(data.checkpoints)).get(sessionID)?.generation === expected
            return [current, admission] as const
          }),
        ),
      )
    })

    const retain = Effect.fn("SessionRunState.retain")(function* (sessionID: SessionID, expected: number) {
      return yield* InstanceState.useEffect(state, (data) =>
        SynchronizedRef.modifyEffect(
          data.lifecycle,
          Effect.fnUntraced(function* (admission) {
            if (admission.removing.has(sessionID)) return [Option.none(), admission] as const
            const retained = yield* SynchronizedRef.modify(data.checkpoints, (checkpoints) => {
              const current = checkpoints.get(sessionID)
              if (current?.generation !== expected) return [Option.none(), checkpoints] as const
              const released = { value: false }
              const release = Effect.sync(() => {
                if (released.value) return false
                released.value = true
                return true
              }).pipe(
                Effect.flatMap((active) =>
                  active
                    ? SynchronizedRef.modify(data.checkpoints, (latest) => {
                        const held = latest.get(sessionID)
                        if (held?.generation !== expected || held.leases === 0) return [false, latest] as const
                        const next = new Map(latest)
                        if (held.leases === 1 && !data.runners.get(sessionID)?.busy) {
                          next.delete(sessionID)
                          return [true, next] as const
                        }
                        next.set(sessionID, { ...held, leases: held.leases - 1 })
                        return [false, next] as const
                      })
                    : Effect.succeed(false),
                ),
                Effect.flatMap((settled) => (settled ? publishIdleAfterRelease(data, sessionID) : Effect.void)),
              )
              const next = new Map(checkpoints).set(sessionID, { ...current, leases: current.leases + 1 })
              return [Option.some(release), next] as const
            })
            return [retained, admission] as const
          }),
        ),
      )
    })

    const admit: Interface["admit"] = (requirements, effect) =>
      InstanceState.useEffect(state, (data) =>
        SynchronizedRef.modifyEffect(
          data.lifecycle,
          Effect.fnUntraced(function* (admission) {
            const rejected = requirements.some((requirement) => {
              if (admission.removing.has(requirement.sessionID)) return true
              if (requirement.checkpoint === undefined) return false
              return (
                SynchronizedRef.getUnsafe(data.checkpoints).get(requirement.sessionID)?.generation !==
                requirement.checkpoint
              )
            })
            if (rejected) return [Option.none(), admission] as const
            return [Option.some(yield* effect), admission] as const
          }),
        ),
      )

    const admitIfCurrent: Interface["admitIfCurrent"] = (sessionID, expected, effect) =>
      admit([{ sessionID, checkpoint: expected }], effect)

    const ensureRunning = Effect.fn("SessionRunState.ensureRunning")(function* (
      sessionID: SessionID,
      onInterrupt: Effect.Effect<SessionV1.WithParts>,
      work: Effect.Effect<SessionV1.WithParts>,
    ) {
      const admitted = yield* InstanceState.useEffect(state, (data) =>
        SynchronizedRef.modifyEffect(
          data.lifecycle,
          Effect.fnUntraced(function* (admission) {
            if (admission.removing.has(sessionID)) return [Option.none(), admission] as const
            // A stopped turn does not restart, for example from a queued task notification, until the revert finishes.
            if (admission.stopped.has(sessionID)) return [Option.some(onInterrupt), admission] as const
            const existing = yield* runner(sessionID, onInterrupt)
            return [Option.some(yield* existing.enqueueWake(work)), admission] as const
          }),
        ),
      )
      if (Option.isNone(admitted)) return yield* removingError(sessionID)
      return yield* admitted.value
    })

    const startShell = Effect.fn("SessionRunState.startShell")(function* (
      sessionID: SessionID,
      onInterrupt: Effect.Effect<SessionV1.WithParts>,
      work: Effect.Effect<SessionV1.WithParts>,
      ready?: Latch.Latch,
    ) {
      const admitted = yield* InstanceState.useEffect(state, (data) =>
        SynchronizedRef.modifyEffect(
          data.lifecycle,
          Effect.fnUntraced(function* (admission) {
            if (admission.removing.has(sessionID)) return [Option.none(), admission] as const
            if (admission.stopped.has(sessionID))
              return [Option.some(Effect.fail(new Runner.Busy())), admission] as const
            const existing = yield* runner(sessionID, onInterrupt)
            return [Option.some(yield* existing.enqueueShell(work, ready)), admission] as const
          }),
        ),
      )
      if (Option.isNone(admitted)) return yield* removingError(sessionID)
      return yield* admitted.value.pipe(Effect.catchTag("RunnerBusy", () => Effect.fail(busyError(sessionID))))
    })

    const unsubscribe = yield* events.listen((event) => {
      if (event.type !== Session.Event.Removing.type) return Effect.void
      const data = Schema.decodeUnknownOption(Session.Event.Removing.fields.data)(event.data)
      return Option.isSome(data) ? remove(data.value.sessionID) : Effect.void
    })
    yield* Effect.addFinalizer(() => unsubscribe)

    return Service.of({
      assertNotBusy,
      cancel,
      interruptTurn,
      notificationTurnsPaused,
      resumeNotificationTurns,
      stopTurn,
      awaitRevert,
      revertChanged,
      cancelTasks,
      checkpoint,
      isCurrent,
      retain,
      admit,
      admitIfCurrent,
      ensureRunning,
      startShell,
    })
  }),
)

const cancelBackgroundJobs = Effect.fn("SessionRunState.cancelBackgroundJobs")(function* (
  background: BackgroundJob.Interface,
  sessionID: SessionID,
  messageIDs?: ReadonlySet<string>,
  taskIDs?: ReadonlySet<string>,
) {
  const jobs = yield* background.list()
  // Without message IDs the Session owns every job under it. With message IDs it owns
  // only the jobs those messages started or resumed, and every job under them.
  const owned = new Set<string>(messageIDs ? [] : [sessionID])
  const pending = jobs.slice()

  while (pending.length) {
    const index = pending.findIndex((job) => {
      const ancestors = job.metadata?.ancestorSessionIds
      const messageID = job.metadata?.messageId
      return (
        (messageIDs !== undefined &&
          job.metadata?.parentSessionId === sessionID &&
          ((typeof messageID === "string" && messageIDs.has(messageID)) || taskIDs?.has(job.id) === true)) ||
        owned.has(job.id) ||
        (typeof job.metadata?.sessionId === "string" && owned.has(job.metadata.sessionId)) ||
        (typeof job.metadata?.parentSessionId === "string" && owned.has(job.metadata.parentSessionId)) ||
        (Array.isArray(ancestors) && ancestors.some((item) => typeof item === "string" && owned.has(item)))
      )
    })
    if (index === -1) break
    const [job] = pending.splice(index, 1)
    owned.add(job.id)
    if (typeof job.metadata?.sessionId === "string") owned.add(job.metadata.sessionId)
  }

  const running = jobs.filter((job) => job.status === "running" && owned.has(job.id))
  // A revert holds the results of the Tasks it keeps, so it must discard the results of the Tasks it cancels.
  // Claim every completion delivery before any cancel, so a cancelled descendant cannot deliver either.
  if (messageIDs)
    yield* Effect.forEach(
      running,
      (job) => background.extend({ id: job.id, claimCompletion: true, run: Effect.interrupt }),
      { discard: true },
    )
  yield* Effect.forEach(running, (job) => background.cancel(job.id), { concurrency: "unbounded", discard: true })
  return running.map((job) => job.id)
})

function resume(admission: Lifecycle, sessionID: SessionID): Lifecycle {
  if (!admission.paused.has(sessionID)) return admission
  const paused = new Set(admission.paused)
  paused.delete(sessionID)
  return { ...admission, paused }
}

// Task jobs use the Task Session ID as the job ID. Cancellation can finish delivering a Task result later, but it
// must stop the Task runner first, so a cancelled Task cannot keep working, for example after a revert restores files.
function stopTaskRunners(data: State, jobIDs: readonly string[]) {
  // Only Task jobs have a Session runner. Other job IDs are not Session IDs, and decoding them must not fail abort.
  return Effect.forEach(
    jobIDs.flatMap((id) => Option.toArray(Schema.decodeUnknownOption(SessionID)(id))),
    (id) => data.runners.get(id)?.cancel ?? Effect.void,
    { concurrency: "unbounded", discard: true },
  )
}

function publishRevertChange(data: State, sessionID: SessionID) {
  return Effect.suspend(() => {
    const change = data.revertChanges.get(sessionID)
    if (!change) return Effect.void
    data.revertChanges.delete(sessionID)
    return Deferred.succeed(change, undefined).pipe(Effect.asVoid)
  })
}

function busyError(sessionID: SessionID) {
  return new Session.BusyError({ sessionID })
}

function removingError(sessionID: SessionID) {
  return Effect.die(new Session.RemovingError({ sessionID }))
}

export const node = LayerNode.make({
  service: Service,
  layer: layer,
  deps: [BackgroundJob.node, SessionStatus.node, EventV2Bridge.node],
})

export * as SessionRunState from "./run-state"
