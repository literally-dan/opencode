import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { InstanceState } from "@/effect/instance-state"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { Runner } from "@/effect/runner"
import { BackgroundJob } from "@/background/job"
import { Effect, Latch, Layer, Scope, Context, Option, Schema, SynchronizedRef } from "effect"
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
}

export type Admission = {
  sessionID: SessionID
  checkpoint?: number
}

export interface Interface {
  readonly assertNotBusy: (sessionID: SessionID) => Effect.Effect<void, Session.BusyError>
  readonly cancel: (sessionID: SessionID) => Effect.Effect<void>
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

    const state = yield* InstanceState.make(
      Effect.fn("SessionRunState.state")(function* () {
        const scope = yield* Scope.Scope
        const runners = new Map<SessionID, Runner.Runner<SessionV1.WithParts>>()
        const checkpoints = yield* SynchronizedRef.make(new Map<SessionID, Checkpoint>())
        const lifecycle = yield* SynchronizedRef.make<Lifecycle>({ removing: new Set() })
        yield* Effect.addFinalizer(
          Effect.fnUntraced(function* () {
            yield* Effect.forEach(runners.values(), (runner) => runner.cancel, {
              concurrency: "unbounded",
              discard: true,
            })
            runners.clear()
          }),
        )
        return { runners, scope, checkpoints, lifecycle, generation: 0 }
      }),
    )

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
                data.runners.delete(sessionID)
                yield* status.set(sessionID, { type: "idle" })
                yield* SynchronizedRef.update(data.checkpoints, (checkpoints) => {
                  const checkpoint = checkpoints.get(sessionID)
                  if (!checkpoint || checkpoint.leases > 0) return checkpoints
                  const next = new Map(checkpoints)
                  next.delete(sessionID)
                  return next
                })
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
              return [undefined, admission] as const
            }),
          )
          yield* cancelBackgroundJobs(background, sessionID)
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
        }),
      )
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
            return [undefined, { removing }] as const
          }),
        ),
      )
    })

    const remove = Effect.fn("SessionRunState.remove")(function* (sessionID: SessionID) {
      yield* markRemoving(sessionID)
      yield* cancel(sessionID)
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
                    ? SynchronizedRef.update(data.checkpoints, (latest) => {
                        const held = latest.get(sessionID)
                        if (held?.generation !== expected || held.leases === 0) return latest
                        const next = new Map(latest)
                        if (held.leases === 1 && !data.runners.get(sessionID)?.busy) {
                          next.delete(sessionID)
                          return next
                        }
                        next.set(sessionID, { ...held, leases: held.leases - 1 })
                        return next
                      })
                    : Effect.void,
                ),
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
      const admitted = yield* admit(
        [{ sessionID }],
        Effect.gen(function* () {
          return yield* (yield* runner(sessionID, onInterrupt)).enqueueWake(work)
        }),
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
      const admitted = yield* admit(
        [{ sessionID }],
        Effect.gen(function* () {
          return yield* (yield* runner(sessionID, onInterrupt)).enqueueShell(work, ready)
        }),
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
) {
  const jobs = yield* background.list()
  const owned = new Set<string>([sessionID])
  const pending = jobs.slice()

  while (pending.length) {
    const index = pending.findIndex((job) => {
      const ancestors = job.metadata?.ancestorSessionIds
      return (
        owned.has(job.id) ||
        (typeof job.metadata?.sessionId === "string" && owned.has(job.metadata.sessionId)) ||
        (typeof job.metadata?.parentSessionId === "string" && owned.has(job.metadata.parentSessionId)) ||
        (Array.isArray(ancestors) && ancestors.some((item) => typeof item === "string" && item === sessionID))
      )
    })
    if (index === -1) break
    const [job] = pending.splice(index, 1)
    owned.add(job.id)
    if (typeof job.metadata?.sessionId === "string") owned.add(job.metadata.sessionId)
  }

  yield* Effect.forEach(
    jobs.filter((job) => job.status === "running" && owned.has(job.id)),
    (job) => background.cancel(job.id),
    { concurrency: "unbounded", discard: true },
  )
})

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
