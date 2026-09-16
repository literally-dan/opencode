export * as BackgroundJob from "./background-job"

import { Cause, Clock, Context, Deferred, Effect, Exit, Layer, Option, Scope, SynchronizedRef } from "effect"
import { Identifier } from "./id/id"
import { makeGlobalNode } from "./effect/app-node"

export type Status = "running" | "completed" | "error" | "cancelled"

export type Info = {
  id: string
  type: string
  title?: string
  status: Status
  started_at: number
  completed_at?: number
  /** While running, the output of the latest settled run. */
  output?: string
  error?: string
  metadata?: Record<string, unknown>
}

type Settlement = {
  sequence: number
  status: Exclude<Status, "running">
  error?: string
}

type Active = {
  info: Info
  done: Deferred.Deferred<Info>
  scope: Scope.Closeable
  token: object
  settling: "running" | "claimable" | "finalizing"
  pending: number
  next: number
  output?: { sequence: number; text: string }
  settlement?: Settlement
  tail: Deferred.Deferred<void>
  onComplete?: (info: Info) => Effect.Effect<void>
  awaitOnComplete: boolean
  notifyOnComplete: boolean
}

type State = {
  jobs: SynchronizedRef.SynchronizedRef<Map<string, Active>>
  scope: Scope.Scope
}

type FinalizeResult = {
  info: Info
  done: Deferred.Deferred<Info>
  scope: Scope.Closeable
  token: object
  onComplete?: (info: Info) => Effect.Effect<void>
  awaitOnComplete: boolean
}

type FinishResult = {
  info?: Info
  token?: object
  version?: number
  finalize?: FinalizeResult
}

type StartResult =
  | { info: Info }
  | { info: Info; scope: Scope.Closeable; token: object }
  | { wait: Deferred.Deferred<Info> }

type ExtendResult =
  | { extended: false }
  | {
      extended: true
      previous: Deferred.Deferred<void>
      scope: Scope.Closeable
      tail: Deferred.Deferred<void>
      token: object
      sequence: number
    }

export type StartInput = {
  id?: string
  type: string
  title?: string
  metadata?: Record<string, unknown>
  /** Runs once for this in-memory generation when completion delivery is enabled. */
  onComplete?: (info: Info) => Effect.Effect<void>
  /** Waits for `onComplete` to enqueue delivery before publishing completion. */
  awaitOnComplete?: boolean
  notifyOnComplete?: boolean
  /** Runs once when this in-memory generation closes. */
  onFinalize?: Effect.Effect<void>
  run: Effect.Effect<string, unknown>
}

export type ExtendInput = {
  id: string
  expectedType?: string
  /** Atomically transfers this generation's completion delivery to the extending caller. */
  claimCompletion?: boolean
  /** Runs once after an accepted extension finishes or is cancelled while queued. */
  onFinalize?: Effect.Effect<void>
  run: Effect.Effect<string, unknown>
}

export type HoldInput = {
  id: string
  expectedType?: string
}

export type WaitInput = {
  id: string
  timeout?: number
}

export type WaitResult = {
  info?: Info
  timedOut: boolean
}

export interface Interface {
  readonly list: () => Effect.Effect<Info[]>
  readonly get: (id: string) => Effect.Effect<Info | undefined>
  readonly start: (input: StartInput) => Effect.Effect<Info>
  readonly extend: (input: ExtendInput) => Effect.Effect<boolean>
  /**
   * Keeps a running generation from settling until the returned release runs. A hold runs no work and does not
   * change output. The release is idempotent and does not wait for completion delivery.
   */
  readonly hold: (input: HoldInput) => Effect.Effect<Option.Option<Effect.Effect<void>>>
  readonly wait: (input: WaitInput) => Effect.Effect<WaitResult>
  /** Marks a running generation cancelled. It does not wait for an awaited completion callback; use `wait` for that. */
  readonly cancel: (id: string) => Effect.Effect<Info | undefined>
}

export class Service extends Context.Service<Service, Interface>()("@opencode/BackgroundJob") {}

function snapshot(job: Active): Info {
  return {
    ...job.info,
    ...(job.info.status === "running" && job.output ? { output: job.output.text } : {}),
    ...(job.info.metadata ? { metadata: { ...job.info.metadata } } : {}),
  }
}

function errorText(error: unknown) {
  if (error instanceof Error) return error.message
  return String(error)
}

/**
 * Makes one scoped, process-local registry. Entries are intentionally not
 * durable: process restart or owner-scope closure loses status and interrupts
 * live work. Persisted observation, restart recovery, and remote workers need a
 * separate durable ownership slice rather than pretending this registry has
 * those semantics.
 */
export const make = Effect.gen(function* () {
  const state: State = {
    jobs: yield* SynchronizedRef.make(new Map()),
    scope: yield* Scope.Scope,
  }

  const finalize = Effect.fn("BackgroundJob.finalize")(function* (
    id: string,
    result: FinalizeResult,
    restore: <A, E, R>(effect: Effect.Effect<A, E, R>) => Effect.Effect<A, E, R>,
  ) {
    const publish = Effect.gen(function* () {
      const published = yield* SynchronizedRef.modifyEffect(state.jobs, (jobs) =>
        Effect.sync(() => {
          const job = jobs.get(id)
          if (!job || job.token !== result.token || job.settling !== "finalizing") return [false, jobs] as const
          return [true, new Map(jobs).set(id, { ...job, info: result.info })] as const
        }),
      )
      if (!published) return
      yield* Deferred.succeed(result.done, result.info).pipe(Effect.ignore)
      yield* SynchronizedRef.modifyEffect(state.jobs, (jobs) =>
        Effect.sync(() => {
          const job = jobs.get(id)
          if (!job || job.token !== result.token || job.settling !== "finalizing") return [undefined, jobs] as const
          return [undefined, new Map(jobs).set(id, { ...job, settling: "running" })] as const
        }),
      )
    })
    const cleanup = Scope.close(result.scope, Exit.void).pipe(Effect.ensuring(publish), Effect.uninterruptible)
    const onComplete = result.onComplete
    const delivery = onComplete
      ? result.awaitOnComplete
        ? restore(Effect.suspend(() => onComplete(result.info)).pipe(Effect.ignore))
        : Effect.suspend(() => onComplete(result.info)).pipe(
            Effect.ignore,
            Effect.forkIn(state.scope, { startImmediately: true }),
            Effect.asVoid,
          )
      : Effect.void
    yield* delivery.pipe(Effect.ensuring(cleanup))
  })

  const settle = Effect.fn("BackgroundJob.settle")(
    // A hold release settles without a run result.
    (id: string, token: object, outcome?: { sequence: number; exit: Exit.Exit<string, unknown> }) =>
      Effect.uninterruptibleMask((restore) =>
        Effect.gen(function* () {
          const completed_at = yield* Clock.currentTimeMillis
          const candidate = yield* SynchronizedRef.modifyEffect(state.jobs, (jobs) =>
            Effect.sync((): readonly [FinishResult, Map<string, Active>] => {
              const job = jobs.get(id)
              if (!job) return [{}, jobs]
              if (job.token !== token) return [{}, jobs]
              if (job.info.status !== "running" || job.settling !== "running") return [{ info: snapshot(job) }, jobs]
              const pending = job.pending - 1
              const output =
                outcome && Exit.isSuccess(outcome.exit) && (!job.output || outcome.sequence > job.output.sequence)
                  ? { sequence: outcome.sequence, text: outcome.exit.value }
                  : job.output
              const settled: Settlement | undefined = outcome && {
                sequence: outcome.sequence,
                status: Exit.isSuccess(outcome.exit)
                  ? "completed"
                  : Cause.hasInterruptsOnly(outcome.exit.cause)
                    ? "cancelled"
                    : "error",
                ...(Exit.isFailure(outcome.exit) ? { error: errorText(Cause.squash(outcome.exit.cause)) } : {}),
              }
              // Terminal state follows the highest accepted sequence, not settle order.
              const settlement =
                settled && (!job.settlement || settled.sequence > job.settlement.sequence) ? settled : job.settlement
              if (pending > 0 || !settlement) {
                return [{}, new Map(jobs).set(id, { ...job, pending, output, settlement })]
              }
              const next = {
                ...job,
                settling: "claimable" as const,
                pending: 0,
                output,
                settlement,
              }
              const info = {
                ...job.info,
                status: settlement.status,
                completed_at,
                ...(output ? { output: output.text } : {}),
                ...(settlement.error !== undefined ? { error: settlement.error } : {}),
              }
              // Give a related completion callback one scheduling turn to append work
              // before this generation takes final ownership of delivery.
              return [{ info, token: job.token, version: job.next }, new Map(jobs).set(id, next)]
            }),
          )
          if (!candidate.info || !candidate.token || candidate.version === undefined) return candidate.info
          const info = candidate.info
          const candidateToken = candidate.token
          const candidateVersion = candidate.version
          yield* Effect.yieldNow
          const result = yield* SynchronizedRef.modifyEffect(state.jobs, (jobs) =>
            Effect.sync((): readonly [FinishResult, Map<string, Active>] => {
              const job = jobs.get(id)
              if (!job || job.token !== candidateToken || job.next !== candidateVersion) return [{}, jobs]
              if (job.info.status !== "running" || job.settling !== "claimable") return [{ info: snapshot(job) }, jobs]
              return [
                {
                  info,
                  finalize: {
                    info,
                    done: job.done,
                    scope: job.scope,
                    token: job.token,
                    ...(job.notifyOnComplete && job.onComplete ? { onComplete: job.onComplete } : {}),
                    awaitOnComplete: job.awaitOnComplete,
                  },
                },
                new Map(jobs).set(id, {
                  ...job,
                  settling: "finalizing",
                  onComplete: undefined,
                  notifyOnComplete: false,
                }),
              ]
            }),
          )
          if (result.finalize) yield* finalize(id, result.finalize, restore)
          return result.info
        }),
      ),
  )

  const fork = Effect.fn("BackgroundJob.fork")(function* (
    scope: Scope.Scope,
    id: string,
    token: object,
    sequence: number,
    run: Effect.Effect<string, unknown>,
    tail: Deferred.Deferred<void>,
  ) {
    return yield* run.pipe(
      Effect.matchCauseEffect({
        onSuccess: (output) => settle(id, token, { sequence, exit: Exit.succeed(output) }),
        onFailure: (cause) => settle(id, token, { sequence, exit: Exit.failCause(cause) }),
      }),
      // The next queued run starts when this tail resolves, so it must see the output that settle recorded.
      // A queued run keeps `pending` above zero, so this settle cannot finalize while one waits.
      Effect.ensuring(Deferred.succeed(tail, undefined)),
      Effect.asVoid,
      Effect.forkIn(scope, { startImmediately: true }),
    )
  })

  const list: Interface["list"] = Effect.fn("BackgroundJob.list")(function* () {
    return Array.from((yield* SynchronizedRef.get(state.jobs)).values())
      .map(snapshot)
      .toSorted((a, b) => a.started_at - b.started_at)
  })

  const get: Interface["get"] = Effect.fn("BackgroundJob.get")(function* (id) {
    const job = (yield* SynchronizedRef.get(state.jobs)).get(id)
    if (!job) return
    return snapshot(job)
  })

  const start: Interface["start"] = Effect.fn("BackgroundJob.start")(function* (input) {
    const id = input.id ?? Identifier.ascending("job")
    const result = yield* Effect.uninterruptibleMask((restore) =>
      Effect.gen(function* () {
        const started_at = yield* Clock.currentTimeMillis
        const done = yield* Deferred.make<Info>()
        const tail = yield* Deferred.make<void>()
        const result = yield* SynchronizedRef.modifyEffect(
          state.jobs,
          Effect.fnUntraced(function* (jobs) {
            const existing = jobs.get(id)
            if (existing?.settling !== undefined && existing.settling !== "running") {
              return [{ wait: existing.done }, jobs] as readonly [StartResult, Map<string, Active>]
            }
            if (existing?.info.status === "running") {
              return [{ info: snapshot(existing) }, jobs] as readonly [StartResult, Map<string, Active>]
            }
            const scope = yield* Scope.fork(state.scope, "sequential")
            if (input.onFinalize) yield* Scope.addFinalizer(scope, input.onFinalize)
            const token = {}
            const job = {
              info: {
                id,
                type: input.type,
                title: input.title,
                status: "running" as const,
                started_at,
                metadata: input.metadata,
              },
              done,
              scope,
              token,
              settling: "running" as const,
              pending: 1,
              next: 1,
              tail,
              onComplete: input.onComplete,
              awaitOnComplete: input.awaitOnComplete ?? false,
              notifyOnComplete: input.notifyOnComplete ?? false,
            }
            return [{ info: snapshot(job), scope, token }, new Map(jobs).set(id, job)] as readonly [
              StartResult,
              Map<string, Active>,
            ]
          }),
        )
        if ("scope" in result) yield* fork(result.scope, id, result.token, 0, restore(input.run), tail)
        return result
      }),
    )
    if ("wait" in result) {
      yield* Deferred.await(result.wait).pipe(Effect.interruptible)
      return yield* start({ ...input, id })
    }
    return result.info
  })

  const extend: Interface["extend"] = Effect.fn("BackgroundJob.extend")(function* (input) {
    return yield* Effect.uninterruptibleMask((restore) =>
      Effect.gen(function* () {
        const tail = yield* Deferred.make<void>()
        const result = yield* SynchronizedRef.modifyEffect(state.jobs, (jobs) =>
          Effect.sync((): readonly [ExtendResult, Map<string, Active>] => {
            const job = jobs.get(input.id)
            if (
              !job ||
              job.info.status !== "running" ||
              job.settling === "finalizing" ||
              (input.expectedType !== undefined && job.info.type !== input.expectedType)
            )
              return [{ extended: false }, jobs]
            return [
              { extended: true, previous: job.tail, scope: job.scope, tail, token: job.token, sequence: job.next },
              new Map(jobs).set(input.id, {
                ...job,
                settling: "running",
                pending: job.pending + 1,
                next: job.next + 1,
                tail,
                notifyOnComplete: input.claimCompletion ? false : job.notifyOnComplete,
              }),
            ]
          }),
        )
        if (!result.extended) return false
        yield* fork(
          result.scope,
          input.id,
          result.token,
          result.sequence,
          Deferred.await(result.previous).pipe(
            Effect.andThen(restore(input.run)),
            Effect.ensuring(input.onFinalize ?? Effect.void),
          ),
          result.tail,
        )
        return true
      }),
    )
  })

  const hold: Interface["hold"] = Effect.fn("BackgroundJob.hold")(function* (input) {
    const token = yield* SynchronizedRef.modifyEffect(state.jobs, (jobs) =>
      Effect.sync((): readonly [object | undefined, Map<string, Active>] => {
        const job = jobs.get(input.id)
        if (
          !job ||
          job.info.status !== "running" ||
          job.settling === "finalizing" ||
          (input.expectedType !== undefined && job.info.type !== input.expectedType)
        )
          return [undefined, jobs]
        // Like an extension, a hold invalidates a claimable settlement that has not taken delivery yet.
        return [
          job.token,
          new Map(jobs).set(input.id, { ...job, settling: "running", pending: job.pending + 1, next: job.next + 1 }),
        ]
      }),
    )
    if (!token) return Option.none()
    const released = { value: false }
    return Option.some(
      Effect.suspend(() => {
        if (released.value) return Effect.void
        released.value = true
        // The last release can finalize the generation. Its completion delivery must not run in the releasing fiber.
        return settle(input.id, token).pipe(Effect.forkIn(state.scope, { startImmediately: true }), Effect.asVoid)
      }),
    )
  })

  const wait: Interface["wait"] = Effect.fn("BackgroundJob.wait")(function* (input) {
    const job = (yield* SynchronizedRef.get(state.jobs)).get(input.id)
    if (!job) return { timedOut: false }
    if (job.info.status !== "running") return { info: snapshot(job), timedOut: false }
    if (input.timeout === undefined) return { info: yield* Deferred.await(job.done), timedOut: false }
    if (input.timeout <= 0) return { info: snapshot(job), timedOut: true }
    const info = yield* Deferred.await(job.done).pipe(Effect.timeoutOption(input.timeout))
    if (info._tag === "Some") return { info: info.value, timedOut: false }
    return { info: snapshot(job), timedOut: true }
  })

  const cancel: Interface["cancel"] = Effect.fn("BackgroundJob.cancel")((id) =>
    Effect.uninterruptibleMask((restore) =>
      Effect.gen(function* () {
        const completed_at = yield* Clock.currentTimeMillis
        const result = yield* SynchronizedRef.modifyEffect(state.jobs, (jobs) =>
          Effect.sync((): readonly [FinishResult, Map<string, Active>] => {
            const job = jobs.get(id)
            if (!job) return [{}, jobs]
            if (job.info.status !== "running" || job.settling === "finalizing") return [{ info: snapshot(job) }, jobs]
            const info = {
              ...job.info,
              status: "cancelled" as const,
              completed_at,
            }
            return [
              {
                info,
                finalize: {
                  info,
                  done: job.done,
                  scope: job.scope,
                  token: job.token,
                  ...(job.notifyOnComplete && job.onComplete ? { onComplete: job.onComplete } : {}),
                  awaitOnComplete: job.awaitOnComplete,
                },
              },
              new Map(jobs).set(id, {
                ...job,
                settling: "finalizing",
                onComplete: undefined,
                notifyOnComplete: false,
                pending: 0,
              }),
            ]
          }),
        )
        if (!result.finalize) return result.info
        // An awaited completion callback can wait for other work, such as an ancestor turn. Cancellation returns once
        // the generation is marked. Delivery and scope cleanup finish in the registry scope.
        if (result.finalize.onComplete && result.finalize.awaitOnComplete) {
          yield* finalize(id, result.finalize, restore).pipe(Effect.forkIn(state.scope, { startImmediately: true }))
          return result.info
        }
        yield* finalize(id, result.finalize, restore)
        return result.info
      }),
    ),
  )

  return Service.of({ list, get, start, extend, hold, wait, cancel })
})

const layer = Layer.effect(Service, make)

export const node = makeGlobalNode({ service: Service, layer, deps: [] })
