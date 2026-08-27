import { Cause, Deferred, Effect, Exit, Fiber, Latch, Schema, Scope, SynchronizedRef } from "effect"

export interface Runner<A, E = never> {
  readonly state: State<A, E>
  readonly busy: boolean
  readonly ensureRunning: (work: Effect.Effect<A, E>) => Effect.Effect<A, E>
  readonly enqueueWake: (work: Effect.Effect<A, E>) => Effect.Effect<Effect.Effect<A, E>>
  readonly wake: (work: Effect.Effect<A, E>) => Effect.Effect<A, E>
  readonly enqueueShell: (work: Effect.Effect<A, E>, ready?: Latch.Latch) => Effect.Effect<Effect.Effect<A, E | Busy>>
  readonly startShell: (work: Effect.Effect<A, E>, ready?: Latch.Latch) => Effect.Effect<A, E | Busy>
  readonly enqueueCancel: Effect.Effect<Effect.Effect<void>>
  readonly cancel: Effect.Effect<void>
}

export class Cancelled extends Schema.TaggedErrorClass<Cancelled>()("RunnerCancelled", {}) {}
export class Busy extends Schema.TaggedErrorClass<Busy>()("RunnerBusy", {}) {}

interface RunHandle<A, E> {
  id: number
  done: Deferred.Deferred<A, E | Cancelled>
  fiber: Fiber.Fiber<A, E>
}

interface ShellHandle<A, E> {
  id: number
  cancelled: Deferred.Deferred<void>
  ready?: Latch.Latch
  fiber: Fiber.Fiber<A, E>
}

interface PendingHandle<A, E> {
  id: number
  done: Deferred.Deferred<A, E | Cancelled>
  work: Effect.Effect<A, E>
}

interface CancellationHandle {
  id: number
  done: Deferred.Deferred<void>
}

export type State<A, E> =
  | { readonly _tag: "Idle" }
  | { readonly _tag: "Running"; readonly run: RunHandle<A, E> }
  | { readonly _tag: "RunningThenRun"; readonly run: RunHandle<A, E>; readonly next: PendingHandle<A, E> }
  | { readonly _tag: "Finalizing"; readonly run: RunHandle<A, E> }
  | { readonly _tag: "FinalizingThenRun"; readonly run: RunHandle<A, E>; readonly next: PendingHandle<A, E> }
  | { readonly _tag: "Shell"; readonly shell: ShellHandle<A, E> }
  | { readonly _tag: "ShellThenRun"; readonly shell: ShellHandle<A, E>; readonly run: PendingHandle<A, E> }
  | { readonly _tag: "ShellFinalizing"; readonly shell: ShellHandle<A, E> }
  | { readonly _tag: "ShellFinalizingThenRun"; readonly shell: ShellHandle<A, E>; readonly run: PendingHandle<A, E> }
  | { readonly _tag: "Cancelling"; readonly cancellation: CancellationHandle }
  | {
      readonly _tag: "CancellingThenRun"
      readonly cancellation: CancellationHandle
      readonly run: PendingHandle<A, E>
    }

export const make = <A, E = never>(
  scope: Scope.Scope,
  opts?: {
    onIdle?: Effect.Effect<void>
    onBusy?: Effect.Effect<void>
    onInterrupt?: Effect.Effect<A, E>
  },
): Runner<A, E> => {
  const ref = SynchronizedRef.makeUnsafe<State<A, E>>({ _tag: "Idle" })
  const idle = opts?.onIdle ?? Effect.void
  const onBusy = opts?.onBusy ?? Effect.void
  const onInterrupt = opts?.onInterrupt
  let ids = 0

  const state = () => SynchronizedRef.getUnsafe(ref)
  const next = () => {
    ids += 1
    return ids
  }

  const complete = (done: Deferred.Deferred<A, E | Cancelled>, exit: Exit.Exit<A, E>) =>
    Exit.isFailure(exit) && Cause.hasInterruptsOnly(exit.cause)
      ? Deferred.fail(done, new Cancelled()).pipe(Effect.asVoid)
      : Deferred.done(done, exit).pipe(Effect.asVoid)

  const awaitDone = (done: Deferred.Deferred<A, E | Cancelled>) =>
    Deferred.await(done).pipe(Effect.catchTag("RunnerCancelled", (e) => onInterrupt ?? Effect.die(e)))

  const finalizeRun = (
    id: number,
    done: Deferred.Deferred<A, E | Cancelled>,
    exit: Exit.Exit<A, E>,
  ): Effect.Effect<void> =>
    SynchronizedRef.modifyEffect(
      ref,
      Effect.fnUntraced(function* (st) {
        if (st._tag === "Finalizing" && st.run.id === id) {
          return [complete(done, exit), { _tag: "Idle" }] as const
        }
        if (st._tag === "FinalizingThenRun" && st.run.id === id) {
          const run = yield* startRun(st.next.work, st.next.done)
          return [complete(done, exit), { _tag: "Running", run }] as const
        }
        return [complete(done, exit), st] as const
      }),
    ).pipe(Effect.flatten)

  const finishRun = (
    id: number,
    done: Deferred.Deferred<A, E | Cancelled>,
    exit: Exit.Exit<A, E>,
  ): Effect.Effect<void> =>
    SynchronizedRef.modifyEffect(ref, (st): Effect.Effect<readonly [Effect.Effect<void>, State<A, E>]> => {
      if (st._tag === "Running" && st.run.id === id)
        return Effect.succeed([
          idle.pipe(Effect.andThen(finalizeRun(id, done, exit))),
          { _tag: "Finalizing", run: st.run },
        ] as const)
      if (st._tag === "RunningThenRun" && st.run.id === id) {
        return startRun(st.next.work, st.next.done).pipe(
          Effect.map((run) => [complete(done, exit), { _tag: "Running", run }] as const),
        )
      }
      if ((st._tag === "Cancelling" || st._tag === "CancellingThenRun") && st.cancellation.id === id) {
        return Effect.succeed([complete(done, exit), st] as const)
      }
      return Effect.succeed([complete(done, exit), st] as const)
    }).pipe(Effect.flatten)

  function startRun(
    work: Effect.Effect<A, E>,
    done: Deferred.Deferred<A, E | Cancelled>,
  ): Effect.Effect<RunHandle<A, E>> {
    return Effect.gen(function* () {
      const id = next()
      const fiber = yield* work.pipe(
        Effect.onExit((exit) => finishRun(id, done, exit)),
        Effect.forkIn(scope),
      )
      return { id, done, fiber } satisfies RunHandle<A, E>
    })
  }

  const finalizeShell = (id: number) =>
    SynchronizedRef.modifyEffect(
      ref,
      Effect.fnUntraced(function* (st) {
        if (st._tag === "ShellFinalizing" && st.shell.id === id) {
          return [Effect.void, { _tag: "Idle" }] as const
        }
        if (st._tag === "ShellFinalizingThenRun" && st.shell.id === id) {
          const run = yield* startRun(st.run.work, st.run.done)
          return [Effect.void, { _tag: "Running", run }] as const
        }
        return [Effect.void, st] as const
      }),
    ).pipe(Effect.flatten)

  const finishShell = (id: number) =>
    SynchronizedRef.modifyEffect(
      ref,
      Effect.fnUntraced(function* (st) {
        if (st._tag === "Shell" && st.shell.id === id) {
          return [idle.pipe(Effect.andThen(finalizeShell(id))), { _tag: "ShellFinalizing", shell: st.shell }] as const
        }
        if (st._tag === "ShellThenRun" && st.shell.id === id) {
          const run = yield* startRun(st.run.work, st.run.done)
          return [Effect.void, { _tag: "Running", run }] as const
        }
        if ((st._tag === "Cancelling" || st._tag === "CancellingThenRun") && st.cancellation.id === id) {
          return [Effect.void, st] as const
        }
        return [Effect.void, st] as const
      }),
    ).pipe(Effect.flatten)

  const stopShell = (shell: ShellHandle<A, E>) =>
    Effect.gen(function* () {
      if (shell.ready) yield* shell.ready.await.pipe(Effect.exit, Effect.asVoid)
      yield* Deferred.succeed(shell.cancelled, undefined).pipe(Effect.asVoid)
      yield* Fiber.interrupt(shell.fiber)
    })

  const ensureRunning = (work: Effect.Effect<A, E>) =>
    SynchronizedRef.modifyEffect(
      ref,
      Effect.fnUntraced(function* (st) {
        switch (st._tag) {
          case "Running":
          case "RunningThenRun":
          case "Finalizing":
          case "FinalizingThenRun":
          case "ShellThenRun":
            return [awaitDone(st.run.done), st] as const
          case "Shell": {
            const run = {
              id: next(),
              done: yield* Deferred.make<A, E | Cancelled>(),
              work,
            } satisfies PendingHandle<A, E>
            return [awaitDone(run.done), { _tag: "ShellThenRun", shell: st.shell, run }] as const
          }
          case "ShellFinalizing": {
            const run = {
              id: next(),
              done: yield* Deferred.make<A, E | Cancelled>(),
              work,
            } satisfies PendingHandle<A, E>
            return [awaitDone(run.done), { _tag: "ShellFinalizingThenRun", shell: st.shell, run }] as const
          }
          case "ShellFinalizingThenRun":
          case "CancellingThenRun":
            return [awaitDone(st.run.done), st] as const
          case "Cancelling": {
            const run = {
              id: next(),
              done: yield* Deferred.make<A, E | Cancelled>(),
              work,
            } satisfies PendingHandle<A, E>
            return [awaitDone(run.done), { _tag: "CancellingThenRun", cancellation: st.cancellation, run }] as const
          }
          case "Idle": {
            const done = yield* Deferred.make<A, E | Cancelled>()
            const run = yield* startRun(work, done)
            return [awaitDone(done), { _tag: "Running", run }] as const
          }
        }
      }),
    ).pipe(Effect.flatten)

  const enqueueWake = (work: Effect.Effect<A, E>) =>
    SynchronizedRef.modifyEffect(
      ref,
      Effect.fnUntraced(function* (st) {
        switch (st._tag) {
          case "Running": {
            const pendingRun = {
              id: next(),
              done: yield* Deferred.make<A, E | Cancelled>(),
              work,
            } satisfies PendingHandle<A, E>
            return [awaitDone(pendingRun.done), { _tag: "RunningThenRun", run: st.run, next: pendingRun }] as const
          }
          case "Finalizing": {
            const pendingRun = {
              id: next(),
              done: yield* Deferred.make<A, E | Cancelled>(),
              work,
            } satisfies PendingHandle<A, E>
            return [awaitDone(pendingRun.done), { _tag: "FinalizingThenRun", run: st.run, next: pendingRun }] as const
          }
          case "RunningThenRun":
            return [awaitDone(st.next.done), st] as const
          case "FinalizingThenRun":
            return [awaitDone(st.next.done), st] as const
          case "ShellThenRun":
            return [awaitDone(st.run.done), st] as const
          case "Shell": {
            const run = {
              id: next(),
              done: yield* Deferred.make<A, E | Cancelled>(),
              work,
            } satisfies PendingHandle<A, E>
            return [awaitDone(run.done), { _tag: "ShellThenRun", shell: st.shell, run }] as const
          }
          case "ShellFinalizing": {
            const run = {
              id: next(),
              done: yield* Deferred.make<A, E | Cancelled>(),
              work,
            } satisfies PendingHandle<A, E>
            return [awaitDone(run.done), { _tag: "ShellFinalizingThenRun", shell: st.shell, run }] as const
          }
          case "ShellFinalizingThenRun":
          case "CancellingThenRun":
            return [awaitDone(st.run.done), st] as const
          case "Cancelling": {
            const run = {
              id: next(),
              done: yield* Deferred.make<A, E | Cancelled>(),
              work,
            } satisfies PendingHandle<A, E>
            return [awaitDone(run.done), { _tag: "CancellingThenRun", cancellation: st.cancellation, run }] as const
          }
          case "Idle": {
            const done = yield* Deferred.make<A, E | Cancelled>()
            const run = yield* startRun(work, done)
            return [awaitDone(done), { _tag: "Running", run }] as const
          }
        }
      }),
    )

  const wake = (work: Effect.Effect<A, E>) => enqueueWake(work).pipe(Effect.flatten)

  const enqueueShell = (work: Effect.Effect<A, E>, ready?: Latch.Latch): Effect.Effect<Effect.Effect<A, E | Busy>> =>
    SynchronizedRef.modifyEffect(
      ref,
      Effect.fnUntraced(function* (st) {
        if (st._tag !== "Idle") {
          const reject: Effect.Effect<A, E | Busy> = Effect.fail(new Busy())
          return [reject, st] as const
        }
        yield* onBusy
        const id = next()
        const cancelled = yield* Deferred.make<void>()
        const fiber = yield* work.pipe(Effect.ensuring(finishShell(id)), Effect.forkChild)
        const shell = { id, cancelled, ready, fiber } satisfies ShellHandle<A, E>
        return [
          Effect.gen(function* () {
            const exit = yield* Fiber.await(fiber)
            if (Exit.isSuccess(exit)) return exit.value
            if (
              Cause.hasInterruptsOnly(exit.cause) ||
              ((yield* Deferred.isDone(cancelled)) && Cause.hasInterrupts(exit.cause) && !Cause.hasDies(exit.cause))
            ) {
              if (onInterrupt) return yield* onInterrupt
              return yield* Effect.die(new Cancelled())
            }
            return yield* Effect.failCause(exit.cause)
          }),
          { _tag: "Shell", shell },
        ] as const
      }),
    )

  const startShell = (work: Effect.Effect<A, E>, ready?: Latch.Latch): Effect.Effect<A, E | Busy> =>
    enqueueShell(work, ready).pipe(Effect.flatten)

  const finishCancellation = (cancellation: CancellationHandle) => {
    const completeCancellation = Deferred.succeed(cancellation.done, undefined).pipe(Effect.asVoid)
    const finalize = SynchronizedRef.modifyEffect(
      ref,
      Effect.fnUntraced(function* (st) {
        if (st._tag === "Cancelling" && st.cancellation.id === cancellation.id) {
          return [completeCancellation, { _tag: "Idle" }] as const
        }
        if (st._tag === "CancellingThenRun" && st.cancellation.id === cancellation.id) {
          const run = yield* startRun(st.run.work, st.run.done)
          return [completeCancellation, { _tag: "Running", run }] as const
        }
        return [completeCancellation, st] as const
      }),
    ).pipe(Effect.flatten)
    return SynchronizedRef.modifyEffect(
      ref,
      Effect.fnUntraced(function* (st) {
        if (st._tag === "Cancelling" && st.cancellation.id === cancellation.id) {
          return [idle.pipe(Effect.andThen(finalize)), st] as const
        }
        if (st._tag === "CancellingThenRun" && st.cancellation.id === cancellation.id) {
          const run = yield* startRun(st.run.work, st.run.done)
          return [completeCancellation, { _tag: "Running", run }] as const
        }
        return [completeCancellation, st] as const
      }),
    ).pipe(Effect.flatten)
  }

  const enqueueCancel = SynchronizedRef.modifyEffect(
    ref,
    Effect.fnUntraced(function* (st) {
      switch (st._tag) {
        case "Idle":
          return [Effect.void, st] as const
        case "Finalizing":
          return [Deferred.await(st.run.done).pipe(Effect.exit, Effect.asVoid), st] as const
        case "FinalizingThenRun":
          return [
            Deferred.fail(st.next.done, new Cancelled()).pipe(
              Effect.andThen(Deferred.await(st.run.done).pipe(Effect.exit)),
              Effect.asVoid,
            ),
            { _tag: "Finalizing", run: st.run },
          ] as const
        case "ShellFinalizing":
          return [Fiber.await(st.shell.fiber).pipe(Effect.asVoid), st] as const
        case "ShellFinalizingThenRun":
          return [
            Deferred.fail(st.run.done, new Cancelled()).pipe(
              Effect.andThen(Fiber.await(st.shell.fiber)),
              Effect.asVoid,
            ),
            { _tag: "ShellFinalizing", shell: st.shell },
          ] as const
        case "Cancelling":
          return [Deferred.await(st.cancellation.done), st] as const
        case "CancellingThenRun":
          return [
            Deferred.fail(st.run.done, new Cancelled()).pipe(
              Effect.andThen(Deferred.await(st.cancellation.done)),
              Effect.asVoid,
            ),
            { _tag: "Cancelling", cancellation: st.cancellation },
          ] as const
        case "Running": {
          const cancellation = { id: st.run.id, done: yield* Deferred.make<void>() } satisfies CancellationHandle
          return [
            Effect.gen(function* () {
              yield* Fiber.interrupt(st.run.fiber)
              yield* Deferred.fail(st.run.done, new Cancelled()).pipe(Effect.asVoid)
              yield* finishCancellation(cancellation)
            }),
            { _tag: "Cancelling", cancellation },
          ] as const
        }
        case "RunningThenRun": {
          const cancellation = { id: st.run.id, done: yield* Deferred.make<void>() } satisfies CancellationHandle
          return [
            Effect.gen(function* () {
              yield* Fiber.interrupt(st.run.fiber)
              yield* Deferred.fail(st.run.done, new Cancelled()).pipe(Effect.asVoid)
              yield* Deferred.fail(st.next.done, new Cancelled()).pipe(Effect.asVoid)
              yield* finishCancellation(cancellation)
            }),
            { _tag: "Cancelling", cancellation },
          ] as const
        }
        case "Shell": {
          const cancellation = { id: st.shell.id, done: yield* Deferred.make<void>() } satisfies CancellationHandle
          return [
            stopShell(st.shell).pipe(Effect.andThen(finishCancellation(cancellation))),
            { _tag: "Cancelling", cancellation },
          ] as const
        }
        case "ShellThenRun": {
          const cancellation = { id: st.shell.id, done: yield* Deferred.make<void>() } satisfies CancellationHandle
          return [
            stopShell(st.shell).pipe(
              Effect.andThen(Deferred.fail(st.run.done, new Cancelled())),
              Effect.andThen(finishCancellation(cancellation)),
              Effect.asVoid,
            ),
            { _tag: "Cancelling", cancellation },
          ] as const
        }
      }
    }),
  )

  const cancel = enqueueCancel.pipe(Effect.flatten)

  return {
    get state() {
      return state()
    },
    get busy() {
      return state()._tag !== "Idle"
    },
    ensureRunning,
    enqueueWake,
    wake,
    enqueueShell,
    startShell,
    enqueueCancel,
    cancel,
  }
}

export * as Runner from "./runner"
