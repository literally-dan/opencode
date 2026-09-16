import { describe, expect } from "bun:test"
import { BackgroundJob } from "@opencode-ai/core/background-job"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { Clock, Deferred, Effect, Exit, Fiber, Scope } from "effect"
import { it } from "./lib/effect"

const jobsLayer = LayerNode.compile(BackgroundJob.node)

describe("BackgroundJob", () => {
  it.live("tracks process-local work through explicit observation", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const latch = yield* Deferred.make<void>()
      const job = yield* jobs.start({
        type: "test",
        metadata: { durable: false },
        run: Deferred.await(latch).pipe(Effect.as("done")),
      })

      expect(job).toMatchObject({ type: "test", status: "running", metadata: { durable: false } })
      expect(yield* jobs.wait({ id: job.id, timeout: 0 })).toMatchObject({
        timedOut: true,
        info: { status: "running" },
      })

      yield* Deferred.succeed(latch, undefined)
      expect(yield* jobs.wait({ id: job.id })).toMatchObject({
        timedOut: false,
        info: { status: "completed", output: "done" },
      })
    }).pipe(Effect.provide(jobsLayer)),
  )

  it.live("publishes jobs before starting immediately settling work", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service

      yield* Effect.forEach(Array.from({ length: 100 }), (_, index) => {
        const id = `job_immediate_start_${index}`
        return Effect.gen(function* () {
          const job = yield* jobs.start({
            id,
            type: "test",
            run: jobs
              .get(id)
              .pipe(
                Effect.flatMap((info) =>
                  info?.status === "running"
                    ? Effect.succeed(`done-${index}`)
                    : Effect.fail("job started before publish"),
                ),
              ),
          })

          expect(yield* jobs.wait({ id: job.id })).toMatchObject({
            timedOut: false,
            info: { status: "completed", output: `done-${index}` },
          })
        })
      })
    }).pipe(Effect.provide(jobsLayer)),
  )

  it.live("increments pending work before starting immediately settling extensions", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service

      yield* Effect.forEach(Array.from({ length: 100 }), (_, index) =>
        Effect.gen(function* () {
          const first = yield* Deferred.make<void>()
          const job = yield* jobs.start({
            type: "test",
            run: Deferred.await(first).pipe(Effect.as(`first-${index}`)),
          })

          expect(yield* jobs.extend({ id: job.id, run: Effect.succeed(`second-${index}`) })).toBe(true)
          expect((yield* jobs.get(job.id))?.status).toBe("running")

          yield* Deferred.succeed(first, undefined)
          expect(yield* jobs.wait({ id: job.id })).toMatchObject({
            timedOut: false,
            info: { status: "completed", output: `second-${index}` },
          })
        }),
      )
    }).pipe(Effect.provide(jobsLayer)),
  )

  it.live("preserves accepted extension work after preceding work fails", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const fail = yield* Deferred.make<void>()
      const extensionStarted = yield* Deferred.make<void>()
      const finishExtension = yield* Deferred.make<void>()
      const finalized: string[] = []
      const job = yield* jobs.start({
        type: "test",
        onFinalize: Effect.sync(() => finalized.push("generation")).pipe(Effect.asVoid),
        run: Deferred.await(fail).pipe(Effect.andThen(Effect.fail(new Error("first failed")))),
      })
      expect(
        yield* jobs.extend({
          id: job.id,
          onFinalize: Effect.sync(() => finalized.push("extension")).pipe(Effect.asVoid),
          run: Deferred.succeed(extensionStarted, undefined).pipe(
            Effect.andThen(Deferred.await(finishExtension)),
            Effect.as("recovered"),
          ),
        }),
      ).toBe(true)

      yield* Deferred.succeed(fail, undefined)
      yield* Deferred.await(extensionStarted).pipe(Effect.timeout("1 second"))
      expect(yield* jobs.wait({ id: job.id, timeout: 0 })).toMatchObject({
        timedOut: true,
        info: { status: "running" },
      })
      expect(finalized).toEqual([])

      yield* Deferred.succeed(finishExtension, undefined)
      expect((yield* jobs.wait({ id: job.id })).info).toMatchObject({
        status: "completed",
        output: "recovered",
      })
      expect(finalized.toSorted()).toEqual(["extension", "generation"])
    }).pipe(Effect.provide(jobsLayer)),
  )

  it.live("uses the highest accepted sequence when immediate work settles out of order", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const clock = yield* Clock.Clock
      const firstSettlement = yield* Deferred.make<void>()
      const releaseFirstSettlement = yield* Deferred.make<void>()
      const extensionSettlement = yield* Deferred.make<void>()
      const reads = { value: 0 }
      const controlledClock: Clock.Clock = {
        currentTimeMillisUnsafe: () => clock.currentTimeMillisUnsafe(),
        currentTimeMillis: Effect.suspend(() => {
          const read = ++reads.value
          const now = clock.currentTimeMillisUnsafe()
          if (read === 2)
            return Deferred.succeed(firstSettlement, undefined).pipe(
              Effect.andThen(Deferred.await(releaseFirstSettlement)),
              Effect.as(now),
            )
          if (read === 3) return Deferred.succeed(extensionSettlement, undefined).pipe(Effect.as(now))
          return Effect.succeed(now)
        }),
        currentTimeNanosUnsafe: () => clock.currentTimeNanosUnsafe(),
        currentTimeNanos: clock.currentTimeNanos,
        sleep: (duration) => clock.sleep(duration),
      }

      yield* Effect.gen(function* () {
        const job = yield* jobs.start({
          type: "test",
          run: Effect.fail(new Error("first failed")),
        })

        yield* Deferred.await(firstSettlement)
        expect(yield* jobs.extend({ id: job.id, run: Effect.succeed("recovered") })).toBe(true)
        yield* Deferred.await(extensionSettlement)
        yield* Deferred.succeed(releaseFirstSettlement, undefined)
        expect((yield* jobs.wait({ id: job.id })).info).toMatchObject({
          status: "completed",
          output: "recovered",
        })
      }).pipe(Effect.provideService(Clock.Clock, controlledClock))
    }).pipe(Effect.provide(jobsLayer)),
  )

  it.live("atomically claims completion delivery with an accepted extension", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const first = yield* Deferred.make<void>()
      const second = yield* Deferred.make<void>()
      let notifications = 0
      const job = yield* jobs.start({
        id: "job_claimed_completion",
        type: "test",
        notifyOnComplete: true,
        onComplete: () => Effect.sync(() => notifications++).pipe(Effect.asVoid),
        run: Deferred.await(first).pipe(Effect.as("first")),
      })

      expect(
        yield* jobs.extend({
          id: job.id,
          claimCompletion: true,
          run: Deferred.await(second).pipe(Effect.as("second")),
        }),
      ).toBe(true)
      yield* Deferred.succeed(first, undefined)
      expect((yield* jobs.get(job.id))?.status).toBe("running")
      yield* Deferred.succeed(second, undefined)

      expect((yield* jobs.wait({ id: job.id })).info?.output).toBe("second")
      expect(notifications).toBe(0)
    }).pipe(Effect.provide(jobsLayer)),
  )

  it.live("keeps completion ownership isolated when an extension loses to settlement", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const oldSettling = yield* Deferred.make<void>()
      const releaseOld = yield* Deferred.make<void>()
      let oldNotifications = 0
      let newNotifications = 0
      const id = "job_completion_generations"
      yield* jobs.start({
        id,
        type: "test",
        notifyOnComplete: true,
        onComplete: () =>
          Effect.gen(function* () {
            oldNotifications++
            yield* Deferred.succeed(oldSettling, undefined)
            yield* Deferred.await(releaseOld)
          }),
        run: Effect.succeed("old"),
      })

      yield* Deferred.await(oldSettling)
      expect(yield* jobs.extend({ id, claimCompletion: true, run: Effect.succeed("late") })).toBe(false)

      const replacement = yield* jobs.start({
        id,
        type: "test",
        notifyOnComplete: true,
        onComplete: () => Effect.sync(() => newNotifications++).pipe(Effect.asVoid),
        run: Effect.succeed("new"),
      })
      expect((yield* jobs.wait({ id: replacement.id })).info?.output).toBe("new")
      yield* Deferred.succeed(releaseOld, undefined)

      expect(oldNotifications).toBe(1)
      expect(newNotifications).toBe(1)
    }).pipe(Effect.provide(jobsLayer)),
  )

  it.live("finalizes a generation without waiting for its completion callback", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const callbackStarted = yield* Deferred.make<void>()
      const releaseCallback = yield* Deferred.make<void>()
      const id = "job_blocked_completion"
      yield* jobs.start({
        id,
        type: "test",
        notifyOnComplete: true,
        onComplete: () =>
          Deferred.succeed(callbackStarted, undefined).pipe(Effect.andThen(Deferred.await(releaseCallback))),
        run: Effect.succeed("done"),
      })

      yield* Deferred.await(callbackStarted)
      expect((yield* jobs.wait({ id })).info).toMatchObject({ status: "completed", output: "done" })
      expect((yield* jobs.get(id))?.status).toBe("completed")
      yield* Deferred.succeed(releaseCallback, undefined)
    }).pipe(Effect.provide(jobsLayer)),
  )

  it.live("can enqueue completion delivery before publishing completion", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const callbackStarted = yield* Deferred.make<void>()
      const releaseCallback = yield* Deferred.make<void>()
      const id = "job_awaited_completion"
      yield* jobs.start({
        id,
        type: "test",
        notifyOnComplete: true,
        awaitOnComplete: true,
        onComplete: () =>
          Deferred.succeed(callbackStarted, undefined).pipe(Effect.andThen(Deferred.await(releaseCallback))),
        run: Effect.succeed("done"),
      })

      yield* Deferred.await(callbackStarted)
      expect(yield* jobs.wait({ id, timeout: 0 })).toMatchObject({ timedOut: true, info: { status: "running" } })
      yield* Deferred.succeed(releaseCallback, undefined)
      expect((yield* jobs.wait({ id })).info).toMatchObject({ status: "completed", output: "done" })
    }).pipe(Effect.provide(jobsLayer)),
  )

  it.live("cancels live work without waiting for its completion callback", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const callbackStarted = yield* Deferred.make<void>()
      const releaseCallback = yield* Deferred.make<void>()
      const interrupted = yield* Deferred.make<void>()
      const id = "job_blocked_cancel"
      yield* jobs.start({
        id,
        type: "test",
        notifyOnComplete: true,
        onComplete: () =>
          Deferred.succeed(callbackStarted, undefined).pipe(Effect.andThen(Deferred.await(releaseCallback))),
        run: Effect.never.pipe(Effect.ensuring(Deferred.succeed(interrupted, undefined))),
      })
      const waiter = yield* jobs.wait({ id }).pipe(Effect.forkChild)

      expect((yield* jobs.cancel(id))?.status).toBe("cancelled")
      yield* Deferred.await(callbackStarted)
      expect((yield* Fiber.join(waiter)).info?.status).toBe("cancelled")
      yield* Deferred.await(interrupted).pipe(Effect.timeout("1 second"))
      yield* Deferred.succeed(releaseCallback, undefined)
    }).pipe(Effect.provide(jobsLayer)),
  )

  it.live("does not restart an id until the cancelled generation closes", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const finalizing = yield* Deferred.make<void>()
      const releaseFinalizer = yield* Deferred.make<void>()
      const replacementStarted = yield* Deferred.make<void>()
      const id = "job_cancel_restart"
      yield* jobs.start({
        id,
        type: "test",
        onFinalize: Deferred.succeed(finalizing, undefined).pipe(Effect.andThen(Deferred.await(releaseFinalizer))),
        run: Effect.never,
      })

      const cancelling = yield* jobs.cancel(id).pipe(Effect.forkChild)
      yield* Deferred.await(finalizing)
      const replacement = yield* jobs
        .start({
          id,
          type: "test",
          run: Deferred.succeed(replacementStarted, undefined).pipe(Effect.as("replacement")),
        })
        .pipe(Effect.uninterruptible, Effect.forkChild)
      yield* Effect.yieldNow

      expect(yield* Deferred.isDone(replacementStarted)).toBe(false)
      expect((yield* jobs.get(id))?.status).toBe("running")
      yield* Deferred.succeed(releaseFinalizer, undefined)
      expect((yield* Fiber.join(cancelling))?.status).toBe("cancelled")
      const restarted = yield* Fiber.join(replacement)
      expect((yield* jobs.wait({ id: restarted.id })).info).toMatchObject({
        status: "completed",
        output: "replacement",
      })
    }).pipe(Effect.provide(jobsLayer)),
  )

  it.live("finishes cancellation when the cancelling fiber is interrupted", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const callbackStarted = yield* Deferred.make<void>()
      const workInterrupted = yield* Deferred.make<void>()
      const finalized: string[] = []
      const id = "job_interrupted_cancel"
      yield* jobs.start({
        id,
        type: "test",
        notifyOnComplete: true,
        awaitOnComplete: true,
        onComplete: () => Deferred.succeed(callbackStarted, undefined).pipe(Effect.andThen(Effect.never)),
        onFinalize: Effect.sync(() => finalized.push(id)).pipe(Effect.asVoid),
        run: Effect.never.pipe(Effect.ensuring(Deferred.succeed(workInterrupted, undefined))),
      })
      const waiter = yield* jobs.wait({ id }).pipe(Effect.forkChild)
      const cancelling = yield* jobs.cancel(id).pipe(Effect.forkChild)

      yield* Deferred.await(callbackStarted)
      cancelling.interruptUnsafe()
      expect(Exit.hasInterrupts(yield* Fiber.await(cancelling))).toBe(true)
      expect((yield* Fiber.join(waiter)).info?.status).toBe("cancelled")
      yield* Deferred.await(workInterrupted).pipe(Effect.timeout("1 second"))
      expect(finalized).toEqual([id])
    }).pipe(Effect.provide(jobsLayer)),
  )

  it.live("finalizes when an awaited completion callback interrupts its owner", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const finalized: string[] = []
      const id = "job_interrupted_completion"
      yield* jobs.start({
        id,
        type: "test",
        notifyOnComplete: true,
        awaitOnComplete: true,
        onComplete: () => Effect.interrupt,
        onFinalize: Effect.sync(() => finalized.push(id)).pipe(Effect.asVoid),
        run: Effect.succeed("done"),
      })

      expect((yield* jobs.wait({ id }).pipe(Effect.timeout("1 second"))).info).toMatchObject({
        status: "completed",
        output: "done",
      })
      expect(finalized).toEqual([id])
    }).pipe(Effect.provide(jobsLayer)),
  )

  it.live("keeps a same-id start waiter interruptible during finalization", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const callbackStarted = yield* Deferred.make<void>()
      const releaseCallback = yield* Deferred.make<void>()
      const replacementStarted = yield* Deferred.make<void>()
      const id = "job_interruptible_restart"
      yield* jobs.start({
        id,
        type: "test",
        notifyOnComplete: true,
        awaitOnComplete: true,
        onComplete: () =>
          Deferred.succeed(callbackStarted, undefined).pipe(Effect.andThen(Deferred.await(releaseCallback))),
        run: Effect.succeed("old"),
      })
      yield* Deferred.await(callbackStarted)
      const replacement = yield* jobs
        .start({
          id,
          type: "test",
          run: Deferred.succeed(replacementStarted, undefined).pipe(Effect.as("replacement")),
        })
        .pipe(Effect.forkChild)

      yield* Effect.yieldNow
      replacement.interruptUnsafe()
      expect(Exit.hasInterrupts(yield* Fiber.await(replacement).pipe(Effect.timeout("1 second")))).toBe(true)
      expect(yield* Deferred.isDone(replacementStarted)).toBe(false)
      yield* Deferred.succeed(releaseCallback, undefined)
      expect((yield* jobs.wait({ id })).info?.output).toBe("old")
    }).pipe(Effect.provide(jobsLayer)),
  )

  it.live("finalizes an extension cancelled before its run starts exactly once", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const first = yield* Deferred.make<void>()
      const extensionRan: string[] = []
      const finalized: string[] = []
      const job = yield* jobs.start({
        type: "test",
        run: Deferred.await(first).pipe(Effect.as("first")),
      })
      expect(
        yield* jobs.extend({
          id: job.id,
          onFinalize: Effect.sync(() => finalized.push("extension")).pipe(Effect.asVoid),
          run: Effect.sync(() => extensionRan.push("extension")).pipe(Effect.as("second")),
        }),
      ).toBe(true)

      expect((yield* jobs.cancel(job.id))?.status).toBe("cancelled")
      expect((yield* jobs.cancel(job.id))?.status).toBe("cancelled")
      expect(extensionRan).toEqual([])
      expect(finalized).toEqual(["extension"])
    }).pipe(Effect.provide(jobsLayer)),
  )

  it.live("runs generation and completed extension finalizers exactly once", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const first = yield* Deferred.make<void>()
      const finalized: string[] = []
      const job = yield* jobs.start({
        type: "test",
        onFinalize: Effect.sync(() => finalized.push("generation")).pipe(Effect.asVoid),
        run: Deferred.await(first).pipe(Effect.as("first")),
      })
      expect(
        yield* jobs.extend({
          id: job.id,
          onFinalize: Effect.sync(() => finalized.push("extension")).pipe(Effect.asVoid),
          run: Effect.succeed("second"),
        }),
      ).toBe(true)

      yield* Deferred.succeed(first, undefined)
      expect((yield* jobs.wait({ id: job.id })).info?.output).toBe("second")
      expect(finalized.toSorted()).toEqual(["extension", "generation"])
      expect((yield* jobs.cancel(job.id))?.status).toBe("completed")
      expect(finalized.toSorted()).toEqual(["extension", "generation"])
    }).pipe(Effect.provide(jobsLayer)),
  )

  it.live("interrupts live work without promising settlement after the owning process-local scope closes", () =>
    Effect.gen(function* () {
      const scope = yield* Scope.make()
      const interrupted = yield* Deferred.make<void>()
      const jobs = yield* BackgroundJob.make.pipe(Scope.provide(scope))
      const job = yield* jobs.start({
        type: "test",
        run: Effect.never.pipe(Effect.ensuring(Deferred.succeed(interrupted, undefined))),
      })

      yield* Scope.close(scope, Exit.void)

      yield* Deferred.await(interrupted).pipe(Effect.timeout("1 second"))
      // The abandoned in-memory registry is not a durable observation channel.
      expect((yield* jobs.get(job.id))?.status).toBe("running")
    }),
  )
})
