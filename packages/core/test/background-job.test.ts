import { describe, expect } from "bun:test"
import { BackgroundJob } from "@opencode-ai/core/background-job"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { Deferred, Effect, Exit, Scope } from "effect"
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

  it.live("finalizes a settling generation when its completion callback is interrupted", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const callbackStarted = yield* Deferred.make<void>()
      const id = "job_interrupted_completion"
      yield* jobs.start({
        id,
        type: "test",
        notifyOnComplete: true,
        onComplete: () => Deferred.succeed(callbackStarted, undefined).pipe(Effect.andThen(Effect.never)),
        run: Effect.succeed("done"),
      })

      yield* Deferred.await(callbackStarted)
      expect((yield* jobs.get(id))?.status).toBe("running")
      expect((yield* jobs.cancel(id))?.status).toBe("cancelled")
      expect((yield* jobs.wait({ id })).info?.status).toBe("cancelled")

      yield* jobs.start({ id, type: "test", run: Effect.succeed("replacement") })
      expect((yield* jobs.wait({ id })).info?.output).toBe("replacement")
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
