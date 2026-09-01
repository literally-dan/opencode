import { describe, expect } from "bun:test"
import { BackgroundJob } from "@opencode-ai/core/background-job"
import { Deferred, Effect, Layer } from "effect"
import { MessageID, SessionID } from "@/session/schema"
import { make } from "@/tool/task_control"
import { testEffect } from "../lib/effect"

const it = testEffect(Layer.empty)

const root = SessionID.make("ses_root")
const child = SessionID.make("ses_child")
const grandchild = SessionID.make("ses_grandchild")
const sibling = SessionID.make("ses_sibling")
const unrelated = SessionID.make("ses_unrelated")

const context = {
  sessionID: root,
  messageID: MessageID.make("msg_test"),
  agent: "build",
  abort: new AbortController().signal,
  messages: [],
  metadata: () => Effect.void,
  ask: () => Effect.void,
}

describe("tool.task_control", () => {
  it.instance("lists only running Task descendants", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.make
      yield* jobs.start({
        id: child,
        type: "task",
        title: "Child task",
        metadata: { parentSessionId: root, sessionId: child },
        run: Effect.never,
      })
      yield* jobs.start({
        id: grandchild,
        type: "task",
        title: "Grandchild task",
        metadata: { parentSessionId: child, sessionId: grandchild },
        run: Effect.never,
      })
      yield* jobs.start({
        id: unrelated,
        type: "task",
        title: "Unrelated task",
        metadata: { parentSessionId: SessionID.make("ses_other"), sessionId: unrelated },
        run: Effect.never,
      })
      yield* jobs.start({
        id: "job_other",
        type: "other",
        title: "Other job",
        metadata: { parentSessionId: root, sessionId: "job_other" },
        run: Effect.never,
      })
      const def = make(jobs)

      const result = yield* def.execute({ action: "list" }, context)

      expect(result.metadata.count).toBe(2)
      expect(result.output).toContain(`${child}: Child task`)
      expect(result.output).toContain(`${grandchild}: Grandchild task`)
      expect(result.output).not.toContain("Unrelated task")
      expect(result.output).not.toContain("Other job")
    }),
  )

  it.instance("stops one Task and its descendants without cancelling unrelated work", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.make
      const childStarted = yield* Deferred.make<void>()
      const grandchildStarted = yield* Deferred.make<void>()
      yield* jobs.start({
        id: child,
        type: "task",
        metadata: { parentSessionId: root, sessionId: child },
        run: Deferred.succeed(childStarted, undefined).pipe(Effect.andThen(Effect.never)),
      })
      yield* jobs.start({
        id: grandchild,
        type: "task",
        metadata: { parentSessionId: child, sessionId: grandchild },
        run: Deferred.succeed(grandchildStarted, undefined).pipe(Effect.andThen(Effect.never)),
      })
      yield* jobs.start({
        id: unrelated,
        type: "task",
        metadata: { parentSessionId: SessionID.make("ses_other"), sessionId: unrelated },
        run: Effect.never,
      })
      yield* Deferred.await(childStarted)
      yield* Deferred.await(grandchildStarted)
      const def = make(jobs)

      const result = yield* def.execute({ action: "stop", task_id: child }, context)

      expect(result.metadata).toMatchObject({ taskId: child, stopped: true, count: 2 })
      expect((yield* jobs.get(child))?.status).toBe("cancelled")
      expect((yield* jobs.get(grandchild))?.status).toBe("cancelled")
      expect((yield* jobs.get(unrelated))?.status).toBe("running")
    }),
  )

  it.instance("stops a running descendant after its parent Task completes", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.make
      yield* jobs.start({
        id: child,
        type: "task",
        metadata: { parentSessionId: root, sessionId: child, ancestorSessionIds: [root] },
        run: Effect.succeed("done"),
      })
      yield* jobs.wait({ id: child })
      yield* jobs.start({
        id: grandchild,
        type: "task",
        title: "Orphaned grandchild",
        metadata: { parentSessionId: child, sessionId: grandchild, ancestorSessionIds: [child, root] },
        run: Effect.never,
      })
      const def = make(jobs)

      const listed = yield* def.execute({ action: "list" }, context)
      expect(listed.metadata.count).toBe(1)
      expect(listed.output).toContain(`${grandchild}: Orphaned grandchild`)

      const stopped = yield* def.execute({ action: "stop", task_id: grandchild }, context)
      expect(stopped.metadata).toMatchObject({ taskId: grandchild, stopped: true, count: 1 })
      expect((yield* jobs.get(grandchild))?.status).toBe("cancelled")
    }),
  )

  it.instance("uses legacy metadata through a completed intermediate Task", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.make
      yield* jobs.start({
        id: child,
        type: "task",
        metadata: { parentSessionId: root, sessionId: child },
        run: Effect.succeed("done"),
      })
      yield* jobs.wait({ id: child })
      yield* jobs.start({
        id: grandchild,
        type: "task",
        title: "Legacy grandchild",
        metadata: { parentSessionId: child, sessionId: grandchild },
        run: Effect.never,
      })
      const def = make(jobs)

      const listed = yield* def.execute({ action: "list" }, context)
      expect(listed.output).toContain(`${grandchild}: Legacy grandchild`)

      const stopped = yield* def.execute({ action: "stop", task_id: grandchild }, context)
      expect(stopped.metadata).toMatchObject({ taskId: grandchild, stopped: true, count: 1 })
      expect((yield* jobs.get(grandchild))?.status).toBe("cancelled")
    }),
  )

  it.instance("does not expose a sibling Task to a child session", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.make
      yield* jobs.start({
        id: grandchild,
        type: "task",
        title: "Owned grandchild",
        metadata: { parentSessionId: child, sessionId: grandchild, ancestorSessionIds: [child, root] },
        run: Effect.never,
      })
      yield* jobs.start({
        id: sibling,
        type: "task",
        title: "Sibling task",
        metadata: { parentSessionId: root, sessionId: sibling, ancestorSessionIds: [root] },
        run: Effect.never,
      })
      const def = make(jobs)
      const childContext = { ...context, sessionID: child }

      const listed = yield* def.execute({ action: "list" }, childContext)
      expect(listed.metadata.count).toBe(1)
      expect(listed.output).toContain(`${grandchild}: Owned grandchild`)
      expect(listed.output).not.toContain("Sibling task")

      const stopped = yield* def.execute({ action: "stop", task_id: sibling }, childContext)
      expect(stopped.metadata).toMatchObject({ taskId: sibling, stopped: false })
      expect((yield* jobs.get(sibling))?.status).toBe("running")
    }),
  )

  it.instance("refuses to stop an unrelated Task", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.make
      yield* jobs.start({
        id: unrelated,
        type: "task",
        metadata: { parentSessionId: SessionID.make("ses_other"), sessionId: unrelated },
        run: Effect.never,
      })
      const def = make(jobs)

      const result = yield* def.execute({ action: "stop", task_id: unrelated }, context)

      expect(result.metadata).toMatchObject({ taskId: unrelated, stopped: false })
      expect((yield* jobs.get(unrelated))?.status).toBe("running")
    }),
  )
})
