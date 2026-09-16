import { describe, expect } from "bun:test"
import { Effect, Layer } from "effect"
import { MessageID, SessionID } from "@/session/schema"
import { SessionTaskState } from "@/session/task-state"
import { make } from "@/tool/task_control"
import { testEffect } from "../lib/effect"

const it = testEffect(Layer.empty)
const root = sessionID(1)
const child = sessionID(2)
const grandchild = sessionID(3)

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
  it.effect("renders the canonical nested task tree", () =>
    Effect.gen(function* () {
      const def = make(
        state({
          allowed: true,
          scope: "descendants",
          nodes: [
            node(child, {
              title: "Child task",
              phase: "model",
              children: [
                node(grandchild, {
                  title: "Grandchild task",
                  phase: "tool",
                  currentTool: { name: "bash", title: "Run checks", startedAt: 1 },
                }),
              ],
            }),
          ],
        }),
      )

      const result = yield* def.execute({ action: "list" }, context)

      expect(result.metadata.count).toBe(2)
      expect(result.metadata.tasks[0]?.children[0]?.taskID).toBe(grandchild)
      expect(result.output).toContain(`${child} [running/model`)
      expect(result.output).toContain(`${grandchild} [running/tool`)
      expect(result.output).toContain("current tool: bash (Run checks)")
    }),
  )

  it.effect("renders shortened errors for failed and cancelled tasks", () =>
    Effect.gen(function* () {
      const def = make(
        state({
          allowed: true,
          scope: "descendants",
          nodes: [
            node(child, {
              title: "Failed task",
              execution: "error",
              phase: "idle",
              error: `Subagent failed (task_id: ${child}): The subagent stopped.\n\nPartial output before the failure:\n${"x".repeat(300)}`,
            }),
            node(grandchild, {
              title: "Cancelled task",
              execution: "cancelled",
              phase: "idle",
              error: "Task cancelled",
            }),
            node(sessionID(4), { title: "Completed task", execution: "completed", phase: "idle", error: "stale" }),
          ],
        }),
      )

      const result = yield* def.execute({ action: "list" }, context)
      const lines = result.output.split("\n")

      expect(lines[2]).toBe(
        `  error: Subagent failed (task_id: ${child}): The subagent stopped. Partial output before the failure: ${"x".repeat(83)}…`,
      )
      expect(lines[3]).toContain(`${grandchild} [cancelled/idle`)
      expect(lines[4]).toBe("  error: Task cancelled")
      expect(lines[5]).toContain(`${sessionID(4)} [completed/idle`)
      expect(lines).toHaveLength(6)
    }),
  )

  it.effect("reports denied ancestor visibility", () =>
    Effect.gen(function* () {
      const def = make(state({ allowed: false, scope: "ancestors", nodes: [] }))

      const result = yield* def.execute({ action: "list", scope: "ancestors" }, context)

      expect(result.metadata).toMatchObject({ count: 0, scope: "ancestors" })
      expect(result.output).toContain("not granted")
    }),
  )

  it.effect("delegates descendant cancellation", () =>
    Effect.gen(function* () {
      let stopped: { sessionID: SessionID; taskID: SessionID } | undefined
      const def = make(
        state({ allowed: true, scope: "descendants", nodes: [node(child)] }, (input) => {
          stopped = input
          return Effect.succeed({ stopped: [child, grandchild], unavailable: [] })
        }),
      )

      const result = yield* def.execute({ action: "stop", task_id: child }, context)

      expect(stopped).toEqual({ sessionID: root, taskID: child })
      expect(result.metadata).toMatchObject({ taskId: child, stopped: true, count: 2 })
      expect(result.output).toContain("and 1 active descendant")
    }),
  )

  it.effect("reports tasks it cannot stop from another location", () =>
    Effect.gen(function* () {
      const def = make(
        state({ allowed: true, scope: "descendants", nodes: [node(child, { execution: "unknown" })] }, () =>
          Effect.succeed({ stopped: [], unavailable: [child] }),
        ),
      )

      const result = yield* def.execute({ action: "stop", task_id: child }, context)

      expect(result.title).toBe("Task in another location")
      expect(result.metadata.stopped).toBe(false)
      expect(result.output).toBe(
        `Task ${child} was not stopped. 1 unfinished task in another location could not be observed or stopped from here: ${child}.`,
      )
    }),
  )

  it.effect("never delegates an ancestor stop", () =>
    Effect.gen(function* () {
      let called = false
      const def = make(
        state({ allowed: true, scope: "ancestors", nodes: [node(child)] }, () => {
          called = true
          return Effect.succeed({ stopped: [child], unavailable: [] })
        }),
      )

      const result = yield* def.execute({ action: "stop", task_id: child, scope: "ancestors" }, context)

      expect(called).toBe(false)
      expect(result.metadata.stopped).toBe(false)
      expect(result.output).toContain("read-only")
    }),
  )
})

function state(
  inspection: SessionTaskState.Inspection,
  stop: SessionTaskState.Interface["stop"] = () => Effect.succeed({ stopped: [], unavailable: [] }),
): SessionTaskState.Interface {
  return {
    withLock: () => (effect) => effect,
    begin: () => Effect.succeed(1),
    ensure: () => Effect.succeed(1),
    settle: () => Effect.succeed(true),
    deliver: () => Effect.succeed(true),
    grant: (input) => Effect.succeed(input.requested),
    get: () => Effect.succeed(undefined),
    inspect: () => Effect.succeed(inspection),
    stop,
    canReadHistory: () => Effect.succeed(false),
  }
}

function node(taskID: SessionID, overrides: Partial<SessionTaskState.Node> = {}): SessionTaskState.Node {
  return {
    taskID,
    title: "Task",
    execution: "running",
    phase: "starting",
    startedAt: 1,
    lastActivityAt: 2,
    quietForMs: 3,
    delivery: { status: "pending" },
    children: [],
    ...overrides,
  }
}

function sessionID(index: number) {
  return SessionID.make(`ses_${index.toString(16).padStart(12, "0")}${index.toString().padStart(14, "0")}`)
}
