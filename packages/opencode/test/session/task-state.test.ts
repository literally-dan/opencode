import { afterEach, describe, expect } from "bun:test"
import { Database } from "@opencode-ai/core/database/database"
import { CrossSpawnSpawner } from "@opencode-ai/core/cross-spawn-spawner"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { EventV2 } from "@opencode-ai/core/event"
import { ModelV2 } from "@opencode-ai/core/model"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { SessionProjector } from "@opencode-ai/core/session/projector"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { WorkspaceV2 } from "@opencode-ai/core/workspace"
import { Deferred, Effect, Fiber } from "effect"
import { BackgroundJob } from "@/background/job"
import { EventV2Bridge } from "@/event-v2-bridge"
import { RuntimeFlags } from "@/effect/runtime-flags"
import { Permission } from "@/permission"
import { Question } from "@/question"
import { SessionActivity } from "@/session/activity"
import { MessageID, PartID, SessionID } from "@/session/schema"
import { Session } from "@/session/session"
import { SessionStatus } from "@/session/status"
import { SessionTaskState } from "@/session/task-state"
import { disposeAllInstances } from "../fixture/fixture"
import { pollWithTimeout, testEffect } from "../lib/effect"

const layer = LayerNode.compile(
  LayerNode.group([
    SessionTaskState.node,
    Session.node,
    SessionProjector.node,
    SessionStatus.node,
    SessionActivity.node,
    BackgroundJob.node,
    Permission.node,
    Question.node,
    RuntimeFlags.node,
    Database.node,
    EventV2Bridge.node,
    CrossSpawnSpawner.node,
  ]),
)
const it = testEffect(layer)
const ref = { providerID: ProviderV2.ID.make("test"), modelID: ModelV2.ID.make("model") }
const LocalSessionEvent = EventV2.define({ type: "test.session.local", schema: { sessionID: SessionID } })

afterEach(async () => {
  await disposeAllInstances()
})

describe("session.task-state", () => {
  it.live("publishes non-durable session events without an instance", () =>
    Effect.gen(function* () {
      const events = yield* EventV2Bridge.Service
      yield* SessionActivity.Service
      const event = yield* events.publish(LocalSessionEvent, { sessionID: SessionID.make("ses_no_instance") })
      expect(event.data.sessionID).toBe(SessionID.make("ses_no_instance"))
    }),
  )

  it.instance("persists lifecycle, delivery, and independent ancestor access", () =>
    Effect.gen(function* () {
      const sessions = yield* Session.Service
      const tasks = yield* SessionTaskState.Service
      const root = yield* sessions.create({ title: "Root" })
      const blind = yield* sessions.createTask({ parentID: root.id, title: "Blind" })
      const history = yield* sessions.createTask({ parentID: root.id, title: "History" })
      const status = yield* sessions.createTask({ parentID: root.id, title: "Status" })
      const all = yield* sessions.createTask({ parentID: root.id, title: "All" })

      const blindGeneration = yield* tasks.begin({ sessionID: blind.id, ancestorAccess: "none" })
      expect(blindGeneration).toBe(1)
      expect(yield* tasks.begin({ sessionID: history.id, ancestorAccess: "history" })).toBe(1)
      expect(yield* tasks.begin({ sessionID: status.id, ancestorAccess: "status" })).toBe(1)
      expect(yield* tasks.begin({ sessionID: all.id, ancestorAccess: "all" })).toBe(1)

      expect(yield* tasks.canReadHistory({ sessionID: blind.id, ancestorID: root.id })).toBe(false)
      expect(yield* tasks.canReadHistory({ sessionID: history.id, ancestorID: root.id })).toBe(true)
      expect(yield* tasks.canReadHistory({ sessionID: status.id, ancestorID: root.id })).toBe(false)
      expect(yield* tasks.canReadHistory({ sessionID: all.id, ancestorID: root.id })).toBe(true)
      expect(yield* tasks.grant({ sessionID: blind.id, requested: "all" })).toBe("none")
      expect(yield* tasks.grant({ sessionID: history.id, requested: "all" })).toBe("history")
      expect(yield* tasks.grant({ sessionID: status.id, requested: "all" })).toBe("status")
      expect(yield* tasks.grant({ sessionID: all.id, requested: "all" })).toBe("all")
      expect((yield* tasks.inspect({ sessionID: blind.id, scope: "ancestors" })).allowed).toBe(false)
      expect((yield* tasks.inspect({ sessionID: history.id, scope: "ancestors" })).allowed).toBe(false)
      expect((yield* tasks.inspect({ sessionID: status.id, scope: "ancestors" })).nodes[0]?.taskID).toBe(root.id)
      expect((yield* tasks.inspect({ sessionID: all.id, scope: "ancestors" })).nodes[0]?.taskID).toBe(root.id)

      const detached = yield* tasks.inspect({ sessionID: root.id, scope: "descendants" })
      expect(detached.nodes.find((node) => node.taskID === blind.id)?.execution).toBe("detached")

      yield* tasks.settle({ sessionID: blind.id, generation: blindGeneration ?? 0, status: "completed" })
      let completed = (yield* tasks.inspect({ sessionID: root.id, scope: "descendants" })).nodes.find(
        (node) => node.taskID === blind.id,
      )
      expect(completed).toMatchObject({ execution: "completed", delivery: { status: "pending" } })

      yield* tasks.deliver({ sessionID: blind.id, generation: blindGeneration ?? 0, targetID: root.id })
      completed = (yield* tasks.inspect({ sessionID: root.id, scope: "descendants" })).nodes.find(
        (node) => node.taskID === blind.id,
      )
      expect(completed?.delivery).toMatchObject({ status: "delivered", targetID: root.id })
    }),
  )

  it.instance("hides ancestor failure text from status-only ancestor listing", () =>
    Effect.gen(function* () {
      const sessions = yield* Session.Service
      const tasks = yield* SessionTaskState.Service
      const root = yield* sessions.create({ title: "Root" })
      const parent = yield* sessions.createTask({ parentID: root.id, title: "Parent" })
      const child = yield* sessions.createTask({ parentID: parent.id, title: "Child" })
      const parentGeneration = yield* tasks.begin({ sessionID: parent.id, ancestorAccess: "all" })
      yield* tasks.begin({ sessionID: child.id, ancestorAccess: "status" })
      const error = `Subagent failed (task_id: ${parent.id}): boom\n\nPartial output before the failure:\nprivate transcript`
      yield* tasks.settle({ sessionID: parent.id, generation: parentGeneration ?? 0, status: "error", error })

      const ancestors = yield* tasks.inspect({ sessionID: child.id, scope: "ancestors" })
      expect(ancestors.nodes.map((node) => node.taskID)).toEqual([parent.id, root.id])
      expect(ancestors.nodes[0]).toMatchObject({ execution: "error" })
      expect(ancestors.nodes.map((node) => node.error)).toEqual([undefined, undefined])

      const descendants = yield* tasks.inspect({ sessionID: root.id, scope: "descendants" })
      expect(descendants.nodes[0]?.error).toBe(error)
    }),
  )

  it.instance("rejects stale generations and distinguishes delivering from skipped delivery", () =>
    Effect.gen(function* () {
      const sessions = yield* Session.Service
      const tasks = yield* SessionTaskState.Service
      const jobs = yield* BackgroundJob.Service
      const root = yield* sessions.create({ title: "Root" })
      const child = yield* sessions.createTask({ parentID: root.id, title: "Child" })
      const first = yield* tasks.begin({ sessionID: child.id, ancestorAccess: "history" })
      const firstStartedAt = (yield* tasks.get(child.id))?.time_created ?? 0
      yield* Effect.sleep("2 millis")
      const second = yield* tasks.begin({ sessionID: child.id, ancestorAccess: "history" })
      expect(first).toBe(1)
      expect(second).toBe(2)
      expect((yield* tasks.get(child.id))?.time_created).toBeGreaterThan(firstStartedAt)
      expect(yield* tasks.settle({ sessionID: child.id, generation: first ?? 0, status: "completed" })).toBe(false)
      expect(yield* tasks.deliver({ sessionID: child.id, generation: first ?? 0, targetID: root.id })).toBe(false)

      const done = yield* Deferred.make<void>()
      yield* jobs.start({
        id: child.id,
        type: "task",
        run: Deferred.await(done).pipe(Effect.as("finished")),
      })
      expect(yield* tasks.settle({ sessionID: child.id, generation: second ?? 0, status: "completed" })).toBe(true)
      let node = (yield* tasks.inspect({ sessionID: root.id, scope: "descendants" })).nodes[0]
      expect(node).toMatchObject({ execution: "completed", phase: "delivering", delivery: { status: "pending" } })
      expect((yield* tasks.stop({ sessionID: root.id, taskID: child.id })).stopped).toEqual([])

      yield* Deferred.succeed(done, undefined)
      yield* jobs.wait({ id: child.id })
      node = (yield* tasks.inspect({ sessionID: root.id, scope: "descendants" })).nodes[0]
      expect(node).toMatchObject({ execution: "completed", phase: "idle", delivery: { status: "pending" } })
      expect(yield* tasks.deliver({ sessionID: child.id, generation: second ?? 0, targetID: root.id })).toBe(true)
      expect((yield* tasks.get(child.id))?.delivery_session_id).toBe(root.id)
    }),
  )

  it.instance("keeps completed connectors and infers legacy live tasks", () =>
    Effect.gen(function* () {
      const sessions = yield* Session.Service
      const tasks = yield* SessionTaskState.Service
      const jobs = yield* BackgroundJob.Service
      const root = yield* sessions.create({ title: "Root" })
      const connector = yield* sessions.createTask({ parentID: root.id, title: "Connector" })
      const grandchild = yield* sessions.createTask({ parentID: connector.id, title: "Grandchild" })
      const legacy = yield* sessions.createTask({ parentID: root.id, title: "Legacy" })
      const connectorGeneration = yield* tasks.begin({ sessionID: connector.id, ancestorAccess: "history" })
      yield* tasks.begin({ sessionID: grandchild.id, ancestorAccess: "history" })
      yield* tasks.settle({
        sessionID: connector.id,
        generation: connectorGeneration ?? 0,
        status: "completed",
      })
      yield* jobs.start({ id: grandchild.id, type: "task", run: Effect.never })
      yield* jobs.start({ id: legacy.id, type: "task", run: Effect.never })

      const tree = yield* tasks.inspect({ sessionID: root.id, scope: "descendants" })
      const connectorNode = tree.nodes.find((node) => node.taskID === connector.id)
      expect(connectorNode?.execution).toBe("completed")
      expect(connectorNode?.children[0]).toMatchObject({ taskID: grandchild.id, execution: "running" })
      expect(tree.nodes.find((node) => node.taskID === legacy.id)).toMatchObject({
        execution: "running",
        delivery: { status: "unknown" },
      })
      yield* jobs.cancel(grandchild.id)
      yield* jobs.cancel(legacy.id)
    }),
  )

  it.instance("serializes stop with a fresh task handoff", () =>
    Effect.gen(function* () {
      const sessions = yield* Session.Service
      const tasks = yield* SessionTaskState.Service
      const jobs = yield* BackgroundJob.Service
      const root = yield* sessions.create({ title: "Root" })
      const child = yield* sessions.createTask({ parentID: root.id, title: "Child" })
      const began = yield* Deferred.make<number>()
      const releaseStart = yield* Deferred.make<void>()
      const handoff = yield* tasks
        .withLock(child.id)(
          Effect.gen(function* () {
            const generation = yield* tasks.begin({ sessionID: child.id, ancestorAccess: "history" })
            if (!generation) return
            yield* Deferred.succeed(began, generation)
            yield* Deferred.await(releaseStart)
            yield* jobs.start({ id: child.id, type: "task", run: Effect.never })
          }),
        )
        .pipe(Effect.forkChild)

      yield* Deferred.await(began)
      expect((yield* tasks.inspect({ sessionID: root.id, scope: "descendants" })).nodes[0]?.execution).toBe("detached")
      const stopping = yield* tasks.stop({ sessionID: root.id, taskID: child.id }).pipe(Effect.forkChild)
      yield* Effect.yieldNow
      expect(stopping.pollUnsafe()).toBeUndefined()

      yield* Deferred.succeed(releaseStart, undefined)
      yield* Fiber.join(handoff)
      expect((yield* Fiber.join(stopping)).stopped).toEqual([child.id])
      expect((yield* jobs.get(child.id))?.status).toBe("cancelled")
      expect((yield* tasks.get(child.id))?.status).toBe("cancelled")
    }),
  )

  it.instance("shows nested live activity and stops only the selected subtree", () =>
    Effect.gen(function* () {
      const sessions = yield* Session.Service
      const tasks = yield* SessionTaskState.Service
      const jobs = yield* BackgroundJob.Service
      const activity = yield* SessionActivity.Service
      const root = yield* sessions.create({ title: "Root" })
      const child = yield* sessions.createTask({ parentID: root.id, title: "Child" })
      const grandchild = yield* sessions.createTask({ parentID: child.id, title: "Grandchild" })
      const sibling = yield* sessions.createTask({ parentID: root.id, title: "Sibling" })
      const otherRoot = yield* sessions.create({ title: "Other root" })
      const unrelated = yield* sessions.createTask({ parentID: otherRoot.id, title: "Unrelated" })

      yield* tasks.begin({ sessionID: child.id, ancestorAccess: "history" })
      yield* tasks.begin({ sessionID: grandchild.id, ancestorAccess: "history" })
      yield* tasks.begin({ sessionID: sibling.id, ancestorAccess: "history" })
      yield* tasks.begin({ sessionID: unrelated.id, ancestorAccess: "history" })
      yield* Effect.forEach(
        [child, grandchild, sibling, unrelated],
        (session) => jobs.start({ id: session.id, type: "task", title: session.title, run: Effect.never }),
        { discard: true },
      )
      const toolStartedAt = Date.now()
      const tool = yield* runningTool(grandchild, toolStartedAt)
      const activityBeforeDelta = (yield* activity.list()).get(grandchild.id) ?? 0
      yield* Effect.sleep("2 millis")
      yield* sessions.updatePartDelta({
        sessionID: grandchild.id,
        messageID: tool.messageID,
        partID: tool.id,
        field: "output",
        delta: "progress",
      })
      expect((yield* activity.list()).get(grandchild.id)).toBeGreaterThan(activityBeforeDelta)

      const tree = yield* tasks.inspect({ sessionID: root.id, scope: "descendants" })
      const childNode = tree.nodes.find((node) => node.taskID === child.id)
      const grandchildNode = childNode?.children[0]
      expect(childNode?.execution).toBe("running")
      expect(grandchildNode).toMatchObject({
        taskID: grandchild.id,
        execution: "running",
        phase: "tool",
        currentTool: { name: "bash", title: "Run checks", startedAt: toolStartedAt },
      })
      expect(grandchildNode?.lastActivityAt).toBeGreaterThanOrEqual(toolStartedAt)
      expect(tree.nodes.some((node) => node.taskID === unrelated.id)).toBe(false)

      const childTree = yield* tasks.inspect({ sessionID: child.id, scope: "descendants" })
      expect(childTree.nodes.map((node) => node.taskID)).toEqual([grandchild.id])

      expect((yield* tasks.stop({ sessionID: root.id, taskID: child.id })).stopped).toEqual([child.id, grandchild.id])
      expect((yield* jobs.get(child.id))?.status).toBe("cancelled")
      expect((yield* jobs.get(grandchild.id))?.status).toBe("cancelled")
      expect((yield* jobs.get(sibling.id))?.status).toBe("running")
      expect((yield* jobs.get(unrelated.id))?.status).toBe("running")
      expect((yield* tasks.get(child.id))?.status).toBe("cancelled")
      expect((yield* tasks.get(grandchild.id))?.status).toBe("cancelled")
      yield* jobs.cancel(sibling.id)
      yield* jobs.cancel(unrelated.id)
    }),
  )

  it.instance("keeps task provenance after the root moves and does not stop work it cannot observe", () =>
    Effect.gen(function* () {
      const sessions = yield* Session.Service
      const tasks = yield* SessionTaskState.Service
      const root = yield* sessions.create({ title: "Root" })
      const child = yield* sessions.createTask({ parentID: root.id, title: "Child" })
      const generation = yield* tasks.begin({ sessionID: child.id, ancestorAccess: "all" })
      yield* sessions.setWorkspace({ sessionID: root.id, workspaceID: WorkspaceV2.ID.create() })

      const tree = yield* tasks.inspect({ sessionID: root.id, scope: "descendants" })
      expect(tree.nodes.map((node) => node.taskID)).toEqual([child.id])
      // The child job runs in the child's location, so this location cannot tell whether it is detached.
      expect(tree.nodes[0]).toMatchObject({ execution: "unknown", phase: "unknown" })
      expect(yield* tasks.canReadHistory({ sessionID: child.id, ancestorID: root.id })).toBe(true)
      expect(
        (yield* tasks.inspect({ sessionID: child.id, scope: "ancestors" })).nodes.map((node) => node.taskID),
      ).toEqual([root.id])

      expect(yield* tasks.stop({ sessionID: root.id, taskID: child.id })).toEqual({
        stopped: [],
        unavailable: [child.id],
      })
      expect(yield* tasks.get(child.id)).toMatchObject({ generation, status: "running" })
    }),
  )

  it.instance("stops a moved task child whose job runs in this location", () =>
    Effect.gen(function* () {
      const sessions = yield* Session.Service
      const tasks = yield* SessionTaskState.Service
      const jobs = yield* BackgroundJob.Service
      const root = yield* sessions.create({ title: "Root" })
      const child = yield* sessions.createTask({ parentID: root.id, title: "Child" })
      yield* tasks.begin({ sessionID: child.id, ancestorAccess: "history" })
      yield* jobs.start({ id: child.id, type: "task", run: Effect.never })
      yield* sessions.setWorkspace({ sessionID: child.id, workspaceID: WorkspaceV2.ID.create() })

      expect((yield* tasks.inspect({ sessionID: root.id, scope: "descendants" })).nodes[0]).toMatchObject({
        taskID: child.id,
        execution: "running",
      })
      expect(yield* tasks.stop({ sessionID: root.id, taskID: child.id })).toEqual({
        stopped: [child.id],
        unavailable: [],
      })
      expect((yield* jobs.wait({ id: child.id, timeout: 1_000 })).info?.status).toBe("cancelled")
    }),
  )

  it.instance("lists a task that waits on its own running child as idle", () =>
    Effect.gen(function* () {
      const sessions = yield* Session.Service
      const tasks = yield* SessionTaskState.Service
      const jobs = yield* BackgroundJob.Service
      const root = yield* sessions.create({ title: "Root" })
      const coordinator = yield* sessions.createTask({ parentID: root.id, title: "Coordinator" })
      const worker = yield* sessions.createTask({ parentID: coordinator.id, title: "Worker" })
      yield* tasks.begin({ sessionID: coordinator.id, ancestorAccess: "history" })
      yield* tasks.begin({ sessionID: worker.id, ancestorAccess: "history" })
      yield* jobs.start({ id: coordinator.id, type: "task", run: Effect.never })
      // The coordinator turn has ended and its worker still runs.
      yield* finishedTurn(coordinator, Date.now() - 1_000)
      yield* jobs.start({
        id: worker.id,
        type: "task",
        metadata: { parentSessionId: coordinator.id, sessionId: worker.id },
        run: Effect.never,
      })

      const tree = yield* tasks.inspect({ sessionID: root.id, scope: "descendants" })
      expect(tree.nodes[0]).toMatchObject({ taskID: coordinator.id, execution: "running", phase: "idle" })
      expect(tree.nodes[0]?.children[0]).toMatchObject({ taskID: worker.id, execution: "running" })

      yield* jobs.cancel(worker.id)
      expect((yield* tasks.inspect({ sessionID: root.id, scope: "descendants" })).nodes[0]?.phase).toBe("delivering")
    }),
  )

  it.instance("reports permission, question, and retry blockers", () =>
    Effect.gen(function* () {
      const sessions = yield* Session.Service
      const tasks = yield* SessionTaskState.Service
      const jobs = yield* BackgroundJob.Service
      const permissions = yield* Permission.Service
      const questions = yield* Question.Service
      const statuses = yield* SessionStatus.Service
      const root = yield* sessions.create({ title: "Root" })
      const child = yield* sessions.createTask({ parentID: root.id, title: "Child" })
      yield* tasks.begin({ sessionID: child.id, ancestorAccess: "history" })
      yield* jobs.start({ id: child.id, type: "task", run: Effect.never })

      const permissionFiber = yield* permissions
        .ask({
          sessionID: child.id,
          permission: "bash",
          patterns: ["bun test"],
          metadata: {},
          always: [],
          ruleset: [],
        })
        .pipe(Effect.forkChild)
      const permission = yield* pollWithTimeout(
        permissions.list().pipe(Effect.map((items) => items[0])),
        "permission was not pending",
      )
      expect((yield* tasks.inspect({ sessionID: root.id, scope: "descendants" })).nodes[0]?.phase).toBe("permission")
      yield* permissions.reply({ requestID: permission.id, reply: "reject" })
      yield* Fiber.await(permissionFiber)

      const questionFiber = yield* questions
        .ask({
          sessionID: child.id,
          questions: [{ question: "Continue?", header: "Continue", options: [] }],
        })
        .pipe(Effect.forkChild)
      const question = yield* pollWithTimeout(
        questions.list().pipe(Effect.map((items) => items[0])),
        "question was not pending",
      )
      expect((yield* tasks.inspect({ sessionID: root.id, scope: "descendants" })).nodes[0]?.phase).toBe("question")
      yield* questions.reject(question.id)
      yield* Fiber.await(questionFiber)

      yield* statuses.set(child.id, { type: "retry", attempt: 1, message: "retrying", next: Date.now() + 1_000 })
      expect((yield* tasks.inspect({ sessionID: root.id, scope: "descendants" })).nodes[0]?.phase).toBe("retry")
    }),
  )
})

const finishedTurn = Effect.fn("SessionTaskStateTest.finishedTurn")(function* (
  session: Session.Info,
  startedAt: number,
) {
  const sessions = yield* Session.Service
  const user = yield* sessions.updateMessage({
    id: MessageID.ascending(),
    role: "user",
    sessionID: session.id,
    agent: "general",
    model: ref,
    time: { created: startedAt - 1 },
  })
  yield* sessions.updateMessage({
    id: MessageID.ascending(),
    role: "assistant",
    parentID: user.id,
    sessionID: session.id,
    mode: "general",
    agent: "general",
    cost: 0,
    path: { cwd: session.directory, root: session.directory },
    tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
    modelID: ref.modelID,
    providerID: ref.providerID,
    time: { created: startedAt, completed: startedAt + 1 },
    finish: "stop",
  })
})

const runningTool = Effect.fn("SessionTaskStateTest.runningTool")(function* (session: Session.Info, startedAt: number) {
  const sessions = yield* Session.Service
  const user = yield* sessions.updateMessage({
    id: MessageID.ascending(),
    role: "user",
    sessionID: session.id,
    agent: "general",
    model: ref,
    time: { created: startedAt - 1 },
  })
  const assistant: SessionV1.Assistant = {
    id: MessageID.ascending(),
    role: "assistant",
    parentID: user.id,
    sessionID: session.id,
    mode: "general",
    agent: "general",
    cost: 0,
    path: { cwd: session.directory, root: session.directory },
    tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
    modelID: ref.modelID,
    providerID: ref.providerID,
    time: { created: startedAt },
  }
  yield* sessions.updateMessage(assistant)
  return yield* sessions.updatePart({
    id: PartID.ascending(),
    sessionID: session.id,
    messageID: assistant.id,
    type: "tool",
    callID: "call-checks",
    tool: "bash",
    state: {
      status: "running",
      input: { command: "bun test" },
      title: "Run checks",
      metadata: {},
      time: { start: startedAt },
    },
  })
})
