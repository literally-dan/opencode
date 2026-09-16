import { BackgroundJob } from "@/background/job"
import { Permission } from "@/permission"
import { Question } from "@/question"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { KeyedMutex } from "@opencode-ai/core/effect/keyed-mutex"
import { Database } from "@opencode-ai/core/database/database"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import {
  MessageTable,
  PartTable,
  SessionTable,
  SessionMessageTable,
  SessionTaskTable,
} from "@opencode-ai/core/session/sql"
import { and, eq, inArray, max, sql } from "drizzle-orm"
import { Context, Effect, Layer, Schema, Scope } from "effect"
import { SessionActivity } from "./activity"
import { SessionID } from "./schema"
import { Session } from "./session"
import { SessionStatus } from "./status"

export const AncestorAccess = Schema.Literals(["none", "history", "status", "all"])
export type AncestorAccess = typeof AncestorAccess.Type

export type Execution = "running" | "completed" | "error" | "cancelled" | "detached" | "unknown"
export type Phase =
  | "starting"
  | "model"
  | "tool"
  | "retry"
  | "permission"
  | "question"
  | "delivering"
  | "idle"
  | "unknown"

export type Node = {
  taskID: SessionID
  parentID?: SessionID
  agent?: string
  title: string
  execution: Execution
  phase: Phase
  currentTool?: {
    name: string
    title?: string
    startedAt: number
  }
  startedAt: number
  lastActivityAt: number
  quietForMs: number
  completedAt?: number
  error?: string
  delivery: {
    status: "pending" | "delivered" | "unknown"
    targetID?: SessionID
    deliveredAt?: number
  }
  children: Node[]
}

type Lifecycle = typeof SessionTaskTable.$inferSelect

export type Inspection = {
  allowed: boolean
  scope: "descendants" | "ancestors"
  nodes: Node[]
}

export type StopResult = {
  stopped: SessionID[]
  /** Unfinished Tasks whose job can run only in another location, so this location cannot stop them. */
  unavailable: SessionID[]
}

export interface Interface {
  readonly withLock: KeyedMutex.KeyedMutex<SessionID>["withLock"]
  readonly begin: (input: { sessionID: SessionID; ancestorAccess: AncestorAccess }) => Effect.Effect<number | undefined>
  readonly ensure: (input: {
    sessionID: SessionID
    ancestorAccess: AncestorAccess
  }) => Effect.Effect<number | undefined>
  readonly settle: (input: {
    sessionID: SessionID
    generation: number
    status: "completed" | "error" | "cancelled"
    error?: string
  }) => Effect.Effect<boolean>
  readonly deliver: (input: { sessionID: SessionID; generation: number; targetID: SessionID }) => Effect.Effect<boolean>
  readonly grant: (input: { sessionID: SessionID; requested: AncestorAccess }) => Effect.Effect<AncestorAccess>
  readonly get: (sessionID: SessionID) => Effect.Effect<Lifecycle | undefined>
  readonly inspect: (input: { sessionID: SessionID; scope: "descendants" | "ancestors" }) => Effect.Effect<Inspection>
  readonly stop: (input: { sessionID: SessionID; taskID: SessionID }) => Effect.Effect<StopResult>
  readonly canReadHistory: (input: { sessionID: SessionID; ancestorID: SessionID }) => Effect.Effect<boolean>
}

export class Service extends Context.Service<Service, Interface>()("@opencode/SessionTaskState") {}

// Location is not compared: moving a Session does not move its Task children.
export function isTaskChild(parent: Session.Info, child: Session.Info) {
  return child.parentID === parent.id && child.taskParentID === parent.id && child.projectID === parent.projectID
}

const layer = Layer.effect(
  Service,
  Effect.gen(function* () {
    const { db } = yield* Database.Service
    const sessions = yield* Session.Service
    const background = yield* BackgroundJob.Service
    const statuses = yield* SessionStatus.Service
    const activity = yield* SessionActivity.Service
    const permissions = yield* Permission.Service
    const questions = yield* Question.Service
    const scope = yield* Scope.Scope
    const mutex = KeyedMutex.makeUnsafe<SessionID>()
    const withLock = mutex.withLock

    const get = Effect.fn("SessionTaskState.get")(function* (sessionID: SessionID) {
      return yield* db
        .select()
        .from(SessionTaskTable)
        .where(eq(SessionTaskTable.session_id, sessionID))
        .get()
        .pipe(Effect.orDie)
    })

    const begin = Effect.fn("SessionTaskState.begin")(function* (input: {
      sessionID: SessionID
      ancestorAccess: AncestorAccess
    }) {
      const exists = yield* db
        .select({ id: SessionTable.id })
        .from(SessionTable)
        .where(eq(SessionTable.id, input.sessionID))
        .get()
        .pipe(Effect.orDie)
      if (!exists) return undefined
      const now = Date.now()
      const row = yield* db
        .insert(SessionTaskTable)
        .values({
          session_id: input.sessionID,
          ancestor_access: input.ancestorAccess,
          generation: 1,
          status: "running",
          time_created: now,
          time_updated: now,
        })
        .onConflictDoUpdate({
          target: SessionTaskTable.session_id,
          set: {
            ancestor_access: input.ancestorAccess,
            generation: sql`${SessionTaskTable.generation} + 1`,
            status: "running",
            error: null,
            time_created: now,
            time_updated: now,
            time_completed: null,
            delivery_session_id: null,
            time_delivered: null,
          },
        })
        .returning({ generation: SessionTaskTable.generation })
        .get()
        .pipe(Effect.orDie)
      yield* activity.touch(input.sessionID, now)
      return row?.generation
    })

    const ensure = Effect.fn("SessionTaskState.ensure")(function* (input: {
      sessionID: SessionID
      ancestorAccess: AncestorAccess
    }) {
      const lifecycle = yield* get(input.sessionID)
      if (!lifecycle) return yield* begin(input)
      if (lifecycle.ancestor_access === input.ancestorAccess) return lifecycle.generation
      yield* db
        .update(SessionTaskTable)
        .set({ ancestor_access: input.ancestorAccess, time_updated: Date.now() })
        .where(eq(SessionTaskTable.session_id, input.sessionID))
        .run()
        .pipe(Effect.orDie)
      return lifecycle.generation
    })

    const settle = Effect.fn("SessionTaskState.settle")(function* (input: {
      sessionID: SessionID
      generation: number
      status: "completed" | "error" | "cancelled"
      error?: string
    }) {
      const exists = yield* db
        .select({ id: SessionTable.id })
        .from(SessionTable)
        .where(eq(SessionTable.id, input.sessionID))
        .get()
        .pipe(Effect.orDie)
      if (!exists) return false
      const now = Date.now()
      const row = yield* db
        .update(SessionTaskTable)
        .set({
          status: input.status,
          error: input.error ?? null,
          time_updated: now,
          time_completed: now,
        })
        .where(
          and(
            eq(SessionTaskTable.session_id, input.sessionID),
            eq(SessionTaskTable.generation, input.generation),
            eq(SessionTaskTable.status, "running"),
          ),
        )
        .returning({ sessionID: SessionTaskTable.session_id })
        .get()
        .pipe(Effect.orDie)
      if (row) yield* activity.touch(input.sessionID, now)
      return !!row
    })

    const deliver = Effect.fn("SessionTaskState.deliver")(function* (input: {
      sessionID: SessionID
      generation: number
      targetID: SessionID
    }) {
      const now = Date.now()
      const row = yield* db
        .update(SessionTaskTable)
        .set({ delivery_session_id: input.targetID, time_delivered: now, time_updated: now })
        .where(and(eq(SessionTaskTable.session_id, input.sessionID), eq(SessionTaskTable.generation, input.generation)))
        .returning({ sessionID: SessionTaskTable.session_id })
        .get()
        .pipe(Effect.orDie)
      if (row) yield* activity.touch(input.sessionID, now)
      return !!row
    })

    const grant = Effect.fn("SessionTaskState.grant")(function* (input: {
      sessionID: SessionID
      requested: AncestorAccess
    }) {
      const lifecycle = yield* get(input.sessionID)
      if (lifecycle) return intersectAccess(lifecycle.ancestor_access, input.requested)
      const session = yield* sessions.get(input.sessionID).pipe(Effect.catch(() => Effect.succeed(undefined)))
      return session?.taskParentID ? intersectAccess("history", input.requested) : input.requested
    })

    const descendants = Effect.fn("SessionTaskState.descendants")(function* (root: Session.Info) {
      const result: Session.Info[] = []
      const queue = [root]
      const seen = new Set<SessionID>([root.id])
      while (queue.length) {
        const parent = queue.shift()
        if (!parent) break
        const children = yield* sessions.children(parent.id)
        for (const child of children) {
          if (seen.has(child.id) || !isTaskChild(parent, child)) continue
          seen.add(child.id)
          result.push(child)
          queue.push(child)
        }
      }
      return result
    })

    const ancestors = Effect.fn("SessionTaskState.ancestors")(function* (sessionID: SessionID, strict: boolean) {
      const result: Session.Info[] = []
      const seen = new Set<SessionID>()
      let child = yield* sessions.get(sessionID).pipe(Effect.orDie)
      while (child.parentID && !seen.has(child.id) && result.length < 64) {
        seen.add(child.id)
        const parent = yield* sessions.get(child.parentID).pipe(Effect.catch(() => Effect.succeed(undefined)))
        if (!parent) break
        if (strict && !isTaskChild(parent, child)) break
        result.push(parent)
        child = parent
      }
      return result
    })

    const canReadHistory = Effect.fn("SessionTaskState.canReadHistory")(function* (input: {
      sessionID: SessionID
      ancestorID: SessionID
    }) {
      if (input.sessionID === input.ancestorID) return true
      const lifecycle = yield* get(input.sessionID)
      if (lifecycle && lifecycle.ancestor_access !== "history" && lifecycle.ancestor_access !== "all") return false
      return (yield* ancestors(input.sessionID, !!lifecycle)).some((item) => item.id === input.ancestorID)
    })

    const inspect = Effect.fn("SessionTaskState.inspect")(function* (input: {
      sessionID: SessionID
      scope: "descendants" | "ancestors"
    }) {
      if (input.scope === "ancestors") {
        const lifecycle = yield* get(input.sessionID)
        if (!lifecycle || (lifecycle.ancestor_access !== "status" && lifecycle.ancestor_access !== "all")) {
          return { allowed: false, scope: input.scope, nodes: [] }
        }
        const chain = yield* ancestors(input.sessionID, true)
        const caller = yield* sessions.get(input.sessionID).pipe(Effect.orDie)
        return { allowed: true, scope: input.scope, nodes: yield* buildNodes(chain, false, caller) }
      }

      const caller = yield* sessions.get(input.sessionID).pipe(Effect.orDie)
      const children = yield* descendants(caller)
      return { allowed: true, scope: input.scope, nodes: yield* buildNodes(children, true, caller) }
    })

    const stop = Effect.fn("SessionTaskState.stop")(function* (input: { sessionID: SessionID; taskID: SessionID }) {
      const inspection = yield* inspect({ sessionID: input.sessionID, scope: "descendants" })
      const selected = findNode(inspection.nodes, input.taskID)
      if (!selected) return { stopped: [], unavailable: [] }
      const unavailable = yield* Effect.filter(flatten(selected), (node) =>
        node.execution === "unknown"
          ? get(node.taskID).pipe(Effect.map((lifecycle) => lifecycle?.status === "running"))
          : Effect.succeed(false),
      ).pipe(Effect.map((nodes) => nodes.map((node) => node.taskID)))
      if (selected.execution !== "running" && selected.execution !== "detached") return { stopped: [], unavailable }
      const targets = flatten(selected).filter((node) => node.execution === "running" || node.execution === "detached")
      const stopped = yield* Effect.forEach(
        targets,
        Effect.fnUntraced(function* (node) {
          const decision = yield* withLock(node.taskID)(
            Effect.gen(function* () {
              const lifecycle = yield* get(node.taskID)
              const job = yield* background.get(node.taskID)
              if (job?.status === "running") return { type: "cancel" as const, generation: lifecycle?.generation }
              if (node.execution !== "detached" || !lifecycle) return { type: "none" as const }
              const stopped = yield* settle({
                sessionID: node.taskID,
                generation: lifecycle.generation,
                status: "cancelled",
                error: "Task cancelled",
              })
              return { type: stopped ? ("stopped" as const) : ("none" as const) }
            }),
          )
          if (decision.type === "none") return false
          if (decision.type === "stopped") return true

          const result = yield* background.cancel(node.taskID)
          if (result?.status !== "cancelled") return false
          // Completion delivery settles Task jobs before routing, so the fallback settle waits for the job to close.
          // Delivery to a running ancestor Task waits for that ancestor's turn, which can be the caller's turn.
          yield* background.wait({ id: node.taskID }).pipe(
            Effect.andThen(
              withLock(node.taskID)(
                Effect.gen(function* () {
                  const generation =
                    decision.generation ?? (yield* ensure({ sessionID: node.taskID, ancestorAccess: "history" }))
                  if (!generation) return
                  yield* settle({
                    sessionID: node.taskID,
                    generation,
                    status: "cancelled",
                    error: "Task cancelled",
                  })
                }),
              ),
            ),
            Effect.forkIn(scope, { startImmediately: true }),
          )
          return true
        }),
        { concurrency: "unbounded" },
      )
      return { stopped: targets.filter((_, index) => stopped[index]).map((node) => node.taskID), unavailable }
    })

    const buildNodes = Effect.fnUntraced(function* (
      infos: Session.Info[],
      nested: boolean,
      caller: Session.Info,
    ): Effect.fn.Return<Node[]> {
      if (infos.length === 0) return []
      const ids = infos.map((info) => info.id)
      const [lifecycles, jobs, statusMap, activityMap, pendingPermissions, pendingQuestions, activityRows, messages] =
        yield* Effect.all([
          db.select().from(SessionTaskTable).where(inArray(SessionTaskTable.session_id, ids)).all().pipe(Effect.orDie),
          background.list(),
          statuses.list(),
          activity.list(),
          permissions.list(),
          questions.list(),
          durableActivity(ids),
          Effect.forEach(
            infos,
            (info) => sessions.messages({ sessionID: info.id, limit: 1 }).pipe(Effect.catch(() => Effect.succeed([]))),
            { concurrency: "unbounded" },
          ),
        ])
      const now = Date.now()
      const lifecycleMap = new Map(lifecycles.map((item) => [item.session_id, item]))
      const jobMap = new Map(jobs.filter((job) => job.type === "task").map((job) => [job.id, job]))
      const waitingParents = new Set(
        jobs.flatMap((job) =>
          job.type === "task" && job.status === "running" && typeof job.metadata?.parentSessionId === "string"
            ? [job.metadata.parentSessionId]
            : [],
        ),
      )
      const permissionSessions = new Set(pendingPermissions.map((item) => item.sessionID))
      const questionSessions = new Set(pendingQuestions.map((item) => item.sessionID))
      const nodes = new Map<SessionID, Node>()

      infos.forEach((info, index) => {
        const lifecycle = lifecycleMap.get(info.id)
        const job = jobMap.get(info.id)
        const status = statusMap.get(info.id)
        const message = messages[index]?.at(-1)
        const tool = message?.parts.findLast((part) => part.type === "tool" && part.state.status === "running")
        const execution = executionOf(lifecycle, job, status, message, sameLocation(caller, info))
        const completedAt = lifecycle?.time_completed ?? job?.completed_at ?? completedAtOf(message)
        const lastActivityAt = Math.max(
          info.time.updated,
          lifecycle?.time_updated ?? 0,
          activityMap.get(info.id) ?? 0,
          activityRows.get(info.id) ?? 0,
          job?.started_at ?? 0,
          job?.completed_at ?? 0,
          completedAt ?? 0,
          tool?.type === "tool" && tool.state.status === "running" ? tool.state.time.start : 0,
        )
        const currentTool =
          tool?.type === "tool" && tool.state.status === "running"
            ? { name: tool.tool, title: tool.state.title, startedAt: tool.state.time.start }
            : undefined
        const delivery = lifecycle
          ? lifecycle.time_delivered && lifecycle.delivery_session_id
            ? {
                status: "delivered" as const,
                targetID: lifecycle.delivery_session_id,
                deliveredAt: lifecycle.time_delivered,
              }
            : { status: "pending" as const }
          : { status: "unknown" as const }
        nodes.set(info.id, {
          taskID: info.id,
          parentID: info.parentID,
          agent: info.agent,
          title: info.title,
          execution,
          phase: phaseOf({
            execution,
            status,
            hasPermission: permissionSessions.has(info.id),
            hasQuestion: questionSessions.has(info.id),
            currentTool,
            message,
            jobRunning: job?.status === "running",
            waitingForChild: waitingParents.has(info.id),
          }),
          currentTool,
          startedAt: job?.started_at ?? lifecycle?.time_created ?? info.time.created,
          lastActivityAt,
          quietForMs: Math.max(0, now - lastActivityAt),
          completedAt,
          // Failure text can quote transcript output. Ancestor status access does not grant transcript access.
          error: nested ? (lifecycle?.error ?? job?.error ?? messageError(message)) : undefined,
          delivery,
          children: [],
        })
      })

      if (!nested)
        return infos.flatMap((info) => {
          const node = nodes.get(info.id)
          return node ? [node] : []
        })
      for (const info of infos) {
        const node = nodes.get(info.id)
        if (!node || !info.parentID) continue
        nodes.get(info.parentID)?.children.push(node)
      }
      return infos.flatMap((info) => {
        if (info.parentID && nodes.has(info.parentID)) return []
        const node = nodes.get(info.id)
        return node ? [node] : []
      })
    })

    const durableActivity = Effect.fnUntraced(function* (ids: SessionID[]) {
      const [legacyMessages, parts, currentMessages] = yield* Effect.all([
        db
          .select({ sessionID: MessageTable.session_id, time: max(MessageTable.time_updated) })
          .from(MessageTable)
          .where(inArray(MessageTable.session_id, ids))
          .groupBy(MessageTable.session_id)
          .all()
          .pipe(Effect.orDie),
        db
          .select({ sessionID: PartTable.session_id, time: max(PartTable.time_updated) })
          .from(PartTable)
          .where(inArray(PartTable.session_id, ids))
          .groupBy(PartTable.session_id)
          .all()
          .pipe(Effect.orDie),
        db
          .select({ sessionID: SessionMessageTable.session_id, time: max(SessionMessageTable.time_updated) })
          .from(SessionMessageTable)
          .where(inArray(SessionMessageTable.session_id, ids))
          .groupBy(SessionMessageTable.session_id)
          .all()
          .pipe(Effect.orDie),
      ])
      const result = new Map<SessionID, number>()
      for (const row of [...legacyMessages, ...parts, ...currentMessages]) {
        if (row.time !== null) result.set(row.sessionID, Math.max(result.get(row.sessionID) ?? 0, row.time))
      }
      return result
    })

    return Service.of({ withLock, begin, ensure, settle, deliver, grant, get, inspect, stop, canReadHistory })
  }),
)

function executionOf(
  lifecycle: Lifecycle | undefined,
  job: BackgroundJob.Info | undefined,
  status: SessionStatus.Info | undefined,
  message: SessionV1Message | undefined,
  local: boolean,
): Execution {
  if (job && job.status !== "running") return job.status
  if (lifecycle?.status !== undefined && lifecycle.status !== "running") return lifecycle.status
  if (lifecycle?.status === "running" && job?.status === "running") return "running"
  // Jobs are process-local per location. A Task in another location can run where this location cannot observe it.
  if (lifecycle?.status === "running") return local ? "detached" : "unknown"
  if (job?.status === "running" || (status && status.type !== "idle")) return "running"
  if (message?.info.role === "assistant" && message.info.error) return "error"
  if (completedAtOf(message) !== undefined) return "completed"
  return "unknown"
}

type SessionV1Message = SessionV1.WithParts

function completedAtOf(message: SessionV1Message | undefined) {
  return message?.info.role === "assistant" ? message.info.time.completed : undefined
}

function messageError(message: SessionV1Message | undefined) {
  if (message?.info.role !== "assistant" || !message.info.error) return undefined
  return "message" in message.info.error.data && typeof message.info.error.data.message === "string"
    ? message.info.error.data.message
    : message.info.error.name
}

function phaseOf(input: {
  execution: Execution
  status: SessionStatus.Info | undefined
  hasPermission: boolean
  hasQuestion: boolean
  currentTool?: Node["currentTool"]
  message: SessionV1Message | undefined
  jobRunning: boolean
  waitingForChild: boolean
}): Phase {
  if (input.hasPermission) return "permission"
  if (input.hasQuestion) return "question"
  if (input.status?.type === "retry") return "retry"
  if (input.currentTool) return "tool"
  if (input.execution === "detached") return "unknown"
  if (input.execution !== "running" && input.execution !== "unknown" && input.jobRunning) return "delivering"
  if (input.execution !== "running") return input.execution === "unknown" ? "unknown" : "idle"
  // A Task whose turn ended stays open while its own child runs. It is waiting, not delivering.
  if (completedAtOf(input.message) !== undefined) return input.waitingForChild ? "idle" : "delivering"
  if (input.status?.type === "busy") return "model"
  return "starting"
}

function intersectAccess(parent: AncestorAccess, requested: AncestorAccess): AncestorAccess {
  const history = (parent === "history" || parent === "all") && (requested === "history" || requested === "all")
  const status = (parent === "status" || parent === "all") && (requested === "status" || requested === "all")
  if (history && status) return "all"
  if (history) return "history"
  if (status) return "status"
  return "none"
}

function sameLocation(a: Session.Info, b: Session.Info) {
  return (
    a.workspaceID === b.workspaceID &&
    a.directory === b.directory &&
    (a.path === undefined || b.path === undefined || a.path === b.path)
  )
}

function findNode(nodes: Node[], taskID: SessionID): Node | undefined {
  for (const node of nodes) {
    if (node.taskID === taskID) return node
    const found = findNode(node.children, taskID)
    if (found) return found
  }
  return undefined
}

function flatten(node: Node): Node[] {
  return [node, ...node.children.flatMap(flatten)]
}

export const node = LayerNode.make({
  service: Service,
  layer,
  deps: [
    Database.node,
    Session.node,
    BackgroundJob.node,
    SessionStatus.node,
    SessionActivity.node,
    Permission.node,
    Question.node,
  ],
})

export * as SessionTaskState from "./task-state"
