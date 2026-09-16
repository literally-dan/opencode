import * as Tool from "./tool"
import DESCRIPTION from "./task_control.txt"
import { SessionID } from "@/session/schema"
import { SessionTaskState } from "@/session/task-state"
import { Effect, Exit, Schema } from "effect"

const id = "task_control"

export const Parameters = Schema.Struct({
  action: Schema.Literals(["list", "stop"]).annotate({
    description: "List subagent state or stop one task and its active descendants",
  }),
  task_id: Schema.optional(Schema.String).annotate({
    description: "The task_id to stop. Required when action is stop.",
  }),
  scope: Schema.optional(Schema.Literals(["descendants", "ancestors"])).annotate({
    description:
      'State to list. Defaults to "descendants". Ancestors require read-only status access granted when this task was spawned and cannot be stopped.',
  }),
})

type Metadata = {
  count: number
  scope: "descendants" | "ancestors"
  tasks: SessionTaskState.Node[]
  taskId?: string
  stopped?: boolean
}

export const TaskControlTool = Tool.define(
  id,
  Effect.gen(function* () {
    return make(yield* SessionTaskState.Service)
  }),
)

export function make(taskState: SessionTaskState.Interface): Tool.DefWithoutID<typeof Parameters, Metadata> {
  return {
    description: DESCRIPTION,
    parameters: Parameters,
    execute: (params: Schema.Schema.Type<typeof Parameters>, ctx: Tool.Context) =>
      Effect.gen(function* () {
        const scope = params.scope ?? "descendants"
        const inspection = yield* taskState.inspect({ sessionID: ctx.sessionID, scope })
        const count = inspection.nodes.reduce((total, node) => total + flatten(node).length, 0)
        const metadata: Metadata = {
          count,
          scope,
          tasks: inspection.nodes,
        }
        if (params.action === "list") {
          if (!inspection.allowed) {
            return {
              title: "Ancestor status unavailable",
              metadata,
              output: "This subagent was not granted read-only ancestor status access.",
            }
          }
          return {
            title: `${count} ${scope === "ancestors" ? "ancestor" : "subagent"}${count === 1 ? "" : "s"}`,
            metadata,
            output: count
              ? [scope === "ancestors" ? "Ancestor sessions:" : "Subagent tree:", ...format(inspection.nodes)].join(
                  "\n",
                )
              : scope === "ancestors"
                ? "No ancestor sessions."
                : "No subagents.",
          }
        }

        if (!params.task_id) {
          return {
            title: "Missing task_id",
            metadata: { ...metadata, count: 0, tasks: [], stopped: false },
            output: "task_id is required when action is stop.",
          }
        }

        if (scope === "ancestors") {
          return {
            title: "Ancestor stop denied",
            metadata: { ...metadata, count: 0, tasks: [], taskId: params.task_id, stopped: false },
            output: "Ancestor status access is read-only. A subagent cannot stop an ancestor.",
          }
        }

        const taskID = Schema.decodeUnknownExit(SessionID)(params.task_id)
        if (Exit.isFailure(taskID)) {
          return {
            title: "Task not running",
            metadata: { ...metadata, count: 0, tasks: [], taskId: params.task_id, stopped: false },
            output: `Task ${params.task_id} is not a running subagent owned by this session.`,
          }
        }

        const result = yield* taskState.stop({ sessionID: ctx.sessionID, taskID: taskID.value })
        const unavailable = result.unavailable.length
          ? ` ${result.unavailable.length} unfinished task${result.unavailable.length === 1 ? "" : "s"} in another location could not be observed or stopped from here: ${result.unavailable.join(", ")}.`
          : ""
        if (result.stopped.length === 0) {
          return {
            title: result.unavailable.length ? "Task in another location" : "Task not running",
            metadata: { ...metadata, count: 0, tasks: [], taskId: params.task_id, stopped: false },
            output: result.unavailable.length
              ? `Task ${params.task_id} was not stopped.${unavailable}`
              : `Task ${params.task_id} is not a running subagent owned by this session.`,
          }
        }
        return {
          title: `Stopped subagent: ${params.task_id}`,
          metadata: { ...metadata, taskId: params.task_id, stopped: true, count: result.stopped.length },
          output: `Stopped task ${params.task_id}${result.stopped.length > 1 ? ` and ${result.stopped.length - 1} active descendant${result.stopped.length === 2 ? "" : "s"}` : ""}.${unavailable}`,
        }
      }).pipe(Effect.orDie),
  }
}

function format(nodes: SessionTaskState.Node[], depth = 0): string[] {
  return nodes.flatMap((node) => {
    const error =
      node.execution === "error" || node.execution === "cancelled" ? node.error?.replace(/\s+/g, " ").trim() : undefined
    return [
      `${"  ".repeat(depth)}- ${node.taskID} [${node.execution}/${node.phase}, quiet ${duration(node.quietForMs)}, delivery ${node.delivery.status}]: ${node.title}`,
      ...(node.currentTool
        ? [
            `${"  ".repeat(depth + 1)}current tool: ${node.currentTool.name}${node.currentTool.title ? ` (${node.currentTool.title})` : ""}`,
          ]
        : []),
      ...(error ? [`${"  ".repeat(depth + 1)}error: ${error.length > 200 ? `${error.slice(0, 199)}…` : error}`] : []),
      ...format(node.children, depth + 1),
    ]
  })
}

function duration(ms: number) {
  if (ms < 1_000) return `${ms}ms`
  if (ms < 60_000) return `${Math.floor(ms / 1_000)}s`
  if (ms < 3_600_000) return `${Math.floor(ms / 60_000)}m`
  return `${Math.floor(ms / 3_600_000)}h`
}

function flatten(node: SessionTaskState.Node): SessionTaskState.Node[] {
  return [node, ...node.children.flatMap(flatten)]
}
