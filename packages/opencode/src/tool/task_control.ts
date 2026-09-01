import * as Tool from "./tool"
import DESCRIPTION from "./task_control.txt"
import { BackgroundJob } from "@/background/job"
import { Effect, Schema } from "effect"

const id = "task_control"

export const Parameters = Schema.Struct({
  action: Schema.Literals(["list", "stop"]).annotate({
    description: "List running subagents or stop one task and its running descendants",
  }),
  task_id: Schema.optional(Schema.String).annotate({
    description: "The task_id to stop. Required when action is stop.",
  }),
})

type Metadata = {
  count: number
  taskId?: string
  stopped?: boolean
}

export const TaskControlTool = Tool.define(
  id,
  Effect.gen(function* () {
    return make(yield* BackgroundJob.Service)
  }),
)

export function make(background: BackgroundJob.Interface): Tool.DefWithoutID<typeof Parameters, Metadata> {
  return {
    description: DESCRIPTION,
    parameters: Parameters,
    execute: (params: Schema.Schema.Type<typeof Parameters>, ctx: Tool.Context) =>
      Effect.gen(function* () {
        const jobs = yield* background.list()
        const tasks = ownedTasks(jobs, ctx.sessionID)
        const metadata: Metadata = {
          count: tasks.length,
        }
        if (params.action === "list") {
          return {
            title: `${tasks.length} running subagent${tasks.length === 1 ? "" : "s"}`,
            metadata,
            output: tasks.length
              ? ["Running subagents:", ...tasks.map((task) => `- ${task.id}: ${task.title ?? "Untitled task"}`)].join(
                  "\n",
                )
              : "No running subagents.",
          }
        }

        if (!params.task_id) {
          return {
            title: "Missing task_id",
            metadata: { ...metadata, count: 0, stopped: false },
            output: "task_id is required when action is stop.",
          }
        }

        const selected = tasks.find((task) => task.id === params.task_id)
        if (!selected) {
          return {
            title: "Task not running",
            metadata: { ...metadata, count: 0, taskId: params.task_id, stopped: false },
            output: `Task ${params.task_id} is not a running subagent owned by this session.`,
          }
        }

        const descendants = ownedTasks(jobs, selected.id)
        yield* Effect.forEach([selected, ...descendants], (task) => background.cancel(task.id), {
          concurrency: "unbounded",
          discard: true,
        })
        return {
          title: `Stopped subagent: ${selected.title ?? selected.id}`,
          metadata: { ...metadata, taskId: selected.id, stopped: true, count: descendants.length + 1 },
          output: `Stopped task ${selected.id}${descendants.length ? ` and ${descendants.length} running descendant${descendants.length === 1 ? "" : "s"}` : ""}.`,
        }
      }).pipe(Effect.orDie),
  }
}

function ownedTasks(jobs: BackgroundJob.Info[], sessionID: string) {
  const owners = new Set([sessionID])
  const pending = jobs.filter((job) => job.type === "task")

  while (pending.length) {
    const index = pending.findIndex((job) => {
      const parentSessionID = job.metadata?.parentSessionId
      const childSessionID = job.metadata?.sessionId
      const ancestors = job.metadata?.ancestorSessionIds
      return (
        typeof childSessionID === "string" &&
        childSessionID === job.id &&
        ((typeof parentSessionID === "string" && owners.has(parentSessionID)) ||
          (Array.isArray(ancestors) && ancestors.some((item) => typeof item === "string" && item === sessionID)))
      )
    })
    if (index === -1) break
    const [job] = pending.splice(index, 1)
    owners.add(job.id)
  }

  return jobs.filter(
    (job) => job.type === "task" && job.status === "running" && job.id !== sessionID && owners.has(job.id),
  )
}
