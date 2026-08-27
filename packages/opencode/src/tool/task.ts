import * as Tool from "./tool"
import DESCRIPTION from "./task.txt"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { BackgroundJob } from "@/background/job"
import { Session } from "@/session/session"
import { SessionID, MessageID } from "../session/schema"
import { MessageV2 } from "../session/message-v2"
import { Agent } from "../agent/agent"
import { deriveSubagentSessionPermission } from "../agent/subagent-permissions"
import type { SessionPrompt } from "../session/prompt"
import { Config } from "@/config/config"
import { Effect, Exit, Schema, Scope } from "effect"
import { EffectBridge } from "@/effect/bridge"
import { RuntimeFlags } from "@/effect/runtime-flags"
import { Database } from "@opencode-ai/core/database/database"
import { Provider } from "@/provider/provider"

export interface TaskPromptOps {
  cancel(sessionID: SessionID): Effect.Effect<void>
  checkpoint(sessionID: SessionID): Effect.Effect<number>
  resolvePromptParts(template: string): Effect.Effect<SessionPrompt.PromptInput["parts"]>
  prompt(input: SessionPrompt.PromptInput): Effect.Effect<SessionV1.WithParts>
  notify(input: SessionPrompt.PromptInput, checkpoint: number): Effect.Effect<void>
}

const id = "task"
// Two levels so a coordinating subagent (for example the `/target` coordinator)
// can delegate to a worker and a reviewer. The worker sits at the limit and
// cannot nest further.
const DEFAULT_SUBAGENT_DEPTH = 2
const BACKGROUND_DESCRIPTION = [
  "Background mode is the default: the subagent launches asynchronously and returns a resumable task_id immediately.",
  "Set background=false only when the parent must wait synchronously for the result.",
  "You will be notified automatically when it finishes.",
].join(" ")
const BACKGROUND_STARTED = [
  "The task is working in the background. You will be notified automatically when it finishes.",
  "DO NOT sleep, poll for progress, ask the task for status, or duplicate this task's work — avoid working with the same files or topics it is using.",
  "Keep the main conversation moving with non-overlapping work. If there is nothing else to do, briefly tell the user what you launched and end your response.",
].join("\n")
const BACKGROUND_UPDATED = [
  "Additional context sent to the running background task.",
  "The task is still working in the background. You will be notified automatically when it finishes.",
  "DO NOT sleep, poll for progress, ask the task for status, or duplicate this task's work — avoid working with the same files or topics it is using.",
  "Keep the main conversation moving with non-overlapping work. If there is nothing else to do, briefly tell the user what you sent and end your response.",
].join("\n")

const BaseParameterFields = {
  description: Schema.String.annotate({ description: "A short (3-5 words) description of the task" }),
  prompt: Schema.String.annotate({ description: "The task for the agent to perform" }),
  subagent_type: Schema.String.annotate({ description: "The type of specialized agent to use for this task" }),
  model: Schema.optional(Schema.String).annotate({
    description:
      "Optional provider/model to use for the subagent (for example, anthropic/claude-sonnet-4-5). Defaults to the subagent's configured model, or the current model when none is configured",
  }),
  task_id: Schema.optional(Schema.String).annotate({
    description:
      "This should only be set if you mean to resume a previous task (you can pass a prior task_id and the task will continue the same subagent session as before instead of creating a fresh one)",
  }),
  command: Schema.optional(Schema.String).annotate({ description: "The command that triggered this task" }),
}

export const Parameters = Schema.Struct({
  ...BaseParameterFields,
  background: Schema.optional(Schema.Boolean).annotate({
    description:
      "Run the agent in the background (default true). Set false only to wait synchronously. You will be notified when it completes. DO NOT sleep, poll, or proactively check on its progress",
    default: true,
  }),
})

function renderOutput(input: {
  sessionID: SessionID
  state: "running" | "completed" | "error"
  summary?: string
  text: string
}) {
  const tag = input.state === "error" ? "task_error" : "task_result"
  return [
    `<task id="${input.sessionID}" state="${input.state}">`,
    ...(input.summary ? [`<summary>${input.summary}</summary>`] : []),
    `<${tag}>`,
    input.text,
    `</${tag}>`,
    "</task>",
  ].join("\n")
}

function childFailure(error: NonNullable<SessionV1.Assistant["error"]>, text: string) {
  const detail = "message" in error.data && typeof error.data.message === "string" ? error.data.message : error.name
  const advice = SessionV1.ContentFilterError.isInstance(error)
    ? "The provider's safety classifier refused this turn. Resuming with the same instructions will be refused again — change the approach or report the blocker."
    : undefined
  return [
    `The subagent stopped with ${error.name}: ${detail}`,
    advice,
    text && `Partial output before the failure:\n${text}`,
  ]
    .filter((part): part is string => !!part)
    .join("\n\n")
}

export const TaskTool = Tool.define(
  id,
  Effect.gen(function* () {
    const agent = yield* Agent.Service
    const background = yield* BackgroundJob.Service
    const config = yield* Config.Service
    const sessions = yield* Session.Service
    const scope = yield* Scope.Scope
    const flags = yield* RuntimeFlags.Service
    const database = yield* Database.Service

    const run = Effect.fn("TaskTool.execute")(function* (
      params: Schema.Schema.Type<typeof Parameters>,
      ctx: Tool.Context,
    ) {
      const cfg = yield* config.get()
      const runInBackground = params.background !== false

      const parent = yield* sessions.get(ctx.sessionID)
      const ancestors = [parent]
      let current = parent
      let depth = 0
      while (current.parentID) {
        depth++
        current = yield* sessions.get(current.parentID)
        ancestors.push(current)
      }
      const maxDepth = cfg.subagent_depth ?? DEFAULT_SUBAGENT_DEPTH
      if (depth >= maxDepth) {
        return yield* Effect.fail(
          new Error(
            `Subagent depth limit reached (${maxDepth}). Increase "subagent_depth" to allow nested subagents.${params.task_id ? ` (task_id: ${params.task_id})` : ""}`,
          ),
        )
      }

      const next = yield* agent.get(params.subagent_type)
      if (!next) {
        return yield* Effect.fail(
          new Error(
            `Unknown agent type: ${params.subagent_type} is not a valid agent type${params.task_id ? ` (task_id: ${params.task_id})` : ""}`,
          ),
        )
      }

      const resume = Effect.fn("TaskTool.resume")(function* (taskID: string) {
        const decoded = Schema.decodeUnknownExit(SessionID)(taskID)
        if (Exit.isFailure(decoded))
          return yield* Effect.fail(new Error(`Cannot resume task_id ${taskID}: invalid session ID`))
        const session = yield* sessions.get(decoded.value).pipe(Effect.catchCause(() => Effect.succeed(undefined)))
        if (!session) return yield* Effect.fail(new Error(`Cannot resume task_id ${taskID}: session not found`))
        if (session.id === parent.id)
          return yield* Effect.fail(
            new Error(`Cannot resume task_id ${taskID}: a task cannot resume its parent session`),
          )
        if (session.parentID !== parent.id)
          return yield* Effect.fail(
            new Error(`Cannot resume task_id ${taskID}: session is not a direct child of ${parent.id}`),
          )
        if (
          session.projectID !== parent.projectID ||
          session.workspaceID !== parent.workspaceID ||
          session.directory !== parent.directory ||
          session.path !== parent.path
        )
          return yield* Effect.fail(
            new Error(`Cannot resume task_id ${taskID}: session belongs to a different project or location`),
          )
        if (session.agent !== undefined && session.agent !== next.name)
          return yield* Effect.fail(
            new Error(`Cannot resume task_id ${taskID}: session agent ${session.agent} does not match ${next.name}`),
          )
        return session
      })
      const session = params.task_id ? yield* resume(params.task_id) : undefined

      if (!ctx.extra?.bypassAgentCheck) {
        yield* ctx.ask({
          permission: id,
          patterns: [params.subagent_type],
          always: ["*"],
          metadata: {
            description: params.description,
            subagent_type: params.subagent_type,
          },
        })
      }

      const childPermission = deriveSubagentSessionPermission({
        parentSessionPermission: parent.permission ?? [],
        subagent: next,
      })
      const childToolDenies = [
        ...(next.permission.some((rule) => rule.permission === "todowrite")
          ? []
          : [{ permission: "todowrite" as const, pattern: "*" as const, action: "deny" as const }]),
        ...(next.permission.some((rule) => rule.permission === id)
          ? []
          : [{ permission: id, pattern: "*" as const, action: "deny" as const }]),
        ...(cfg.experimental?.primary_tools?.map((permission) => ({
          permission,
          pattern: "*" as const,
          action: "deny" as const,
        })) ?? []),
      ]
      const nextSession =
        session ??
        (yield* sessions.create({
          parentID: ctx.sessionID,
          title: params.description + ` (@${next.name} subagent)`,
          agent: next.name,
          permission: [
            ...childPermission,
            ...childToolDenies.filter(
              (deny) =>
                !childPermission.some(
                  (rule) =>
                    rule.permission === deny.permission && rule.pattern === deny.pattern && rule.action === deny.action,
                ),
            ),
          ],
        }))

      const msg = yield* MessageV2.get({ sessionID: ctx.sessionID, messageID: ctx.messageID }).pipe(
        Effect.provideService(Database.Service, database),
        Effect.orDie,
      )
      if (msg.info.role !== "assistant") return yield* Effect.fail(new Error("Not an assistant message"))
      const variant = msg.info.variant

      const model = (params.model ? Provider.parseModel(params.model) : undefined) ??
        next.model ?? {
          modelID: msg.info.modelID,
          providerID: msg.info.providerID,
        }
      const metadata = {
        parentSessionId: ctx.sessionID,
        sessionId: nextSession.id,
        model,
        ...(runInBackground ? { background: true } : {}),
      }

      yield* ctx.metadata({
        title: params.description,
        metadata,
      })

      const ops = ctx.extra?.promptOps as TaskPromptOps
      if (!ops) return yield* Effect.fail(new Error("TaskTool requires promptOps in ctx.extra"))
      const notificationTargets = yield* Effect.forEach(ancestors, (session) =>
        ops.checkpoint(session.id).pipe(Effect.map((checkpoint) => ({ session, checkpoint }))),
      )
      const resumableFailure = (message: string) =>
        message.includes(`task_id: ${nextSession.id}`)
          ? message
          : `Subagent failed (task_id: ${nextSession.id}): ${message}`
      const resumableError = (error: unknown) =>
        new Error(resumableFailure(error instanceof Error ? error.message : String(error)))

      const runTask = Effect.fn("TaskTool.runTask")(function* () {
        const basicParts = yield* ops.resolvePromptParts(params.prompt)
        // Surface the parent session ID so the subagent can pass it to
        // `read_part` / `search_session_history` when it needs to recover
        // compacted history from the spawning session. Gated with those tools:
        // without them the id is unusable, and naming the parent session at all
        // is a capability worth withholding when the feature is off.
        const parts = flags.disableContextCompaction
          ? basicParts
          : [
              ...basicParts,
              {
                type: "text" as const,
                text: `\n<subagent-context parent-session-id="${ctx.sessionID}" />`,
              },
            ]
        const result = yield* ops.prompt({
          messageID: MessageID.ascending(),
          sessionID: nextSession.id,
          model: {
            modelID: model.modelID,
            providerID: model.providerID,
          },
          variant: params.model || next.model ? undefined : variant,
          agent: next.name,
          parts,
        })
        const text = result.parts.findLast((item) => item.type === "text")?.text ?? ""
        if (result.info.role === "assistant" && result.info.error) {
          return yield* Effect.fail(
            new Error(`Subagent failed (task_id: ${nextSession.id}): ${childFailure(result.info.error, text)}`),
          )
        }
        const failed = result.parts.findLast((item) => item.type === "tool" && item.state.status === "error")
        if (failed?.type === "tool" && failed.state.status === "error") {
          return yield* Effect.fail(new Error(`Subagent failed (task_id: ${nextSession.id}): ${failed.state.error}`))
        }
        return text
      })

      const notificationInput = (
        target: (typeof notificationTargets)[number],
        state: "completed" | "error",
        text: string,
      ): SessionPrompt.PromptInput => ({
        sessionID: target.session.id,
        agent: target.session.agent ?? (target.session.id === ctx.sessionID ? ctx.agent : undefined),
        variant,
        parts: [
          {
            type: "text",
            synthetic: true,
            text: renderOutput({
              sessionID: nextSession.id,
              state,
              summary:
                state === "completed"
                  ? `Background task completed: ${params.description}`
                  : `Background task failed: ${params.description}`,
              text,
            }),
          },
        ],
      })

      const forward = Effect.fn("TaskTool.forwardBackgroundResult")(function* (
        target: (typeof notificationTargets)[number],
        state: "completed" | "error",
        text: string,
      ) {
        yield* ops.notify(notificationInput(target, state, text), target.checkpoint)
        const latest = (yield* sessions.messages({ sessionID: target.session.id })).findLast(
          (message) => message.info.role === "assistant",
        )
        const part = latest?.parts.findLast((item) => item.type === "text")
        return part?.type === "text" ? part.text : text
      })

      const route = Effect.fn("TaskTool.routeBackgroundResult")(function* (
        state: "completed" | "error",
        text: string,
      ) {
        for (const target of notificationTargets.slice(0, -1)) {
          if (
            yield* background.extend({
              id: target.session.id,
              expectedType: id,
              run: forward(target, state, text),
            })
          )
            return
        }
        const root = notificationTargets.at(-1)
        if (!root) return
        yield* ops
          .notify(notificationInput(root, state, text), root.checkpoint)
          .pipe(Effect.ignore, Effect.forkIn(scope, { startImmediately: true }))
      })

      const complete = (info: BackgroundJob.Info) => {
        if (info.status === "completed") return route("completed", info.output ?? "")
        if (info.status === "error") return route("error", info.error ?? "")
        if (info.status === "cancelled") return route("error", "Task cancelled")
        return Effect.void
      }

      function backgroundResult(summary: string, text: string) {
        return {
          title: params.description,
          metadata: {
            ...metadata,
            background: true,
            jobId: nextSession.id,
          },
          output: renderOutput({
            sessionID: nextSession.id,
            state: "running",
            summary,
            text,
          }),
        }
      }

      const waitForeground = Effect.fn("TaskTool.waitForeground")(function* (allowPromotion: boolean) {
        const runCancel = yield* EffectBridge.make()
        const cancel = ops.cancel(nextSession.id)

        function onAbort() {
          runCancel.fork(cancel)
        }

        return yield* Effect.acquireUseRelease(
          Effect.sync(() => {
            ctx.abort.addEventListener("abort", onAbort)
          }),
          () =>
            Effect.gen(function* () {
              const result = allowPromotion
                ? yield* Effect.raceFirst(
                    background.wait({ id: nextSession.id }).pipe(Effect.map((waited) => waited.info)),
                    background.waitForPromotion(nextSession.id),
                  )
                : (yield* background.wait({ id: nextSession.id })).info
              if (result?.status === "running" && result.metadata?.background === true)
                return backgroundResult("Background task started", BACKGROUND_STARTED)
              if (!result) return yield* Effect.fail(new Error(resumableFailure("Task result unavailable")))
              if (result.status === "error")
                return yield* Effect.fail(new Error(resumableFailure(result.error ?? "Task failed")))
              if (result.status === "cancelled")
                return yield* Effect.fail(new Error(resumableFailure("Task cancelled")))
              return {
                title: params.description,
                metadata,
                output: renderOutput({ sessionID: nextSession.id, state: "completed", text: result.output ?? "" }),
              }
            }),
          (_, exit) =>
            Effect.gen(function* () {
              if (Exit.hasInterrupts(exit))
                yield* Effect.all([cancel, background.cancel(nextSession.id)], { discard: true })
            }).pipe(
              Effect.ensuring(
                Effect.sync(() => {
                  ctx.abort.removeEventListener("abort", onAbort)
                }),
              ),
            ),
        )
      })

      const taskRun = runTask().pipe(
        Effect.mapError(resumableError),
        Effect.catchDefect((defect) => Effect.fail(resumableError(defect))),
        Effect.onInterrupt(() => ops.cancel(nextSession.id)),
      )
      if (
        yield* background.extend({
          id: nextSession.id,
          expectedType: id,
          claimCompletion: !runInBackground,
          run: taskRun,
        })
      ) {
        if (runInBackground) return backgroundResult("Background task updated", BACKGROUND_UPDATED)
        const active = yield* background.get(nextSession.id)
        return yield* waitForeground(active?.metadata?.background !== true)
      }

      yield* background.start({
        id: nextSession.id,
        type: id,
        title: params.description,
        metadata,
        onPromote: ctx.metadata({
          title: params.description,
          metadata: { ...metadata, background: true, jobId: nextSession.id },
        }),
        onComplete: complete,
        notifyOnComplete: runInBackground,
        run: taskRun,
      })

      if (runInBackground) {
        return backgroundResult("Background task started", BACKGROUND_STARTED)
      }

      return yield* waitForeground(true)
    })

    return {
      description: [DESCRIPTION, BACKGROUND_DESCRIPTION].join("\n\n"),
      parameters: Parameters,
      execute: (params: Schema.Schema.Type<typeof Parameters>, ctx: Tool.Context) =>
        run(params, ctx).pipe(Effect.orDie),
    }
  }),
)
