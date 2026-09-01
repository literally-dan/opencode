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
import { Deferred, Effect, Exit, Option, Ref, Schema, Scope } from "effect"
import { RuntimeFlags } from "@/effect/runtime-flags"
import { Database } from "@opencode-ai/core/database/database"
import { Provider } from "@/provider/provider"

export interface TaskPromptOps {
  cancel(sessionID: SessionID): Effect.Effect<void>
  checkpoint(sessionID: SessionID): Effect.Effect<number>
  retain(sessionID: SessionID, checkpoint: number): Effect.Effect<Option.Option<Effect.Effect<void>>>
  admitIfCurrent<A, E, R>(effect: Effect.Effect<A, E, R>): Effect.Effect<Option.Option<A>, E, R>
  admitChild<A, E, R>(sessionID: SessionID, effect: Effect.Effect<A, E, R>): Effect.Effect<Option.Option<A>, E, R>
  resolvePromptParts(template: string): Effect.Effect<SessionPrompt.PromptInput["parts"]>
  prompt(input: SessionPrompt.PromptInput): Effect.Effect<SessionV1.WithParts>
  admitNotification(
    input: SessionPrompt.PromptInput,
    checkpoint: number,
  ): Effect.Effect<Option.Option<SessionPrompt.NotificationContinuation>>
  notify(input: SessionPrompt.PromptInput, checkpoint: number): Effect.Effect<SessionV1.WithParts | void>
}

const id = "task"
// Two levels so a coordinating subagent (for example the `/target` coordinator)
// can delegate to a worker and a reviewer. The worker sits at the limit and
// cannot nest further.
const DEFAULT_SUBAGENT_DEPTH = 2
const BACKGROUND_DESCRIPTION = [
  "Every subagent launches asynchronously and returns a resumable task_id immediately.",
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

export const Parameters = Schema.Struct(BaseParameterFields)

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

      const parent = yield* sessions.get(ctx.sessionID)
      const ancestors = [parent]
      const seen = new Set([parent.id])
      let current = parent
      let depth = 0
      while (current.parentID) {
        const parentID = current.parentID
        if (current.taskParentID !== parentID || seen.has(parentID)) break
        const next = yield* sessions.get(parentID)
        if (
          current.projectID !== next.projectID ||
          current.workspaceID !== next.workspaceID ||
          current.directory !== next.directory ||
          (current.path !== undefined && next.path !== undefined && current.path !== next.path)
        )
          break
        depth++
        current = next
        seen.add(current.id)
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
          (session.path !== undefined && parent.path !== undefined && session.path !== parent.path)
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
      const ops = ctx.extra?.promptOps as TaskPromptOps
      if (!ops) return yield* Effect.fail(new Error("TaskTool requires promptOps in ctx.extra"))
      const child = session
        ? Option.some(session)
        : yield* ops.admitIfCurrent(
            sessions.createTask({
              parentID: ctx.sessionID,
              title: params.description + ` (@${next.name} subagent)`,
              agent: next.name,
              permission: [
                ...childPermission,
                ...childToolDenies.filter(
                  (deny) =>
                    !childPermission.some(
                      (rule) =>
                        rule.permission === deny.permission &&
                        rule.pattern === deny.pattern &&
                        rule.action === deny.action,
                    ),
                ),
              ],
            }),
          )
      if (Option.isNone(child)) return yield* Effect.interrupt
      const nextSession = child.value
      if (session) {
        const claimed = yield* ops.admitIfCurrent(
          sessions.claimTask({ sessionID: nextSession.id, parentID: ctx.sessionID }),
        )
        if (Option.isNone(claimed)) return yield* Effect.interrupt
      }

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
        ancestorSessionIds: ancestors.map((item) => item.id),
        model,
        background: true,
      }

      yield* ctx.metadata({
        title: params.description,
        metadata,
      })

      const notificationTargets: Array<{
        session: (typeof ancestors)[number]
        checkpoint: number
        release: Effect.Effect<void>
      }> = []
      const releasedTargets = yield* Ref.make(false)
      const releaseTargets = Effect.gen(function* () {
        if (yield* Ref.getAndSet(releasedTargets, true)) return
        yield* Effect.forEach(notificationTargets, (target) => target.release, { discard: true }).pipe(Effect.ignore)
      }).pipe(Effect.uninterruptible)
      const routeSnapshot = [nextSession, ...ancestors]
      const validRoute = Effect.fn("TaskTool.validCompletionRoute")(function* () {
        const matches = yield* Effect.forEach(routeSnapshot, (expected) =>
          sessions.get(expected.id).pipe(
            Effect.map(
              (current) =>
                current.projectID === expected.projectID &&
                current.workspaceID === expected.workspaceID &&
                current.directory === expected.directory &&
                (current.path === undefined || expected.path === undefined || current.path === expected.path),
            ),
            Effect.catch(() => Effect.succeed(false)),
          ),
        )
        return matches.every(Boolean)
      })
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
        admitted: Deferred.Deferred<void>,
      ) {
        if (!(yield* validRoute())) return text
        const continuation = yield* ops.admitNotification(notificationInput(target, state, text), target.checkpoint)
        if (Option.isNone(continuation)) return text
        yield* Deferred.succeed(admitted, undefined)
        const result = yield* continuation.value
        if (!result || result.info.role !== "assistant") return text
        const part = result.parts.findLast((item) => item.type === "text")
        return part?.type === "text" ? part.text : text
      })

      const route = Effect.fn("TaskTool.routeBackgroundResult")(function* (state: "completed" | "error", text: string) {
        if (!(yield* validRoute())) {
          yield* releaseTargets
          return
        }
        for (const target of notificationTargets.slice(0, -1)) {
          const admitted = yield* Deferred.make<void>()
          if (!(yield* validRoute())) {
            yield* releaseTargets
            return
          }
          if (
            yield* background.extend({
              id: target.session.id,
              expectedType: id,
              onFinalize: Deferred.succeed(admitted, undefined).pipe(Effect.andThen(releaseTargets)),
              run: forward(target, state, text, admitted),
            })
          )
            return yield* Deferred.await(admitted)
        }
        const root = notificationTargets.at(-1)
        if (!root) {
          yield* releaseTargets
          return
        }
        if (!(yield* validRoute())) return yield* releaseTargets
        const continuation = yield* ops.admitNotification(notificationInput(root, state, text), root.checkpoint)
        if (Option.isSome(continuation))
          yield* continuation.value.pipe(Effect.ignore, Effect.forkIn(scope, { startImmediately: true }))
        yield* releaseTargets
      })

      const complete = (info: BackgroundJob.Info) => {
        if (info.status === "completed") return route("completed", info.output ?? "")
        if (info.status === "error") return route("error", info.error ?? "")
        if (info.status === "cancelled") return route("error", "Task cancelled")
        return releaseTargets
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

      const taskRun = runTask().pipe(
        Effect.mapError(resumableError),
        Effect.catchDefect((defect) => Effect.fail(resumableError(defect))),
        Effect.onInterrupt(() => ops.cancel(nextSession.id)),
      )
      // Defer interruption until every acquired lease is released or owned by BackgroundJob.
      const handoff = { value: false }
      const admitted = yield* Effect.uninterruptibleMask(() =>
        Effect.gen(function* () {
          const retainedTargets = yield* Effect.forEach(
            ancestors,
            Effect.fnUntraced(function* (session) {
              const checkpoint = yield* ops.checkpoint(session.id)
              const release = yield* ops.retain(session.id, checkpoint)
              return Option.map(release, (release) => {
                const target = { session, checkpoint, release }
                notificationTargets.push(target)
                return target
              })
            }),
          )
          if (retainedTargets.some(Option.isNone)) return Option.none()

          const admitted = yield* ops.admitChild(
            nextSession.id,
            Effect.gen(function* () {
              if (
                yield* background.extend({
                  id: nextSession.id,
                  expectedType: id,
                  onFinalize: releaseTargets,
                  run: taskRun,
                })
              )
                return "extended" as const

              const started = yield* background.start({
                id: nextSession.id,
                type: id,
                title: params.description,
                metadata,
                onComplete: complete,
                awaitOnComplete: true,
                onFinalize: releaseTargets,
                notifyOnComplete: true,
                run: taskRun,
              })
              if (started.type !== id) return yield* Effect.interrupt
              return "started" as const
            }),
          )
          if (Option.isSome(admitted)) handoff.value = true
          return admitted
        }).pipe(Effect.ensuring(Effect.suspend(() => (handoff.value ? Effect.void : releaseTargets)))),
      )
      if (Option.isNone(admitted)) return yield* Effect.interrupt
      if (admitted.value === "extended") return backgroundResult("Background task updated", BACKGROUND_UPDATED)
      return backgroundResult("Background task started", BACKGROUND_STARTED)
    })

    return {
      description: [DESCRIPTION, BACKGROUND_DESCRIPTION].join("\n\n"),
      parameters: Parameters,
      execute: (params: Schema.Schema.Type<typeof Parameters>, ctx: Tool.Context) =>
        run(params, ctx).pipe(Effect.orDie),
    }
  }),
)
