import * as Tool from "./tool"
import DESCRIPTION from "./task.txt"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { BackgroundJob } from "@/background/job"
import { Session } from "@/session/session"
import { SessionTaskState } from "@/session/task-state"
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
    source?: SessionPrompt.NotificationSource,
    onAdmitted?: Effect.Effect<void>,
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
  ancestor_access: Schema.optional(SessionTaskState.AncestorAccess).annotate({
    description:
      'Optional access to ancestor sessions: "history" (default), "none" (blind), "status" (read-only state), or "all" (history and state). Omitted resumes preserve the existing task policy.',
  }),
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
    const taskState = yield* SessionTaskState.Service
    const scope = yield* Scope.Scope
    const flags = yield* RuntimeFlags.Service
    const database = yield* Database.Service
    const provider = yield* Provider.Service

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
        if (current.projectID !== next.projectID) break
        depth++
        // A moved ancestor still counts toward the depth limit, but it is not a notification target:
        // checkpoints and notification turns run in this Instance, not in the ancestor's new location.
        if (
          ancestors.at(-1) === current &&
          current.workspaceID === next.workspaceID &&
          current.directory === next.directory &&
          (current.path === undefined || next.path === undefined || current.path === next.path)
        )
          ancestors.push(next)
        current = next
        seen.add(current.id)
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
      // parseModel accepts any string. Check the model before a child Session is created or claimed.
      const explicitModel = params.model ? Provider.parseModel(params.model) : undefined
      if (explicitModel) yield* provider.getModel(explicitModel.providerID, explicitModel.modelID)

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
      const requestedAccess =
        params.ancestor_access ??
        (session ? (yield* taskState.get(session.id))?.ancestor_access : undefined) ??
        "history"
      const ancestorAccess = yield* taskState.grant({ sessionID: parent.id, requested: requestedAccess })

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

      const model = explicitModel ??
        next.model ?? {
          modelID: msg.info.modelID,
          providerID: msg.info.providerID,
        }
      const metadata = {
        parentSessionId: ctx.sessionID,
        sessionId: nextSession.id,
        ancestorSessionIds: ancestors.map((item) => item.id),
        messageId: ctx.messageID,
        ancestorAccess,
        model,
        background: true,
      }

      yield* ctx.metadata({
        title: params.description,
        metadata,
      })

      // A result held during a revert is dropped when the revert removes this message.
      const source = { sessionID: ctx.sessionID, messageID: ctx.messageID }
      const notificationTargets: Array<{
        session: (typeof ancestors)[number]
        checkpoint: number
        release: Effect.Effect<void>
      }> = []
      const parentHolds: Array<Effect.Effect<void>> = []
      const releasedTargets = yield* Ref.make(false)
      const releaseTargets = Effect.gen(function* () {
        if (yield* Ref.getAndSet(releasedTargets, true)) return
        yield* Effect.forEach(notificationTargets, (target) => target.release, { discard: true }).pipe(Effect.ignore)
        yield* Effect.forEach(parentHolds, (release) => release, { discard: true })
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
        const parts =
          flags.disableContextCompaction || (ancestorAccess !== "history" && ancestorAccess !== "all")
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

      const labeledResult = (state: "completed" | "error", text: string) =>
        renderOutput({
          sessionID: nextSession.id,
          state,
          summary:
            state === "completed"
              ? `Background task completed: ${params.description}`
              : `Background task failed: ${params.description}`,
          text,
        })

      const notificationInput = (
        target: (typeof notificationTargets)[number],
        state: "completed" | "error",
        text: string,
      ): SessionPrompt.PromptInput => ({
        sessionID: target.session.id,
        agent: target.session.agent ?? (target.session.id === ctx.sessionID ? ctx.agent : undefined),
        variant,
        parts: [{ type: "text", synthetic: true, text: labeledResult(state, text) }],
      })

      // The forward run output becomes the target job output, which the target's parent receives as the target's reply.
      const forward = Effect.fn("TaskTool.forwardBackgroundResult")(function* (
        target: (typeof notificationTargets)[number],
        state: "completed" | "error",
        text: string,
        admitted: Deferred.Deferred<void>,
      ) {
        const previous = (yield* background.get(target.session.id))?.output
        // The target never saw this result. Keep the target reply and pass the result through under its own task id.
        const passThrough = () => [previous, labeledResult(state, text)].filter(Boolean).join("\n\n")
        if (!(yield* validRoute())) return passThrough()
        const continuation = yield* admitResult(target, state, text)
        if (Option.isNone(continuation)) return passThrough()
        // Resolving `admitted` lets this child job close and release its leases. The target keeps its own lease
        // until the continuation settles; otherwise its checkpoint is gone when the continuation turn ends.
        const result = yield* Effect.uninterruptibleMask((restore) =>
          Effect.gen(function* () {
            const release = yield* ops.retain(target.session.id, target.checkpoint)
            yield* Deferred.succeed(admitted, undefined)
            return yield* restore(continuation.value).pipe(
              Effect.ensuring(Option.getOrElse(release, () => Effect.void)),
            )
          }),
        )
        // No reply means no turn answered this result, for example a turn-only interrupt paused the target Session.
        // Completing with the target's older output would look like a final answer.
        if (result?.info.role !== "assistant")
          return yield* Effect.fail(
            new Error(
              `Subagent failed (task_id: ${target.session.id}): Task interrupted before it answered the worker result (task_id: ${nextSession.id}).`,
            ),
          )
        const reply = result.parts.findLast((item) => item.type === "text")?.text ?? ""
        // A failed continuation turn fails the target job, as a failed first turn does in runTask.
        if (result.info.error)
          return yield* Effect.fail(
            new Error(`Subagent failed (task_id: ${target.session.id}): ${childFailure(result.info.error, reply)}`),
          )
        const failed = result.parts.findLast((item) => item.type === "tool" && item.state.status === "error")
        if (failed?.type === "tool" && failed.state.status === "error")
          return yield* Effect.fail(new Error(`Subagent failed (task_id: ${target.session.id}): ${failed.state.error}`))
        return reply || (previous ?? "")
      })

      const admitResult = Effect.fn("TaskTool.admitBackgroundResult")(function* (
        target: (typeof notificationTargets)[number],
        state: "completed" | "error",
        text: string,
      ) {
        return yield* taskState.withLock(nextSession.id)(
          Effect.gen(function* () {
            const generation = lifecycle.generation
            if (!generation) return Option.none()
            const current = yield* taskState.get(nextSession.id)
            if (!current || current.generation !== generation || current.status === "running") return Option.none()
            // A result held during a revert can be dropped later, so record the delivery when the message is saved.
            return yield* ops.admitNotification(
              notificationInput(target, state, text),
              target.checkpoint,
              source,
              taskState
                .deliver({ sessionID: nextSession.id, generation, targetID: target.session.id })
                .pipe(Effect.asVoid),
            )
          }),
        )
      })

      const route = Effect.fn("TaskTool.routeBackgroundResult")(function* (
        state: "completed" | "error",
        text: string,
        cancelled: boolean,
      ) {
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
              // Interrupting the forward run only stops waiting for the continuation. Cancel the turn itself too.
              run: forward(target, state, text, admitted).pipe(Effect.onInterrupt(() => ops.cancel(target.session.id))),
            })
          )
            return yield* Deferred.await(admitted)
          // A cancelled intermediate Task reports the cancellation of its whole subtree, so skip a second notice.
          if (cancelled && (yield* taskState.get(target.session.id))?.status === "cancelled")
            return yield* releaseTargets
        }
        const root = notificationTargets.at(-1)
        if (!root) {
          yield* releaseTargets
          return
        }
        if (!(yield* validRoute())) return yield* releaseTargets
        const continuation = yield* admitResult(root, state, text)
        // The root keeps a lease until its continuation settles, so it stays busy and its checkpoint stays current.
        if (Option.isSome(continuation))
          yield* ops.retain(root.session.id, root.checkpoint).pipe(
            Effect.flatMap((release) =>
              continuation.value.pipe(
                Effect.ignore,
                Effect.ensuring(Option.getOrElse(release, () => Effect.void)),
                Effect.forkIn(scope, { startImmediately: true }),
              ),
            ),
            Effect.uninterruptible,
          )
        yield* releaseTargets
      })

      // A revert claims the completion of the Tasks it cancels, so BackgroundJob skips `complete` and no notification is
      // sent. The job still closes, so settle the durable Task row here. After `complete` this finds nothing to settle.
      const settleUnfinished = Effect.suspend(() =>
        lifecycle.generation === undefined
          ? Effect.void
          : taskState
              .settle({
                sessionID: nextSession.id,
                generation: lifecycle.generation,
                status: "cancelled",
                error: "Task cancelled",
              })
              .pipe(Effect.asVoid),
      )

      const complete = Effect.fnUntraced(function* (info: BackgroundJob.Info) {
        const result = yield* taskState.withLock(nextSession.id)(
          Effect.gen(function* () {
            const generation = lifecycle.generation
            if (!generation) return undefined
            if (info.status === "completed") {
              const settled = yield* taskState.settle({
                sessionID: nextSession.id,
                generation,
                status: "completed",
              })
              return settled ? ({ state: "completed", text: info.output ?? "" } as const) : undefined
            }
            if (info.status === "error") {
              const settled = yield* taskState.settle({
                sessionID: nextSession.id,
                generation,
                status: "error",
                error: info.error,
              })
              return settled ? ({ state: "error", text: info.error ?? "" } as const) : undefined
            }
            if (info.status === "cancelled") {
              const settled = yield* taskState.settle({
                sessionID: nextSession.id,
                generation,
                status: "cancelled",
                error: "Task cancelled",
              })
              return settled ? ({ state: "error", text: "Task cancelled" } as const) : undefined
            }
            return undefined
          }),
        )
        if (!result) return yield* releaseTargets
        // BackgroundJob closes the job scope only after delivery, and delivery can wait for an ancestor turn.
        // Stop the cancelled child Session now so it does not keep working until then.
        if (info.status === "cancelled") yield* ops.cancel(nextSession.id)
        return yield* route(result.state, result.text, info.status === "cancelled")
      })

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
      const lifecycle = { generation: undefined as number | undefined, fresh: false }
      const admitted = yield* Effect.uninterruptibleMask((restore) =>
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
          // A parent Task generation stays open until this result is delivered. Otherwise the parent settles when its
          // turn ends, and the result skips the parent and reaches a higher ancestor as raw output.
          const parentHold = yield* background.hold({ id: ctx.sessionID, expectedType: id })
          if (Option.isSome(parentHold)) parentHolds.push(parentHold.value)
          function handoffTask(): Effect.Effect<Option.Option<"started" | "extended">, Error> {
            return Effect.suspend(() =>
              taskState
                .withLock(nextSession.id)(
                  Effect.gen(function* () {
                    const current = yield* background.get(nextSession.id)
                    if (current?.status === "running" && current.type !== id) return Option.none()
                    if (current?.status === "running") {
                      const generation = yield* taskState.ensure({ sessionID: nextSession.id, ancestorAccess })
                      if (!generation) return Option.none()
                      lifecycle.generation = generation
                      lifecycle.fresh = false
                      const extended = yield* ops.admitChild(
                        nextSession.id,
                        background.extend({
                          id: nextSession.id,
                          expectedType: id,
                          onFinalize: releaseTargets,
                          run: taskRun,
                        }),
                      )
                      if (Option.isNone(extended)) return Option.none()
                      return extended.value ? Option.some("extended" as const) : ("retry" as const)
                    }

                    const admission = yield* ops.admitChild(
                      nextSession.id,
                      Effect.gen(function* () {
                        // Begin after admission so a rejected handoff keeps the previous lifecycle state.
                        const generation = yield* taskState.begin({ sessionID: nextSession.id, ancestorAccess })
                        if (!generation) return Option.none()
                        lifecycle.generation = generation
                        lifecycle.fresh = true
                        const started = yield* background.start({
                          id: nextSession.id,
                          type: id,
                          title: params.description,
                          metadata,
                          onComplete: complete,
                          awaitOnComplete: true,
                          onFinalize: releaseTargets.pipe(Effect.andThen(settleUnfinished)),
                          notifyOnComplete: true,
                          run: taskRun,
                        })
                        if (started.type !== id) return yield* Effect.interrupt
                        return Option.some("started" as const)
                      }),
                    )
                    return Option.flatten(admission)
                  }),
                )
                .pipe(
                  Effect.flatMap((result) => {
                    if (result !== "retry") return Effect.succeed(result)
                    lifecycle.generation = undefined
                    lifecycle.fresh = false
                    return retryHandoff()
                  }),
                ),
            )
          }

          const retryHandoff = Effect.fnUntraced(function* () {
            // The finalizing child waits for its result to reach the nearest running ancestor Task. When this caller
            // is that Task, the result waits for this turn to end, so waiting here would never finish. This call holds
            // the caller job only when that job can still take the result.
            const child = yield* background.get(nextSession.id)
            if (child?.status === "running" && parentHolds.length > 0) {
              const settled = (yield* taskState.get(nextSession.id))?.status
              const outcome =
                settled === "cancelled"
                  ? "was cancelled; its cancellation notice is queued"
                  : settled === "error"
                    ? "failed; its error is queued"
                    : settled === "completed"
                      ? "finished; its result is queued"
                      : "is finishing; its result will be queued"
              return yield* Effect.fail(
                new Error(`Task ${nextSession.id} ${outcome} for this session. End this turn, then resume it.`),
              )
            }
            yield* restore(background.wait({ id: nextSession.id }))
            return yield* handoffTask()
          })

          const admitted = yield* handoffTask()
          if (Option.isSome(admitted)) handoff.value = true
          return admitted
        }).pipe(
          Effect.ensuring(
            Effect.suspend(() =>
              handoff.value
                ? Effect.void
                : lifecycle.fresh && lifecycle.generation
                  ? taskState
                      .settle({
                        sessionID: nextSession.id,
                        generation: lifecycle.generation,
                        status: "cancelled",
                        error: "Task handoff interrupted",
                      })
                      .pipe(Effect.andThen(releaseTargets))
                  : releaseTargets,
            ),
          ),
        ),
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
