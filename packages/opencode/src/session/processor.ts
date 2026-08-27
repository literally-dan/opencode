import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { PermissionV1 } from "@opencode-ai/core/v1/permission"
import { Image } from "@/image/image"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { Cause, Deferred, Effect, Exit, Layer, Context, Scope, Schema, Semaphore } from "effect"
import * as Stream from "effect/Stream"
import { Agent } from "@/agent/agent"
import { Config } from "@/config/config"
import { Permission } from "@/permission"
import { Plugin } from "@/plugin"
import { Snapshot } from "@/snapshot"
import { Session } from "./session"
import { LLM } from "./llm"
import { MessageV2 } from "./message-v2"
import { check as checkOverflow, estimateRequest, REQUEST_REMINDER_HEADROOM } from "./overflow"
import { PartID } from "./schema"
import type { SessionID } from "./schema"
import { SessionRetry } from "./retry"
import { SessionStatus } from "./status"
import { SessionSummary } from "./summary"
import type { Provider } from "@/provider/provider"
import { Question } from "@/question"
import { errorMessage } from "@/util/error"
import { isRecord } from "@/util/record"
import { EventV2Bridge } from "@/event-v2-bridge"
import { Database } from "@opencode-ai/core/database/database"
import { Usage, type LLMEvent } from "@opencode-ai/llm"
import { compactionOf } from "./compaction-pruning"

const DOOM_LOOP_THRESHOLD = 3
const toolGeneration = Symbol("SessionProcessor.toolGeneration")

export function getToolGeneration(options: object) {
  const value = Object.getOwnPropertyDescriptor(options, toolGeneration)?.value
  return typeof value === "number" ? value : undefined
}

function setToolGeneration<T extends object>(options: T, generation: number) {
  return Object.assign({}, options, { [toolGeneration]: generation })
}

function toolsForGeneration(tools: LLM.StreamInput["tools"], generation: number): LLM.StreamInput["tools"] {
  return Object.fromEntries(
    Object.entries(tools).map(([name, item]) => {
      if (!item.execute) return [name, item]
      const execute = item.execute
      return [name, { ...item, execute: (args, options) => execute(args, setToolGeneration(options, generation)) }]
    }),
  )
}

export type Result = "compact" | "stop" | "continue"

export interface Handle {
  readonly message: SessionV1.Assistant
  readonly toolSignal?: AbortSignal
  readonly startToolCall?: (
    generation: number | undefined,
    toolCallID: string,
    name: string,
    input: Record<string, unknown>,
  ) => Effect.Effect<boolean>
  readonly updateToolCall: (
    toolCallID: string,
    update: (part: SessionV1.ToolPart) => SessionV1.ToolPart,
  ) => Effect.Effect<SessionV1.ToolPart | undefined>
  readonly completeToolCall: (
    toolCallID: string,
    output: {
      title: string
      metadata: Record<string, any>
      output: string
      attachments?: SessionV1.FilePart[]
    },
  ) => Effect.Effect<void>
  readonly failToolCall?: (toolCallID: string, error: unknown) => Effect.Effect<void>
  readonly process: (streamInput: LLM.StreamInput) => Effect.Effect<Result>
}

type Input = {
  assistantMessage: SessionV1.Assistant
  sessionID: SessionID
  model: Provider.Model
}

export interface Interface {
  readonly create: (input: Input) => Effect.Effect<Handle>
}

type ToolCall = {
  partID: SessionV1.ToolPart["id"]
  messageID: SessionV1.ToolPart["messageID"]
  sessionID: SessionV1.ToolPart["sessionID"]
  done: Deferred.Deferred<void>
  awaitSettlement: boolean
}

type ToolOutput = {
  title: string
  metadata: Record<string, any>
  output: string
  attachments?: SessionV1.FilePart[]
}

type ToolSettlement = { type: "success"; output: ToolOutput } | { type: "failure"; error: unknown }

interface ProcessorContext extends Input {
  toolcalls: Record<string, ToolCall>
  shouldBreak: boolean
  snapshot: string | undefined
  blocked: boolean
  needsCompaction: boolean
  currentText: SessionV1.TextPart | undefined
  reasoningMap: Record<string, SessionV1.ReasoningPart>
  // Sticky once a tool has started executing. A retry replays the same request,
  // so the model re-emits the same call and a side-effecting tool would run a
  // second time. Deleting the part cannot undo what already touched the disk,
  // so this is the one failure the turn must not replay.
  executed: boolean
  // A failed local tool is terminal for this attempt. It must not be mistaken
  // for a transport failure that happened after a successful side effect.
  toolFailed: boolean
  toolFailure: Deferred.Deferred<void>
  // Parts written by the attempt currently in flight. A retry removes them
  // first, otherwise the truncated output from the failed attempt stays in the
  // transcript alongside its replacement.
  attemptParts: SessionV1.Part["id"][]
  request: LLM.StreamInput | undefined
}

type StreamEvent = LLMEvent

export class Service extends Context.Service<Service, Interface>()("@opencode/SessionProcessor") {}

const layer = Layer.effect(
  Service,
  Effect.gen(function* () {
    const session = yield* Session.Service
    const config = yield* Config.Service
    const snapshot = yield* Snapshot.Service
    const agents = yield* Agent.Service
    const llm = yield* LLM.Service
    const permission = yield* Permission.Service
    const plugin = yield* Plugin.Service
    const summary = yield* SessionSummary.Service
    const scope = yield* Scope.Scope
    const status = yield* SessionStatus.Service
    const image = yield* Image.Service
    const events = yield* EventV2Bridge.Service
    const database = yield* Database.Service

    const create = Effect.fn("SessionProcessor.create")(function* (input: Input) {
      // Pre-capture snapshot before the LLM stream starts. The AI SDK
      // may execute tools internally before emitting start-step events,
      // so capturing inside the event handler can be too late.
      const initialSnapshot = yield* snapshot.track()
      const ctx: ProcessorContext = {
        assistantMessage: input.assistantMessage,
        sessionID: input.sessionID,
        model: input.model,
        toolcalls: {},
        shouldBreak: false,
        snapshot: initialSnapshot,
        blocked: false,
        needsCompaction: false,
        currentText: undefined,
        reasoningMap: {},
        executed: false,
        toolFailed: false,
        toolFailure: yield* Deferred.make<void>(),
        attemptParts: [],
        request: undefined,
      }
      let aborted = false
      const toolController = new AbortController()
      const toolLock = Semaphore.makeUnsafe(1)
      const pendingSettlements = new Map<string, ToolSettlement>()
      const settling = new Set<string>()
      const settled = new Set<string>()
      const admittedToolCalls = new Set<string>()
      let generation = 0
      let activeGeneration: number | undefined
      let admissionClosed = false

      const parse = (e: unknown) =>
        MessageV2.fromError(e, {
          providerID: input.model.providerID,
          aborted,
        })

      const finishToolCall = Effect.fn("SessionProcessor.finishToolCall")(function* (toolCallID: string) {
        const done = ctx.toolcalls[toolCallID]?.done
        delete ctx.toolcalls[toolCallID]
        pendingSettlements.delete(toolCallID)
        settling.delete(toolCallID)
        settled.add(toolCallID)
        if (done) yield* Deferred.succeed(done, undefined).pipe(Effect.ignore)
      })

      const readToolCall = Effect.fn("SessionProcessor.readToolCall")(function* (toolCallID: string) {
        const call = ctx.toolcalls[toolCallID]
        if (!call) return undefined
        const part = yield* session.getPart({
          partID: call.partID,
          messageID: call.messageID,
          sessionID: call.sessionID,
        })
        if (!part || part.type !== "tool") {
          yield* finishToolCall(toolCallID)
          return undefined
        }
        return { call, part }
      })

      const applyToolSettlement = Effect.fn("SessionProcessor.applyToolSettlement")(function* (
        toolCallID: string,
        settlement: ToolSettlement,
      ) {
        const match = yield* readToolCall(toolCallID)
        if (!match) {
          if (!settled.has(toolCallID)) pendingSettlements.set(toolCallID, settlement)
          return
        }
        if (match.part.state.status === "pending") {
          pendingSettlements.set(toolCallID, settlement)
          return
        }
        if (match.part.state.status !== "running") {
          yield* finishToolCall(toolCallID)
          return
        }
        if (settlement.type === "success") {
          yield* session.updatePart({
            ...match.part,
            state: {
              status: "completed",
              input: match.part.state.input,
              output: settlement.output.output,
              metadata: settlement.output.metadata,
              title: settlement.output.title,
              time: { start: match.part.state.time.start, end: Date.now() },
              attachments: settlement.output.attachments,
            },
          })
          yield* finishToolCall(toolCallID)
          return
        }
        yield* session.updatePart({
          ...match.part,
          state: {
            status: "error",
            input: match.part.state.input,
            error: errorMessage(settlement.error),
            // Keep metadata streamed while running so failures retain progress detail (e.g. execute's child calls).
            metadata: match.part.state.metadata,
            time: { start: match.part.state.time.start, end: Date.now() },
          },
        })
        if (
          settlement.error instanceof PermissionV1.RejectedError ||
          settlement.error instanceof Question.RejectedError
        ) {
          ctx.blocked = ctx.shouldBreak
        }
        yield* finishToolCall(toolCallID)
      })

      const updateToolCall = Effect.fn("SessionProcessor.updateToolCall")(function* (
        toolCallID: string,
        update: (part: SessionV1.ToolPart) => SessionV1.ToolPart,
      ) {
        return yield* toolLock.withPermits(1)(
          Effect.gen(function* () {
            const match = yield* readToolCall(toolCallID)
            if (!match) return undefined
            const part = yield* session.updatePart(update(match.part))
            ctx.toolcalls[toolCallID] = {
              ...match.call,
              partID: part.id,
              messageID: part.messageID,
              sessionID: part.sessionID,
            }
            const pending = part.state.status === "running" ? pendingSettlements.get(toolCallID) : undefined
            if (pending) {
              pendingSettlements.delete(toolCallID)
              yield* applyToolSettlement(toolCallID, pending)
            }
            return part
          }),
        )
      })

      const normalizeToolOutput = Effect.fn("SessionProcessor.normalizeToolOutput")(function* (output: ToolOutput) {
        const normalized = yield* Effect.forEach(output.attachments ?? [], (attachment) =>
          attachment.mime.startsWith("image/")
            ? image.normalize(attachment).pipe(
                Effect.catchIf(
                  (error) => error instanceof Image.ResizerUnavailableError,
                  () => Effect.succeed(attachment),
                ),
                Effect.exit,
              )
            : Effect.succeed(Exit.succeed<SessionV1.FilePart>(attachment)),
        )
        const omitted = normalized.filter(Exit.isFailure).length
        const attachments = normalized.filter(Exit.isSuccess).map((item) => item.value)
        return {
          ...output,
          output:
            omitted === 0
              ? output.output
              : `${output.output}\n\n[${omitted} image${omitted === 1 ? "" : "s"} omitted: could not be resized below the image size limit.]`,
          attachments: attachments.length ? attachments : undefined,
        }
      })

      const completeToolCall = Effect.fn("SessionProcessor.completeToolCall")(function* (
        toolCallID: string,
        output: ToolOutput,
      ) {
        const claimed = yield* toolLock.withPermits(1)(
          Effect.sync(() => {
            if (settled.has(toolCallID) || settling.has(toolCallID) || pendingSettlements.has(toolCallID)) return false
            settling.add(toolCallID)
            return true
          }),
        )
        if (!claimed) return
        const normalized = yield* normalizeToolOutput(output)
        yield* toolLock.withPermits(1)(
          Effect.gen(function* () {
            settling.delete(toolCallID)
            if (settled.has(toolCallID)) return
            yield* applyToolSettlement(toolCallID, { type: "success", output: normalized })
          }),
        )
      })

      const failToolCall = Effect.fn("SessionProcessor.failToolCall")(function* (
        toolCallID: string,
        error: unknown,
        local = true,
      ) {
        yield* toolLock.withPermits(1)(
          Effect.gen(function* () {
            if (settled.has(toolCallID) || settling.has(toolCallID) || pendingSettlements.has(toolCallID)) return
            if (local) {
              ctx.toolFailed = true
              yield* Deferred.succeed(ctx.toolFailure, undefined).pipe(Effect.ignore)
            }
            yield* applyToolSettlement(toolCallID, { type: "failure", error })
          }),
        )
      })

      const finishReasoning = Effect.fn("SessionProcessor.finishReasoning")(function* (reasoningID: string) {
        if (!(reasoningID in ctx.reasoningMap)) return
        // oxlint-disable-next-line no-self-assign -- reactivity trigger
        ctx.reasoningMap[reasoningID].text = ctx.reasoningMap[reasoningID].text
        ctx.reasoningMap[reasoningID].time = { ...ctx.reasoningMap[reasoningID].time, end: Date.now() }
        yield* session.updatePart(ctx.reasoningMap[reasoningID])
        delete ctx.reasoningMap[reasoningID]
      })

      const ensureToolCallUnlocked = Effect.fn("SessionProcessor.ensureToolCallUnlocked")(function* (input: {
        id: string
        name: string
        providerExecuted?: boolean
      }) {
        if (settled.has(input.id)) return undefined
        const existing = yield* readToolCall(input.id)
        if (existing) {
          if (!input.providerExecuted || existing.part.metadata?.providerExecuted) return existing
          const part = yield* session.updatePart({
            ...existing.part,
            metadata: { ...existing.part.metadata, providerExecuted: true },
          })
          ctx.toolcalls[input.id] = {
            ...existing.call,
            partID: part.id,
            messageID: part.messageID,
            sessionID: part.sessionID,
          }
          return { call: ctx.toolcalls[input.id], part }
        }
        const part = yield* session.updatePart({
          id: PartID.ascending(),
          messageID: ctx.assistantMessage.id,
          sessionID: ctx.assistantMessage.sessionID,
          type: "tool",
          tool: input.name,
          callID: input.id,
          state: { status: "pending", input: {}, raw: "" },
          metadata: input.providerExecuted ? { providerExecuted: true } : undefined,
        } satisfies SessionV1.ToolPart)
        ctx.attemptParts.push(part.id)
        ctx.toolcalls[input.id] = {
          done: yield* Deferred.make<void>(),
          awaitSettlement: false,
          partID: part.id,
          messageID: part.messageID,
          sessionID: part.sessionID,
        }
        return { call: ctx.toolcalls[input.id], part }
      })

      const ensureToolCall = Effect.fn("SessionProcessor.ensureToolCall")(function* (input: {
        id: string
        name: string
        providerExecuted?: boolean
      }) {
        return yield* toolLock.withPermits(1)(
          Effect.suspend(() =>
            admissionClosed || activeGeneration === undefined
              ? Effect.succeed(undefined)
              : ensureToolCallUnlocked(input),
          ),
        )
      })

      const startToolCallUnlocked = Effect.fn("SessionProcessor.startToolCallUnlocked")(function* (input: {
        id: string
        name: string
        args: Record<string, unknown>
        providerExecuted?: boolean
        admitted?: boolean
        metadata?: SessionV1.ToolPart["metadata"]
      }) {
        ctx.executed = true
        const match = yield* ensureToolCallUnlocked(input)
        if (!match) return
        match.call.awaitSettlement = input.providerExecuted ? false : input.admitted ? true : match.call.awaitSettlement
        const part = yield* session.updatePart({
          ...match.part,
          tool: input.name,
          state:
            match.part.state.status === "running"
              ? { ...match.part.state, input: input.args }
              : {
                  status: "running",
                  input: input.args,
                  time: { start: Date.now() },
                },
          metadata: match.part.metadata?.providerExecuted
            ? { ...input.metadata, providerExecuted: true }
            : (input.metadata ?? match.part.metadata),
        })
        ctx.toolcalls[input.id] = {
          ...match.call,
          partID: part.id,
          messageID: part.messageID,
          sessionID: part.sessionID,
        }
        const pending = pendingSettlements.get(input.id)
        if (!pending) return
        pendingSettlements.delete(input.id)
        yield* applyToolSettlement(input.id, pending)
      })

      const recordToolCall = (input: Parameters<typeof startToolCallUnlocked>[0]) =>
        toolLock.withPermits(1)(
          Effect.suspend(() =>
            admissionClosed || activeGeneration === undefined ? Effect.void : startToolCallUnlocked(input),
          ),
        )

      const startToolCall = (
        requestedGeneration: number | undefined,
        toolCallID: string,
        name: string,
        args: Record<string, unknown>,
      ) =>
        toolLock.withPermits(1)(
          Effect.gen(function* () {
            if (admissionClosed || requestedGeneration === undefined || requestedGeneration !== activeGeneration) {
              return false
            }
            if (admittedToolCalls.has(toolCallID)) return false
            admittedToolCalls.add(toolCallID)
            yield* startToolCallUnlocked({ id: toolCallID, name, args, admitted: true })
            return true
          }),
        )

      const openToolGeneration = toolLock.withPermits(1)(
        Effect.gen(function* () {
          if (admissionClosed) return yield* Effect.interrupt
          generation += 1
          activeGeneration = generation
          admittedToolCalls.clear()
          return generation
        }),
      )

      const closeToolGeneration = (closing: number) =>
        toolLock.withPermits(1)(
          Effect.sync(() => {
            if (activeGeneration === closing) activeGeneration = undefined
          }),
        )

      const isFilePart = (value: unknown): value is SessionV1.FilePart => Schema.is(SessionV1.FilePart)(value)

      const toolResultOutput = (
        value: Extract<StreamEvent, { type: "tool-result" }>,
      ): { title: string; metadata: Record<string, any>; output: string; attachments?: SessionV1.FilePart[] } => {
        if (isRecord(value.result.value) && typeof value.result.value.output === "string") {
          return {
            title: typeof value.result.value.title === "string" ? value.result.value.title : value.name,
            metadata: isRecord(value.result.value.metadata) ? value.result.value.metadata : {},
            output: value.result.value.output,
            attachments: Array.isArray(value.result.value.attachments)
              ? value.result.value.attachments.filter(isFilePart)
              : undefined,
          }
        }
        return {
          title: value.name,
          metadata: value.result.type === "json" && isRecord(value.result.value) ? value.result.value : {},
          output:
            typeof value.result.value === "string" ? value.result.value : (JSON.stringify(value.result.value) ?? ""),
        }
      }

      const handleEvent = Effect.fnUntraced(function* (value: StreamEvent) {
        switch (value.type) {
          case "reasoning-start":
            if (value.id in ctx.reasoningMap) return
            ctx.reasoningMap[value.id] = {
              id: PartID.ascending(),
              messageID: ctx.assistantMessage.id,
              sessionID: ctx.assistantMessage.sessionID,
              type: "reasoning",
              text: "",
              time: { start: Date.now() },
              metadata: value.providerMetadata,
            }
            ctx.attemptParts.push(ctx.reasoningMap[value.id].id)
            yield* session.updatePart(ctx.reasoningMap[value.id])
            return

          case "reasoning-delta":
            // Match dev: silently drop orphan deltas (no preceding reasoning-start).
            if (!(value.id in ctx.reasoningMap)) return
            ctx.reasoningMap[value.id].text += value.text
            if (value.providerMetadata) ctx.reasoningMap[value.id].metadata = value.providerMetadata
            yield* session.updatePartDelta({
              sessionID: ctx.reasoningMap[value.id].sessionID,
              messageID: ctx.reasoningMap[value.id].messageID,
              partID: ctx.reasoningMap[value.id].id,
              field: "text",
              delta: value.text,
            })
            return

          case "reasoning-end":
            if (value.providerMetadata && value.id in ctx.reasoningMap) {
              ctx.reasoningMap[value.id].metadata = value.providerMetadata
            }
            yield* finishReasoning(value.id)
            return

          case "tool-input-start":
            if (ctx.assistantMessage.summary) {
              throw new Error(`Tool call not allowed while generating summary: ${value.name}`)
            }
            yield* ensureToolCall(value)
            return

          case "tool-input-delta":
            yield* ensureToolCall(value)
            return

          case "tool-input-end": {
            yield* ensureToolCall(value)
            return
          }

          case "tool-call": {
            if (ctx.assistantMessage.summary) {
              throw new Error(`Tool call not allowed while generating summary: ${value.name}`)
            }
            const input = isRecord(value.input) ? value.input : { value: value.input }
            // The call is about to execute, so the turn can no longer be replayed.
            yield* recordToolCall({
              id: value.id,
              name: value.name,
              args: input,
              providerExecuted: value.providerExecuted,
              metadata: value.providerMetadata,
            })

            const parts = yield* MessageV2.parts(ctx.assistantMessage.id).pipe(
              Effect.provideService(Database.Service, database),
            )
            const recentParts = parts.slice(-DOOM_LOOP_THRESHOLD)

            if (
              recentParts.length !== DOOM_LOOP_THRESHOLD ||
              !recentParts.every(
                (part) =>
                  part.type === "tool" &&
                  part.tool === value.name &&
                  part.state.status !== "pending" &&
                  JSON.stringify(part.state.input) === JSON.stringify(input),
              )
            ) {
              return
            }

            const agent = yield* agents.get(ctx.assistantMessage.agent)
            yield* permission.ask({
              permission: "doom_loop",
              patterns: [value.name],
              sessionID: ctx.assistantMessage.sessionID,
              metadata: { tool: value.name, input },
              always: [value.name],
              ruleset: agent.permission,
            })
            return
          }

          case "tool-result": {
            if (value.result.type === "error") {
              yield* failToolCall(value.id, value.result.value, !value.providerExecuted)
              return
            }
            yield* completeToolCall(value.id, toolResultOutput(value))
            return
          }

          case "tool-error": {
            yield* failToolCall(value.id, value.error ?? new Error(value.message))
            return
          }

          case "provider-error":
            throw new Error(value.message)

          case "step-start":
            if (!ctx.snapshot) ctx.snapshot = yield* snapshot.track()
            yield* session.updatePart({
              id: PartID.ascending(),
              messageID: ctx.assistantMessage.id,
              sessionID: ctx.sessionID,
              snapshot: ctx.snapshot,
              type: "step-start",
            })
            return

          case "step-finish": {
            const completedSnapshot = yield* snapshot.track()
            yield* Effect.forEach(Object.keys(ctx.reasoningMap), finishReasoning)
            const usage = Session.getUsage({
              model: ctx.model,
              usage: value.usage ?? new Usage({}),
              metadata: value.providerMetadata,
            })
            ctx.assistantMessage.finish = value.reason
            ctx.assistantMessage.cost += usage.cost
            ctx.assistantMessage.tokens = usage.tokens
            yield* session.updatePart({
              id: PartID.ascending(),
              reason: value.reason,
              snapshot: completedSnapshot,
              messageID: ctx.assistantMessage.id,
              sessionID: ctx.assistantMessage.sessionID,
              type: "step-finish",
              tokens: usage.tokens,
              cost: usage.cost,
            })
            yield* session.updateMessage(ctx.assistantMessage)
            if (ctx.snapshot) {
              const patch = yield* snapshot.patch(ctx.snapshot)
              if (patch.files.length) {
                yield* session.updatePart({
                  id: PartID.ascending(),
                  messageID: ctx.assistantMessage.id,
                  sessionID: ctx.sessionID,
                  type: "patch",
                  hash: patch.hash,
                  files: patch.files,
                })
              }
              ctx.snapshot = undefined
            }
            yield* summary
              .summarize({
                sessionID: ctx.sessionID,
                messageID: ctx.assistantMessage.parentID,
              })
              .pipe(Effect.ignore, Effect.forkIn(scope))
            if (!ctx.assistantMessage.summary) {
              const cfg = yield* config.get()
              const overflow = checkOverflow({ cfg, tokens: usage.tokens, model: ctx.model })
              if (overflow.status === "impossible") {
                ctx.assistantMessage.error = new SessionV1.ContextOverflowError({
                  message: overflow.budget.message,
                }).toObject()
                ctx.assistantMessage.finish = "error"
                yield* session.updateMessage(ctx.assistantMessage)
                yield* events.publish(Session.Event.Error, {
                  sessionID: ctx.sessionID,
                  error: ctx.assistantMessage.error,
                })
                return
              }
              if (overflow.overflow) {
                const messages = MessageV2.filterCompacted(yield* session.messages({ sessionID: ctx.sessionID }))
                const manuallyCompacted = messages.some((message) =>
                  message.parts.some((part) => compactionOf(part)?.native === false),
                )
                const current = manuallyCompacted
                  ? yield* MessageV2.toModelMessagesEffect(messages, ctx.model).pipe(
                      Effect.map((messages) =>
                        estimateRequest({
                          system: ctx.request?.system ?? [],
                          messages,
                          tools: ctx.request?.tools,
                          reminderHeadroom: REQUEST_REMINDER_HEADROOM,
                        }),
                      ),
                      Effect.map((tokens) =>
                        checkOverflow({
                          cfg,
                          tokens: {
                            total: tokens,
                            input: tokens,
                            output: 0,
                            reasoning: 0,
                            cache: { read: 0, write: 0 },
                          },
                          model: ctx.model,
                        }),
                      ),
                    )
                  : overflow
                if (current.overflow) ctx.needsCompaction = true
              }
            }
            return
          }

          case "text-start":
            ctx.currentText = {
              id: PartID.ascending(),
              messageID: ctx.assistantMessage.id,
              sessionID: ctx.assistantMessage.sessionID,
              type: "text",
              text: "",
              time: { start: Date.now() },
              metadata: value.providerMetadata,
            }
            ctx.attemptParts.push(ctx.currentText.id)
            yield* session.updatePart(ctx.currentText)
            return

          case "text-delta":
            if (!ctx.currentText) return
            ctx.currentText.text += value.text
            if (value.providerMetadata) ctx.currentText.metadata = value.providerMetadata
            yield* session.updatePartDelta({
              sessionID: ctx.currentText.sessionID,
              messageID: ctx.currentText.messageID,
              partID: ctx.currentText.id,
              field: "text",
              delta: value.text,
            })
            return

          case "text-end":
            if (!ctx.currentText) return
            // oxlint-disable-next-line no-self-assign -- reactivity trigger
            ctx.currentText.text = ctx.currentText.text
            ctx.currentText.text = (yield* plugin.trigger(
              "experimental.text.complete",
              {
                sessionID: ctx.sessionID,
                messageID: ctx.assistantMessage.id,
                partID: ctx.currentText.id,
              },
              { text: ctx.currentText.text },
            )).text
            {
              const end = Date.now()
              ctx.currentText.time = { start: ctx.currentText.time?.start ?? end, end }
            }
            if (value.providerMetadata) ctx.currentText.metadata = value.providerMetadata
            yield* session.updatePart(ctx.currentText)
            ctx.currentText = undefined
            return

          case "finish":
            return
        }
      })

      const cleanup = Effect.fn("SessionProcessor.cleanup")(function* () {
        yield* toolLock.withPermits(1)(
          Effect.sync(() => {
            admissionClosed = true
            activeGeneration = undefined
          }),
        )
        if (ctx.snapshot) {
          const patch = yield* snapshot.patch(ctx.snapshot)
          if (patch.files.length) {
            yield* session.updatePart({
              id: PartID.ascending(),
              messageID: ctx.assistantMessage.id,
              sessionID: ctx.sessionID,
              type: "patch",
              hash: patch.hash,
              files: patch.files,
            })
          }
          ctx.snapshot = undefined
        }

        if (ctx.currentText) {
          const end = Date.now()
          ctx.currentText.time = { start: ctx.currentText.time?.start ?? end, end }
          yield* session.updatePart(ctx.currentText)
          ctx.currentText = undefined
        }

        for (const part of Object.values(ctx.reasoningMap)) {
          const end = Date.now()
          yield* session.updatePart({
            ...part,
            time: { start: part.time.start ?? end, end },
          })
        }
        ctx.reasoningMap = {}

        yield* Effect.forEach(
          Object.values(ctx.toolcalls),
          (call) => Deferred.await(call.done).pipe(Effect.timeout("250 millis"), Effect.ignore),
          { concurrency: "unbounded" },
        )

        yield* toolLock.withPermits(1)(
          Effect.gen(function* () {
            for (const toolCallID of Object.keys(ctx.toolcalls)) {
              const match = yield* readToolCall(toolCallID)
              if (!match) continue
              const part = match.part
              const end = Date.now()
              const metadata = "metadata" in part.state && isRecord(part.state.metadata) ? part.state.metadata : {}
              yield* session.updatePart({
                ...part,
                state: {
                  ...part.state,
                  status: "error",
                  error: "Tool execution aborted",
                  metadata: { ...metadata, interrupted: true },
                  time: { start: "time" in part.state ? part.state.time.start : end, end },
                },
              })
              yield* finishToolCall(toolCallID)
            }
            for (const toolCallID of [...pendingSettlements.keys(), ...settling]) settled.add(toolCallID)
            pendingSettlements.clear()
            settling.clear()
          }),
        )
        ctx.assistantMessage.time.completed = Date.now()
        yield* session.updateMessage(ctx.assistantMessage)
      })

      const halt = Effect.fn("SessionProcessor.halt")(function* (e: unknown) {
        toolController.abort()
        yield* Effect.logError("process", {
          "session.id": input.sessionID,
          messageID: input.assistantMessage.id,
          error: errorMessage(e),
          stack: e instanceof Error ? e.stack : undefined,
        })
        const error = parse(e)
        if (SessionV1.ContextOverflowError.isInstance(error)) {
          if ((yield* config.get()).compaction?.auto === false && !ctx.assistantMessage.summary) {
            ctx.assistantMessage.error = error
            ctx.assistantMessage.finish = "error"
            yield* events.publish(Session.Event.Error, { sessionID: ctx.sessionID, error })
            yield* status.set(ctx.sessionID, { type: "idle" })
            return
          }
          ctx.needsCompaction = true
          yield* events.publish(Session.Event.Error, { sessionID: ctx.sessionID, error })
          return
        }
        ctx.assistantMessage.error = error
        yield* events.publish(Session.Event.Error, {
          sessionID: ctx.assistantMessage.sessionID,
          error: ctx.assistantMessage.error,
        })
        yield* status.set(ctx.sessionID, { type: "idle" })
      })

      const process = Effect.fn("SessionProcessor.process")(function* (streamInput: LLM.StreamInput) {
        yield* Effect.logInfo("process", {
          "session.id": input.sessionID,
          messageID: input.assistantMessage.id,
        })
        ctx.needsCompaction = false
        ctx.request = streamInput
        ctx.shouldBreak = (yield* config.get()).experimental?.continue_loop_on_deny !== true

        return yield* Effect.gen(function* () {
          yield* Effect.acquireUseRelease(
            openToolGeneration,
            (attemptGeneration) =>
              Effect.gen(function* () {
                // Drop whatever the previous attempt managed to write. The request is
                // replayed unchanged, so leaving those parts behind would show the
                // truncated output followed by the full replacement.
                yield* Effect.forEach(ctx.attemptParts, (partID) =>
                  session.removePart({
                    sessionID: ctx.assistantMessage.sessionID,
                    messageID: ctx.assistantMessage.id,
                    partID,
                  }),
                )
                ctx.attemptParts = []
                ctx.currentText = undefined
                ctx.reasoningMap = {}
                if (attemptGeneration > 1)
                  yield* toolLock.withPermits(1)(
                    Effect.sync(() => {
                      ctx.toolcalls = {}
                      pendingSettlements.clear()
                      settling.clear()
                      settled.clear()
                    }),
                  )
                yield* status.set(ctx.sessionID, { type: "busy" })
                const stream = llm.stream({
                  ...streamInput,
                  tools: toolsForGeneration(streamInput.tools, attemptGeneration),
                })

                yield* stream.pipe(
                  Stream.tap((event) => handleEvent(event)),
                  Stream.takeUntil(() => ctx.needsCompaction),
                  Stream.runDrain,
                )
              }),
            closeToolGeneration,
          ).pipe(
            Effect.catchCauseIf(
              (cause) => !Cause.hasInterruptsOnly(cause),
              (cause) => Effect.fail(Cause.squash(cause)),
            ),
            Effect.retry({
              schedule: SessionRetry.policy({
                provider: input.model.providerID,
                parse,
                set: (info) => {
                  return status.set(ctx.sessionID, {
                    type: "retry",
                    attempt: info.attempt,
                    message: info.message,
                    action: info.action,
                    next: info.next,
                  })
                },
              }),
              // Text and reasoning written by a failed attempt are removed before
              // the next one, but a tool that already ran cannot be undone.
              while: () => !ctx.executed,
            }),
            Effect.catch((error) => {
              if (!ctx.executed || !SessionRetry.retryable(parse(error), input.model.providerID)) return halt(error)
              return Effect.gen(function* () {
                if (ctx.toolFailed) return yield* halt(error)
                yield* Effect.race(
                  Effect.forEach(
                    Object.values(ctx.toolcalls).filter((call) => call.awaitSettlement),
                    (call) => Deferred.await(call.done),
                    { concurrency: "unbounded", discard: true },
                  ),
                  Deferred.await(ctx.toolFailure),
                )
                if (ctx.toolFailed) return yield* halt(error)
                ctx.assistantMessage.error = undefined
              })
            }),
            Effect.onInterrupt(() =>
              Effect.gen(function* () {
                aborted = true
                toolController.abort()
                if (!ctx.assistantMessage.error) {
                  yield* halt(new DOMException("Aborted", "AbortError"))
                }
              }),
            ),
            Effect.ensuring(cleanup()),
          )

          if (ctx.needsCompaction) return "compact"
          if (ctx.blocked || ctx.assistantMessage.error) return "stop"
          return "continue"
        })
      })

      return {
        get message() {
          return ctx.assistantMessage
        },
        toolSignal: toolController.signal,
        startToolCall,
        updateToolCall,
        completeToolCall,
        failToolCall,
        process,
      } satisfies Handle
    })

    return Service.of({ create })
  }),
)

export const node = LayerNode.make({
  service: Service,
  layer: layer,
  deps: [
    Session.node,
    Config.node,
    Snapshot.node,
    Agent.node,
    LLM.node,
    Permission.node,
    Plugin.node,
    SessionSummary.node,
    SessionStatus.node,
    Image.node,
    EventV2Bridge.node,
    Database.node,
  ],
})

export * as SessionProcessor from "./processor"
