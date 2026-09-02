import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { Database } from "@opencode-ai/core/database/database"
import { EventV2 } from "@opencode-ai/core/event"
import { ModelV2 } from "@opencode-ai/core/model"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { statics } from "@opencode-ai/core/schema"
import { SessionAskEvent } from "@opencode-ai/schema/session-ask-event"
import {
  SESSION_ASK_TITLE_MAX_CHARS,
  SESSION_ASK_TOOL_ACTIVITY_MAX_BYTES,
  SessionAskThreadTable,
  SessionAskTurnTable,
  type SessionAskToolActivity,
} from "@opencode-ai/core/session/sql"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { LLMEvent, type LLMEvent as LLMEventType } from "@opencode-ai/llm"
import { Agent } from "@/agent/agent"
import { Config } from "@/config/config"
import { EffectBridge } from "@/effect/bridge"
import { EventV2Bridge } from "@/event-v2-bridge"
import { Identifier } from "@/id/id"
import { Permission } from "@/permission"
import { Plugin } from "@/plugin"
import { Provider } from "@/provider/provider"
import { ProviderTransform } from "@/provider/transform"
import { ToolJsonSchema } from "@/tool/json-schema"
import { ToolRegistry } from "@/tool/registry"
import { jsonSchema, tool as aiTool, type Tool as AITool } from "ai"
import { and, desc, eq, lt, or } from "drizzle-orm"
import { Cause, Context, Deferred, Effect, Layer, Schema, Stream } from "effect"
import { isRecord } from "@/util/record"
import { LLM } from "./llm"
import { LLMRequestPrep } from "./llm/request"
import { Instruction } from "./instruction"
import { MessageV2 } from "./message-v2"
import { budget, estimateRequest } from "./overflow"
import { Session } from "./session"
import { MessageID, PartID, SessionID } from "./schema"
import { SystemPrompt } from "./system"

const MAX_PROVIDER_CALLS = 4
const MAX_HISTORY_TURNS = 20
const MAX_ACTIVITY_ITEMS = 16
const MAX_ACTIVITY_CALL_ID_BYTES = 256
const MAX_ACTIVITY_TOOL_BYTES = 256
const MAX_ACTIVITY_INPUT_BYTES = 8 * 1024
const MAX_ACTIVITY_OUTPUT_BYTES = 16 * 1024
const MAX_ACTIVITY_ERROR_BYTES = 4 * 1024

export const IsolationInstruction =
  "You are answering in an isolated Ask thread. Nothing in the main Session was paused or interrupted. You are seeing a read-only snapshot that can become stale while the main Session continues. Treat current Session state as authoritative. Do not claim that Ask text entered normal Session history or changed its execution."

export const ID = Schema.String.check(Schema.isStartsWith("ask_")).pipe(
  Schema.brand("SessionAskThreadID"),
  statics((schema) => ({ ascending: (id?: string) => schema.make(Identifier.ascending("ask", id)) })),
)
export type ID = typeof ID.Type

export const TurnID = Schema.String.check(Schema.isStartsWith("atn_")).pipe(
  Schema.brand("SessionAskTurnID"),
  statics((schema) => ({ ascending: (id?: string) => schema.make(Identifier.ascending("askTurn", id)) })),
)
export type TurnID = typeof TurnID.Type

export const ToolActivity = Schema.Struct({
  callID: Schema.String,
  tool: Schema.String,
  status: Schema.Literals(["completed", "error"]),
  input: Schema.String,
  output: Schema.optional(Schema.String),
  error: Schema.optional(Schema.String),
})
export type ToolActivity = typeof ToolActivity.Type

export const Thread = Schema.Struct({
  id: ID,
  sessionID: SessionID,
  title: Schema.String,
  time: Schema.Struct({ created: Schema.Number, updated: Schema.Number }),
})
export type Thread = typeof Thread.Type

export const Turn = Schema.Struct({
  id: TurnID,
  threadID: ID,
  question: Schema.String,
  answer: Schema.String,
  toolActivity: Schema.Array(ToolActivity),
  time: Schema.Struct({ created: Schema.Number }),
})
export type Turn = typeof Turn.Type

const ModelRef = Schema.Struct({
  providerID: ProviderV2.ID,
  modelID: ModelV2.ID,
})

export const AskInput = Schema.Struct({
  sessionID: SessionID,
  requestID: Schema.optional(SessionAskEvent.RequestID),
  threadID: Schema.optional(ID),
  question: Schema.String.check(Schema.isMinLength(1)),
  model: Schema.optional(ModelRef),
  agent: Schema.optional(Schema.String),
  variant: Schema.optional(Schema.String),
})
export type AskInput = typeof AskInput.Type

export const AskOutput = Schema.Struct({
  ...Turn.fields,
  requestID: SessionAskEvent.RequestID,
  text: Schema.String,
})
export type AskOutput = typeof AskOutput.Type

export const ThreadsInput = Schema.Struct({
  sessionID: SessionID,
  limit: Schema.optional(Schema.Number),
  before: Schema.optional(Schema.String),
})
export type ThreadsInput = typeof ThreadsInput.Type

export const TurnsInput = Schema.Struct({
  sessionID: SessionID,
  threadID: ID,
  limit: Schema.optional(Schema.Number),
  before: Schema.optional(Schema.String),
})
export type TurnsInput = typeof TurnsInput.Type

export const ThreadsOutput = Schema.Struct({
  items: Schema.Array(Thread),
  more: Schema.Boolean,
  cursor: Schema.optional(Schema.String),
})
export type ThreadsOutput = typeof ThreadsOutput.Type

export const TurnsOutput = Schema.Struct({
  items: Schema.Array(Turn),
  more: Schema.Boolean,
  cursor: Schema.optional(Schema.String),
})
export type TurnsOutput = typeof TurnsOutput.Type

export const CancelInput = Schema.Struct({
  sessionID: SessionID,
  threadID: Schema.optional(ID),
})
export type CancelInput = typeof CancelInput.Type

export class ThreadNotFoundError extends Schema.TaggedErrorClass<ThreadNotFoundError>()(
  "SessionAskThreadNotFoundError",
  {
    sessionID: SessionID,
    threadID: ID,
  },
) {
  override get message() {
    return `Ask thread not found: ${this.threadID}`
  }
}

export class ThreadBusyError extends Schema.TaggedErrorClass<ThreadBusyError>()("SessionAskThreadBusyError", {
  threadID: ID,
}) {
  override get message() {
    return `Ask thread already has an active turn: ${this.threadID}`
  }
}

export class RequestBusyError extends Schema.TaggedErrorClass<RequestBusyError>()("SessionAskRequestBusyError", {
  sessionID: SessionID,
  requestID: SessionAskEvent.RequestID,
}) {
  override get message() {
    return `Ask request already active: ${this.requestID}`
  }
}

export class CurrentMessageMissingError extends Schema.TaggedErrorClass<CurrentMessageMissingError>()(
  "SessionAskCurrentMessageMissingError",
  { sessionID: SessionID, requestID: SessionAskEvent.RequestID, messageID: MessageID },
) {
  override get message() {
    return `Ask message transform produced an invalid current question: ${this.messageID}`
  }
}

export class UnsupportedModelError extends Schema.TaggedErrorClass<UnsupportedModelError>()(
  "SessionAskUnsupportedModelError",
  { providerID: ProviderV2.ID, modelID: ModelV2.ID },
) {
  override get message() {
    return `Ask does not support GitLab workflow model ${this.providerID}/${this.modelID}`
  }
}

export const PermissionMetadata = Schema.Struct({
  requestID: SessionAskEvent.RequestID,
  threadID: ID,
  turnID: TurnID,
  callID: Schema.String,
  tool: Schema.String,
})
export type PermissionMetadata = typeof PermissionMetadata.Type

export const Event = SessionAskEvent

export interface Interface {
  readonly ask: (input: AskInput) => Effect.Effect<AskOutput, unknown>
  readonly threads: (input: ThreadsInput) => Effect.Effect<ThreadsOutput, unknown>
  readonly turns: (input: TurnsInput) => Effect.Effect<TurnsOutput, unknown>
  readonly cancel: (input: CancelInput) => Effect.Effect<void, unknown>
}

export class Service extends Context.Service<Service, Interface>()("@opencode/SessionAsk") {}

type Active = {
  readonly sessionID: SessionID
  readonly requestID: SessionAskEvent.RequestID
  readonly cancelled: Deferred.Deferred<void>
}

const layer = Layer.effect(
  Service,
  Effect.gen(function* () {
    const database = yield* Database.Service
    const { db } = database
    const sessions = yield* Session.Service
    const agents = yield* Agent.Service
    const provider = yield* Provider.Service
    const registry = yield* ToolRegistry.Service
    const permission = yield* Permission.Service
    const llm = yield* LLM.Service
    const sys = yield* SystemPrompt.Service
    const instruction = yield* Instruction.Service
    const config = yield* Config.Service
    const plugin = yield* Plugin.Service
    const events = yield* EventV2Bridge.Service
    // Same-thread exclusion is process-local. Distributed ownership needs a separate clustered execution design.
    const active = new Map<ID, Active>()

    const requireThread = Effect.fnUntraced(function* (sessionID: SessionID, threadID: ID) {
      const row = yield* db
        .select()
        .from(SessionAskThreadTable)
        .where(and(eq(SessionAskThreadTable.id, threadID), eq(SessionAskThreadTable.session_id, sessionID)))
        .get()
        .pipe(Effect.orDie)
      if (!row) return yield* new ThreadNotFoundError({ sessionID, threadID })
      return row
    })

    const turns = Effect.fn("SessionAsk.turns")(function* (input: TurnsInput) {
      yield* requireThread(input.sessionID, input.threadID)
      const limit = pageLimit(input.limit)
      const before = input.before ? decodeTurnCursor(decodeCursor(input.before)) : undefined
      const rows = yield* db
        .select()
        .from(SessionAskTurnTable)
        .where(
          before
            ? and(eq(SessionAskTurnTable.thread_id, input.threadID), lt(SessionAskTurnTable.id, before.id))
            : eq(SessionAskTurnTable.thread_id, input.threadID),
        )
        .orderBy(desc(SessionAskTurnTable.id))
        .limit(limit + 1)
        .all()
        .pipe(Effect.orDie)
      const more = rows.length > limit
      const slice = more ? rows.slice(0, limit) : rows
      const tail = slice.at(-1)
      return {
        items: slice.reverse().map(turnFromRow),
        more,
        cursor: more && tail ? encodeCursor({ id: TurnID.make(tail.id) }) : undefined,
      }
    })

    const threads = Effect.fn("SessionAsk.threads")(function* (input: ThreadsInput) {
      yield* sessions.get(input.sessionID)
      const limit = pageLimit(input.limit)
      const before = input.before ? decodeThreadCursor(decodeCursor(input.before)) : undefined
      const rows = yield* db
        .select()
        .from(SessionAskThreadTable)
        .where(
          before
            ? and(
                eq(SessionAskThreadTable.session_id, input.sessionID),
                or(
                  lt(SessionAskThreadTable.updated_at, before.updated),
                  and(eq(SessionAskThreadTable.updated_at, before.updated), lt(SessionAskThreadTable.id, before.id)),
                ),
              )
            : eq(SessionAskThreadTable.session_id, input.sessionID),
        )
        .orderBy(desc(SessionAskThreadTable.updated_at), desc(SessionAskThreadTable.id))
        .limit(limit + 1)
        .all()
        .pipe(Effect.orDie)
      const more = rows.length > limit
      const slice = more ? rows.slice(0, limit) : rows
      const tail = slice.at(-1)
      return {
        items: slice.map(threadFromRow),
        more,
        cursor: more && tail ? encodeCursor({ id: ID.make(tail.id), updated: tail.updated_at }) : undefined,
      }
    })

    const cancel = Effect.fn("SessionAsk.cancel")(function* (input: CancelInput) {
      if (input.threadID && active.get(input.threadID)?.sessionID !== input.sessionID)
        yield* requireThread(input.sessionID, input.threadID)
      const pending = [...active.entries()].filter(
        ([threadID, entry]) => entry.sessionID === input.sessionID && (!input.threadID || threadID === input.threadID),
      )
      yield* Effect.forEach(pending, ([, entry]) => Deferred.succeed(entry.cancelled, undefined), { discard: true })
    })

    const askImpl = Effect.fnUntraced(function* (
      input: AskInput,
      requestID: SessionAskEvent.RequestID,
      threadID: ID,
      turnID: TurnID,
      claimMessageID: MessageID,
    ) {
      const session = yield* sessions.get(input.sessionID)
      const bridge = yield* EffectBridge.make()
      const normal = yield* MessageV2.filterCompactedEffect(input.sessionID).pipe(
        Effect.provideService(Database.Service, database),
      )
      const latest = MessageV2.latest(normal).user
      const agentName = input.agent ?? latest?.agent ?? session.agent ?? (yield* agents.defaultAgent())
      const agent = (yield* agents.list()).find((item) => item.name === agentName)
      if (!agent) throw new Error(`Agent not found: "${agentName}"`)
      const selected =
        input.model ??
        (session.model ? { providerID: session.model.providerID, modelID: session.model.id } : undefined) ??
        latest?.model ??
        (yield* provider.defaultModel())
      const model = yield* provider.getModel(selected.providerID, selected.modelID)
      if (model.providerID === ProviderV2.ID.make("gitlab") && model.api.id.startsWith("duo-workflow-")) {
        return yield* new UnsupportedModelError({ providerID: model.providerID, modelID: model.id })
      }
      const variant = input.variant ?? (input.model ? undefined : (session.model?.variant ?? latest?.model.variant))
      const previous = input.threadID
        ? yield* db
            .select()
            .from(SessionAskTurnTable)
            .where(eq(SessionAskTurnTable.thread_id, threadID))
            .orderBy(desc(SessionAskTurnTable.id))
            .limit(MAX_HISTORY_TURNS)
            .all()
            .pipe(Effect.orDie)
            .pipe(Effect.map((rows) => rows.reverse()))
        : []
      const current = userMessage({
        sessionID: input.sessionID,
        question: input.question,
        agent: agent.name,
        model,
        variant,
        created: Date.now(),
      })
      const currentMessageID = current.info.id
      const priorTurns = previous.map((row) => turnMessages(row, input.sessionID, agent, model, variant))
      const history = priorTurns.flat()
      const activity: SessionAskToolActivity[] = []
      const system = [
        ...(yield* sys.environment(model)),
        ...(yield* instruction.system().pipe(Effect.orDie)),
        IsolationInstruction,
      ]
      const cfg = yield* config.get()
      const synthetic: SessionV1.WithParts[] = [...normal, ...history, current]
      yield* plugin.trigger("experimental.chat.messages.transform", {}, { messages: synthetic })
      const transformedCurrent = synthetic.filter((message) => message.info.id === currentMessageID)
      const roundUser = transformedCurrent.length === 1 ? transformedCurrent[0]?.info : undefined
      if (!roundUser || roundUser.role !== "user")
        return yield* new CurrentMessageMissingError({
          sessionID: input.sessionID,
          requestID,
          messageID: currentMessageID,
        })
      const tools = yield* makeTools({
        registry,
        permission,
        events,
        bridge,
        session,
        agent,
        model,
        requestID,
        threadID,
        turnID,
        claimMessageID,
        messages: synthetic,
        activity,
      })
      const answer: string[] = []

      for (let call = 0; call < MAX_PROVIDER_CALLS; call++) {
        const final = call === MAX_PROVIDER_CALLS - 1
        const enabledTools = final ? {} : tools
        const preparedSystem = yield* LLMRequestPrep.prepareSystem({
          agent,
          model,
          system,
          user: roundUser,
          sessionID: input.sessionID,
          plugin,
        })
        const modelMessages = yield* boundedModelMessages({
          messages: synthetic,
          priorTurns: priorTurns.map((messages) => new Set(messages.map((message) => message.info.id))),
          model,
          system: preparedSystem,
          tools: enabledTools,
          cfg,
        })
        const streamed = yield* llm
          .stream({
            agent,
            user: roundUser,
            permission: session.permission,
            system,
            preparedSystem,
            tools: enabledTools,
            toolChoice: final ? "none" : "auto",
            model,
            sessionID: input.sessionID,
            retries: 2,
            messages: modelMessages,
          })
          .pipe(
            Stream.runCollect,
            Effect.mapError((error) => MessageV2.fromError(error, { providerID: model.providerID })),
          )
        const providerError = streamed.find(LLMEvent.is.providerError)
        if (providerError) {
          return yield* Effect.fail(MessageV2.fromError(providerError.message, { providerID: model.providerID }))
        }
        const round = roundMessage({
          events: streamed,
          sessionID: input.sessionID,
          parentID: roundUser.id,
          agent,
          model,
        })
        answer.push(...streamed.filter(LLMEvent.is.textDelta).map((event) => event.text))
        synthetic.push(round.message)
        if (!round.calledTool || final) break
      }

      const persistedActivity = boundedActivity(activity)
      const created = Date.now()
      const text = answer.join("")
      yield* db
        .transaction((tx) =>
          Effect.gen(function* () {
            if (!input.threadID) {
              yield* tx
                .insert(SessionAskThreadTable)
                .values({
                  id: threadID,
                  session_id: input.sessionID,
                  title: title(input.question),
                  created_at: created,
                  updated_at: created,
                })
                .run()
            }
            yield* tx
              .insert(SessionAskTurnTable)
              .values({
                id: turnID,
                thread_id: threadID,
                question: input.question,
                answer: text,
                tool_activity: persistedActivity,
                created_at: created,
              })
              .run()
            if (input.threadID) {
              yield* tx
                .update(SessionAskThreadTable)
                .set({ updated_at: created })
                .where(
                  and(eq(SessionAskThreadTable.id, threadID), eq(SessionAskThreadTable.session_id, input.sessionID)),
                )
                .run()
            }
          }),
        )
        .pipe(Effect.orDie)
      return {
        id: turnID,
        requestID,
        threadID,
        question: input.question,
        answer: text,
        text,
        toolActivity: persistedActivity,
        time: { created },
      }
    })

    const ask = Effect.fn("SessionAsk.ask")(function* (input: AskInput) {
      if (input.threadID) yield* requireThread(input.sessionID, input.threadID)
      const requestID = input.requestID ?? SessionAskEvent.RequestID.make(crypto.randomUUID())
      const threadID = input.threadID ?? ID.ascending()
      const turnID = TurnID.ascending()
      const claimMessageID = MessageID.ascending()
      const cancelled = yield* Deferred.make<void>()
      const conflict = yield* Effect.sync(() => {
        if ([...active.values()].some((entry) => entry.sessionID === input.sessionID && entry.requestID === requestID))
          return "request" as const
        if (active.has(threadID)) return "thread" as const
        active.set(threadID, { sessionID: input.sessionID, requestID, cancelled })
        return undefined
      })
      if (conflict === "request") return yield* new RequestBusyError({ sessionID: input.sessionID, requestID })
      if (conflict === "thread") return yield* new ThreadBusyError({ threadID })
      return yield* Effect.raceFirst(
        askImpl(input, requestID, threadID, turnID, claimMessageID),
        Deferred.await(cancelled).pipe(Effect.flatMap(() => Effect.interrupt)),
      ).pipe(
        Effect.ensuring(instruction.clear(claimMessageID)),
        Effect.ensuring(
          Effect.sync(() => {
            if (active.get(threadID)?.cancelled === cancelled) active.delete(threadID)
          }),
        ),
      )
    })

    return Service.of({ ask, threads, turns, cancel })
  }),
)

function userMessage(input: {
  sessionID: SessionID
  question: string
  agent: string
  model: Provider.Model
  variant?: string
  created: number
}): SessionV1.WithParts & { info: SessionV1.User } {
  const id = MessageID.ascending()
  return {
    info: {
      id,
      sessionID: input.sessionID,
      role: "user",
      time: { created: input.created },
      agent: input.agent,
      model: { providerID: input.model.providerID, modelID: input.model.id, variant: input.variant },
    },
    parts: [{ id: PartID.ascending(), messageID: id, sessionID: input.sessionID, type: "text", text: input.question }],
  }
}

function turnMessages(
  row: typeof SessionAskTurnTable.$inferSelect,
  sessionID: SessionID,
  agent: Agent.Info,
  model: Provider.Model,
  variant?: string,
): SessionV1.WithParts[] {
  const user = userMessage({
    sessionID,
    question: row.question,
    agent: agent.name,
    model,
    variant,
    created: row.created_at,
  })
  const id = MessageID.ascending()
  return [
    user,
    {
      info: assistantInfo({
        id,
        sessionID,
        parentID: user.info.id,
        agent,
        model,
        created: row.created_at,
        finish: "stop",
      }),
      parts: [{ id: PartID.ascending(), messageID: id, sessionID, type: "text", text: row.answer }],
    },
  ]
}

function roundMessage(input: {
  events: readonly LLMEventType[]
  sessionID: SessionID
  parentID: MessageID
  agent: Agent.Info
  model: Provider.Model
}) {
  const id = MessageID.ascending()
  const created = Date.now()
  const parts: SessionV1.Part[] = []
  const reasoning = new Map<string, SessionV1.ReasoningPart>()
  const text = new Map<string, SessionV1.TextPart>()
  const tools = new Map<string, SessionV1.ToolPart>()
  let finish = "stop"
  let calledTool = false

  for (const event of input.events) {
    switch (event.type) {
      case "reasoning-start": {
        const part: SessionV1.ReasoningPart = {
          id: PartID.ascending(),
          messageID: id,
          sessionID: input.sessionID,
          type: "reasoning",
          text: "",
          time: { start: created },
          metadata: event.providerMetadata,
        }
        reasoning.set(event.id, part)
        parts.push(part)
        break
      }
      case "reasoning-delta": {
        const part = reasoning.get(event.id)
        if (!part) break
        part.text += event.text
        if (event.providerMetadata) part.metadata = event.providerMetadata
        break
      }
      case "reasoning-end": {
        const part = reasoning.get(event.id)
        if (!part) break
        part.time.end = Date.now()
        if (event.providerMetadata) part.metadata = event.providerMetadata
        break
      }
      case "text-start": {
        const part: SessionV1.TextPart = {
          id: PartID.ascending(),
          messageID: id,
          sessionID: input.sessionID,
          type: "text",
          text: "",
          time: { start: created },
          metadata: event.providerMetadata,
        }
        text.set(event.id, part)
        parts.push(part)
        break
      }
      case "text-delta": {
        const part = text.get(event.id)
        if (!part) break
        part.text += event.text
        if (event.providerMetadata) part.metadata = event.providerMetadata
        break
      }
      case "text-end": {
        const part = text.get(event.id)
        if (!part) break
        if (part.time) part.time.end = Date.now()
        if (event.providerMetadata) part.metadata = event.providerMetadata
        break
      }
      case "tool-call": {
        calledTool = true
        const part: SessionV1.ToolPart = {
          id: PartID.ascending(),
          messageID: id,
          sessionID: input.sessionID,
          type: "tool",
          callID: event.id,
          tool: event.name,
          state: {
            status: "running",
            input: toRecord(event.input),
            time: { start: created },
          },
          metadata: event.providerMetadata,
        }
        tools.set(event.id, part)
        parts.push(part)
        break
      }
      case "tool-result": {
        const part = tools.get(event.id)
        if (!part) break
        if (event.result.type === "error") {
          part.state = {
            status: "error",
            input: part.state.input,
            error: outputText(event.result.value),
            time: { start: created, end: Date.now() },
          }
          break
        }
        const result = toolResult(event)
        part.state = {
          status: "completed",
          input: part.state.input,
          title: result.title,
          metadata: result.metadata,
          output: result.output,
          attachments: result.attachments,
          time: { start: created, end: Date.now() },
        }
        break
      }
      case "tool-error": {
        const part = tools.get(event.id)
        if (!part) break
        part.state = {
          status: "error",
          input: part.state.input,
          error: event.message,
          time: { start: created, end: Date.now() },
        }
        break
      }
      case "step-finish":
      case "finish":
        finish = event.reason
        break
    }
  }
  for (const part of reasoning.values()) if (!part.time.end) part.time.end = Date.now()
  for (const part of tools.values()) {
    if (part.state.status !== "running") continue
    part.state = {
      status: "error",
      input: part.state.input,
      error: "Tool execution was interrupted",
      time: { start: created, end: Date.now() },
    }
  }
  return {
    calledTool,
    message: {
      info: assistantInfo({
        id,
        sessionID: input.sessionID,
        parentID: input.parentID,
        agent: input.agent,
        model: input.model,
        created,
        finish,
      }),
      parts,
    } satisfies SessionV1.WithParts,
  }
}

function assistantInfo(input: {
  id: MessageID
  sessionID: SessionID
  parentID: MessageID
  agent: Agent.Info
  model: Provider.Model
  created: number
  finish: string
}): SessionV1.Assistant {
  return {
    id: input.id,
    sessionID: input.sessionID,
    role: "assistant",
    time: { created: input.created, completed: input.created },
    parentID: input.parentID,
    modelID: input.model.id,
    providerID: input.model.providerID,
    mode: input.agent.name,
    agent: input.agent.name,
    path: { cwd: "", root: "" },
    cost: 0,
    tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
    finish: input.finish,
  }
}

function makeTools(input: {
  registry: ToolRegistry.Interface
  permission: Permission.Interface
  events: EventV2.Interface
  bridge: EffectBridge.Shape
  session: Session.Info
  agent: Agent.Info
  model: Provider.Model
  requestID: SessionAskEvent.RequestID
  threadID: ID
  turnID: TurnID
  claimMessageID: MessageID
  messages: SessionV1.WithParts[]
  activity: SessionAskToolActivity[]
}) {
  return Effect.gen(function* () {
    const definitions = yield* input.registry.askTools({ providerID: input.model.providerID })
    return Object.fromEntries(
      definitions.map((definition) => [
        definition.id,
        aiTool({
          description: definition.description,
          inputSchema: jsonSchema(ProviderTransform.schema(input.model, ToolJsonSchema.fromTool(definition))),
          execute: async (args, options) => {
            const signal = options.abortSignal ?? new AbortController().signal
            const activityInput = boundedJson(args, MAX_ACTIVITY_INPUT_BYTES)
            const base = {
              sessionID: input.session.id,
              requestID: input.requestID,
              threadID: input.threadID,
              turnID: input.turnID,
              callID: truncateUtf8(options.toolCallId, MAX_ACTIVITY_CALL_ID_BYTES),
              tool: truncateUtf8(definition.id, MAX_ACTIVITY_TOOL_BYTES),
              input: activityInput,
            }
            const run = Effect.gen(function* () {
              yield* input.events.publish(Event.ToolActivity, { ...base, status: "running" })
              const result = yield* definition.execute(toRecord(args), {
                sessionID: input.session.id,
                messageID: input.claimMessageID,
                agent: input.agent.name,
                abort: signal,
                callID: options.toolCallId,
                messages: input.messages,
                metadata: () => Effect.void,
                ask: (request) =>
                  input.permission
                    .ask({
                      ...request,
                      sessionID: input.session.id,
                      metadata: {
                        ...request.metadata,
                        ask: {
                          requestID: input.requestID,
                          threadID: input.threadID,
                          turnID: input.turnID,
                          callID: options.toolCallId,
                          tool: definition.id,
                        },
                      },
                      ruleset: Permission.merge(input.agent.permission, input.session.permission ?? []),
                    })
                    .pipe(Effect.orDie),
              })
              const completed: SessionAskToolActivity = {
                callID: base.callID,
                tool: base.tool,
                status: "completed",
                input: activityInput,
                output: truncateUtf8(result.output, MAX_ACTIVITY_OUTPUT_BYTES),
              }
              input.activity.push(completed)
              yield* input.events.publish(Event.ToolActivity, {
                ...base,
                status: "completed",
                output: completed.output,
              })
              return result
            }).pipe(
              Effect.tapCause((cause) => {
                const error = truncateUtf8(Cause.pretty(cause), MAX_ACTIVITY_ERROR_BYTES)
                input.activity.push({
                  callID: base.callID,
                  tool: base.tool,
                  status: "error",
                  input: activityInput,
                  error,
                })
                return input.events.publish(Event.ToolActivity, { ...base, status: "error", error })
              }),
            )
            return input.bridge.promise(input.bridge.run(interruptOnAbort(signal, run)))
          },
        }),
      ]),
    )
  })
}

function interruptOnAbort<A, E, R>(signal: AbortSignal, effect: Effect.Effect<A, E, R>) {
  const aborted = Effect.callback<void>((resume) => {
    if (signal.aborted) return resume(Effect.void)
    const handler = () => resume(Effect.void)
    signal.addEventListener("abort", handler, { once: true })
    return Effect.sync(() => signal.removeEventListener("abort", handler))
  })
  return Effect.raceFirst(effect, aborted.pipe(Effect.flatMap(() => Effect.interrupt)))
}

function boundedModelMessages(input: {
  priorTurns: ReadonlyArray<ReadonlySet<MessageID>>
  messages: SessionV1.WithParts[]
  model: Provider.Model
  system: string[]
  tools: Record<string, AITool>
  cfg: Parameters<typeof budget>[0]["cfg"]
}) {
  return Effect.gen(function* () {
    const available = budget({ cfg: input.cfg, model: input.model })
    const removed = new Set<MessageID>()
    let offset = 0
    while (true) {
      const messages = yield* MessageV2.toModelMessagesEffect(
        input.messages.filter((message) => !removed.has(message.info.id)),
        input.model,
        { stampUser: true },
      )
      if (
        available.status !== "available" ||
        estimateRequest({ system: input.system, messages, tools: input.tools }) <= available.tokens ||
        offset >= input.priorTurns.length
      )
        return messages
      for (const id of input.priorTurns[offset] ?? []) removed.add(id)
      offset++
    }
  })
}

function toolResult(event: Extract<LLMEventType, { type: "tool-result" }>) {
  const value = event.result.value
  if (isRecord(value) && typeof value.output === "string") {
    return {
      title: typeof value.title === "string" ? value.title : event.name,
      metadata: isRecord(value.metadata) ? value.metadata : {},
      output: value.output,
      attachments: Array.isArray(value.attachments)
        ? value.attachments.filter((item): item is SessionV1.FilePart => Schema.is(SessionV1.FilePart)(item))
        : undefined,
    }
  }
  return {
    title: event.name,
    metadata: event.result.type === "json" && isRecord(value) ? value : {},
    output: outputText(value),
  }
}

function boundedActivity(activity: SessionAskToolActivity[]) {
  const result = activity.slice(-MAX_ACTIVITY_ITEMS).map((item) => ({
    callID: truncateUtf8(item.callID, MAX_ACTIVITY_CALL_ID_BYTES),
    tool: truncateUtf8(item.tool, MAX_ACTIVITY_TOOL_BYTES),
    status: item.status,
    input: truncateUtf8(item.input, MAX_ACTIVITY_INPUT_BYTES),
    output: item.output === undefined ? undefined : truncateUtf8(item.output, MAX_ACTIVITY_OUTPUT_BYTES),
    error: item.error === undefined ? undefined : truncateUtf8(item.error, MAX_ACTIVITY_ERROR_BYTES),
  }))
  while (new TextEncoder().encode(JSON.stringify(result)).length > SESSION_ASK_TOOL_ACTIVITY_MAX_BYTES) result.shift()
  return result
}

function boundedJson(value: unknown, bytes: number) {
  return truncateUtf8(outputText(value), bytes)
}

function outputText(value: unknown) {
  if (typeof value === "string") return value
  try {
    return JSON.stringify(value) ?? String(value)
  } catch {
    return String(value)
  }
}

function truncateUtf8(value: string, bytes: number) {
  const encoded = new TextEncoder().encode(value)
  if (encoded.length <= bytes) return value
  return new TextDecoder().decode(encoded.slice(0, bytes))
}

function toRecord(value: unknown): Record<string, unknown> {
  if (isRecord(value)) return Object.fromEntries(Object.entries(value))
  return { value }
}

function title(question: string) {
  return question.trim().slice(0, SESSION_ASK_TITLE_MAX_CHARS)
}

function pageLimit(value?: number) {
  if (!value || !Number.isFinite(value)) return 50
  return Math.min(100, Math.max(1, Math.floor(value)))
}

const ThreadCursor = Schema.Struct({ id: ID, updated: Schema.Number })
const TurnCursor = Schema.Struct({ id: TurnID })
const decodeThreadCursor = Schema.decodeUnknownSync(Schema.fromJsonString(ThreadCursor))
const decodeTurnCursor = Schema.decodeUnknownSync(Schema.fromJsonString(TurnCursor))

function encodeCursor(value: object) {
  return Buffer.from(JSON.stringify(value)).toString("base64url")
}

function decodeCursor(value: string) {
  return Buffer.from(value, "base64url").toString("utf8")
}

function threadFromRow(row: typeof SessionAskThreadTable.$inferSelect): Thread {
  return {
    id: ID.make(row.id),
    sessionID: row.session_id,
    title: row.title,
    time: { created: row.created_at, updated: row.updated_at },
  }
}

function turnFromRow(row: typeof SessionAskTurnTable.$inferSelect): Turn {
  return {
    id: TurnID.make(row.id),
    threadID: ID.make(row.thread_id),
    question: row.question,
    answer: row.answer,
    toolActivity: row.tool_activity,
    time: { created: row.created_at },
  }
}

export const node = LayerNode.make({
  service: Service,
  layer,
  deps: [
    Database.node,
    Session.node,
    Agent.node,
    Provider.node,
    ToolRegistry.node,
    Permission.node,
    LLM.node,
    SystemPrompt.node,
    Instruction.node,
    Config.node,
    Plugin.node,
    EventV2Bridge.node,
  ],
})

export * as SessionAsk from "./ask"
