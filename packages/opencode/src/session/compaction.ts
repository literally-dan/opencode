import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { ConfigV1 } from "@opencode-ai/core/v1/config/config"
import { Session } from "./session"
import { SessionID, MessageID, PartID } from "./schema"
import { Provider } from "@/provider/provider"
import { MessageV2 } from "./message-v2"
import { Token } from "@/util/token"
import { SessionProcessor } from "./processor"
import { Agent } from "@/agent/agent"
import { Plugin } from "@/plugin"
import { Config } from "@/config/config"
import { NotFoundError } from "@/storage/storage"

import { Effect, Layer, Context } from "effect"
import { InstanceState } from "@/effect/instance-state"
import {
  budget,
  check as checkOverflow,
  estimateRequest,
  isOverflow as overflow,
  REQUEST_REMINDER_HEADROOM,
  usable,
} from "./overflow"
import { serviceUse } from "@opencode-ai/core/effect/service-use"
import { RuntimeFlags } from "@/effect/runtime-flags"
import { EventV2Bridge } from "@/event-v2-bridge"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { ModelV2 } from "@opencode-ai/core/model"
import { buildPrompt } from "@opencode-ai/core/session/compaction"
import { SessionCompactionEvent } from "@opencode-ai/schema/session-compaction-event"
import {
  compactionInput,
  compactionOf,
  isManualToolCompaction,
  projectionChannel,
  projectionChannelKey,
  withCompactionLock,
} from "./compaction-pruning"

export const Event = SessionCompactionEvent

export const PRUNE_MINIMUM = 20_000
export const PRUNE_PROTECT = 40_000
const TOOL_OUTPUT_MAX_CHARS = 2_000
const COMPACTION_SUMMARY_MAX_CHARS = 4_000
const PRUNE_PROTECTED_TOOLS = ["skill"]
const DEFAULT_TAIL_TURNS = 2
const MIN_PRESERVE_RECENT_TOKENS = 2_000
const MAX_PRESERVE_RECENT_TOKENS = 8_000
const MAX_AUTO_COMPACTION_ATTEMPTS = 3
type Turn = {
  start: number
  end: number
  id: MessageID
}

type Tail = {
  start: number
  id: MessageID
}

type CompletedCompaction = {
  userIndex: number
  assistantIndex: number
  summary: string
}

const truncate = (value: string) =>
  value.length <= TOOL_OUTPUT_MAX_CHARS ? value : `${value.slice(0, TOOL_OUTPUT_MAX_CHARS)}\n[truncated]`

const truncateSummary = (value: string) =>
  value.length <= COMPACTION_SUMMARY_MAX_CHARS ? value : `${value.slice(0, COMPACTION_SUMMARY_MAX_CHARS)}\n[truncated]`

function serializer() {
  const seen = new Set<string>()
  const current = (part: SessionV1.Part, label: string, role: SessionV1.Info["role"]) => {
    const record = compactionOf(part)
    if (!record) return undefined
    const key = projectionChannelKey(record.group ?? part.id, projectionChannel(part, role))
    if (seen.has(key)) return `${label}: [Compacted ${part.id}]`
    seen.add(key)
    const content = record.summary ? truncateSummary(compactionInput(part).trim()) : ""
    return `${label}: [Compacted ${part.id}]${content ? `\n${content}` : ""}`
  }

  return (stored: SessionV1.WithParts) => {
    const message = MessageV2.projectErroredTurns([stored])[0]!
    if (
      message.info.role === "assistant" &&
      message.info.error &&
      SessionV1.ContentFilterError.isInstance(message.info.error)
    )
      return `[Assistant]: ${MessageV2.CONTENT_FILTER_NOTE}`
    if (message.info.role === "user") {
      return message.parts
        .flatMap((part) => {
          if (part.type === "text") {
            if (part.ignored) return []
            const compacted = current(part, "[User]", message.info.role)
            return compacted ? [compacted] : part.text ? [`[User]: ${part.text}`] : []
          }
          if (part.type !== "file") return []
          const compacted = current(part, "[User attachment]", message.info.role)
          return compacted ? [compacted] : [`[Attached ${part.mime}: ${part.filename ?? "file"}]`]
        })
        .join("\n")
    }
    return message.parts
      .flatMap((part) => {
        if (part.type === "text") {
          const compacted = current(part, "[Assistant]", message.info.role)
          return compacted ? [compacted] : part.text ? [`[Assistant]: ${part.text}`] : []
        }
        if (part.type === "reasoning") {
          const compacted = current(part, "[Assistant reasoning]", message.info.role)
          return compacted ? [compacted] : part.text ? [`[Assistant reasoning]: ${part.text}`] : []
        }
        if (part.type === "file") {
          const compacted = current(part, "[Assistant]", message.info.role)
          return compacted ? [compacted] : [`[Attached ${part.mime}: ${part.filename ?? "file"}]`]
        }
        if (part.type !== "tool") return []
        const compacted = current(part, "[Tool]", message.info.role)
        if (compacted) return [compacted]
        const call = `[Assistant tool call]: ${part.tool}(${JSON.stringify(part.state.input)})`
        if (part.state.status === "completed") {
          const attachments = (part.state.attachments ?? []).map(
            (item) => `[Attached ${item.mime}: ${item.filename ?? "file"}]`,
          )
          return [call, `[Tool result]: ${truncate([part.state.output, ...attachments].join("\n"))}`]
        }
        if (part.state.status === "error") return [call, `[Tool error]: ${part.state.error}`]
        return [call]
      })
      .join("\n")
  }
}

function summaryText(message: SessionV1.WithParts) {
  const text = message.parts
    .filter((part): part is SessionV1.TextPart => part.type === "text")
    .map((part) => part.text.trim())
    .filter(Boolean)
    .join("\n\n")
    .trim()
  return text || undefined
}

function completedCompactions(messages: SessionV1.WithParts[]) {
  const users = new Map<MessageID, number>()
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i]
    if (msg.info.role !== "user") continue
    if (!msg.parts.some((part) => part.type === "compaction")) continue
    users.set(msg.info.id, i)
  }

  return messages.flatMap((msg, assistantIndex): CompletedCompaction[] => {
    if (msg.info.role !== "assistant") return []
    if (!msg.info.summary || !msg.info.finish || msg.info.error) return []
    const userIndex = users.get(msg.info.parentID)
    if (userIndex === undefined) return []
    const summary = summaryText(msg)
    if (!summary) return []
    return [{ userIndex, assistantIndex, summary }]
  })
}

function preserveRecentBudget(input: { cfg: ConfigV1.Info; model: Provider.Model }) {
  return (
    input.cfg.compaction?.preserve_recent_tokens ??
    Math.min(MAX_PRESERVE_RECENT_TOKENS, Math.max(MIN_PRESERVE_RECENT_TOKENS, Math.floor(usable(input) * 0.25)))
  )
}

function turns(messages: SessionV1.WithParts[]) {
  const result: Turn[] = []
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i]
    if (msg.info.role !== "user") continue
    if (msg.parts.some((part) => part.type === "compaction")) continue
    result.push({
      start: i,
      end: messages.length,
      id: msg.info.id,
    })
  }
  for (let i = 0; i < result.length - 1; i++) {
    result[i].end = result[i + 1].start
  }
  return result
}

function splitTurn(input: {
  messages: SessionV1.WithParts[]
  turn: Turn
  model: Provider.Model
  budget: number
  estimate: (input: { messages: SessionV1.WithParts[]; model: Provider.Model }) => Effect.Effect<number>
}) {
  return Effect.gen(function* () {
    if (input.budget <= 0) return undefined
    if (input.turn.end - input.turn.start <= 1) return undefined
    for (let start = input.turn.start + 1; start < input.turn.end; start++) {
      const size = yield* input.estimate({
        messages: input.messages.slice(start, input.turn.end),
        model: input.model,
      })
      if (size > input.budget) continue
      return {
        start,
        id: input.messages[start]!.info.id,
      } satisfies Tail
    }
    return undefined
  })
}

export interface Interface {
  readonly checkOverflow: (input: {
    tokens: SessionV1.Assistant["tokens"]
    model: Provider.Model
  }) => Effect.Effect<ReturnType<typeof checkOverflow>>
  readonly isOverflow: (input: {
    tokens: SessionV1.Assistant["tokens"]
    model: Provider.Model
  }) => Effect.Effect<boolean>
  readonly prune: (input: { sessionID: SessionID }) => Effect.Effect<void>
  readonly process: (input: {
    parentID: MessageID
    messages: SessionV1.WithParts[]
    sessionID: SessionID
    auto: boolean
    overflow?: boolean
    attempt?: number
    requestOverhead?: number
  }) => Effect.Effect<"continue" | "stop">
  readonly create: (input: {
    sessionID: SessionID
    agent: string
    model: { providerID: ProviderV2.ID; modelID: ModelV2.ID }
    auto: boolean
    overflow?: boolean
  }) => Effect.Effect<void>
}

export class Service extends Context.Service<Service, Interface>()("@opencode/SessionCompaction") {}

export const use = serviceUse(Service)

const layer = Layer.effect(
  Service,
  Effect.gen(function* () {
    const config = yield* Config.Service
    const session = yield* Session.Service
    const agents = yield* Agent.Service
    const plugin = yield* Plugin.Service
    const processors = yield* SessionProcessor.Service
    const provider = yield* Provider.Service
    const events = yield* EventV2Bridge.Service
    const flags = yield* RuntimeFlags.Service

    const isOverflow = Effect.fn("SessionCompaction.isOverflow")(function* (input: {
      tokens: SessionV1.Assistant["tokens"]
      model: Provider.Model
    }) {
      return overflow({
        cfg: yield* config.get(),
        tokens: input.tokens,
        model: input.model,
        outputTokenMax: flags.outputTokenMax,
      })
    })

    const overflowStatus = Effect.fn("SessionCompaction.checkOverflow")(function* (input: {
      tokens: SessionV1.Assistant["tokens"]
      model: Provider.Model
    }) {
      return checkOverflow({
        cfg: yield* config.get(),
        tokens: input.tokens,
        model: input.model,
        outputTokenMax: flags.outputTokenMax,
      })
    })

    const projection = Effect.fnUntraced(function* (input: { messages: SessionV1.WithParts[]; model: Provider.Model }) {
      const msgs = yield* MessageV2.toModelMessagesEffect(input.messages, input.model)
      return JSON.stringify(msgs)
    })

    const estimate = Effect.fn("SessionCompaction.estimate")(function* (input: {
      messages: SessionV1.WithParts[]
      model: Provider.Model
    }) {
      return Token.estimate(yield* projection(input))
    })

    const select = Effect.fn("SessionCompaction.select")(function* (input: {
      messages: SessionV1.WithParts[]
      cfg: ConfigV1.Info
      model: Provider.Model
      requestOverhead: number
    }) {
      const limit = input.cfg.compaction?.tail_turns ?? DEFAULT_TAIL_TURNS
      if (limit <= 0) return { head: input.messages, tail_start_id: undefined }
      const budget = Math.min(
        preserveRecentBudget({ cfg: input.cfg, model: input.model }),
        Math.max(0, usable(input) - input.requestOverhead),
      )
      const all = turns(input.messages)
      if (!all.length) return { head: input.messages, tail_start_id: undefined }
      const recent = limit === undefined ? all : all.slice(-limit)

      let total = 0
      let keep: Tail | undefined
      for (let i = recent.length - 1; i >= 0; i--) {
        const turn = recent[i]!
        // estimate lazily so cost stays proportional to the retained tail, not the whole session
        const size = yield* estimate({
          messages: input.messages.slice(turn.start, turn.end),
          model: input.model,
        })
        if (total + size <= budget) {
          total += size
          keep = { start: turn.start, id: turn.id }
          continue
        }
        const remaining = budget - total
        const split = yield* splitTurn({
          messages: input.messages,
          turn,
          model: input.model,
          budget: remaining,
          estimate,
        })
        if (split) keep = split
        else if (!keep) {
          yield* Effect.logInfo("tail fallback", { budget, size, total })
        }
        break
      }

      if (!keep || keep.start === 0) return { head: input.messages, tail_start_id: undefined }
      return {
        head: input.messages.slice(0, keep.start),
        tail_start_id: keep.id,
      }
    })

    // goes backwards through parts until there are PRUNE_PROTECT tokens worth of tool
    // calls, then erases output of older tool calls to free context space
    const prune = Effect.fn("SessionCompaction.prune")(function* (input: { sessionID: SessionID }) {
      const cfg = yield* config.get()
      if (!cfg.compaction?.prune) return
      return yield* withCompactionLock(
        input.sessionID,
        undefined,
        Effect.gen(function* () {
          yield* Effect.logInfo("pruning")

          const msgs = yield* session
            .messages({ sessionID: input.sessionID })
            .pipe(Effect.catchIf(NotFoundError.isInstance, () => Effect.succeed(undefined)))
          if (!msgs) return
          const current = msgs.findLast((msg) => msg.info.role === "user")
          if (!current || current.info.role !== "user") return
          const model = yield* provider
            .getModel(current.info.model.providerID, current.info.model.modelID)
            .pipe(Effect.orDie)
          const projectedBefore = yield* projection({ messages: msgs, model })

          let total = 0
          let pruned = 0
          const toPrune: SessionV1.ToolPart[] = []
          const projections = new Map<MessageID, string>()
          let turns = 0

          loop: for (let msgIndex = msgs.length - 1; msgIndex >= 0; msgIndex--) {
            const msg = msgs[msgIndex]
            if (msg.info.role === "user") turns++
            if (turns < 2) continue
            if (msg.info.role === "assistant" && msg.info.summary) break loop
            for (let partIndex = msg.parts.length - 1; partIndex >= 0; partIndex--) {
              const part = msg.parts[partIndex]
              if (part.type !== "tool") continue
              if (part.state.status !== "completed") continue
              if (PRUNE_PROTECTED_TOOLS.includes(part.tool)) continue
              if (isManualToolCompaction(part)) continue
              if (part.state.time.compacted !== undefined) break loop
              const cached = projections.get(msg.info.id)
              const original = cached ?? (yield* projection({ messages: [msg], model }))
              if (cached === undefined) projections.set(msg.info.id, original)
              const next = structuredClone(msg)
              const candidate = next.parts.find((item) => item.id === part.id)
              if (candidate?.type !== "tool" || candidate.state.status !== "completed") continue
              candidate.state.time.compacted = Date.now()
              const replacement = yield* projection({ messages: [next], model })
              if (replacement.length >= original.length) continue
              const size = Token.estimate(original) - Token.estimate(replacement)
              if (size <= 0) continue
              total += size
              if (total <= PRUNE_PROTECT) continue
              pruned += size
              toPrune.push(part)
            }
          }

          yield* Effect.logInfo("found", { pruned, total })
          if (pruned <= PRUNE_MINIMUM) return
          const projected = structuredClone(msgs)
          const selected = new Set(toPrune.map((part) => part.id))
          for (const msg of projected) {
            for (const part of msg.parts) {
              if (part.type !== "tool" || part.state.status !== "completed" || !selected.has(part.id)) continue
              part.state.time.compacted = Date.now()
            }
          }
          const projectedAfter = yield* projection({ messages: projected, model })
          const savings = Token.estimate(projectedBefore) - Token.estimate(projectedAfter)
          if (projectedAfter.length >= projectedBefore.length || savings <= PRUNE_MINIMUM) return
          for (const part of toPrune) {
            if (part.state.status === "completed") {
              part.state.time.compacted = Date.now()
              yield* session.updatePart(part)
            }
          }
          yield* Effect.logInfo("pruned", { count: toPrune.length, savings })
        }),
      )
    })

    const processCompaction = Effect.fn("SessionCompaction.process")(function* (input: {
      parentID: MessageID
      messages: SessionV1.WithParts[]
      sessionID: SessionID
      auto: boolean
      overflow?: boolean
      attempt?: number
      requestOverhead?: number
    }) {
      const parent = input.messages.findLast((m) => m.info.id === input.parentID)
      if (!parent || parent.info.role !== "user") {
        throw new Error(`Compaction parent must be a user message: ${input.parentID}`)
      }
      const userMessage = parent.info
      const compactionPart = parent.parts.find((part): part is SessionV1.CompactionPart => part.type === "compaction")

      let messages = input.messages
      let replay:
        | {
            info: SessionV1.User
            parts: SessionV1.Part[]
          }
        | undefined
      if (input.overflow) {
        const idx = input.messages.findIndex((m) => m.info.id === input.parentID)
        for (let i = idx - 1; i >= 0; i--) {
          const msg = input.messages[i]
          if (msg.info.role === "user" && !msg.parts.some((p) => p.type === "compaction")) {
            replay = { info: msg.info, parts: msg.parts }
            messages = input.messages.slice(0, i)
            break
          }
        }
        const hasContent =
          replay && messages.some((m) => m.info.role === "user" && !m.parts.some((p) => p.type === "compaction"))
        if (!hasContent) {
          replay = undefined
          messages = input.messages
        }
      }

      const agent = yield* agents.get("compaction")
      const contextModel = yield* provider
        .getModel(userMessage.model.providerID, userMessage.model.modelID)
        .pipe(Effect.orDie)
      const model = agent.model
        ? yield* provider.getModel(agent.model.providerID, agent.model.modelID).pipe(Effect.orDie)
        : contextModel
      const cfg = yield* config.get()
      const projectedBefore = yield* estimate({ messages: input.messages, model: contextModel })
      const observed = input.messages.findLast(
        (message) => message.info.role === "assistant" && message.info.finish && !message.info.summary,
      )
      const observedTokens =
        observed?.info.role === "assistant"
          ? checkOverflow({
              cfg,
              tokens: observed.info.tokens,
              model: contextModel,
              outputTokenMax: flags.outputTokenMax,
            }).count
          : 0
      const requestOverhead = Math.max(
        REQUEST_REMINDER_HEADROOM,
        input.requestOverhead ?? 0,
        observedTokens - projectedBefore,
      )
      const parentIndex = messages.findIndex((message) => message.info.id === input.parentID)
      const history = compactionPart && parentIndex >= 0 ? messages.slice(0, parentIndex) : messages
      const prior = completedCompactions(history)
      const hidden = new Set(prior.flatMap((item) => [item.userIndex, item.assistantIndex]))
      const previousSummary = prior.at(-1)?.summary
      const visible = history.filter((_, index) => !hidden.has(index))
      const selected = yield* select({
        messages: visible,
        cfg,
        model: contextModel,
        requestOverhead,
      })
      // Allow plugins to inject context or replace compaction prompt.
      const compacting = yield* plugin.trigger(
        "experimental.session.compacting",
        { sessionID: input.sessionID },
        { context: [], prompt: undefined },
      )
      const msgs = structuredClone(MessageV2.projectErroredTurns(selected.head))
      yield* plugin.trigger("experimental.chat.messages.transform", {}, { messages: msgs })
      const serialize = serializer()
      const conversation = msgs.map(serialize).filter(Boolean).join("\n\n")
      const nextPrompt =
        compacting.prompt ??
        [
          buildPrompt({
            previousSummary,
            context: [conversation],
          }),
          ...compacting.context,
        ]
          .filter(Boolean)
          .join("\n\n")
      const request = [
        nextPrompt,
        ...(compacting.prompt ? ["The following is the conversation history:", conversation] : []),
      ]
        .filter(Boolean)
        .join("\n\n")
      const ctx = yield* InstanceState.context
      const msg: SessionV1.Assistant = {
        id: MessageID.ascending(),
        role: "assistant",
        parentID: input.parentID,
        sessionID: input.sessionID,
        mode: "compaction",
        agent: "compaction",
        variant: userMessage.model.variant,
        summary: true,
        path: {
          cwd: ctx.directory,
          root: ctx.worktree,
        },
        cost: 0,
        tokens: {
          output: 0,
          input: 0,
          reasoning: 0,
          cache: { read: 0, write: 0 },
        },
        modelID: model.id,
        providerID: model.providerID,
        time: {
          created: Date.now(),
        },
      }
      yield* session.updateMessage(msg)
      return yield* Effect.gen(function* () {
        const fail = Effect.fnUntraced(function* (message: string, target: SessionV1.Assistant = msg) {
          target.error = new SessionV1.ContextOverflowError({ message }).toObject()
          target.finish = "error"
          target.time.completed = Date.now()
          yield* session.updateMessage(target)
          yield* events
            .publish(Session.Event.Error, { sessionID: input.sessionID, error: target.error })
            .pipe(Effect.orDie)
          return "stop" as const
        })
        const contextBudget = budget({ cfg, model: contextModel, outputTokenMax: flags.outputTokenMax })
        if (contextBudget.status === "impossible") return yield* fail(contextBudget.message)
        const contextAvailable =
          contextBudget.status === "available" ? Math.max(0, contextBudget.tokens - requestOverhead) : 0
        if (contextBudget.status === "available" && contextAvailable === 0) {
          return yield* fail(
            `Compaction cannot leave enough room for the current request's system instructions, tools, and reminder headroom (${requestOverhead} estimated tokens in a ${contextBudget.tokens} token usable budget). Reduce configured tools or choose a larger-context model.`,
          )
        }
        if (
          contextBudget.status === "available" &&
          cfg.compaction?.preserve_recent_tokens !== undefined &&
          cfg.compaction.preserve_recent_tokens >= contextAvailable
        ) {
          return yield* fail(
            `Compaction cannot preserve ${cfg.compaction.preserve_recent_tokens} recent tokens after reserving ${requestOverhead} tokens for fixed request overhead within the model's ${contextBudget.tokens} token usable budget. Lower compaction.preserve_recent_tokens.`,
          )
        }
        if (input.auto && (input.attempt ?? 1) > MAX_AUTO_COMPACTION_ATTEMPTS) {
          return yield* fail(
            `Automatic compaction stopped after ${MAX_AUTO_COMPACTION_ATTEMPTS} attempts without reaching a safe context size. Shorten the latest prompt or compact large content manually.`,
          )
        }
        const summaryBudget = budget({ cfg, model, outputTokenMax: flags.outputTokenMax })
        if (summaryBudget.status === "impossible") return yield* fail(summaryBudget.message)
        const requestTokens = estimateRequest({ system: [], messages: [{ role: "user", content: request }] })
        if (summaryBudget.status === "available" && requestTokens >= summaryBudget.tokens) {
          return yield* fail(
            `Conversation history is too large to summarize safely: the compaction request needs about ${requestTokens} tokens but the compaction model has ${summaryBudget.tokens} usable tokens. Compact large content manually or choose a compaction model with a larger context window.`,
          )
        }
        const before = projectedBefore + requestOverhead
        const processor = yield* processors.create({
          assistantMessage: msg,
          sessionID: input.sessionID,
          model,
        })
        const result = yield* processor.process({
          user: userMessage,
          agent,
          sessionID: input.sessionID,
          tools: {},
          system: [],
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "text",
                  text: request,
                },
              ],
            },
          ],
          model,
        })

        if (result === "compact") {
          return yield* fail(
            replay
              ? "Conversation history too large to compact - exceeds model context limit"
              : "Session too large to compact - context exceeds model limit even after stripping media",
            processor.message,
          )
        }

        const all = yield* session.messages({ sessionID: input.sessionID }).pipe(Effect.orDie)
        const completed = all.find((item) => item.info.id === processor.message.id)
        if (
          result === "continue" &&
          (!completed || completed.info.role !== "assistant" || !completed.info.finish || !summaryText(completed))
        ) {
          return yield* fail(
            "Compaction failed because the model did not return a completed non-empty summary. No history was discarded.",
            processor.message,
          )
        }

        if (result === "continue") {
          const projected = structuredClone(all)
          if (compactionPart && selected.tail_start_id) {
            const boundary = projected.flatMap((message) => message.parts).find((part) => part.id === compactionPart.id)
            if (boundary?.type === "compaction") boundary.tail_start_id = selected.tail_start_id
          }
          const after =
            (yield* estimate({ messages: MessageV2.filterCompacted(projected.toReversed()), model: contextModel })) +
            requestOverhead
          if (after >= before) {
            return yield* fail(
              `Compaction did not reduce the current context (${before} tokens before, ${after} after). Shorten the generated summary or keep more original history.`,
              processor.message,
            )
          }
          if (input.auto && contextBudget.status === "available" && after >= contextBudget.tokens) {
            return yield* fail(
              `Automatic compaction still leaves about ${after} tokens in a ${contextBudget.tokens} token usable budget. Reduce compaction.preserve_recent_tokens, shorten the latest prompt, or compact large content manually.`,
              processor.message,
            )
          }
        }

        if (compactionPart && selected.tail_start_id && compactionPart.tail_start_id !== selected.tail_start_id) {
          yield* session.updatePart({
            ...compactionPart,
            tail_start_id: selected.tail_start_id,
          })
        }

        if (result === "continue" && input.auto) {
          if (replay) {
            const original = replay.info
            const replayMsg = yield* session.updateMessage({
              id: MessageID.ascending(),
              role: "user",
              sessionID: input.sessionID,
              time: { created: Date.now() },
              agent: original.agent,
              model: original.model,
              format: original.format,
              tools: original.tools,
              system: original.system,
            })
            for (const part of replay.parts) {
              if (part.type === "compaction") continue
              const replayPart =
                part.type === "file" && MessageV2.isMedia(part.mime)
                  ? { type: "text" as const, text: `[Attached ${part.mime}: ${part.filename ?? "file"}]` }
                  : part
              yield* session.updatePart({
                ...replayPart,
                id: PartID.ascending(),
                messageID: replayMsg.id,
                sessionID: input.sessionID,
              })
            }
          }

          if (!replay) {
            const info = yield* provider.getProvider(userMessage.model.providerID)
            if (
              (yield* plugin.trigger(
                "experimental.compaction.autocontinue",
                {
                  sessionID: input.sessionID,
                  agent: userMessage.agent,
                  model: yield* provider
                    .getModel(userMessage.model.providerID, userMessage.model.modelID)
                    .pipe(Effect.orDie),
                  provider: {
                    source: info.source,
                    info,
                    options: info.options,
                  },
                  message: userMessage,
                  overflow: input.overflow === true,
                },
                { enabled: true },
              )).enabled
            ) {
              const continueMsg = yield* session.updateMessage({
                id: MessageID.ascending(),
                role: "user",
                sessionID: input.sessionID,
                time: { created: Date.now() },
                agent: userMessage.agent,
                model: userMessage.model,
              })
              const text =
                (input.overflow
                  ? "The previous request exceeded the provider's size limit due to large media attachments. The conversation was compacted and media files were removed from context. If the user was asking about attached images or files, explain that the attachments were too large to process and suggest they try again with smaller or fewer files.\n\n"
                  : "") +
                "Continue if you have next steps, or stop and ask for clarification if you are unsure how to proceed."
              yield* session.updatePart({
                id: PartID.ascending(),
                messageID: continueMsg.id,
                sessionID: input.sessionID,
                type: "text",
                // Internal marker for auto-compaction followups so provider plugins
                // can distinguish them from manual post-compaction user prompts.
                // This is not a stable plugin contract and may change or disappear.
                metadata: { compaction_continue: true },
                synthetic: true,
                text,
                time: {
                  start: Date.now(),
                  end: Date.now(),
                },
              })
            }
          }
        }

        if (processor.message.error) return "stop"
        if (result === "continue") {
          yield* events.publish(Event.Compacted, { sessionID: input.sessionID })
        }
        return result
      }).pipe(
        Effect.onInterrupt(() =>
          Effect.gen(function* () {
            msg.error ??= new SessionV1.AbortedError({ message: "Compaction interrupted" }).toObject()
            msg.finish = "error"
            msg.time.completed ??= Date.now()
            yield* session.updateMessage(msg)
          }),
        ),
      )
    })

    const create = Effect.fn("SessionCompaction.create")(function* (input: {
      sessionID: SessionID
      agent: string
      model: { providerID: ProviderV2.ID; modelID: ModelV2.ID }
      auto: boolean
      overflow?: boolean
    }) {
      const msg = yield* session.updateMessage({
        id: MessageID.ascending(),
        role: "user",
        model: input.model,
        sessionID: input.sessionID,
        agent: input.agent,
        time: { created: Date.now() },
      })
      yield* session.updatePart({
        id: PartID.ascending(),
        messageID: msg.id,
        sessionID: msg.sessionID,
        type: "compaction",
        auto: input.auto,
        overflow: input.overflow,
      })
    })

    return Service.of({
      checkOverflow: overflowStatus,
      isOverflow,
      prune,
      process: processCompaction,
      create,
    })
  }),
)

export const node = LayerNode.make({
  service: Service,
  layer: layer,
  deps: [
    Config.node,
    Session.node,
    Agent.node,
    Plugin.node,
    SessionProcessor.node,
    Provider.node,
    EventV2Bridge.node,
    RuntimeFlags.node,
  ],
})

export * as SessionCompaction from "./compaction"
