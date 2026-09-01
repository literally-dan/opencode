import { afterEach, describe, expect, mock, test } from "bun:test"
import { ConfigV1 } from "@opencode-ai/core/v1/config/config"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { Database } from "@opencode-ai/core/database/database"
import { EventV2Bridge } from "@/event-v2-bridge"
import { APICallError } from "ai"
import { Cause, Deferred, Effect, Exit, Fiber, Layer, Schema } from "effect"
import * as Stream from "effect/Stream"
import { Config } from "@/config/config"
import { LLM } from "../../src/session/llm"
import { SessionCompaction } from "../../src/session/compaction"
import { compactionLockReferences, withCompactionLock } from "../../src/session/compaction-pruning"
import { estimateRequest, REQUEST_REMINDER_HEADROOM } from "../../src/session/overflow"
import { Token } from "@/util/token"
import { Plugin } from "../../src/plugin"
import { provideTmpdirInstance, TestInstance } from "../fixture/fixture"
import { Session as SessionNs } from "@/session/session"
import { MessageV2 } from "../../src/session/message-v2"
import { MessageID, PartID, SessionID } from "../../src/session/schema"
import { SessionStatus } from "../../src/session/status"
import { SessionSummary } from "../../src/session/summary"
import { SessionV2 } from "@opencode-ai/core/session"
import { SessionExecution } from "@opencode-ai/core/session/execution"
import { SessionProjector } from "@opencode-ai/core/session/projector"

import { Provider } from "@/provider/provider"
import * as SessionProcessorModule from "../../src/session/processor"
import { ProviderTest } from "../fake/provider"
import { pollWithTimeout, testEffect } from "../lib/effect"
import { CrossSpawnSpawner } from "@opencode-ai/core/cross-spawn-spawner"
import { TestConfig } from "../fixture/config"
import { RuntimeFlags } from "@/effect/runtime-flags"
import { LLMEvent, Usage } from "@opencode-ai/llm"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { ModelV2 } from "@opencode-ai/core/model"
import { AppNodeBuilder } from "@opencode-ai/core/effect/app-node-builder"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { Agent } from "@/agent/agent"
import { ReadPartTool } from "@/tool/read_part"
import { Truncate } from "@/tool/truncate"

const summary = Layer.succeed(
  SessionSummary.Service,
  SessionSummary.Service.of({
    summarize: () => Effect.void,
    diff: () => Effect.succeed([]),
    computeDiff: () => Effect.succeed([]),
  }),
)

const ref = {
  providerID: ProviderV2.ID.make("test"),
  modelID: ModelV2.ID.make("test-model"),
}

const usage = (input: ConstructorParameters<typeof Usage>[0]) => new Usage(input)

const basicUsage = () => usage({ inputTokens: 1, outputTokens: 1, totalTokens: 2 })

afterEach(() => {
  mock.restore()
})

function createModel(opts: {
  context: number
  output: number
  input?: number
  cost?: Provider.Model["cost"]
  npm?: string
}): Provider.Model {
  return {
    id: "test-model",
    providerID: "test",
    name: "Test",
    limit: {
      context: opts.context,
      input: opts.input,
      output: opts.output,
    },
    cost: opts.cost ?? { input: 0, output: 0, cache: { read: 0, write: 0 } },
    capabilities: {
      toolcall: true,
      attachment: false,
      reasoning: false,
      temperature: true,
      input: { text: true, image: false, audio: false, video: false },
      output: { text: true, image: false, audio: false, video: false },
    },
    api: { npm: opts.npm ?? "@ai-sdk/anthropic" },
    options: {},
  } as Provider.Model
}

const wide = () => ProviderTest.fake({ model: createModel({ context: 100_000, output: 32_000 }) })

function createUserMessage(sessionID: SessionID, text: string) {
  return Effect.gen(function* () {
    const ssn = yield* SessionNs.Service
    const msg = yield* ssn.updateMessage({
      id: MessageID.ascending(),
      role: "user",
      sessionID,
      agent: "build",
      model: ref,
      time: { created: Date.now() },
    })
    yield* ssn.updatePart({
      id: PartID.ascending(),
      messageID: msg.id,
      sessionID,
      type: "text",
      text,
    })
    return msg
  })
}

function createAssistantMessage(sessionID: SessionID, parentID: MessageID, root: string) {
  return SessionNs.Service.use((ssn) =>
    ssn.updateMessage({
      id: MessageID.ascending(),
      role: "assistant",
      sessionID,
      mode: "build",
      agent: "build",
      path: { cwd: root, root },
      cost: 0,
      tokens: {
        output: 0,
        input: 0,
        reasoning: 0,
        cache: { read: 0, write: 0 },
      },
      modelID: ref.modelID,
      providerID: ref.providerID,
      parentID,
      time: { created: Date.now() },
      finish: "end_turn",
    }),
  )
}

function createStepFinish(sessionID: SessionID, messageID: MessageID, reason = "stop") {
  return SessionNs.Service.use((ssn) =>
    ssn.updatePart({
      id: PartID.ascending(),
      messageID,
      sessionID,
      type: "step-finish",
      reason,
      cost: 0,
      tokens: { input: 1, output: 1, reasoning: 0, cache: { read: 0, write: 0 } },
    }),
  )
}

function createContextReceipt(sessionID: SessionID, messageID: MessageID, output: string) {
  return SessionNs.Service.use((ssn) =>
    ssn.updatePart({
      id: PartID.ascending(),
      messageID,
      sessionID,
      type: "tool",
      callID: `call-list-context-${output.length}`,
      tool: "list_context",
      state: {
        status: "completed",
        input: { compactable_only: false, limit: 80 },
        output,
        title: "12 parts, 7 compactable",
        metadata: { total: 12, compactable: 7, charsCompactable: 34_567 },
        time: { start: 1, end: 2 },
      },
    }),
  )
}

function createSummaryAssistantMessage(sessionID: SessionID, parentID: MessageID, root: string, text: string) {
  return SessionNs.Service.use((ssn) =>
    Effect.gen(function* () {
      const msg = yield* ssn.updateMessage({
        id: MessageID.ascending(),
        role: "assistant",
        sessionID,
        mode: "compaction",
        agent: "compaction",
        path: { cwd: root, root },
        cost: 0,
        tokens: {
          output: 0,
          input: 0,
          reasoning: 0,
          cache: { read: 0, write: 0 },
        },
        modelID: ref.modelID,
        providerID: ref.providerID,
        parentID,
        summary: true,
        time: { created: Date.now() },
        finish: "end_turn",
      })
      yield* ssn.updatePart({
        id: PartID.ascending(),
        messageID: msg.id,
        sessionID,
        type: "text",
        text,
      })
      return msg
    }),
  )
}

function createCompactionMarker(sessionID: SessionID) {
  return SessionNs.Service.use((ssn) =>
    Effect.gen(function* () {
      const msg = yield* ssn.updateMessage({
        id: MessageID.ascending(),
        role: "user",
        model: ref,
        sessionID,
        agent: "build",
        time: { created: Date.now() },
      })
      yield* ssn.updatePart({
        id: PartID.ascending(),
        messageID: msg.id,
        sessionID: msg.sessionID,
        type: "compaction",
        auto: false,
      })
    }),
  )
}

function fake(
  input: Parameters<SessionProcessorModule.SessionProcessor.Interface["create"]>[0],
  result: "continue" | "compact",
) {
  const msg = input.assistantMessage
  return {
    get message() {
      return msg
    },
    toolSignal: new AbortController().signal,
    updateToolCall: Effect.fn("TestSessionProcessor.updateToolCall")(() => Effect.succeed(undefined)),
    completeToolCall: Effect.fn("TestSessionProcessor.completeToolCall")(() => Effect.void),
    failToolCall: Effect.fn("TestSessionProcessor.failToolCall")(() => Effect.void),
    process: Effect.fn("TestSessionProcessor.process")(() => Effect.succeed(result)),
  } satisfies SessionProcessorModule.SessionProcessor.Handle
}

function processorLayer(result: "continue" | "compact") {
  return Layer.succeed(
    SessionProcessorModule.SessionProcessor.Service,
    SessionProcessorModule.SessionProcessor.Service.of({
      create: Effect.fn("TestSessionProcessor.create")((input) => Effect.succeed(fake(input, result))),
    }),
  )
}

function cfg(compaction?: ConfigV1.Info["compaction"]) {
  const base = Schema.decodeUnknownSync(ConfigV1.Info)({}) as ConfigV1.Info
  return Layer.succeed(Config.Service, TestConfig.make({ get: () => Effect.succeed({ ...base, compaction }) }))
}

const defaultProvider = wide()
const compactionTestNode = LayerNode.group([
  SessionCompaction.node,
  SessionNs.node,
  SessionProjector.node,
  Database.node,
  EventV2Bridge.node,
  CrossSpawnSpawner.node,
])
const env = AppNodeBuilder.build(compactionTestNode, [
  [Provider.node, defaultProvider.layer],
  [LLM.node, summaryLLMLayer()],
  [RuntimeFlags.node, RuntimeFlags.layer({ experimentalEventSystem: true })],
])

const it = testEffect(env)

const compactionEnv = AppNodeBuilder.build(
  LayerNode.group([SessionNs.node, SessionProjector.node, Database.node, EventV2Bridge.node, CrossSpawnSpawner.node]),
)
const itCompaction = testEffect(compactionEnv)

type CompactionProcessOptions = {
  result?: "continue" | "compact"
  llm?: Layer.Layer<LLM.Service>
  plugin?: Layer.Layer<Plugin.Service>
  provider?: ReturnType<typeof wide>
  config?: Layer.Layer<Config.Service>
  session?: Layer.Layer<SessionNs.Service>
}

function withCompaction(options?: CompactionProcessOptions) {
  return Effect.provide(compactionProcessLayer(options))
}

function compactionProcessLayer(options?: CompactionProcessOptions) {
  const replacements: LayerNode.Replacements = [
    [Provider.node, (options?.provider ?? wide()).layer],
    [RuntimeFlags.node, RuntimeFlags.layer({ experimentalEventSystem: true })],
    [SessionSummary.node, summary],
    ...(options?.session ? ([[SessionNs.node, options.session]] as const) : []),
  ]
  if (options?.result) {
    return AppNodeBuilder.build(compactionTestNode, [
      ...replacements,
      [SessionProcessorModule.SessionProcessor.node, processorLayer(options.result)],
      ...(options?.plugin ? ([[Plugin.node, options.plugin]] as const) : []),
      ...(options?.config ? ([[Config.node, options.config]] as const) : []),
    ])
  }
  return AppNodeBuilder.build(compactionTestNode, [
    ...replacements,
    [LLM.node, options?.llm ?? summaryLLMLayer()],
    ...(options?.plugin ? ([[Plugin.node, options.plugin]] as const) : []),
    ...(options?.config ? ([[Config.node, options.config]] as const) : []),
  ])
}

function createSummaryCompaction(sessionID: SessionID) {
  return SessionCompaction.use.create({ sessionID, agent: "build", model: ref, auto: false })
}

function readCompactionPart(sessionID: SessionID) {
  return SessionNs.use
    .messages({ sessionID })
    .pipe(
      Effect.map((messages) =>
        messages.at(-2)?.parts.find((item): item is SessionV1.CompactionPart => item.type === "compaction"),
      ),
    )
}

function llm() {
  const queue: Array<
    Stream.Stream<LLMEvent, unknown> | ((input: LLM.StreamInput) => Stream.Stream<LLMEvent, unknown>)
  > = []

  return {
    push(stream: Stream.Stream<LLMEvent, unknown> | ((input: LLM.StreamInput) => Stream.Stream<LLMEvent, unknown>)) {
      queue.push(stream)
    },
    llmLayer: Layer.succeed(
      LLM.Service,
      LLM.Service.of({
        stream: (input) => {
          const item = queue.shift() ?? Stream.empty
          const stream = typeof item === "function" ? item(input) : item
          return stream.pipe(Stream.mapEffect((event) => Effect.succeed(event)))
        },
      }),
    ),
  }
}

function reply(
  text: string,
  capture?: (input: LLM.StreamInput) => void,
): (input: LLM.StreamInput) => Stream.Stream<LLMEvent, unknown> {
  return (input) => {
    capture?.(input)
    return Stream.make(
      LLMEvent.textStart({ id: "txt-0" }),
      LLMEvent.textDelta({ id: "txt-0", text }),
      LLMEvent.textEnd({ id: "txt-0" }),
      LLMEvent.stepFinish({
        index: 0,
        reason: "stop",
        usage: basicUsage(),
      }),
      LLMEvent.finish({
        reason: "stop",
        usage: basicUsage(),
      }),
    )
  }
}

function summaryLLMLayer() {
  return Layer.succeed(
    LLM.Service,
    LLM.Service.of({
      stream: reply("summary"),
    }),
  )
}

function plugin(ready: Deferred.Deferred<void>) {
  return Layer.mock(Plugin.Service)({
    trigger: <Name extends string, Input, Output>(name: Name, _input: Input, output: Output) => {
      if (name !== "experimental.session.compacting") return Effect.succeed(output)
      return Effect.sync(() => Deferred.doneUnsafe(ready, Effect.void)).pipe(
        Effect.andThen(Effect.never),
        Effect.as(output),
      )
    },
    list: () => Effect.succeed([]),
    init: () => Effect.void,
  })
}

function autocontinue(enabled: boolean) {
  return Layer.mock(Plugin.Service)({
    trigger: <Name extends string, Input, Output>(name: Name, _input: Input, output: Output) => {
      if (name !== "experimental.compaction.autocontinue") return Effect.succeed(output)
      return Effect.sync(() => {
        ;(output as { enabled: boolean }).enabled = enabled
        return output
      })
    },
    list: () => Effect.succeed([]),
    init: () => Effect.void,
  })
}

function compactionContext(context: string) {
  return Layer.mock(Plugin.Service)({
    trigger: <Name extends string, Input, Output>(name: Name, _input: Input, output: Output) => {
      if (name !== "experimental.session.compacting") return Effect.succeed(output)
      return Effect.sync(() => {
        ;(output as { context: string[] }).context.push(context)
        return output
      })
    },
    list: () => Effect.succeed([]),
    init: () => Effect.void,
  })
}

describe("session.compaction.isOverflow", () => {
  it.live(
    "returns true when token count exceeds usable context",
    provideTmpdirInstance(() =>
      Effect.gen(function* () {
        const compact = yield* SessionCompaction.Service
        const model = createModel({ context: 100_000, output: 32_000 })
        const tokens = { input: 75_000, output: 5_000, reasoning: 0, cache: { read: 0, write: 0 } }
        expect(yield* compact.isOverflow({ tokens, model })).toBe(true)
      }),
    ),
  )

  it.live(
    "returns false when token count within usable context",
    provideTmpdirInstance(() =>
      Effect.gen(function* () {
        const compact = yield* SessionCompaction.Service
        const model = createModel({ context: 200_000, output: 32_000 })
        const tokens = { input: 100_000, output: 10_000, reasoning: 0, cache: { read: 0, write: 0 } }
        expect(yield* compact.isOverflow({ tokens, model })).toBe(false)
      }),
    ),
  )

  it.live(
    "includes cache.read in token count",
    provideTmpdirInstance(() =>
      Effect.gen(function* () {
        const compact = yield* SessionCompaction.Service
        const model = createModel({ context: 100_000, output: 32_000 })
        const tokens = { input: 60_000, output: 10_000, reasoning: 0, cache: { read: 10_000, write: 0 } }
        expect(yield* compact.isOverflow({ tokens, model })).toBe(true)
      }),
    ),
  )

  it.live(
    "respects input limit for input caps",
    provideTmpdirInstance(() =>
      Effect.gen(function* () {
        const compact = yield* SessionCompaction.Service
        const model = createModel({ context: 400_000, input: 272_000, output: 128_000 })
        const tokens = { input: 271_000, output: 1_000, reasoning: 0, cache: { read: 2_000, write: 0 } }
        expect(yield* compact.isOverflow({ tokens, model })).toBe(true)
      }),
    ),
  )

  it.live(
    "returns false when input/output are within input caps",
    provideTmpdirInstance(() =>
      Effect.gen(function* () {
        const compact = yield* SessionCompaction.Service
        const model = createModel({ context: 400_000, input: 272_000, output: 128_000 })
        const tokens = { input: 200_000, output: 20_000, reasoning: 0, cache: { read: 10_000, write: 0 } }
        expect(yield* compact.isOverflow({ tokens, model })).toBe(false)
      }),
    ),
  )

  it.live(
    "returns false when output within limit with input caps",
    provideTmpdirInstance(() =>
      Effect.gen(function* () {
        const compact = yield* SessionCompaction.Service
        const model = createModel({ context: 200_000, input: 120_000, output: 10_000 })
        const tokens = { input: 50_000, output: 9_999, reasoning: 0, cache: { read: 0, write: 0 } }
        expect(yield* compact.isOverflow({ tokens, model })).toBe(false)
      }),
    ),
  )

  // ─── Bug reproduction tests ───────────────────────────────────────────
  // These tests demonstrate that when limit.input is set, isOverflow()
  // does not subtract any headroom for the next model response. This means
  // compaction only triggers AFTER we've already consumed the full input
  // budget, leaving zero room for the next API call's output tokens.
  //
  // Compare: without limit.input, usable = context - output (reserves space).
  // With limit.input, usable = limit.input (reserves nothing).
  //
  // Related issues: #10634, #8089, #11086, #12621
  // Open PRs: #6875, #12924

  it.live(
    "BUG: no headroom when limit.input is set — compaction should trigger near boundary but does not",
    provideTmpdirInstance(() =>
      Effect.gen(function* () {
        const compact = yield* SessionCompaction.Service
        // Simulate Claude with prompt caching: input limit = 200K, output limit = 32K
        const model = createModel({ context: 200_000, input: 200_000, output: 32_000 })

        // We've used 198K tokens total. Only 2K under the input limit.
        // On the next turn, the full conversation (198K) becomes input,
        // plus the model needs room to generate output — this WILL overflow.
        const tokens = { input: 180_000, output: 15_000, reasoning: 0, cache: { read: 3_000, write: 0 } }
        // count = 180K + 3K + 15K = 198K
        // usable = limit.input = 200K (no output subtracted!)
        // 198K > 200K = false → no compaction triggered

        // WITHOUT limit.input: usable = 200K - 32K = 168K, and 198K > 168K = true ✓
        // WITH limit.input: usable = 200K, and 198K > 200K = false ✗

        // With 198K used and only 2K headroom, the next turn will overflow.
        // Compaction MUST trigger here.
        expect(yield* compact.isOverflow({ tokens, model })).toBe(true)
      }),
    ),
  )

  it.live(
    "BUG: without limit.input, same token count correctly triggers compaction",
    provideTmpdirInstance(() =>
      Effect.gen(function* () {
        const compact = yield* SessionCompaction.Service
        // Same model but without limit.input — uses context - output instead
        const model = createModel({ context: 200_000, output: 32_000 })

        // Same token usage as above
        const tokens = { input: 180_000, output: 15_000, reasoning: 0, cache: { read: 3_000, write: 0 } }
        // count = 198K
        // usable = context - output = 200K - 32K = 168K
        // 198K > 168K = true → compaction correctly triggered

        const result = yield* compact.isOverflow({ tokens, model })
        expect(result).toBe(true) // ← Correct: headroom is reserved
      }),
    ),
  )

  it.live(
    "BUG: asymmetry — limit.input model allows 30K more usage before compaction than equivalent model without it",
    provideTmpdirInstance(() =>
      Effect.gen(function* () {
        const compact = yield* SessionCompaction.Service
        // Two models with identical context/output limits, differing only in limit.input
        const withInputLimit = createModel({ context: 200_000, input: 200_000, output: 32_000 })
        const withoutInputLimit = createModel({ context: 200_000, output: 32_000 })

        // 170K total tokens — well above context-output (168K) but below input limit (200K)
        const tokens = { input: 166_000, output: 10_000, reasoning: 0, cache: { read: 5_000, write: 0 } }

        const withLimit = yield* compact.isOverflow({ tokens, model: withInputLimit })
        const withoutLimit = yield* compact.isOverflow({ tokens, model: withoutInputLimit })

        // Both models have identical real capacity — they should agree:
        expect(withLimit).toBe(true) // should compact (170K leaves no room for 32K output)
        expect(withoutLimit).toBe(true) // correctly compacts (170K > 168K)
      }),
    ),
  )

  it.live(
    "returns false when model context limit is 0",
    provideTmpdirInstance(() =>
      Effect.gen(function* () {
        const compact = yield* SessionCompaction.Service
        const model = createModel({ context: 0, output: 32_000 })
        const tokens = { input: 100_000, output: 10_000, reasoning: 0, cache: { read: 0, write: 0 } }
        expect(yield* compact.isOverflow({ tokens, model })).toBe(false)
      }),
    ),
  )

  it.live(
    "returns false when compaction.auto is disabled",
    provideTmpdirInstance(
      () =>
        Effect.gen(function* () {
          const compact = yield* SessionCompaction.Service
          const model = createModel({ context: 100_000, output: 32_000 })
          const tokens = { input: 75_000, output: 5_000, reasoning: 0, cache: { read: 0, write: 0 } }
          expect(yield* compact.isOverflow({ tokens, model })).toBe(false)
        }),
      {
        config: {
          compaction: { auto: false },
        },
      },
    ),
  )

  it.live(
    "returns an explicit impossible status when reserved tokens consume the input limit",
    provideTmpdirInstance(
      () =>
        Effect.gen(function* () {
          const compact = yield* SessionCompaction.Service
          const model = createModel({ context: 400_000, input: 200_000, output: 32_000 })
          const tokens = { input: 10, output: 0, reasoning: 0, cache: { read: 0, write: 0 } }

          const status = yield* compact.checkOverflow({ tokens, model })

          expect(status.status).toBe("impossible")
          expect(status.overflow).toBe(false)
          if (status.status === "impossible") expect(status.budget.message).toContain("leaves no usable input space")
          expect(yield* compact.isOverflow({ tokens, model })).toBe(false)
        }),
      { config: { compaction: { reserved: 200_000 } } },
    ),
  )

  it.live(
    "returns disabled before validating an impossible budget when auto compaction is off",
    provideTmpdirInstance(
      () =>
        Effect.gen(function* () {
          const compact = yield* SessionCompaction.Service
          const model = createModel({ context: 400_000, input: 200_000, output: 32_000 })
          const tokens = { input: 300_000, output: 0, reasoning: 0, cache: { read: 0, write: 0 } }

          const status = yield* compact.checkOverflow({ tokens, model })

          expect(status.status).toBe("disabled")
          expect(status.overflow).toBe(false)
        }),
      { config: { compaction: { auto: false, reserved: 200_000 } } },
    ),
  )

  it.live(
    "uses the larger valid cache-inclusive count when provider totals conflict",
    provideTmpdirInstance(() =>
      Effect.gen(function* () {
        const compact = yield* SessionCompaction.Service
        const model = createModel({ context: 100_000, output: 20_000 })
        const tokens = {
          total: 1,
          input: 50_000,
          output: 5_000,
          reasoning: 0,
          cache: { read: 30_000, write: 0 },
        }

        const status = yield* compact.checkOverflow({ tokens, model })

        expect(status.count).toBe(85_000)
        expect(status.overflow).toBe(true)
      }),
    ),
  )

  it.live(
    "includes reasoning when the provider total is missing or malformed",
    provideTmpdirInstance(() =>
      Effect.gen(function* () {
        const compact = yield* SessionCompaction.Service
        const model = createModel({ context: 100_000, output: 20_000 })
        const tokens = {
          total: Number.NaN,
          input: -20,
          output: 5_000,
          reasoning: 10_000,
          cache: { read: 80_000, write: Number.NaN },
        }

        const status = yield* compact.checkOverflow({ tokens, model })

        expect(status.count).toBe(95_000)
        expect(status.overflow).toBe(true)
      }),
    ),
  )
})

describe("session.compaction.create", () => {
  it.live(
    "creates a compaction user message and part",
    provideTmpdirInstance(() =>
      Effect.gen(function* () {
        const compact = yield* SessionCompaction.Service
        const ssn = yield* SessionNs.Service

        const info = yield* ssn.create({})

        yield* compact.create({
          sessionID: info.id,
          agent: "build",
          model: ref,
          auto: true,
          overflow: true,
        })

        const msgs = yield* ssn.messages({ sessionID: info.id })
        expect(msgs).toHaveLength(1)
        expect(msgs[0].info.role).toBe("user")
        expect(msgs[0].parts).toHaveLength(1)
        expect(msgs[0].parts[0]).toMatchObject({
          type: "compaction",
          auto: true,
          overflow: true,
        })
      }),
    ),
  )

  it.live.skip(
    "projects a compaction message to v2 (v2 projector disabled)",
    provideTmpdirInstance(() =>
      Effect.gen(function* () {
        const compact = yield* SessionCompaction.Service
        const ssn = yield* SessionNs.Service
        const info = yield* ssn.create({})

        yield* compact.create({
          sessionID: info.id,
          agent: "build",
          model: ref,
          auto: true,
          overflow: true,
        })

        const v2 = yield* SessionV2.Service.use((svc) => svc.messages({ sessionID: info.id })).pipe(
          Effect.provide(AppNodeBuilder.build(SessionV2.node, [[SessionExecution.node, SessionExecution.noopLayer]])),
        )
        expect(v2.at(-1)).toMatchObject({
          type: "compaction",
          reason: "auto",
          summary: "",
        })
      }),
    ),
  )
})

describe("session.compaction.receipts", () => {
  it.instance(
    "shows a receipt in full once, then collapses it once without losing recovery data",
    Effect.gen(function* () {
      const compact = yield* SessionCompaction.Service
      const events = yield* EventV2Bridge.Service
      const ssn = yield* SessionNs.Service
      const test = yield* TestInstance
      const session = yield* ssn.create({})
      const user = yield* createUserMessage(session.id, "inspect context")
      const assistant = yield* createAssistantMessage(session.id, user.id, test.directory)
      const receipt = yield* createContextReceipt(session.id, assistant.id, "FULL_CONTEXT_INVENTORY")
      // The step that requested the tool did not consume its result.
      yield* createStepFinish(session.id, assistant.id, "tool-calls")
      let writes = 0
      const unsub = yield* events.listen((event) => {
        if (event.type !== SessionV1.Event.PartUpdated.type) return Effect.void
        if ((event.data as typeof SessionV1.Event.PartUpdated.data.Type).part.id === receipt.id) writes++
        return Effect.void
      })
      yield* Effect.addFinalizer(() => unsub)

      const full = yield* compact.collapseReceipts({ sessionID: session.id })
      const fullProjection = JSON.stringify(yield* MessageV2.toModelMessagesEffect(full, defaultProvider.model))
      expect(fullProjection).toContain("FULL_CONTEXT_INVENTORY")
      expect(writes).toBe(0)

      const nextUser = yield* createUserMessage(session.id, "continue")
      const shell = yield* createAssistantMessage(session.id, nextUser.id, test.directory)
      const preStream = yield* compact.collapseReceipts({ sessionID: session.id })
      const preStreamProjection = JSON.stringify(yield* MessageV2.toModelMessagesEffect(preStream, defaultProvider.model))
      expect(preStreamProjection).toContain("FULL_CONTEXT_INVENTORY")
      expect(writes).toBe(0)

      yield* createStepFinish(session.id, shell.id)
      const bounded = yield* compact.collapseReceipts({ sessionID: session.id })
      const firstProjection = JSON.stringify(yield* MessageV2.toModelMessagesEffect(bounded, defaultProvider.model))
      expect(firstProjection).not.toContain("FULL_CONTEXT_INVENTORY")
      expect(firstProjection).toContain("total parts=12; compactable parts=7; estimated reclaimable characters=34567")
      expect(firstProjection).toContain(receipt.id)
      expect(writes).toBe(1)

      const repeated = yield* compact.collapseReceipts({ sessionID: session.id })
      const repeatedProjection = JSON.stringify(yield* MessageV2.toModelMessagesEffect(repeated, defaultProvider.model))
      expect(repeatedProjection).toBe(firstProjection)
      expect(writes).toBe(1)

      const stored = yield* ssn.findPart({ sessionID: session.id, partID: receipt.id })
      expect(stored).toMatchObject({
        type: "tool",
        callID: "call-list-context-22",
        tool: "list_context",
        state: {
          status: "completed",
          input: { compactable_only: false, limit: 80 },
          output: "FULL_CONTEXT_INVENTORY",
          metadata: { total: 12, compactable: 7, charsCompactable: 34_567 },
          time: { compacted: expect.any(Number) },
        },
      })
      expect(stored?.type === "tool" && stored.state.status === "completed" && stored.state.compactionGroup).toBeUndefined()
      expect(stored?.type === "tool" && stored.state.status === "completed" && stored.state.compactionSummary).toBeUndefined()
      expect(stored?.type === "tool" && stored.state.status === "completed" && stored.state.compactionGeneration).toBeUndefined()

      const readPart = yield* ReadPartTool.pipe(
        Effect.provide(
          Layer.merge(
            Layer.mock(Agent.Service, {
              get: () => Effect.succeed({ name: "build", mode: "primary", permission: [], options: {} }),
            }),
            Layer.mock(Truncate.Service, {
              output: (text) => Effect.succeed({ content: text, truncated: false }),
            }),
          ),
        ),
        Effect.map((info) => info.init()),
        Effect.flatten,
      )
      const recovered = yield* readPart.execute(
        { part_id: receipt.id },
        {
          sessionID: session.id,
          messageID: nextUser.id,
          agent: "build",
          abort: new AbortController().signal,
          messages: repeated,
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )
      expect(recovered.output).toContain("FULL_CONTEXT_INVENTORY")
      expect(recovered.output).toContain('"compactable_only": false')
    }),
  )

  it.instance(
    "keeps receipts full after crashed or cancelled assistant shells",
    Effect.gen(function* () {
      const compact = yield* SessionCompaction.Service
      const ssn = yield* SessionNs.Service
      const test = yield* TestInstance
      const failures = [
        new SessionV1.APIError({ message: "provider stream crashed", isRetryable: false }).toObject(),
        new SessionV1.AbortedError({ message: "cancelled" }).toObject(),
      ]

      for (const [index, error] of failures.entries()) {
        const session = yield* ssn.create({})
        const user = yield* createUserMessage(session.id, `inspect context ${index}`)
        const assistant = yield* createAssistantMessage(session.id, user.id, test.directory)
        const receipt = yield* createContextReceipt(session.id, assistant.id, `FULL_FAILURE_RECEIPT_${index}`)
        yield* createStepFinish(session.id, assistant.id, "tool-calls")
        const nextUser = yield* createUserMessage(session.id, `continue ${index}`)
        const shell = yield* createAssistantMessage(session.id, nextUser.id, test.directory)
        yield* ssn.updateMessage({
          ...shell,
          finish: "error",
          error,
          time: { ...shell.time, completed: Date.now() },
        })

        const projected = yield* compact.collapseReceipts({ sessionID: session.id })
        const serialized = JSON.stringify(yield* MessageV2.toModelMessagesEffect(projected, defaultProvider.model))
        const stored = yield* ssn.findPart({ sessionID: session.id, partID: receipt.id })
        expect(serialized).toContain(`FULL_FAILURE_RECEIPT_${index}`)
        expect(
          stored?.type === "tool" && stored.state.status === "completed" && stored.state.time.compacted,
        ).toBeUndefined()
      }
    }),
  )

  itCompaction.instance(
    "retries only the unmarked receipt after a partial write failure",
    Effect.gen(function* () {
      const ssn = yield* SessionNs.Service
      const test = yield* TestInstance
      const session = yield* ssn.create({})
      const user = yield* createUserMessage(session.id, "inspect context")
      const assistant = yield* createAssistantMessage(session.id, user.id, test.directory)
      const first = yield* createContextReceipt(session.id, assistant.id, "FIRST_RECEIPT")
      const second = yield* createContextReceipt(session.id, assistant.id, "SECOND_RECEIPT")
      yield* createStepFinish(session.id, assistant.id, "tool-calls")
      const nextUser = yield* createUserMessage(session.id, "continue")
      const completed = yield* createAssistantMessage(session.id, nextUser.id, test.directory)
      yield* createStepFinish(session.id, completed.id)

      let writes = 0
      let fail = true
      const updatePart: SessionNs.Interface["updatePart"] = (part) =>
        Effect.gen(function* () {
          writes++
          if (fail && writes === 2) return yield* Effect.die(new Error("injected receipt write failure"))
          return yield* ssn.updatePart(part)
        })
      const sessionLayer = Layer.mock(SessionNs.Service, {
        messages: (input) => ssn.messages(input),
        updatePart,
      })

      yield* Effect.gen(function* () {
        const compact = yield* SessionCompaction.Service
        const failed = yield* Effect.exit(compact.collapseReceipts({ sessionID: session.id }))
        expect(Exit.isFailure(failed)).toBe(true)
        expect(writes).toBe(2)

        const firstAfterFailure = yield* ssn.findPart({ sessionID: session.id, partID: first.id })
        const secondAfterFailure = yield* ssn.findPart({ sessionID: session.id, partID: second.id })
        const firstMarker =
          firstAfterFailure?.type === "tool" && firstAfterFailure.state.status === "completed"
            ? firstAfterFailure.state.time.compacted
            : undefined
        expect(firstMarker).toBeNumber()
        expect(
          secondAfterFailure?.type === "tool" &&
            secondAfterFailure.state.status === "completed" &&
            secondAfterFailure.state.time.compacted,
        ).toBeUndefined()

        fail = false
        const retried = yield* compact.collapseReceipts({ sessionID: session.id })
        const retriedProjection = JSON.stringify(yield* MessageV2.toModelMessagesEffect(retried, defaultProvider.model))
        expect(writes).toBe(3)
        expect(retriedProjection).not.toContain("FIRST_RECEIPT")
        expect(retriedProjection).not.toContain("SECOND_RECEIPT")

        const firstAfterRetry = yield* ssn.findPart({ sessionID: session.id, partID: first.id })
        const secondAfterRetry = yield* ssn.findPart({ sessionID: session.id, partID: second.id })
        expect(
          firstAfterRetry?.type === "tool" &&
            firstAfterRetry.state.status === "completed" &&
            firstAfterRetry.state.time.compacted,
        ).toBe(firstMarker)
        expect(
          secondAfterRetry?.type === "tool" &&
            secondAfterRetry.state.status === "completed" &&
            secondAfterRetry.state.time.compacted,
        ).toBeNumber()

        const repeated = yield* compact.collapseReceipts({ sessionID: session.id })
        expect(JSON.stringify(yield* MessageV2.toModelMessagesEffect(repeated, defaultProvider.model))).toBe(
          retriedProjection,
        )
        expect(writes).toBe(3)
      }).pipe(withCompaction({ session: sessionLayer }))
    }),
  )
})

describe("session.compaction.prune", () => {
  it.live(
    "continues native pruning past a collapsed context-management receipt",
    provideTmpdirInstance(
      (dir) =>
        Effect.gen(function* () {
          const compact = yield* SessionCompaction.Service
          const ssn = yield* SessionNs.Service
          const session = yield* ssn.create({})
          const user = yield* createUserMessage(session.id, "first")
          const assistant = yield* createAssistantMessage(session.id, user.id, dir)
          const old = yield* ssn.updatePart({
            id: PartID.ascending(),
            messageID: assistant.id,
            sessionID: session.id,
            type: "tool",
            callID: "old-output-before-receipt",
            tool: "bash",
            state: {
              status: "completed",
              input: {},
              output: "x".repeat(300_000),
              title: "old output",
              metadata: {},
              time: { start: 1, end: 2 },
            },
          })
          const marker = 123
          const receipt = yield* ssn.updatePart({
            id: PartID.ascending(),
            messageID: assistant.id,
            sessionID: session.id,
            type: "tool",
            callID: "collapsed-receipt",
            tool: "list_context",
            state: {
              status: "completed",
              input: {},
              output: "full inventory remains durable",
              title: "inventory",
              metadata: { total: 2, compactable: 1, charsCompactable: 100_000 },
              time: { start: 3, end: 4, compacted: marker },
            },
          })
          yield* createUserMessage(session.id, "second")
          yield* createUserMessage(session.id, "third")

          yield* compact.prune({ sessionID: session.id })

          const storedOld = yield* ssn.findPart({ sessionID: session.id, partID: old.id })
          const storedReceipt = yield* ssn.findPart({ sessionID: session.id, partID: receipt.id })
          expect(
            storedOld?.type === "tool" &&
              storedOld.state.status === "completed" &&
              storedOld.state.time.compacted,
          ).toBeNumber()
          expect(
            storedReceipt?.type === "tool" &&
              storedReceipt.state.status === "completed" &&
              storedReceipt.state.time.compacted,
          ).toBe(marker)
          expect(
            storedReceipt?.type === "tool" && storedReceipt.state.status === "completed" && storedReceipt.state.output,
          ).toBe("full inventory remains durable")
        }),
      { config: { compaction: { prune: true } } },
    ),
  )

  it.live(
    "reloads after overlapping manual compaction before pruning",
    provideTmpdirInstance(
      (dir) =>
        Effect.gen(function* () {
          const compact = yield* SessionCompaction.Service
          const ssn = yield* SessionNs.Service
          const info = yield* ssn.create({})
          const user = yield* createUserMessage(info.id, "first")
          const assistant = yield* createAssistantMessage(info.id, user.id, dir)
          const part = yield* ssn.updatePart({
            id: PartID.ascending(),
            messageID: assistant.id,
            sessionID: info.id,
            type: "tool",
            callID: "overlapping-manual-compaction",
            tool: "bash",
            state: {
              status: "completed",
              input: {},
              output: "x".repeat(200_000),
              title: "old output",
              metadata: {},
              time: { start: 1, end: 2 },
            },
          })
          yield* createUserMessage(info.id, "second")
          yield* createUserMessage(info.id, "third")
          const entered = yield* Deferred.make<void>()
          const apply = yield* Deferred.make<void>()
          const persisted = yield* Deferred.make<void>()
          const release = yield* Deferred.make<void>()
          const manual = yield* withCompactionLock(
            info.id,
            undefined,
            Effect.gen(function* () {
              yield* Deferred.succeed(entered, undefined)
              yield* Deferred.await(apply)
              const latest = yield* ssn.findPart({ sessionID: info.id, partID: part.id })
              if (latest?.type !== "tool" || latest.state.status !== "completed") throw new Error("tool part missing")
              latest.state.compactionGroup = "manual-group"
              latest.state.compactionSummary = "manual summary"
              latest.state.compactionGeneration = 1
              yield* ssn.updatePart(latest)
              yield* Deferred.succeed(persisted, undefined)
              yield* Deferred.await(release)
            }),
          ).pipe(Effect.forkScoped)
          yield* Deferred.await(entered)
          const pruning = yield* compact.prune({ sessionID: info.id }).pipe(Effect.forkScoped)
          yield* pollWithTimeout(
            Effect.sync(() => (compactionLockReferences(info.id) === 2 ? true : undefined)),
            "native pruning did not enter the shared compaction lock",
          )
          expect(pruning.pollUnsafe()).toBeUndefined()

          yield* Deferred.succeed(apply, undefined)
          yield* Deferred.await(persisted)
          yield* Deferred.succeed(release, undefined)
          yield* Fiber.join(manual)
          yield* Fiber.join(pruning)

          const stored = yield* ssn.findPart({ sessionID: info.id, partID: part.id })
          expect(stored).toMatchObject({
            type: "tool",
            state: {
              status: "completed",
              compactionGroup: "manual-group",
              compactionSummary: "manual summary",
              compactionGeneration: 1,
            },
          })
          expect(
            stored?.type === "tool" && stored.state.status === "completed" && stored.state.time.compacted,
          ).toBeUndefined()
        }),
      { config: { compaction: { prune: true } } },
    ),
  )

  it.live(
    "skips manual output without consuming the native protection budget",
    provideTmpdirInstance(
      (dir) =>
        Effect.gen(function* () {
          const compact = yield* SessionCompaction.Service
          const ssn = yield* SessionNs.Service
          const info = yield* ssn.create({})
          const a = yield* ssn.updateMessage({
            id: MessageID.ascending(),
            role: "user",
            sessionID: info.id,
            agent: "build",
            model: ref,
            time: { created: Date.now() },
          })
          yield* ssn.updatePart({
            id: PartID.ascending(),
            messageID: a.id,
            sessionID: info.id,
            type: "text",
            text: "first",
          })
          const b: SessionV1.Assistant = {
            id: MessageID.ascending(),
            role: "assistant",
            sessionID: info.id,
            mode: "build",
            agent: "build",
            path: { cwd: dir, root: dir },
            cost: 0,
            tokens: {
              output: 0,
              input: 0,
              reasoning: 0,
              cache: { read: 0, write: 0 },
            },
            modelID: ref.modelID,
            providerID: ref.providerID,
            parentID: a.id,
            time: { created: Date.now() },
            finish: "end_turn",
          }
          yield* ssn.updateMessage(b)
          const old = yield* ssn.updatePart({
            id: PartID.ascending(),
            messageID: b.id,
            sessionID: info.id,
            type: "tool",
            callID: crypto.randomUUID(),
            tool: "bash",
            state: {
              status: "completed",
              input: {},
              output: "x".repeat(200_000),
              title: "done",
              metadata: {},
              time: { start: Date.now(), end: Date.now() },
            },
          })
          const protectedPart = yield* ssn.updatePart({
            id: PartID.ascending(),
            messageID: b.id,
            sessionID: info.id,
            type: "tool",
            callID: crypto.randomUUID(),
            tool: "bash",
            state: {
              status: "completed",
              input: {},
              output: "p".repeat(150_000),
              title: "native protection budget",
              metadata: {},
              time: { start: Date.now(), end: Date.now() },
            },
          })
          const legacy = yield* ssn.updatePart({
            id: PartID.ascending(),
            messageID: b.id,
            sessionID: info.id,
            type: "tool",
            callID: crypto.randomUUID(),
            tool: "read",
            state: {
              status: "completed",
              input: {},
              output: "l".repeat(200_000),
              title: "legacy manual compaction",
              metadata: {},
              time: { start: Date.now(), end: Date.now(), compacted: Date.now() },
              compactionSummary: "legacy settled output",
            },
          })
          const manual = yield* ssn.updatePart({
            id: PartID.ascending(),
            messageID: b.id,
            sessionID: info.id,
            type: "tool",
            callID: crypto.randomUUID(),
            tool: "read",
            state: {
              status: "completed",
              input: {},
              output: "y".repeat(200_000),
              title: "manually compacted",
              metadata: {},
              time: { start: Date.now(), end: Date.now() },
              compactionGroup: crypto.randomUUID(),
              compactionSummary: "settled read output",
            },
          })
          for (const text of ["second", "third"]) {
            const msg = yield* ssn.updateMessage({
              id: MessageID.ascending(),
              role: "user",
              sessionID: info.id,
              agent: "build",
              model: ref,
              time: { created: Date.now() },
            })
            yield* ssn.updatePart({
              id: PartID.ascending(),
              messageID: msg.id,
              sessionID: info.id,
              type: "text",
              text,
            })
          }

          yield* compact.prune({ sessionID: info.id })

          const msgs = yield* ssn.messages({ sessionID: info.id })
          const parts = msgs.flatMap((msg) => msg.parts)
          const oldAfter = parts.find((part) => part.id === old.id)
          const protectedAfter = parts.find((part) => part.id === protectedPart.id)
          const legacyAfter = parts.find((part) => part.id === legacy.id)
          const manualAfter = parts.find((part) => part.id === manual.id)
          expect(oldAfter).toMatchObject({ type: "tool", state: { status: "completed" } })
          if (oldAfter?.type !== "tool" || oldAfter.state.status !== "completed") return
          expect(oldAfter.state.time.compacted).toBeNumber()
          expect(
            protectedAfter?.type === "tool" &&
              protectedAfter.state.status === "completed" &&
              protectedAfter.state.time.compacted,
          ).toBeUndefined()
          expect(
            legacyAfter?.type === "tool" &&
              legacyAfter.state.status === "completed" &&
              legacyAfter.state.time.compacted,
          ).toBeNumber()
          expect(
            legacyAfter?.type === "tool" &&
              legacyAfter.state.status === "completed" &&
              legacyAfter.state.compactionSummary,
          ).toBe("legacy settled output")
          expect(
            manualAfter?.type === "tool" &&
              manualAfter.state.status === "completed" &&
              manualAfter.state.time.compacted,
          ).toBeUndefined()
          expect(
            manualAfter?.type === "tool" &&
              manualAfter.state.status === "completed" &&
              manualAfter.state.compactionGroup,
          ).toBe(manual.type === "tool" && manual.state.status === "completed" && manual.state.compactionGroup)
        }),

      {
        config: {
          compaction: { prune: true },
        },
      },
    ),
  )

  it.live(
    "does not count retained tool input as native pruning savings",
    provideTmpdirInstance(
      (dir) =>
        Effect.gen(function* () {
          const ssn = yield* SessionNs.Service
          const compact = yield* SessionCompaction.Service
          const info = yield* ssn.create({})
          const user = yield* createUserMessage(info.id, "first")
          const assistant = yield* createAssistantMessage(info.id, user.id, dir)
          const old = yield* ssn.updatePart({
            id: PartID.ascending(),
            messageID: assistant.id,
            sessionID: info.id,
            type: "tool",
            callID: "old-output",
            tool: "bash",
            state: {
              status: "completed",
              input: {},
              output: "o".repeat(120_000),
              title: "old",
              metadata: {},
              time: { start: 1, end: 2 },
            },
          })
          const large = yield* ssn.updatePart({
            id: PartID.ascending(),
            messageID: assistant.id,
            sessionID: info.id,
            type: "tool",
            callID: "large-input",
            tool: "edit",
            state: {
              status: "completed",
              input: { patch: "i".repeat(180_000) },
              output: "done",
              title: "large input",
              metadata: {},
              time: { start: 3, end: 4 },
            },
          })
          yield* createUserMessage(info.id, "second")
          yield* createUserMessage(info.id, "third")

          yield* compact.prune({ sessionID: info.id })

          const parts = (yield* ssn.messages({ sessionID: info.id })).flatMap((message) => message.parts)
          const stored = parts.find((part) => part.id === old.id)
          const largeStored = parts.find((part) => part.id === large.id)
          expect(
            stored?.type === "tool" && stored.state.status === "completed" && stored.state.time.compacted,
          ).toBeUndefined()
          expect(
            largeStored?.type === "tool" &&
              largeStored.state.status === "completed" &&
              largeStored.state.time.compacted,
          ).toBeUndefined()
        }),
      { config: { compaction: { prune: true } } },
    ),
  )

  it.live(
    "does not prune external tool attachments that are absent from the provider projection",
    provideTmpdirInstance(
      (dir) =>
        Effect.gen(function* () {
          const ssn = yield* SessionNs.Service
          const compact = yield* SessionCompaction.Service
          const info = yield* ssn.create({})
          const user = yield* createUserMessage(info.id, "first")
          const assistant = yield* createAssistantMessage(info.id, user.id, dir)
          const tool = yield* ssn.updatePart({
            id: PartID.ascending(),
            messageID: assistant.id,
            sessionID: info.id,
            type: "tool",
            callID: "external-attachment",
            tool: "read",
            state: {
              status: "completed",
              input: { filePath: "huge.txt" },
              output: "x",
              title: "huge.txt",
              metadata: {},
              time: { start: 1, end: 2 },
              attachments: [
                {
                  id: PartID.ascending(),
                  messageID: assistant.id,
                  sessionID: info.id,
                  type: "file",
                  mime: "application/octet-stream",
                  filename: "huge.txt",
                  url: `file:///${"a".repeat(200_000)}`,
                },
              ],
            },
          })
          yield* createUserMessage(info.id, "second")
          yield* createUserMessage(info.id, "third")
          const beforeMessages = yield* ssn.messages({ sessionID: info.id })
          const before = JSON.stringify(yield* MessageV2.toModelMessagesEffect(beforeMessages, defaultProvider.model))

          yield* compact.prune({ sessionID: info.id })

          const afterMessages = yield* ssn.messages({ sessionID: info.id })
          const after = JSON.stringify(yield* MessageV2.toModelMessagesEffect(afterMessages, defaultProvider.model))
          const stored = afterMessages.flatMap((message) => message.parts).find((part) => part.id === tool.id)
          expect(after).toBe(before)
          expect(
            stored?.type === "tool" && stored.state.status === "completed" && stored.state.time.compacted,
          ).toBeUndefined()
        }),
      { config: { compaction: { prune: true } } },
    ),
  )

  it.live(
    "skips protected skill tool output",
    provideTmpdirInstance((dir) =>
      Effect.gen(function* () {
        const compact = yield* SessionCompaction.Service
        const ssn = yield* SessionNs.Service
        const info = yield* ssn.create({})
        const a = yield* ssn.updateMessage({
          id: MessageID.ascending(),
          role: "user",
          sessionID: info.id,
          agent: "build",
          model: ref,
          time: { created: Date.now() },
        })
        yield* ssn.updatePart({
          id: PartID.ascending(),
          messageID: a.id,
          sessionID: info.id,
          type: "text",
          text: "first",
        })
        const b: SessionV1.Assistant = {
          id: MessageID.ascending(),
          role: "assistant",
          sessionID: info.id,
          mode: "build",
          agent: "build",
          path: { cwd: dir, root: dir },
          cost: 0,
          tokens: {
            output: 0,
            input: 0,
            reasoning: 0,
            cache: { read: 0, write: 0 },
          },
          modelID: ref.modelID,
          providerID: ref.providerID,
          parentID: a.id,
          time: { created: Date.now() },
          finish: "end_turn",
        }
        yield* ssn.updateMessage(b)
        yield* ssn.updatePart({
          id: PartID.ascending(),
          messageID: b.id,
          sessionID: info.id,
          type: "tool",
          callID: crypto.randomUUID(),
          tool: "skill",
          state: {
            status: "completed",
            input: {},
            output: "x".repeat(200_000),
            title: "done",
            metadata: {},
            time: { start: Date.now(), end: Date.now() },
          },
        })
        for (const text of ["second", "third"]) {
          const msg = yield* ssn.updateMessage({
            id: MessageID.ascending(),
            role: "user",
            sessionID: info.id,
            agent: "build",
            model: ref,
            time: { created: Date.now() },
          })
          yield* ssn.updatePart({
            id: PartID.ascending(),
            messageID: msg.id,
            sessionID: info.id,
            type: "text",
            text,
          })
        }

        yield* compact.prune({ sessionID: info.id })

        const msgs = yield* ssn.messages({ sessionID: info.id })
        const part = msgs.flatMap((msg) => msg.parts).find((part) => part.type === "tool")
        expect(part?.type).toBe("tool")
        if (part?.type === "tool" && part.state.status === "completed") {
          expect(part.state.time.compacted).toBeUndefined()
        }
      }),
    ),
  )
})

describe("session.compaction.process", () => {
  it.instance(
    "throws when parent is not a user message",
    Effect.gen(function* () {
      const test = yield* TestInstance
      const ssn = yield* SessionNs.Service
      const session = yield* ssn.create({})
      const msg = yield* createUserMessage(session.id, "hello")
      const reply = yield* createAssistantMessage(session.id, msg.id, test.directory)
      const msgs = yield* ssn.messages({ sessionID: session.id })

      const exit = yield* Effect.exit(
        SessionCompaction.use.process({
          parentID: reply.id,
          messages: msgs,
          sessionID: session.id,
          auto: false,
        }),
      )

      expect(Exit.isFailure(exit)).toBe(true)
      if (Exit.isFailure(exit)) {
        const error = Cause.squash(exit.cause)
        expect(error).toBeInstanceOf(Error)
        if (error instanceof Error) {
          expect(error.message).toContain(`Compaction parent must be a user message: ${reply.id}`)
        }
      }
    }),
  )

  it.instance(
    "publishes compacted event on continue",
    Effect.gen(function* () {
      const events = yield* EventV2Bridge.Service
      const ssn = yield* SessionNs.Service
      const session = yield* ssn.create({})
      yield* createUserMessage(session.id, "h".repeat(10_000))
      yield* createCompactionMarker(session.id)
      const msgs = yield* ssn.messages({ sessionID: session.id })
      const parent = msgs.at(-1)!.info.id
      const done = yield* Deferred.make<void, Error>()
      const seen: string[] = []
      const unsub = yield* events.listen((evt) => {
        seen.push(evt.type)
        if (evt.type !== SessionCompaction.Event.Compacted.type) return Effect.void
        if ((evt.data as typeof SessionCompaction.Event.Compacted.data.Type).sessionID !== session.id)
          return Effect.void
        Deferred.doneUnsafe(done, Effect.void)
        return Effect.void
      })
      yield* Effect.addFinalizer(() => unsub)

      const result = yield* SessionCompaction.use.process({
        parentID: parent,
        messages: msgs,
        sessionID: session.id,
        auto: false,
      })

      yield* Deferred.await(done).pipe(Effect.timeout("500 millis"))
      expect(result).toBe("continue")
      expect(seen).toContain(SessionCompaction.Event.Compacted.type)
      expect(seen.filter((type) => type.startsWith("session.next."))).toEqual([])
    }),
  )

  itCompaction.instance(
    "marks summary message as errored on compact result",
    Effect.gen(function* () {
      const ssn = yield* SessionNs.Service
      const session = yield* ssn.create({})
      const msg = yield* createUserMessage(session.id, "hello")
      const msgs = yield* ssn.messages({ sessionID: session.id })

      const result = yield* SessionCompaction.use.process({
        parentID: msg.id,
        messages: msgs,
        sessionID: session.id,
        auto: false,
      })

      const summary = (yield* ssn.messages({ sessionID: session.id })).find(
        (msg) => msg.info.role === "assistant" && msg.info.summary,
      )

      expect(result).toBe("stop")
      expect(summary?.info.role).toBe("assistant")
      if (summary?.info.role === "assistant") {
        expect(summary.info.finish).toBe("error")
        expect(JSON.stringify(summary.info.error)).toContain("Session too large to compact")
      }
    }).pipe(withCompaction({ result: "compact" })),
  )

  itCompaction.instance(
    "rejects an empty summary without activating the compaction boundary",
    () => {
      const stub = llm()
      stub.push(
        Stream.make(
          LLMEvent.stepFinish({ index: 0, reason: "stop", usage: basicUsage() }),
          LLMEvent.finish({ reason: "stop", usage: basicUsage() }),
        ),
      )
      return Effect.gen(function* () {
        const ssn = yield* SessionNs.Service
        const session = yield* ssn.create({})
        const original = yield* createUserMessage(session.id, "keep original history")
        yield* createCompactionMarker(session.id)
        const msgs = yield* ssn.messages({ sessionID: session.id })
        const parent = msgs.at(-1)!.info.id

        const result = yield* SessionCompaction.use.process({
          parentID: parent,
          messages: msgs,
          sessionID: session.id,
          auto: false,
        })

        const all = yield* ssn.messages({ sessionID: session.id })
        const summary = all.find((message) => message.info.role === "assistant" && message.info.summary)
        const filtered = MessageV2.filterCompacted(all)
        expect(result).toBe("stop")
        expect(summary?.info.role === "assistant" ? summary.info.error?.name : undefined).toBe("ContextOverflowError")
        expect(filtered.map((message) => message.info.id)).toContain(original.id)
      }).pipe(withCompaction({ llm: stub.llmLayer }))
    },
    { git: true },
  )

  itCompaction.instance(
    "fails before generation when the summary request exceeds the compaction model",
    () => {
      const stub = llm()
      let called = 0
      stub.push((input) => {
        called++
        return reply("summary")(input)
      })
      const provider = ProviderTest.fake({ model: createModel({ context: 4_000, output: 1_000 }) })
      return Effect.gen(function* () {
        const ssn = yield* SessionNs.Service
        const session = yield* ssn.create({})
        const user = yield* createUserMessage(session.id, "x".repeat(40_000))
        const msgs = yield* ssn.messages({ sessionID: session.id })

        const result = yield* SessionCompaction.use.process({
          parentID: user.id,
          messages: msgs,
          sessionID: session.id,
          auto: false,
        })

        const summary = (yield* ssn.messages({ sessionID: session.id })).find(
          (message) => message.info.role === "assistant" && message.info.summary,
        )
        expect(result).toBe("stop")
        expect(called).toBe(0)
        expect(summary?.info.role === "assistant" ? JSON.stringify(summary.info.error) : "").toContain(
          "too large to summarize safely",
        )
      }).pipe(withCompaction({ llm: stub.llmLayer, provider }))
    },
    { git: true },
  )

  itCompaction.instance(
    "truncates oversized tool input only in the native summary request",
    () => {
      const stub = llm()
      let captured = ""
      stub.push(reply("summary", (input) => (captured = JSON.stringify(input.messages))))
      return Effect.gen(function* () {
        const ssn = yield* SessionNs.Service
        const session = yield* ssn.create({})
        const root = yield* createUserMessage(session.id, "root")
        const assistant = yield* createAssistantMessage(session.id, root.id, (yield* TestInstance).directory)
        yield* ssn.updatePart({
          id: PartID.ascending(),
          messageID: assistant.id,
          sessionID: session.id,
          type: "tool",
          callID: "large-input",
          tool: "read",
          state: {
            status: "completed",
            input: { payload: `INPUT_START${"x".repeat(20_000)}INPUT_END` },
            output: "done",
            title: "done",
            metadata: {},
            time: { start: 1, end: 2 },
          },
        })
        yield* createCompactionMarker(session.id)
        const msgs = yield* ssn.messages({ sessionID: session.id })

        const result = yield* SessionCompaction.use.process({
          parentID: msgs.at(-1)!.info.id,
          messages: msgs,
          sessionID: session.id,
          auto: false,
        })

        expect(result).toBe("continue")
        expect(captured).toContain("INPUT_START")
        expect(captured).toContain("[truncated]")
        expect(captured).not.toContain("INPUT_END")
        const stored = yield* MessageV2.parts(assistant.id)
        expect(JSON.stringify(stored)).toContain("INPUT_END")
      }).pipe(withCompaction({ llm: stub.llmLayer, config: cfg({ tail_turns: 0 }) }))
    },
    { git: true },
  )

  itCompaction.instance(
    "caps an oversized native summary before validating context reduction",
    () => {
      const stub = llm()
      stub.push(reply("summary ".repeat(20_000)))
      return Effect.gen(function* () {
        const ssn = yield* SessionNs.Service
        const session = yield* ssn.create({})
        yield* createUserMessage(session.id, "history ".repeat(5_000))
        yield* createCompactionMarker(session.id)
        const msgs = yield* ssn.messages({ sessionID: session.id })

        const result = yield* SessionCompaction.use.process({
          parentID: msgs.at(-1)!.info.id,
          messages: msgs,
          sessionID: session.id,
          auto: false,
        })

        const generated = (yield* ssn.messages({ sessionID: session.id })).find(
          (message) => message.info.role === "assistant" && message.info.summary,
        )
        const text = generated?.parts
          .filter((part): part is SessionV1.TextPart => part.type === "text")
          .map((part) => part.text)
          .join("")
        expect(result).toBe("continue")
        expect(text?.length).toBeLessThanOrEqual(4_000)
        expect(text).toEndWith("[truncated]")
      }).pipe(withCompaction({ llm: stub.llmLayer, config: cfg({ tail_turns: 0 }) }))
    },
    { git: true },
  )

  itCompaction.instance(
    "keeps a concurrent prompt last without appending an automatic continuation",
    () =>
      Effect.gen(function* () {
        const started = yield* Deferred.make<void>()
        const release = yield* Deferred.make<void>()
        const stub = llm()
        stub.push((input) =>
          Stream.concat(
            Stream.fromEffect(Effect.sync(() => Deferred.doneUnsafe(started, Effect.void))).pipe(Stream.drain),
            Stream.concat(Stream.fromEffect(Deferred.await(release)).pipe(Stream.drain), reply("summary")(input)),
          ),
        )
        return yield* Effect.gen(function* () {
          const ssn = yield* SessionNs.Service
          const session = yield* ssn.create({})
          yield* createUserMessage(session.id, "old history ".repeat(4_000))
          yield* createCompactionMarker(session.id)
          const msgs = yield* ssn.messages({ sessionID: session.id })
          const fiber = yield* SessionCompaction.use
            .process({
              parentID: msgs.at(-1)!.info.id,
              messages: msgs,
              sessionID: session.id,
              auto: true,
            })
            .pipe(Effect.forkChild)

          yield* Deferred.await(started)
          const appended = yield* createUserMessage(session.id, "new prompt ".repeat(10_000))
          Deferred.doneUnsafe(release, Effect.void)
          const result = yield* Fiber.join(fiber)
          const all = yield* ssn.messages({ sessionID: session.id })
          const generated = all.find((message) => message.info.role === "assistant" && message.info.summary)

          expect(result).toBe("continue")
          expect(all.at(-1)?.info.id).toBe(appended.id)
          expect(
            all.some((message) =>
              message.parts.some(
                (part) => part.type === "text" && part.synthetic && part.metadata?.compaction_continue === true,
              ),
            ),
          ).toBe(false)
          expect(generated?.info.role === "assistant" ? generated.info.error : undefined).toBeUndefined()
        }).pipe(withCompaction({ llm: stub.llmLayer, config: cfg({ tail_turns: 0 }) }))
      }),
    { git: true },
  )

  itCompaction.instance(
    "reloads stale caller messages after acquiring the compaction lock",
    () => {
      const stub = llm()
      let captured = ""
      stub.push(reply("native summary", (input) => (captured = JSON.stringify(input.messages))))
      return Effect.gen(function* () {
        const ssn = yield* SessionNs.Service
        const session = yield* ssn.create({})
        const root = yield* createUserMessage(session.id, "old history")
        const assistant = yield* createAssistantMessage(session.id, root.id, (yield* TestInstance).directory)
        const part = yield* ssn.updatePart({
          id: PartID.ascending(),
          messageID: assistant.id,
          sessionID: session.id,
          type: "tool",
          callID: "stale-lock-input",
          tool: "read",
          state: {
            status: "completed",
            input: {},
            output: `STALE_TOOL_OUTPUT${"x".repeat(20_000)}`,
            title: "done",
            metadata: {},
            time: { start: 1, end: 2 },
          },
        })
        yield* createCompactionMarker(session.id)
        const stale = yield* ssn.messages({ sessionID: session.id })
        const entered = yield* Deferred.make<void>()
        const mutate = yield* Deferred.make<void>()
        const persisted = yield* Deferred.make<void>()
        const release = yield* Deferred.make<void>()
        const holder = yield* withCompactionLock(
          session.id,
          undefined,
          Effect.gen(function* () {
            yield* Deferred.succeed(entered, undefined)
            yield* Deferred.await(mutate)
            const latest = yield* ssn.findPart({ sessionID: session.id, partID: part.id })
            if (latest?.type !== "tool" || latest.state.status !== "completed") throw new Error("tool part missing")
            latest.state.compactionGroup = "lock-rebase-group"
            latest.state.compactionSummary = `LOCK_REBASED_SUMMARY${"y".repeat(8_000)}`
            latest.state.compactionGeneration = 1
            yield* ssn.updatePart(latest)
            yield* Deferred.succeed(persisted, undefined)
            yield* Deferred.await(release)
          }),
        ).pipe(Effect.forkScoped)
        yield* Deferred.await(entered)
        const compaction = yield* SessionCompaction.use
          .process({
            parentID: stale.at(-1)!.info.id,
            messages: stale,
            sessionID: session.id,
            auto: false,
          })
          .pipe(Effect.forkScoped)
        yield* pollWithTimeout(
          Effect.sync(() => (compactionLockReferences(session.id) === 2 ? true : undefined)),
          "native compaction did not wait for the shared lock",
        )

        yield* Deferred.succeed(mutate, undefined)
        yield* Deferred.await(persisted)
        yield* Deferred.succeed(release, undefined)
        yield* Fiber.join(holder)
        expect(yield* Fiber.join(compaction)).toBe("continue")
        expect(captured).toContain("LOCK_REBASED_SUMMARY")
        expect(captured).not.toContain("STALE_TOOL_OUTPUT")
      }).pipe(withCompaction({ llm: stub.llmLayer, config: cfg({ tail_turns: 0 }) }))
    },
    { git: true },
  )

  itCompaction.instance(
    "fails explicitly when preserve_recent_tokens consumes the usable budget",
    () => {
      const stub = llm()
      let called = 0
      stub.push((input) => {
        called++
        return reply("summary")(input)
      })
      return Effect.gen(function* () {
        const ssn = yield* SessionNs.Service
        const session = yield* ssn.create({})
        const user = yield* createUserMessage(session.id, "hello")
        const msgs = yield* ssn.messages({ sessionID: session.id })

        const result = yield* SessionCompaction.use.process({
          parentID: user.id,
          messages: msgs,
          sessionID: session.id,
          auto: true,
        })

        const summary = (yield* ssn.messages({ sessionID: session.id })).find(
          (message) => message.info.role === "assistant" && message.info.summary,
        )
        expect(result).toBe("stop")
        expect(called).toBe(0)
        expect(summary?.info.role === "assistant" ? JSON.stringify(summary.info.error) : "").toContain("preserve")
      }).pipe(withCompaction({ llm: stub.llmLayer, config: cfg({ preserve_recent_tokens: 68_000 }) }))
    },
    { git: true },
  )

  itCompaction.instance(
    "includes system and tool schema overhead in post-compaction safety",
    () => {
      const stub = llm()
      let called = 0
      stub.push((input) => {
        called++
        return reply("summary")(input)
      })
      const requestOverhead = estimateRequest({
        system: ["system instructions".repeat(1_000)],
        messages: [],
        tools: {
          large_mcp_tool: {
            description: "large MCP tool",
            inputSchema: { type: "object", description: "s".repeat(300_000) },
          },
        },
        reminderHeadroom: REQUEST_REMINDER_HEADROOM,
      })
      return Effect.gen(function* () {
        const ssn = yield* SessionNs.Service
        const session = yield* ssn.create({})
        const user = yield* createUserMessage(session.id, "hello")
        const msgs = yield* ssn.messages({ sessionID: session.id })

        const result = yield* SessionCompaction.use.process({
          parentID: user.id,
          messages: msgs,
          sessionID: session.id,
          auto: true,
          requestOverhead,
        })

        const summary = (yield* ssn.messages({ sessionID: session.id })).find(
          (message) => message.info.role === "assistant" && message.info.summary,
        )
        expect(result).toBe("stop")
        expect(called).toBe(0)
        expect(requestOverhead).toBeGreaterThan(68_000)
        expect(summary?.info.role === "assistant" ? JSON.stringify(summary.info.error) : "").toContain(
          "system instructions, tools, and reminder headroom",
        )
      }).pipe(withCompaction({ llm: stub.llmLayer }))
    },
    { git: true },
  )

  itCompaction.instance(
    "caps an oversized automatic summary before validating projected context",
    () => {
      const stub = llm()
      stub.push(reply("s".repeat(100_000)))
      return Effect.gen(function* () {
        const ssn = yield* SessionNs.Service
        const test = yield* TestInstance
        const session = yield* ssn.create({})
        yield* createUserMessage(session.id, "h".repeat(100_000))
        const recent = yield* createUserMessage(session.id, "keep recent")
        const assistant = yield* createAssistantMessage(session.id, recent.id, test.directory)
        yield* ssn.updatePart({
          id: PartID.ascending(),
          messageID: assistant.id,
          sessionID: session.id,
          type: "tool",
          callID: "large-tail",
          tool: "read",
          state: {
            status: "completed",
            input: { filePath: "large.txt" },
            output: "t".repeat(220_000),
            title: "large.txt",
            metadata: {},
            time: { start: 1, end: 2 },
          },
        })
        yield* createCompactionMarker(session.id)
        const msgs = yield* ssn.messages({ sessionID: session.id })
        const marker = msgs.at(-1)!.info.id

        const result = yield* SessionCompaction.use.process({
          parentID: marker,
          messages: msgs,
          sessionID: session.id,
          auto: true,
        })

        const all = yield* ssn.messages({ sessionID: session.id })
        const generated = all.find((message) => message.info.role === "assistant" && message.info.summary)
        const text = generated?.parts
          .filter((part): part is SessionV1.TextPart => part.type === "text")
          .map((part) => part.text)
          .join("")
        expect(result).toBe("continue")
        expect(generated?.info.role === "assistant" ? generated.info.error : undefined).toBeUndefined()
        expect(text?.length).toBeLessThanOrEqual(4_000)
        expect(text).toEndWith("[truncated]")
        expect(all.filter((message) => message.info.role === "user")).toHaveLength(4)
      }).pipe(
        withCompaction({
          llm: stub.llmLayer,
          config: cfg({ tail_turns: 1, preserve_recent_tokens: 65_000 }),
        }),
      )
    },
    { git: true },
  )

  itCompaction.instance(
    "rejects a growing manual summary without activating its compaction boundary",
    () => {
      const stub = llm()
      stub.push(reply("s".repeat(10_000)))
      return Effect.gen(function* () {
        const ssn = yield* SessionNs.Service
        const session = yield* ssn.create({})
        const original = yield* createUserMessage(session.id, "tiny")
        yield* createCompactionMarker(session.id)
        const msgs = yield* ssn.messages({ sessionID: session.id })

        const result = yield* SessionCompaction.use.process({
          parentID: msgs.at(-1)!.info.id,
          messages: msgs,
          sessionID: session.id,
          auto: false,
        })

        const all = yield* ssn.messages({ sessionID: session.id })
        const generated = all.find((message) => message.info.role === "assistant" && message.info.summary)
        const boundary = all
          .flatMap((message) => message.parts)
          .find((part): part is SessionV1.CompactionPart => part.type === "compaction")
        expect(result).toBe("stop")
        expect(generated?.info.role === "assistant" ? generated.info.error?.name : undefined).toBe(
          "ContextOverflowError",
        )
        expect(generated?.info.role === "assistant" ? JSON.stringify(generated.info.error) : "").toContain(
          "did not reduce",
        )
        expect(boundary?.tail_start_id).toBeUndefined()
        expect(MessageV2.filterCompacted(all.toReversed()).map((message) => message.info.id)).toContain(original.id)
        expect(
          all.some(
            (message) =>
              message.info.role === "assistant" && message.info.summary && message.info.finish && !message.info.error,
          ),
        ).toBe(false)
      }).pipe(withCompaction({ llm: stub.llmLayer }))
    },
    { git: true },
  )

  it.instance(
    "adds synthetic continue prompt when auto is enabled",
    Effect.gen(function* () {
      const ssn = yield* SessionNs.Service
      const session = yield* ssn.create({})
      yield* createUserMessage(session.id, "h".repeat(10_000))
      yield* createCompactionMarker(session.id)
      const msgs = yield* ssn.messages({ sessionID: session.id })

      const result = yield* SessionCompaction.use.process({
        parentID: msgs.at(-1)!.info.id,
        messages: msgs,
        sessionID: session.id,
        auto: true,
      })

      const all = yield* ssn.messages({ sessionID: session.id })
      const last = all.at(-1)

      expect(result).toBe("continue")
      expect(last?.info.role).toBe("user")
      expect(last?.parts[0]).toMatchObject({
        type: "text",
        synthetic: true,
        metadata: { compaction_continue: true },
      })
      if (last?.parts[0]?.type === "text") {
        expect(last.parts[0].text).toContain("Continue if you have next steps")
      }
    }),
  )

  itCompaction.instance(
    "persists tail_start_id for retained recent turns",
    Effect.gen(function* () {
      const ssn = yield* SessionNs.Service
      const session = yield* ssn.create({})
      yield* createUserMessage(session.id, "f".repeat(10_000))
      const keep = yield* createUserMessage(session.id, "second")
      yield* createUserMessage(session.id, "third")
      yield* createSummaryCompaction(session.id)

      const msgs = yield* ssn.messages({ sessionID: session.id })
      const parent = msgs.at(-1)?.info.id
      expect(parent).toBeTruthy()
      yield* SessionCompaction.use.process({
        parentID: parent!,
        messages: msgs,
        sessionID: session.id,
        auto: false,
      })

      const part = yield* readCompactionPart(session.id)
      expect(part?.type).toBe("compaction")
      expect(part?.tail_start_id).toBe(keep.id)
    }).pipe(withCompaction({ config: cfg({ tail_turns: 2, preserve_recent_tokens: 10_000 }) })),
  )

  itCompaction.instance(
    "shrinks retained tail to fit preserve token budget",
    Effect.gen(function* () {
      const ssn = yield* SessionNs.Service
      const session = yield* ssn.create({})
      yield* createUserMessage(session.id, "f".repeat(10_000))
      yield* createUserMessage(session.id, "x".repeat(2_000))
      const keep = yield* createUserMessage(session.id, "tiny")
      yield* createSummaryCompaction(session.id)

      const msgs = yield* ssn.messages({ sessionID: session.id })
      const parent = msgs.at(-1)?.info.id
      expect(parent).toBeTruthy()
      yield* SessionCompaction.use.process({
        parentID: parent!,
        messages: msgs,
        sessionID: session.id,
        auto: false,
      })

      const part = yield* readCompactionPart(session.id)
      expect(part?.type).toBe("compaction")
      expect(part?.tail_start_id).toBe(keep.id)
    }).pipe(withCompaction({ config: cfg({ tail_turns: 2, preserve_recent_tokens: 100 }) })),
  )

  itCompaction.instance(
    "falls back to full summary when even one recent turn exceeds preserve token budget",
    () => {
      const stub = llm()
      let captured = ""
      stub.push(reply("summary", (input) => (captured = JSON.stringify(input.messages))))
      return Effect.gen(function* () {
        const ssn = yield* SessionNs.Service
        const session = yield* ssn.create({})
        yield* createUserMessage(session.id, "first")
        yield* createUserMessage(session.id, "y".repeat(2_000))
        yield* createSummaryCompaction(session.id)

        const msgs = yield* ssn.messages({ sessionID: session.id })
        const parent = msgs.at(-1)?.info.id
        expect(parent).toBeTruthy()
        yield* SessionCompaction.use.process({ parentID: parent!, messages: msgs, sessionID: session.id, auto: false })

        const part = yield* readCompactionPart(session.id)
        expect(part?.type).toBe("compaction")
        expect(part?.tail_start_id).toBeUndefined()
        expect(captured).toContain("yyyy")
      }).pipe(withCompaction({ llm: stub.llmLayer, config: cfg({ tail_turns: 1, preserve_recent_tokens: 20 }) }))
    },
    { git: true },
  )

  itCompaction.instance(
    "falls back to full summary when retained tail media exceeds preserve token budget",
    () => {
      const stub = llm()
      let captured = ""
      stub.push(reply("summary", (input) => (captured = JSON.stringify(input.messages))))
      return Effect.gen(function* () {
        const ssn = yield* SessionNs.Service
        const session = yield* ssn.create({})
        yield* createUserMessage(session.id, "older")
        const recent = yield* createUserMessage(session.id, "recent image turn")
        yield* ssn.updatePart({
          id: PartID.ascending(),
          messageID: recent.id,
          sessionID: session.id,
          type: "file",
          mime: "image/png",
          filename: "big.png",
          url: `data:image/png;base64,${"a".repeat(4_000)}`,
        })
        yield* createSummaryCompaction(session.id)

        const msgs = yield* ssn.messages({ sessionID: session.id })
        const parent = msgs.at(-1)?.info.id
        expect(parent).toBeTruthy()
        yield* SessionCompaction.use.process({ parentID: parent!, messages: msgs, sessionID: session.id, auto: false })

        const part = yield* readCompactionPart(session.id)
        expect(part?.type).toBe("compaction")
        expect(part?.tail_start_id).toBeUndefined()
        expect(captured).toContain("recent image turn")
        expect(captured).toContain("Attached image/png: big.png")
      }).pipe(withCompaction({ llm: stub.llmLayer, config: cfg({ tail_turns: 1, preserve_recent_tokens: 100 }) }))
    },
    { git: true },
  )

  itCompaction.instance(
    "retains a split turn suffix when a later message fits the preserve token budget",
    () => {
      const stub = llm()
      let captured = ""
      stub.push(reply("summary", (input) => (captured = JSON.stringify(input.messages))))
      return Effect.gen(function* () {
        const test = yield* TestInstance
        const ssn = yield* SessionNs.Service
        const session = yield* ssn.create({})
        yield* createUserMessage(session.id, "o".repeat(10_000))
        const recent = yield* createUserMessage(session.id, "recent turn")
        const large = yield* createAssistantMessage(session.id, recent.id, test.directory)
        yield* ssn.updatePart({
          id: PartID.ascending(),
          messageID: large.id,
          sessionID: session.id,
          type: "text",
          text: "z".repeat(2_000),
        })
        const keep = yield* createAssistantMessage(session.id, recent.id, test.directory)
        yield* ssn.updatePart({
          id: PartID.ascending(),
          messageID: keep.id,
          sessionID: session.id,
          type: "text",
          text: "keep tail",
        })
        yield* createSummaryCompaction(session.id)

        const msgs = yield* ssn.messages({ sessionID: session.id })
        const parent = msgs.at(-1)?.info.id
        expect(parent).toBeTruthy()
        yield* SessionCompaction.use.process({ parentID: parent!, messages: msgs, sessionID: session.id, auto: false })

        const part = yield* readCompactionPart(session.id)
        expect(part?.type).toBe("compaction")
        expect(part?.tail_start_id).toBe(keep.id)
        expect(captured).toContain("zzzz")
        expect(captured).not.toContain("keep tail")

        const filtered = MessageV2.filterCompacted(yield* MessageV2.stream(session.id))
        expect(filtered.map((msg) => msg.info.id).slice(0, 3)).toEqual([parent!, expect.any(String), keep.id])
        expect(filtered[1]?.info.role).toBe("assistant")
        expect(filtered[1]?.info.role === "assistant" ? filtered[1].info.summary : false).toBe(true)
        expect(filtered.map((msg) => msg.info.id)).not.toContain(large.id)
      }).pipe(withCompaction({ llm: stub.llmLayer, config: cfg({ tail_turns: 1, preserve_recent_tokens: 100 }) }))
    },
    { git: true },
  )

  itCompaction.instance(
    "allows plugins to disable synthetic continue prompt",
    Effect.gen(function* () {
      const ssn = yield* SessionNs.Service
      const session = yield* ssn.create({})
      yield* createUserMessage(session.id, "h".repeat(10_000))
      yield* createCompactionMarker(session.id)
      const msgs = yield* ssn.messages({ sessionID: session.id })

      const result = yield* SessionCompaction.use.process({
        parentID: msgs.at(-1)!.info.id,
        messages: msgs,
        sessionID: session.id,
        auto: true,
      })

      const all = yield* ssn.messages({ sessionID: session.id })
      const last = all.at(-1)

      expect(result).toBe("continue")
      expect(last?.info.role).toBe("assistant")
      expect(
        all.some(
          (msg) =>
            msg.info.role === "user" &&
            msg.parts.some(
              (part) => part.type === "text" && part.synthetic && part.text.includes("Continue if you have next steps"),
            ),
        ),
      ).toBe(false)
    }).pipe(withCompaction({ plugin: autocontinue(false) })),
  )

  itCompaction.instance(
    "replays the prior user turn on overflow when earlier context exists",
    () => {
      const stub = llm()
      let request = ""
      stub.push(
        reply("summary", (input) => {
          request = JSON.stringify(input.messages)
        }),
      )
      return Effect.gen(function* () {
        const ssn = yield* SessionNs.Service
        const session = yield* ssn.create({})
        const root = yield* createUserMessage(session.id, "root")
        const assistant = yield* createAssistantMessage(session.id, root.id, (yield* TestInstance).directory)
        yield* ssn.updatePart({
          id: PartID.ascending(),
          messageID: assistant.id,
          sessionID: session.id,
          type: "text",
          text: `FINAL_PRE_REPLAY_ASSISTANT_MESSAGE${"x".repeat(10_000)}`,
        })
        const replay = yield* createUserMessage(session.id, "image")
        yield* ssn.updatePart({
          id: PartID.ascending(),
          messageID: replay.id,
          sessionID: session.id,
          type: "file",
          mime: "image/png",
          filename: "cat.png",
          url: "https://example.com/cat.png",
        })
        yield* createCompactionMarker(session.id)
        const msgs = yield* ssn.messages({ sessionID: session.id })

        const result = yield* SessionCompaction.use.process({
          parentID: msgs.at(-1)!.info.id,
          messages: msgs,
          sessionID: session.id,
          auto: true,
          overflow: true,
        })

        const last = (yield* ssn.messages({ sessionID: session.id })).at(-1)

        expect(result).toBe("continue")
        expect(last?.info.role).toBe("user")
        expect(last?.parts.some((part) => part.type === "file")).toBe(false)
        expect(
          last?.parts.some((part) => part.type === "text" && part.text.includes("Attached image/png: cat.png")),
        ).toBe(true)
        expect(request).toContain("FINAL_PRE_REPLAY_ASSISTANT_MESSAGE")
      }).pipe(withCompaction({ llm: stub.llmLayer }))
    },
    { git: true },
  )

  it.instance(
    "falls back to overflow guidance when no replayable turn exists",
    Effect.gen(function* () {
      const ssn = yield* SessionNs.Service
      const session = yield* ssn.create({})
      yield* createUserMessage(session.id, `current${"x".repeat(10_000)}`)
      yield* createCompactionMarker(session.id)
      const msgs = yield* ssn.messages({ sessionID: session.id })

      const result = yield* SessionCompaction.use.process({
        parentID: msgs.at(-1)!.info.id,
        messages: msgs,
        sessionID: session.id,
        auto: true,
        overflow: true,
      })

      const last = (yield* ssn.messages({ sessionID: session.id })).at(-1)

      expect(result).toBe("continue")
      expect(last?.info.role).toBe("user")
      if (last?.parts[0]?.type === "text") {
        expect(last.parts[0].text).toContain("previous request exceeded the provider's size limit")
      }
    }),
  )

  itCompaction.instance(
    "stops quickly when aborted during retry backoff",
    () => {
      const stub = llm()
      stub.push(
        Stream.fromAsyncIterable(
          {
            async *[Symbol.asyncIterator]() {
              yield LLMEvent.stepStart({ index: 0 })
              throw new APICallError({
                message: "boom",
                url: "https://example.com/v1/chat/completions",
                requestBodyValues: {},
                statusCode: 503,
                responseHeaders: { "retry-after-ms": "10000" },
                responseBody: '{"error":"boom"}',
                isRetryable: true,
              })
            },
          },
          (err) => err,
        ),
      )

      return Effect.gen(function* () {
        const ssn = yield* SessionNs.Service
        const events = yield* EventV2Bridge.Service
        const ready = yield* Deferred.make<void>()
        const session = yield* ssn.create({})
        const msg = yield* createUserMessage(session.id, "hello")
        const msgs = yield* ssn.messages({ sessionID: session.id })
        const off = yield* events.listen((evt) => {
          if (evt.type !== SessionStatus.Event.Status.type) return Effect.void
          const data = evt.data as typeof SessionStatus.Event.Status.data.Type
          if (data.sessionID !== session.id || data.status.type !== "retry") return Effect.void
          Deferred.doneUnsafe(ready, Effect.void)
          return Effect.void
        })
        yield* Effect.addFinalizer(() => off)

        const fiber = yield* SessionCompaction.use
          .process({
            parentID: msg.id,
            messages: msgs,
            sessionID: session.id,
            auto: false,
          })
          .pipe(Effect.forkChild)

        yield* Deferred.await(ready).pipe(Effect.timeout("5 seconds"))
        const start = Date.now()
        // Fiber.interrupt already awaits the fiber, so this bound is the only
        // one that matters. The stub is mid-backoff on a 10s retry-after, so
        // 500ms still proves the interrupt short-circuits it by 20x while
        // leaving room for GC and scheduler pressure.
        yield* Fiber.interrupt(fiber)
        const exit = yield* Fiber.await(fiber)

        expect(Exit.isFailure(exit)).toBe(true)
        if (Exit.isFailure(exit)) {
          expect(Cause.hasInterrupts(exit.cause)).toBe(true)
          expect(Date.now() - start).toBeLessThan(500)
        }
      }).pipe(withCompaction({ llm: stub.llmLayer }))
    },
    { git: true },
    { timeout: 10_000 },
  )

  itCompaction.instance(
    "does not leave a summary assistant when aborted before processor setup",
    () =>
      Effect.gen(function* () {
        const ready = yield* Deferred.make<void>()
        return yield* Effect.gen(function* () {
          const ssn = yield* SessionNs.Service
          const session = yield* ssn.create({})
          const msg = yield* createUserMessage(session.id, "hello")
          const msgs = yield* ssn.messages({ sessionID: session.id })
          const fiber = yield* SessionCompaction.use
            .process({
              parentID: msg.id,
              messages: msgs,
              sessionID: session.id,
              auto: false,
            })
            .pipe(Effect.forkChild)

          yield* Deferred.await(ready).pipe(Effect.timeout("1 second"))
          yield* Fiber.interrupt(fiber)
          const exit = yield* Fiber.await(fiber).pipe(Effect.timeout("250 millis"))
          const all = yield* ssn.messages({ sessionID: session.id })

          expect(Exit.isFailure(exit)).toBe(true)
          if (Exit.isFailure(exit)) expect(Cause.hasInterrupts(exit.cause)).toBe(true)
          expect(all.some((msg) => msg.info.role === "assistant" && msg.info.summary)).toBe(false)
        }).pipe(withCompaction({ plugin: plugin(ready) }))
      }),
    { git: true },
  )

  itCompaction.instance(
    "finalizes the summary assistant when interrupted after persistence",
    () => {
      const stub = llm()
      let ready: Deferred.Deferred<void> | undefined
      stub.push(() => {
        Deferred.doneUnsafe(ready!, Effect.void)
        return Stream.never
      })
      return Effect.gen(function* () {
        ready = yield* Deferred.make<void>()
        const ssn = yield* SessionNs.Service
        const session = yield* ssn.create({})
        const msg = yield* createUserMessage(session.id, "hello")
        const msgs = yield* ssn.messages({ sessionID: session.id })
        const fiber = yield* SessionCompaction.use
          .process({
            parentID: msg.id,
            messages: msgs,
            sessionID: session.id,
            auto: false,
          })
          .pipe(Effect.forkChild)

        yield* Deferred.await(ready).pipe(Effect.timeout("1 second"))
        yield* Fiber.interrupt(fiber)

        const summary = (yield* ssn.messages({ sessionID: session.id })).find(
          (message) => message.info.role === "assistant" && message.info.summary,
        )
        expect(summary?.info.role).toBe("assistant")
        if (summary?.info.role !== "assistant") return
        expect(summary.info.finish).toBe("error")
        expect(summary.info.error?.name).toBe("MessageAbortedError")
        expect(summary.info.time.completed).toBeNumber()
      }).pipe(withCompaction({ llm: stub.llmLayer }))
    },
    { git: true },
  )

  itCompaction.instance(
    "silently drops reasoning-delta arriving without prior reasoning-start",
    () => {
      // Regression: PR initially auto-created a reasoning Part for orphan deltas (no preceding
      // reasoning-start). Reverted to match dev — drop silently. Pinned here so any future
      // change to processor.ts reasoning-delta handling triggers this test.
      const stub = llm()
      stub.push(
        Stream.make(
          LLMEvent.reasoningDelta({ id: "orphan-1", text: "stray reasoning" }),
          LLMEvent.textStart({ id: "txt-0" }),
          LLMEvent.textDelta({ id: "txt-0", text: "summary" }),
          LLMEvent.textEnd({ id: "txt-0" }),
          LLMEvent.stepFinish({ index: 0, reason: "stop", usage: basicUsage() }),
          LLMEvent.finish({ reason: "stop", usage: basicUsage() }),
        ),
      )
      return Effect.gen(function* () {
        const ssn = yield* SessionNs.Service
        const session = yield* ssn.create({})
        const msg = yield* createUserMessage(session.id, "hello")
        const msgs = yield* ssn.messages({ sessionID: session.id })
        yield* SessionCompaction.use.process({
          parentID: msg.id,
          messages: msgs,
          sessionID: session.id,
          auto: false,
        })

        const summary = (yield* ssn.messages({ sessionID: session.id })).find(
          (item) => item.info.role === "assistant" && item.info.summary,
        )
        expect(summary?.parts.some((part) => part.type === "reasoning")).toBe(false)
        // Sanity: the text part still got through.
        expect(summary?.parts.some((part) => part.type === "text" && part.text === "summary")).toBe(true)
      }).pipe(withCompaction({ llm: stub.llmLayer }))
    },
    { git: true },
  )

  itCompaction.instance(
    "does not allow tool calls while generating the summary",
    () => {
      const stub = llm()
      stub.push(
        Stream.make(
          LLMEvent.toolCall({ id: "call-1", name: "_noop", input: {} }),
          LLMEvent.stepFinish({
            index: 0,
            reason: "tool-calls",
            usage: basicUsage(),
          }),
          LLMEvent.finish({
            reason: "tool-calls",
            usage: basicUsage(),
          }),
        ),
      )
      return Effect.gen(function* () {
        const ssn = yield* SessionNs.Service
        const session = yield* ssn.create({})
        const msg = yield* createUserMessage(session.id, "hello")
        const msgs = yield* ssn.messages({ sessionID: session.id })
        yield* SessionCompaction.use.process({ parentID: msg.id, messages: msgs, sessionID: session.id, auto: false })

        const summary = (yield* ssn.messages({ sessionID: session.id })).find(
          (item) => item.info.role === "assistant" && item.info.summary,
        )

        expect(summary?.info.role).toBe("assistant")
        expect(summary?.parts.some((part) => part.type === "tool")).toBe(false)
      }).pipe(withCompaction({ llm: stub.llmLayer }))
    },
    { git: true },
  )

  itCompaction.instance(
    "summarizes only the head while keeping recent tail out of summary input",
    () => {
      const stub = llm()
      let messages: LLM.StreamInput["messages"] = []
      stub.push(
        reply("summary", (input) => {
          messages = input.messages
        }),
      )
      return Effect.gen(function* () {
        const ssn = yield* SessionNs.Service
        const session = yield* ssn.create({})
        yield* createUserMessage(session.id, `older context${"x".repeat(10_000)}`)
        yield* createUserMessage(session.id, "keep this turn")
        yield* createUserMessage(session.id, "and this one too")
        yield* createCompactionMarker(session.id)

        const msgs = yield* ssn.messages({ sessionID: session.id })
        const parent = msgs.at(-1)?.info.id
        expect(parent).toBeTruthy()
        yield* SessionCompaction.use.process({
          parentID: parent!,
          messages: msgs,
          sessionID: session.id,
          auto: false,
        })

        const captured = JSON.stringify(messages)
        expect(messages).toHaveLength(1)
        expect(messages[0]?.role).toBe("user")
        expect(captured).toContain("Here is the conversation so far:")
        expect(captured).toContain("<conversation>")
        expect(captured.indexOf("[User]: older context")).toBeLessThan(
          captured.indexOf("Create a new anchored summary"),
        )
        expect(captured).toContain("[User]: older context")
        expect(captured).not.toContain("keep this turn")
        expect(captured).not.toContain("and this one too")
        expect(captured).not.toContain("What did we do so far?")
      }).pipe(
        withCompaction({
          llm: stub.llmLayer,
          config: cfg({ tail_turns: 2, preserve_recent_tokens: 10_000 }),
        }),
      )
    },
    { git: true },
  )

  itCompaction.instance(
    "anchors repeated compactions with the previous summary",
    () => {
      const stub = llm()
      let captured = ""
      stub.push(reply("summary one"))
      stub.push(
        reply("summary two", (input) => {
          captured = JSON.stringify(input.messages)
        }),
      )

      return Effect.gen(function* () {
        const ssn = yield* SessionNs.Service
        const session = yield* ssn.create({})
        yield* createUserMessage(session.id, `older context${"x".repeat(10_000)}`)
        yield* createUserMessage(session.id, "keep this turn")
        yield* createCompactionMarker(session.id)

        let msgs = yield* ssn.messages({ sessionID: session.id })
        let parent = msgs.at(-1)?.info.id
        expect(parent).toBeTruthy()
        yield* SessionCompaction.use.process({ parentID: parent!, messages: msgs, sessionID: session.id, auto: false })

        yield* createUserMessage(session.id, `latest turn${"x".repeat(10_000)}`)
        yield* createCompactionMarker(session.id)

        msgs = MessageV2.filterCompacted(yield* MessageV2.stream(session.id))
        parent = msgs.at(-1)?.info.id
        expect(parent).toBeTruthy()
        yield* SessionCompaction.use.process({ parentID: parent!, messages: msgs, sessionID: session.id, auto: false })

        expect(captured).toContain("<prior-summary>")
        expect(captured).toContain("summary one")
        expect(captured.match(/summary one/g)?.length).toBe(1)
        expect(captured.indexOf("latest turn")).toBeLessThan(captured.indexOf("<prior-summary>"))
        expect(captured).toContain("summary of the conversation before the <conversation> above")
        expect(captured).toContain("## Important Details")
        expect(captured).toContain("## Work State")
      }).pipe(withCompaction({ llm: stub.llmLayer }))
    },
    { git: true },
  )

  itCompaction.instance(
    "keeps plugin context outside the serialized conversation",
    () => {
      const stub = llm()
      let captured = ""
      stub.push(
        reply("summary", (input) => {
          captured = JSON.stringify(input.messages)
        }),
      )

      return Effect.gen(function* () {
        const ssn = yield* SessionNs.Service
        const session = yield* ssn.create({})
        yield* createUserMessage(session.id, "older context")
        yield* createUserMessage(session.id, "keep this turn")
        yield* createUserMessage(session.id, "and this one too")
        yield* createCompactionMarker(session.id)

        const msgs = yield* ssn.messages({ sessionID: session.id })
        const parent = msgs.at(-1)?.info.id
        expect(parent).toBeTruthy()
        yield* SessionCompaction.use.process({
          parentID: parent!,
          messages: msgs,
          sessionID: session.id,
          auto: false,
        })

        expect(captured).toContain("Prioritize unresolved migration details")
        expect(captured.indexOf("</conversation>")).toBeLessThan(
          captured.indexOf("Prioritize unresolved migration details"),
        )
      }).pipe(
        withCompaction({
          llm: stub.llmLayer,
          plugin: compactionContext("Prioritize unresolved migration details"),
        }),
      )
    },
    { git: true },
  )

  itCompaction.instance(
    "serializes repeated compaction history as one user message",
    () => {
      const stub = llm()
      let captured: LLM.StreamInput["messages"] = []
      stub.push(
        reply("summary two", (input) => {
          captured = input.messages
        }),
      )

      return Effect.gen(function* () {
        const ssn = yield* SessionNs.Service
        const test = yield* TestInstance
        const session = yield* ssn.create({})
        const turn = yield* createUserMessage(session.id, "original request")
        const kept = yield* createAssistantMessage(session.id, turn.id, test.directory)
        yield* ssn.updatePart({
          id: PartID.ascending(),
          messageID: kept.id,
          sessionID: session.id,
          type: "tool",
          callID: "read-call",
          tool: "read",
          state: {
            status: "completed",
            input: { filePath: "src/index.ts" },
            output: "file contents",
            title: "src/index.ts",
            metadata: {},
            time: { start: Date.now(), end: Date.now() },
          },
        })

        const previous = yield* ssn.updateMessage({
          id: MessageID.ascending(),
          role: "user",
          model: ref,
          sessionID: session.id,
          agent: "build",
          time: { created: Date.now() },
        })
        yield* ssn.updatePart({
          id: PartID.ascending(),
          messageID: previous.id,
          sessionID: session.id,
          type: "compaction",
          auto: false,
          tail_start_id: kept.id,
        })
        yield* createSummaryAssistantMessage(session.id, previous.id, test.directory, "summary one")
        yield* createCompactionMarker(session.id)

        const msgs = MessageV2.filterCompacted(yield* MessageV2.stream(session.id))
        const parent = msgs.at(-1)?.info.id
        expect(parent).toBeTruthy()
        yield* SessionCompaction.use.process({ parentID: parent!, messages: msgs, sessionID: session.id, auto: false })

        expect(captured).toHaveLength(1)
        expect(captured[0]?.role).toBe("user")
        expect(JSON.stringify(captured)).toContain('[Assistant tool call]: read({\\"filePath\\":\\"src/index.ts\\"})')
        expect(JSON.stringify(captured)).toContain("[Tool result]: file contents")
        expect(JSON.stringify(captured)).not.toContain('\\"role\\":\\"assistant\\"')
      }).pipe(withCompaction({ llm: stub.llmLayer, config: cfg({ tail_turns: 0 }) }))
    },
    { git: true },
  )

  itCompaction.instance(
    "omits classifier-rejected output from native compaction model input",
    () => {
      const stub = llm()
      let captured = ""
      stub.push(
        reply("native summary", (input) => {
          captured = JSON.stringify(input.messages)
        }),
      )

      return Effect.gen(function* () {
        const ssn = yield* SessionNs.Service
        const test = yield* TestInstance
        const session = yield* ssn.create({})
        const rejectedUser = yield* createUserMessage(session.id, "rejected request")
        const rejected = yield* createAssistantMessage(session.id, rejectedUser.id, test.directory)
        yield* ssn.updateMessage({
          ...rejected,
          error: new SessionV1.ContentFilterError({ message: "blocked by classifier" }).toObject(),
        })
        yield* ssn.updatePart({
          id: PartID.ascending(),
          messageID: rejected.id,
          sessionID: session.id,
          type: "text",
          text: `CLASSIFIER_REJECTED_TEXT_${"x".repeat(6_000)}`,
        })
        yield* ssn.updatePart({
          id: PartID.ascending(),
          messageID: rejected.id,
          sessionID: session.id,
          type: "tool",
          callID: "rejected-tool",
          tool: "read",
          state: {
            status: "completed",
            input: { filePath: "CLASSIFIER_REJECTED_INPUT" },
            output: "CLASSIFIER_REJECTED_TOOL_OUTPUT",
            title: "rejected",
            metadata: {},
            time: { start: 1, end: 2 },
          },
        })
        const interruptedUser = yield* createUserMessage(session.id, "interrupted request")
        const interrupted = yield* createAssistantMessage(session.id, interruptedUser.id, test.directory)
        yield* ssn.updateMessage({
          ...interrupted,
          error: new SessionV1.AbortedError({ message: "interrupted" }).toObject(),
        })
        yield* ssn.updatePart({
          id: PartID.ascending(),
          messageID: interrupted.id,
          sessionID: session.id,
          type: "tool",
          callID: "completed-before-interrupt",
          tool: "read",
          state: {
            status: "completed",
            input: { filePath: "kept.txt" },
            output: `LEGITIMATE_TOOL_RESULT_${"y".repeat(5_000)}`,
            title: "kept",
            metadata: {},
            time: { start: 3, end: 4 },
          },
        })
        yield* createCompactionMarker(session.id)
        const messages = yield* ssn.messages({ sessionID: session.id })

        yield* SessionCompaction.use.process({
          parentID: messages.at(-1)!.info.id,
          messages,
          sessionID: session.id,
          auto: false,
        })

        expect(captured).toContain(MessageV2.CONTENT_FILTER_NOTE)
        expect(captured).toContain("LEGITIMATE_TOOL_RESULT")
        expect(captured).not.toContain("CLASSIFIER_REJECTED_TEXT")
        expect(captured).not.toContain("CLASSIFIER_REJECTED_INPUT")
        expect(captured).not.toContain("CLASSIFIER_REJECTED_TOOL_OUTPUT")
      }).pipe(withCompaction({ llm: stub.llmLayer, config: cfg({ tail_turns: 0 }) }))
    },
    { git: true },
  )

  itCompaction.instance(
    "serializes manual compaction summaries without expanding original parts",
    () => {
      const stub = llm()
      const longSummary = `LONG_SUMMARY_START${"u".repeat(3_400)}LONG_SUMMARY_END`
      const mixedSummary = "settled assistant text and tool result"
      let captured = ""
      stub.push(
        reply("native summary", (input) => {
          captured = JSON.stringify(input.messages)
        }),
      )

      return Effect.gen(function* () {
        const ssn = yield* SessionNs.Service
        const test = yield* TestInstance
        const session = yield* ssn.create({})
        const user = yield* createUserMessage(session.id, "ORIGINAL_USER_SECRET")
        const userMessage = (yield* ssn.messages({ sessionID: session.id })).find(
          (message) => message.info.id === user.id,
        )!
        const userPart = userMessage.parts.find((part): part is SessionV1.TextPart => part.type === "text")!
        yield* ssn.updatePart({
          ...userPart,
          compacted: Date.now(),
          compactionGroup: "cross-role-group",
          compactionSummary: longSummary,
        })
        const assistant = yield* createAssistantMessage(session.id, user.id, test.directory)
        yield* ssn.updatePart({
          id: PartID.ascending(),
          messageID: assistant.id,
          sessionID: session.id,
          type: "text",
          text: "ORIGINAL_REASONING_SECRET",
          time: { start: 1, end: 2 },
          compacted: Date.now(),
          compactionGroup: "assistant-mixed-group",
          compactionSummary: mixedSummary,
        })
        yield* ssn.updatePart({
          id: PartID.ascending(),
          messageID: assistant.id,
          sessionID: session.id,
          type: "tool",
          callID: "manual-tool",
          tool: "read",
          state: {
            status: "completed",
            input: { secret: "ORIGINAL_INPUT_SECRET" },
            output: "ORIGINAL_OUTPUT_SECRET",
            title: "manual",
            metadata: {},
            time: { start: 1, end: 2 },
            compactionGroup: "assistant-mixed-group",
            compactionSummary: mixedSummary,
          },
        })
        yield* ssn.updatePart({
          id: PartID.ascending(),
          messageID: assistant.id,
          sessionID: session.id,
          type: "tool",
          callID: "manual-tool-without-summary",
          tool: "read",
          state: {
            status: "completed",
            input: { secret: "UNSUMMARIZED_INPUT_SECRET" },
            output: "UNSUMMARIZED_OUTPUT_SECRET",
            title: "manual marker",
            metadata: {},
            time: { start: 1, end: 2 },
            compactionGroup: "marker-only-group",
          },
        })
        yield* createCompactionMarker(session.id)
        const msgs = yield* ssn.messages({ sessionID: session.id })

        yield* SessionCompaction.use.process({
          parentID: msgs.at(-1)!.info.id,
          messages: msgs,
          sessionID: session.id,
          auto: false,
        })

        expect(captured).toContain("LONG_SUMMARY_START")
        expect(captured).toContain("LONG_SUMMARY_END")
        expect(captured.split(mixedSummary)).toHaveLength(3)
        expect(captured).toContain("[Assistant]: [Compacted")
        expect(captured).toContain("[Tool]: [Compacted")
        expect(captured).not.toContain("ORIGINAL_USER_SECRET")
        expect(captured).not.toContain("ORIGINAL_REASONING_SECRET")
        expect(captured).not.toContain("ORIGINAL_INPUT_SECRET")
        expect(captured).not.toContain("ORIGINAL_OUTPUT_SECRET")
        expect(captured).not.toContain("UNSUMMARIZED_INPUT_SECRET")
        expect(captured).not.toContain("UNSUMMARIZED_OUTPUT_SECRET")
      }).pipe(withCompaction({ llm: stub.llmLayer, config: cfg({ tail_turns: 0 }) }))
    },
    { git: true },
  )

  itCompaction.instance("keeps recent pre-compaction turns across repeated compactions", () => {
    const stub = llm()
    stub.push(reply("summary one"))
    stub.push(reply("summary two"))

    return Effect.gen(function* () {
      const ssn = yield* SessionNs.Service
      const session = yield* ssn.create({})
      const u1 = yield* createUserMessage(session.id, `one${"x".repeat(10_000)}`)
      const u2 = yield* createUserMessage(session.id, `two${"x".repeat(10_000)}`)
      const u3 = yield* createUserMessage(session.id, "three")
      yield* createCompactionMarker(session.id)

      let msgs = yield* ssn.messages({ sessionID: session.id })
      let parent = msgs.at(-1)?.info.id
      expect(parent).toBeTruthy()
      yield* SessionCompaction.use.process({ parentID: parent!, messages: msgs, sessionID: session.id, auto: false })

      const u4 = yield* createUserMessage(session.id, "four")
      yield* createCompactionMarker(session.id)

      msgs = MessageV2.filterCompacted(yield* MessageV2.stream(session.id))
      parent = msgs.at(-1)?.info.id
      expect(parent).toBeTruthy()
      yield* SessionCompaction.use.process({ parentID: parent!, messages: msgs, sessionID: session.id, auto: false })

      const filtered = MessageV2.filterCompacted(yield* MessageV2.stream(session.id))
      const ids = filtered.map((msg) => msg.info.id)

      expect(ids).not.toContain(u1.id)
      expect(ids).not.toContain(u2.id)
      expect(ids).toContain(u3.id)
      expect(ids).toContain(u4.id)
      expect(filtered.some((msg) => msg.info.role === "assistant" && msg.info.summary)).toBe(true)
      expect(
        filtered.some((msg) => msg.info.role === "user" && msg.parts.some((part) => part.type === "compaction")),
      ).toBe(true)
    }).pipe(withCompaction({ llm: stub.llmLayer, config: cfg({ tail_turns: 2, preserve_recent_tokens: 10_000 }) }))
  })

  itCompaction.instance(
    "ignores previous summaries when sizing the retained tail",
    Effect.gen(function* () {
      const ssn = yield* SessionNs.Service
      const test = yield* TestInstance
      const session = yield* ssn.create({})
      yield* createUserMessage(session.id, "o".repeat(10_000))
      const keep = yield* createUserMessage(session.id, "keep this turn")
      const keepReply = yield* createAssistantMessage(session.id, keep.id, test.directory)
      yield* ssn.updatePart({
        id: PartID.ascending(),
        messageID: keepReply.id,
        sessionID: session.id,
        type: "text",
        text: "keep reply",
      })

      yield* createCompactionMarker(session.id)
      const firstCompaction = (yield* ssn.messages({ sessionID: session.id })).at(-1)?.info.id
      expect(firstCompaction).toBeTruthy()
      yield* createSummaryAssistantMessage(session.id, firstCompaction!, test.directory, "summary ".repeat(800))

      const recent = yield* createUserMessage(session.id, "recent turn")
      const recentReply = yield* createAssistantMessage(session.id, recent.id, test.directory)
      yield* ssn.updatePart({
        id: PartID.ascending(),
        messageID: recentReply.id,
        sessionID: session.id,
        type: "text",
        text: "recent reply",
      })

      yield* createCompactionMarker(session.id)
      const msgs = yield* ssn.messages({ sessionID: session.id })
      const parent = msgs.at(-1)?.info.id
      expect(parent).toBeTruthy()
      yield* SessionCompaction.use.process({ parentID: parent!, messages: msgs, sessionID: session.id, auto: false })

      const part = yield* readCompactionPart(session.id)
      expect(part?.type).toBe("compaction")
      expect(part?.tail_start_id).toBe(keep.id)
    }).pipe(withCompaction({ config: cfg({ tail_turns: 2, preserve_recent_tokens: 500 }) })),
  )
})

describe("util.token.estimate", () => {
  test("estimates tokens from text (4 chars per token)", () => {
    const text = "x".repeat(4000)
    expect(Token.estimate(text)).toBe(1000)
  })

  test("estimates tokens from larger text", () => {
    const text = "y".repeat(20_000)
    expect(Token.estimate(text)).toBe(5000)
  })

  test("returns 0 for empty string", () => {
    expect(Token.estimate("")).toBe(0)
  })
})

describe("SessionNs.getUsage", () => {
  test("normalizes standard usage to token format", () => {
    const model = createModel({ context: 100_000, output: 32_000 })
    const result = SessionNs.getUsage({
      model,
      usage: usage({ inputTokens: 1000, outputTokens: 500, totalTokens: 1500 }),
    })

    expect(result.tokens.input).toBe(1000)
    expect(result.tokens.output).toBe(500)
    expect(result.tokens.reasoning).toBe(0)
    expect(result.tokens.cache.read).toBe(0)
    expect(result.tokens.cache.write).toBe(0)
  })

  test("extracts cached tokens to cache.read", () => {
    const model = createModel({ context: 100_000, output: 32_000 })
    const result = SessionNs.getUsage({
      model,
      usage: usage({ inputTokens: 1000, outputTokens: 500, totalTokens: 1500, cacheReadInputTokens: 200 }),
    })

    expect(result.tokens.input).toBe(800)
    expect(result.tokens.cache.read).toBe(200)
  })

  test("handles anthropic cache write metadata", () => {
    const model = createModel({ context: 100_000, output: 32_000 })
    const result = SessionNs.getUsage({
      model,
      usage: usage({ inputTokens: 1000, outputTokens: 500, totalTokens: 1500 }),
      metadata: {
        anthropic: {
          cacheCreationInputTokens: 300,
        },
      },
    })

    expect(result.tokens.cache.write).toBe(300)
  })

  test("subtracts cached tokens for anthropic provider", () => {
    const model = createModel({ context: 100_000, output: 32_000 })
    // AI SDK v6 normalizes inputTokens to include cached tokens for all providers
    const result = SessionNs.getUsage({
      model,
      usage: usage({ inputTokens: 1000, outputTokens: 500, totalTokens: 1500, cacheReadInputTokens: 200 }),
      metadata: {
        anthropic: {},
      },
    })

    expect(result.tokens.input).toBe(800)
    expect(result.tokens.cache.read).toBe(200)
  })

  test("separates reasoning tokens from output tokens", () => {
    const model = createModel({ context: 100_000, output: 32_000 })
    const result = SessionNs.getUsage({
      model,
      usage: usage({ inputTokens: 1000, outputTokens: 500, reasoningTokens: 100, totalTokens: 1500 }),
    })

    expect(result.tokens.input).toBe(1000)
    expect(result.tokens.output).toBe(400)
    expect(result.tokens.reasoning).toBe(100)
    expect(result.tokens.total).toBe(1500)
  })

  test("does not double count reasoning tokens in cost", () => {
    const model = createModel({
      context: 100_000,
      output: 32_000,
      cost: {
        input: 0,
        output: 15,
        cache: { read: 0, write: 0 },
      },
    })
    const result = SessionNs.getUsage({
      model,
      usage: usage({ inputTokens: 0, outputTokens: 1_000_000, reasoningTokens: 250_000, totalTokens: 1_000_000 }),
    })

    expect(result.tokens.output).toBe(750_000)
    expect(result.tokens.reasoning).toBe(250_000)
    expect(result.cost).toBe(15)
  })

  test("handles undefined optional values gracefully", () => {
    const model = createModel({ context: 100_000, output: 32_000 })
    const result = SessionNs.getUsage({
      model,
      usage: usage({ inputTokens: 0, outputTokens: 0, totalTokens: 0 }),
    })

    expect(result.tokens.input).toBe(0)
    expect(result.tokens.output).toBe(0)
    expect(result.tokens.reasoning).toBe(0)
    expect(result.tokens.cache.read).toBe(0)
    expect(result.tokens.cache.write).toBe(0)
    expect(Number.isNaN(result.cost)).toBe(false)
  })

  test("ignores malformed cost fields", () => {
    const model = createModel({
      context: 100_000,
      output: 32_000,
      cost: { input: 3, output: 15, cache: { read: 0.3, write: 3.75 } },
    })
    Object.assign(model.cost, { input: {} })

    const result = SessionNs.getUsage({
      model,
      usage: usage({ inputTokens: 1_000_000, outputTokens: 100_000, totalTokens: 1_100_000 }),
    })

    expect(result.cost).toBe(1.5)
  })

  test("calculates cost correctly", () => {
    const model = createModel({
      context: 100_000,
      output: 32_000,
      cost: {
        input: 3,
        output: 15,
        cache: { read: 0.3, write: 3.75 },
      },
    })
    const result = SessionNs.getUsage({
      model,
      usage: usage({ inputTokens: 1_000_000, outputTokens: 100_000, totalTokens: 1_100_000 }),
    })

    expect(result.cost).toBe(3 + 1.5)
  })

  test("uses authoritative Copilot billed cost when provided", () => {
    const result = SessionNs.getUsage({
      model: createModel({
        context: 100_000,
        output: 32_000,
        cost: { input: 3, output: 15, cache: { read: 0.3, write: 0.3 } },
      }),
      usage: usage({ inputTokens: 11_774, outputTokens: 39, totalTokens: 11_813 }),
      metadata: { copilot: { totalNanoAiu: 4_473_525_000 } },
    })

    expect(result.cost).toBe(0.04473525)
  })

  test("uses matching context cost tier before over-200k fallback", () => {
    const model = createModel({
      context: 1_000_000,
      output: 32_000,
      cost: {
        input: 1,
        output: 2,
        cache: { read: 0.1, write: 0.5 },
        tiers: [
          {
            input: 3,
            output: 4,
            cache: { read: 0.3, write: 1.5 },
            tier: { type: "context", size: 200_000 },
          },
          {
            input: 5,
            output: 6,
            cache: { read: 0.5, write: 2.5 },
            tier: { type: "context", size: 500_000 },
          },
        ],
        experimentalOver200K: {
          input: 100,
          output: 100,
          cache: { read: 100, write: 100 },
        },
      },
    })
    const result = SessionNs.getUsage({
      model,
      usage: usage({
        inputTokens: 650_000,
        outputTokens: 100_000,
        totalTokens: 750_000,
        cacheReadInputTokens: 100_000,
      }),
    })

    expect(result.tokens.input).toBe(550_000)
    expect(result.cost).toBe(2.75 + 0.6 + 0.05)
  })

  test("falls back to over-200k pricing when no cost tier matches", () => {
    const model = createModel({
      context: 1_000_000,
      output: 32_000,
      cost: {
        input: 1,
        output: 2,
        cache: { read: 0.1, write: 0.5 },
        tiers: [
          {
            input: 5,
            output: 6,
            cache: { read: 0.5, write: 2.5 },
            tier: { type: "context", size: 500_000 },
          },
        ],
        experimentalOver200K: {
          input: 3,
          output: 4,
          cache: { read: 0.3, write: 1.5 },
        },
      },
    })
    const result = SessionNs.getUsage({
      model,
      usage: usage({ inputTokens: 300_000, outputTokens: 100_000, totalTokens: 400_000 }),
    })

    expect(result.cost).toBe(0.9 + 0.4)
  })

  test.each(["@ai-sdk/anthropic", "@ai-sdk/amazon-bedrock", "@ai-sdk/google-vertex/anthropic"])(
    "computes total from components for %s models",
    (npm) => {
      const model = createModel({ context: 100_000, output: 32_000, npm })
      // AI SDK v6: inputTokens includes cached tokens for all providers
      const item = usage({
        inputTokens: 1000,
        outputTokens: 500,
        totalTokens: 1500,
        cacheReadInputTokens: 200,
      })
      if (npm === "@ai-sdk/amazon-bedrock") {
        const result = SessionNs.getUsage({
          model,
          usage: item,
          metadata: {
            bedrock: {
              usage: {
                cacheWriteInputTokens: 300,
              },
            },
          },
        })

        // inputTokens (1000) includes cache, so adjusted = 1000 - 200 - 300 = 500
        expect(result.tokens.input).toBe(500)
        expect(result.tokens.cache.read).toBe(200)
        expect(result.tokens.cache.write).toBe(300)
        // total = adjusted (500) + output (500) + cacheRead (200) + cacheWrite (300)
        expect(result.tokens.total).toBe(1500)
        return
      }

      const result = SessionNs.getUsage({
        model,
        usage: item,
        metadata: {
          anthropic: {
            cacheCreationInputTokens: 300,
          },
        },
      })

      // inputTokens (1000) includes cache, so adjusted = 1000 - 200 - 300 = 500
      expect(result.tokens.input).toBe(500)
      expect(result.tokens.cache.read).toBe(200)
      expect(result.tokens.cache.write).toBe(300)
      // total = adjusted (500) + output (500) + cacheRead (200) + cacheWrite (300)
      expect(result.tokens.total).toBe(1500)
    },
  )

  test("extracts cache write tokens from vertex metadata key", () => {
    const model = createModel({ context: 100_000, output: 32_000, npm: "@ai-sdk/google-vertex/anthropic" })
    const result = SessionNs.getUsage({
      model,
      usage: usage({ inputTokens: 1000, outputTokens: 500, totalTokens: 1500, cacheReadInputTokens: 200 }),
      metadata: {
        vertex: {
          cacheCreationInputTokens: 300,
        },
      },
    })

    expect(result.tokens.input).toBe(500)
    expect(result.tokens.cache.read).toBe(200)
    expect(result.tokens.cache.write).toBe(300)
  })
})
