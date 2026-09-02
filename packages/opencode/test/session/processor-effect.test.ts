import { SessionV1 } from "@opencode-ai/core/v1/session"
import { Database } from "@opencode-ai/core/database/database"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { EventV2Bridge } from "@/event-v2-bridge"
import { expect } from "bun:test"
import { tool } from "ai"
import { Cause, Context, Deferred, Effect, Exit, Fiber, Layer, Option, Schema, Stream } from "effect"
import path from "path"
import z from "zod"
import type { Agent } from "../../src/agent/agent"
import { Provider } from "@/provider/provider"

import { Session } from "@/session/session"
import { LLM } from "../../src/session/llm"
import { MessageV2 } from "../../src/session/message-v2"
import { SessionProcessor } from "../../src/session/processor"
import { MessageID, PartID, SessionID } from "../../src/session/schema"
import { SessionStatus } from "../../src/session/status"
import { SessionSummary } from "../../src/session/summary"
import { CrossSpawnSpawner } from "@opencode-ai/core/cross-spawn-spawner"
import { provideTmpdirInstance, provideTmpdirServer } from "../fixture/fixture"
import { testEffect } from "../lib/effect"
import { raw, reply, TestLLMServer } from "../lib/llm-server"
import { RuntimeFlags } from "@/effect/runtime-flags"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { ModelV2 } from "@opencode-ai/core/model"
import { SessionProjector } from "@opencode-ai/core/session/projector"
import { LLMEvent, Usage } from "@opencode-ai/llm"
import { SessionTools } from "@/session/tools"
import { ToolRegistry } from "@/tool/registry"
import { Tool } from "@/tool/tool"
import { MCP } from "@/mcp"
import { Truncate } from "@/tool/truncate"
import { Plugin } from "@/plugin"
import { Permission } from "@/permission"
import { Image } from "@/image/image"

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

const cfg = {
  provider: {
    test: {
      name: "Test",
      id: "test",
      env: [],
      npm: "@ai-sdk/openai-compatible",
      models: {
        "test-model": {
          id: "test-model",
          name: "Test Model",
          attachment: false,
          reasoning: false,
          temperature: false,
          tool_call: true,
          release_date: "2025-01-01",
          limit: { context: 100000, output: 10000 },
          cost: { input: 0, output: 0 },
          options: {},
        },
      },
      options: {
        apiKey: "test-key",
        baseURL: "http://localhost:1/v1",
      },
    },
  },
}

function providerCfg(url: string) {
  return {
    ...cfg,
    provider: {
      ...cfg.provider,
      test: {
        ...cfg.provider.test,
        options: {
          ...cfg.provider.test.options,
          baseURL: url,
        },
      },
    },
  }
}

function agent(): Agent.Info {
  return {
    name: "build",
    mode: "primary",
    options: {},
    permission: [{ permission: "*", pattern: "*", action: "allow" }],
  }
}

function defer<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

const waitFor = <A>(check: Effect.Effect<A | undefined>, message: string) =>
  Effect.gen(function* () {
    const stop = Date.now() + 500
    while (Date.now() < stop) {
      const value = yield* check
      if (value !== undefined) return value
      yield* Effect.sleep("10 millis")
    }
    return yield* Effect.fail(new Error(message))
  })

const user = Effect.fn("TestSession.user")(function* (sessionID: SessionID, text: string) {
  const session = yield* Session.Service
  const msg = yield* session.updateMessage({
    id: MessageID.ascending(),
    role: "user",
    sessionID,
    agent: "build",
    model: ref,
    time: { created: Date.now() },
  })
  yield* session.updatePart({
    id: PartID.ascending(),
    messageID: msg.id,
    sessionID,
    type: "text",
    text,
  })
  return msg
})

const assistant = Effect.fn("TestSession.assistant")(function* (
  sessionID: SessionID,
  parentID: MessageID,
  root: string,
) {
  const session = yield* Session.Service
  const msg: SessionV1.Assistant = {
    id: MessageID.ascending(),
    role: "assistant",
    sessionID,
    mode: "build",
    agent: "build",
    path: { cwd: root, root },
    cost: 0,
    tokens: {
      total: 0,
      input: 0,
      output: 0,
      reasoning: 0,
      cache: { read: 0, write: 0 },
    },
    modelID: ref.modelID,
    providerID: ref.providerID,
    parentID,
    time: { created: Date.now() },
    finish: "end_turn",
  }
  yield* session.updateMessage(msg)
  return msg
})

const root = LayerNode.group([
  SessionProcessor.node,
  Session.node,
  SessionProjector.node,
  Provider.node,
  Database.node,
  EventV2Bridge.node,
  SessionStatus.node,
  CrossSpawnSpawner.node,
])
const replacements = [
  [SessionSummary.node, summary],
  [RuntimeFlags.node, RuntimeFlags.layer({ experimentalEventSystem: true })],
] as const
const env = LayerNode.compile(
  LayerNode.group([
    root,
    LayerNode.make({ service: TestLLMServer, layer: TestLLMServer.layer, deps: [] }),
    Plugin.node,
    Permission.node,
    RuntimeFlags.node,
  ]),
  replacements,
)

const it = testEffect(env)

const providerErrorLLM = Layer.succeed(
  LLM.Service,
  LLM.Service.of({
    stream: () =>
      Stream.make(
        LLMEvent.stepStart({ index: 0 }),
        LLMEvent.toolInputStart({ id: "call-1", name: "lookup" }),
        LLMEvent.toolInputEnd({ id: "call-1", name: "lookup" }),
        LLMEvent.toolCall({ id: "call-1", name: "lookup", input: {}, providerExecuted: true }),
        LLMEvent.toolResult({
          id: "call-1",
          name: "lookup",
          result: { type: "error", value: "provider boom" },
          providerExecuted: true,
        }),
        LLMEvent.stepFinish({ index: 0, reason: "stop" }),
        LLMEvent.finish({ reason: "stop" }),
      ),
  }),
)
const providerErrorEnv = LayerNode.compile(root, [...replacements, [LLM.node, providerErrorLLM]])
const itProviderError = testEffect(providerErrorEnv)

const fragmentFailureLLM = Layer.succeed(
  LLM.Service,
  LLM.Service.of({
    stream: () =>
      Stream.make(
        LLMEvent.stepStart({ index: 0 }),
        LLMEvent.reasoningStart({ id: "reasoning-1" }),
        LLMEvent.reasoningDelta({ id: "reasoning-1", text: "thinking" }),
        LLMEvent.textStart({ id: "text-1" }),
        LLMEvent.textDelta({ id: "text-1", text: "partial" }),
        LLMEvent.providerError({ message: "provider boom" }),
      ),
  }),
)
const fragmentFailureEnv = LayerNode.compile(root, [...replacements, [LLM.node, fragmentFailureLLM]])
const itFragmentFailure = testEffect(fragmentFailureEnv)

const discardedUsage = new Usage({ inputTokens: 19, outputTokens: 11, totalTokens: 30 })
const successfulUsage = new Usage({ inputTokens: 5, outputTokens: 3, totalTokens: 8 })

class RetryAccountingState extends Context.Service<RetryAccountingState, { calls: number }>()(
  "@test/RetryAccountingState",
) {}

const retryAccountingState = Layer.effect(
  RetryAccountingState,
  Effect.sync(() => RetryAccountingState.of({ calls: 0 })),
)
const retryAccountingStateNode = LayerNode.make({
  service: RetryAccountingState,
  layer: retryAccountingState,
  deps: [],
})
const retryAccountingLLM = Layer.effect(
  LLM.Service,
  Effect.gen(function* () {
    const state = yield* RetryAccountingState
    return LLM.Service.of({
      stream: () => {
        state.calls++
        const first = state.calls === 1
        const events = Stream.make(
          LLMEvent.stepStart({ index: 0 }),
          LLMEvent.stepFinish({
            index: 0,
            reason: first ? "length" : "stop",
            usage: first ? discardedUsage : successfulUsage,
          }),
        )
        if (first) return events.pipe(Stream.concat(Stream.make(LLMEvent.providerError({ message: "terminated" }))))
        return events.pipe(Stream.concat(Stream.make(LLMEvent.finish({ reason: "stop", usage: successfulUsage }))))
      },
    })
  }),
)
const retryAccountingLLMNode = LayerNode.make({
  service: LLM.Service,
  layer: retryAccountingLLM,
  deps: [retryAccountingStateNode],
})
const retryAccountingEnv = LayerNode.compile(LayerNode.group([root, retryAccountingStateNode]), [
  ...replacements,
  [LLM.node, retryAccountingLLMNode],
])
const itRetryAccounting = testEffect(retryAccountingEnv)

type RecoveryStateValue = {
  mode: "recovery" | "cleanup" | "retry" | "sibling"
  started: Deferred.Deferred<void>
  release: Deferred.Deferred<void>
  finished: Deferred.Deferred<void>
  cancelled: Deferred.Deferred<void>
  callbackReady: Deferred.Deferred<void>
  retryStarted: Deferred.Deferred<void>
  finishRetry: Deferred.Deferred<void>
  delayed?: () => Promise<unknown>
  providerCalls: number
  toolCalls: number
  cancellations: number
  image: boolean
  normalizations: number
}

class RecoveryState extends Context.Service<RecoveryState, RecoveryStateValue>()("@test/RecoveryState") {}

const recoveryStateLayer = Layer.effect(
  RecoveryState,
  Effect.gen(function* () {
    return RecoveryState.of({
      mode: "recovery",
      started: yield* Deferred.make<void>(),
      release: yield* Deferred.make<void>(),
      finished: yield* Deferred.make<void>(),
      cancelled: yield* Deferred.make<void>(),
      callbackReady: yield* Deferred.make<void>(),
      retryStarted: yield* Deferred.make<void>(),
      finishRetry: yield* Deferred.make<void>(),
      providerCalls: 0,
      toolCalls: 0,
      cancellations: 0,
      image: false,
      normalizations: 0,
    })
  }),
)
const recoveryStateNode = LayerNode.make({ service: RecoveryState, layer: recoveryStateLayer, deps: [] })
const recoveryLLM = Layer.effect(
  LLM.Service,
  Effect.gen(function* () {
    const state = yield* RecoveryState
    return LLM.Service.of({
      stream: (input) => {
        state.providerCalls += 1
        const execute = input.tools.lookup?.execute
        if (!execute) return Stream.fail(new Error("lookup tool is missing"))
        const transport = new AbortController()
        const invoke = () =>
          Promise.resolve(
            execute({}, { toolCallId: "call-1", messages: input.messages, abortSignal: transport.signal }),
          )
        if (state.mode === "cleanup") {
          return Stream.fromEffectDrain(
            Effect.sync(() => {
              state.delayed = invoke
            }).pipe(Effect.andThen(Deferred.succeed(state.callbackReady, undefined))),
          )
        }
        if (state.mode === "retry") {
          if (state.providerCalls === 1) {
            return Stream.fromEffectDrain(
              Effect.sync(() => {
                state.delayed = invoke
              }).pipe(Effect.andThen(Deferred.succeed(state.callbackReady, undefined))),
            ).pipe(Stream.concat(Stream.fail(new Error("terminated"))))
          }
          return Stream.fromEffectDrain(Deferred.succeed(state.retryStarted, undefined)).pipe(
            Stream.concat(Stream.fromEffectDrain(Deferred.await(state.finishRetry))),
          )
        }
        const events = Stream.make(
          LLMEvent.stepStart({ index: 0 }),
          LLMEvent.toolInputStart({ id: "call-1", name: "lookup" }),
          LLMEvent.toolInputEnd({ id: "call-1", name: "lookup" }),
          LLMEvent.toolCall({ id: "call-1", name: "lookup", input: {} }),
          ...(state.mode === "sibling"
            ? [
                LLMEvent.toolInputStart({ id: "call-fail", name: "failing" }),
                LLMEvent.toolInputEnd({ id: "call-fail", name: "failing" }),
                LLMEvent.toolCall({ id: "call-fail", name: "failing", input: {} }),
                LLMEvent.toolResult({
                  id: "call-fail",
                  name: "failing",
                  result: { type: "error", value: "sibling failed" },
                }),
              ]
            : []),
        )
        return events.pipe(
          Stream.concat(
            Stream.fromEffectDrain(
              Effect.gen(function* () {
                yield* Effect.sync(() => {
                  void invoke().catch(() => {})
                })
                yield* Deferred.await(state.started)
                transport.abort()
                return yield* Effect.fail(new Error("terminated"))
              }),
            ),
          ),
        )
      },
    })
  }),
)
const recoveryLLMNode = LayerNode.make({
  service: LLM.Service,
  layer: recoveryLLM,
  deps: [recoveryStateNode],
})
const recoveryImage = Layer.effect(
  Image.Service,
  Effect.gen(function* () {
    const state = yield* RecoveryState
    return Image.Service.of({
      normalize: (input) =>
        Effect.sync(() => {
          state.normalizations += 1
          return { ...input, mime: "image/webp", url: "data:image/webp;base64,normalized" }
        }),
    })
  }),
)
const recoveryImageNode = LayerNode.make({
  service: Image.Service,
  layer: recoveryImage,
  deps: [recoveryStateNode],
})
const recoveryEnv = LayerNode.compile(
  LayerNode.group([root, recoveryStateNode, Plugin.node, Permission.node, RuntimeFlags.node]),
  [...replacements, [LLM.node, recoveryLLMNode], [Image.node, recoveryImageNode]],
)
const itRecovery = testEffect(recoveryEnv)

const emptyParameters = Schema.Struct({})

function recoveryRegistry(state: RecoveryStateValue) {
  const held: Tool.Def<typeof emptyParameters> = {
    id: "lookup",
    description: "Held lookup",
    parameters: emptyParameters,
    jsonSchema: { type: "object", properties: {}, additionalProperties: false },
    execute: (_args, ctx) =>
      Effect.gen(function* () {
        state.toolCalls += 1
        if (state.mode === "cleanup" || state.mode === "retry") {
          return { title: "Unexpected lookup", metadata: {}, output: "unexpected" }
        }
        const onAbort = () => {
          state.cancellations += 1
          Effect.runSync(Deferred.succeed(state.cancelled, undefined).pipe(Effect.ignore))
        }
        ctx.abort.addEventListener("abort", onAbort, { once: true })
        if (ctx.abort.aborted) onAbort()
        yield* Deferred.succeed(state.started, undefined)
        yield* Deferred.await(state.release)
        ctx.abort.removeEventListener("abort", onAbort)
        yield* Deferred.succeed(state.finished, undefined)
        return {
          title: "Lookup complete",
          metadata: { source: "held" },
          output: "held result",
          attachments: state.image
            ? [{ type: "file", mime: "image/png", url: "data:image/png;base64,original" }]
            : undefined,
        }
      }),
  }
  return ToolRegistry.Service.of({
    ids: () => Effect.succeed([held.id]),
    all: () => Effect.succeed([held]),
    named: () => Effect.die("unused"),
    askTools: () => Effect.die("unused"),
    tools: () => Effect.succeed([held]),
  })
}

const emptyMcp = MCP.Service.of({
  status: () => Effect.succeed({}),
  clients: () => Effect.succeed({}),
  instructions: () => Effect.succeed([]),
  tools: () => Effect.succeed({}),
  prompts: () => Effect.succeed({}),
  resources: () => Effect.succeed({}),
  resourceTemplates: () => Effect.succeed({}),
  add: () => Effect.die("unused"),
  connect: () => Effect.die("unused"),
  disconnect: () => Effect.die("unused"),
  getPrompt: () => Effect.die("unused"),
  readResource: () => Effect.die("unused"),
  startAuth: () => Effect.die("unused"),
  authenticate: () => Effect.die("unused"),
  finishAuth: () => Effect.die("unused"),
  removeAuth: () => Effect.die("unused"),
  supportsOAuth: () => Effect.die("unused"),
  hasStoredTokens: () => Effect.die("unused"),
  getAuthStatus: () => Effect.succeed("not_authenticated"),
})

const passthroughTruncate = Truncate.Service.of({
  cleanup: () => Effect.void,
  write: (text) => Effect.succeed(text),
  output: (text) => Effect.succeed({ content: text, truncated: false }),
  limits: () => Effect.succeed({ maxLines: Truncate.MAX_LINES, maxBytes: Truncate.MAX_BYTES }),
})

const recoverySetup = Effect.fn("test.recoverySetup")(function* (root: string) {
  const state = yield* RecoveryState
  const { processors, session, provider } = yield* boot()
  const chat = yield* session.create({})
  const parent = yield* user(chat.id, "recover tool")
  const msg = yield* assistant(chat.id, parent.id, root)
  const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
  const handle = yield* processors.create({ assistantMessage: msg, sessionID: chat.id, model: mdl })
  const tools = yield* SessionTools.resolve({
    agent: agent(),
    model: mdl,
    session: chat,
    processor: handle,
    bypassAgentCheck: false,
    messages: yield* session.messages({ sessionID: chat.id }),
    promptOps: {
      cancel: () => Effect.void,
      checkpoint: () => Effect.succeed(0),
      retain: () => Effect.succeed(Option.some(Effect.void)),
      admitIfCurrent: (effect) => effect.pipe(Effect.map(Option.some)),
      admitChild: (_sessionID, effect) => effect.pipe(Effect.map(Option.some)),
      resolvePromptParts: () => Effect.die("unused"),
      prompt: () => Effect.die("unused"),
      admitNotification: () => Effect.die("unused"),
      notify: () => Effect.void,
    },
  }).pipe(
    Effect.provideService(ToolRegistry.Service, recoveryRegistry(state)),
    Effect.provideService(MCP.Service, emptyMcp),
    Effect.provideService(Truncate.Service, passthroughTruncate),
  )
  return {
    state,
    session,
    msg,
    handle,
    input: {
      user: parent,
      sessionID: chat.id,
      model: mdl,
      agent: agent(),
      system: [],
      messages: [{ role: "user" as const, content: "recover tool" }],
      tools,
    } satisfies LLM.StreamInput,
  }
})

const boot = Effect.fn("test.boot")(function* () {
  const processors = yield* SessionProcessor.Service
  const session = yield* Session.Service
  const provider = yield* Provider.Service
  return { processors, session, provider }
})

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

it.live("session.processor effect tests capture llm input cleanly", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const database = yield* Database.Service
        const { processors, session, provider } = yield* boot()

        yield* llm.text("hello")

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "hi")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const input = {
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "hi" }],
          tools: {},
        } satisfies LLM.StreamInput

        const value = yield* handle.process(input)
        const parts = yield* MessageV2.parts(msg.id)
        const calls = yield* llm.calls

        expect(value).toBe("continue")
        expect(calls).toBe(1)
        expect(parts.some((part) => part.type === "text" && part.text === "hello")).toBe(true)
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests preserve text start time", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const database = yield* Database.Service
        const gate = defer<void>()
        const { processors, session, provider } = yield* boot()

        yield* llm.push(
          raw({
            head: [
              {
                id: "chatcmpl-test",
                object: "chat.completion.chunk",
                choices: [{ delta: { role: "assistant" } }],
              },
              {
                id: "chatcmpl-test",
                object: "chat.completion.chunk",
                choices: [{ delta: { content: "hello" } }],
              },
            ],
            wait: gate.promise,
            tail: [
              {
                id: "chatcmpl-test",
                object: "chat.completion.chunk",
                choices: [{ delta: {}, finish_reason: "stop" }],
              },
            ],
          }),
        )

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "hi")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const run = yield* handle
          .process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "hi" }],
            tools: {},
          })
          .pipe(Effect.forkChild)

        yield* waitFor(
          MessageV2.parts(msg.id).pipe(
            Effect.map((parts) => parts.find((part): part is SessionV1.TextPart => part.type === "text")),
            Effect.provideService(Database.Service, database),
          ),
          "timed out waiting for text part",
        )
        yield* Effect.sleep("20 millis")
        gate.resolve()

        const exit = yield* Fiber.await(run)
        const text = (yield* MessageV2.parts(msg.id)).find((part): part is SessionV1.TextPart => part.type === "text")

        expect(Exit.isSuccess(exit)).toBe(true)
        expect(text?.text).toBe("hello")
        expect(text?.time?.start).toBeDefined()
        expect(text?.time?.end).toBeDefined()
        if (!text?.time?.start || !text.time.end) return
        expect(text.time.start).toBeLessThan(text.time.end)
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests stop after token overflow requests compaction", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const database = yield* Database.Service
        const { processors, session, provider } = yield* boot()

        yield* llm.text("after", { usage: { input: 100, output: 0 } })

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "compact")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const base = yield* provider.getModel(ref.providerID, ref.modelID)
        const mdl = { ...base, limit: { context: 20, output: 10 } }
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const value = yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "compact" }],
          tools: {},
        })

        const parts = yield* MessageV2.parts(msg.id)

        expect(value).toBe("compact")
        expect(parts.some((part) => part.type === "text" && part.text === "after")).toBe(true)
        expect(parts.some((part) => part.type === "step-finish")).toBe(true)
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests capture reasoning from http mock", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const database = yield* Database.Service
        const { processors, session, provider } = yield* boot()

        yield* llm.push(reply().reason("think").text("done").stop())

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "reason")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const value = yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "reason" }],
          tools: {},
        })

        const parts = yield* MessageV2.parts(msg.id)
        const reasoning = parts.find((part): part is SessionV1.ReasoningPart => part.type === "reasoning")
        const text = parts.find((part): part is SessionV1.TextPart => part.type === "text")

        expect(value).toBe("continue")
        expect(yield* llm.calls).toBe(1)
        expect(reasoning?.text).toBe("think")
        expect(text?.text).toBe("done")
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests reset reasoning state across retries", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()

        yield* llm.push(reply().reason("one").reset(), reply().reason("two").stop())

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "reason")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const value = yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "reason" }],
          tools: {},
        })

        const parts = yield* MessageV2.parts(msg.id)
        const reasoning = parts.filter((part): part is SessionV1.ReasoningPart => part.type === "reasoning")

        expect(value).toBe("continue")
        expect(yield* llm.calls).toBe(2)
        // The retry neither appends to the abandoned part nor leaves it behind:
        // "one" is removed with the rest of the failed attempt.
        expect(reasoning.map((part) => part.text)).toEqual(["two"])
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests do not retry unknown json errors", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()

        yield* llm.error(400, { error: { message: "no_kv_space" } })

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "json")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const value = yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "json" }],
          tools: {},
        })

        expect(value).toBe("stop")
        expect(yield* llm.calls).toBe(1)
        expect(handle.message.error?.name).toBe("APIError")
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests retry recognized structured json errors", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()

        yield* llm.error(429, { type: "error", error: { type: "too_many_requests" } })
        yield* llm.text("after")

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "retry json")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const value = yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "retry json" }],
          tools: {},
        })

        const parts = yield* MessageV2.parts(msg.id)

        expect(value).toBe("continue")
        expect(yield* llm.calls).toBe(2)
        expect(parts.some((part) => part.type === "text" && part.text === "after")).toBe(true)
        expect(handle.message.error).toBeUndefined()
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests retry OpenAI-compatible midstream server errors", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()

        yield* llm.push(raw({ chunks: [{ error: { type: "server_error", code: "server_error", message: "xxx" } }] }))
        yield* llm.text("after")

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "retry midstream server error")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const value = yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "retry midstream server error" }],
          tools: {},
        })

        const parts = yield* MessageV2.parts(msg.id)

        expect(value).toBe("continue")
        expect(yield* llm.calls).toBe(2)
        expect(parts.some((part) => part.type === "text" && part.text === "after")).toBe(true)
        expect(handle.message.error).toBeUndefined()
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests retry network_error finish reasons", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()

        yield* llm.push(
          raw({
            chunks: [
              {
                id: "chatcmpl-network-error",
                object: "chat.completion.chunk",
                choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: "network_error" }],
              },
            ],
          }),
        )
        yield* llm.text("after retry")

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "retry network error")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const value = yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "retry network error" }],
          tools: {},
        })

        const parts = yield* MessageV2.parts(msg.id)

        expect(value).toBe("continue")
        expect(yield* llm.calls).toBe(2)
        expect(parts.some((part) => part.type === "text" && part.text === "after retry")).toBe(true)
        expect(handle.message.error).toBeUndefined()
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests publish retry status updates", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()
        const events = yield* EventV2Bridge.Service

        yield* llm.error(503, { error: "boom" })
        yield* llm.text("")

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "retry")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const states: number[] = []
        const off = yield* events.listen((evt) => {
          if (evt.type !== SessionStatus.Event.Status.type) return Effect.void
          const data = evt.data as typeof SessionStatus.Event.Status.data.Type
          if (data.sessionID === chat.id && data.status.type === "retry") states.push(data.status.attempt)
          return Effect.void
        })
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const value = yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "retry" }],
          tools: {},
        })

        yield* off

        expect(value).toBe("continue")
        expect(yield* llm.calls).toBe(2)
        expect(states).toStrictEqual([1])
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests discard a failed attempt's parts before retrying", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()

        // A genuine mid-stream drop: text is written, then the connection is
        // reset. Retrying replays the request unchanged, so the truncated text
        // has to be removed or the transcript ends up holding both it and the
        // full replacement.
        yield* llm.push(reply().text("partial out").reset())
        yield* llm.text("recovered")

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "partial")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const value = yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "partial" }],
          tools: {},
        })

        expect(value).toBe("continue")
        expect(yield* llm.calls).toBe(2)
        expect(handle.message.error).toBeUndefined()
        const parts = yield* MessageV2.parts(msg.id)
        expect(
          parts.filter((part): part is SessionV1.TextPart => part.type === "text").map((part) => part.text),
        ).toEqual(["recovered"])
        expect(parts.filter((part) => part.type === "step-start")).toHaveLength(1)
      }),
    { config: (url) => providerCfg(url) },
  ),
)

itRetryAccounting.live("session.processor effect tests reverse terminal artifacts and usage before retrying", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()
        const events = yield* EventV2Bridge.Service
        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "retry accounting")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })
        yield* Effect.promise(() => Bun.write(path.join(dir, "discarded-attempt.txt"), "discarded"))
        let patchWritten = false
        const off = yield* events.listen((event) => {
          if (event.type !== SessionV1.Event.PartUpdated.type) return Effect.void
          const data = event.data as typeof SessionV1.Event.PartUpdated.data.Type
          if (data.sessionID === chat.id && data.part.messageID === msg.id && data.part.type === "patch")
            patchWritten = true
          return Effect.void
        })

        const value = yield* handle.process({
          user: parent,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "retry accounting" }],
          tools: {},
        })
        yield* off

        const expected = Session.getUsage({ model: mdl, usage: successfulUsage })
        const parts = yield* MessageV2.parts(msg.id)
        const aggregate = yield* session.get(chat.id)

        expect(value).toBe("continue")
        expect(patchWritten).toBe(true)
        expect(parts.filter((part) => part.type === "step-start")).toHaveLength(1)
        expect(parts.filter((part) => part.type === "step-finish")).toHaveLength(1)
        expect(parts.filter((part) => part.type === "patch")).toHaveLength(0)
        expect(handle.message.finish).toBe("stop")
        expect(handle.message.cost).toBe(expected.cost)
        expect(handle.message.tokens).toEqual(expected.tokens)
        expect(aggregate.cost).toBe(expected.cost)
        expect(aggregate.tokens).toEqual({
          input: expected.tokens.input,
          output: expected.tokens.output,
          reasoning: expected.tokens.reasoning,
          cache: expected.tokens.cache,
        })
      }),
    {
      git: true,
      config: {
        ...cfg,
        provider: {
          test: {
            ...cfg.provider.test,
            models: {
              "test-model": {
                ...cfg.provider.test.models["test-model"],
                cost: { input: 1, output: 2 },
              },
            },
          },
        },
      },
    },
  ),
)

itRetryAccounting.live("session.processor effect tests roll back before cancellation during retry backoff", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()
        const state = yield* RetryAccountingState
        const statuses = yield* SessionStatus.Service
        const events = yield* EventV2Bridge.Service
        const retryStarted = yield* Deferred.make<void>()
        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "cancel retry accounting")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })
        yield* Effect.promise(() => Bun.write(path.join(dir, "discarded-attempt.txt"), "discarded"))
        let patchWritten = false
        const off = yield* events.listen((event) => {
          if (event.type === SessionV1.Event.PartUpdated.type) {
            const data = event.data as typeof SessionV1.Event.PartUpdated.data.Type
            if (data.sessionID === chat.id && data.part.messageID === msg.id && data.part.type === "patch") {
              patchWritten = true
            }
            return Effect.void
          }
          if (event.type !== SessionStatus.Event.Status.type) return Effect.void
          const data = event.data as typeof SessionStatus.Event.Status.data.Type
          if (data.sessionID !== chat.id || data.status.type !== "retry") return Effect.void
          return Deferred.succeed(retryStarted, undefined).pipe(Effect.asVoid)
        })

        const run = yield* handle
          .process({
            user: parent,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "cancel retry accounting" }],
            tools: {},
          })
          .pipe(Effect.forkChild)

        yield* Deferred.await(retryStarted)
        yield* waitFor(
          statuses.get(chat.id).pipe(Effect.map((status) => (status.type === "retry" ? true : undefined))),
          "retry status was not stored",
        )
        yield* Effect.yieldNow
        expect(state.calls).toBe(1)
        yield* Fiber.interrupt(run)
        const exit = yield* Fiber.await(run)
        yield* off

        const parts = yield* MessageV2.parts(msg.id)
        const stored = yield* MessageV2.get({ sessionID: chat.id, messageID: msg.id })
        const aggregate = yield* session.get(chat.id)
        const emptyTokens = {
          total: 0,
          input: 0,
          output: 0,
          reasoning: 0,
          cache: { read: 0, write: 0 },
        }

        expect(Exit.isFailure(exit)).toBe(true)
        if (Exit.isFailure(exit)) expect(Cause.hasInterruptsOnly(exit.cause)).toBe(true)
        expect(patchWritten).toBe(true)
        expect(state.calls).toBe(1)
        expect(parts.filter((part) => part.type === "step-start")).toHaveLength(0)
        expect(parts.filter((part) => part.type === "step-finish")).toHaveLength(0)
        expect(parts.filter((part) => part.type === "patch")).toHaveLength(0)
        expect(handle.message.finish).toBe("end_turn")
        expect(handle.message.cost).toBe(0)
        expect(handle.message.tokens).toEqual(emptyTokens)
        expect(stored.info.role).toBe("assistant")
        if (stored.info.role === "assistant") {
          expect(stored.info.finish).toBe("end_turn")
          expect(stored.info.cost).toBe(0)
          expect(stored.info.tokens).toEqual(emptyTokens)
        }
        expect(aggregate.cost).toBe(0)
        expect(aggregate.tokens).toEqual({
          input: 0,
          output: 0,
          reasoning: 0,
          cache: { read: 0, write: 0 },
        })
      }),
    {
      git: true,
      config: {
        ...cfg,
        provider: {
          test: {
            ...cfg.provider.test,
            models: {
              "test-model": {
                ...cfg.provider.test.models["test-model"],
                cost: { input: 1, output: 2 },
              },
            },
          },
        },
      },
    },
  ),
)

it.live(
  "session.processor effect tests settle a reused tool call ID after retrying",
  () =>
    provideTmpdirServer(
      ({ dir, llm }) =>
        Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()
          let executions = 0

          yield* llm.push(reply().pendingTool("lookup", { query: "par" }).reset())
          yield* llm.tool("lookup", { query: "weather" })

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "retry tool")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "retry tool" }],
            tools: {
              lookup: tool({
                description: "Look up information",
                inputSchema: z.object({ query: z.string() }),
                execute: async (input) => {
                  executions += 1
                  return {
                    title: "Weather lookup",
                    output: `result:${input.query}`,
                    metadata: { source: "test" },
                  }
                },
              }),
            },
          })

          const calls = (yield* MessageV2.parts(msg.id)).filter(
            (part): part is SessionV1.ToolPart => part.type === "tool",
          )
          expect(value).toBe("continue")
          expect(yield* llm.calls).toBe(2)
          expect(executions).toBe(1)
          expect(calls).toHaveLength(1)
          expect(calls[0]?.state.status).toBe("completed")
          if (calls[0]?.state.status !== "completed") return
          expect(calls[0].state.input).toEqual({ query: "weather" })
          expect(calls[0].state.output).toBe("result:weather")
        }),
      { config: (url) => providerCfg(url) },
    ),
  30_000,
)

it.live("session.processor effect tests compact on structured context overflow", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()

        yield* llm.error(400, { type: "error", error: { code: "context_length_exceeded" } })

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "compact json")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const value = yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "compact json" }],
          tools: {},
        })

        expect(value).toBe("compact")
        expect(yield* llm.calls).toBe(1)
        expect(handle.message.error).toBeUndefined()
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests complete AI SDK tool calls when native flag is off", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()

        yield* llm.tool("lookup", { query: "weather" })

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "tool")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const value = yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "tool" }],
          tools: {
            lookup: tool({
              description: "Look up information",
              inputSchema: z.object({ query: z.string() }),
              execute: async (input) => ({
                title: "Weather lookup",
                output: `result:${input.query}`,
                metadata: { source: "test" },
              }),
            }),
          },
        })

        const parts = yield* MessageV2.parts(msg.id)
        const call = parts.find((part): part is SessionV1.ToolPart => part.type === "tool")

        expect(value).toBe("continue")
        expect(yield* llm.calls).toBe(1)
        expect(call?.callID).toBe("call_1")
        expect(call?.tool).toBe("lookup")
        expect(call?.state.status).toBe("completed")
        if (call?.state.status !== "completed") return
        expect(call.state.input).toEqual({ query: "weather" })
        expect(call.state.output).toBe("result:weather")
        expect(call.state.title).toBe("Weather lookup")
        expect(call.state.metadata).toEqual({ source: "test" })
        expect(call.state.time.start).toBeDefined()
        expect(call.state.time.end).toBeDefined()
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests admit a duplicate provider tool call ID once per generation", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()
        let executions = 0
        const parameters = Schema.Struct({ query: Schema.String })
        const lookup: Tool.Def<typeof parameters> = {
          id: "lookup",
          description: "Look up information",
          parameters,
          jsonSchema: {
            type: "object",
            properties: { query: { type: "string" } },
            required: ["query"],
            additionalProperties: false,
          },
          execute: (args) =>
            Effect.sync(() => {
              executions += 1
              return {
                title: "Weather lookup",
                output: `result:${args.query}`,
                metadata: { source: "test" },
              }
            }),
        }
        const registry = ToolRegistry.Service.of({
          ids: () => Effect.succeed([lookup.id]),
          all: () => Effect.succeed([lookup]),
          named: () => Effect.die("unused"),
          askTools: () => Effect.die("unused"),
          tools: () => Effect.succeed([lookup]),
        })
        const toolChunk = (index: number) => ({
          id: "chatcmpl-duplicate-tool",
          object: "chat.completion.chunk",
          choices: [
            {
              delta: {
                tool_calls: [
                  {
                    index,
                    id: "same-id",
                    type: "function",
                    function: { name: "lookup", arguments: '{"query":"weather"}' },
                  },
                ],
              },
            },
          ],
        })

        yield* llm.push(
          raw({
            chunks: [
              {
                id: "chatcmpl-duplicate-tool",
                object: "chat.completion.chunk",
                choices: [{ delta: { role: "assistant" } }],
              },
              toolChunk(0),
              toolChunk(1),
              {
                id: "chatcmpl-duplicate-tool",
                object: "chat.completion.chunk",
                choices: [{ delta: {}, finish_reason: "tool_calls" }],
              },
            ],
          }),
        )

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "duplicate tool")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })
        const tools = yield* SessionTools.resolve({
          agent: agent(),
          model: mdl,
          session: chat,
          processor: handle,
          bypassAgentCheck: false,
          messages: yield* session.messages({ sessionID: chat.id }),
          promptOps: {
            cancel: () => Effect.void,
            checkpoint: () => Effect.succeed(0),
            retain: () => Effect.succeed(Option.some(Effect.void)),
            admitIfCurrent: (effect) => effect.pipe(Effect.map(Option.some)),
            admitChild: (_sessionID, effect) => effect.pipe(Effect.map(Option.some)),
            resolvePromptParts: () => Effect.die("unused"),
            prompt: () => Effect.die("unused"),
            admitNotification: () => Effect.die("unused"),
            notify: () => Effect.void,
          },
        }).pipe(
          Effect.provideService(ToolRegistry.Service, registry),
          Effect.provideService(MCP.Service, emptyMcp),
          Effect.provideService(Truncate.Service, passthroughTruncate),
        )

        const value = yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "duplicate tool" }],
          tools,
        })

        const calls = (yield* MessageV2.parts(msg.id)).filter(
          (part): part is SessionV1.ToolPart => part.type === "tool",
        )
        expect(value).toBe("continue")
        expect(yield* llm.calls).toBe(1)
        expect(executions).toBe(1)
        expect(calls).toHaveLength(1)
        expect(calls[0]?.callID).toBe("same-id")
        expect(calls[0]?.state.status).toBe("completed")
        if (calls[0]?.state.status !== "completed") return
        expect(calls[0].state.input).toEqual({ query: "weather" })
        expect(calls[0].state.output).toBe("result:weather")
      }),
    { config: (url) => providerCfg(url) },
  ),
)

itRecovery.live("session.processor effect tests reject a tool callback delayed past cleanup", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const setup = yield* recoverySetup(path.resolve(dir))
        setup.state.mode = "cleanup"

        expect(yield* setup.handle.process(setup.input)).toBe("continue")
        yield* Deferred.await(setup.state.callbackReady)
        const delayed = setup.state.delayed
        if (!delayed) return yield* Effect.die("delayed callback is missing")

        const first = yield* Effect.tryPromise({ try: delayed, catch: (error) => error }).pipe(Effect.exit)
        const second = yield* Effect.tryPromise({ try: delayed, catch: (error) => error }).pipe(Effect.exit)
        const calls = (yield* MessageV2.parts(setup.msg.id)).filter((part) => part.type === "tool")

        expect(Exit.isFailure(first)).toBe(true)
        expect(Exit.isFailure(second)).toBe(true)
        expect(setup.state.providerCalls).toBe(1)
        expect(setup.state.toolCalls).toBe(0)
        expect(calls).toHaveLength(0)
        expect(setup.handle.message.time.completed).toBeDefined()
      }),
    { config: cfg },
  ),
)

itRecovery.live("session.processor effect tests reject a stale tool callback after retry starts", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const setup = yield* recoverySetup(path.resolve(dir))
        setup.state.mode = "retry"
        const run = yield* setup.handle.process(setup.input).pipe(Effect.forkChild)

        yield* Deferred.await(setup.state.callbackReady)
        yield* Deferred.await(setup.state.retryStarted)
        const delayed = setup.state.delayed
        if (!delayed) return yield* Effect.die("delayed callback is missing")

        const stale = yield* Effect.tryPromise({ try: delayed, catch: (error) => error }).pipe(Effect.exit)
        expect(Exit.isFailure(stale)).toBe(true)
        expect(setup.state.providerCalls).toBe(2)
        expect(setup.state.toolCalls).toBe(0)
        expect(run.pollUnsafe()).toBeUndefined()

        yield* Deferred.succeed(setup.state.finishRetry, undefined)
        expect(yield* Fiber.join(run)).toBe("continue")
        const calls = (yield* MessageV2.parts(setup.msg.id)).filter((part) => part.type === "tool")
        expect(calls).toHaveLength(0)
        expect(setup.state.toolCalls).toBe(0)
      }),
    { config: cfg },
  ),
)

itRecovery.live("session.processor effect tests continue after retryable stream failure with a running tool", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const setup = yield* recoverySetup(path.resolve(dir))
        const run = yield* setup.handle.process(setup.input).pipe(Effect.forkChild)

        yield* Deferred.await(setup.state.started)

        expect(setup.state.providerCalls).toBe(1)
        expect(setup.state.toolCalls).toBe(1)
        expect(setup.state.cancellations).toBe(0)
        expect(run.pollUnsafe()).toBeUndefined()

        yield* Deferred.succeed(setup.state.release, undefined)

        const value = yield* Fiber.join(run)
        const call = (yield* MessageV2.parts(setup.msg.id)).find(
          (part): part is SessionV1.ToolPart => part.type === "tool",
        )

        expect(value).toBe("continue")
        expect(setup.state.providerCalls).toBe(1)
        expect(setup.state.toolCalls).toBe(1)
        expect(setup.state.cancellations).toBe(0)
        expect(setup.handle.message.error).toBeUndefined()
        expect(call?.state.status).toBe("completed")
        if (call?.state.status !== "completed") return
        expect(call.state.title).toBe("Lookup complete")
        expect(call.state.output).toBe("held result")
        expect(call.state.metadata).toEqual({ source: "held" })
      }),
    { config: cfg },
  ),
)

itRecovery.live("session.processor effect tests apply completion buffered before running registration", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const setup = yield* recoverySetup(path.resolve(dir))
        yield* setup.handle.completeToolCall("call-1", {
          title: "Fast lookup",
          metadata: { source: "early" },
          output: "fast result",
        })

        const value = yield* setup.handle.process(setup.input)
        const call = (yield* MessageV2.parts(setup.msg.id)).find(
          (part): part is SessionV1.ToolPart => part.type === "tool",
        )

        expect(value).toBe("continue")
        expect(setup.state.providerCalls).toBe(1)
        expect(setup.state.toolCalls).toBe(1)
        expect(call?.state.status).toBe("completed")
        if (call?.state.status === "completed") {
          expect(call.state.title).toBe("Fast lookup")
          expect(call.state.output).toBe("fast result")
          expect(call.state.metadata).toEqual({ source: "early" })
        }

        yield* Deferred.succeed(setup.state.release, undefined)
        yield* Deferred.await(setup.state.finished)
      }),
    { config: cfg },
  ),
)

itRecovery.live("session.processor effect tests apply failure buffered before running registration", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const setup = yield* recoverySetup(path.resolve(dir))
        const fail = setup.handle.failToolCall
        if (!fail) return yield* Effect.die("processor does not support direct tool failure")
        yield* fail("call-1", new Error("fast failure"))

        const value = yield* setup.handle.process(setup.input)
        const call = (yield* MessageV2.parts(setup.msg.id)).find(
          (part): part is SessionV1.ToolPart => part.type === "tool",
        )

        expect(value).toBe("stop")
        expect(setup.state.providerCalls).toBe(1)
        expect(setup.state.toolCalls).toBe(1)
        expect(setup.state.cancellations).toBe(1)
        expect(call?.state.status).toBe("error")
        if (call?.state.status === "error") expect(call.state.error).toBe("fast failure")

        yield* Deferred.succeed(setup.state.release, undefined)
        yield* Deferred.await(setup.state.finished)
      }),
    { config: cfg },
  ),
)

itRecovery.live("session.processor effect tests stop waiting when one local tool fails", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const setup = yield* recoverySetup(path.resolve(dir))
        setup.state.mode = "sibling"
        const run = yield* setup.handle.process(setup.input).pipe(Effect.forkChild)

        yield* Deferred.await(setup.state.started)

        const value = yield* Fiber.join(run)
        const calls = (yield* MessageV2.parts(setup.msg.id)).filter(
          (part): part is SessionV1.ToolPart => part.type === "tool",
        )
        const failed = calls.find((part) => part.callID === "call-fail")
        const hanging = calls.find((part) => part.callID === "call-1")

        expect(value).toBe("stop")
        expect(setup.state.cancellations).toBe(1)
        expect(failed?.state.status).toBe("error")
        if (failed?.state.status === "error") expect(failed.state.error).toBe("sibling failed")
        expect(hanging?.state.status).toBe("error")
        if (hanging?.state.status === "error") expect(hanging.state.metadata?.interrupted).toBe(true)

        yield* Deferred.succeed(setup.state.release, undefined)
        yield* Deferred.await(setup.state.finished)
      }),
    { config: cfg },
  ),
)

itRecovery.live("session.processor effect tests normalize images settled after provider stream failure", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const setup = yield* recoverySetup(path.resolve(dir))
        setup.state.image = true
        const run = yield* setup.handle.process(setup.input).pipe(Effect.forkChild)

        yield* Deferred.await(setup.state.started)
        yield* Deferred.succeed(setup.state.release, undefined)

        expect(yield* Fiber.join(run)).toBe("continue")
        const call = (yield* MessageV2.parts(setup.msg.id)).find(
          (part): part is SessionV1.ToolPart => part.type === "tool",
        )

        expect(setup.state.normalizations).toBe(1)
        expect(call?.state.status).toBe("completed")
        if (call?.state.status !== "completed") return
        expect(call.state.attachments).toHaveLength(1)
        expect(call.state.attachments?.[0]?.mime).toBe("image/webp")
        expect(call.state.attachments?.[0]?.url).toBe("data:image/webp;base64,normalized")
      }),
    { config: cfg },
  ),
)

itRecovery.live("session.processor effect tests preserve prompt tool completion that wins cancellation grace", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const setup = yield* recoverySetup(path.resolve(dir))
        const run = yield* setup.handle.process(setup.input).pipe(Effect.forkChild)

        yield* Deferred.await(setup.state.started)
        const cancelling = yield* Fiber.interrupt(run).pipe(Effect.forkChild)
        yield* Deferred.await(setup.state.cancelled)
        yield* Deferred.succeed(setup.state.release, undefined)
        yield* Fiber.join(cancelling)

        const call = (yield* MessageV2.parts(setup.msg.id)).find(
          (part): part is SessionV1.ToolPart => part.type === "tool",
        )
        expect((yield* Fiber.await(run))._tag).toBe("Failure")
        expect(call?.state.status).toBe("completed")
        if (call?.state.status === "completed") expect(call.state.output).toBe("held result")
      }),
    { config: cfg },
  ),
)

itRecovery.live("session.processor effect tests abort the running tool when recovery is interrupted", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const setup = yield* recoverySetup(path.resolve(dir))
        const run = yield* setup.handle.process(setup.input).pipe(Effect.forkChild)

        yield* Deferred.await(setup.state.started)
        yield* Fiber.interrupt(run)

        const exit = yield* Fiber.await(run)
        const call = (yield* MessageV2.parts(setup.msg.id)).find(
          (part): part is SessionV1.ToolPart => part.type === "tool",
        )

        expect(Exit.isFailure(exit)).toBe(true)
        expect(setup.state.providerCalls).toBe(1)
        expect(setup.state.toolCalls).toBe(1)
        expect(setup.state.cancellations).toBe(1)
        expect(call?.state.status).toBe("error")
        if (call?.state.status !== "error") return
        expect(call.state.error).toBe("Tool execution aborted")
        expect(call.state.metadata?.interrupted).toBe(true)

        yield* Deferred.succeed(setup.state.release, undefined)
        yield* Deferred.await(setup.state.finished)
        yield* Effect.yieldNow
        const after = (yield* MessageV2.parts(setup.msg.id)).find(
          (part): part is SessionV1.ToolPart => part.type === "tool",
        )
        expect(after?.state.status).toBe("error")
        if (after?.state.status === "error") expect(after.state.error).toBe("Tool execution aborted")
      }),
    { config: cfg },
  ),
)

it.live("session.processor effect tests mark pending tools as aborted on cleanup", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const database = yield* Database.Service
        const { processors, session, provider } = yield* boot()

        yield* llm.toolHang("bash", { cmd: "pwd" })

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "tool abort")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const run = yield* handle
          .process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "tool abort" }],
            tools: {},
          })
          .pipe(Effect.forkChild)

        yield* llm.wait(1)
        yield* waitFor(
          MessageV2.parts(msg.id).pipe(
            Effect.map((parts) => parts.find((part): part is SessionV1.ToolPart => part.type === "tool")),
            Effect.provideService(Database.Service, database),
          ),
          "timed out waiting for tool part",
        )
        yield* Fiber.interrupt(run)

        const exit = yield* Fiber.await(run)
        const parts = yield* MessageV2.parts(msg.id)
        const call = parts.find((part): part is SessionV1.ToolPart => part.type === "tool")

        expect(Exit.isFailure(exit)).toBe(true)
        if (Exit.isFailure(exit)) {
          expect(Cause.hasInterruptsOnly(exit.cause)).toBe(true)
        }
        expect(yield* llm.calls).toBe(1)
        expect(call?.state.status).toBe("error")
        if (call?.state.status === "error") {
          expect(call.state.error).toBe("Tool execution aborted")
          expect(call.state.metadata?.interrupted).toBe(true)
          expect(call.state.time.end).toBeDefined()
        }
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests record aborted errors and idle state", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const seen = defer<void>()
        const { processors, session, provider } = yield* boot()
        const events = yield* EventV2Bridge.Service
        const sts = yield* SessionStatus.Service

        yield* llm.hang

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "abort")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const errs: string[] = []
        const off = yield* events.listen((evt) => {
          if (evt.type !== Session.Event.Error.type) return Effect.void
          const data = evt.data as typeof Session.Event.Error.data.Type
          if (data.sessionID !== chat.id || !data.error) return Effect.void
          errs.push(data.error.name)
          seen.resolve()
          return Effect.void
        })
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const run = yield* handle
          .process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "abort" }],
            tools: {},
          })
          .pipe(Effect.forkChild)

        yield* llm.wait(1)
        yield* Fiber.interrupt(run)

        const exit = yield* Fiber.await(run)
        yield* Effect.promise(() => seen.promise)
        const stored = yield* MessageV2.get({ sessionID: chat.id, messageID: msg.id })
        const state = yield* sts.get(chat.id)
        yield* off

        expect(Exit.isFailure(exit)).toBe(true)
        if (Exit.isFailure(exit)) {
          expect(Cause.hasInterruptsOnly(exit.cause)).toBe(true)
        }
        expect(handle.message.error?.name).toBe("MessageAbortedError")
        expect(stored.info.role).toBe("assistant")
        if (stored.info.role === "assistant") {
          expect(stored.info.error?.name).toBe("MessageAbortedError")
        }
        expect(state).toMatchObject({ type: "idle" })
        expect(errs).toContain("MessageAbortedError")
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests mark interruptions aborted without manual abort", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()
        const sts = yield* SessionStatus.Service

        yield* llm.hang

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "interrupt")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const run = yield* handle
          .process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "interrupt" }],
            tools: {},
          })
          .pipe(Effect.forkChild)

        yield* llm.wait(1)
        yield* Fiber.interrupt(run)

        const exit = yield* Fiber.await(run)
        const stored = yield* MessageV2.get({ sessionID: chat.id, messageID: msg.id })
        const state = yield* sts.get(chat.id)

        expect(Exit.isFailure(exit)).toBe(true)
        expect(handle.message.error?.name).toBe("MessageAbortedError")
        expect(stored.info.role).toBe("assistant")
        if (stored.info.role === "assistant") {
          expect(stored.info.error?.name).toBe("MessageAbortedError")
        }
        expect(state).toMatchObject({ type: "idle" })
      }),
    { config: (url) => providerCfg(url) },
  ),
)

itProviderError.live("session.processor effect tests fail provider-executed error results", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()
        const events = yield* EventV2Bridge.Service

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "provider tool error")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const seen: string[] = []
        const off = yield* events.listen((event) => {
          seen.push(event.type)
          return Effect.void
        })
        const handle = yield* processors.create({ assistantMessage: msg, sessionID: chat.id, model: mdl })

        yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "provider tool error" }],
          tools: {},
        })
        yield* off

        const parts = yield* MessageV2.parts(msg.id)
        const call = parts.find((part): part is SessionV1.ToolPart => part.type === "tool")
        expect(call?.state.status).toBe("error")
        if (call?.state.status === "error") expect(call.state.error).toBe("provider boom")
        expect(seen).toContain(MessageV2.Event.PartUpdated.type)
        expect(seen).toContain(MessageV2.Event.Updated.type)
        expect(seen.filter((type) => type.startsWith("session.next."))).toEqual([])
      }),
    { config: cfg },
  ),
)

itFragmentFailure.live("session.processor effect tests retain partial legacy parts without v2 events", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()
        const events = yield* EventV2Bridge.Service

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "provider failure")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const seen: string[] = []
        const off = yield* events.listen((event) => {
          seen.push(event.type)
          return Effect.void
        })
        const handle = yield* processors.create({ assistantMessage: msg, sessionID: chat.id, model: mdl })

        expect(
          yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "provider failure" }],
            tools: {},
          }),
        ).toBe("stop")
        yield* off

        const parts = yield* MessageV2.parts(msg.id)
        expect(parts).toEqual(
          expect.arrayContaining([
            expect.objectContaining({ type: "text", text: "partial" }),
            expect.objectContaining({ type: "reasoning", text: "thinking" }),
          ]),
        )
        expect(seen).toContain(MessageV2.Event.PartUpdated.type)
        expect(seen).toContain(Session.Event.Error.type)
        expect(seen.filter((type) => type.startsWith("session.next."))).toEqual([])
      }),
    { config: cfg },
  ),
)
