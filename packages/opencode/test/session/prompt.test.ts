import { ConfigV1 } from "@opencode-ai/core/v1/config/config"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { Database } from "@opencode-ai/core/database/database"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { SessionProjector } from "@opencode-ai/core/session/projector"
import { eq } from "drizzle-orm"
import { EventV2Bridge } from "@/event-v2-bridge"
import { expect, test } from "bun:test"
import { Cause, Deferred, Duration, Effect, Exit, Fiber, Layer, Option, Schema, Stream } from "effect"
import { LLMEvent, Usage } from "@opencode-ai/llm"
import path from "path"
import { fileURLToPath } from "url"
import { NamedError } from "@opencode-ai/core/util/error"
import { Agent as AgentSvc } from "../../src/agent/agent"
import { BackgroundJob } from "@/background/job"
import { Command } from "../../src/command"
import { Config } from "@/config/config"
import { LSP } from "@/lsp/lsp"
import { MCP } from "../../src/mcp"
import { Permission } from "../../src/permission"
import { Plugin } from "../../src/plugin"
import { Provider as ProviderSvc } from "@/provider/provider"
import { Env } from "../../src/env"
import { Git } from "../../src/git"
import { Image } from "../../src/image/image"

import { Question } from "../../src/question"
import { Todo } from "../../src/session/todo"
import { Session } from "@/session/session"
import { SessionContextEpochTable, SessionInputTable, SessionMessageTable } from "@opencode-ai/core/session/sql"
import { EventTable } from "@opencode-ai/core/event/sql"
import { LLM } from "../../src/session/llm"
import { MessageV2 } from "../../src/session/message-v2"
import { FSUtil } from "@opencode-ai/core/fs-util"
import { SessionCompaction } from "../../src/session/compaction"
import { SessionSummary } from "../../src/session/summary"
import { Instruction } from "../../src/session/instruction"
import { SessionProcessor } from "../../src/session/processor"
import { SessionPrompt } from "../../src/session/prompt"
import { SessionAsk } from "../../src/session/ask"
import { SessionRevert } from "../../src/session/revert"
import { SessionRunState } from "../../src/session/run-state"
import { MessageID, PartID, SessionID } from "../../src/session/schema"
import { SessionStatus } from "../../src/session/status"
import { SessionV2 } from "@opencode-ai/core/session"
import { SessionExecution } from "@opencode-ai/core/session/execution"
import { Skill } from "../../src/skill"
import { SystemPrompt } from "../../src/session/system"
import { Shell } from "@opencode-ai/core/shell"
import { Snapshot } from "../../src/snapshot"
import { ToolRegistry } from "@/tool/registry"
import { Truncate } from "@/tool/truncate"
import { CrossSpawnSpawner } from "@opencode-ai/core/cross-spawn-spawner"
import { Ripgrep } from "@opencode-ai/core/ripgrep"
import { Format } from "../../src/format"
import {
  disposeAllInstancesEffect,
  provideInstanceEffect,
  reloadInstance,
  TestInstance,
  tmpdirScoped,
} from "../fixture/fixture"
import { awaitWithTimeout, pollWithTimeout, testEffect } from "../lib/effect"
import { reply, TestLLMServer } from "../lib/llm-server"
import { RuntimeFlags } from "@/effect/runtime-flags"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { ModelV2 } from "@opencode-ai/core/model"
import { LocationServiceMap, locationServiceMapLayer } from "@opencode-ai/core/location-services"

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

function withSh<A, E, R>(fx: () => Effect.Effect<A, E, R>) {
  return Effect.acquireUseRelease(
    Effect.sync(() => {
      const prev = process.env.SHELL
      process.env.SHELL = "/bin/sh"
      Shell.preferred.reset()
      return prev
    }),
    () => fx(),
    (prev) =>
      Effect.sync(() => {
        if (prev === undefined) delete process.env.SHELL
        else process.env.SHELL = prev
        Shell.preferred.reset()
      }),
  )
}

function toolPart(parts: SessionV1.Part[]) {
  return parts.find((part): part is SessionV1.ToolPart => part.type === "tool")
}

type CompletedToolPart = SessionV1.ToolPart & { state: SessionV1.ToolStateCompleted }

function completedTool(parts: SessionV1.Part[]) {
  const part = toolPart(parts)
  expect(part?.state.status).toBe("completed")
  return part?.state.status === "completed" ? (part as CompletedToolPart) : undefined
}

function makeMcp(instructions: MCP.ServerInstructions[] = [], tools: Record<string, MCP.McpTool> = {}) {
  return Layer.succeed(
    MCP.Service,
    MCP.Service.of({
      status: () => Effect.succeed({}),
      clients: () => Effect.succeed({}),
      instructions: () => Effect.succeed(instructions),
      tools: () => Effect.succeed(tools),
      prompts: () => Effect.succeed({}),
      resources: () => Effect.succeed({}),
      resourceTemplates: () => Effect.succeed({}),
      add: () => Effect.succeed({ status: { status: "disabled" as const } }),
      connect: () => Effect.void,
      disconnect: () => Effect.void,
      getPrompt: () => Effect.succeed(undefined),
      readResource: () => Effect.succeed(undefined),
      startAuth: () => Effect.die("unexpected MCP auth in prompt-effect tests"),
      authenticate: () => Effect.die("unexpected MCP auth in prompt-effect tests"),
      finishAuth: () => Effect.die("unexpected MCP auth in prompt-effect tests"),
      removeAuth: () => Effect.void,
      supportsOAuth: () => Effect.succeed(false),
      hasStoredTokens: () => Effect.succeed(false),
      getAuthStatus: () => Effect.succeed("not_authenticated" as const),
    }),
  )
}

const lsp = Layer.succeed(
  LSP.Service,
  LSP.Service.of({
    init: () => Effect.void,
    status: () => Effect.succeed([]),
    hasClients: () => Effect.succeed(false),
    touchFile: () => Effect.void,
    diagnostics: () => Effect.succeed({}),
    hover: () => Effect.succeed(undefined),
    definition: () => Effect.succeed([]),
    references: () => Effect.succeed([]),
    implementation: () => Effect.succeed([]),
    documentSymbol: () => Effect.succeed([]),
    workspaceSymbol: () => Effect.succeed([]),
    prepareCallHierarchy: () => Effect.succeed([]),
    incomingCalls: () => Effect.succeed([]),
    outgoingCalls: () => Effect.succeed([]),
  }),
)

const processorCreateStarted: Array<() => void> = []
const blockingProcessor = Layer.succeed(
  SessionProcessor.Service,
  SessionProcessor.Service.of({
    create: () => Effect.sync(() => processorCreateStarted.shift()?.()).pipe(Effect.andThen(Effect.never)),
  }),
)

const promptPreparationGates: Array<{
  started: Deferred.Deferred<void>
  release: Deferred.Deferred<void>
}> = []
const blockingPromptPreparation = Layer.succeed(
  Plugin.Service,
  Plugin.Service.of({
    trigger: (name, _input, output) => {
      if (name !== "chat.message") return Effect.succeed(output)
      const gate = promptPreparationGates.shift()
      if (!gate) return Effect.succeed(output)
      return Deferred.succeed(gate.started, undefined).pipe(
        Effect.andThen(Deferred.await(gate.release)),
        Effect.as(output),
      )
    },
    list: () => Effect.succeed([]),
    init: () => Effect.void,
  }),
)

const runtimeFlags = RuntimeFlags.layer({ experimentalEventSystem: true })

const testLLMServerNode = LayerNode.make({ service: TestLLMServer, layer: TestLLMServer.layer, deps: [] })

const promptRoot = LayerNode.group([
  SessionPrompt.node,
  SessionAsk.node,
  Session.node,
  SessionProjector.node,
  MessageV2.node,
  Snapshot.node,
  LLM.node,
  Env.node,
  AgentSvc.node,
  Command.node,
  Permission.node,
  Plugin.node,
  Config.node,
  ProviderSvc.node,
  LSP.node,
  MCP.node,
  FSUtil.node,
  BackgroundJob.node,
  SessionStatus.node,
  SessionRunState.node,
  Database.node,
  EventV2Bridge.node,
  Question.node,
  Todo.node,
  ToolRegistry.node,
  Skill.node,
  Git.node,
  Ripgrep.node,
  Format.node,
  Truncate.node,
  SessionProcessor.node,
  Image.node,
  SessionCompaction.node,
  SessionRevert.node,
  Instruction.node,
  SystemPrompt.node,
  CrossSpawnSpawner.node,
  RuntimeFlags.node,
])

function makePrompt(input?: {
  mcpInstructions?: MCP.ServerInstructions[]
  mcpTools?: Record<string, MCP.McpTool>
  processor?: "blocking" | Layer.Layer<SessionProcessor.Service>
  llm?: Layer.Layer<LLM.Service>
  plugin?: Layer.Layer<Plugin.Service>
}) {
  const replacements = [
    [SessionSummary.node, summary],
    [LSP.node, lsp],
    [MCP.node, makeMcp(input?.mcpInstructions, input?.mcpTools)],
    [RuntimeFlags.node, runtimeFlags],
  ] as const
  const plugin = input?.plugin ? ([Plugin.node, input.plugin] as const) : undefined
  if (input?.processor && input.llm) {
    return LayerNode.compile(promptRoot, [
      ...replacements,
      ...(plugin ? [plugin] : []),
      [SessionProcessor.node, input.processor === "blocking" ? blockingProcessor : input.processor],
      [LLM.node, input.llm],
    ])
  }
  if (input?.processor) {
    return LayerNode.compile(promptRoot, [
      ...replacements,
      ...(plugin ? [plugin] : []),
      [SessionProcessor.node, input.processor === "blocking" ? blockingProcessor : input.processor],
    ])
  }
  if (input?.llm)
    return LayerNode.compile(promptRoot, [...replacements, ...(plugin ? [plugin] : []), [LLM.node, input.llm]])
  if (plugin) return LayerNode.compile(promptRoot, [...replacements, plugin])
  return LayerNode.compile(promptRoot, replacements)
}

function makeHttp(input?: {
  mcpInstructions?: MCP.ServerInstructions[]
  mcpTools?: Record<string, MCP.McpTool>
  processor?: "blocking" | Layer.Layer<SessionProcessor.Service>
  plugin?: Layer.Layer<Plugin.Service>
  llm?: Layer.Layer<LLM.Service>
}) {
  const root = LayerNode.group([promptRoot, testLLMServerNode])
  const replacements = [
    [SessionSummary.node, summary],
    [LSP.node, lsp],
    [MCP.node, makeMcp(input?.mcpInstructions, input?.mcpTools)],
    [RuntimeFlags.node, runtimeFlags],
  ] as const
  const plugin = input?.plugin ? ([Plugin.node, input.plugin] as const) : undefined
  if (input?.processor && input.llm) {
    return LayerNode.compile(root, [
      ...replacements,
      ...(plugin ? [plugin] : []),
      [SessionProcessor.node, input.processor === "blocking" ? blockingProcessor : input.processor],
      [LLM.node, input.llm],
    ])
  }
  if (input?.processor) {
    return LayerNode.compile(root, [
      ...replacements,
      ...(plugin ? [plugin] : []),
      [SessionProcessor.node, input.processor === "blocking" ? blockingProcessor : input.processor],
    ])
  }
  if (plugin) return LayerNode.compile(root, [...replacements, plugin])
  return LayerNode.compile(root, replacements)
}

function makeHttpNoLLMServer(input?: { mcpInstructions?: MCP.ServerInstructions[]; processor?: "blocking" }) {
  return makePrompt(input)
}

const it = testEffect(makeHttp())
const finishOnlyCalls = { value: 0 }
const cachePrefixRequests: LLM.StreamInput[] = []
const askCleanupGates: Array<{
  started: Deferred.Deferred<void>
  finalizing: Deferred.Deferred<void>
  release: Deferred.Deferred<void>
}> = []
const finishOnly = testEffect(
  makePrompt({
    llm: Layer.succeed(
      LLM.Service,
      LLM.Service.of({
        stream: () => {
          finishOnlyCalls.value++
          return Stream.make(
            LLMEvent.finish({
              reason: "stop",
              usage: new Usage({ inputTokens: 1, outputTokens: 0, totalTokens: 1 }),
            }),
          )
        },
      }),
    ),
  }),
)
const cachePrefix = testEffect(
  makePrompt({
    llm: Layer.succeed(
      LLM.Service,
      LLM.Service.of({
        stream: (input) => {
          cachePrefixRequests.push(input)
          return Stream.never
        },
      }),
    ),
  }),
)
const noLLMServer = testEffect(makeHttpNoLLMServer())
const raceNoLLMServer = testEffect(makeHttpNoLLMServer({ processor: "blocking" }))
const promptRemovalRace = testEffect(makePrompt({ plugin: blockingPromptPreparation }))
const askTransform = testEffect(
  makeHttp({
    plugin: Layer.succeed(
      Plugin.Service,
      Plugin.Service.of({
        trigger: (name, _input, output) =>
          Effect.sync(() => {
            if (name !== "experimental.chat.messages.transform") return output
            const messages = (output as unknown as { messages: SessionV1.WithParts[] }).messages
            const question = messages.flatMap((message) => message.parts).find((part) => part.type === "text")
            if (question?.type === "text" && question.text === "untransformed question") {
              question.text = "transformed question"
            }
            return output
          }),
        list: () => Effect.succeed([]),
        init: () => Effect.void,
      }),
    ),
  }),
)
const askSystemTransformCalls: string[] = []
const askHistoryTransform = testEffect(
  makeHttp({
    plugin: Layer.succeed(
      Plugin.Service,
      Plugin.Service.of({
        trigger: ((name, _input, output) =>
          Effect.sync(() => {
            if (name !== "experimental.chat.messages.transform") return output
            if (typeof output !== "object" || output === null || !("messages" in output)) return output
            if (!Array.isArray(output.messages)) return output
            const mode = [
              "length",
              "reorder",
              "removal",
              "add",
              "current-removal",
              "id-mutation",
              "duplicate",
              "wrong-role",
            ].find((value) => JSON.stringify(output.messages).includes(`current ${value} transform question`))
            if (!mode) return output
            if (mode === "length") {
              const normal = output.messages.find((message) =>
                JSON.stringify(message).includes("normal transform marker"),
              )
              if (normal) output.messages.unshift(normal)
            }
            if (mode === "reorder") {
              const index = output.messages.findIndex((message) =>
                JSON.stringify(message).includes("current reorder transform question"),
              )
              const current = index >= 0 ? output.messages.splice(index, 1)[0] : undefined
              if (current) output.messages.unshift(current)
              const stale = output.messages.find((message) =>
                JSON.stringify(message).includes("prior reorder transform question"),
              )
              if (stale && typeof stale === "object" && "info" in stale && typeof stale.info === "object" && stale.info)
                stale.info.system = "stale reordered user settings"
            }
            if (mode === "removal") {
              const index = output.messages.findIndex((message) =>
                JSON.stringify(message).includes("prior removal transform answer"),
              )
              if (index >= 0) output.messages.splice(index, 1)
            }
            if (mode === "add") {
              const current = output.messages.find((message) =>
                JSON.stringify(message).includes("current add transform question"),
              )
              if (current && typeof current === "object" && "info" in current && typeof current.info === "object") {
                const added = structuredClone(current)
                added.info.id = MessageID.ascending()
                added.info.system = "stale added user settings"
                output.messages.push(added)
              }
            }
            if (mode === "current-removal") {
              const index = output.messages.findIndex((message) =>
                JSON.stringify(message).includes("current current-removal transform question"),
              )
              if (index >= 0) output.messages.splice(index, 1)
            }
            if (mode === "id-mutation") {
              const current = output.messages.find((message) =>
                JSON.stringify(message).includes("current id-mutation transform question"),
              )
              if (current && typeof current === "object" && "info" in current && current.info)
                current.info.id = MessageID.ascending()
            }
            if (mode === "duplicate") {
              const current = output.messages.find((message) =>
                JSON.stringify(message).includes("current duplicate transform question"),
              )
              if (current) output.messages.push(structuredClone(current))
            }
            if (mode === "wrong-role") {
              const current = output.messages.find((message) =>
                JSON.stringify(message).includes("current wrong-role transform question"),
              )
              if (current && typeof current === "object" && "info" in current && current.info)
                current.info.role = "assistant"
            }
            return output
          })) as Plugin.Interface["trigger"],
        list: () => Effect.succeed([]),
        init: () => Effect.void,
      }),
    ),
  }),
)
const transformedSystemMarker = "ask transformed system marker 6248"
const askSystemBudget = testEffect(
  makeHttp({
    plugin: Layer.succeed(
      Plugin.Service,
      Plugin.Service.of({
        trigger: ((name, _input, output) =>
          Effect.sync(() => {
            if (name !== "experimental.chat.system.transform") return output
            if (typeof output !== "object" || output === null || !("system" in output) || !Array.isArray(output.system))
              return output
            askSystemTransformCalls.push(name)
            output.system.push(`${transformedSystemMarker} ${"system ".repeat(4_000)}`)
            return output
          })) as Plugin.Interface["trigger"],
        list: () => Effect.succeed([]),
        init: () => Effect.void,
      }),
    ),
  }),
)
const askBoundaryHooks: string[] = []
const askBoundaryPlugin = Layer.succeed(
  Plugin.Service,
  Plugin.Service.of({
    init: () => Effect.void,
    list: () =>
      Effect.succeed([
        {
          tool: {
            custom_forbidden: {
              description: "must not be visible to Ask",
              args: {},
              execute: async () => "forbidden",
            },
          },
        },
      ]),
    trigger: ((name, _input, output) =>
      Effect.sync(() => {
        askBoundaryHooks.push(name)
        return output
      })) as Plugin.Interface["trigger"],
  }),
)
const askBoundary = testEffect(
  makeHttp({
    plugin: askBoundaryPlugin,
    mcpTools: {
      mcp_forbidden: {
        client: {} as MCP.McpTool["client"],
        def: {
          name: "mcp_forbidden",
          description: "must not be visible to Ask",
          inputSchema: { type: "object", properties: {} },
        } as MCP.McpTool["def"],
      },
    },
  }),
)

const askProviderFailure = testEffect(
  makePrompt({
    llm: Layer.succeed(
      LLM.Service,
      LLM.Service.of({
        stream: () =>
          Stream.make(
            LLMEvent.providerError({
              message: JSON.stringify({
                type: "error",
                error: { code: "server_error", message: "midstream unavailable" },
              }),
            }),
          ),
      }),
    ),
  }),
)
const askCleanupRace = testEffect(
  makePrompt({
    llm: Layer.succeed(
      LLM.Service,
      LLM.Service.of({
        stream: () => {
          const gate = askCleanupGates.shift()
          if (!gate) return Stream.never
          return Stream.fromEffect(
            Deferred.succeed(gate.started, undefined).pipe(
              Effect.andThen(Effect.never),
              Effect.ensuring(
                Deferred.succeed(gate.finalizing, undefined).pipe(Effect.andThen(Deferred.await(gate.release))),
              ),
            ),
          )
        },
      }),
    ),
  }),
)
let cancelNormalGate:
  | {
      started: Deferred.Deferred<void>
      finalizing: Deferred.Deferred<void>
      release: Deferred.Deferred<void>
    }
  | undefined
let cancelAskStarted: Deferred.Deferred<void> | undefined

const cancelConcurrency = testEffect(
  makePrompt({
    processor: Layer.succeed(
      SessionProcessor.Service,
      SessionProcessor.Service.of({
        create: () => {
          const gate = cancelNormalGate
          if (!gate) return Effect.die("missing normal cancellation gate")
          return Deferred.succeed(gate.started, undefined).pipe(
            Effect.andThen(Effect.never),
            Effect.ensuring(
              Deferred.succeed(gate.finalizing, undefined).pipe(Effect.andThen(Deferred.await(gate.release))),
            ),
          )
        },
      }),
    ),
    llm: Layer.succeed(
      LLM.Service,
      LLM.Service.of({
        stream: () => {
          const started = cancelAskStarted
          if (!started) return Stream.die("missing Ask cancellation gate")
          return Stream.fromEffect(Deferred.succeed(started, undefined).pipe(Effect.andThen(Effect.never)))
        },
      }),
    ),
  }),
)

const withMcpInstructions = testEffect(
  makeHttp({
    mcpInstructions: [
      {
        name: "guide-server",
        instructions: "Use lookup before mutate.",
        tools: ["guide-server_lookup"],
      },
    ],
  }),
)
const unix = process.platform !== "win32" ? it.instance : it.instance.skip
const unixNoLLMServer = process.platform !== "win32" ? noLLMServer.instance : noLLMServer.instance.skip

// Config that registers a custom "test" provider with a "test-model" model
// so provider model lookup succeeds inside the loop.
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

function askModelBudgetCfg(url: string) {
  const config = providerCfg(url)
  return {
    ...config,
    provider: {
      ...config.provider,
      test: {
        ...config.provider.test,
        models: {
          "test-model": {
            ...config.provider.test.models["test-model"],
            limit: { context: 6_000, output: 1_000 },
          },
        },
      },
    },
  }
}

function askBudgetCfg(url: string) {
  return {
    ...askModelBudgetCfg(url),
    agent: { build: { prompt: `ask effective system ${"system ".repeat(4_000)}` } },
  }
}

function gitlabWorkflowCfg(url: string) {
  return {
    ...cfg,
    provider: {
      ...cfg.provider,
      gitlab: {
        ...cfg.provider.test,
        id: "gitlab",
        models: {
          "duo-workflow-test": {
            ...cfg.provider.test.models["test-model"],
            id: "duo-workflow-test",
            name: "GitLab Workflow Test",
          },
          "workflow-alias": {
            ...cfg.provider.test.models["test-model"],
            id: "duo-workflow-test",
            name: "GitLab Workflow Alias",
          },
        },
        options: { ...cfg.provider.test.options, baseURL: url },
      },
    },
  }
}

const writeText = Effect.fn("test.writeText")(function* (file: string, text: string) {
  const fs = yield* FSUtil.Service
  yield* fs.writeWithDirs(file, text)
})

const writeConfig = Effect.fn("test.writeConfig")(function* (dir: string, config: Partial<ConfigV1.Info>) {
  yield* writeText(
    path.join(dir, "opencode.json"),
    JSON.stringify({ $schema: "https://opencode.ai/config.json", ...config }),
  )
})

const useServerConfig = Effect.fn("test.useServerConfig")(function* (config: (url: string) => Partial<ConfigV1.Info>) {
  const { directory: dir } = yield* TestInstance
  const llm = yield* TestLLMServer
  yield* writeConfig(dir, config(llm.url))
  return { dir, llm }
})

// Wait for a session's runner to enter a busy state. SessionStatus is flipped
// inside Runner.startShell's serialized transition, so cancel can't no-op once
// we observe it.
const waitForBusy = (sessionID: SessionID, duration: Duration.Input = "2 seconds") =>
  pollWithTimeout(
    Effect.gen(function* () {
      const status = yield* SessionStatus.Service
      const s = yield* status.get(sessionID)
      return s.type === "busy" ? (true as const) : undefined
    }),
    `session ${sessionID} never became busy`,
    duration,
  )

const hasBash = Effect.sync(() => Bun.which("bash") !== null)

const deferredAsPromise = <A>(deferred: Deferred.Deferred<A>): PromiseLike<A> => ({
  then: (onfulfilled, onrejected) => {
    Effect.runFork(
      Deferred.await(deferred).pipe(
        Effect.match({
          onFailure: (error) => {
            onrejected?.(error)
          },
          onSuccess: (value) => {
            onfulfilled?.(value)
          },
        }),
      ),
    )
    return deferredAsPromise(deferred) as PromiseLike<never>
  },
})

function defer<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

const succeedVoid = (deferred: Deferred.Deferred<void>) => {
  Effect.runSync(Deferred.succeed(deferred, void 0).pipe(Effect.ignore))
}

function strings(input: unknown): string[] {
  if (typeof input === "string") return [input]
  if (Array.isArray(input)) return input.flatMap(strings)
  if (input && typeof input === "object") return Object.values(input).flatMap(strings)
  return []
}

const user = Effect.fn("test.user")(function* (sessionID: SessionID, text: string) {
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

const seed = Effect.fn("test.seed")(function* (sessionID: SessionID, opts?: { finish?: string }) {
  const session = yield* Session.Service
  const msg = yield* user(sessionID, "hello")
  const assistant: SessionV1.Assistant = {
    id: MessageID.ascending(),
    role: "assistant",
    parentID: msg.id,
    sessionID,
    mode: "build",
    agent: "build",
    cost: 0,
    path: { cwd: "/tmp", root: "/tmp" },
    tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
    modelID: ref.modelID,
    providerID: ref.providerID,
    time: { created: Date.now() },
    ...(opts?.finish ? { finish: opts.finish } : {}),
  }
  yield* session.updateMessage(assistant)
  yield* session.updatePart({
    id: PartID.ascending(),
    messageID: assistant.id,
    sessionID,
    type: "text",
    text: "hi there",
  })
  return { user: msg, assistant }
})

const addSubtask = (
  sessionID: SessionID,
  messageID: MessageID,
  model = ref,
  opts?: { command?: string; background?: boolean },
) =>
  Effect.gen(function* () {
    const session = yield* Session.Service
    yield* session.updatePart({
      id: PartID.ascending(),
      messageID,
      sessionID,
      type: "subtask",
      prompt: "look into the cache key path",
      description: "inspect bug",
      agent: "general",
      model,
      ...opts,
    })
  })

const boot = Effect.fn("test.boot")(function* (input?: { title?: string }) {
  const config = yield* Config.Service
  const prompt = yield* SessionPrompt.Service
  const run = yield* SessionRunState.Service
  const sessions = yield* Session.Service
  yield* config.get()
  const chat = yield* sessions.create(input ?? { title: "Pinned" })
  return { prompt, run, sessions, chat }
})

noLLMServer.instance(
  "idle run-state runners are recreated with the current interrupt handler",
  () =>
    Effect.gen(function* () {
      const { prompt, run, chat } = yield* boot()
      const message = (text: string) =>
        prompt.prompt({
          sessionID: chat.id,
          agent: "build",
          model: ref,
          noReply: true,
          parts: [{ type: "text", text }],
        })
      const first = yield* message("first")
      const second = yield* message("second")

      yield* run.ensureRunning(chat.id, Effect.succeed(first), Effect.succeed(first))
      const ready = yield* Deferred.make<void>()
      const fiber = yield* run
        .ensureRunning(
          chat.id,
          Effect.succeed(second),
          Deferred.succeed(ready, undefined).pipe(Effect.andThen(Effect.never)),
        )
        .pipe(Effect.forkChild)
      yield* Deferred.await(ready)
      yield* run.cancel(chat.id)

      expect((yield* Fiber.join(fiber)).info.id).toBe(second.info.id)
    }),
  3_000,
)

noLLMServer.instance(
  "a wake admitted during finalization is not removed by the idle callback",
  () =>
    Effect.gen(function* () {
      const { prompt, run, chat } = yield* boot()
      const message = (text: string) =>
        prompt.prompt({
          sessionID: chat.id,
          agent: "build",
          model: ref,
          noReply: true,
          parts: [{ type: "text", text }],
        })
      const first = yield* message("first")
      const second = yield* message("second")
      const third = yield* message("third")
      const firstStarted = yield* Deferred.make<void>()
      const finishFirst = yield* Deferred.make<void>()
      const admissionHeld = yield* Deferred.make<void>()
      const releaseAdmission = yield* Deferred.make<void>()
      const secondStarted = yield* Deferred.make<void>()
      const finishSecond = yield* Deferred.make<void>()
      const thirdStarted = yield* Deferred.make<void>()

      yield* Effect.gen(function* () {
        const firstRun = yield* run
          .ensureRunning(
            chat.id,
            Effect.succeed(first),
            Deferred.succeed(firstStarted, undefined).pipe(
              Effect.andThen(Deferred.await(finishFirst)),
              Effect.as(first),
            ),
          )
          .pipe(Effect.forkChild)
        yield* Deferred.await(firstStarted)

        const held = yield* run
          .admit(
            [{ sessionID: chat.id }],
            Deferred.succeed(admissionHeld, undefined).pipe(Effect.andThen(Deferred.await(releaseAdmission))),
          )
          .pipe(Effect.forkChild)
        yield* Deferred.await(admissionHeld)

        const secondRun = yield* run
          .ensureRunning(
            chat.id,
            Effect.succeed(second),
            Deferred.succeed(secondStarted, undefined).pipe(
              Effect.andThen(Deferred.await(finishSecond)),
              Effect.as(second),
            ),
          )
          .pipe(Effect.forkChild)
        yield* Effect.yieldNow
        yield* Deferred.succeed(finishFirst, undefined)
        yield* Deferred.succeed(releaseAdmission, undefined)
        yield* Fiber.join(held)
        yield* Deferred.await(secondStarted).pipe(Effect.timeout("1 second"))

        const thirdRun = yield* run
          .ensureRunning(
            chat.id,
            Effect.succeed(third),
            Deferred.succeed(thirdStarted, undefined).pipe(Effect.as(third)),
          )
          .pipe(Effect.forkChild)
        yield* run.admit([{ sessionID: chat.id }], Effect.void)
        yield* Effect.yieldNow
        expect(yield* Deferred.isDone(thirdStarted)).toBe(false)

        yield* Deferred.succeed(finishSecond, undefined)
        expect((yield* Fiber.join(firstRun)).info.id).toBe(first.info.id)
        expect((yield* Fiber.join(secondRun)).info.id).toBe(second.info.id)
        expect((yield* Fiber.join(thirdRun)).info.id).toBe(third.info.id)
      }).pipe(
        Effect.ensuring(
          Effect.all(
            [
              Deferred.succeed(finishFirst, undefined),
              Deferred.succeed(releaseAdmission, undefined),
              Deferred.succeed(finishSecond, undefined),
            ],
            { discard: true },
          ).pipe(Effect.ignore),
        ),
      )
    }),
  30_000,
)

noLLMServer.instance("run-state checkpoints survive leases and never reuse a generation", () =>
  Effect.gen(function* () {
    const { run, chat } = yield* boot()
    const first = yield* run.checkpoint(chat.id)
    const leaseA = yield* run.retain(chat.id, first)
    const leaseB = yield* run.retain(chat.id, first)
    expect(Option.isSome(leaseA)).toBe(true)
    expect(Option.isSome(leaseB)).toBe(true)
    if (Option.isNone(leaseA) || Option.isNone(leaseB)) return

    yield* leaseA.value
    expect(yield* run.isCurrent(chat.id, first)).toBe(true)
    yield* leaseB.value
    expect(yield* run.isCurrent(chat.id, first)).toBe(false)
    yield* leaseA.value

    const second = yield* run.checkpoint(chat.id)
    expect(second).not.toBe(first)
  }),
)

noLLMServer.instance("session removal tombstones before cancellation and rejects later run-state admission", () =>
  Effect.gen(function* () {
    const { run, sessions, chat } = yield* boot()
    const checkpoint = yield* run.checkpoint(chat.id)
    const lease = yield* run.retain(chat.id, checkpoint)
    const started = yield* Deferred.make<void>()
    const cancelling = yield* Deferred.make<void>()
    const releaseCancellation = yield* Deferred.make<void>()
    expect(Option.isSome(lease)).toBe(true)

    yield* Effect.gen(function* () {
      const active = yield* run
        .ensureRunning(
          chat.id,
          Effect.die("interrupt fallback admitted"),
          Deferred.succeed(started, undefined).pipe(
            Effect.andThen(Effect.never),
            Effect.ensuring(
              Deferred.succeed(cancelling, undefined).pipe(Effect.andThen(Deferred.await(releaseCancellation))),
            ),
          ),
        )
        .pipe(Effect.forkChild)
      yield* Deferred.await(started)
      const removal = yield* sessions.remove(chat.id).pipe(Effect.forkChild)
      yield* Deferred.await(cancelling)

      expect(yield* run.isCurrent(chat.id, checkpoint)).toBe(false)
      if (Option.isSome(lease)) yield* lease.value

      const admitted = { value: false }
      expect(
        Option.isNone(
          yield* run.admit(
            [{ sessionID: chat.id }],
            Effect.sync(() => {
              admitted.value = true
            }),
          ),
        ),
      ).toBe(true)
      expect(admitted.value).toBe(false)

      const checkpointExit = yield* run.checkpoint(chat.id).pipe(Effect.exit)
      expect(Exit.isFailure(checkpointExit)).toBe(true)
      if (Exit.isFailure(checkpointExit))
        expect(Cause.squash(checkpointExit.cause)).toBeInstanceOf(Session.RemovingError)

      const runExit = yield* run
        .ensureRunning(chat.id, Effect.die("interrupt fallback admitted"), Effect.die("work admitted"))
        .pipe(Effect.exit)
      expect(Exit.isFailure(runExit)).toBe(true)
      if (Exit.isFailure(runExit)) expect(Cause.squash(runExit.cause)).toBeInstanceOf(Session.RemovingError)

      yield* Deferred.succeed(releaseCancellation, undefined)
      yield* Fiber.join(removal)
      yield* Fiber.await(active)
    }).pipe(Effect.ensuring(Deferred.succeed(releaseCancellation, undefined).pipe(Effect.ignore)))
  }),
)

promptRemovalRace.instance("normal prompt rejects with RemovingError when removal wins before persistence", () =>
  Effect.gen(function* () {
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const chat = yield* sessions.create({})
    const messageID = MessageID.ascending()
    const started = yield* Deferred.make<void>()
    const release = yield* Deferred.make<void>()
    const gate = { started, release }
    promptPreparationGates.push(gate)

    yield* Effect.gen(function* () {
      const pending = yield* prompt
        .prompt({
          sessionID: chat.id,
          messageID,
          agent: "build",
          model: ref,
          noReply: true,
          parts: [{ type: "text", text: "removed before persistence" }],
        })
        .pipe(Effect.forkChild)
      yield* Deferred.await(started)

      yield* sessions.remove(chat.id)
      expect(Exit.isFailure(yield* sessions.get(chat.id).pipe(Effect.exit))).toBe(true)
      yield* Deferred.succeed(release, undefined)

      const exit = yield* Fiber.await(pending)
      expect(Exit.isFailure(exit)).toBe(true)
      if (Exit.isFailure(exit)) {
        expect(Cause.squash(exit.cause)).toBeInstanceOf(Session.RemovingError)
        expect(Cause.squash(exit.cause)).toMatchObject({ _tag: "SessionRemovingError", sessionID: chat.id })
      }
    }).pipe(
      Effect.ensuring(
        Effect.all(
          [
            Deferred.succeed(release, undefined),
            Effect.sync(() => {
              const index = promptPreparationGates.indexOf(gate)
              if (index >= 0) promptPreparationGates.splice(index, 1)
            }),
          ],
          { discard: true },
        ).pipe(Effect.ignore),
      ),
    )
  }),
)

noLLMServer.instance("run-state removal tombstone clears when the same Instance reloads", () =>
  Effect.gen(function* () {
    const events = yield* EventV2Bridge.Service
    const run = yield* SessionRunState.Service
    const directory = yield* tmpdirScoped()
    const sessionID = SessionID.make("session-run-state-reload")

    yield* events.publish(Session.Event.Removing, { sessionID }).pipe(provideInstanceEffect(directory))
    const removed = yield* run.checkpoint(sessionID).pipe(provideInstanceEffect(directory), Effect.exit)
    expect(Exit.isFailure(removed)).toBe(true)
    if (Exit.isFailure(removed)) expect(Cause.squash(removed.cause)).toBeInstanceOf(Session.RemovingError)

    yield* reloadInstance({ directory })
    expect(yield* run.checkpoint(sessionID).pipe(provideInstanceEffect(directory))).toBe(1)
  }),
)

noLLMServer.instance("run-state removal tombstone clears after Instance disposal", () =>
  Effect.gen(function* () {
    const events = yield* EventV2Bridge.Service
    const run = yield* SessionRunState.Service
    const directory = yield* tmpdirScoped()
    const sessionID = SessionID.make("session-run-state-dispose")

    yield* events.publish(Session.Event.Removing, { sessionID }).pipe(provideInstanceEffect(directory))
    const removed = yield* run.checkpoint(sessionID).pipe(provideInstanceEffect(directory), Effect.exit)
    expect(Exit.isFailure(removed)).toBe(true)
    if (Exit.isFailure(removed)) expect(Cause.squash(removed.cause)).toBeInstanceOf(Session.RemovingError)
    yield* disposeAllInstancesEffect

    expect(yield* run.checkpoint(sessionID).pipe(provideInstanceEffect(directory))).toBe(1)
  }),
)

noLLMServer.instance("run-state admission locks are independent across Instances", () =>
  Effect.gen(function* () {
    const run = yield* SessionRunState.Service
    const first = yield* tmpdirScoped()
    const second = yield* tmpdirScoped()
    const sessionID = SessionID.make("session-run-state-independent")
    const started = yield* Deferred.make<void>()
    const release = yield* Deferred.make<void>()

    yield* Effect.gen(function* () {
      const held = yield* run
        .admit([{ sessionID }], Deferred.succeed(started, undefined).pipe(Effect.andThen(Deferred.await(release))))
        .pipe(provideInstanceEffect(first), Effect.forkChild)
      yield* Deferred.await(started)

      const admitted = yield* awaitWithTimeout(
        run.admit([{ sessionID }], Effect.succeed("independent")).pipe(provideInstanceEffect(second)),
        "second Instance admission was serialized behind the first",
      )
      expect(admitted).toEqual(Option.some("independent"))

      yield* Deferred.succeed(release, undefined)
      yield* Fiber.join(held)
    }).pipe(Effect.ensuring(Deferred.succeed(release, undefined).pipe(Effect.ignore)))
  }),
)

noLLMServer.instance(
  "new subtask parts always persist background mode",
  () =>
    Effect.gen(function* () {
      const { prompt, chat } = yield* boot()
      const create = (background?: boolean) =>
        prompt.prompt({
          sessionID: chat.id,
          agent: "build",
          model: ref,
          noReply: true,
          parts: [
            {
              type: "subtask",
              prompt: "inspect the cache",
              description: "inspect cache",
              agent: "general",
              model: ref,
              ...(background === undefined ? {} : { background }),
            },
          ],
        })

      const implicit = yield* create()
      const foreground = yield* create(false)
      const implicitPart = (yield* MessageV2.parts(implicit.info.id)).find((part) => part.type === "subtask")
      const foregroundPart = (yield* MessageV2.parts(foreground.info.id)).find((part) => part.type === "subtask")

      expect(implicitPart?.type === "subtask" ? implicitPart.background : undefined).toBe(true)
      expect(foregroundPart?.type === "subtask" ? foregroundPart.background : undefined).toBe(true)
    }),
  { config: cfg },
)

// Loop semantics

noLLMServer.instance(
  "loop exits immediately when last assistant has stop finish",
  () =>
    Effect.gen(function* () {
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Pinned" })
      yield* seed(chat.id, { finish: "stop" })

      const result = yield* prompt.loop({ sessionID: chat.id })
      expect(result.info.role).toBe("assistant")
      if (result.info.role === "assistant") expect(result.info.finish).toBe("stop")
    }),
  { config: cfg },
)

noLLMServer.instance(
  "loop exits for a completed parent turn with nonmonotonic message IDs",
  () =>
    Effect.gen(function* () {
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Pinned" })
      const userID = MessageID.make("msg_z_user")
      const assistantID = MessageID.make("msg_a_assistant")
      yield* sessions.updateMessage({
        id: userID,
        role: "user",
        sessionID: chat.id,
        agent: "build",
        model: ref,
        time: { created: 100 },
      })
      yield* sessions.updateMessage({
        id: assistantID,
        role: "assistant",
        parentID: userID,
        sessionID: chat.id,
        mode: "build",
        agent: "build",
        cost: 0,
        path: { cwd: "/tmp", root: "/tmp" },
        tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
        modelID: ref.modelID,
        providerID: ref.providerID,
        time: { created: 200, completed: 201 },
        finish: "stop",
      })

      const result = yield* prompt.loop({ sessionID: chat.id })

      expect(result.info.id).toBe(assistantID)
    }),
  { config: cfg },
)

it.instance("loop exits without an LLM request for interrupted orphan tool calls", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const chat = yield* sessions.create({ title: "Pinned" })
    const seeded = yield* seed(chat.id, { finish: "stop" })
    yield* sessions.updatePart({
      id: PartID.ascending(),
      messageID: seeded.assistant.id,
      sessionID: chat.id,
      type: "tool",
      callID: "interrupted-call",
      tool: "edit",
      state: {
        status: "error",
        input: {},
        error: "Tool execution aborted",
        metadata: { interrupted: true },
        time: { start: 1, end: 2 },
      },
    })

    const result = yield* prompt.loop({ sessionID: chat.id })
    expect(result.info.id).toBe(seeded.assistant.id)
    expect(yield* llm.hits).toHaveLength(0)
  }),
)

it.instance("loop calls LLM and returns assistant message", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const chat = yield* sessions.create({
      title: "Pinned",
      permission: [{ permission: "*", pattern: "*", action: "allow" }],
    })
    yield* prompt.prompt({
      sessionID: chat.id,
      agent: "build",
      noReply: true,
      parts: [{ type: "text", text: "hello" }],
    })
    yield* llm.text("world")

    const result = yield* prompt.loop({ sessionID: chat.id })
    expect(result.info.role).toBe("assistant")
    const parts = result.parts.filter((p) => p.type === "text")
    expect(parts.some((p) => p.type === "text" && p.text === "world")).toBe(true)
    expect(yield* llm.hits).toHaveLength(1)
  }),
)

test("ask input request ID is optional and bounded", () => {
  const input = { sessionID: "ses_test", question: "question" }
  expect(Schema.is(SessionAsk.AskInput)(input)).toBe(true)
  expect(Schema.is(SessionAsk.AskInput)({ ...input, requestID: "client-request" })).toBe(true)
  expect(Schema.is(SessionAsk.AskInput)({ ...input, requestID: "" })).toBe(false)
  expect(Schema.is(SessionAsk.AskInput)({ ...input, requestID: "x".repeat(129) })).toBe(false)
})

it.instance(
  "ask uses session context without persisting the question or answer",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const prompt = yield* SessionPrompt.Service
      const asks = yield* SessionAsk.Service
      const sessions = yield* Session.Service
      const status = yield* SessionStatus.Service
      const chat = yield* sessions.create({ title: "Pinned" })
      yield* seed(chat.id, { finish: "stop" })
      const before = yield* sessions.messages({ sessionID: chat.id })
      const beforeInfo = yield* sessions.get(chat.id)
      yield* llm.text("The answer is 42.")

      const result = yield* prompt.ask({ sessionID: chat.id, question: "What number did we establish?" })
      const after = yield* sessions.messages({ sessionID: chat.id })
      const hit = (yield* llm.hits)[0]

      expect(result).toMatchObject({ text: "The answer is 42." })
      expect(after.map((message) => message.info.id)).toEqual(before.map((message) => message.info.id))
      expect((yield* sessions.get(chat.id)).time.updated).toBe(beforeInfo.time.updated)
      expect((yield* status.get(chat.id)).type).toBe("idle")
      const threads = yield* asks.threads({ sessionID: chat.id })
      expect(threads.items).toHaveLength(1)
      expect((yield* asks.turns({ sessionID: chat.id, threadID: result.threadID })).items).toEqual([
        {
          id: result.id,
          threadID: result.threadID,
          question: "What number did we establish?",
          answer: "The answer is 42.",
          toolActivity: [],
          time: result.time,
        },
      ])
      expect(strings(hit?.body)).toContain("hello")
      expect(strings(hit?.body)).toContain("hi there")
      expect(strings(hit?.body)).toContain("What number did we establish?")
      expect(strings(hit?.body.tools)).toEqual(expect.arrayContaining(["read", "glob", "grep", "webfetch"]))
    }),
  30_000,
)

it.instance(
  "ask correlates concurrent client requests and generates unique defaults",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const asks = yield* SessionAsk.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Ask request correlation" })
      yield* llm.text("first answer")
      yield* llm.text("second answer")

      const explicit = yield* Effect.all(
        [
          asks.ask({ sessionID: chat.id, requestID: "client-first", question: "first question" }),
          asks.ask({ sessionID: chat.id, requestID: "client-second", question: "second question" }),
        ],
        { concurrency: "unbounded" },
      )
      yield* llm.text("third answer")
      yield* llm.text("fourth answer")
      const generated = yield* Effect.all(
        [
          asks.ask({ sessionID: chat.id, question: "third question" }),
          asks.ask({ sessionID: chat.id, question: "fourth question" }),
        ],
        { concurrency: "unbounded" },
      )

      expect(explicit.map((result) => result.requestID)).toEqual(["client-first", "client-second"])
      expect(generated[0].requestID).not.toBe(generated[1].requestID)
      expect(generated.every((result) => Schema.is(SessionAsk.Event.RequestID)(result.requestID))).toBe(true)
      const persisted = yield* Effect.forEach([...explicit, ...generated], (result) =>
        asks.turns({ sessionID: chat.id, threadID: result.threadID }).pipe(Effect.map((page) => page.items[0])),
      )
      expect(persisted.every((turn) => turn && !("requestID" in turn))).toBe(true)
    }),
  30_000,
)

askTransform.instance(
  "ask transforms the ephemeral question before model serialization",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Pinned" })
      yield* llm.text("transformed")

      expect(yield* prompt.ask({ sessionID: chat.id, question: "untransformed question" })).toMatchObject({
        text: "transformed",
      })
      const hit = (yield* llm.hits)[0]
      expect(strings(hit?.body)).toContain("transformed question")
      expect(strings(hit?.body)).not.toContain("untransformed question")
    }),
  30_000,
)

it.instance(
  "Ask runs while the normal runner stays busy and receives the exact isolation instruction",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const prompt = yield* SessionPrompt.Service
      const asks = yield* SessionAsk.Service
      const sessions = yield* Session.Service
      const status = yield* SessionStatus.Service
      const chat = yield* sessions.create({ title: "Busy normal runner" })
      const release = defer<void>()
      yield* llm.hold("normal answer", release.promise)
      const normal = yield* prompt
        .prompt({ sessionID: chat.id, parts: [{ type: "text", text: "normal busy snapshot marker" }] })
        .pipe(Effect.forkChild)
      yield* awaitWithTimeout(llm.wait(1), "normal runner did not reach the provider", "10 seconds")
      expect((yield* status.get(chat.id)).type).toBe("busy")
      yield* llm.text("isolated answer")

      const result = yield* asks.ask({ sessionID: chat.id, question: "isolated busy question" })
      const request = strings((yield* llm.hits)[1]?.body)

      expect(result.text).toBe("isolated answer")
      expect(request.some((item) => item.includes(SessionAsk.IsolationInstruction))).toBe(true)
      expect(request).toContain("normal busy snapshot marker")
      expect((yield* status.get(chat.id)).type).toBe("busy")
      release.resolve()
      yield* Fiber.join(normal)
    }),
  30_000,
)

it.instance(
  "cancel interrupts every active ask for the session",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const prompt = yield* SessionPrompt.Service
      const asks = yield* SessionAsk.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Pinned" })
      yield* llm.hang
      yield* llm.hang

      const first = yield* prompt.ask({ sessionID: chat.id, question: "first" }).pipe(Effect.forkChild)
      const second = yield* prompt.ask({ sessionID: chat.id, question: "second" }).pipe(Effect.forkChild)
      yield* awaitWithTimeout(llm.wait(2), "timed out waiting for active asks", "10 seconds")

      yield* prompt.cancel(chat.id)
      const [firstExit, secondExit] = yield* Effect.all([Fiber.await(first), Fiber.await(second)])

      expect(Exit.isFailure(firstExit) && Cause.hasInterruptsOnly(firstExit.cause)).toBe(true)
      expect(Exit.isFailure(secondExit) && Cause.hasInterruptsOnly(secondExit.cause)).toBe(true)
      expect((yield* asks.threads({ sessionID: chat.id })).items).toEqual([])
    }),
  30_000,
)

cancelConcurrency.instance(
  "cancel interrupts Ask while normal runner finalization is blocked",
  () =>
    Effect.gen(function* () {
      const prompt = yield* SessionPrompt.Service
      const asks = yield* SessionAsk.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Concurrent cancellation" })
      const normal = {
        started: yield* Deferred.make<void>(),
        finalizing: yield* Deferred.make<void>(),
        release: yield* Deferred.make<void>(),
      }
      const askStarted = yield* Deferred.make<void>()
      yield* Effect.addFinalizer(() =>
        Effect.sync(() => {
          cancelNormalGate = undefined
          cancelAskStarted = undefined
        }).pipe(Effect.andThen(Deferred.succeed(normal.release, undefined)), Effect.asVoid),
      )
      cancelNormalGate = normal
      cancelAskStarted = askStarted
      const normalFiber = yield* prompt
        .prompt({ sessionID: chat.id, parts: [{ type: "text", text: "normal work" }] })
        .pipe(Effect.forkChild)
      yield* awaitWithTimeout(Deferred.await(normal.started), "normal runner did not start", "10 seconds")
      const askFiber = yield* asks.ask({ sessionID: chat.id, question: "isolated work" }).pipe(Effect.forkChild)
      yield* awaitWithTimeout(Deferred.await(askStarted), "Ask did not start", "10 seconds")

      const cancelling = yield* prompt.cancel(chat.id).pipe(Effect.forkChild)
      yield* awaitWithTimeout(
        Deferred.await(normal.finalizing),
        "normal runner did not enter finalization",
        "10 seconds",
      )
      const askExit = yield* awaitWithTimeout(Fiber.await(askFiber), "Ask was not cancelled concurrently", "2 seconds")

      expect(Exit.isFailure(askExit) && Cause.hasInterruptsOnly(askExit.cause)).toBe(true)
      expect((yield* asks.threads({ sessionID: chat.id })).items).toEqual([])
      yield* Deferred.succeed(normal.release, undefined)
      yield* awaitWithTimeout(Fiber.join(cancelling), "Session cancel did not finish", "5 seconds")
      yield* awaitWithTimeout(Fiber.interrupt(normalFiber), "normal prompt did not finish", "5 seconds")
    }),
  30_000,
)

askCleanupRace.instance(
  "stale ask cleanup preserves a replacement ask registration",
  () =>
    Effect.gen(function* () {
      const test = yield* TestInstance
      yield* writeConfig(test.directory, cfg)
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Ask replacement cleanup" })
      const first = {
        started: yield* Deferred.make<void>(),
        finalizing: yield* Deferred.make<void>(),
        release: yield* Deferred.make<void>(),
      }
      const second = {
        started: yield* Deferred.make<void>(),
        finalizing: yield* Deferred.make<void>(),
        release: yield* Deferred.make<void>(),
      }
      askCleanupGates.push(first)

      return yield* Effect.gen(function* () {
        const firstAsk = yield* prompt
          .ask({ sessionID: chat.id, agent: "build", model: ref, question: "first" })
          .pipe(Effect.forkChild)
        yield* Deferred.await(first.started).pipe(Effect.timeout("5 seconds"))
        yield* prompt.cancel(chat.id)
        yield* Deferred.await(first.finalizing).pipe(Effect.timeout("5 seconds"))

        askCleanupGates.push(second)
        const secondAsk = yield* prompt
          .ask({ sessionID: chat.id, agent: "build", model: ref, question: "second" })
          .pipe(Effect.forkChild)
        yield* Deferred.await(second.started).pipe(Effect.timeout("5 seconds"))
        yield* Deferred.succeed(first.release, undefined)
        const firstExit = yield* Fiber.await(firstAsk).pipe(Effect.timeout("5 seconds"))
        expect(Exit.isFailure(firstExit) && Cause.hasInterruptsOnly(firstExit.cause)).toBe(true)

        yield* Deferred.succeed(second.release, undefined)
        yield* prompt.cancel(chat.id)
        const secondExit = yield* Fiber.await(secondAsk).pipe(Effect.timeout("5 seconds"))
        expect(Exit.isFailure(secondExit) && Cause.hasInterruptsOnly(secondExit.cause)).toBe(true)
      }).pipe(
        Effect.ensuring(
          Effect.sync(() => askCleanupGates.splice(0)).pipe(
            Effect.andThen(Deferred.succeed(first.release, undefined)),
            Effect.andThen(Deferred.succeed(second.release, undefined)),
            Effect.asVoid,
          ),
        ),
      )
    }),
  30_000,
)

askProviderFailure.instance("ask preserves structured provider stream failures", () =>
  Effect.gen(function* () {
    const test = yield* TestInstance
    yield* writeConfig(test.directory, cfg)
    const prompt = yield* SessionPrompt.Service
    const asks = yield* SessionAsk.Service
    const sessions = yield* Session.Service
    const chat = yield* sessions.create({ title: "Pinned" })

    const exit = yield* prompt
      .ask({ sessionID: chat.id, requestID: "failed-request", question: "fail" })
      .pipe(Effect.exit)

    expect(Exit.isFailure(exit)).toBe(true)
    if (Exit.isSuccess(exit)) return
    const error = Cause.squash(exit.cause)
    expect(SessionV1.APIError.isInstance(error)).toBe(true)
    if (!SessionV1.APIError.isInstance(error)) return
    expect(error.data).toMatchObject({
      message: "midstream unavailable",
      isRetryable: true,
    })
    expect(error.data.responseBody).toContain("server_error")
    const reused = yield* prompt
      .ask({ sessionID: chat.id, requestID: "failed-request", question: "fail again" })
      .pipe(Effect.exit)
    expect(Exit.isFailure(reused) && SessionV1.APIError.isInstance(Cause.squash(reused.cause))).toBe(true)
    expect((yield* asks.threads({ sessionID: chat.id })).items).toEqual([])
  }),
)

it.instance(
  "ask follow-ups use only the selected thread history",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const asks = yield* SessionAsk.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Thread isolation" })
      yield* llm.text("alpha answer")
      const alpha = yield* asks.ask({ sessionID: chat.id, question: "alpha question" })
      yield* llm.text("beta answer")
      yield* asks.ask({ sessionID: chat.id, question: "beta question" })
      yield* llm.text("alpha follow-up answer")

      const followup = yield* asks.ask({
        sessionID: chat.id,
        threadID: alpha.threadID,
        question: "alpha follow-up",
      })
      const hit = (yield* llm.hits)[2]
      const context = strings(hit?.body)

      expect(context).toEqual(expect.arrayContaining(["alpha question", "alpha answer", "alpha follow-up"]))
      expect(context).not.toContain("beta question")
      expect(context).not.toContain("beta answer")
      expect(
        (yield* asks.turns({ sessionID: chat.id, threadID: alpha.threadID })).items.map((turn) => turn.id),
      ).toEqual([alpha.id, followup.id])
    }),
  30_000,
)

it.instance(
  "ask history pruning budgets the effective agent system prompt",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(askBudgetCfg)
      const asks = yield* SessionAsk.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Ask budget" })
      yield* llm.text("prior ask budget answer")
      const first = yield* asks.ask({ sessionID: chat.id, question: "prior ask budget question" })
      yield* llm.text("current answer")

      yield* asks.ask({ sessionID: chat.id, threadID: first.threadID, question: "current budget question" })
      const request = strings((yield* llm.hits)[1]?.body)

      expect(request.some((item) => item.includes("ask effective system"))).toBe(true)
      expect(request).toContain("current budget question")
      expect(request).not.toContain("prior ask budget question")
      expect(request).not.toContain("prior ask budget answer")
    }),
  30_000,
)

askSystemBudget.instance(
  "ask budgets the transformed system prompt once before pruning",
  () =>
    Effect.gen(function* () {
      askSystemTransformCalls.splice(0)
      const { llm } = yield* useServerConfig(askModelBudgetCfg)
      const asks = yield* SessionAsk.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Ask transformed system budget" })
      yield* llm.text("prior transformed budget answer")
      const first = yield* asks.ask({ sessionID: chat.id, question: "prior transformed budget question" })
      yield* llm.text("current answer")

      yield* asks.ask({ sessionID: chat.id, threadID: first.threadID, question: "current transformed question" })
      const request = JSON.stringify((yield* llm.hits)[1]?.body)

      expect(askSystemTransformCalls).toHaveLength(2)
      expect(request.split(transformedSystemMarker)).toHaveLength(2)
      expect(request).toContain("current transformed question")
      expect(request).not.toContain("prior transformed budget question")
      expect(request).not.toContain("prior transformed budget answer")
    }),
  30_000,
)

askHistoryTransform.instance(
  "ask prunes only prior thread turns after length, reorder, removal, and add transforms",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(askBudgetCfg)
      const asks = yield* SessionAsk.Service
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service

      yield* Effect.forEach(["length", "reorder", "removal", "add"] as const, (mode) =>
        Effect.gen(function* () {
          const chat = yield* sessions.create({ title: `Ask ${mode} transform pruning` })
          yield* llm.text(`normal ${mode} answer`)
          yield* prompt.prompt({
            sessionID: chat.id,
            parts: [{ type: "text", text: `normal transform marker ${mode}` }],
          })
          yield* llm.text(`prior ${mode} transform answer`)
          const first = yield* asks.ask({
            sessionID: chat.id,
            question: `prior ${mode} transform question`,
          })
          yield* llm.text(`current ${mode} answer`)
          yield* asks.ask({
            sessionID: chat.id,
            threadID: first.threadID,
            question: `current ${mode} transform question`,
          })
          const request = JSON.stringify((yield* llm.hits).at(-1)?.body)

          expect(request).toContain(`normal transform marker ${mode}`)
          expect(request).toContain(`current ${mode} transform question`)
          expect(request).not.toContain(`prior ${mode} transform question`)
          expect(request).not.toContain(`prior ${mode} transform answer`)
          expect(request).not.toContain("stale reordered user settings")
          expect(request).not.toContain("stale added user settings")
        }),
      )
    }),
  60_000,
)

askHistoryTransform.instance(
  "ask rejects transformed current-message mutation, collision, role changes, and removal",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const asks = yield* SessionAsk.Service
      const sessions = yield* Session.Service
      yield* Effect.forEach(["id-mutation", "duplicate", "wrong-role", "current-removal"] as const, (mode) =>
        Effect.gen(function* () {
          const chat = yield* sessions.create({ title: `Invalid current Ask ${mode}` })
          const exit = yield* asks
            .ask({
              sessionID: chat.id,
              requestID: `invalid-current-${mode}`,
              question: `current ${mode} transform question`,
            })
            .pipe(Effect.exit)

          expect(Exit.isFailure(exit)).toBe(true)
          if (Exit.isFailure(exit))
            expect(Cause.squash(exit.cause)).toMatchObject({
              _tag: "SessionAskCurrentMessageMissingError",
              sessionID: chat.id,
              requestID: `invalid-current-${mode}`,
              messageID: expect.stringMatching(/^msg_/),
            })
          expect((yield* asks.threads({ sessionID: chat.id })).items).toEqual([])
        }),
      )
      expect(yield* llm.hits).toHaveLength(0)
    }),
  30_000,
)

it.instance(
  "later normal prompts exclude all Ask thread data",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const prompt = yield* SessionPrompt.Service
      const asks = yield* SessionAsk.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Ask exclusion" })
      yield* llm.text("ask private answer marker 4831")
      const result = yield* asks.ask({ sessionID: chat.id, question: "ask private question marker 9274" })
      yield* llm.text("normal answer")

      yield* prompt.prompt({ sessionID: chat.id, parts: [{ type: "text", text: "normal request marker" }] })
      const request = JSON.stringify((yield* llm.hits)[1]?.body)

      expect(request).toContain("normal request marker")
      for (const value of [
        "ask private question marker 9274",
        "ask private answer marker 4831",
        result.threadID,
        result.id,
        SessionAsk.IsolationInstruction,
      ]) {
        expect(request).not.toContain(value)
      }
    }),
  30_000,
)

it.instance(
  "Ask writes no normal messages, durable inputs, context epochs, or replay events",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const asks = yield* SessionAsk.Service
      const sessions = yield* Session.Service
      const { db } = yield* Database.Service
      const chat = yield* sessions.create({ title: "Ask durable isolation" })
      const beforeEvents = (yield* db.select().from(EventTable).all().pipe(Effect.orDie)).length
      yield* llm.text("isolated durable answer")

      const result = yield* asks.ask({ sessionID: chat.id, question: "isolated durable question" })

      expect(
        yield* db.select().from(SessionMessageTable).where(eq(SessionMessageTable.session_id, chat.id)).all(),
      ).toEqual([])
      expect(yield* db.select().from(SessionInputTable).where(eq(SessionInputTable.session_id, chat.id)).all()).toEqual(
        [],
      )
      expect(
        yield* db.select().from(SessionContextEpochTable).where(eq(SessionContextEpochTable.session_id, chat.id)).all(),
      ).toEqual([])
      expect((yield* db.select().from(EventTable).all().pipe(Effect.orDie)).length).toBe(beforeEvents)
      expect((yield* asks.turns({ sessionID: chat.id, threadID: result.threadID })).items).toHaveLength(1)
    }),
  30_000,
)

it.instance(
  "Ask thread operations authenticate the owning session",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const asks = yield* SessionAsk.Service
      const sessions = yield* Session.Service
      const owner = yield* sessions.create({ title: "Ask owner" })
      const other = yield* sessions.create({ title: "Ask other" })
      yield* llm.text("owned answer")
      const result = yield* asks.ask({ sessionID: owner.id, question: "owned question" })

      const attempts = yield* Effect.all([
        asks
          .ask({ sessionID: other.id, threadID: result.threadID, question: "unauthorized follow-up" })
          .pipe(Effect.asVoid, Effect.exit),
        asks.turns({ sessionID: other.id, threadID: result.threadID }).pipe(Effect.asVoid, Effect.exit),
        asks.cancel({ sessionID: other.id, threadID: result.threadID }).pipe(Effect.exit),
      ])

      for (const attempt of attempts) {
        expect(Exit.isFailure(attempt)).toBe(true)
        if (Exit.isFailure(attempt)) expect(Cause.squash(attempt.cause)).toBeInstanceOf(SessionAsk.ThreadNotFoundError)
      }
      expect(yield* llm.hits).toHaveLength(1)
      expect((yield* asks.turns({ sessionID: owner.id, threadID: result.threadID })).items).toHaveLength(1)
    }),
  30_000,
)

it.instance(
  "ask thread and turn pagination follows stable cursors",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const asks = yield* SessionAsk.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Ask pagination" })
      yield* llm.text("one")
      const one = yield* asks.ask({ sessionID: chat.id, question: "one" })
      yield* llm.text("one follow-up")
      const followup = yield* asks.ask({ sessionID: chat.id, threadID: one.threadID, question: "continue one" })
      yield* llm.text("two")
      const two = yield* asks.ask({ sessionID: chat.id, question: "two" })

      const threadPage = yield* asks.threads({ sessionID: chat.id, limit: 1 })
      expect(threadPage.items.map((thread) => thread.id)).toEqual([two.threadID])
      expect(threadPage.more).toBe(true)
      const olderThreads = yield* asks.threads({ sessionID: chat.id, limit: 1, before: threadPage.cursor })
      expect(olderThreads.items.map((thread) => thread.id)).toEqual([one.threadID])

      const turnPage = yield* asks.turns({ sessionID: chat.id, threadID: one.threadID, limit: 1 })
      expect(turnPage.items.map((turn) => turn.id)).toEqual([followup.id])
      expect(turnPage.more).toBe(true)
      const olderTurns = yield* asks.turns({
        sessionID: chat.id,
        threadID: one.threadID,
        limit: 1,
        before: turnPage.cursor,
      })
      expect(olderTurns.items.map((turn) => turn.id)).toEqual([one.id])
    }),
  30_000,
)

askBoundary.instance(
  "Ask excludes custom and MCP tools and suppresses plugin tool execution hooks",
  () =>
    Effect.gen(function* () {
      askBoundaryHooks.splice(0)
      const { dir, llm } = yield* useServerConfig(providerCfg)
      const asks = yield* SessionAsk.Service
      const sessions = yield* Session.Service
      const file = path.join(dir, "ask-boundary.txt")
      yield* writeText(file, "allowed read result")
      const chat = yield* sessions.create({
        title: "Ask tool boundary",
        permission: [{ permission: "*", pattern: "*", action: "allow" }],
      })
      yield* llm.tool("read", { filePath: file })
      yield* llm.text("boundary answer")

      yield* asks.ask({ sessionID: chat.id, question: "Use only an allowed source tool" })
      const tools = strings((yield* llm.hits)[0]?.body.tools)

      expect(tools).toContain("read")
      expect(tools).not.toContain("custom_forbidden")
      expect(tools).not.toContain("mcp_forbidden")
      expect(askBoundaryHooks).toContain("experimental.chat.messages.transform")
      expect(askBoundaryHooks).not.toContain("tool.execute.before")
      expect(askBoundaryHooks).not.toContain("tool.execute.after")
    }),
  30_000,
)

it.instance(
  "ask executes allowlisted read tools and sends results to the next provider call",
  () =>
    Effect.gen(function* () {
      const { dir, llm } = yield* useServerConfig(providerCfg)
      const asks = yield* SessionAsk.Service
      const sessions = yield* Session.Service
      const events = yield* EventV2Bridge.Service
      const activity: { status: string; requestID: string; threadID: string; turnID: string }[] = []
      const unsub = yield* events.listen((event) => {
        if (Schema.is(SessionAsk.Event.ToolActivity)(event)) activity.push(event.data)
        return Effect.void
      })
      yield* Effect.addFinalizer(() => unsub)
      const chat = yield* sessions.create({
        title: "Read tool",
        permission: [{ permission: "*", pattern: "*", action: "allow" }],
      })
      const file = path.join(dir, "ask-read.txt")
      yield* writeText(
        file,
        [
          "isolated ask tool result 8472",
          ...Array.from({ length: 220 }, (_, index) => `${index}:${"x".repeat(90)}`),
          "ask full result end 3941",
        ].join("\n"),
      )
      yield* llm.tool("read", { filePath: file })
      yield* llm.text("I read it.")

      const result = yield* asks.ask({
        sessionID: chat.id,
        requestID: "read-request",
        question: "Read the test file",
      })
      const hits = yield* llm.hits
      const available = strings(hits[0]?.body.tools)

      expect(hits).toHaveLength(2)
      expect(available).toEqual(expect.arrayContaining(["read", "glob", "grep", "webfetch"]))
      expect(available).not.toEqual(expect.arrayContaining(["bash", "write", "edit", "task"]))
      expect(strings(hits[1]?.body).some((value) => value.includes("isolated ask tool result 8472"))).toBe(true)
      expect(strings(hits[1]?.body).some((value) => value.includes("ask full result end 3941"))).toBe(true)
      expect(result.toolActivity).toHaveLength(1)
      expect(result.toolActivity[0]).toMatchObject({ tool: "read", status: "completed" })
      expect(result.toolActivity[0]?.output).toContain("isolated ask tool result 8472")
      expect(result.toolActivity[0]?.output).not.toContain("ask full result end 3941")
      expect(new TextEncoder().encode(result.toolActivity[0]?.output).length).toBeLessThanOrEqual(16 * 1024)
      expect(result.requestID).toBe("read-request")
      expect(activity.map((item) => item.status)).toEqual(["running", "completed"])
      expect(activity.map((item) => item.requestID)).toEqual(["read-request", "read-request"])
      expect(activity.every((item) => item.threadID === result.threadID && item.turnID === result.id)).toBe(true)
    }),
  30_000,
)

it.instance(
  "ask reuses one Instruction claim across tool calls",
  () =>
    Effect.gen(function* () {
      const { dir, llm } = yield* useServerConfig(providerCfg)
      const asks = yield* SessionAsk.Service
      const sessions = yield* Session.Service
      const fs = yield* FSUtil.Service
      const nested = path.join(dir, "instruction-claim")
      const marker = "request local instruction claim 7381"
      yield* fs.makeDirectory(nested, { recursive: true })
      yield* writeText(path.join(nested, "AGENTS.md"), marker)
      const file = path.join(nested, "input.txt")
      yield* writeText(file, "claim input")
      yield* llm.tool("read", { filePath: file })
      yield* llm.tool("read", { filePath: file })
      yield* llm.text("done")
      const chat = yield* sessions.create({ title: "Instruction claim" })

      yield* asks.ask({ sessionID: chat.id, question: "Read twice" })
      const request = JSON.stringify((yield* llm.hits).at(-1)?.body)

      expect(request.split(marker)).toHaveLength(2)
    }),
  30_000,
)

it.instance(
  "ask disables tools on the fourth provider call",
  () =>
    Effect.gen(function* () {
      const { dir, llm } = yield* useServerConfig(providerCfg)
      const asks = yield* SessionAsk.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({
        title: "Turn limit",
        permission: [{ permission: "*", pattern: "*", action: "allow" }],
      })
      const file = path.join(dir, "ask-limit.txt")
      yield* writeText(file, "limit")
      yield* llm.tool("read", { filePath: file })
      yield* llm.tool("read", { filePath: file })
      yield* llm.tool("read", { filePath: file })
      yield* llm.text("final answer")

      const result = yield* asks.ask({ sessionID: chat.id, question: "Keep reading" })
      const hits = yield* llm.hits

      expect(hits).toHaveLength(4)
      expect(strings(hits[3]?.body.tools)).not.toContain("read")
      expect(result.text).toBe("final answer")
      expect(result.toolActivity).toHaveLength(3)
    }),
  30_000,
)

it.instance(
  "ask rejects a concurrent turn in the same thread",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const asks = yield* SessionAsk.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Busy thread" })
      yield* llm.text("first answer")
      const first = yield* asks.ask({ sessionID: chat.id, question: "first" })
      yield* llm.hang
      const pending = yield* asks
        .ask({ sessionID: chat.id, threadID: first.threadID, question: "pending" })
        .pipe(Effect.forkChild)
      yield* awaitWithTimeout(llm.wait(2), "timed out waiting for same-thread ask", "10 seconds")

      const duplicate = yield* asks
        .ask({ sessionID: chat.id, threadID: first.threadID, question: "duplicate" })
        .pipe(Effect.exit)
      expect(Exit.isFailure(duplicate)).toBe(true)
      if (Exit.isFailure(duplicate)) expect(Cause.squash(duplicate.cause)).toBeInstanceOf(SessionAsk.ThreadBusyError)

      yield* asks.cancel({ sessionID: chat.id, threadID: first.threadID })
      const exit = yield* Fiber.await(pending)
      expect(Exit.isFailure(exit) && Cause.hasInterruptsOnly(exit.cause)).toBe(true)
      expect((yield* asks.turns({ sessionID: chat.id, threadID: first.threadID })).items).toHaveLength(1)
    }),
  30_000,
)

it.instance(
  "ask rejects duplicate active request IDs and releases them after terminal cleanup",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const asks = yield* SessionAsk.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Duplicate Ask request" })
      const other = yield* sessions.create({ title: "Same request other Session" })
      yield* llm.hang
      yield* llm.hang
      const first = yield* asks
        .ask({ sessionID: chat.id, requestID: "duplicate-request", question: "first" })
        .pipe(Effect.forkChild)
      yield* awaitWithTimeout(llm.wait(1), "timed out waiting for first duplicate Ask", "10 seconds")

      const duplicate = yield* asks
        .ask({ sessionID: chat.id, requestID: "duplicate-request", question: "duplicate" })
        .pipe(Effect.exit)
      expect(Exit.isFailure(duplicate)).toBe(true)
      if (Exit.isFailure(duplicate))
        expect(Cause.squash(duplicate.cause)).toMatchObject({
          _tag: "SessionAskRequestBusyError",
          sessionID: chat.id,
          requestID: "duplicate-request",
        })

      const otherSession = yield* asks
        .ask({ sessionID: other.id, requestID: "duplicate-request", question: "other" })
        .pipe(Effect.forkChild)
      yield* awaitWithTimeout(llm.wait(2), "timed out waiting for cross-Session Ask", "10 seconds")
      yield* Fiber.interrupt(first)
      yield* Fiber.interrupt(otherSession)

      yield* llm.text("reused after cancel")
      const reused = yield* asks.ask({ sessionID: chat.id, requestID: "duplicate-request", question: "reuse" })
      yield* llm.text("reused after success")
      const reusedAgain = yield* asks.ask({
        sessionID: chat.id,
        requestID: "duplicate-request",
        question: "reuse again",
      })
      expect([reused.text, reusedAgain.text]).toEqual(["reused after cancel", "reused after success"])
    }),
  30_000,
)

it.instance(
  "ask permission requests carry Ask metadata without a Session Part pointer",
  () =>
    Effect.gen(function* () {
      const { dir, llm } = yield* useServerConfig(providerCfg)
      const asks = yield* SessionAsk.Service
      const sessions = yield* Session.Service
      const permissions = yield* Permission.Service
      const chat = yield* sessions.create({
        title: "Ask permission",
        permission: [{ permission: "read", pattern: "*", action: "ask" }],
      })
      const file = path.join(dir, "ask-permission.txt")
      yield* writeText(file, "permission result")
      yield* llm.tool("read", { filePath: file })
      yield* llm.text("approved")
      const clientRequestID = "ask-permission-request"
      const pending = yield* asks
        .ask({ sessionID: chat.id, requestID: clientRequestID, question: "Read with approval" })
        .pipe(Effect.forkChild)
      const request = yield* pollWithTimeout(
        permissions.list().pipe(Effect.map((items) => items[0])),
        "Ask permission was not published",
        "10 seconds",
      )

      expect(request.tool).toBeUndefined()
      expect(request.metadata).toMatchObject({
        ask: {
          requestID: clientRequestID,
          threadID: expect.stringMatching(/^ask_/),
          turnID: expect.stringMatching(/^atn_/),
          callID: "call_1",
          tool: "read",
        },
      })
      yield* permissions.reply({ requestID: request.id, reply: "once" })
      expect(yield* Fiber.join(pending)).toMatchObject({ requestID: clientRequestID, text: "approved" })
    }),
  30_000,
)

it.instance(
  "Ask always permission approves repeated matching tool calls",
  () =>
    Effect.gen(function* () {
      const { dir, llm } = yield* useServerConfig(providerCfg)
      const asks = yield* SessionAsk.Service
      const sessions = yield* Session.Service
      const permissions = yield* Permission.Service
      const file = path.join(dir, "ask-always.txt")
      yield* writeText(file, "always permission")
      const chat = yield* sessions.create({
        title: "Ask always permission",
        permission: [{ permission: "read", pattern: "*", action: "ask" }],
      })
      yield* llm.tool("read", { filePath: file })
      yield* llm.tool("read", { filePath: file })
      yield* llm.text("always approved")
      const pending = yield* asks.ask({ sessionID: chat.id, question: "Read twice" }).pipe(Effect.forkChild)
      const request = yield* pollWithTimeout(
        permissions.list().pipe(Effect.map((items) => items[0])),
        "Ask permission was not published",
        "10 seconds",
      )

      yield* permissions.reply({ requestID: request.id, reply: "always" })
      const result = yield* awaitWithTimeout(Fiber.join(pending), "always permission did not resume Ask", "20 seconds")

      expect(result.text).toBe("always approved")
      expect(result.toolActivity.map((item) => item.status)).toEqual(["completed", "completed"])
      expect(yield* permissions.list()).toEqual([])
    }),
  45_000,
)

it.instance(
  "Ask deny permission removes the tool before provider execution",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const asks = yield* SessionAsk.Service
      const sessions = yield* Session.Service
      const permissions = yield* Permission.Service
      const chat = yield* sessions.create({
        title: "Ask deny permission",
        permission: [{ permission: "read", pattern: "*", action: "deny" }],
      })
      yield* llm.text("denied tool answer")

      const result = yield* asks.ask({ sessionID: chat.id, question: "Do not expose read" })

      expect(strings((yield* llm.hits)[0]?.body.tools)).not.toContain("read")
      expect(result.toolActivity).toEqual([])
      expect(yield* permissions.list()).toEqual([])
    }),
  30_000,
)

it.instance(
  "Ask reject permission returns a tool error and continues without stale requests",
  () =>
    Effect.gen(function* () {
      const { dir, llm } = yield* useServerConfig(providerCfg)
      const asks = yield* SessionAsk.Service
      const sessions = yield* Session.Service
      const permissions = yield* Permission.Service
      const file = path.join(dir, "ask-reject.txt")
      yield* writeText(file, "rejected permission")
      const chat = yield* sessions.create({
        title: "Ask reject permission",
        permission: [{ permission: "read", pattern: "*", action: "ask" }],
      })
      yield* llm.tool("read", { filePath: file })
      yield* llm.text("rejection handled")
      const pending = yield* asks.ask({ sessionID: chat.id, question: "Reject this read" }).pipe(Effect.forkChild)
      const request = yield* pollWithTimeout(
        permissions.list().pipe(Effect.map((items) => items[0])),
        "Ask permission was not published",
        "10 seconds",
      )

      yield* permissions.reply({ requestID: request.id, reply: "reject" })
      const result = yield* awaitWithTimeout(Fiber.join(pending), "rejected tool did not continue Ask", "20 seconds")

      expect(result.text).toBe("rejection handled")
      expect(result.toolActivity.map((item) => item.status)).toEqual(["error"])
      expect(yield* permissions.list()).toEqual([])
    }),
  45_000,
)

it.instance(
  "ask rejects GitLab workflow aliases by API identity before provider execution",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(gitlabWorkflowCfg)
      const asks = yield* SessionAsk.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "GitLab workflow" })

      const exit = yield* asks
        .ask({
          sessionID: chat.id,
          question: "unsupported",
          model: {
            providerID: ProviderV2.ID.make("gitlab"),
            modelID: ModelV2.ID.make("workflow-alias"),
          },
        })
        .pipe(Effect.exit)

      expect(Exit.isFailure(exit)).toBe(true)
      if (Exit.isFailure(exit)) expect(Cause.squash(exit.cause)).toBeInstanceOf(SessionAsk.UnsupportedModelError)
      expect(yield* llm.hits).toHaveLength(0)
      expect((yield* asks.threads({ sessionID: chat.id })).items).toEqual([])
    }),
  30_000,
)

it.instance(
  "ask cancellation interrupts a pending tool permission and persists nothing",
  () =>
    Effect.gen(function* () {
      const { dir, llm } = yield* useServerConfig(providerCfg)
      const asks = yield* SessionAsk.Service
      const sessions = yield* Session.Service
      const permissions = yield* Permission.Service
      const chat = yield* sessions.create({
        title: "Cancel Ask permission",
        permission: [{ permission: "read", pattern: "*", action: "ask" }],
      })
      const file = path.join(dir, "ask-cancel-permission.txt")
      yield* writeText(file, "cancel")
      yield* llm.tool("read", { filePath: file })
      yield* llm.text("unused")
      const pending = yield* asks.ask({ sessionID: chat.id, question: "Cancel this read" }).pipe(Effect.forkChild)
      yield* pollWithTimeout(
        permissions.list().pipe(Effect.map((items) => (items.length === 1 ? true : undefined))),
        "Ask permission was not published",
        "20 seconds",
      )

      yield* asks.cancel({ sessionID: chat.id })
      const exit = yield* Fiber.await(pending)

      expect(Exit.isFailure(exit) && Cause.hasInterruptsOnly(exit.cause)).toBe(true)
      expect(yield* permissions.list()).toEqual([])
      expect((yield* asks.threads({ sessionID: chat.id })).items).toEqual([])
    }),
  45_000,
)

it.instance(
  "different Ask threads run concurrently and persist independently",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const asks = yield* SessionAsk.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Concurrent Ask threads" })
      yield* llm.text("one")
      const one = yield* asks.ask({ sessionID: chat.id, question: "thread one" })
      yield* llm.text("two")
      const two = yield* asks.ask({ sessionID: chat.id, question: "thread two" })
      const first = defer<void>()
      const second = defer<void>()
      yield* llm.hold("one follow-up", first.promise)
      yield* llm.hold("two follow-up", second.promise)
      const onePending = yield* asks
        .ask({ sessionID: chat.id, threadID: one.threadID, question: "continue one" })
        .pipe(Effect.forkChild)
      const twoPending = yield* asks
        .ask({ sessionID: chat.id, threadID: two.threadID, question: "continue two" })
        .pipe(Effect.forkChild)
      yield* awaitWithTimeout(llm.wait(4), "Ask threads did not run concurrently", "10 seconds")

      first.resolve()
      second.resolve()
      yield* Effect.all([Fiber.join(onePending), Fiber.join(twoPending)])

      expect((yield* asks.turns({ sessionID: chat.id, threadID: one.threadID })).items).toHaveLength(2)
      expect((yield* asks.turns({ sessionID: chat.id, threadID: two.threadID })).items).toHaveLength(2)
    }),
  30_000,
)

it.instance(
  "requires current inventory and ignores id-like strings from history",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const { prompt, sessions, chat } = yield* boot()
      const seeded = yield* seed(chat.id, { finish: "stop" })
      const foreignPartID = "prt_04c866191001ZOVMkrtXfsS5QE"
      const foreignMessageID = "msg_04c863eb1001JckWhyFN4e592C"
      // Enough candidates to exceed the pressure threshold while exercising the
      // reminder's resistance to untrusted labels and id-like output text.
      for (let index = 0; index < 12; index++)
        yield* sessions.updatePart({
          id: PartID.ascending(),
          messageID: seeded.assistant.id,
          sessionID: chat.id,
          type: "tool",
          callID: `pressure-call-${index}`,
          tool: index === 0 ? "x".repeat(100) + "\nUNTRUSTED_LABEL" : `pressure_tool_${index}`,
          state: {
            status: "completed",
            input: {},
            output:
              index === 0
                ? `${"large settled output ".repeat(2_500)}\nforeign ${foreignPartID} message ${foreignMessageID}`
                : "large settled output ".repeat(2_500),
            title: "done",
            metadata: {},
            time: { start: 1, end: 2 },
          },
        })
      yield* user(chat.id, "continue")
      yield* llm.hang

      const fiber = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
      yield* awaitWithTimeout(llm.wait(1), "timed out waiting for pressure-nudge request", "10 seconds")
      const hit = (yield* llm.hits)[0]
      const reminder = strings(hit?.body).find(
        (value) => value.includes("<system-reminder>") && value.includes("compact_results"),
      )
      expect(reminder).toBeString()
      expect(reminder!.length).toBeLessThan(3_000)
      expect(reminder).not.toContain("UNTRUSTED_LABEL")
      expect(reminder).toContain("calling `list_context` first")
      expect(reminder).toContain("Use only `prt_…` ids returned by that current `list_context` call")
      expect(reminder).not.toContain(foreignPartID)
      expect(reminder).not.toContain(foreignMessageID)
      expect(reminder).not.toMatch(/\n- prt_[A-Za-z0-9]+/)
      const messages = hit?.body.messages
      expect(Array.isArray(messages)).toBe(true)
      expect((messages as { role?: unknown }[]).at(-1)?.role).toBe("user")
      expect(strings((messages as unknown[]).at(-1)).some((value) => value.includes("<system-reminder>"))).toBe(true)
      yield* Fiber.interrupt(fiber)
    }),
  { timeout: 15_000 },
)

cachePrefix.instance("limits caching before grouped receipts and a final-step instruction", () =>
  Effect.gen(function* () {
    cachePrefixRequests.splice(0)
    const test = yield* TestInstance
    yield* writeConfig(test.directory, {
      ...cfg,
      agent: { build: { steps: 1 } },
    })
    const { prompt, sessions, chat } = yield* boot()
    const seeded = yield* seed(chat.id, { finish: "stop" })
    yield* Effect.forEach(
      ["context-call-1", "context-call-2"],
      (callID) =>
        sessions.updatePart({
          id: PartID.ascending(),
          messageID: seeded.assistant.id,
          sessionID: chat.id,
          type: "tool",
          callID,
          tool: "list_context",
          state: {
            status: "completed",
            input: {},
            output: `Context inventory from ${callID}`,
            title: "Context inventory",
            metadata: { total: 1, compactable: 1, charsCompactable: 1_000 },
            time: { start: 1, end: 2 },
          },
        }),
      { discard: true },
    )
    yield* user(chat.id, "inspect context")

    const fiber = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
    yield* pollWithTimeout(
      Effect.sync(() => (cachePrefixRequests.length >= 1 ? true : undefined)),
      "cache-prefix request was not prepared",
    )

    expect(cachePrefixRequests).toHaveLength(1)
    const receipts = (yield* sessions.messages({ sessionID: chat.id }))
      .flatMap((message) => message.parts)
      .filter((part): part is SessionV1.ToolPart => part.type === "tool" && part.tool === "list_context")
    expect(receipts).toHaveLength(2)
    expect(receipts.map((part) => part.state.status)).toEqual(["completed", "completed"])
    expect(receipts.map((part) => (part.state.status === "completed" ? part.state.time.compacted : undefined))).toEqual(
      [undefined, undefined],
    )
    const request = cachePrefixRequests[0]!
    expect(request.cachePrefixLimit).toBeNumber()
    const limit = request.cachePrefixLimit ?? -1
    expect(limit).toBeLessThan(request.messages.length - 1)
    const boundary = request.messages[limit]
    expect(boundary?.role).toBe("assistant")
    const callIDs =
      boundary?.role === "assistant" && Array.isArray(boundary.content)
        ? boundary.content.flatMap((part) => (part.type === "tool-call" ? [part.toolCallId] : []))
        : []
    expect(callIDs).toEqual(["context-call-1", "context-call-2"])
    expect(request.messages.at(-1)?.role).toBe("assistant")
    yield* Fiber.interrupt(fiber)
  }),
)

cachePrefix.instance("limits caching before an unconsumed receipt and pressure reminder", () =>
  Effect.gen(function* () {
    cachePrefixRequests.splice(0)
    const test = yield* TestInstance
    yield* writeConfig(test.directory, cfg)
    const { prompt, sessions, chat } = yield* boot()
    const seeded = yield* seed(chat.id, { finish: "stop" })
    yield* sessions.updatePart({
      id: PartID.ascending(),
      messageID: seeded.assistant.id,
      sessionID: chat.id,
      type: "tool",
      callID: "pressure-source",
      tool: "read",
      state: {
        status: "completed",
        input: {},
        output: "settled output ".repeat(18_000),
        title: "done",
        metadata: {},
        time: { start: 1, end: 2 },
      },
    })
    yield* sessions.updatePart({
      id: PartID.ascending(),
      messageID: seeded.assistant.id,
      sessionID: chat.id,
      type: "tool",
      callID: "unconsumed-context",
      tool: "list_context",
      state: {
        status: "completed",
        input: {},
        output: "Context inventory for interrupted request",
        title: "Context inventory",
        metadata: { total: 1, compactable: 1, charsCompactable: 100_000 },
        time: { start: 1, end: 2 },
      },
    })
    yield* user(chat.id, "continue")
    const fiber = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
    yield* pollWithTimeout(
      Effect.sync(() => (cachePrefixRequests.length >= 1 ? true : undefined)),
      "cache-prefix request was not prepared",
    )

    expect(cachePrefixRequests).toHaveLength(1)
    const request = cachePrefixRequests[0]!
    expect(request.cachePrefixLimit).toBeNumber()
    const limit = request.cachePrefixLimit ?? -1
    expect(limit).toBeLessThan(request.messages.length - 1)
    expect(request.messages.at(-1)?.role).toBe("user")
    const boundary = request.messages[limit]
    const callIDs =
      boundary?.role === "assistant" && Array.isArray(boundary.content)
        ? boundary.content.flatMap((part) => (part.type === "tool-call" ? [part.toolCallId] : []))
        : []
    expect(callIDs).toContain("unconsumed-context")
    yield* Fiber.interrupt(fiber)
  }),
)

it.instance("does not renudge from stale usage after only bounded compaction markers remain", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const { prompt, sessions, chat } = yield* boot()
    const seeded = yield* seed(chat.id, { finish: "stop" })
    seeded.assistant.tokens.input = 65_000
    yield* sessions.updateMessage(seeded.assistant)
    yield* sessions.updatePart({
      id: PartID.ascending(),
      messageID: seeded.assistant.id,
      sessionID: chat.id,
      type: "tool",
      callID: "compacted-pressure-call",
      tool: "bash",
      state: {
        status: "completed",
        input: {},
        output: "large settled output ".repeat(500),
        title: "done",
        metadata: {},
        time: { start: 1, end: 2 },
        compactionGroup: crypto.randomUUID(),
        compactionSummary: "s".repeat(10_000),
      },
    })
    yield* user(chat.id, "continue")
    yield* llm.hang

    const fiber = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
    yield* awaitWithTimeout(llm.wait(1), "timed out waiting for compacted-history request", "10 seconds")
    const values = strings((yield* llm.hits)[0]?.body)
    expect(values.some((value) => value.includes("Context is ~"))).toBe(false)
    const marker = values.find((value) => value.includes("<compacted prt="))
    expect(marker).toContain("[summary truncated]")
    expect(marker!.length).toBeLessThan(4_500)
    yield* Fiber.interrupt(fiber)
  }),
)

it.instance("does not schedule native compaction from stale usage after manual reclamation", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const { prompt, sessions, chat } = yield* boot()
    const seeded = yield* seed(chat.id, { finish: "stop" })
    seeded.assistant.tokens.total = 195_000
    seeded.assistant.tokens.input = 195_000
    yield* sessions.updateMessage(seeded.assistant)
    const stored = (yield* sessions.messages({ sessionID: chat.id })).find(
      (message) => message.info.id === seeded.assistant.id,
    )!
    const text = stored.parts.find((part): part is SessionV1.TextPart => part.type === "text")!
    yield* sessions.updatePart({
      ...text,
      compacted: Date.now(),
      compactionGroup: "manual-reclamation",
      compactionSummary: "the old response is settled",
    })
    yield* user(chat.id, "continue after reclaiming context")
    yield* llm.text("done")
    yield* llm.text("unexpected second request")

    const result = yield* prompt.loop({ sessionID: chat.id })
    const messages = yield* sessions.messages({ sessionID: chat.id })

    expect(yield* llm.calls).toBe(1)
    expect(result.parts.some((part) => part.type === "text" && part.text === "done")).toBe(true)
    expect(messages.flatMap((message) => message.parts).some((part) => part.type === "compaction")).toBe(false)
  }),
)

it.instance("keeps a failed compaction retry attached to its owning message", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const { prompt, sessions, chat } = yield* boot()
    yield* user(chat.id, "ORIGINAL_COMPACTION_OWNER")
    yield* SessionCompaction.use.create({
      sessionID: chat.id,
      agent: "build",
      model: ref,
      auto: false,
    })
    const marker = (yield* sessions.messages({ sessionID: chat.id })).find((message) =>
      message.parts.some((part) => part.type === "compaction"),
    )!
    const next = yield* user(chat.id, "NEXT_USER_PROMPT_MUST_NOT_BE_CONSUMED")
    yield* llm.push(reply().stop())

    yield* prompt.loop({ sessionID: chat.id })

    const messages = yield* sessions.messages({ sessionID: chat.id })
    const summary = messages.find((message) => message.info.role === "assistant" && message.info.summary)
    const request = JSON.stringify((yield* llm.hits)[0]?.body)
    expect(summary?.info.role === "assistant" ? summary.info.parentID : undefined).toBe(marker.info.id)
    expect(summary?.info.role === "assistant" ? summary.info.parentID : undefined).not.toBe(next.id)
    expect(summary?.info.role === "assistant" ? summary.info.error?.name : undefined).toBe("ContextOverflowError")
    expect(request).toContain("ORIGINAL_COMPACTION_OWNER")
    expect(request).not.toContain("NEXT_USER_PROMPT_MUST_NOT_BE_CONSUMED")
  }),
)

finishOnly.instance("terminalizes a finish-only manual compaction without retrying", () =>
  Effect.gen(function* () {
    finishOnlyCalls.value = 0
    const test = yield* TestInstance
    yield* writeConfig(test.directory, cfg)
    const { prompt, sessions, chat } = yield* boot()
    yield* user(chat.id, "compact this")
    yield* SessionCompaction.use.create({
      sessionID: chat.id,
      agent: "build",
      model: ref,
      auto: false,
    })

    const result = yield* prompt.loop({ sessionID: chat.id })
    const summaries = (yield* sessions.messages({ sessionID: chat.id })).filter(
      (message) => message.info.role === "assistant" && message.info.summary,
    )

    expect(finishOnlyCalls.value).toBe(1)
    expect(summaries).toHaveLength(1)
    expect(result.info.role).toBe("assistant")
    if (result.info.role !== "assistant") return
    expect(result.info.finish).toBe("error")
    expect(result.info.error?.name).toBe("ContextOverflowError")
  }),
)

it.instance("attaches impossible-budget preflight errors to the current user request", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig((url) => ({
      ...providerCfg(url),
      compaction: { reserved: 100_000 },
    }))
    const { prompt, sessions, chat } = yield* boot()
    const seeded = yield* seed(chat.id, { finish: "tool-calls" })
    const current = yield* user(chat.id, "CURRENT_USER_REQUEST")

    const result = yield* prompt.loop({ sessionID: chat.id })
    const previous = (yield* sessions.messages({ sessionID: chat.id })).find(
      (message) => message.info.id === seeded.assistant.id,
    )

    expect(yield* llm.calls).toBe(0)
    expect(result.info.role).toBe("assistant")
    if (result.info.role !== "assistant") return
    expect(result.info.parentID).toBe(current.id)
    expect(result.info.error?.name).toBe("ContextOverflowError")
    expect(previous?.info.role === "assistant" ? previous.info.error : undefined).toBeUndefined()
    expect(previous?.info.role === "assistant" ? previous.info.finish : undefined).toBe("tool-calls")
  }),
)

it.instance("nudges from the live request size, not the previous response's usage", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const { prompt, sessions, chat } = yield* boot()
    const seeded = yield* seed(chat.id, { finish: "stop" })
    // Zero recorded provider usage: steering off the last response would see an
    // empty context and stay silent, however much is actually about to be sent.
    seeded.assistant.tokens.input = 0
    seeded.assistant.tokens.output = 0
    yield* sessions.updateMessage(seeded.assistant)
    // Usable budget is context 200k - output 10k = 190k, and the heuristic is
    // 4 chars per token, so ~600k chars sits well past the 65% soft threshold.
    yield* sessions.updatePart({
      id: PartID.ascending(),
      messageID: seeded.assistant.id,
      sessionID: chat.id,
      type: "tool",
      callID: "live-pressure-call",
      tool: "bash",
      state: {
        status: "completed",
        input: {},
        output: "large live output ".repeat(34_000),
        title: "done",
        metadata: {},
        time: { start: 1, end: 2 },
      },
    })
    yield* user(chat.id, "continue")
    yield* llm.hang

    const fiber = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
    yield* awaitWithTimeout(llm.wait(1), "timed out waiting for live-pressure request", "10 seconds")
    const reminder = strings((yield* llm.hits)[0]?.body).find((value) => value.includes("Context is ~"))
    expect(reminder).toContain("compact_results")
    // The ask states the gap down to the target band, not just enough to clear
    // the threshold it tripped.
    expect(reminder).toContain("Aim to reclaim roughly")
    expect(reminder).toContain("down to about 45%")
    yield* Fiber.interrupt(fiber)
  }),
)

it.instance("does not repeat a reminder when no actionable compaction candidate changed", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const { prompt, sessions, chat } = yield* boot()
    const seeded = yield* seed(chat.id, { finish: "stop" })
    yield* sessions.updatePart({
      id: PartID.ascending(),
      messageID: seeded.assistant.id,
      sessionID: chat.id,
      type: "tool",
      callID: "soft-pressure-candidate",
      tool: "read",
      state: {
        status: "completed",
        input: {},
        output: "settled output ".repeat(3_000),
        title: "done",
        metadata: {},
        time: { start: 1, end: 2 },
      },
    })
    yield* user(chat.id, "x".repeat(200_000))
    yield* llm.push(reply().tool("glob", { pattern: "missing-one" }).stop())
    yield* llm.push(reply().tool("glob", { pattern: "missing-two" }).stop())
    yield* llm.push(reply().tool("glob", { pattern: "missing-three" }).stop())
    yield* llm.text("done")

    yield* prompt.loop({ sessionID: chat.id })

    const hits = yield* llm.hits
    const first = strings(hits[0]?.body).find((value) => value.includes("Context is ~"))
    const repeated = strings(hits[3]?.body).find((value) => value.includes("Context is ~"))
    expect(hits).toHaveLength(4)
    expect(first).toContain("Before it gets critical")
    expect(repeated).toBeUndefined()
  }),
)

it.instance("does not feed context-management receipts back into pressure reminders", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const { prompt, sessions, chat } = yield* boot()
    const seeded = yield* seed(chat.id, { finish: "stop" })
    const selected = yield* sessions.updatePart({
      id: PartID.ascending(),
      messageID: seeded.assistant.id,
      sessionID: chat.id,
      type: "tool",
      callID: "feedback-selected",
      tool: "read",
      state: {
        status: "completed",
        input: {},
        output: "selected output ".repeat(5_000),
        title: "done",
        metadata: {},
        time: { start: 1, end: 2 },
      },
    })
    yield* sessions.updatePart({
      id: PartID.ascending(),
      messageID: seeded.assistant.id,
      sessionID: chat.id,
      type: "tool",
      callID: "feedback-retained",
      tool: "read",
      state: {
        status: "completed",
        input: {},
        output: "retained output ".repeat(12_000),
        title: "done",
        metadata: {},
        time: { start: 1, end: 2 },
      },
    })
    yield* user(chat.id, "x".repeat(50_000))
    yield* llm.push(reply().tool("list_context", { compactable_only: true }).stop())
    yield* llm.push(
      reply()
        .tool("compact_results", {
          part_ids: [selected.id],
          summary: "The selected output is no longer needed.",
        })
        .stop(),
    )
    yield* llm.push(reply().tool("glob", { pattern: "missing-feedback-loop-*" }).stop())
    yield* llm.text("done")

    const first = yield* prompt.loop({ sessionID: chat.id })
    yield* sessions.updatePart({
      id: PartID.ascending(),
      messageID: first.info.id,
      sessionID: chat.id,
      type: "tool",
      callID: "feedback-new-candidate",
      tool: "read",
      state: {
        status: "completed",
        input: {},
        output: "new settled output ".repeat(5_000),
        title: "done",
        metadata: {},
        time: { start: 1, end: 2 },
      },
    })
    yield* user(chat.id, "continue with new evidence")
    yield* llm.text("done again")
    yield* prompt.loop({ sessionID: chat.id })

    const hits = yield* llm.hits
    expect(hits).toHaveLength(5)
    const requests = hits.map((hit) => strings(hit.body))
    const reminders = requests.map((values) => values.find((value) => value.includes("Context is ~")))
    expect(reminders[0]).toContain("compact_results")
    expect(reminders.slice(1, 4)).toEqual([undefined, undefined, undefined])
    expect(reminders[4]).toContain("compact_results")

    const fullInventory = requests[1]!.find((value) => value.includes("Context inventory for"))
    expect(fullInventory).toBeString()
    if (!fullInventory) return
    expect(fullInventory).toContain(selected.id)
    const inventorySummary = requests[2]!.find(
      (value) => value.includes("Context-management receipt") && value.includes("(list_context)"),
    )
    expect(inventorySummary).toBeString()
    if (!inventorySummary) return
    expect(requests[2]).not.toContain(fullInventory)
    expect(requests[2]!.some((value) => value.includes("Compacted 1 of 1 parts"))).toBe(true)
    expect(requests[3]).toContain(inventorySummary)
    expect(
      requests[3]!.some((value) => value.includes("Context-management receipt") && value.includes("(compact_results)")),
    ).toBe(true)
    expect(requests[3]!.some((value) => value.includes("Compacted 1 of 1 parts"))).toBe(false)

    const stored = (yield* sessions.messages({ sessionID: chat.id })).flatMap((message) => message.parts)
    const inventoryReceipt = stored.find(
      (part): part is SessionV1.ToolPart => part.type === "tool" && part.tool === "list_context",
    )
    const compactionReceipt = stored.find(
      (part): part is SessionV1.ToolPart => part.type === "tool" && part.tool === "compact_results",
    )
    expect(inventoryReceipt?.state.status === "completed" ? inventoryReceipt.state.output : undefined).toContain(
      "Context inventory for",
    )
    expect(compactionReceipt?.state.status === "completed" ? compactionReceipt.state.output : undefined).toContain(
      "Compacted 1 of 1 parts",
    )
  }),
)

it.instance("remembers reminders across runs and renudges only for new actionable content", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const { prompt, sessions, chat } = yield* boot()
    const seeded = yield* seed(chat.id, { finish: "stop" })
    yield* sessions.updatePart({
      id: PartID.ascending(),
      messageID: seeded.assistant.id,
      sessionID: chat.id,
      type: "tool",
      callID: "persistent-pressure-candidate",
      tool: "read",
      state: {
        status: "completed",
        input: {},
        output: "settled output ".repeat(40_000),
        title: "done",
        metadata: {},
        time: { start: 1, end: 2 },
      },
    })

    yield* user(chat.id, "first request")
    yield* llm.text("first response")
    const first = yield* prompt.loop({ sessionID: chat.id })
    yield* user(chat.id, "second request")
    yield* llm.text("second response")
    yield* prompt.loop({ sessionID: chat.id })
    yield* sessions.updatePart({
      id: PartID.ascending(),
      messageID: first.info.id,
      sessionID: chat.id,
      type: "tool",
      callID: "new-pressure-candidate",
      tool: "read",
      state: {
        status: "completed",
        input: {},
        output: "new settled output ".repeat(500),
        title: "done",
        metadata: {},
        time: { start: 1, end: 2 },
      },
    })
    yield* user(chat.id, "third request")
    yield* llm.text("third response")
    yield* prompt.loop({ sessionID: chat.id })

    const hits = yield* llm.hits
    const reminders = hits.map((hit) => strings(hit.body).find((value) => value.includes("Context is ~")))
    expect(reminders[0]).toContain("compact_results")
    expect(reminders[1]).toBeUndefined()
    expect(reminders[2]).toContain("compact_results")
  }),
)

it.instance("reserves reminder headroom at the soft pressure boundary", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const { prompt, sessions, chat } = yield* boot()
    const seeded = yield* seed(chat.id, { finish: "stop" })
    yield* sessions.updatePart({
      id: PartID.ascending(),
      messageID: seeded.assistant.id,
      sessionID: chat.id,
      type: "tool",
      callID: "boundary-candidate",
      tool: "read",
      state: {
        status: "completed",
        input: {},
        output: "settled output ".repeat(1_000),
        title: "done",
        metadata: {},
        time: { start: 1, end: 2 },
      },
    })
    yield* user(chat.id, "x".repeat(178_500))
    yield* llm.hang

    const fiber = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
    yield* awaitWithTimeout(llm.wait(1), "timed out waiting for boundary request", "10 seconds")
    const reminder = strings((yield* llm.hits)[0]?.body).find((value) => value.includes("Context is ~"))
    expect(reminder).toContain("compact_results")
    expect(reminder).toContain("Context is ~65%")
    yield* Fiber.interrupt(fiber)
  }),
)

// Builds a session whose only reclaimable content is notes that were already
// folded once, with the live request sized to land at a chosen pressure level.
function alreadyFolded(request: number) {
  return Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const { prompt, sessions, chat } = yield* boot()
    const seeded = yield* seed(chat.id, { finish: "stop" })
    for (let index = 0; index < 3; index++)
      yield* sessions.updatePart({
        id: PartID.ascending(),
        messageID: seeded.assistant.id,
        sessionID: chat.id,
        type: "tool",
        callID: `folded-call-${index}`,
        tool: `folded_tool_${index}`,
        state: {
          status: "completed",
          input: {},
          output: "original output ".repeat(1_000),
          title: "done",
          metadata: {},
          time: { start: 1, end: 2 },
          compactionGroup: `folded-${index}`,
          compactionGeneration: 1,
          compactionSummary: "settled note ".repeat(300),
        },
      })
    // Pressure comes from the live request itself, which is never a compaction
    // candidate, so the only reclaimable content left is the notes.
    yield* user(chat.id, "x".repeat(request))
    yield* llm.hang
    const fiber = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
    yield* awaitWithTimeout(llm.wait(1), "timed out waiting for fold request", "10 seconds")
    return { fiber, reminder: strings((yield* llm.hits)[0]?.body).find((value) => value.includes("Context is ~")) }
  })
}

it.instance("does not advertise already-folded notes that cannot guarantee savings", () =>
  Effect.gen(function* () {
    // Sized past the hard threshold, where native compaction is imminent.
    const { fiber, reminder } = yield* alreadyFolded(680_000)
    expect(reminder).toBeUndefined()
    yield* Fiber.interrupt(fiber)
  }),
)

it.instance("does not offer to refold notes below critical pressure", () =>
  Effect.gen(function* () {
    // Refolding erodes detail permanently in exchange for very little space,
    // so it is offered only when native whole-history compaction is the
    // alternative — not merely because the context is getting full.
    // Sized into the soft band: pressure is real, but not yet critical.
    const { fiber, reminder } = yield* alreadyFolded(240_000)
    expect(reminder).toBeUndefined()
    yield* Fiber.interrupt(fiber)
  }),
)

it.instance("suppresses context-pressure nudges on the agent's last step", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig((url) => ({
      ...providerCfg(url),
      agent: { build: { steps: 1 } },
    }))
    const { prompt, sessions, chat } = yield* boot()
    const seeded = yield* seed(chat.id, { finish: "stop" })
    seeded.assistant.tokens.input = 65_000
    yield* sessions.updateMessage(seeded.assistant)
    yield* sessions.updatePart({
      id: PartID.ascending(),
      messageID: seeded.assistant.id,
      sessionID: chat.id,
      type: "tool",
      callID: "last-step-pressure-call",
      tool: "bash",
      state: {
        status: "completed",
        input: {},
        output: "large settled output ".repeat(500),
        title: "done",
        metadata: {},
        time: { start: 1, end: 2 },
      },
    })
    yield* user(chat.id, "continue")
    yield* llm.hang

    const fiber = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
    yield* awaitWithTimeout(llm.wait(1), "timed out waiting for last-step request", "10 seconds")
    expect(strings((yield* llm.hits)[0]?.body).some((value) => value.includes("Context is ~"))).toBe(false)
    yield* Fiber.interrupt(fiber)
  }),
)

it.instance("suppresses context-pressure nudges when compact_results is denied", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const { prompt, sessions, chat } = yield* boot()
    const seeded = yield* seed(chat.id, { finish: "stop" })
    seeded.assistant.tokens.input = 65_000
    yield* sessions.updateMessage(seeded.assistant)
    yield* sessions.updatePart({
      id: PartID.ascending(),
      messageID: seeded.assistant.id,
      sessionID: chat.id,
      type: "tool",
      callID: "denied-pressure-call",
      tool: "bash",
      state: {
        status: "completed",
        input: {},
        output: "large settled output ".repeat(500),
        title: "done",
        metadata: {},
        time: { start: 1, end: 2 },
      },
    })
    yield* sessions.setPermission({
      sessionID: chat.id,
      permission: [{ permission: "compact_results", pattern: "*", action: "deny" }],
    })
    yield* user(chat.id, "continue")
    yield* llm.hang

    const fiber = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
    yield* awaitWithTimeout(llm.wait(1), "timed out waiting for denied-pressure request", "10 seconds")
    const values = strings((yield* llm.hits)[0]?.body)
    expect(values.some((value) => value.includes("Context is ~"))).toBe(false)
    expect(values.some((value) => value.includes("large settled output"))).toBe(true)
    yield* Fiber.interrupt(fiber)
  }),
)

withMcpInstructions.instance(
  "loop includes MCP instructions in model system context",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({
        title: "Pinned",
        permission: [{ permission: "*", pattern: "*", action: "allow" }],
      })
      yield* llm.hang
      yield* user(chat.id, "hello")

      const fiber = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
      yield* awaitWithTimeout(llm.wait(1), "timed out waiting for MCP instruction request", "10 seconds")

      const hits = yield* llm.hits
      const body = JSON.stringify(hits[0]?.body)
      expect(body).toContain('<server name=\\"guide-server\\">')
      expect(body).toContain("Use lookup before mutate.")
      yield* Fiber.interrupt(fiber)
    }),
  15_000,
)

it.instance("legacy prompt emits message events without session.next events", () =>
  Effect.gen(function* () {
    const events = yield* EventV2Bridge.Service
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const chat = yield* sessions.create({
      title: "Pinned",
      agent: "plan",
      model: { providerID: ProviderV2.ID.make("old"), id: ModelV2.ID.make("old-model") },
    })
    const seen: string[] = []
    const off = yield* events.listen((event) => {
      seen.push(event.type)
      return Effect.void
    })

    const first = yield* prompt.prompt({
      sessionID: chat.id,
      agent: "build",
      model: ref,
      noReply: true,
      parts: [{ type: "text", text: "hello" }],
    })
    const second = yield* prompt.prompt({
      sessionID: chat.id,
      agent: "build",
      noReply: true,
      parts: [{ type: "text", text: "again" }],
    })
    yield* off

    expect(first.info.role).toBe("user")
    expect(second.info.role).toBe("user")
    if (first.info.role === "user" && second.info.role === "user") {
      expect(first.info.model).toEqual(ref)
      expect(second.info.model).toEqual(ref)
    }
    expect(yield* sessions.get(chat.id)).toMatchObject({
      agent: "build",
      model: { providerID: ref.providerID, id: ref.modelID },
    })
    expect(seen).toContain(Session.Event.Updated.type)
    expect(seen).toContain(MessageV2.Event.Updated.type)
    expect(seen).toContain(MessageV2.Event.PartUpdated.type)
    expect(seen.filter((type) => type.startsWith("session.next."))).toEqual([])
  }),
)

it.instance("loop surfaces content-filter finishes as session errors", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const events = yield* EventV2Bridge.Service
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const chat = yield* sessions.create({ title: "Pinned" })
    const errors: NonNullable<SessionV1.Assistant["error"]>[] = []
    const expected = {
      name: "ContentFilterError",
      data: { message: "The response was blocked by the provider's content filter" },
    } satisfies NonNullable<SessionV1.Assistant["error"]>
    const off = yield* events.listen((event) => {
      if (event.type !== Session.Event.Error.type) return Effect.void
      const data = event.data as typeof Session.Event.Error.data.Type
      if (data.sessionID === chat.id && data.error) errors.push(data.error)
      return Effect.void
    })

    yield* prompt.prompt({
      sessionID: chat.id,
      agent: "build",
      noReply: true,
      parts: [{ type: "text", text: "hello" }],
    })
    yield* llm.push(reply().text("partial response").contentFilter())

    const result = yield* prompt.loop({ sessionID: chat.id })
    const stored = yield* MessageV2.get({ sessionID: chat.id, messageID: result.info.id })
    yield* off

    expect(yield* llm.hits).toHaveLength(1)
    expect(result.info.role).toBe("assistant")
    expect(stored.info.role).toBe("assistant")
    if (result.info.role === "assistant" && stored.info.role === "assistant") {
      expect(result.info.finish).toBe("content-filter")
      expect(result.info.error).toEqual(expected)
      expect(stored.info.error).toEqual(result.info.error)
      expect(errors).toContainEqual(expected)
    }
    expect(result.parts).toEqual(
      expect.arrayContaining([expect.objectContaining({ type: "text", text: "partial response" })]),
    )
  }),
)

it.instance("loop stops provider overflow instead of auto-compacting when disabled", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig((url) => ({
      ...providerCfg(url),
      compaction: { auto: false },
    }))
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const chat = yield* sessions.create({ title: "Pinned" })

    yield* llm.error(413, { error: { message: "request entity too large" } })
    yield* prompt.prompt({
      sessionID: chat.id,
      agent: "build",
      noReply: true,
      parts: [{ type: "text", text: "hello" }],
    })

    const result = yield* prompt.loop({ sessionID: chat.id })
    const messages = yield* sessions.messages({ sessionID: chat.id })

    expect(result.info.role).toBe("assistant")
    if (result.info.role === "assistant") {
      expect(result.info.error?.name).toBe("ContextOverflowError")
      expect(result.info.finish).toBe("error")
    }
    expect(messages.some((message) => message.parts.some((part) => part.type === "compaction"))).toBe(false)
  }),
)

noLLMServer.instance.skip(
  "prompt emits v2 prompted and synthetic events (v2 projector disabled)",
  () =>
    Effect.gen(function* () {
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Pinned" })

      yield* prompt.prompt({
        sessionID: chat.id,
        agent: "build",
        noReply: true,
        parts: [
          { type: "text", text: "hello v2" },
          {
            type: "file",
            mime: "text/plain",
            filename: "note.txt",
            url: "data:text/plain;base64,bm90ZSBjb250ZW50",
          },
        ],
      })

      const messages = yield* SessionV2.Service.use((session) => session.messages({ sessionID: chat.id })).pipe(
        Effect.provide(
          LayerNode.compile(SessionV2.node, [
            [SessionExecution.node, SessionExecution.noopLayer],
            [LocationServiceMap.node, locationServiceMapLayer],
          ]),
        ),
      )
      const { db } = yield* Database.Service
      const row = yield* db
        .select()
        .from(SessionMessageTable)
        .where(eq(SessionMessageTable.session_id, chat.id))
        .get()
        .pipe(Effect.orDie)
      expect(messages.find((message) => message.type === "user")).toMatchObject({ type: "user", text: "hello v2" })
      expect(typeof row?.data.time.created).toBe("number")
      expect(messages).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ type: "synthetic", text: expect.stringContaining("Called the Read tool") }),
          expect.objectContaining({ type: "synthetic", text: "note content" }),
        ]),
      )
    }),
  { config: cfg },
)

it.instance("static loop returns assistant text through local provider", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const session = yield* sessions.create({
      title: "Prompt provider",
      permission: [{ permission: "*", pattern: "*", action: "allow" }],
    })

    yield* prompt.prompt({
      sessionID: session.id,
      agent: "build",
      noReply: true,
      parts: [{ type: "text", text: "hello" }],
    })

    yield* llm.text("world")

    const result = yield* prompt.loop({ sessionID: session.id })
    expect(result.info.role).toBe("assistant")
    expect(result.parts.some((part) => part.type === "text" && part.text === "world")).toBe(true)
    expect(yield* llm.hits).toHaveLength(1)
    expect(yield* llm.pending).toBe(0)
  }),
)

it.instance("static loop consumes queued replies across turns", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const session = yield* sessions.create({
      title: "Prompt provider turns",
      permission: [{ permission: "*", pattern: "*", action: "allow" }],
    })

    yield* prompt.prompt({
      sessionID: session.id,
      agent: "build",
      noReply: true,
      parts: [{ type: "text", text: "hello one" }],
    })

    yield* llm.text("world one")

    const first = yield* prompt.loop({ sessionID: session.id })
    expect(first.info.role).toBe("assistant")
    expect(first.parts.some((part) => part.type === "text" && part.text === "world one")).toBe(true)

    yield* prompt.prompt({
      sessionID: session.id,
      agent: "build",
      noReply: true,
      parts: [{ type: "text", text: "hello two" }],
    })

    yield* llm.text("world two")

    const second = yield* prompt.loop({ sessionID: session.id })
    expect(second.info.role).toBe("assistant")
    expect(second.parts.some((part) => part.type === "text" && part.text === "world two")).toBe(true)

    expect(yield* llm.hits).toHaveLength(2)
    expect(yield* llm.pending).toBe(0)
  }),
)

it.instance("loop continues when finish is tool-calls", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const session = yield* sessions.create({
      title: "Pinned",
      permission: [{ permission: "*", pattern: "*", action: "allow" }],
    })
    yield* prompt.prompt({
      sessionID: session.id,
      agent: "build",
      noReply: true,
      parts: [{ type: "text", text: "hello" }],
    })
    yield* llm.tool("first", { value: "first" })
    yield* llm.text("second")

    const result = yield* prompt.loop({ sessionID: session.id })
    expect(yield* llm.calls).toBe(2)
    expect(result.info.role).toBe("assistant")
    if (result.info.role === "assistant") {
      expect(result.parts.some((part) => part.type === "text" && part.text === "second")).toBe(true)
      expect(result.info.finish).toBe("stop")
    }
  }),
)

it.instance("loop continues when finish is unknown", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const session = yield* sessions.create({
      title: "Pinned",
      permission: [{ permission: "*", pattern: "*", action: "allow" }],
    })
    yield* prompt.prompt({
      sessionID: session.id,
      agent: "build",
      noReply: true,
      parts: [{ type: "text", text: "hello" }],
    })
    yield* llm.push(reply())
    yield* llm.text("second")

    const result = yield* prompt.loop({ sessionID: session.id })
    expect(yield* llm.calls).toBe(2)
    expect(result.info.role).toBe("assistant")
    if (result.info.role === "assistant") {
      expect(result.parts.some((part) => part.type === "text" && part.text === "second")).toBe(true)
      expect(result.info.finish).toBe("stop")
    }
  }),
)

it.instance("glob tool keeps instance context during prompt runs", () =>
  Effect.gen(function* () {
    const { dir, llm } = yield* useServerConfig(providerCfg)
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const session = yield* sessions.create({
      title: "Glob context",
      permission: [{ permission: "*", pattern: "*", action: "allow" }],
    })
    const file = path.join(dir, "probe.txt")
    yield* writeText(file, "probe")

    yield* prompt.prompt({
      sessionID: session.id,
      agent: "build",
      noReply: true,
      parts: [{ type: "text", text: "find text files" }],
    })
    yield* llm.tool("glob", { pattern: "**/*.txt" })
    yield* llm.text("done")

    const result = yield* prompt.loop({ sessionID: session.id })
    expect(result.info.role).toBe("assistant")

    const msgs = yield* MessageV2.filterCompactedEffect(session.id)
    const tool = msgs
      .flatMap((msg) => msg.parts)
      .find(
        (part): part is CompletedToolPart =>
          part.type === "tool" && part.tool === "glob" && part.state.status === "completed",
      )
    if (!tool) return

    expect(tool.state.output).toContain(file)
    expect(tool.state.output).not.toContain("No context found for instance")
    expect(result.parts.some((part) => part.type === "text" && part.text === "done")).toBe(true)
  }),
)

it.instance("loop continues when finish is stop but assistant has tool parts", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const session = yield* sessions.create({
      title: "Pinned",
      permission: [{ permission: "*", pattern: "*", action: "allow" }],
    })
    yield* prompt.prompt({
      sessionID: session.id,
      agent: "build",
      noReply: true,
      parts: [{ type: "text", text: "hello" }],
    })
    yield* llm.push(reply().tool("first", { value: "first" }).stop())
    yield* llm.text("second")

    const result = yield* prompt.loop({ sessionID: session.id })
    expect(yield* llm.calls).toBe(2)
    expect(result.info.role).toBe("assistant")
    if (result.info.role === "assistant") {
      expect(result.parts.some((part) => part.type === "text" && part.text === "second")).toBe(true)
      expect(result.info.finish).toBe("stop")
    }
  }),
)

it.instance("subtask child inherits parent session external_directory allow", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const chat = yield* sessions.create({
      title: "Parent",
      permission: [{ permission: "external_directory", pattern: "/tmp/allowed/*", action: "allow" }],
    })
    yield* llm.text("done")
    const msg = yield* user(chat.id, "hello")
    yield* addSubtask(chat.id, msg.id)

    yield* prompt.loop({ sessionID: chat.id })

    const kids = yield* sessions.children(chat.id)
    expect(kids).toHaveLength(1)
    const child = kids[0]!
    const rules = child.permission ?? []
    expect(rules).toEqual(
      expect.arrayContaining([{ permission: "external_directory", pattern: "/tmp/allowed/*", action: "allow" }]),
    )
    expect(Permission.evaluate("external_directory", "/tmp/allowed/file", rules).action).toBe("allow")
    expect(Permission.evaluate("task", "anything", rules).action).toBe("deny")
  }),
)

noLLMServer.instance("prompt tools replace previous prompt tool rules", () =>
  Effect.gen(function* () {
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const session = yield* sessions.create({ title: "Prompt tools" })

    yield* prompt.prompt({
      sessionID: session.id,
      agent: "build",
      noReply: true,
      tools: { bash: false },
      parts: [{ type: "text", text: "first" }],
    })
    yield* prompt.prompt({
      sessionID: session.id,
      agent: "build",
      noReply: true,
      tools: { read: true },
      parts: [{ type: "text", text: "second" }],
    })

    const reloaded = yield* sessions.get(session.id)
    expect(reloaded.permission).toEqual([{ permission: "read", pattern: "*", action: "allow" }])
    expect(Permission.evaluate("bash", "anything", reloaded.permission ?? []).action).toBe("ask")
  }),
)

it.instance(
  "background subtask preserves metadata after tool-call transition",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Pinned" })
      yield* llm.hang
      const msg = yield* user(chat.id, "hello")
      yield* addSubtask(chat.id, msg.id)

      const fiber = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)

      const tool = yield* pollWithTimeout(
        Effect.gen(function* () {
          const msgs = yield* MessageV2.filterCompactedEffect(chat.id)
          const taskMsg = msgs.find((item) => item.info.role === "assistant" && item.info.agent === "general")
          const tool = taskMsg?.parts.find((part): part is SessionV1.ToolPart => part.type === "tool")
          if (tool?.state.status === "completed" && tool.state.metadata?.sessionId) return tool
        }),
        "timed out waiting for background subtask metadata",
        "20 seconds",
      )

      if (tool.state.status !== "completed") return
      expect(typeof tool.state.metadata?.sessionId).toBe("string")
      expect(tool.state.title).toBeDefined()
      expect(tool.state.metadata?.model).toBeDefined()

      yield* prompt.cancel(chat.id)
      yield* Fiber.await(fiber)
    }),
  30_000,
)

it.instance(
  "background task tool preserves metadata after tool-call transition",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({
        title: "Pinned",
        permission: [{ permission: "*", pattern: "*", action: "allow" }],
      })
      yield* llm.tool("task", {
        description: "inspect bug",
        prompt: "look into the cache key path",
        subagent_type: "general",
      })
      yield* llm.hang
      yield* user(chat.id, "hello")

      const fiber = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)

      const tool = yield* pollWithTimeout(
        Effect.gen(function* () {
          const msgs = yield* MessageV2.filterCompactedEffect(chat.id)
          const assistant = msgs.findLast((item) => item.info.role === "assistant" && item.info.agent === "build")
          const tool = assistant?.parts.find(
            (part): part is SessionV1.ToolPart => part.type === "tool" && part.tool === "task",
          )
          if (tool?.state.status === "completed" && tool.state.metadata?.sessionId) return tool
        }),
        "timed out waiting for background task metadata",
        "20 seconds",
      )

      if (tool.state.status !== "completed") return
      expect(typeof tool.state.metadata?.sessionId).toBe("string")
      expect(tool.state.title).toBe("inspect bug")
      expect(tool.state.metadata?.model).toBeDefined()

      yield* prompt.cancel(chat.id)
      yield* Fiber.await(fiber)
    }),
  30_000,
)

it.instance(
  "loop sets status to busy then idle",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const status = yield* SessionStatus.Service

      yield* llm.hang

      const chat = yield* sessions.create({})
      yield* user(chat.id, "hi")

      const fiber = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
      yield* llm.wait(1)
      expect((yield* status.get(chat.id)).type).toBe("busy")
      yield* prompt.cancel(chat.id)
      yield* Fiber.await(fiber)
      expect((yield* status.get(chat.id)).type).toBe("idle")
    }),
  3_000,
)

// Cancel semantics

it.instance("cancel interrupts loop and resolves with an assistant message", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const chat = yield* sessions.create({ title: "Pinned" })
    yield* seed(chat.id)

    yield* llm.hang

    yield* user(chat.id, "more")

    const fiber = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
    yield* llm.wait(1)
    yield* waitForBusy(chat.id)
    yield* prompt.cancel(chat.id)
    const exit = yield* Fiber.await(fiber)
    expect(Exit.isSuccess(exit)).toBe(true)
    if (Exit.isSuccess(exit)) {
      expect(exit.value.info.role).toBe("assistant")
    }
  }),
)

it.instance("cancel records MessageAbortedError on interrupted process", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const chat = yield* sessions.create({ title: "Pinned" })
    yield* llm.hang
    yield* user(chat.id, "hello")

    const fiber = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
    yield* llm.wait(1)
    yield* waitForBusy(chat.id)
    yield* prompt.cancel(chat.id)
    const exit = yield* Fiber.await(fiber)
    expect(Exit.isSuccess(exit)).toBe(true)
    if (Exit.isSuccess(exit)) {
      const info = exit.value.info
      if (info.role === "assistant") {
        expect(info.error?.name).toBe("MessageAbortedError")
      }
    }
  }),
)

raceNoLLMServer.instance(
  "finalizes assistant when cancelled before processor creation completes",
  () =>
    Effect.gen(function* () {
      processorCreateStarted.length = 0
      yield* Effect.addFinalizer(() =>
        Effect.sync(() => {
          processorCreateStarted.length = 0
        }),
      )

      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Processor creation race" })

      yield* prompt.prompt({
        sessionID: chat.id,
        agent: "build",
        noReply: true,
        parts: [{ type: "text", text: "first" }],
      })

      const firstCreate = defer<void>()
      processorCreateStarted.push(firstCreate.resolve)
      const first = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
      yield* Effect.promise(() => firstCreate.promise)

      yield* prompt.cancel(chat.id)
      const firstExit = yield* Fiber.await(first)
      expect(Exit.isSuccess(firstExit)).toBe(true)

      let messages = yield* sessions.messages({ sessionID: chat.id })
      const firstInterrupted = messages.at(-1)
      expect(firstInterrupted?.info.role).toBe("assistant")
      expect(firstInterrupted?.parts).toHaveLength(0)
      if (firstInterrupted?.info.role === "assistant") {
        expect(firstInterrupted.info.finish).toBeUndefined()
        expect(firstInterrupted.info.time.completed).toBeNumber()
        expect(firstInterrupted.info.error?.name).toBe("MessageAbortedError")
      }

      yield* prompt.prompt({
        sessionID: chat.id,
        agent: "build",
        noReply: true,
        parts: [{ type: "text", text: "second" }],
      })

      const secondCreate = defer<void>()
      processorCreateStarted.push(secondCreate.resolve)
      const second = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
      yield* Effect.promise(() => secondCreate.promise)

      yield* prompt.cancel(chat.id)
      const secondExit = yield* Fiber.await(second)
      expect(Exit.isSuccess(secondExit)).toBe(true)

      messages = yield* sessions.messages({ sessionID: chat.id })
      const poisonMessages = messages.filter(
        (message) =>
          message.info.role === "assistant" &&
          message.parts.length === 0 &&
          !message.info.finish &&
          !message.info.time.completed &&
          !message.info.error,
      )
      expect(poisonMessages).toHaveLength(0)

      const interruptedMessages = messages.filter(
        (message) =>
          message.info.role === "assistant" &&
          message.parts.length === 0 &&
          message.info.time.completed &&
          message.info.error?.name === "MessageAbortedError",
      )
      expect(interruptedMessages).toHaveLength(2)

      const lastUser = messages.at(-2)
      const lastAssistant = messages.at(-1)
      expect(lastUser?.info.role).toBe("user")
      expect(lastAssistant?.info.role).toBe("assistant")
      if (lastUser?.info.role === "user" && lastAssistant?.info.role === "assistant") {
        expect(lastAssistant.info.parentID).toBe(lastUser?.info.id)
      }
    }),
  { config: cfg },
  3_000,
)

noLLMServer.instance(
  "cancel finalizes subtask tool state",
  () =>
    Effect.gen(function* () {
      const ready = yield* Deferred.make<void>()
      const aborted = yield* Deferred.make<void>()
      const registry = yield* ToolRegistry.Service
      const { task } = yield* registry.named()
      const original = task.execute
      task.execute = (_args, ctx) =>
        Effect.callback<never>((_resume) => {
          ctx.abort.addEventListener("abort", () => succeedVoid(aborted), { once: true })
          if (ctx.abort.aborted) succeedVoid(aborted)
          succeedVoid(ready)
          return Effect.sync(() => succeedVoid(aborted))
        })
      yield* Effect.addFinalizer(() => Effect.sync(() => void (task.execute = original)))

      const { prompt, chat } = yield* boot()
      const msg = yield* user(chat.id, "hello")
      yield* addSubtask(chat.id, msg.id)

      const fiber = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
      yield* awaitWithTimeout(Deferred.await(ready), "timed out waiting for task tool to start", "10 seconds")
      yield* prompt.cancel(chat.id)

      const exit = yield* Fiber.await(fiber)
      expect(Exit.isSuccess(exit)).toBe(true)
      yield* awaitWithTimeout(Deferred.await(aborted), "timed out waiting for task tool abort", "10 seconds")

      const msgs = yield* MessageV2.filterCompactedEffect(chat.id)
      const taskMsg = msgs.find((item) => item.info.role === "assistant" && item.info.agent === "general")
      expect(taskMsg?.info.role).toBe("assistant")
      if (!taskMsg || taskMsg.info.role !== "assistant") return

      const tool = toolPart(taskMsg.parts)
      expect(tool?.type).toBe("tool")
      if (!tool) return

      expect(tool.state.status).not.toBe("running")
      expect(taskMsg.info.time.completed).toBeDefined()
      expect(taskMsg.info.finish).toBeDefined()
    }),
  { config: cfg },
  30_000,
)

it.instance(
  "cancel propagates from slash command subtask to child session",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const status = yield* SessionStatus.Service
      const chat = yield* sessions.create({ title: "Pinned" })
      yield* llm.hang
      const msg = yield* user(chat.id, "hello")
      yield* addSubtask(chat.id, msg.id)

      const fiber = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
      yield* llm.wait(1)

      const sessionID = yield* pollWithTimeout(
        Effect.gen(function* () {
          const msgs = yield* MessageV2.filterCompactedEffect(chat.id)
          const taskMsg = msgs.find((item) => item.info.role === "assistant" && item.info.agent === "general")
          const tool = taskMsg ? toolPart(taskMsg.parts) : undefined
          if (tool?.state.status === "completed" && typeof tool.state.metadata?.sessionId === "string") {
            return tool.state.metadata.sessionId
          }
        }),
        "timed out waiting for slash command subtask metadata",
        "20 seconds",
      )
      expect(typeof sessionID).toBe("string")
      const childID = SessionID.make(sessionID)
      expect((yield* status.get(childID)).type).toBe("busy")

      yield* prompt.cancel(chat.id)
      const exit = yield* Fiber.await(fiber)
      expect(Exit.isSuccess(exit)).toBe(true)

      expect((yield* status.get(chat.id)).type).toBe("idle")
      expect((yield* status.get(childID)).type).toBe("idle")
    }),
  30_000,
)

it.instance(
  "cancel with queued callers resolves all cleanly",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Pinned" })
      yield* llm.hang
      yield* user(chat.id, "hello")

      const a = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
      yield* llm.wait(1)
      const b = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
      yield* Effect.sleep(50)

      yield* prompt.cancel(chat.id)
      const [exitA, exitB] = yield* Effect.all([Fiber.await(a), Fiber.await(b)])
      expect(Exit.isSuccess(exitA)).toBe(true)
      expect(Exit.isSuccess(exitB)).toBe(true)
      if (Exit.isSuccess(exitA) && Exit.isSuccess(exitB)) {
        expect(exitA.value.info.id).toBe(exitB.value.info.id)
      }
    }),
  { git: true },
  10_000,
)

// Queue semantics

noLLMServer.instance("concurrent loop callers get same result", () =>
  Effect.gen(function* () {
    const { prompt, run, chat } = yield* boot()
    yield* seed(chat.id, { finish: "stop" })

    const [a, b] = yield* Effect.all([prompt.loop({ sessionID: chat.id }), prompt.loop({ sessionID: chat.id })], {
      concurrency: "unbounded",
    })

    expect(a.info.id).toBe(b.info.id)
    expect(a.info.role).toBe("assistant")
    yield* run.assertNotBusy(chat.id)
  }),
)

it.instance("concurrent loop callers all receive same error result", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const chat = yield* sessions.create({ title: "Pinned" })

    yield* llm.fail("boom")
    yield* user(chat.id, "hello")

    const [a, b] = yield* Effect.all([prompt.loop({ sessionID: chat.id }), prompt.loop({ sessionID: chat.id })], {
      concurrency: "unbounded",
    })
    expect(a.info.id).toBe(b.info.id)
    expect(a.info.role).toBe("assistant")
  }),
)

it.instance(
  "prompt submitted during an active run is included in the next LLM input",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const gate = yield* Deferred.make<void>()
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Pinned" })

      yield* llm.hold("first", deferredAsPromise(gate))
      yield* llm.text("second")

      const a = yield* prompt
        .prompt({
          sessionID: chat.id,
          agent: "build",
          model: ref,
          parts: [{ type: "text", text: "first" }],
        })
        .pipe(Effect.forkChild)

      yield* llm.wait(1)
      yield* waitForBusy(chat.id)

      const id = MessageID.ascending()
      const b = yield* prompt
        .prompt({
          sessionID: chat.id,
          messageID: id,
          agent: "build",
          model: ref,
          parts: [{ type: "text", text: "second" }],
        })
        .pipe(Effect.forkChild)

      yield* pollWithTimeout(
        sessions
          .messages({ sessionID: chat.id })
          .pipe(
            Effect.map((msgs) =>
              msgs.some((msg) => msg.info.role === "user" && msg.info.id === id) ? true : undefined,
            ),
          ),
        "timed out waiting for second prompt to save",
      )

      yield* Deferred.succeed(gate, void 0)

      const [ea, eb] = yield* Effect.all([Fiber.await(a), Fiber.await(b)])
      expect(Exit.isSuccess(ea)).toBe(true)
      expect(Exit.isSuccess(eb)).toBe(true)
      expect(yield* llm.calls).toBe(2)

      const msgs = yield* sessions.messages({ sessionID: chat.id })
      const assistants = msgs.filter((msg) => msg.info.role === "assistant")
      expect(assistants).toHaveLength(2)
      const last = assistants.at(-1)
      if (!last || last.info.role !== "assistant") throw new Error("expected second assistant")
      expect(last.info.parentID).toBe(id)
      expect(last.parts.some((part) => part.type === "text" && part.text === "second")).toBe(true)

      const inputs = yield* llm.inputs
      expect(inputs).toHaveLength(2)
      const messages = inputs.at(-1)?.messages
      if (!Array.isArray(messages)) throw new Error("expected LLM messages")
      // This test is about prompt admission during an active run, so assert the
      // admitted text and leave the metadata stamp to message-v2's own tests.
      const sent = messages.at(-1) as { role: string; content: { type: string; text: string }[] }
      expect(sent.role).toBe("user")
      expect(sent.content[0]).toEqual({ type: "text", text: "second" })
    }),
  { timeout: 15_000 },
)

it.instance("assertNotBusy fails with BusyError when loop running", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const prompt = yield* SessionPrompt.Service
    const run = yield* SessionRunState.Service
    const sessions = yield* Session.Service
    yield* llm.hang

    const chat = yield* sessions.create({})
    yield* user(chat.id, "hi")

    const fiber = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
    yield* llm.wait(1)
    yield* waitForBusy(chat.id)

    const exit = yield* run.assertNotBusy(chat.id).pipe(Effect.exit)
    expect(Exit.isFailure(exit)).toBe(true)
    if (Exit.isFailure(exit)) {
      expect(Cause.squash(exit.cause)).toBeInstanceOf(Session.BusyError)
      expect(Cause.squash(exit.cause)).toMatchObject({ _tag: "SessionBusyError", sessionID: chat.id })
    }

    yield* prompt.cancel(chat.id)
    yield* Fiber.await(fiber)
  }),
)

noLLMServer.instance("assertNotBusy succeeds when idle", () =>
  Effect.gen(function* () {
    const run = yield* SessionRunState.Service
    const sessions = yield* Session.Service

    const chat = yield* sessions.create({})
    const exit = yield* run.assertNotBusy(chat.id).pipe(Effect.exit)
    expect(Exit.isSuccess(exit)).toBe(true)
  }),
)

// Shell semantics

it.instance("shell rejects with BusyError when loop running", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const chat = yield* sessions.create({ title: "Pinned" })
    yield* llm.hang
    yield* user(chat.id, "hi")

    const fiber = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
    yield* llm.wait(1)
    yield* waitForBusy(chat.id)

    const exit = yield* prompt.shell({ sessionID: chat.id, agent: "build", command: "echo hi" }).pipe(Effect.exit)
    expect(Exit.isFailure(exit)).toBe(true)
    if (Exit.isFailure(exit)) {
      expect(Cause.squash(exit.cause)).toBeInstanceOf(Session.BusyError)
      expect(Cause.squash(exit.cause)).toMatchObject({ _tag: "SessionBusyError", sessionID: chat.id })
    }

    yield* prompt.cancel(chat.id)
    yield* Fiber.await(fiber)
  }),
)

unixNoLLMServer(
  "shell captures stdout and stderr in completed tool output",
  () =>
    Effect.gen(function* () {
      const { prompt, run, chat } = yield* boot()
      const result = yield* prompt.shell({
        sessionID: chat.id,
        agent: "build",
        command: "printf out && printf err >&2",
      })

      expect(result.info.role).toBe("assistant")
      const tool = completedTool(result.parts)
      if (!tool) return

      expect(tool.state.output).toContain("out")
      expect(tool.state.output).toContain("err")
      expect(tool.state.metadata.output).toContain("out")
      expect(tool.state.metadata.output).toContain("err")
      yield* run.assertNotBusy(chat.id)
    }),
  { config: cfg },
)

unixNoLLMServer(
  "shell completes a fast command on the preferred shell",
  () =>
    Effect.gen(function* () {
      const { directory: dir } = yield* TestInstance
      const { prompt, run, chat } = yield* boot()
      const result = yield* prompt.shell({
        sessionID: chat.id,
        agent: "build",
        command: "pwd",
      })

      expect(result.info.role).toBe("assistant")
      const tool = completedTool(result.parts)
      if (!tool) return

      expect(tool.state.input.command).toBe("pwd")
      expect(tool.state.output).toContain(dir)
      expect(tool.state.metadata.output).toContain(dir)
      yield* run.assertNotBusy(chat.id)
    }),
  { config: cfg },
)

unixNoLLMServer(
  "shell uses configured shell over env shell",
  () =>
    withSh(() =>
      Effect.gen(function* () {
        if (!(yield* hasBash)) return

        const { prompt, chat } = yield* boot()
        const result = yield* prompt.shell({
          sessionID: chat.id,
          agent: "build",
          command: "[[ 1 -eq 1 ]] && printf configured",
        })

        const tool = completedTool(result.parts)
        if (!tool) return
        expect(tool.state.output).toContain("configured")
      }),
    ),
  { config: { ...cfg, shell: "bash" } },
  30_000,
)

unixNoLLMServer(
  "shell commands can change directory after startup",
  () =>
    withSh(() =>
      Effect.gen(function* () {
        const { directory: dir } = yield* TestInstance
        const { prompt, run, chat } = yield* boot()
        const parent = path.dirname(dir)
        const result = yield* prompt.shell({
          sessionID: chat.id,
          agent: "build",
          command: "cd .. && pwd",
        })

        expect(result.info.role).toBe("assistant")
        const tool = completedTool(result.parts)
        if (!tool) return

        expect(tool.state.output).toContain(parent)
        expect(tool.state.metadata.output).toContain(parent)
        yield* run.assertNotBusy(chat.id)
      }),
    ),
  { config: cfg },
)

unixNoLLMServer(
  "shell lists files from the project directory",
  () =>
    Effect.gen(function* () {
      const { directory: dir } = yield* TestInstance
      const { prompt, run, chat } = yield* boot()
      yield* writeText(path.join(dir, "README.md"), "# e2e\n")

      const result = yield* prompt.shell({
        sessionID: chat.id,
        agent: "build",
        command: "command ls",
      })

      expect(result.info.role).toBe("assistant")
      const tool = completedTool(result.parts)
      if (!tool) return

      expect(tool.state.input.command).toBe("command ls")
      expect(tool.state.output).toContain("README.md")
      expect(tool.state.metadata.output).toContain("README.md")
      yield* run.assertNotBusy(chat.id)
    }),
  { config: cfg },
)

unixNoLLMServer(
  "shell captures stderr from a failing command",
  () =>
    Effect.gen(function* () {
      const { prompt, run, chat } = yield* boot()
      const result = yield* prompt.shell({
        sessionID: chat.id,
        agent: "build",
        command: "command -v __nonexistent_cmd_e2e__ || echo 'not found' >&2; exit 1",
      })

      expect(result.info.role).toBe("assistant")
      const tool = completedTool(result.parts)
      if (!tool) return

      expect(tool.state.output).toContain("not found")
      expect(tool.state.metadata.output).toContain("not found")
      yield* run.assertNotBusy(chat.id)
    }),
  { config: cfg },
)

unixNoLLMServer(
  "shell updates running metadata before process exit",
  () =>
    withSh(() =>
      Effect.gen(function* () {
        const { prompt, chat } = yield* boot()

        const fiber = yield* prompt
          .shell({ sessionID: chat.id, agent: "build", command: "printf first && sleep 0.2 && printf second" })
          .pipe(Effect.forkChild)

        yield* pollWithTimeout(
          Effect.gen(function* () {
            const msgs = yield* MessageV2.filterCompactedEffect(chat.id)
            const taskMsg = msgs.find((item) => item.info.role === "assistant")
            const tool = taskMsg ? toolPart(taskMsg.parts) : undefined
            if (tool?.state.status === "running" && tool.state.metadata?.output.includes("first")) return true
          }),
          "timed out waiting for running shell metadata",
        )

        const exit = yield* Fiber.await(fiber)
        expect(Exit.isSuccess(exit)).toBe(true)
      }),
    ),
  { config: cfg },
  30_000,
)

it.instance(
  "loop waits while shell runs and starts after shell exits",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({
        title: "Pinned",
        permission: [{ permission: "*", pattern: "*", action: "allow" }],
      })
      yield* llm.text("after-shell")

      const sh = yield* prompt
        .shell({ sessionID: chat.id, agent: "build", command: "sleep 0.2" })
        .pipe(Effect.forkChild)
      yield* waitForBusy(chat.id)

      const loop = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
      yield* Effect.sleep(50)

      expect(yield* llm.calls).toBe(0)

      yield* Fiber.await(sh)
      const exit = yield* Fiber.await(loop)

      expect(Exit.isSuccess(exit)).toBe(true)
      if (Exit.isSuccess(exit)) {
        expect(exit.value.info.role).toBe("assistant")
        expect(exit.value.parts.some((part) => part.type === "text" && part.text === "after-shell")).toBe(true)
      }
      expect(yield* llm.calls).toBe(1)
    }),
  { git: true },
  10_000,
)

it.instance(
  "shell completion resumes queued loop callers",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({
        title: "Pinned",
        permission: [{ permission: "*", pattern: "*", action: "allow" }],
      })
      yield* llm.text("done")

      const sh = yield* prompt
        .shell({ sessionID: chat.id, agent: "build", command: "sleep 0.2" })
        .pipe(Effect.forkChild)
      yield* waitForBusy(chat.id)

      const a = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
      const b = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
      yield* Effect.sleep(50)

      expect(yield* llm.calls).toBe(0)

      yield* Fiber.await(sh)
      const [ea, eb] = yield* Effect.all([Fiber.await(a), Fiber.await(b)])

      expect(Exit.isSuccess(ea)).toBe(true)
      expect(Exit.isSuccess(eb)).toBe(true)
      if (Exit.isSuccess(ea) && Exit.isSuccess(eb)) {
        expect(ea.value.info.id).toBe(eb.value.info.id)
        expect(ea.value.info.role).toBe("assistant")
      }
      expect(yield* llm.calls).toBe(1)
    }),
  { git: true },
  10_000,
)

unix(
  "command ! expansion uses configured shell over env shell",
  () =>
    withSh(() =>
      Effect.gen(function* () {
        if (!(yield* hasBash)) return
        const { llm } = yield* useServerConfig((url) => ({
          ...providerCfg(url),
          shell: "bash",
          command: {
            probe: {
              template: "Probe: !`[[ 1 -eq 1 ]] && printf configured`",
            },
          },
        }))

        const { prompt, chat } = yield* boot()
        yield* llm.text("done")

        const result = yield* prompt.command({
          sessionID: chat.id,
          command: "probe",
          arguments: "",
        })

        expect(result.info.role).toBe("assistant")
        const inputs = yield* llm.inputs
        expect(JSON.stringify(inputs.at(-1)?.messages)).toContain("configured")
      }),
    ),
  30_000,
)

it.instance(
  "notification admitted during a running turn is processed exactly once",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const { prompt, chat } = yield* boot()
      const release = defer<void>()
      yield* prompt.prompt({
        sessionID: chat.id,
        agent: "build",
        noReply: true,
        parts: [{ type: "text", text: "initial" }],
      })
      yield* llm.hold("first", release.promise)
      yield* llm.text("notification")

      const running = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
      yield* llm.wait(1)
      const notification = yield* prompt
        .notify(
          {
            sessionID: chat.id,
            agent: "build",
            parts: [{ type: "text", text: "background done", synthetic: true }],
          },
          yield* prompt.checkpoint(chat.id),
        )
        .pipe(Effect.forkChild)
      yield* pollWithTimeout(
        MessageV2.filterCompactedEffect(chat.id).pipe(
          Effect.map((messages) =>
            messages.some((message) =>
              message.parts.some((part) => part.type === "text" && part.text === "background done"),
            )
              ? true
              : undefined,
          ),
        ),
        "background notification was not admitted",
      )

      release.resolve()
      yield* Fiber.join(running)
      yield* Fiber.join(notification)

      expect(yield* llm.calls).toBe(2)
      const messages = yield* MessageV2.filterCompactedEffect(chat.id)
      const user = messages.findLast((message) =>
        message.parts.some((part) => part.type === "text" && part.text === "background done"),
      )
      const assistant = messages.findLast((message) => message.info.role === "assistant")
      expect(assistant?.info.role === "assistant" ? assistant.info.parentID : undefined).toBe(user?.info.id)
    }),
  { config: cfg },
  10_000,
)

it.instance(
  "notification released after cancellation is not admitted",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const { prompt, chat } = yield* boot()
      const checkpoint = yield* prompt.checkpoint(chat.id)
      const release = yield* Deferred.make<void>()
      yield* llm.text("unexpected")
      const notification = yield* Deferred.await(release).pipe(
        Effect.andThen(
          prompt.notify(
            {
              sessionID: chat.id,
              agent: "build",
              parts: [{ type: "text", text: "background done", synthetic: true }],
            },
            checkpoint,
          ),
        ),
        Effect.forkChild,
      )

      yield* prompt.cancel(chat.id)
      yield* Deferred.succeed(release, undefined)
      yield* Fiber.join(notification)

      expect(yield* llm.calls).toBe(0)
      expect(yield* MessageV2.filterCompactedEffect(chat.id)).toEqual([])
    }),
  { config: cfg },
  10_000,
)

unixNoLLMServer(
  "cancel interrupts shell and resolves cleanly",
  () =>
    withSh(() =>
      Effect.gen(function* () {
        const { prompt, run, chat } = yield* boot()
        const { directory: dir } = yield* TestInstance
        const afs = yield* FSUtil.Service
        const ready = path.join(dir, ".shell-ready")

        const sh = yield* prompt
          .shell({ sessionID: chat.id, agent: "build", command: ": > '.shell-ready'; sleep 30" })
          .pipe(Effect.forkChild)
        yield* pollWithTimeout(
          afs.existsSafe(ready).pipe(Effect.map((exists) => (exists ? (true as const) : undefined))),
          "shell never created readiness marker",
        )

        yield* prompt.cancel(chat.id)

        const status = yield* SessionStatus.Service
        expect((yield* status.get(chat.id)).type).toBe("idle")
        const busy = yield* run.assertNotBusy(chat.id).pipe(Effect.exit)
        expect(Exit.isSuccess(busy)).toBe(true)

        const exit = yield* Fiber.await(sh)
        expect(Exit.isSuccess(exit)).toBe(true)
        if (Exit.isSuccess(exit)) {
          expect(exit.value.info.role).toBe("assistant")
          const tool = completedTool(exit.value.parts)
          if (tool) {
            expect(tool.state.output).toContain("User aborted the command")
          }
        }
      }),
    ),
  { git: true, config: cfg },
  30_000,
)

unixNoLLMServer(
  "cancel persists aborted shell result when shell ignores TERM",
  () =>
    withSh(() =>
      Effect.gen(function* () {
        const { prompt, chat } = yield* boot()
        const { directory: dir } = yield* TestInstance
        const afs = yield* FSUtil.Service
        const ready = path.join(dir, ".trap-ready")

        const sh = yield* prompt
          .shell({
            sessionID: chat.id,
            agent: "build",
            // Touch marker AFTER trap installs so the test waits for the actual
            // ignore-TERM state before cancelling; otherwise SIGTERM can arrive
            // before `trap` runs and the escalation path is never exercised.
            command: `trap '' TERM; touch "${ready}"; sleep 30`,
          })
          .pipe(Effect.forkChild)

        yield* Effect.gen(function* () {
          while (!(yield* afs.existsSafe(ready))) {
            yield* Effect.sleep(Duration.millis(10))
          }
        }).pipe(Effect.timeout(Duration.seconds(5)))

        yield* prompt.cancel(chat.id)

        const exit = yield* Fiber.await(sh)
        expect(Exit.isSuccess(exit)).toBe(true)
        if (Exit.isSuccess(exit)) {
          expect(exit.value.info.role).toBe("assistant")
          const tool = completedTool(exit.value.parts)
          if (tool) {
            expect(tool.state.output).toContain("User aborted the command")
          }
        }
      }),
    ),
  { git: true, config: cfg },
  30_000,
)

unix(
  "cancel finalizes interrupted bash tool output through normal truncation",
  () =>
    Effect.gen(function* () {
      const { dir, llm } = yield* useServerConfig(providerCfg)
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({
        title: "Interrupted bash truncation",
        permission: [{ permission: "*", pattern: "*", action: "allow" }],
      })

      yield* prompt.prompt({
        sessionID: chat.id,
        agent: "build",
        noReply: true,
        parts: [{ type: "text", text: "run bash" }],
      })

      yield* llm.tool("bash", {
        command:
          'i=0; while [ "$i" -lt 4000 ]; do printf "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx %05d\\n" "$i"; i=$((i + 1)); done; printf truncation-ready; sleep 30',
        timeout: 30_000,
        workdir: path.resolve(dir),
      })

      const run = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
      yield* llm.wait(1)
      yield* pollWithTimeout(
        Effect.gen(function* () {
          const msgs = yield* MessageV2.filterCompactedEffect(chat.id)
          const assistant = msgs.findLast((item) => item.info.role === "assistant")
          const tool = assistant ? toolPart(assistant.parts) : undefined
          if (tool?.state.status === "running" && tool.state.metadata?.output.includes("truncation-ready")) return true
        }),
        "timed out waiting for truncated shell output",
      )
      yield* prompt.cancel(chat.id)

      const exit = yield* Fiber.await(run)
      expect(Exit.isSuccess(exit)).toBe(true)
      if (Exit.isFailure(exit)) return

      const tool = completedTool(exit.value.parts)
      if (!tool) return

      expect(tool.state.metadata.truncated).toBe(true)
      expect(typeof tool.state.metadata.outputPath).toBe("string")
      expect(tool.state.output).toMatch(/\.\.\.output truncated\.\.\./)
      expect(tool.state.output).toMatch(/Full output saved to:\s+\S+/)
      expect(tool.state.output).not.toContain("Tool execution aborted")
    }),
  { git: true },
  30_000,
)

unixNoLLMServer(
  "cancel interrupts loop queued behind shell",
  () =>
    Effect.gen(function* () {
      const { prompt, chat } = yield* boot()

      const sh = yield* prompt.shell({ sessionID: chat.id, agent: "build", command: "sleep 30" }).pipe(Effect.forkChild)
      yield* waitForBusy(chat.id)

      const loop = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
      yield* Effect.sleep(50)

      yield* prompt.cancel(chat.id)

      const exit = yield* Fiber.await(loop)
      expect(Exit.isSuccess(exit)).toBe(true)
      if (Exit.isSuccess(exit)) {
        const tool = completedTool(exit.value.parts)
        expect(tool?.state.output).toContain("User aborted the command")
      }

      yield* Fiber.await(sh)
    }),
  { git: true, config: cfg },
  30_000,
)

unixNoLLMServer(
  "shell rejects when another shell is already running",
  () =>
    withSh(() =>
      Effect.gen(function* () {
        const { prompt, chat } = yield* boot()

        const a = yield* prompt
          .shell({ sessionID: chat.id, agent: "build", command: "sleep 30" })
          .pipe(Effect.forkChild)
        yield* waitForBusy(chat.id)

        const exit = yield* prompt.shell({ sessionID: chat.id, agent: "build", command: "echo hi" }).pipe(Effect.exit)
        expect(Exit.isFailure(exit)).toBe(true)
        if (Exit.isFailure(exit)) {
          expect(Cause.squash(exit.cause)).toBeInstanceOf(Session.BusyError)
        }

        yield* prompt.cancel(chat.id)
        yield* Fiber.await(a)
      }),
    ),
  { git: true, config: cfg },
  30_000,
)

// Abort signal propagation tests for inline tool execution

function hangUntilAborted(tool: { execute: (...args: any[]) => any }) {
  return Effect.gen(function* () {
    const ready = yield* Deferred.make<void>()
    const aborted = yield* Deferred.make<void>()
    const original = tool.execute
    tool.execute = (_args: any, ctx: any) => {
      ctx.abort.addEventListener("abort", () => succeedVoid(aborted), { once: true })
      if (ctx.abort.aborted) succeedVoid(aborted)
      succeedVoid(ready)
      return Effect.callback<never>(() => Effect.sync(() => succeedVoid(aborted)))
    }
    const restore = Effect.addFinalizer(() => Effect.sync(() => void (tool.execute = original)))
    return { ready, aborted, restore }
  })
}

noLLMServer.instance(
  "interrupt propagates abort signal to read tool via file part (text/plain)",
  () =>
    Effect.gen(function* () {
      const { directory: dir } = yield* TestInstance
      const registry = yield* ToolRegistry.Service
      const { read } = yield* registry.named()
      const { ready, restore } = yield* hangUntilAborted(read)
      yield* restore

      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Abort Test" })

      const testFile = path.join(dir, "test.txt")
      yield* writeText(testFile, "hello world")

      const fiber = yield* prompt
        .prompt({
          sessionID: chat.id,
          agent: "build",
          parts: [
            { type: "text", text: "read this" },
            { type: "file", url: `file://${testFile}`, filename: "test.txt", mime: "text/plain" },
          ],
        })
        .pipe(Effect.forkChild)

      yield* awaitWithTimeout(Deferred.await(ready), "timed out waiting for read tool to start", "10 seconds")
      yield* prompt.cancel(chat.id)
      yield* Fiber.interrupt(fiber)
      const exit = yield* Fiber.await(fiber)
      expect(Exit.isFailure(exit)).toBe(true)
    }),
  { config: cfg },
  30_000,
)

noLLMServer.instance(
  "interrupt propagates abort signal to read tool via file part (directory)",
  () =>
    Effect.gen(function* () {
      const { directory: dir } = yield* TestInstance
      const registry = yield* ToolRegistry.Service
      const { read } = yield* registry.named()
      const { ready, restore } = yield* hangUntilAborted(read)
      yield* restore

      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Abort Test" })

      const fiber = yield* prompt
        .prompt({
          sessionID: chat.id,
          agent: "build",
          parts: [
            { type: "text", text: "read this" },
            { type: "file", url: `file://${dir}`, filename: "dir", mime: "application/x-directory" },
          ],
        })
        .pipe(Effect.forkChild)

      yield* awaitWithTimeout(Deferred.await(ready), "timed out waiting for read tool to start", "10 seconds")
      yield* prompt.cancel(chat.id)
      yield* Fiber.interrupt(fiber)
      const exit = yield* Fiber.await(fiber)
      expect(Exit.isFailure(exit)).toBe(true)
    }),
  { config: cfg },
  30_000,
)

// Missing file handling

noLLMServer.instance(
  "does not fail the prompt when a file part is missing",
  () =>
    Effect.gen(function* () {
      const { directory: dir } = yield* TestInstance
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const session = yield* sessions.create({})

      const missing = path.join(dir, "does-not-exist.ts")
      const msg = yield* prompt.prompt({
        sessionID: session.id,
        agent: "build",
        noReply: true,
        parts: [
          { type: "text", text: "please review @does-not-exist.ts" },
          {
            type: "file",
            mime: "text/plain",
            url: `file://${missing}`,
            filename: "does-not-exist.ts",
          },
        ],
      })

      if (msg.info.role !== "user") throw new Error("expected user message")
      const hasFailure = msg.parts.some(
        (part) => part.type === "text" && part.synthetic && part.text.includes("Read tool failed to read"),
      )
      expect(hasFailure).toBe(true)

      yield* sessions.remove(session.id)
    }),
  { config: cfg },
)

noLLMServer.instance(
  "keeps stored part order stable when file resolution is async",
  () =>
    Effect.gen(function* () {
      const { directory: dir } = yield* TestInstance
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const session = yield* sessions.create({})

      const missing = path.join(dir, "still-missing.ts")
      const msg = yield* prompt.prompt({
        sessionID: session.id,
        agent: "build",
        noReply: true,
        parts: [
          {
            type: "file",
            mime: "text/plain",
            url: `file://${missing}`,
            filename: "still-missing.ts",
          },
          { type: "text", text: "after-file" },
        ],
      })

      if (msg.info.role !== "user") throw new Error("expected user message")

      const stored = yield* MessageV2.get({
        sessionID: session.id,
        messageID: msg.info.id,
      })
      const text = stored.parts.filter((part) => part.type === "text").map((part) => part.text)

      expect(text[0]?.startsWith("Called the Read tool with the following input:")).toBe(true)
      expect(text[1]?.includes("Read tool failed to read")).toBe(true)
      expect(text[2]).toBe("after-file")

      yield* sessions.remove(session.id)
    }),
  { config: cfg },
)

// Special characters in filenames

noLLMServer.instance(
  "handles filenames with # character",
  () =>
    Effect.gen(function* () {
      const { directory: dir } = yield* TestInstance
      yield* writeText(path.join(dir, "file#name.txt"), "special content\n")

      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const session = yield* sessions.create({})
      const parts = yield* prompt.resolvePromptParts("Read @file#name.txt")
      const fileParts = parts.filter((part) => part.type === "file")

      expect(fileParts.length).toBe(1)
      expect(fileParts[0].filename).toBe("file#name.txt")
      expect(fileParts[0].url).toContain("%23")

      const decodedPath = fileURLToPath(fileParts[0].url)
      expect(decodedPath).toBe(path.join(dir, "file#name.txt"))

      const message = yield* prompt.prompt({
        sessionID: session.id,
        parts,
        noReply: true,
      })
      const stored = yield* MessageV2.get({ sessionID: session.id, messageID: message.info.id })
      const textParts = stored.parts.filter((part) => part.type === "text")
      const hasContent = textParts.some((part) => part.text.includes("special content"))
      expect(hasContent).toBe(true)

      yield* sessions.remove(session.id)
    }),
  { git: true, config: cfg },
)

// Regression: empty assistant turn loop

it.instance("does not loop empty assistant turns for a simple reply", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const session = yield* sessions.create({ title: "Prompt regression" })

    yield* llm.text("packages/opencode/src/session/processor.ts")

    const result = yield* prompt.prompt({
      sessionID: session.id,
      agent: "build",
      parts: [{ type: "text", text: "Where is SessionProcessor?" }],
    })

    expect(result.info.role).toBe("assistant")
    expect(result.parts.some((part) => part.type === "text" && part.text.includes("processor.ts"))).toBe(true)

    const msgs = yield* sessions.messages({ sessionID: session.id })
    expect(msgs.filter((msg) => msg.info.role === "assistant")).toHaveLength(1)
    expect(yield* llm.calls).toBe(1)
  }),
)

it.instance("records aborted errors when prompt is cancelled mid-stream", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const session = yield* sessions.create({ title: "Prompt cancel regression" })

    yield* llm.hang

    const fiber = yield* prompt
      .prompt({
        sessionID: session.id,
        agent: "build",
        parts: [{ type: "text", text: "Cancel me" }],
      })
      .pipe(Effect.forkChild)

    yield* llm.wait(1)
    yield* waitForBusy(session.id)
    yield* prompt.cancel(session.id)

    const exit = yield* Fiber.await(fiber)
    expect(Exit.isSuccess(exit)).toBe(true)
    if (Exit.isSuccess(exit)) {
      expect(exit.value.info.role).toBe("assistant")
      if (exit.value.info.role === "assistant") {
        expect(exit.value.info.error?.name).toBe("MessageAbortedError")
      }
    }

    const msgs = yield* sessions.messages({ sessionID: session.id })
    const last = msgs.findLast((msg) => msg.info.role === "assistant")
    expect(last?.info.role).toBe("assistant")
    if (last?.info.role === "assistant") {
      expect(last.info.error?.name).toBe("MessageAbortedError")
    }
  }),
)

// Agent variant

noLLMServer.instance(
  "applies agent variant only when using agent model",
  () =>
    Effect.gen(function* () {
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const session = yield* sessions.create({})

      const other = yield* prompt.prompt({
        sessionID: session.id,
        agent: "build",
        model: { providerID: ProviderV2.ID.make("opencode"), modelID: ModelV2.ID.make("kimi-k2.5-free") },
        noReply: true,
        parts: [{ type: "text", text: "hello" }],
      })
      if (other.info.role !== "user") throw new Error("expected user message")
      expect(other.info.model.variant).toBeUndefined()

      const match = yield* prompt.prompt({
        sessionID: session.id,
        agent: "build",
        noReply: true,
        parts: [{ type: "text", text: "hello again" }],
      })
      if (match.info.role !== "user") throw new Error("expected user message")
      expect(match.info.model).toEqual({
        providerID: ProviderV2.ID.make("test"),
        modelID: ModelV2.ID.make("test-model"),
        variant: "xhigh",
      })
      expect(match.info.model.variant).toBe("xhigh")

      const override = yield* prompt.prompt({
        sessionID: session.id,
        agent: "build",
        noReply: true,
        variant: "high",
        parts: [{ type: "text", text: "hello third" }],
      })
      if (override.info.role !== "user") throw new Error("expected user message")
      expect(override.info.model.variant).toBe("high")

      yield* sessions.remove(session.id)
    }),
  {
    config: {
      ...cfg,
      provider: {
        ...cfg.provider,
        test: {
          ...cfg.provider.test,
          models: {
            "test-model": {
              ...cfg.provider.test.models["test-model"],
              variants: { xhigh: {}, high: {} },
            },
          },
        },
      },
      agent: {
        build: {
          model: "test/test-model",
          variant: "xhigh",
        },
      },
    },
  },
)

// Agent / command resolution errors

noLLMServer.instance(
  "unknown agent throws typed error",
  () =>
    Effect.gen(function* () {
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const session = yield* sessions.create({})
      const exit = yield* prompt
        .prompt({
          sessionID: session.id,
          agent: "nonexistent-agent-xyz",
          noReply: true,
          parts: [{ type: "text", text: "hello" }],
        })
        .pipe(Effect.exit)

      expect(Exit.isFailure(exit)).toBe(true)
      if (Exit.isFailure(exit)) {
        const err = Cause.squash(exit.cause)
        expect(err).not.toBeInstanceOf(TypeError)
        expect(NamedError.Unknown.isInstance(err)).toBe(true)
        if (NamedError.Unknown.isInstance(err)) {
          expect(err.data.message).toContain('Agent not found: "nonexistent-agent-xyz"')
        }
      }
    }),
  30_000,
)

noLLMServer.instance(
  "unknown agent error includes available agent names",
  () =>
    Effect.gen(function* () {
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const session = yield* sessions.create({})
      const exit = yield* prompt
        .prompt({
          sessionID: session.id,
          agent: "nonexistent-agent-xyz",
          noReply: true,
          parts: [{ type: "text", text: "hello" }],
        })
        .pipe(Effect.exit)

      expect(Exit.isFailure(exit)).toBe(true)
      if (Exit.isFailure(exit)) {
        const err = Cause.squash(exit.cause)
        expect(NamedError.Unknown.isInstance(err)).toBe(true)
        if (NamedError.Unknown.isInstance(err)) {
          expect(err.data.message).toContain("build")
        }
      }
    }),
  30_000,
)

noLLMServer.instance(
  "unknown command throws typed error with available names",
  () =>
    Effect.gen(function* () {
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const session = yield* sessions.create({})
      const exit = yield* prompt
        .command({
          sessionID: session.id,
          command: "nonexistent-command-xyz",
          arguments: "",
        })
        .pipe(Effect.exit)

      expect(Exit.isFailure(exit)).toBe(true)
      if (Exit.isFailure(exit)) {
        const err = Cause.squash(exit.cause)
        expect(err).not.toBeInstanceOf(TypeError)
        expect(NamedError.Unknown.isInstance(err)).toBe(true)
        if (NamedError.Unknown.isInstance(err)) {
          expect(err.data.message).toContain('Command not found: "nonexistent-command-xyz"')
          expect(err.data.message).toContain("init")
        }
      }
    }),
  30_000,
)
