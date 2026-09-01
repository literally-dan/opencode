import { afterEach, describe, expect } from "bun:test"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { Database } from "@opencode-ai/core/database/database"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { SessionProjector } from "@opencode-ai/core/session/projector"
import { Cause, Deferred, Effect, Exit, Fiber, Option } from "effect"
import { Agent } from "../../src/agent/agent"
import { BackgroundJob } from "@/background/job"
import { EventV2Bridge } from "@/event-v2-bridge"
import { Config } from "@/config/config"
import { CrossSpawnSpawner } from "@opencode-ai/core/cross-spawn-spawner"
import { Ripgrep } from "@opencode-ai/core/ripgrep"
import { Session } from "@/session/session"
import type { SessionPrompt } from "../../src/session/prompt"
import { MessageID, PartID, SessionID } from "../../src/session/schema"
import { SessionRunState } from "@/session/run-state"
import { SessionStatus } from "@/session/status"

import { TaskTool, type TaskPromptOps } from "../../src/tool/task"
import { Truncate } from "@/tool/truncate"
import { ToolRegistry } from "@/tool/registry"
import { RuntimeFlags } from "@/effect/runtime-flags"
import { disposeAllInstances } from "../fixture/fixture"
import { pollWithTimeout, testEffect } from "../lib/effect"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { ModelV2 } from "@opencode-ai/core/model"
import { WorkspaceV2 } from "@opencode-ai/core/workspace"

afterEach(async () => {
  await disposeAllInstances()
})

const ref = {
  providerID: ProviderV2.ID.make("test"),
  modelID: ModelV2.ID.make("test-model"),
}

const layer = (flags: Partial<RuntimeFlags.Info> = {}) =>
  LayerNode.compile(
    LayerNode.group([
      Agent.node,
      BackgroundJob.node,
      EventV2Bridge.node,
      Config.node,
      CrossSpawnSpawner.node,
      Session.node,
      SessionProjector.node,
      SessionRunState.node,
      SessionStatus.node,
      Truncate.node,
      ToolRegistry.node,
      Database.node,
      RuntimeFlags.node,
      Ripgrep.node,
    ]),
    [[RuntimeFlags.node, RuntimeFlags.layer(flags)]],
  )

const it = testEffect(layer())
const background = testEffect(layer())

function defer<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

const seed = Effect.fn("TaskToolTest.seed")(function* (title = "Pinned") {
  const session = yield* Session.Service
  const chat = yield* session.create({ title })
  const user = yield* session.updateMessage({
    id: MessageID.ascending(),
    role: "user",
    sessionID: chat.id,
    agent: "build",
    model: ref,
    time: { created: Date.now() },
  })
  const assistant: SessionV1.Assistant = {
    id: MessageID.ascending(),
    role: "assistant",
    parentID: user.id,
    sessionID: chat.id,
    mode: "build",
    agent: "build",
    cost: 0,
    path: { cwd: "/tmp", root: "/tmp" },
    tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
    modelID: ref.modelID,
    providerID: ref.providerID,
    variant: "xhigh",
    time: { created: Date.now() },
  }
  yield* session.updateMessage(assistant)
  return { chat, assistant }
})

function stubOps(opts?: {
  onPrompt?: (input: SessionPrompt.PromptInput) => void
  onNotify?: (input: SessionPrompt.PromptInput) => void
  text?: string
  error?: NonNullable<SessionV1.Assistant["error"]>
  toolError?: string
}): TaskPromptOps {
  return {
    cancel: () => Effect.void,
    checkpoint: () => Effect.succeed(0),
    retain: () => Effect.succeed(Option.some(Effect.void)),
    admitIfCurrent: (effect) => effect.pipe(Effect.map(Option.some)),
    admitChild: (_sessionID, effect) => effect.pipe(Effect.map(Option.some)),
    resolvePromptParts: (template) => Effect.succeed([{ type: "text" as const, text: template }]),
    prompt: (input) =>
      Effect.sync(() => {
        opts?.onPrompt?.(input)
        return reply(input, opts?.text ?? "done", opts?.error, opts?.toolError)
      }),
    admitNotification: (input) =>
      Effect.sync(() => {
        opts?.onNotify?.(input)
        return Option.some(Effect.succeed(undefined))
      }),
    notify: (input) => Effect.sync(() => opts?.onNotify?.(input)),
  }
}

function reply(
  input: SessionPrompt.PromptInput,
  text: string,
  error?: NonNullable<SessionV1.Assistant["error"]>,
  toolError?: string,
): SessionV1.WithParts {
  const id = MessageID.ascending()
  return {
    info: {
      id,
      role: "assistant",
      parentID: input.messageID ?? MessageID.ascending(),
      sessionID: input.sessionID,
      mode: input.agent ?? "general",
      agent: input.agent ?? "general",
      cost: 0,
      path: { cwd: "/tmp", root: "/tmp" },
      tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
      modelID: input.model?.modelID ?? ref.modelID,
      providerID: input.model?.providerID ?? ref.providerID,
      time: { created: Date.now() },
      finish: error ? "content-filter" : "stop",
      error,
    },
    parts: [
      {
        id: PartID.ascending(),
        messageID: id,
        sessionID: input.sessionID,
        type: "text",
        text,
      },
      ...(toolError
        ? [
            {
              id: PartID.ascending(),
              messageID: id,
              sessionID: input.sessionID,
              type: "tool" as const,
              tool: "read",
              callID: "call-1",
              state: {
                status: "error" as const,
                input: { filePath: "/external" },
                error: toolError,
                time: { start: Date.now(), end: Date.now() },
              },
            },
          ]
        : []),
    ],
  }
}

function failureMessage(exit: Exit.Exit<unknown, unknown>) {
  if (Exit.isSuccess(exit)) throw new Error("expected task failure")
  const failure = Cause.squash(exit.cause)
  if (!(failure instanceof Error)) throw new Error("expected Error defect")
  return failure.message
}

describe("tool.task", () => {
  it.instance(
    "description sorts subagents by name and is stable across calls",
    () =>
      Effect.gen(function* () {
        const agent = yield* Agent.Service
        const build = yield* agent.get("build")
        const registry = yield* ToolRegistry.Service
        const get = Effect.fnUntraced(function* () {
          const tools = yield* registry.tools({ ...ref, agent: build })
          return tools.find((tool) => tool.id === TaskTool.id)?.description ?? ""
        })
        const first = yield* get()
        const second = yield* get()

        expect(first).toBe(second)

        const alpha = first.indexOf("- alpha: Alpha agent")
        const explore = first.indexOf("- explore:")
        const general = first.indexOf("- general:")
        const zebra = first.indexOf("- zebra: Zebra agent")

        expect(alpha).toBeGreaterThan(-1)
        expect(explore).toBeGreaterThan(alpha)
        expect(general).toBeGreaterThan(explore)
        expect(zebra).toBeGreaterThan(general)
      }),
    {
      config: {
        agent: {
          zebra: {
            description: "Zebra agent",
            mode: "subagent",
          },
          alpha: {
            description: "Alpha agent",
            mode: "subagent",
          },
        },
      },
    },
  )

  it.instance(
    "description hides denied subagents for the caller",
    () =>
      Effect.gen(function* () {
        const agent = yield* Agent.Service
        const build = yield* agent.get("build")
        const registry = yield* ToolRegistry.Service
        const description =
          (yield* registry.tools({ ...ref, agent: build })).find((tool) => tool.id === TaskTool.id)?.description ?? ""

        expect(description).toContain("- alpha: Alpha agent")
        expect(description).not.toContain("- zebra: Zebra agent")
      }),
    {
      config: {
        permission: {
          task: {
            "*": "allow",
            zebra: "deny",
          },
        },
        agent: {
          zebra: {
            description: "Zebra agent",
            mode: "subagent",
          },
          alpha: {
            description: "Alpha agent",
            mode: "subagent",
          },
        },
      },
    },
  )

  it.instance("execute resumes an existing task session from task_id", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const child = yield* sessions.create({ parentID: chat.id, title: "Existing child", agent: "general" })
      const tool = yield* TaskTool
      const def = yield* tool.init()
      let seen: SessionPrompt.PromptInput | undefined
      const promptOps = stubOps({ text: "resumed", onPrompt: (input) => (seen = input) })

      const result = yield* def.execute(
        {
          description: "inspect bug",
          prompt: "look into the cache key path",
          subagent_type: "general",
          task_id: child.id,
        },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "build",
          abort: new AbortController().signal,
          extra: { promptOps },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      const kids = yield* sessions.children(chat.id)
      expect(kids).toHaveLength(1)
      expect(kids[0]?.id).toBe(child.id)
      expect(result.metadata.sessionId).toBe(child.id)
      expect(result.output).toContain(`<task id="${child.id}" state="running">`)
      expect((yield* jobs.wait({ id: child.id })).info?.output).toBe("resumed")
      expect(seen?.sessionID).toBe(child.id)
      expect(seen?.variant).toBe("xhigh")
    }),
  )

  it.instance("execute surfaces child errors with a resumable task_id", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()

      const result = yield* def.execute(
        {
          description: "inspect bug",
          prompt: "look into the cache key path",
          subagent_type: "general",
        },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "build",
          abort: new AbortController().signal,
          extra: {
            promptOps: stubOps({
              text: "",
              error: new SessionV1.APIError({ message: "Network connection lost", isRetryable: false }).toObject(),
            }),
          },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      const child = (yield* sessions.children(chat.id))[0]
      expect(child).toBeDefined()
      expect(result.output).toContain(`state="running"`)
      const failure = yield* jobs.wait({ id: result.metadata.sessionId })
      expect(failure.info?.status).toBe("error")
      expect(failure.info?.error).toContain(`Subagent failed (task_id: ${child?.id}):`)
      expect(failure.info?.error).toContain("The subagent stopped with APIError: Network connection lost")
    }),
  )

  it.instance("execute surfaces terminal child tool errors with a resumable task_id", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()

      const result = yield* def.execute(
        {
          description: "inspect external directory",
          prompt: "read the external directory",
          subagent_type: "general",
        },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "build",
          abort: new AbortController().signal,
          extra: {
            promptOps: stubOps({
              text: "I will inspect the directory.",
              toolError: "The user rejected permission to use this specific tool call.",
            }),
          },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      const child = (yield* sessions.children(chat.id))[0]
      expect(result.output).toContain(`state="running"`)
      const failure = yield* jobs.wait({ id: result.metadata.sessionId })
      expect(failure.info?.status).toBe("error")
      expect(failure.info?.error).toBe(
        `Subagent failed (task_id: ${child?.id}): The user rejected permission to use this specific tool call.`,
      )
    }),
  )

  it.instance("execute uses an explicit model override", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()
      let seen: SessionPrompt.PromptInput | undefined

      const result = yield* def.execute(
        {
          description: "inspect bug",
          prompt: "look into the cache key path",
          subagent_type: "general",
          model: "other/nested/model",
        },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "build",
          abort: new AbortController().signal,
          extra: { promptOps: stubOps({ onPrompt: (input) => (seen = input) }) },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      expect(result.metadata.model).toEqual({
        providerID: ProviderV2.ID.make("other"),
        modelID: ModelV2.ID.make("nested/model"),
      })
      expect(seen?.model).toEqual({
        providerID: ProviderV2.ID.make("other"),
        modelID: ModelV2.ID.make("nested/model"),
      })
      expect(seen?.variant).toBeUndefined()
    }),
  )

  it.instance(
    "execute prefers an explicit model over the subagent model",
    () =>
      Effect.gen(function* () {
        const { chat, assistant } = yield* seed()
        const tool = yield* TaskTool
        const def = yield* tool.init()
        let seen: SessionPrompt.PromptInput | undefined

        yield* def.execute(
          {
            description: "inspect bug",
            prompt: "look into the cache key path",
            subagent_type: "specialized",
            model: "override/model",
          },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "build",
            abort: new AbortController().signal,
            extra: { promptOps: stubOps({ onPrompt: (input) => (seen = input) }) },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )

        expect(seen?.model).toEqual({
          providerID: ProviderV2.ID.make("override"),
          modelID: ModelV2.ID.make("model"),
        })
      }),
    {
      config: {
        agent: {
          specialized: {
            mode: "subagent",
            model: "configured/model",
          },
        },
      },
    },
  )

  it.instance("execute asks by default and skips checks when bypassed", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const calls: unknown[] = []
      const promptOps = stubOps()

      const exec = (extra?: Record<string, any>) =>
        def.execute(
          {
            description: "inspect bug",
            prompt: "look into the cache key path",
            subagent_type: "general",
          },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "build",
            abort: new AbortController().signal,
            extra: { promptOps, ...extra },
            messages: [],
            metadata: () => Effect.void,
            ask: (input) =>
              Effect.sync(() => {
                calls.push(input)
              }),
          },
        )

      yield* exec()
      yield* exec({ bypassAgentCheck: true })

      expect(calls).toHaveLength(1)
      expect(calls[0]).toEqual({
        permission: "task",
        patterns: ["general"],
        always: ["*"],
        metadata: {
          description: "inspect bug",
          subagent_type: "general",
        },
      })
    }),
  )

  it.instance("execute rejects a missing task_id without creating a replacement", () =>
    Effect.gen(function* () {
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()

      const exit = yield* def
        .execute(
          {
            description: "inspect bug",
            prompt: "look into the cache key path",
            subagent_type: "general",
            task_id: "ses_missing",
          },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "build",
            abort: new AbortController().signal,
            extra: { promptOps: stubOps() },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )
        .pipe(Effect.exit)

      expect(failureMessage(exit)).toContain("task_id ses_missing: session not found")
      expect(yield* sessions.children(chat.id)).toHaveLength(0)
    }),
  )

  it.instance("execute rejects an invalid task_id", () =>
    Effect.gen(function* () {
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()

      const exit = yield* def
        .execute(
          {
            description: "inspect bug",
            prompt: "look into the cache key path",
            subagent_type: "general",
            task_id: "not-a-session",
          },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "build",
            abort: new AbortController().signal,
            extra: { promptOps: stubOps() },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )
        .pipe(Effect.exit)

      expect(failureMessage(exit)).toContain("task_id not-a-session: invalid session ID")
      expect(yield* sessions.children(chat.id)).toHaveLength(0)
    }),
  )

  it.instance("execute rejects its current session as task_id", () =>
    Effect.gen(function* () {
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()

      const exit = yield* def
        .execute(
          {
            description: "inspect bug",
            prompt: "look into the cache key path",
            subagent_type: "general",
            task_id: chat.id,
          },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "build",
            abort: new AbortController().signal,
            extra: { promptOps: stubOps() },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )
        .pipe(Effect.exit)

      expect(failureMessage(exit)).toContain(`task_id ${chat.id}: a task cannot resume its parent session`)
      expect(yield* sessions.children(chat.id)).toHaveLength(0)
    }),
  )

  it.instance("execute rejects an unrelated task_id", () =>
    Effect.gen(function* () {
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const other = yield* sessions.create({ title: "Other parent" })
      const unrelated = yield* sessions.create({ parentID: other.id, title: "Other child", agent: "general" })
      const tool = yield* TaskTool
      const def = yield* tool.init()

      const exit = yield* def
        .execute(
          {
            description: "inspect bug",
            prompt: "look into the cache key path",
            subagent_type: "general",
            task_id: unrelated.id,
          },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "build",
            abort: new AbortController().signal,
            extra: { promptOps: stubOps() },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )
        .pipe(Effect.exit)

      expect(failureMessage(exit)).toContain(`task_id ${unrelated.id}: session is not a direct child of ${chat.id}`)
      expect(yield* sessions.children(chat.id)).toHaveLength(0)
    }),
  )

  it.instance("execute rejects a task_id created for another agent", () =>
    Effect.gen(function* () {
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const child = yield* sessions.create({ parentID: chat.id, title: "Explore child", agent: "explore" })
      const tool = yield* TaskTool
      const def = yield* tool.init()

      const exit = yield* def
        .execute(
          {
            description: "inspect bug",
            prompt: "look into the cache key path",
            subagent_type: "general",
            task_id: child.id,
          },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "build",
            abort: new AbortController().signal,
            extra: { promptOps: stubOps() },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )
        .pipe(Effect.exit)

      expect(failureMessage(exit)).toContain(`task_id ${child.id}: session agent explore does not match general`)
      expect(yield* sessions.children(chat.id)).toEqual([child])
    }),
  )

  it.instance("execute rejects a task_id from another workspace boundary", () =>
    Effect.gen(function* () {
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const child = yield* sessions.create({
        parentID: chat.id,
        title: "Other workspace child",
        agent: "general",
        workspaceID: WorkspaceV2.ID.create(),
      })
      const tool = yield* TaskTool
      const def = yield* tool.init()

      const exit = yield* def
        .execute(
          {
            description: "inspect bug",
            prompt: "look into the cache key path",
            subagent_type: "general",
            task_id: child.id,
          },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "build",
            abort: new AbortController().signal,
            extra: { promptOps: stubOps() },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )
        .pipe(Effect.exit)

      expect(failureMessage(exit)).toContain(`task_id ${child.id}: session belongs to a different project or location`)
      expect(yield* sessions.children(chat.id)).toEqual([child])
    }),
  )

  it.instance(
    "prevents subagents from launching subagents when subagent_depth is 1",
    () =>
      Effect.gen(function* () {
        const jobs = yield* BackgroundJob.Service
        const sessions = yield* Session.Service
        const { chat, assistant } = yield* seed()
        const child = yield* sessions.createTask({ parentID: chat.id, title: "child" })
        yield* jobs.start({
          id: child.id,
          type: "task",
          metadata: { sessionId: child.id, parentSessionId: chat.id },
          run: Effect.never,
        })
        const nestedAssistant = yield* sessions.updateMessage({
          ...assistant,
          id: MessageID.ascending(),
          parentID: MessageID.ascending(),
          sessionID: child.id,
        })
        const tool = yield* TaskTool
        const def = yield* tool.init()
        let asked = false

        const exit = yield* def
          .execute(
            {
              description: "inspect bug",
              prompt: "look into the cache key path",
              subagent_type: "general",
            },
            {
              sessionID: child.id,
              messageID: nestedAssistant.id,
              agent: "general",
              abort: new AbortController().signal,
              extra: { promptOps: stubOps() },
              messages: [],
              metadata: () => Effect.void,
              ask: () => Effect.sync(() => (asked = true)),
            },
          )
          .pipe(Effect.exit)

        expect(Exit.isFailure(exit)).toBe(true)
        expect(asked).toBe(false)
        expect(yield* sessions.children(child.id)).toHaveLength(0)
      }),
    { config: { subagent_depth: 1 } },
  )

  it.instance("lets a coordinating subagent launch a subagent by default", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const child = yield* sessions.createTask({ parentID: chat.id, title: "child" })
      yield* jobs.start({
        id: child.id,
        type: "task",
        metadata: { sessionId: child.id, parentSessionId: chat.id },
        run: Effect.never,
      })
      const nestedAssistant = yield* sessions.updateMessage({
        ...assistant,
        id: MessageID.ascending(),
        parentID: MessageID.ascending(),
        sessionID: child.id,
      })
      const tool = yield* TaskTool
      const def = yield* tool.init()

      const result = yield* def.execute(
        {
          description: "inspect bug",
          prompt: "look into the cache key path",
          subagent_type: "general",
        },
        {
          sessionID: child.id,
          messageID: nestedAssistant.id,
          agent: "general",
          abort: new AbortController().signal,
          extra: { promptOps: stubOps() },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      expect((yield* sessions.get(result.metadata.sessionId)).parentID).toBe(child.id)
    }),
  )

  it.instance("stops nesting once the default depth is reached", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const child = yield* sessions.createTask({ parentID: chat.id, title: "child" })
      const grandchild = yield* sessions.createTask({ parentID: child.id, title: "grandchild" })
      yield* jobs.start({
        id: child.id,
        type: "task",
        metadata: { sessionId: child.id, parentSessionId: chat.id },
        run: Effect.never,
      })
      yield* jobs.start({
        id: grandchild.id,
        type: "task",
        metadata: { sessionId: grandchild.id, parentSessionId: child.id },
        run: Effect.never,
      })
      const nestedAssistant = yield* sessions.updateMessage({
        ...assistant,
        id: MessageID.ascending(),
        parentID: MessageID.ascending(),
        sessionID: grandchild.id,
      })
      const tool = yield* TaskTool
      const def = yield* tool.init()

      const exit = yield* def
        .execute(
          {
            description: "inspect bug",
            prompt: "look into the cache key path",
            subagent_type: "general",
          },
          {
            sessionID: grandchild.id,
            messageID: nestedAssistant.id,
            agent: "general",
            abort: new AbortController().signal,
            extra: { promptOps: stubOps() },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )
        .pipe(Effect.exit)

      expect(Exit.isFailure(exit)).toBe(true)
      expect(yield* sessions.children(grandchild.id)).toHaveLength(0)
    }),
  )

  background.instance("tells the parent when a background subagent is classifier-rejected", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const notified: SessionPrompt.PromptInput[] = []

      const result = yield* def.execute(
        {
          description: "inspect bug",
          prompt: "look into the cache key path",
          subagent_type: "general",
        },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "build",
          abort: new AbortController().signal,
          extra: {
            promptOps: stubOps({
              text: "partial thought",
              error: new SessionV1.ContentFilterError({ message: "blocked by content filtering policy" }).toObject(),
              onNotify: (input) => notified.push(input),
            }),
          },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      const job = yield* jobs.wait({ id: result.metadata.sessionId, timeout: 5_000 })
      expect(job.info?.status).toBe("error")
      const injected = yield* pollWithTimeout(
        Effect.sync(() => notified.at(0)),
        "parent was never notified about the failed background task",
      )
      const text = injected.parts.flatMap((part) => (part.type === "text" ? [part.text] : [])).join("\n")
      expect(text).toContain(`state="error"`)
      expect(text).toContain("Background task failed")
      expect(text).toContain("ContentFilterError")
      expect(text).toContain("partial thought")
    }),
  )

  it.instance(
    "execute shapes child permissions for task, todowrite, and primary tools",
    () =>
      Effect.gen(function* () {
        const sessions = yield* Session.Service
        const { chat, assistant } = yield* seed()
        const tool = yield* TaskTool
        const def = yield* tool.init()
        let seen: SessionPrompt.PromptInput | undefined
        const promptOps = stubOps({ onPrompt: (input) => (seen = input) })

        const result = yield* def.execute(
          {
            description: "inspect bug",
            prompt: "look into the cache key path",
            subagent_type: "delegating",
          },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "build",
            abort: new AbortController().signal,
            extra: { promptOps },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )

        const child = yield* sessions.get(result.metadata.sessionId)
        expect(child.parentID).toBe(chat.id)
        expect(child.agent).toBe("delegating")
        expect(child.permission).toEqual([
          {
            permission: "todowrite",
            pattern: "*",
            action: "deny",
          },
          {
            permission: "bash",
            pattern: "*",
            action: "deny",
          },
          {
            permission: "read",
            pattern: "*",
            action: "deny",
          },
        ])
        expect(seen?.tools).toBeUndefined()
      }),
    {
      config: {
        agent: {
          delegating: {
            mode: "subagent",
            permission: {
              task: "allow",
            },
          },
        },
        experimental: {
          primary_tools: ["bash", "read"],
        },
      },
    },
  )

  background.instance("execute defaults to background without waiting for completion", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()

      const result = yield* def.execute(
        {
          description: "inspect bug",
          prompt: "look into the cache key path",
          subagent_type: "general",
        },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "build",
          abort: new AbortController().signal,
          extra: {
            promptOps: {
              ...stubOps(),
              prompt: () => Effect.never,
            } satisfies TaskPromptOps,
          },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      const job = yield* jobs.get(result.metadata.sessionId)
      expect(result.metadata.background).toBe(true)
      expect(result.output).toContain(`state="running"`)
      expect(job?.status).toBe("running")
    }),
  )

  background.instance("background task completion waits for running updates", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const first = defer<void>()
      const second = defer<void>()
      const updated = defer<SessionPrompt.PromptInput>()
      const injected = defer<SessionPrompt.PromptInput>()
      let prompts = 0
      const promptOps: TaskPromptOps = {
        ...stubOps({ onNotify: (input) => injected.resolve(input) }),
        prompt: (input) => {
          prompts++
          if (prompts === 1) return Effect.promise(() => first.promise).pipe(Effect.as(reply(input, "first done")))
          updated.resolve(input)
          return Effect.promise(() => second.promise).pipe(Effect.as(reply(input, "second done")))
        },
      }
      const context = {
        sessionID: chat.id,
        messageID: assistant.id,
        agent: "build",
        abort: new AbortController().signal,
        extra: { promptOps },
        messages: [],
        metadata: () => Effect.void,
        ask: () => Effect.void,
      }

      const started = yield* def.execute(
        {
          description: "inspect bug",
          prompt: "look into the cache key path",
          subagent_type: "general",
        },
        context,
      )
      const result = yield* def.execute(
        {
          description: "add investigation scope",
          prompt: "also inspect cancellation",
          subagent_type: "general",
          task_id: started.metadata.sessionId,
        },
        context,
      )

      expect(result.metadata.sessionId).toBe(started.metadata.sessionId)
      expect(result.metadata.background).toBe(true)
      expect(result.output).toContain("Background task updated")
      first.resolve()
      expect((yield* jobs.get(started.metadata.sessionId))?.status).toBe("running")
      expect((yield* Effect.promise(() => updated.promise)).parts).toEqual([
        { type: "text", text: "also inspect cancellation" },
        { type: "text", text: `\n<subagent-context parent-session-id="${context.sessionID}" />` },
      ])

      second.resolve()
      const waited = yield* jobs.wait({ id: started.metadata.sessionId, timeout: 1_000 })
      expect(waited.info?.status).toBe("completed")
      expect(waited.info?.output).toBe("second done")
      const notification = yield* Effect.promise(() => injected.promise)
      expect(notification.variant).toBe("xhigh")
      expect(notification.parts[0]?.type).toBe("text")
      if (notification.parts[0]?.type === "text") expect(notification.parts[0].text).toContain("second done")
    }),
  )

  background.instance("background tasks complete through the background job service", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()

      const result = yield* def.execute(
        {
          description: "inspect bug",
          prompt: "look into the cache key path",
          subagent_type: "general",
        },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "build",
          abort: new AbortController().signal,
          extra: { promptOps: stubOps({ text: "background done" }) },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      const waited = yield* jobs.wait({ id: result.metadata.sessionId, timeout: 1_000 })
      expect(waited.timedOut).toBe(false)
      expect(waited.info?.status).toBe("completed")
      expect(waited.info?.output).toBe("background done")
    }),
  )

  background.instance("nested completion atomically joins its active ancestor before root delivery", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const outerStarted = yield* Deferred.make<void>()
      const releaseOuter = yield* Deferred.make<void>()
      const ancestorAdmission = yield* Deferred.make<void>()
      const releaseAdmission = yield* Deferred.make<void>()
      const rootAdmission = yield* Deferred.make<void>()
      const notifications: Array<{ input: SessionPrompt.PromptInput; checkpoint: number }> = []
      let prompts = 0
      let outerSessionID: SessionID | undefined
      const promptOps: TaskPromptOps = {
        ...stubOps(),
        prompt: (input) =>
          Effect.gen(function* () {
            prompts++
            if (prompts === 1) {
              yield* Deferred.succeed(outerStarted, undefined)
              yield* Deferred.await(releaseOuter)
              return reply(input, "outer done")
            }
            return reply(input, "nested done")
          }),
        checkpoint: (sessionID) => Effect.succeed(sessionID === chat.id ? 11 : 7),
        admitNotification: (input, checkpoint) =>
          Effect.gen(function* () {
            notifications.push({ input, checkpoint })
            if (input.sessionID === outerSessionID) {
              yield* Deferred.succeed(ancestorAdmission, undefined)
              yield* Deferred.await(releaseAdmission)
              return Option.some(Effect.succeed(undefined))
            }
            yield* Deferred.succeed(rootAdmission, undefined)
            return Option.some(Effect.succeed(undefined))
          }),
      }
      const rootContext = {
        sessionID: chat.id,
        messageID: assistant.id,
        agent: "build",
        abort: new AbortController().signal,
        extra: { promptOps },
        messages: [],
        metadata: () => Effect.void,
        ask: () => Effect.void,
      }

      const outer = yield* def.execute(
        {
          description: "outer investigation",
          prompt: "coordinate the investigation",
          subagent_type: "general",
        },
        rootContext,
      )
      outerSessionID = outer.metadata.sessionId
      yield* Deferred.await(outerStarted)
      const nestedAssistant = yield* sessions.updateMessage({
        ...assistant,
        id: MessageID.ascending(),
        parentID: MessageID.ascending(),
        sessionID: outerSessionID,
        mode: "general",
        agent: "general",
      })

      const nested = yield* def.execute(
        {
          description: "nested investigation",
          prompt: "inspect the nested path",
          subagent_type: "general",
        },
        {
          ...rootContext,
          sessionID: outerSessionID,
          messageID: nestedAssistant.id,
          agent: "general",
        },
      )

      expect((yield* jobs.get(nested.metadata.sessionId))?.metadata?.ancestorSessionIds).toEqual([
        outerSessionID,
        chat.id,
      ])
      expect(yield* jobs.wait({ id: nested.metadata.sessionId, timeout: 0 })).toMatchObject({
        timedOut: true,
        info: { status: "running" },
      })
      yield* Deferred.succeed(releaseOuter, undefined)
      yield* Deferred.await(ancestorAdmission)
      expect((yield* jobs.get(outerSessionID))?.status).toBe("running")
      expect(notifications.length).toBeGreaterThanOrEqual(1)
      expect(notifications[0]?.input.sessionID).toBe(outerSessionID)
      expect(notifications[0]?.checkpoint).toBe(7)
      expect(notifications[0]?.input.parts[0]?.type).toBe("text")
      if (notifications[0]?.input.parts[0]?.type === "text")
        expect(notifications[0].input.parts[0].text).toContain("nested done")

      yield* Deferred.succeed(releaseAdmission, undefined)
      expect((yield* jobs.wait({ id: nested.metadata.sessionId })).info?.output).toBe("nested done")
      const outerJob = yield* jobs.wait({ id: outerSessionID })
      expect(outerJob.info?.output).toBe("nested done")
      yield* Deferred.await(rootAdmission)

      expect(notifications).toHaveLength(2)
      expect(notifications[1]?.input.sessionID).toBe(chat.id)
      expect(notifications[1]?.checkpoint).toBe(11)
      expect(notifications[1]?.input.parts[0]?.type).toBe("text")
      if (notifications[1]?.input.parts[0]?.type === "text")
        expect(notifications[1].input.parts[0].text).toContain("nested done")
    }),
  )

  background.instance("nested completion falls back to its immediate session after the parent job settles", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const child = yield* sessions.create({ parentID: chat.id, title: "Outer task", agent: "general" })
      const parentSettling = yield* Deferred.make<void>()
      const releaseParent = yield* Deferred.make<void>()
      yield* jobs.start({
        id: child.id,
        type: "task",
        metadata: { sessionId: child.id, parentSessionId: chat.id },
        notifyOnComplete: true,
        onComplete: () =>
          Deferred.succeed(parentSettling, undefined).pipe(Effect.andThen(Deferred.await(releaseParent))),
        run: Effect.succeed("outer done"),
      })
      yield* Deferred.await(parentSettling)
      const nestedAssistant = yield* sessions.updateMessage({
        ...assistant,
        id: MessageID.ascending(),
        parentID: MessageID.ascending(),
        sessionID: child.id,
        mode: "general",
        agent: "general",
      })
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const notifications: Array<{ input: SessionPrompt.PromptInput; checkpoint: number }> = []
      const promptOps: TaskPromptOps = {
        ...stubOps({ text: "nested done" }),
        checkpoint: (sessionID) => Effect.succeed(sessionID === chat.id ? 11 : 7),
        admitNotification: (input, checkpoint) =>
          Effect.sync(() => {
            notifications.push({ input, checkpoint })
          }).pipe(Effect.as(Option.some(Effect.succeed(undefined)))),
      }

      const result = yield* def.execute(
        {
          description: "inspect nested bug",
          prompt: "look into the nested cache path",
          subagent_type: "general",
        },
        {
          sessionID: child.id,
          messageID: nestedAssistant.id,
          agent: "general",
          abort: new AbortController().signal,
          extra: { promptOps },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      expect((yield* jobs.wait({ id: result.metadata.sessionId, timeout: 1_000 })).info?.status).toBe("completed")
      const notification = yield* pollWithTimeout(
        Effect.sync(() => notifications[0]),
        "parent session was never notified about the nested background task",
      )
      yield* Effect.yieldNow

      expect(notification.input.sessionID).toBe(child.id)
      expect(notification.checkpoint).toBe(7)
      expect(notifications).toHaveLength(1)
      expect(notification.input.parts[0]?.type).toBe("text")
      if (notification.input.parts[0]?.type === "text")
        expect(notification.input.parts[0].text).toContain("nested done")
      yield* Deferred.succeed(releaseParent, undefined)
      expect((yield* jobs.wait({ id: child.id })).info?.output).toBe("outer done")
    }),
  )

  background.instance("does not route completion through an unproven same-location parent", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const sessions = yield* Session.Service
      const { assistant } = yield* seed()
      const unrelated = yield* sessions.create({ title: "Unrelated session" })
      const forged = yield* sessions.create({ parentID: unrelated.id, title: "Forged child", agent: "general" })
      const nestedAssistant = yield* sessions.updateMessage({
        ...assistant,
        id: MessageID.ascending(),
        parentID: MessageID.ascending(),
        sessionID: forged.id,
        mode: "general",
        agent: "general",
      })
      const notifications: SessionPrompt.PromptInput[] = []
      const tool = yield* TaskTool
      const def = yield* tool.init()

      const result = yield* def.execute(
        {
          description: "inspect nested bug",
          prompt: "look into the nested cache path",
          subagent_type: "general",
        },
        {
          sessionID: forged.id,
          messageID: nestedAssistant.id,
          agent: "general",
          abort: new AbortController().signal,
          extra: {
            promptOps: {
              ...stubOps({ text: "nested done" }),
              admitNotification: (input: SessionPrompt.PromptInput) =>
                Effect.sync(() => notifications.push(input)).pipe(Effect.as(Option.some(Effect.succeed(undefined)))),
            },
          },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      expect((yield* jobs.wait({ id: result.metadata.sessionId })).info?.status).toBe("completed")
      const notification = yield* pollWithTimeout(
        Effect.sync(() => notifications[0]),
        "forged child was never notified",
      )
      expect(notification.sessionID).toBe(forged.id)
      expect(notification.sessionID).not.toBe(unrelated.id)
    }),
  )

  background.instance("does not route completion across a workspace boundary", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const sessions = yield* Session.Service
      const { assistant } = yield* seed()
      const foreign = yield* sessions.create({ title: "Foreign session" })
      const forged = yield* sessions.create({
        parentID: foreign.id,
        title: "Cross-workspace child",
        agent: "general",
        workspaceID: WorkspaceV2.ID.create(),
      })
      const release = yield* Deferred.make<void>()
      yield* jobs.start({
        id: forged.id,
        type: "task",
        metadata: { sessionId: forged.id, parentSessionId: foreign.id },
        run: Deferred.await(release).pipe(Effect.as("foreign done")),
      })
      const nestedAssistant = yield* sessions.updateMessage({
        ...assistant,
        id: MessageID.ascending(),
        parentID: MessageID.ascending(),
        sessionID: forged.id,
        mode: "general",
        agent: "general",
      })
      const notifications: SessionPrompt.PromptInput[] = []
      const tool = yield* TaskTool
      const def = yield* tool.init()

      const result = yield* def.execute(
        {
          description: "inspect nested bug",
          prompt: "look into the nested cache path",
          subagent_type: "general",
        },
        {
          sessionID: forged.id,
          messageID: nestedAssistant.id,
          agent: "general",
          abort: new AbortController().signal,
          extra: {
            promptOps: {
              ...stubOps({ text: "nested done" }),
              admitNotification: (input: SessionPrompt.PromptInput) =>
                Effect.sync(() => notifications.push(input)).pipe(Effect.as(Option.some(Effect.succeed(undefined)))),
            },
          },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      expect((yield* jobs.wait({ id: result.metadata.sessionId })).info?.status).toBe("completed")
      const notification = yield* pollWithTimeout(
        Effect.sync(() => notifications[0]),
        "cross-workspace child was never notified",
      )
      expect(notification.sessionID).toBe(forged.id)
      expect(notification.sessionID).not.toBe(foreign.id)
      yield* Deferred.succeed(release, undefined)
    }),
  )

  background.instance("background task completion waits for admission but not parent continuation", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const admissionStarted = yield* Deferred.make<void>()
      const releaseAdmission = yield* Deferred.make<void>()
      const continuationStarted = yield* Deferred.make<void>()
      const releaseContinuation = yield* Deferred.make<void>()

      const result = yield* def.execute(
        {
          description: "inspect bug",
          prompt: "look into the cache key path",
          subagent_type: "general",
        },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "build",
          abort: new AbortController().signal,
          extra: {
            promptOps: {
              ...stubOps({ text: "background done" }),
              prompt: (input) => Effect.succeed(reply(input, "background done")),
              admitNotification: () =>
                Deferred.succeed(admissionStarted, undefined).pipe(
                  Effect.andThen(Deferred.await(releaseAdmission)),
                  Effect.as(
                    Option.some(
                      Deferred.succeed(continuationStarted, undefined).pipe(
                        Effect.andThen(Deferred.await(releaseContinuation)),
                        Effect.as(undefined),
                      ),
                    ),
                  ),
                ),
            } satisfies TaskPromptOps,
          },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      yield* Deferred.await(admissionStarted)
      expect(yield* jobs.wait({ id: result.metadata.sessionId, timeout: 0 })).toMatchObject({
        timedOut: true,
        info: { status: "running" },
      })
      yield* Deferred.succeed(releaseAdmission, undefined)
      yield* Deferred.await(continuationStarted)
      expect((yield* jobs.wait({ id: result.metadata.sessionId })).info?.status).toBe("completed")
      yield* Deferred.succeed(releaseContinuation, undefined)
    }),
  )

  background.instance("removing the parent session cancels running background tasks", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()

      const result = yield* def.execute(
        {
          description: "inspect bug",
          prompt: "look into the cache key path",
          subagent_type: "general",
        },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "build",
          abort: new AbortController().signal,
          extra: {
            promptOps: {
              ...stubOps(),
              prompt: () => Effect.never,
            } satisfies TaskPromptOps,
          },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      yield* sessions.remove(chat.id)
      const waited = yield* jobs.wait({ id: result.metadata.sessionId, timeout: 1_000 })
      expect(waited.timedOut).toBe(false)
      expect(waited.info?.status).toBe("cancelled")
    }),
  )

  background.instance("removing the child task session cancels its running background task", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()

      const result = yield* def.execute(
        {
          description: "inspect bug",
          prompt: "look into the cache key path",
          subagent_type: "general",
        },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "build",
          abort: new AbortController().signal,
          extra: {
            promptOps: {
              ...stubOps(),
              prompt: () => Effect.never,
            } satisfies TaskPromptOps,
          },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      yield* sessions.remove(result.metadata.sessionId)
      const waited = yield* jobs.wait({ id: result.metadata.sessionId, timeout: 1_000 })
      expect(waited.timedOut).toBe(false)
      expect(waited.info?.status).toBe("cancelled")
    }),
  )

  background.instance("cancelling the parent run cancels running background tasks", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const runState = yield* SessionRunState.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()

      const result = yield* def.execute(
        {
          description: "inspect bug",
          prompt: "look into the cache key path",
          subagent_type: "general",
        },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "build",
          abort: new AbortController().signal,
          extra: {
            promptOps: {
              ...stubOps(),
              prompt: () => Effect.never,
            } satisfies TaskPromptOps,
          },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      yield* runState.cancel(chat.id)
      const waited = yield* jobs.wait({ id: result.metadata.sessionId, timeout: 1_000 })
      expect(waited.timedOut).toBe(false)
      expect(waited.info?.status).toBe("cancelled")
    }),
  )

  background.instance("cancels a running descendant after its parent Task completes", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const runState = yield* SessionRunState.Service
      const parent = SessionID.make("ses_parent")
      const child = SessionID.make("ses_child")
      const grandchild = SessionID.make("ses_grandchild")
      yield* jobs.start({
        id: child,
        type: "task",
        metadata: { parentSessionId: parent, sessionId: child, ancestorSessionIds: [parent] },
        run: Effect.succeed("done"),
      })
      yield* jobs.wait({ id: child })
      yield* jobs.start({
        id: grandchild,
        type: "task",
        metadata: { parentSessionId: child, sessionId: grandchild, ancestorSessionIds: [child, parent] },
        run: Effect.never,
      })

      yield* runState.cancel(parent)

      expect((yield* jobs.get(grandchild))?.status).toBe("cancelled")
    }),
  )

  background.instance("does not admit a task job after its parent checkpoint is cancelled", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const runState = yield* SessionRunState.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const checkpoint = yield* runState.checkpoint(chat.id)
      const admissionReady = yield* Deferred.make<void>()
      const releaseAdmission = yield* Deferred.make<void>()
      let prompts = 0
      const promptOps: TaskPromptOps = {
        ...stubOps(),
        admitIfCurrent: (effect) =>
          Deferred.succeed(admissionReady, undefined).pipe(
            Effect.andThen(Deferred.await(releaseAdmission)),
            Effect.andThen(runState.admitIfCurrent(chat.id, checkpoint, effect)),
          ),
        prompt: () =>
          Effect.sync(() => {
            prompts += 1
          }).pipe(Effect.andThen(Effect.never)),
      }
      const execute = (ops: TaskPromptOps) =>
        def.execute(
          {
            description: "inspect bug",
            prompt: "look into the cache key path",
            subagent_type: "general",
          },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "build",
            abort: new AbortController().signal,
            extra: { promptOps: ops },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )
      const stale = yield* execute(promptOps).pipe(Effect.forkChild)

      yield* Deferred.await(admissionReady)
      yield* runState.cancel(chat.id)
      yield* Deferred.succeed(releaseAdmission, undefined)
      expect((yield* Fiber.await(stale))._tag).toBe("Failure")
      expect(yield* jobs.list()).toHaveLength(0)
      expect(prompts).toBe(0)

      const current = yield* runState.checkpoint(chat.id)
      const replacement = yield* execute({
        ...stubOps(),
        admitIfCurrent: (effect) => runState.admitIfCurrent(chat.id, current, effect),
        prompt: () => Effect.never,
      })
      expect((yield* jobs.get(replacement.metadata.sessionId))?.status).toBe("running")
      yield* runState.cancel(chat.id)
      expect((yield* jobs.get(replacement.metadata.sessionId))?.status).toBe("cancelled")
    }),
  )

  background.instance("does not admit a task job after its child is removed", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const runState = yield* SessionRunState.Service
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const checkpoint = yield* runState.checkpoint(chat.id)
      const childCreated = yield* Deferred.make<SessionID>()
      const releaseCreation = yield* Deferred.make<void>()
      const released: SessionID[] = []
      let prompts = 0
      const promptOps: TaskPromptOps = {
        ...stubOps(),
        checkpoint: (sessionID) => runState.checkpoint(sessionID),
        retain: (sessionID, checkpoint) =>
          runState
            .retain(sessionID, checkpoint)
            .pipe(
              Effect.map(
                Option.map((release) =>
                  Effect.sync(() => released.push(sessionID)).pipe(Effect.andThen(release), Effect.asVoid),
                ),
              ),
            ),
        admitIfCurrent: (effect) =>
          runState.admitIfCurrent(chat.id, checkpoint, effect).pipe(
            Effect.tap((admitted) => {
              if (Option.isNone(admitted)) return Effect.void
              return sessions.children(chat.id).pipe(
                Effect.flatMap((children) => {
                  const child = children[0]
                  if (!child) return Effect.die("task child was not created")
                  return Deferred.succeed(childCreated, child.id).pipe(Effect.andThen(Deferred.await(releaseCreation)))
                }),
              )
            }),
          ),
        admitChild: (childID, effect) =>
          runState.admit([{ sessionID: chat.id, checkpoint }, { sessionID: childID }], effect),
        prompt: () =>
          Effect.sync(() => {
            prompts += 1
          }).pipe(Effect.andThen(Effect.never)),
      }
      const execute = yield* def
        .execute(
          {
            description: "inspect bug",
            prompt: "look into the cache key path",
            subagent_type: "general",
          },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "build",
            abort: new AbortController().signal,
            extra: { promptOps },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )
        .pipe(Effect.forkChild)

      yield* sessions.remove(yield* Deferred.await(childCreated))
      yield* Deferred.succeed(releaseCreation, undefined)

      expect(Exit.hasInterrupts(yield* Fiber.await(execute))).toBe(true)
      expect(yield* jobs.list()).toEqual([])
      expect(released).toEqual([chat.id])
      expect(prompts).toBe(0)
    }),
  )

  background.instance("releases acquired leases when later ancestor retention is interrupted", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const parent = yield* sessions.createTask({ parentID: chat.id, title: "Parent task", agent: "general" })
      yield* jobs.start({
        id: parent.id,
        type: "task",
        metadata: { parentSessionId: chat.id, sessionId: parent.id },
        run: Effect.never,
      })
      const nestedAssistant = yield* sessions.updateMessage({
        ...assistant,
        id: MessageID.ascending(),
        parentID: MessageID.ascending(),
        sessionID: parent.id,
        mode: "general",
        agent: "general",
      })
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const released: SessionID[] = []
      let admissions = 0
      const exit = yield* def
        .execute(
          {
            description: "inspect nested bug",
            prompt: "look into the nested cache path",
            subagent_type: "general",
          },
          {
            sessionID: parent.id,
            messageID: nestedAssistant.id,
            agent: "general",
            abort: new AbortController().signal,
            extra: {
              promptOps: {
                ...stubOps(),
                checkpoint: () => Effect.succeed(7),
                retain: (sessionID) =>
                  sessionID === parent.id
                    ? Effect.succeed(Option.some(Effect.sync(() => released.push(sessionID)).pipe(Effect.asVoid)))
                    : Effect.interrupt,
                admitChild: (_sessionID, effect) =>
                  Effect.sync(() => {
                    admissions += 1
                  }).pipe(Effect.andThen(effect), Effect.map(Option.some)),
              } satisfies TaskPromptOps,
            },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )
        .pipe(Effect.exit)

      expect(Exit.hasInterrupts(exit)).toBe(true)
      expect(released).toEqual([parent.id])
      expect(admissions).toBe(0)
      expect((yield* jobs.list()).map((job) => job.id)).toEqual([parent.id])
      yield* jobs.cancel(parent.id)
    }),
  )

  background.instance("releases every retained lease once when completion cleanup is interrupted", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const parent = yield* sessions.createTask({ parentID: chat.id, title: "Parent task", agent: "general" })
      const parentSettling = yield* Deferred.make<void>()
      const releaseParent = yield* Deferred.make<void>()
      yield* jobs.start({
        id: parent.id,
        type: "task",
        metadata: { parentSessionId: chat.id, sessionId: parent.id },
        notifyOnComplete: true,
        awaitOnComplete: true,
        onComplete: () =>
          Deferred.succeed(parentSettling, undefined).pipe(Effect.andThen(Deferred.await(releaseParent))),
        run: Effect.succeed("parent done"),
      })
      yield* Deferred.await(parentSettling)

      const nestedAssistant = yield* sessions.updateMessage({
        ...assistant,
        id: MessageID.ascending(),
        parentID: MessageID.ascending(),
        sessionID: parent.id,
        mode: "general",
        agent: "general",
      })
      const releasing = yield* Deferred.make<Fiber.Fiber<unknown, unknown>>()
      const continueRelease = yield* Deferred.make<void>()
      const allReleased = yield* Deferred.make<void>()
      const retained: SessionID[] = []
      const released: SessionID[] = []
      const promptOps: TaskPromptOps = {
        ...stubOps({ text: "nested done" }),
        checkpoint: () => Effect.succeed(7),
        retain: (sessionID) =>
          Effect.sync(() => retained.push(sessionID)).pipe(
            Effect.as(
              Option.some(
                sessionID === parent.id
                  ? Effect.withFiber((fiber) =>
                      Effect.sync(() => released.push(sessionID)).pipe(
                        Effect.andThen(Deferred.succeed(releasing, fiber)),
                        Effect.andThen(Deferred.await(continueRelease)),
                      ),
                    )
                  : Effect.sync(() => released.push(sessionID)).pipe(
                      Effect.andThen(Deferred.succeed(allReleased, undefined)),
                      Effect.asVoid,
                    ),
              ),
            ),
          ),
      }
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const result = yield* def.execute(
        {
          description: "inspect nested bug",
          prompt: "look into the nested cache path",
          subagent_type: "general",
        },
        {
          sessionID: parent.id,
          messageID: nestedAssistant.id,
          agent: "general",
          abort: new AbortController().signal,
          extra: { promptOps },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      expect(retained).toEqual([parent.id, chat.id])
      const releasingFiber = yield* Deferred.await(releasing).pipe(Effect.timeout("1 second"))
      releasingFiber.interruptUnsafe()
      yield* Effect.yieldNow
      yield* Deferred.succeed(continueRelease, undefined)
      yield* Deferred.await(allReleased).pipe(Effect.timeout("1 second"))

      expect((yield* jobs.wait({ id: result.metadata.sessionId })).info).toMatchObject({
        status: "completed",
        output: "nested done",
      })
      expect(released).toEqual([parent.id, chat.id])
      yield* jobs.cancel(result.metadata.sessionId)
      expect(released).toEqual([parent.id, chat.id])

      yield* Deferred.succeed(releaseParent, undefined)
      expect((yield* jobs.wait({ id: parent.id })).info?.output).toBe("parent done")
    }),
  )

  it.instance("cancelling a child run cancels its own pre-runner task job", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const runState = yield* SessionRunState.Service
      const sessions = yield* Session.Service
      const { chat } = yield* seed()
      const child = yield* sessions.create({ parentID: chat.id, title: "child" })

      yield* jobs.start({
        id: child.id,
        type: "task",
        metadata: { parentSessionId: chat.id, sessionId: child.id },
        run: Effect.never,
      })

      yield* runState.cancel(child.id)

      expect((yield* jobs.get(child.id))?.status).toBe("cancelled")
    }),
  )

  it.instance("cancelling a parent run recursively cancels descendant background tasks", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const runState = yield* SessionRunState.Service
      const sessions = yield* Session.Service
      const { chat } = yield* seed()
      const child = yield* sessions.create({ parentID: chat.id, title: "child" })
      const grandchild = yield* sessions.create({ parentID: child.id, title: "grandchild" })

      yield* jobs.start({
        id: child.id,
        type: "task",
        metadata: { parentSessionId: chat.id, sessionId: child.id },
        run: Effect.never,
      })
      yield* jobs.start({
        id: grandchild.id,
        type: "task",
        metadata: { parentSessionId: child.id, sessionId: grandchild.id },
        run: Effect.never,
      })

      yield* runState.cancel(chat.id)

      expect((yield* jobs.get(child.id))?.status).toBe("cancelled")
      expect((yield* jobs.get(grandchild.id))?.status).toBe("cancelled")
    }),
  )
})
