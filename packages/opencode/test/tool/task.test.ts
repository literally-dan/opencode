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
import { SessionActivity } from "@/session/activity"
import { SessionTaskState } from "@/session/task-state"
import { Permission } from "@/permission"
import { Provider } from "@/provider/provider"
import { Question } from "@/question"

import { TaskTool, type TaskPromptOps } from "../../src/tool/task"
import { TaskControlTool } from "@/tool/task_control"
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
      SessionActivity.node,
      SessionStatus.node,
      SessionTaskState.node,
      Permission.node,
      Provider.node,
      Question.node,
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
    admitNotification: (input, _checkpoint, _source, onAdmitted) =>
      Effect.sync(() => opts?.onNotify?.(input)).pipe(
        Effect.andThen(onAdmitted ?? Effect.void),
        Effect.as(Option.some(Effect.succeed(undefined))),
      ),
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

  it.instance(
    "execute uses an explicit model override",
    () =>
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
    { config: { provider: { other: { models: { "nested/model": { name: "Nested model" } } } } } },
  )

  it.instance("execute rejects an unknown model before creating or resuming a child", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const existing = yield* sessions.create({ parentID: chat.id, title: "Existing child", agent: "general" })
      const tool = yield* TaskTool
      const def = yield* tool.init()
      let prompts = 0
      const context = {
        sessionID: chat.id,
        messageID: assistant.id,
        agent: "build",
        abort: new AbortController().signal,
        extra: { promptOps: stubOps({ onPrompt: () => prompts++ }) },
        messages: [],
        metadata: () => Effect.void,
        ask: () => Effect.void,
      }
      const params = {
        description: "inspect bug",
        prompt: "look into the cache key path",
        subagent_type: "general",
        model: "missing/model",
      }

      expect(failureMessage(yield* def.execute(params, context).pipe(Effect.exit))).toStartWith(
        "Model not found: missing/model.",
      )
      expect(
        failureMessage(yield* def.execute({ ...params, task_id: existing.id }, context).pipe(Effect.exit)),
      ).toStartWith("Model not found: missing/model.")
      expect(yield* sessions.children(chat.id)).toEqual([existing])
      expect((yield* sessions.get(existing.id)).taskParentID).toBeUndefined()
      expect(yield* jobs.list()).toEqual([])
      expect(prompts).toBe(0)
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
        provider: { override: { models: { model: { name: "Override model" } } } },
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

  it.instance(
    "counts a root that moved to another location toward subagent_depth",
    () =>
      Effect.gen(function* () {
        const sessions = yield* Session.Service
        const { chat, assistant } = yield* seed()
        const child = yield* sessions.createTask({ parentID: chat.id, title: "child", agent: "general" })
        yield* sessions.setWorkspace({ sessionID: chat.id, workspaceID: WorkspaceV2.ID.create() })
        const nestedAssistant = yield* sessions.updateMessage({
          ...assistant,
          id: MessageID.ascending(),
          parentID: MessageID.ascending(),
          sessionID: child.id,
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
          .pipe(Effect.exit)

        expect(failureMessage(exit)).toContain("Subagent depth limit reached (1)")
        expect(yield* sessions.children(child.id)).toHaveLength(0)
      }),
    { config: { subagent_depth: 1 } },
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

  background.instance("caps legacy task descendants to compatibility history access", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const tasks = yield* SessionTaskState.Service
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const legacy = yield* sessions.createTask({
        parentID: chat.id,
        title: "Legacy task",
        agent: "general",
        model: { providerID: ref.providerID, id: ref.modelID },
      })
      const legacyAssistant = yield* sessions.updateMessage({
        ...assistant,
        id: MessageID.ascending(),
        parentID: MessageID.ascending(),
        sessionID: legacy.id,
        mode: "general",
        agent: "general",
      })
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const promptOps: TaskPromptOps = { ...stubOps(), prompt: () => Effect.never }

      const result = yield* def.execute(
        {
          description: "legacy nested investigation",
          prompt: "do not amplify ancestor access",
          subagent_type: "general",
          ancestor_access: "all",
        },
        {
          sessionID: legacy.id,
          messageID: legacyAssistant.id,
          agent: "general",
          abort: new AbortController().signal,
          extra: { promptOps, bypassAgentCheck: true },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      expect(yield* tasks.get(legacy.id)).toBeUndefined()
      expect((yield* tasks.get(result.metadata.sessionId))?.ancestor_access).toBe("history")
      yield* jobs.cancel(result.metadata.sessionId)
    }),
  )

  background.instance("blind tasks omit ancestor context and preserve access on resume", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const tasks = yield* SessionTaskState.Service
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const prompted = yield* Deferred.make<SessionPrompt.PromptInput>()
      const promptOps: TaskPromptOps = {
        ...stubOps(),
        prompt: (input) => Deferred.succeed(prompted, input).pipe(Effect.andThen(Effect.never)),
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
          description: "blind investigation",
          prompt: "inspect only the supplied context",
          subagent_type: "general",
          ancestor_access: "none",
        },
        context,
      )
      expect((yield* Deferred.await(prompted)).parts).toEqual([
        { type: "text", text: "inspect only the supplied context" },
      ])
      expect((yield* tasks.get(started.metadata.sessionId))?.ancestor_access).toBe("none")

      yield* def.execute(
        {
          description: "continue blind investigation",
          prompt: "also inspect the fallback path",
          subagent_type: "general",
          task_id: started.metadata.sessionId,
        },
        context,
      )
      expect((yield* tasks.get(started.metadata.sessionId))?.ancestor_access).toBe("none")

      const nestedAssistant = yield* sessions.updateMessage({
        ...assistant,
        id: MessageID.ascending(),
        parentID: MessageID.ascending(),
        sessionID: started.metadata.sessionId,
      })
      const nested = yield* def.execute(
        {
          description: "nested blind investigation",
          prompt: "inspect without amplified access",
          subagent_type: "general",
          ancestor_access: "all",
        },
        {
          ...context,
          sessionID: started.metadata.sessionId,
          messageID: nestedAssistant.id,
          agent: "general",
          extra: { promptOps, bypassAgentCheck: true },
        },
      )
      expect((yield* tasks.get(nested.metadata.sessionId))?.ancestor_access).toBe("none")
      yield* Effect.all([jobs.cancel(nested.metadata.sessionId), jobs.cancel(started.metadata.sessionId)], {
        concurrency: "unbounded",
        discard: true,
      })
    }),
  )

  background.instance("resume waits for prior result delivery before starting a new generation", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const tasks = yield* SessionTaskState.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const deliveryStarted = yield* Deferred.make<void>()
      const releaseDelivery = yield* Deferred.make<void>()
      const secondStarted = yield* Deferred.make<void>()
      const promptOps: TaskPromptOps = {
        ...stubOps(),
        prompt: (input) => {
          const text = input.parts.find((part) => part.type === "text")
          if (text?.type === "text" && text.text.startsWith("first")) return Effect.succeed(reply(input, "first done"))
          return Deferred.succeed(secondStarted, undefined).pipe(Effect.andThen(Effect.never))
        },
        admitNotification: () =>
          Deferred.succeed(deliveryStarted, undefined).pipe(
            Effect.andThen(Deferred.await(releaseDelivery)),
            Effect.as(Option.some(Effect.succeed(undefined))),
          ),
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
        { description: "first generation", prompt: "first run", subagent_type: "general" },
        context,
      )
      yield* Deferred.await(deliveryStarted)
      const resumed = yield* def
        .execute(
          {
            description: "second generation",
            prompt: "second run",
            subagent_type: "general",
            task_id: started.metadata.sessionId,
          },
          context,
        )
        .pipe(Effect.forkChild)
      yield* Effect.yieldNow
      expect(resumed.pollUnsafe()).toBeUndefined()

      yield* Deferred.succeed(releaseDelivery, undefined)
      expect((yield* Fiber.join(resumed)).metadata.sessionId).toBe(started.metadata.sessionId)
      yield* Deferred.await(secondStarted)
      expect(yield* tasks.get(started.metadata.sessionId)).toMatchObject({
        generation: 2,
        status: "running",
        delivery_session_id: null,
      })
      yield* jobs.cancel(started.metadata.sessionId)
    }),
  )

  background.instance(
    "resume can be interrupted while its child result waits for another ancestor turn",
    () =>
      Effect.gen(function* () {
        const jobs = yield* BackgroundJob.Service
        const sessions = yield* Session.Service
        const tasks = yield* SessionTaskState.Service
        const { chat, assistant } = yield* seed()
        const outer = yield* sessions.createTask({ parentID: chat.id, title: "Outer task", agent: "general" })
        const releaseOuter = yield* Deferred.make<void>()
        yield* jobs.start({
          id: outer.id,
          type: "task",
          metadata: { parentSessionId: chat.id, sessionId: outer.id },
          run: Deferred.await(releaseOuter).pipe(Effect.as("outer done")),
        })
        const middle = yield* sessions.createTask({ parentID: outer.id, title: "Middle task", agent: "general" })
        const middleAssistant = yield* sessions.updateMessage({
          ...assistant,
          id: MessageID.ascending(),
          parentID: MessageID.ascending(),
          sessionID: middle.id,
          mode: "general",
          agent: "general",
        })
        const tool = yield* TaskTool
        const def = yield* tool.init()
        const notifications: SessionPrompt.PromptInput[] = []
        const promptOps: TaskPromptOps = {
          ...stubOps({ text: "nested done" }),
          admitNotification: (input, _checkpoint, _source, onAdmitted) =>
            Effect.sync(() => notifications.push(input)).pipe(
              Effect.andThen(onAdmitted ?? Effect.void),
              Effect.as(Option.some(Effect.succeed(undefined))),
            ),
        }
        const context = {
          sessionID: middle.id,
          messageID: middleAssistant.id,
          agent: "general",
          abort: new AbortController().signal,
          extra: { promptOps },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        }

        const nested = yield* def.execute(
          { description: "nested investigation", prompt: "inspect the nested path", subagent_type: "general" },
          context,
        )
        yield* pollWithTimeout(
          Effect.gen(function* () {
            const lifecycle = yield* tasks.get(nested.metadata.sessionId)
            const job = yield* jobs.get(nested.metadata.sessionId)
            if (lifecycle?.status === "completed" && job?.status === "running") return true
          }),
          "nested task never started finalizing",
        )
        const resumed = yield* def
          .execute(
            {
              description: "continue nested investigation",
              prompt: "continue the nested path",
              subagent_type: "general",
              task_id: nested.metadata.sessionId,
            },
            context,
          )
          .pipe(Effect.forkChild)
        yield* Effect.sleep("20 millis")
        expect(resumed.pollUnsafe()).toBeUndefined()

        const interrupted = yield* Fiber.interrupt(resumed).pipe(
          Effect.timeout("2 seconds"),
          Effect.ensuring(Deferred.succeed(releaseOuter, undefined)),
        )
        expect(interrupted).toBeUndefined()
        expect(Exit.hasInterrupts(yield* Fiber.await(resumed))).toBe(true)
        expect((yield* jobs.wait({ id: nested.metadata.sessionId, timeout: 2_000 })).info?.status).toBe("completed")
        expect(yield* tasks.get(nested.metadata.sessionId)).toMatchObject({
          generation: 1,
          status: "completed",
          delivery_session_id: outer.id,
        })
        expect(notifications[0]?.sessionID).toBe(outer.id)
      }),
    { config: { subagent_depth: 3 } },
  )

  background.instance("background tasks complete through the background job service", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const tasks = yield* SessionTaskState.Service
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
      expect(yield* tasks.get(result.metadata.sessionId)).toMatchObject({
        status: "completed",
        delivery_session_id: chat.id,
        time_completed: expect.any(Number),
        time_delivered: expect.any(Number),
      })
    }),
  )

  background.instance("a rejected resume keeps the previous task state", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const tasks = yield* SessionTaskState.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const context = {
        sessionID: chat.id,
        messageID: assistant.id,
        agent: "build",
        abort: new AbortController().signal,
        extra: { promptOps: stubOps({ text: "first done" }) },
        messages: [],
        metadata: () => Effect.void,
        ask: () => Effect.void,
      }

      const started = yield* def.execute(
        { description: "first generation", prompt: "first run", subagent_type: "general" },
        context,
      )
      expect((yield* jobs.wait({ id: started.metadata.sessionId, timeout: 1_000 })).info?.status).toBe("completed")
      const previous = yield* tasks.get(started.metadata.sessionId)
      expect(previous).toMatchObject({ generation: 1, status: "completed", delivery_session_id: chat.id })

      const exit = yield* def
        .execute(
          {
            description: "second generation",
            prompt: "second run",
            subagent_type: "general",
            task_id: started.metadata.sessionId,
          },
          {
            ...context,
            extra: {
              promptOps: { ...stubOps(), admitChild: () => Effect.succeed(Option.none()) } satisfies TaskPromptOps,
            },
          },
        )
        .pipe(Effect.exit)

      expect(Exit.hasInterrupts(exit)).toBe(true)
      expect(yield* tasks.get(started.metadata.sessionId)).toEqual(previous)
    }),
  )

  background.instance("cancelling a task job while it forwards a result also stops the forwarded turn", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const outer = yield* sessions.createTask({ parentID: chat.id, title: "Outer task", agent: "general" })
      const releaseOuter = yield* Deferred.make<void>()
      yield* jobs.start({
        id: outer.id,
        type: "task",
        metadata: { parentSessionId: chat.id, sessionId: outer.id },
        run: Deferred.await(releaseOuter).pipe(Effect.as("outer done")),
      })
      const nestedAssistant = yield* sessions.updateMessage({
        ...assistant,
        id: MessageID.ascending(),
        parentID: MessageID.ascending(),
        sessionID: outer.id,
        mode: "general",
        agent: "general",
      })
      const continuationStarted = yield* Deferred.make<void>()
      const cancelled: SessionID[] = []
      const promptOps: TaskPromptOps = {
        ...stubOps({ text: "nested done" }),
        cancel: (sessionID) => Effect.sync(() => cancelled.push(sessionID)),
        // The continuation stands for runNotification waiting for the outer Session runner.
        admitNotification: () =>
          Effect.succeed(
            Option.some(Deferred.succeed(continuationStarted, undefined).pipe(Effect.andThen(Effect.never))),
          ),
      }
      const tool = yield* TaskTool
      const def = yield* tool.init()

      yield* def.execute(
        { description: "nested investigation", prompt: "inspect the nested path", subagent_type: "general" },
        {
          sessionID: outer.id,
          messageID: nestedAssistant.id,
          agent: "general",
          abort: new AbortController().signal,
          extra: { promptOps },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )
      yield* Deferred.succeed(releaseOuter, undefined)
      yield* Deferred.await(continuationStarted).pipe(Effect.timeout("2 seconds"))
      expect(cancelled).toEqual([])

      expect((yield* jobs.cancel(outer.id))?.status).toBe("cancelled")
      yield* pollWithTimeout(
        Effect.sync(() => (cancelled.includes(outer.id) ? true : undefined)),
        "cancelling the outer job did not stop its forwarded turn",
      )
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
              // The outer turn answers the nested result without text.
              return Option.some(Effect.succeed(reply(input, "")))
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
      // The continuation reply has no text, so the outer job keeps its own reply.
      const outerJob = yield* jobs.wait({ id: outerSessionID })
      expect(outerJob.info?.output).toBe("outer done")
      yield* Deferred.await(rootAdmission)

      expect(notifications).toHaveLength(2)
      expect(notifications[1]?.input.sessionID).toBe(chat.id)
      expect(notifications[1]?.checkpoint).toBe(11)
      expect(notifications[1]?.input.parts[0]?.type).toBe("text")
      if (notifications[1]?.input.parts[0]?.type === "text") {
        expect(notifications[1].input.parts[0].text).toContain("outer done")
        expect(notifications[1].input.parts[0].text).not.toContain("nested done")
      }
    }),
  )

  background.instance("an intermediate task fails when its continuation turn fails", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const outer = yield* sessions.createTask({ parentID: chat.id, title: "Outer task", agent: "general" })
      const releaseOuter = yield* Deferred.make<void>()
      yield* jobs.start({
        id: outer.id,
        type: "task",
        metadata: { parentSessionId: chat.id, sessionId: outer.id },
        run: Deferred.await(releaseOuter).pipe(Effect.as("outer interim reply")),
      })
      const nestedAssistant = yield* sessions.updateMessage({
        ...assistant,
        id: MessageID.ascending(),
        parentID: MessageID.ascending(),
        sessionID: outer.id,
        mode: "general",
        agent: "general",
      })
      const promptOps: TaskPromptOps = {
        ...stubOps({ text: "nested done" }),
        admitNotification: (input) =>
          Effect.succeed(
            Option.some(
              Effect.succeed(
                reply(
                  input,
                  "partial coordinator text",
                  new SessionV1.APIError({ message: "Coordinator provider failed", isRetryable: false }).toObject(),
                ),
              ),
            ),
          ),
      }
      const tool = yield* TaskTool
      const def = yield* tool.init()

      yield* def.execute(
        { description: "nested investigation", prompt: "inspect the nested path", subagent_type: "general" },
        {
          sessionID: outer.id,
          messageID: nestedAssistant.id,
          agent: "general",
          abort: new AbortController().signal,
          extra: { promptOps },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )
      yield* Deferred.succeed(releaseOuter, undefined)

      const failure = (yield* jobs.wait({ id: outer.id, timeout: 2_000 })).info
      expect(failure?.status).toBe("error")
      expect(failure?.error).toContain(`Subagent failed (task_id: ${outer.id}):`)
      expect(failure?.error).toContain("The subagent stopped with APIError: Coordinator provider failed")
      expect(failure?.error).toContain("partial coordinator text")
    }),
  )

  background.instance("an intermediate task keeps its reply and labels a nested result it could not receive", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const outer = yield* sessions.createTask({ parentID: chat.id, title: "Outer task", agent: "general" })
      const releaseOuter = yield* Deferred.make<void>()
      yield* jobs.start({
        id: outer.id,
        type: "task",
        metadata: { parentSessionId: chat.id, sessionId: outer.id },
        run: Deferred.await(releaseOuter).pipe(Effect.as("outer interim reply")),
      })
      const nestedAssistant = yield* sessions.updateMessage({
        ...assistant,
        id: MessageID.ascending(),
        parentID: MessageID.ascending(),
        sessionID: outer.id,
        mode: "general",
        agent: "general",
      })
      const tool = yield* TaskTool
      const def = yield* tool.init()

      const nested = yield* def.execute(
        { description: "nested investigation", prompt: "inspect the nested path", subagent_type: "general" },
        {
          sessionID: outer.id,
          messageID: nestedAssistant.id,
          agent: "general",
          abort: new AbortController().signal,
          // The outer Session cannot take notifications, for example because a revert is staged.
          extra: {
            promptOps: {
              ...stubOps({ text: "nested raw output" }),
              admitNotification: () => Effect.succeed(Option.none()),
            },
          },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )
      yield* Deferred.succeed(releaseOuter, undefined)

      expect((yield* jobs.wait({ id: nested.metadata.sessionId, timeout: 2_000 })).info?.status).toBe("completed")
      const output = (yield* jobs.wait({ id: outer.id, timeout: 2_000 })).info?.output ?? ""
      expect(output.startsWith("outer interim reply")).toBe(true)
      expect(output).toContain(`<task id="${nested.metadata.sessionId}" state="completed">`)
      expect(output).toContain("nested raw output")
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

  background.instance("a coordinator task receives its worker result after its own turn ends", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const sessions = yield* Session.Service
      const tasks = yield* SessionTaskState.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const coordinatorStarted = yield* Deferred.make<void>()
      const workerLaunched = yield* Deferred.make<void>()
      const coordinatorTurnEnded = yield* Deferred.make<void>()
      const releaseWorker = yield* Deferred.make<void>()
      const notifications: SessionPrompt.PromptInput[] = []
      let coordinatorID: SessionID | undefined
      let continuations = 0
      const promptOps: TaskPromptOps = {
        ...stubOps(),
        checkpoint: (sessionID) => Effect.succeed(sessionID === chat.id ? 11 : 7),
        prompt: (input) =>
          Effect.gen(function* () {
            const text = input.parts.find((part) => part.type === "text")
            if (text?.type === "text" && text.text === "inspect the worker path") {
              yield* Deferred.await(releaseWorker)
              return reply(input, "worker raw output")
            }
            yield* Deferred.succeed(coordinatorStarted, undefined)
            yield* Deferred.await(workerLaunched)
            yield* Deferred.succeed(coordinatorTurnEnded, undefined)
            return reply(input, "coordinator is waiting for the worker")
          }),
        admitNotification: (input, _checkpoint, _source, onAdmitted) =>
          Effect.gen(function* () {
            notifications.push(input)
            if (onAdmitted) yield* onAdmitted
            if (input.sessionID !== coordinatorID) return Option.some(Effect.succeed(undefined))
            return Option.some(
              Effect.sync(() => {
                continuations++
                return reply(input, "coordinator final reply")
              }),
            )
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

      const coordinator = yield* def.execute(
        { description: "coordinate target", prompt: "coordinate the target", subagent_type: "general" },
        rootContext,
      )
      coordinatorID = coordinator.metadata.sessionId
      yield* Deferred.await(coordinatorStarted)
      const coordinatorAssistant = yield* sessions.updateMessage({
        ...assistant,
        id: MessageID.ascending(),
        parentID: MessageID.ascending(),
        sessionID: coordinator.metadata.sessionId,
        mode: "general",
        agent: "general",
      })
      const worker = yield* def.execute(
        { description: "implement target", prompt: "inspect the worker path", subagent_type: "general" },
        {
          ...rootContext,
          sessionID: coordinator.metadata.sessionId,
          messageID: coordinatorAssistant.id,
          agent: "general",
        },
      )
      expect((yield* sessions.get(worker.metadata.sessionId)).taskParentID).toBe(coordinator.metadata.sessionId)
      yield* Deferred.succeed(workerLaunched, undefined)
      yield* Deferred.await(coordinatorTurnEnded)
      yield* Effect.sleep("50 millis")
      // The root must not receive a final-looking completion while the coordinator still waits for its worker.
      expect(notifications.filter((input) => input.sessionID === chat.id)).toEqual([])

      yield* Deferred.succeed(releaseWorker, undefined)
      expect((yield* jobs.wait({ id: worker.metadata.sessionId, timeout: 2_000 })).info?.status).toBe("completed")
      const delivered = yield* pollWithTimeout(
        Effect.sync(() => notifications.find((input) => input.sessionID === chat.id)),
        "the coordinator final reply never reached the root session",
      )

      const texts = (input: SessionPrompt.PromptInput) =>
        input.parts.flatMap((part) => (part.type === "text" ? [part.text] : [])).join("\n")
      expect(notifications[0]?.sessionID).toBe(coordinator.metadata.sessionId)
      expect(texts(notifications[0]!)).toContain("worker raw output")
      expect(continuations).toBe(1)
      expect(texts(delivered)).toContain("coordinator final reply")
      expect(texts(delivered)).not.toContain("worker raw output")
      expect((yield* jobs.wait({ id: coordinator.metadata.sessionId, timeout: 2_000 })).info?.status).toBe("completed")
      expect(yield* tasks.get(worker.metadata.sessionId)).toMatchObject({
        status: "completed",
        delivery_session_id: coordinator.metadata.sessionId,
      })
      expect(yield* tasks.get(coordinator.metadata.sessionId)).toMatchObject({
        status: "completed",
        delivery_session_id: chat.id,
      })
      yield* Effect.sleep("50 millis")
      expect(notifications.filter((input) => input.sessionID === chat.id)).toHaveLength(1)
    }),
  )

  background.instance("an intermediate task stops its running child from inside its own turn", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const sessions = yield* Session.Service
      const tasks = yield* SessionTaskState.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const controlTool = yield* TaskControlTool
      const control = yield* controlTool.init()
      const outerStarted = yield* Deferred.make<void>()
      const stopRequested = yield* Deferred.make<SessionID>()
      const stopReturned = yield* Deferred.make<string>()
      const notifications: SessionPrompt.PromptInput[] = []
      const promptOps: TaskPromptOps = {
        ...stubOps(),
        prompt: (input) =>
          Effect.gen(function* () {
            const text = input.parts.find((part) => part.type === "text")
            if (text?.type === "text" && text.text === "inspect the nested path") return yield* Effect.never
            yield* Deferred.succeed(outerStarted, undefined)
            // The outer turn stays inside the task_control call until stop returns.
            const taskID = yield* Deferred.await(stopRequested)
            const result = yield* control.execute(
              { action: "stop", task_id: taskID },
              {
                sessionID: input.sessionID,
                messageID: MessageID.ascending(),
                agent: "general",
                abort: new AbortController().signal,
                messages: [],
                metadata: () => Effect.void,
                ask: () => Effect.void,
              },
            )
            yield* Deferred.succeed(stopReturned, result.output)
            return reply(input, "outer done")
          }),
        admitNotification: (input, _checkpoint, _source, onAdmitted) =>
          Effect.sync(() => notifications.push(input)).pipe(
            Effect.andThen(onAdmitted ?? Effect.void),
            Effect.as(Option.some(Effect.succeed(undefined))),
          ),
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
        { description: "outer investigation", prompt: "coordinate the investigation", subagent_type: "general" },
        rootContext,
      )
      yield* Deferred.await(outerStarted)
      const nestedAssistant = yield* sessions.updateMessage({
        ...assistant,
        id: MessageID.ascending(),
        parentID: MessageID.ascending(),
        sessionID: outer.metadata.sessionId,
        mode: "general",
        agent: "general",
      })
      const nested = yield* def.execute(
        { description: "nested investigation", prompt: "inspect the nested path", subagent_type: "general" },
        { ...rootContext, sessionID: outer.metadata.sessionId, messageID: nestedAssistant.id, agent: "general" },
      )

      yield* Deferred.succeed(stopRequested, nested.metadata.sessionId)
      expect(yield* Deferred.await(stopReturned).pipe(Effect.timeout("2 seconds"))).toBe(
        `Stopped task ${nested.metadata.sessionId}.`,
      )
      expect((yield* jobs.wait({ id: nested.metadata.sessionId, timeout: 2_000 })).info?.status).toBe("cancelled")
      expect(yield* tasks.get(nested.metadata.sessionId)).toMatchObject({
        status: "cancelled",
        delivery_session_id: outer.metadata.sessionId,
      })
      expect(notifications[0]?.sessionID).toBe(outer.metadata.sessionId)
      expect(notifications[0]?.parts[0]).toMatchObject({
        type: "text",
        text: expect.stringContaining("Task cancelled"),
      })
    }),
  )

  background.instance("stopping a held coordinator sends one cancellation notice to the root", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const controlTool = yield* TaskControlTool
      const control = yield* controlTool.init()
      const turnEnded = yield* Deferred.make<{ coordinator: SessionID; worker: SessionID }>()
      const notifications: SessionPrompt.PromptInput[] = []
      const promptOps: TaskPromptOps = {
        ...stubOps(),
        prompt: (input) =>
          Effect.gen(function* () {
            const text = input.parts.find((part) => part.type === "text")
            if (text?.type === "text" && text.text === "inspect the worker path") return yield* Effect.never
            const coordinatorAssistant = yield* sessions.updateMessage({
              ...assistant,
              id: MessageID.ascending(),
              parentID: MessageID.ascending(),
              sessionID: input.sessionID,
              mode: "general",
              agent: "general",
            })
            const worker = yield* def.execute(
              { description: "implement target", prompt: "inspect the worker path", subagent_type: "general" },
              { ...rootContext, sessionID: input.sessionID, messageID: coordinatorAssistant.id, agent: "general" },
            )
            yield* Deferred.succeed(turnEnded, { coordinator: input.sessionID, worker: worker.metadata.sessionId })
            return reply(input, "coordinator is waiting for the worker")
          }),
        admitNotification: (input, _checkpoint, _source, onAdmitted) =>
          Effect.sync(() => notifications.push(input)).pipe(
            Effect.andThen(onAdmitted ?? Effect.void),
            Effect.as(Option.some(Effect.succeed(undefined))),
          ),
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

      yield* def.execute(
        { description: "coordinate target", prompt: "coordinate the target", subagent_type: "general" },
        rootContext,
      )
      const ids = yield* Deferred.await(turnEnded)
      yield* Effect.sleep("50 millis")
      expect((yield* jobs.get(ids.coordinator))?.status).toBe("running")

      const stop = yield* control.execute({ action: "stop", task_id: ids.coordinator }, rootContext)
      expect(stop.output).toBe(`Stopped task ${ids.coordinator} and 1 active descendant.`)
      expect((yield* jobs.wait({ id: ids.coordinator, timeout: 2_000 })).info?.status).toBe("cancelled")
      expect((yield* jobs.wait({ id: ids.worker, timeout: 2_000 })).info?.status).toBe("cancelled")
      yield* Effect.sleep("100 millis")

      const rootTexts = notifications
        .filter((input) => input.sessionID === chat.id)
        .map((input) => input.parts.flatMap((part) => (part.type === "text" ? [part.text] : [])).join("\n"))
      expect(rootTexts).toHaveLength(1)
      expect(rootTexts[0]).toContain("Background task failed: coordinate target")
    }),
  )

  background.instance("a nested task stopped from its parent turn stops running before that turn ends", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const controlTool = yield* TaskControlTool
      const control = yield* controlTool.init()
      const outerStarted = yield* Deferred.make<void>()
      const stopRequested = yield* Deferred.make<SessionID>()
      const stopReturned = yield* Deferred.make<string>()
      const releaseOuter = yield* Deferred.make<void>()
      const nestedCancelled = yield* Deferred.make<void>()
      const nestedStopped = yield* Deferred.make<void>()
      const notifications: SessionPrompt.PromptInput[] = []
      let nestedID: SessionID | undefined
      const promptOps: TaskPromptOps = {
        ...stubOps(),
        // Like SessionPrompt.cancel, cancelling the nested Session ends its running prompt.
        cancel: (sessionID) => (sessionID === nestedID ? Deferred.succeed(nestedCancelled, undefined) : Effect.void),
        prompt: (input) =>
          Effect.gen(function* () {
            const text = input.parts.find((part) => part.type === "text")
            if (text?.type === "text" && text.text === "inspect the nested path") {
              yield* Deferred.await(nestedCancelled)
              yield* Deferred.succeed(nestedStopped, undefined)
              return reply(input, "partial nested output")
            }
            yield* Deferred.succeed(outerStarted, undefined)
            const taskID = yield* Deferred.await(stopRequested)
            const result = yield* control.execute(
              { action: "stop", task_id: taskID },
              {
                sessionID: input.sessionID,
                messageID: MessageID.ascending(),
                agent: "general",
                abort: new AbortController().signal,
                messages: [],
                metadata: () => Effect.void,
                ask: () => Effect.void,
              },
            )
            yield* Deferred.succeed(stopReturned, result.output)
            // The outer turn keeps working after the stop.
            yield* Deferred.await(releaseOuter)
            return reply(input, "outer done")
          }),
        admitNotification: (input, _checkpoint, _source, onAdmitted) =>
          Effect.sync(() => notifications.push(input)).pipe(
            Effect.andThen(onAdmitted ?? Effect.void),
            Effect.as(Option.some(Effect.succeed(undefined))),
          ),
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
        { description: "outer investigation", prompt: "coordinate the investigation", subagent_type: "general" },
        rootContext,
      )
      yield* Deferred.await(outerStarted)
      const nestedAssistant = yield* sessions.updateMessage({
        ...assistant,
        id: MessageID.ascending(),
        parentID: MessageID.ascending(),
        sessionID: outer.metadata.sessionId,
        mode: "general",
        agent: "general",
      })
      const nested = yield* def.execute(
        { description: "nested investigation", prompt: "inspect the nested path", subagent_type: "general" },
        { ...rootContext, sessionID: outer.metadata.sessionId, messageID: nestedAssistant.id, agent: "general" },
      )
      nestedID = nested.metadata.sessionId

      yield* Deferred.succeed(stopRequested, nested.metadata.sessionId)
      expect(yield* Deferred.await(stopReturned).pipe(Effect.timeout("2 seconds"))).toBe(
        `Stopped task ${nested.metadata.sessionId}.`,
      )
      yield* Deferred.await(nestedStopped).pipe(Effect.timeout("2 seconds"))
      expect(yield* Deferred.isDone(releaseOuter)).toBe(false)
      expect(notifications).toEqual([])

      yield* Deferred.succeed(releaseOuter, undefined)
      expect((yield* jobs.wait({ id: nested.metadata.sessionId, timeout: 2_000 })).info?.status).toBe("cancelled")
      expect(notifications[0]?.sessionID).toBe(outer.metadata.sessionId)
      expect(notifications[0]?.parts[0]).toMatchObject({
        type: "text",
        text: expect.stringContaining("Task cancelled"),
      })
    }),
  )

  background.instance("aborting a nested task returns while its parent task turn is still running", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const sessions = yield* Session.Service
      const runState = yield* SessionRunState.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const outerStarted = yield* Deferred.make<void>()
      const releaseOuter = yield* Deferred.make<void>()
      const notifications: SessionPrompt.PromptInput[] = []
      const promptOps: TaskPromptOps = {
        ...stubOps(),
        prompt: (input) =>
          Effect.gen(function* () {
            const text = input.parts.find((part) => part.type === "text")
            if (text?.type === "text" && text.text === "inspect the nested path") return yield* Effect.never
            yield* Deferred.succeed(outerStarted, undefined)
            yield* Deferred.await(releaseOuter)
            return reply(input, "outer done")
          }),
        admitNotification: (input, _checkpoint, _source, onAdmitted) =>
          Effect.sync(() => notifications.push(input)).pipe(
            Effect.andThen(onAdmitted ?? Effect.void),
            Effect.as(Option.some(Effect.succeed(undefined))),
          ),
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
        { description: "outer investigation", prompt: "coordinate the investigation", subagent_type: "general" },
        rootContext,
      )
      yield* Deferred.await(outerStarted)
      const nestedAssistant = yield* sessions.updateMessage({
        ...assistant,
        id: MessageID.ascending(),
        parentID: MessageID.ascending(),
        sessionID: outer.metadata.sessionId,
        mode: "general",
        agent: "general",
      })
      const nested = yield* def.execute(
        { description: "nested investigation", prompt: "inspect the nested path", subagent_type: "general" },
        { ...rootContext, sessionID: outer.metadata.sessionId, messageID: nestedAssistant.id, agent: "general" },
      )

      yield* runState.cancel(nested.metadata.sessionId).pipe(Effect.timeout("2 seconds"))
      // The cancellation notice waits for the parent task turn, but abort does not.
      expect(notifications).toEqual([])
      expect(yield* Deferred.isDone(releaseOuter)).toBe(false)

      yield* Deferred.succeed(releaseOuter, undefined)
      expect((yield* jobs.wait({ id: nested.metadata.sessionId, timeout: 2_000 })).info?.status).toBe("cancelled")
      const delivered = yield* pollWithTimeout(
        Effect.sync(() => notifications.find((input) => input.sessionID === outer.metadata.sessionId)),
        "the cancellation notice never reached the parent task",
      )
      expect(delivered.parts[0]).toMatchObject({ type: "text", text: expect.stringContaining("Task cancelled") })
    }),
  )

  background.instance("an intermediate task resuming a finalizing child fails fast", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const sessions = yield* Session.Service
      const tasks = yield* SessionTaskState.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const outerStarted = yield* Deferred.make<void>()
      const resumeRequested = yield* Deferred.make<{ taskID: SessionID; messageID: MessageID }>()
      const resumeReturned = yield* Deferred.make<Exit.Exit<unknown, unknown>>()
      const notifications: SessionPrompt.PromptInput[] = []
      const promptOps: TaskPromptOps = {
        ...stubOps(),
        prompt: (input) =>
          Effect.gen(function* () {
            const text = input.parts.find((part) => part.type === "text")
            if (text?.type === "text" && text.text !== "coordinate the investigation")
              return reply(input, "nested done")
            yield* Deferred.succeed(outerStarted, undefined)
            // The outer turn resumes its child while the child's result waits for this turn to end.
            const request = yield* Deferred.await(resumeRequested)
            const exit = yield* def
              .execute(
                {
                  description: "continue nested investigation",
                  prompt: "continue the nested path",
                  subagent_type: "general",
                  task_id: request.taskID,
                },
                { ...rootContext, sessionID: input.sessionID, messageID: request.messageID, agent: "general" },
              )
              .pipe(Effect.exit)
            yield* Deferred.succeed(resumeReturned, exit)
            return reply(input, "outer done")
          }),
        admitNotification: (input, _checkpoint, _source, onAdmitted) =>
          Effect.sync(() => notifications.push(input)).pipe(
            Effect.andThen(onAdmitted ?? Effect.void),
            Effect.as(Option.some(Effect.succeed(undefined))),
          ),
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
        { description: "outer investigation", prompt: "coordinate the investigation", subagent_type: "general" },
        rootContext,
      )
      yield* Deferred.await(outerStarted)
      const nestedAssistant = yield* sessions.updateMessage({
        ...assistant,
        id: MessageID.ascending(),
        parentID: MessageID.ascending(),
        sessionID: outer.metadata.sessionId,
        mode: "general",
        agent: "general",
      })
      const nested = yield* def.execute(
        { description: "nested investigation", prompt: "inspect the nested path", subagent_type: "general" },
        { ...rootContext, sessionID: outer.metadata.sessionId, messageID: nestedAssistant.id, agent: "general" },
      )
      yield* pollWithTimeout(
        Effect.gen(function* () {
          const lifecycle = yield* tasks.get(nested.metadata.sessionId)
          const job = yield* jobs.get(nested.metadata.sessionId)
          if (lifecycle?.status === "completed" && job?.status === "running") return true
        }),
        "nested task never started finalizing",
      )

      yield* Deferred.succeed(resumeRequested, { taskID: nested.metadata.sessionId, messageID: nestedAssistant.id })
      const exit = yield* Deferred.await(resumeReturned).pipe(Effect.timeout("2 seconds"))
      expect(failureMessage(exit)).toBe(
        `Task ${nested.metadata.sessionId} finished; its result is queued for this session. End this turn, then resume it.`,
      )
      expect((yield* jobs.wait({ id: nested.metadata.sessionId, timeout: 2_000 })).info).toMatchObject({
        status: "completed",
        output: "nested done",
      })
      expect(yield* tasks.get(nested.metadata.sessionId)).toMatchObject({
        generation: 1,
        status: "completed",
        delivery_session_id: outer.metadata.sessionId,
      })
      expect(notifications[0]?.sessionID).toBe(outer.metadata.sessionId)
    }),
  )

  background.instance("an intermediate task resuming a child it just stopped reports the cancellation", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const controlTool = yield* TaskControlTool
      const control = yield* controlTool.init()
      const outerStarted = yield* Deferred.make<void>()
      const resumeRequested = yield* Deferred.make<{ taskID: SessionID; messageID: MessageID }>()
      const resumeReturned = yield* Deferred.make<Exit.Exit<unknown, unknown>>()
      const promptOps: TaskPromptOps = {
        ...stubOps(),
        prompt: (input) =>
          Effect.gen(function* () {
            const text = input.parts.find((part) => part.type === "text")
            if (text?.type === "text" && text.text === "inspect the nested path") return yield* Effect.never
            yield* Deferred.succeed(outerStarted, undefined)
            const request = yield* Deferred.await(resumeRequested)
            const context = {
              ...rootContext,
              sessionID: input.sessionID,
              messageID: request.messageID,
              agent: "general",
            }
            yield* control.execute({ action: "stop", task_id: request.taskID }, context)
            const exit = yield* def
              .execute(
                {
                  description: "continue nested investigation",
                  prompt: "continue the nested path",
                  subagent_type: "general",
                  task_id: request.taskID,
                },
                context,
              )
              .pipe(Effect.exit)
            yield* Deferred.succeed(resumeReturned, exit)
            return reply(input, "outer done")
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
        { description: "outer investigation", prompt: "coordinate the investigation", subagent_type: "general" },
        rootContext,
      )
      yield* Deferred.await(outerStarted)
      const nestedAssistant = yield* sessions.updateMessage({
        ...assistant,
        id: MessageID.ascending(),
        parentID: MessageID.ascending(),
        sessionID: outer.metadata.sessionId,
        mode: "general",
        agent: "general",
      })
      const nested = yield* def.execute(
        { description: "nested investigation", prompt: "inspect the nested path", subagent_type: "general" },
        { ...rootContext, sessionID: outer.metadata.sessionId, messageID: nestedAssistant.id, agent: "general" },
      )

      yield* Deferred.succeed(resumeRequested, { taskID: nested.metadata.sessionId, messageID: nestedAssistant.id })
      const exit = yield* Deferred.await(resumeReturned).pipe(Effect.timeout("2 seconds"))
      expect(failureMessage(exit)).toBe(
        `Task ${nested.metadata.sessionId} was cancelled; its cancellation notice is queued for this session. End this turn, then resume it.`,
      )
      expect((yield* jobs.wait({ id: nested.metadata.sessionId, timeout: 2_000 })).info?.status).toBe("cancelled")
    }),
  )

  background.instance(
    "resuming a finalizing child waits when its result goes past a settling caller job",
    () =>
      Effect.gen(function* () {
        const jobs = yield* BackgroundJob.Service
        const sessions = yield* Session.Service
        const tasks = yield* SessionTaskState.Service
        const { chat, assistant } = yield* seed()
        const ancestor = yield* sessions.createTask({ parentID: chat.id, title: "Ancestor task", agent: "general" })
        const releaseAncestor = yield* Deferred.make<void>()
        yield* jobs.start({
          id: ancestor.id,
          type: "task",
          metadata: { parentSessionId: chat.id, sessionId: ancestor.id },
          run: Deferred.await(releaseAncestor).pipe(Effect.as("ancestor done")),
        })
        const outer = yield* sessions.createTask({ parentID: ancestor.id, title: "Outer task", agent: "general" })
        const outerDelivering = yield* Deferred.make<void>()
        const releaseOuterDelivery = yield* Deferred.make<void>()
        yield* jobs.start({
          id: outer.id,
          type: "task",
          metadata: { parentSessionId: ancestor.id, sessionId: outer.id },
          notifyOnComplete: true,
          awaitOnComplete: true,
          onComplete: () =>
            Deferred.succeed(outerDelivering, undefined).pipe(Effect.andThen(Deferred.await(releaseOuterDelivery))),
          run: Effect.succeed("outer done"),
        })
        yield* Deferred.await(outerDelivering)
        const nestedAssistant = yield* sessions.updateMessage({
          ...assistant,
          id: MessageID.ascending(),
          parentID: MessageID.ascending(),
          sessionID: outer.id,
          mode: "general",
          agent: "general",
        })
        const notifications: SessionPrompt.PromptInput[] = []
        const prompts: string[] = []
        const promptOps: TaskPromptOps = {
          ...stubOps(),
          prompt: (input) =>
            Effect.sync(() => {
              const text = input.parts.find((part) => part.type === "text")
              prompts.push(text?.type === "text" ? text.text : "")
              return reply(input, "nested done")
            }),
          admitNotification: (input, _checkpoint, _source, onAdmitted) =>
            Effect.sync(() => notifications.push(input)).pipe(
              Effect.andThen(onAdmitted ?? Effect.void),
              Effect.as(Option.some(Effect.succeed(undefined))),
            ),
        }
        const context = {
          sessionID: outer.id,
          messageID: nestedAssistant.id,
          agent: "general",
          abort: new AbortController().signal,
          extra: { promptOps },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        }
        const tool = yield* TaskTool
        const def = yield* tool.init()

        const nested = yield* def.execute(
          { description: "nested investigation", prompt: "inspect the nested path", subagent_type: "general" },
          context,
        )
        // The settling caller job cannot take the result, so it waits for the ancestor turn instead.
        yield* pollWithTimeout(
          Effect.gen(function* () {
            const lifecycle = yield* tasks.get(nested.metadata.sessionId)
            const job = yield* jobs.get(nested.metadata.sessionId)
            if (lifecycle?.status === "completed" && job?.status === "running") return true
          }),
          "nested task never started finalizing",
        )
        const resumed = yield* def
          .execute(
            {
              description: "continue nested investigation",
              prompt: "continue the nested path",
              subagent_type: "general",
              task_id: nested.metadata.sessionId,
            },
            context,
          )
          .pipe(Effect.exit, Effect.forkChild)
        yield* Effect.sleep("50 millis")
        expect(resumed.pollUnsafe()).toBeUndefined()

        yield* Deferred.succeed(releaseAncestor, undefined)
        const exit = yield* Fiber.join(resumed).pipe(Effect.timeout("2 seconds"))
        expect(Exit.isSuccess(exit)).toBe(true)
        expect(notifications[0]?.sessionID).toBe(ancestor.id)
        yield* pollWithTimeout(
          Effect.sync(() => (prompts.length === 2 ? true : undefined)),
          "the resumed task never started",
        )
        expect(yield* tasks.get(nested.metadata.sessionId)).toMatchObject({ generation: 2 })
        yield* Deferred.succeed(releaseOuterDelivery, undefined)
      }),
    { config: { subagent_depth: 3 } },
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
              admitNotification: (
                input: SessionPrompt.PromptInput,
                _checkpoint: number,
                _source?: SessionPrompt.NotificationSource,
                onAdmitted?: Effect.Effect<void>,
              ) =>
                Effect.sync(() => notifications.push(input)).pipe(
                  Effect.andThen(onAdmitted ?? Effect.void),
                  Effect.as(Option.some(Effect.succeed(undefined))),
                ),
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
              admitNotification: (
                input: SessionPrompt.PromptInput,
                _checkpoint: number,
                _source?: SessionPrompt.NotificationSource,
                onAdmitted?: Effect.Effect<void>,
              ) =>
                Effect.sync(() => notifications.push(input)).pipe(
                  Effect.andThen(onAdmitted ?? Effect.void),
                  Effect.as(Option.some(Effect.succeed(undefined))),
                ),
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

  background.instance("does not route completion to a root that moved to another location", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const child = yield* sessions.createTask({ parentID: chat.id, title: "Outer task", agent: "general" })
      yield* sessions.setWorkspace({ sessionID: chat.id, workspaceID: WorkspaceV2.ID.create() })
      const nestedAssistant = yield* sessions.updateMessage({
        ...assistant,
        id: MessageID.ascending(),
        parentID: MessageID.ascending(),
        sessionID: child.id,
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
          sessionID: child.id,
          messageID: nestedAssistant.id,
          agent: "general",
          abort: new AbortController().signal,
          extra: {
            promptOps: {
              ...stubOps({ text: "nested done" }),
              admitNotification: (
                input: SessionPrompt.PromptInput,
                _checkpoint: number,
                _source?: SessionPrompt.NotificationSource,
                onAdmitted?: Effect.Effect<void>,
              ) =>
                Effect.sync(() => notifications.push(input)).pipe(
                  Effect.andThen(onAdmitted ?? Effect.void),
                  Effect.as(Option.some(Effect.succeed(undefined))),
                ),
            },
          },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      expect((yield* jobs.get(result.metadata.sessionId))?.metadata?.ancestorSessionIds).toEqual([child.id])
      expect((yield* jobs.wait({ id: result.metadata.sessionId })).info?.status).toBe("completed")
      yield* pollWithTimeout(
        Effect.sync(() => notifications[0]),
        "outer task was never notified",
      )
      expect(notifications.map((item) => item.sessionID)).toEqual([child.id])
    }),
  )

  background.instance(
    "stops notification ancestry at the first ancestor in another location",
    () =>
      Effect.gen(function* () {
        const jobs = yield* BackgroundJob.Service
        const sessions = yield* Session.Service
        const { chat, assistant } = yield* seed()
        const child = yield* sessions.createTask({ parentID: chat.id, title: "Outer task", agent: "general" })
        const grandchild = yield* sessions.createTask({ parentID: child.id, title: "Inner task", agent: "general" })
        yield* sessions.setWorkspace({ sessionID: grandchild.id, workspaceID: WorkspaceV2.ID.create() })
        const nestedAssistant = yield* sessions.updateMessage({
          ...assistant,
          id: MessageID.ascending(),
          parentID: MessageID.ascending(),
          sessionID: grandchild.id,
          mode: "general",
          agent: "general",
        })
        const tool = yield* TaskTool
        const def = yield* tool.init()

        const result = yield* def.execute(
          {
            description: "inspect nested bug",
            prompt: "look into the nested cache path",
            subagent_type: "general",
          },
          {
            sessionID: grandchild.id,
            messageID: nestedAssistant.id,
            agent: "general",
            abort: new AbortController().signal,
            extra: { promptOps: stubOps({ text: "nested done" }) },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )

        // Root and child share a location, but the route must not skip the child to reach the root.
        expect((yield* jobs.get(result.metadata.sessionId))?.metadata?.ancestorSessionIds).toEqual([grandchild.id])
        expect((yield* jobs.wait({ id: result.metadata.sessionId })).info?.status).toBe("completed")
      }),
    { config: { subagent_depth: 3 } },
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

  background.instance("root notification continuation keeps the parent checkpoint until it settles", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const runState = yield* SessionRunState.Service
      const status = yield* SessionStatus.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const continuationStarted = yield* Deferred.make<void>()
      const releaseContinuation = yield* Deferred.make<void>()
      const currentInContinuation = yield* Deferred.make<boolean>()
      yield* status.set(chat.id, { type: "busy" })

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
              checkpoint: (sessionID) => runState.checkpoint(sessionID),
              retain: (sessionID, checkpoint) => runState.retain(sessionID, checkpoint),
              admitNotification: (_input, checkpoint) =>
                Effect.succeed(
                  Option.some(
                    Deferred.succeed(continuationStarted, undefined).pipe(
                      Effect.andThen(Deferred.await(releaseContinuation)),
                      Effect.andThen(runState.isCurrent(chat.id, checkpoint)),
                      Effect.flatMap((current) => Deferred.succeed(currentInContinuation, current)),
                      Effect.as(undefined),
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

      yield* Deferred.await(continuationStarted)
      expect((yield* jobs.wait({ id: result.metadata.sessionId })).info?.status).toBe("completed")
      expect((yield* status.get(chat.id)).type).toBe("busy")

      yield* Deferred.succeed(releaseContinuation, undefined)
      expect(yield* Deferred.await(currentInContinuation)).toBe(true)
      yield* pollWithTimeout(
        status.get(chat.id).pipe(Effect.map((current) => (current.type === "idle" ? true : undefined))),
        "releasing the continuation lease never published idle",
      )
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

  background.instance("cancelling message tasks cancels their descendants and keeps other tasks", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const runState = yield* SessionRunState.Service
      const parent = SessionID.make("ses_parent")
      const kept = SessionID.make("ses_kept")
      const reverted = SessionID.make("ses_reverted")
      const grandchild = SessionID.make("ses_grandchild")
      yield* jobs.start({
        id: kept,
        type: "task",
        metadata: { parentSessionId: parent, sessionId: kept, ancestorSessionIds: [parent], messageId: "msg_kept" },
        run: Effect.never,
      })
      yield* jobs.start({
        id: reverted,
        type: "task",
        metadata: {
          parentSessionId: parent,
          sessionId: reverted,
          ancestorSessionIds: [parent],
          messageId: "msg_reverted",
        },
        run: Effect.never,
      })
      yield* jobs.start({
        id: grandchild,
        type: "task",
        metadata: {
          parentSessionId: reverted,
          sessionId: grandchild,
          ancestorSessionIds: [reverted, parent],
          messageId: "msg_reverted_child",
        },
        run: Effect.never,
      })

      yield* runState.cancelTasks(parent, new Set(["msg_reverted"]))

      expect((yield* jobs.get(reverted))?.status).toBe("cancelled")
      expect((yield* jobs.get(grandchild))?.status).toBe("cancelled")
      expect((yield* jobs.get(kept))?.status).toBe("running")
      yield* jobs.cancel(kept)
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
      // The root notification continuation takes its own lease and releases it when it settles.
      expect(retained).toEqual([parent.id, chat.id, chat.id])
      expect(released).toEqual([chat.id, parent.id, chat.id])
      yield* jobs.cancel(result.metadata.sessionId)
      expect(released).toEqual([chat.id, parent.id, chat.id])

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
