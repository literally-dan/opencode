import { afterEach, describe, expect, mock, spyOn, test } from "bun:test"
import { OpencodeClient, type GlobalEvent } from "@opencode-ai/sdk/v2"
import { runInteractiveMode } from "@/cli/cmd/run/runtime"
import { ShownTurnError } from "@/cli/cmd/run/stream.transport"
import type { FooterApi, FooterEvent, RunPrompt, RunProvider, StreamCommit } from "@/cli/cmd/run/types"

type SessionMessage = NonNullable<Awaited<ReturnType<OpencodeClient["session"]["messages"]>>["data"]>[number]
type GlobalEventStream = Awaited<ReturnType<OpencodeClient["global"]["event"]>>["stream"]

const provider: RunProvider = {
  id: "openai",
  name: "OpenAI",
  source: "api",
  env: [],
  options: {},
  models: {
    "gpt-5": {
      id: "gpt-5",
      providerID: "openai",
      api: {
        id: "openai",
        url: "https://openai.test",
        npm: "@ai-sdk/openai",
      },
      name: "Little Frank",
      capabilities: {
        temperature: true,
        reasoning: true,
        attachment: true,
        toolcall: true,
        input: {
          text: true,
          audio: false,
          image: false,
          video: false,
          pdf: false,
        },
        output: {
          text: true,
          audio: false,
          image: false,
          video: false,
          pdf: false,
        },
        interleaved: false,
      },
      cost: {
        input: 0,
        output: 0,
        cache: {
          read: 0,
          write: 0,
        },
      },
      limit: {
        context: 128000,
        output: 8192,
      },
      status: "active",
      options: {},
      headers: {},
      release_date: "2026-01-01",
    },
  },
}

const transportProviders: RunProvider[][] = []

function defer<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

function ok<T>(data: T) {
  return Promise.resolve({
    data,
    error: undefined,
    request: new Request("https://opencode.test"),
    response: new Response(),
  })
}

function footer(): FooterApi {
  let closed = false
  const closes = new Set<() => void>()

  const notify = () => {
    for (const fn of closes) fn()
  }

  return {
    get isClosed() {
      return closed
    },
    onPrompt: () => () => {},
    onQueuedRemove: () => () => {},
    onClose(fn) {
      if (closed) {
        fn()
        return () => {}
      }

      closes.add(fn)
      return () => {
        closes.delete(fn)
      }
    },
    event() {},
    append() {},
    idle() {
      return Promise.resolve()
    },
    close() {
      if (closed) {
        return
      }

      closed = true
      notify()
    },
    destroy() {
      if (closed) {
        return
      }

      closed = true
      notify()
    },
  }
}

function promptFooter() {
  const api = footer()
  const prompts = new Set<(input: RunPrompt) => void>()
  const events: FooterEvent[] = []
  api.onPrompt = (fn) => {
    prompts.add(fn)
    return () => {
      prompts.delete(fn)
    }
  }
  api.event = (next) => {
    events.push(next)
  }

  return {
    api,
    events,
    get subscribed() {
      return prompts.size > 0
    },
    submit(text: string) {
      for (const fn of prompts) fn({ text, parts: [] })
    },
  }
}

async function waitFor(check: () => boolean, label: string) {
  const end = Date.now() + 2_000
  while (Date.now() < end) {
    if (check()) return
    await Bun.sleep(10)
  }

  throw new Error(`timed out waiting for ${label}`)
}

afterEach(() => {
  mock.restore()
  transportProviders.length = 0
})

describe("run interactive runtime", () => {
  test("waits for provider metadata before eager replay transport bootstrap", async () => {
    const providersStarted = defer<void>()
    const providers = defer<void>()

    const sdk = new OpencodeClient()
    spyOn(sdk.config, "providers").mockImplementation(async () => {
      providersStarted.resolve()
      await providers.promise
      return ok({ providers: [provider], default: {} })
    })
    spyOn(sdk.session, "messages").mockImplementation(() =>
      ok([
        {
          info: {
            id: "msg-user-1",
            sessionID: "ses-1",
            role: "user",
            time: {
              created: 1,
            },
            agent: "build",
            model: {
              providerID: "openai",
              modelID: "gpt-5",
              variant: undefined,
            },
          },
          parts: [
            {
              id: "part-user-1",
              sessionID: "ses-1",
              messageID: "msg-user-1",
              type: "text",
              text: "hello",
            },
          ],
        } satisfies SessionMessage,
      ]),
    )
    spyOn(sdk.session, "get").mockRejectedValue(new Error("not needed"))
    spyOn(sdk.app, "agents").mockImplementation(() => ok([]))
    spyOn(sdk.experimental.resource, "list").mockImplementation(() => ok({}))
    spyOn(sdk.command, "list").mockImplementation(() => ok([]))

    const task = runInteractiveMode(
      {
        sdk,
        directory: "/tmp",
        sessionID: "ses-1",
        sessionTitle: "Session",
        resume: true,
        replay: true,
        replayLimit: 100,
        agent: "build",
        model: {
          providerID: "openai",
          modelID: "gpt-5",
        },
        variant: undefined,
        files: [],
        thinking: true,
      },
      {
        createRuntimeLifecycle: async () => ({
          footer: footer(),
          onResize: () => () => {},
          refreshTheme: () => {},
          resetForReplay: () => Promise.resolve(),
          close: () => Promise.resolve(),
        }),
        streamTransport: Promise.resolve({
          createSessionTransport: async (input: { providers?: () => RunProvider[]; footer: FooterApi }) => {
            transportProviders.push(input.providers?.() ?? [])
            setTimeout(() => {
              input.footer.close()
            }, 0)
            return {
              runPromptTurn: async () => {},
              selectSubagent: () => {},
              replayOnResize: async () => false,
              close: async () => {},
            }
          },
          formatUnknownError: (error: unknown) => (error instanceof Error ? error.message : String(error)),
          ShownTurnError,
        }),
      },
    )

    await providersStarted.promise

    expect(transportProviders).toEqual([])

    providers.resolve()

    await task

    expect(transportProviders).toEqual([[provider]])
  })

  test("sends a queued prompt when the active prompt returns while background tasks keep the session busy", async () => {
    const ui = promptFooter()
    const streamClosed = defer<void>()
    const firstReturns = defer<void>()
    const sent: string[] = []

    const sdk = new OpencodeClient()
    spyOn(sdk.config, "providers").mockImplementation(() => ok({ providers: [provider], default: {} }))
    spyOn(sdk.session, "messages").mockImplementation(() => ok([]))
    spyOn(sdk.session, "get").mockRejectedValue(new Error("not needed"))
    spyOn(sdk.session, "children").mockImplementation(() => ok([]))
    // Undelivered background Task results keep the session busy for the whole test.
    spyOn(sdk.session, "status").mockImplementation(() => ok({ "ses-1": { type: "busy" } }))
    spyOn(sdk.permission, "list").mockImplementation(() => ok([]))
    spyOn(sdk.question, "list").mockImplementation(() => ok([]))
    spyOn(sdk.app, "agents").mockImplementation(() => ok([]))
    spyOn(sdk.experimental.resource, "list").mockImplementation(() => ok({}))
    spyOn(sdk.command, "list").mockImplementation(() => ok([]))
    spyOn(sdk.global, "event").mockImplementation(() =>
      Promise.resolve({
        stream: (async function* (): GlobalEventStream {
          await streamClosed.promise
        })(),
      }),
    )
    spyOn(sdk.session, "prompt").mockImplementation(async (input) => {
      sent.push(input.parts?.find((part) => part.type === "text")?.text ?? "")
      if (sent.length === 1) await firstReturns.promise
      return ok({
        info: {
          id: `msg-assistant-${sent.length}`,
          sessionID: "ses-1",
          role: "assistant",
          time: { created: 1, completed: 2 },
          parentID: "msg-user-1",
          modelID: "gpt-5",
          providerID: "openai",
          mode: "chat",
          agent: "build",
          path: { cwd: "/tmp", root: "/tmp" },
          cost: 0,
          tokens: { input: 1, output: 1, reasoning: 0, cache: { read: 0, write: 0 } },
        },
        parts: [],
      } satisfies SessionMessage)
    })

    const task = runInteractiveMode(
      {
        sdk,
        directory: "/tmp",
        sessionID: "ses-1",
        sessionTitle: "Session",
        resume: false,
        agent: "build",
        model: {
          providerID: "openai",
          modelID: "gpt-5",
        },
        variant: undefined,
        files: [],
        thinking: true,
      },
      {
        createRuntimeLifecycle: async () => ({
          footer: ui.api,
          onResize: () => () => {},
          refreshTheme: () => {},
          resetForReplay: () => Promise.resolve(),
          close: () => Promise.resolve(),
        }),
      },
    )

    try {
      await waitFor(() => ui.subscribed, "the prompt queue")
      ui.submit("start background work")
      await waitFor(() => sent.length === 1, "the first prompt")
      ui.submit("talk while subagents run")
      await waitFor(
        () => ui.events.some((event) => event.type === "queued.prompts" && event.prompts.length === 1),
        "the queued prompt",
      )
      expect(sent).toEqual(["start background work"])

      firstReturns.resolve()
      await waitFor(() => sent.length === 2, "the queued prompt to be sent")
      expect(sent).toEqual(["start background work", "talk while subagents run"])
    } finally {
      ui.api.close()
      firstReturns.resolve()
      streamClosed.resolve()
      await task
    }
  })

  test("keeps the files of a turn whose failure the server already showed", async () => {
    const ui = promptFooter()
    const commits: StreamCommit[] = []
    ui.api.append = (commit) => {
      commits.push(commit)
    }
    const stream = { pending: [] as GlobalEvent[], closed: false, wake: undefined as (() => void) | undefined }
    const push = (event: GlobalEvent) => {
      stream.pending.push(event)
      stream.wake?.()
    }
    const sent: Array<Array<{ type: string }>> = []

    const sdk = new OpencodeClient()
    spyOn(sdk.config, "providers").mockImplementation(() => ok({ providers: [provider], default: {} }))
    spyOn(sdk.session, "messages").mockImplementation(() => ok([]))
    spyOn(sdk.session, "get").mockRejectedValue(new Error("not needed"))
    spyOn(sdk.session, "children").mockImplementation(() => ok([]))
    spyOn(sdk.session, "status").mockImplementation(() => ok({}))
    spyOn(sdk.permission, "list").mockImplementation(() => ok([]))
    spyOn(sdk.question, "list").mockImplementation(() => ok([]))
    spyOn(sdk.app, "agents").mockImplementation(() => ok([]))
    spyOn(sdk.experimental.resource, "list").mockImplementation(() => ok({}))
    spyOn(sdk.command, "list").mockImplementation(() => ok([]))
    spyOn(sdk.global, "event").mockImplementation(() =>
      Promise.resolve({
        stream: (async function* (): GlobalEventStream {
          while (!stream.closed) {
            const next = stream.pending.shift()
            if (next) {
              yield next
              continue
            }
            await new Promise<void>((resolve) => {
              stream.wake = resolve
            })
          }
        })(),
      }),
    )
    spyOn(sdk.session, "prompt").mockImplementation(async (input) => {
      sent.push(input.parts ?? [])
      if (sent.length === 1) {
        // The server publishes session.error before it fails the request.
        push({
          directory: "/tmp",
          payload: {
            id: "evt-error-1",
            type: "session.error",
            properties: { sessionID: "ses-1", error: { name: "UnknownError", data: { message: "Model not found" } } },
          },
        })
        await Bun.sleep(20)
        throw new Error("Unexpected server error. Check server logs for details.")
      }
      return ok({
        info: {
          id: "msg-assistant-2",
          sessionID: "ses-1",
          role: "assistant",
          time: { created: 1, completed: 2 },
          parentID: "msg-user-2",
          modelID: "gpt-5",
          providerID: "openai",
          mode: "chat",
          agent: "build",
          path: { cwd: "/tmp", root: "/tmp" },
          cost: 0,
          tokens: { input: 1, output: 1, reasoning: 0, cache: { read: 0, write: 0 } },
        },
        parts: [],
      } satisfies SessionMessage)
    })

    const task = runInteractiveMode(
      {
        sdk,
        directory: "/tmp",
        sessionID: "ses-1",
        sessionTitle: "Session",
        resume: false,
        agent: "build",
        model: {
          providerID: "openai",
          modelID: "gpt-5",
        },
        variant: undefined,
        files: [{ type: "file", url: "file:///tmp/notes.txt", filename: "notes.txt", mime: "text/plain" }],
        thinking: true,
      },
      {
        createRuntimeLifecycle: async () => ({
          footer: ui.api,
          onResize: () => () => {},
          refreshTheme: () => {},
          resetForReplay: () => Promise.resolve(),
          close: () => Promise.resolve(),
        }),
      },
    )

    try {
      await waitFor(() => ui.subscribed, "the prompt queue")
      ui.submit("first")
      await waitFor(() => ui.events.some((event) => event.type === "turn.idle"), "the failed turn to end")
      ui.submit("second")
      await waitFor(() => sent.length === 2, "the second prompt")

      // The failed turn did not use up the files, and its failure appears once.
      expect(sent.map((parts) => parts.some((part) => part.type === "file"))).toEqual([true, true])
      expect(commits.filter((commit) => commit.kind === "error").map((commit) => commit.text)).toEqual([
        "Model not found",
      ])
    } finally {
      ui.api.close()
      stream.closed = true
      stream.wake?.()
      await task
    }
  })
})
