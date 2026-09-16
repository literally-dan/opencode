/** @jsxImportSource @opentui/solid */
import { createDefaultOpenTuiKeymap } from "@opentui/keymap/opentui"
import { testRender, useRenderer, type JSX } from "@opentui/solid"
import type {
  AssistantMessage,
  Event,
  GlobalEvent,
  Session as SessionInfo,
  TextPart,
  ToolPart,
  UserMessage,
} from "@opencode-ai/sdk/v2"
import { expect, test } from "bun:test"
import { mkdir } from "node:fs/promises"
import path from "node:path"
import { onCleanup } from "solid-js"
import { TuiConfigProvider } from "../../../src/config"
import { ArgsProvider } from "../../../src/context/args"
import { DataProvider } from "../../../src/context/data"
import { EditorContextProvider } from "../../../src/context/editor"
import { EpilogueProvider } from "../../../src/context/epilogue"
import { ExitProvider } from "../../../src/context/exit"
import { KVProvider } from "../../../src/context/kv"
import { LocalProvider } from "../../../src/context/local"
import { LocationProvider } from "../../../src/context/location"
import { PermissionProvider } from "../../../src/context/permission"
import { ProjectProvider } from "../../../src/context/project"
import { PromptRefProvider } from "../../../src/context/prompt"
import { RouteProvider } from "../../../src/context/route"
import { SDKProvider } from "../../../src/context/sdk"
import { SyncProvider, useSync } from "../../../src/context/sync"
import { ThemeProvider } from "../../../src/context/theme"
import { Prompt } from "../../../src/component/prompt"
import { FrecencyProvider } from "../../../src/component/prompt/frecency"
import { PromptHistoryProvider } from "../../../src/component/prompt/history"
import { PromptStashProvider } from "../../../src/component/prompt/stash"
import { OpencodeKeymapProvider, registerOpencodeKeymap } from "../../../src/keymap"
import { createPluginRuntime, PluginRuntimeProvider } from "../../../src/plugin/runtime"
import { Session } from "../../../src/routes/session"
import { DialogProvider } from "../../../src/ui/dialog"
import { ToastProvider } from "../../../src/ui/toast"
import { tmpdir } from "../../fixture/fixture"
import { TestTuiContexts } from "../../fixture/tui-environment"
import { createEventSource, createFetch, directory, json } from "../../fixture/tui-sdk"
import { createTuiResolvedConfig } from "../../fixture/tui-runtime"

type Handler = (request: Request) => Response | Promise<Response | undefined> | undefined

const root = {
  id: "ses_root",
  slug: "root",
  projectID: "proj_test",
  directory,
  title: "Root session",
  version: "0.0.0-test",
  time: { created: 1, updated: 1 },
} satisfies SessionInfo
const child = {
  ...root,
  id: "ses_child",
  slug: "child",
  title: "Child session",
  parentID: root.id,
  taskParentID: root.id,
} satisfies SessionInfo

async function wait(fn: () => boolean | Promise<boolean>, timeout = 3000) {
  const start = Date.now()
  while (!(await fn())) {
    if (Date.now() - start > timeout) throw new Error("timed out waiting for condition")
    await Bun.sleep(10)
  }
}

async function waitForText(app: Awaited<ReturnType<typeof testRender>>, text: string) {
  const start = Date.now()
  while (true) {
    await app.renderOnce()
    const frame = app.captureCharFrame()
    if (frame.includes(text)) return frame
    if (Date.now() - start > 3000) throw new Error(`timed out waiting for ${text}\n${frame}`)
    await Bun.sleep(10)
  }
}

async function mountRoute(dir: string, view: () => JSX.Element, handler: Handler = () => undefined) {
  const state = path.join(dir, "state")
  await mkdir(state, { recursive: true })
  await Bun.write(path.join(state, "kv.json"), "{}")
  const events = createEventSource()
  const fallback = createFetch(undefined, events)
  const fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const request = input instanceof Request ? input : new Request(input, init)
    const url = new URL(request.url)
    const result = await handler(request)
    if (result) return result
    if (url.pathname === "/session" && request.method === "GET") return json([root, child])
    if (url.pathname === `/session/${root.id}`) return json(root)
    if (url.pathname === `/session/${root.id}/message`) return json([])
    if (url.pathname.endsWith("/todo") || url.pathname.endsWith("/diff")) return json([])
    return fallback.fetch(input, init)
  }) as typeof globalThis.fetch
  let sync!: ReturnType<typeof useSync>
  let keymap!: ReturnType<typeof createDefaultOpenTuiKeymap>

  function Probe() {
    sync = useSync()
    return view()
  }

  // Keep the provider order from app.tsx so dialogs and prompts see the same contexts as production.
  function Harness() {
    const renderer = useRenderer()
    keymap = createDefaultOpenTuiKeymap(renderer)
    const config = createTuiResolvedConfig()
    const off = registerOpencodeKeymap(keymap, renderer, config)
    onCleanup(off)
    return (
      <TestTuiContexts directory={dir} paths={{ home: dir, state, worktree: dir }}>
        <ExitProvider exit={() => {}}>
          <EpilogueProvider set={() => {}}>
            <OpencodeKeymapProvider keymap={keymap}>
              <ArgsProvider>
                <KVProvider>
                  <ToastProvider>
                    <RouteProvider initialRoute={{ type: "session", sessionID: root.id }}>
                      <TuiConfigProvider config={config}>
                        <PluginRuntimeProvider value={createPluginRuntime()}>
                          <SDKProvider url="http://test" directory={directory} fetch={fetch} events={events.source}>
                            <PermissionProvider>
                              <ProjectProvider>
                                <SyncProvider>
                                  <DataProvider>
                                    <ThemeProvider mode="dark">
                                      <LocalProvider>
                                        <PromptStashProvider>
                                          <DialogProvider>
                                            <FrecencyProvider>
                                              <PromptHistoryProvider>
                                                <PromptRefProvider>
                                                  <EditorContextProvider integration={{}}>
                                                    <LocationProvider>
                                                      <Probe />
                                                    </LocationProvider>
                                                  </EditorContextProvider>
                                                </PromptRefProvider>
                                              </PromptHistoryProvider>
                                            </FrecencyProvider>
                                          </DialogProvider>
                                        </PromptStashProvider>
                                      </LocalProvider>
                                    </ThemeProvider>
                                  </DataProvider>
                                </SyncProvider>
                              </ProjectProvider>
                            </PermissionProvider>
                          </SDKProvider>
                        </PluginRuntimeProvider>
                      </TuiConfigProvider>
                    </RouteProvider>
                  </ToastProvider>
                </KVProvider>
              </ArgsProvider>
            </OpencodeKeymapProvider>
          </EpilogueProvider>
        </ExitProvider>
      </TestTuiContexts>
    )
  }

  const app = await testRender(() => <Harness />, { width: 100, height: 40, kittyKeyboard: true })
  await wait(() => sync?.status === "complete")
  return {
    app,
    sync,
    dispatch: (command: string) => keymap.dispatchCommand(command),
    emit: (payload: Event) => events.emit({ directory, project: "proj_test", payload } satisfies GlobalEvent),
  }
}

test("mounts a fresh permission prompt when the first request changes", async () => {
  await using tmp = await tmpdir()
  const replies: Record<string, unknown>[] = []
  const mounted = await mountRoute(
    tmp.path,
    () => <Session />,
    async (request) => {
      const url = new URL(request.url)
      if (!url.pathname.startsWith("/permission/")) return
      replies.push({ id: url.pathname.split("/")[2], ...((await request.clone().json()) as Record<string, unknown>) })
      return json(true)
    },
  )
  const permission = (id: string, filePath: string) =>
    mounted.emit({
      id: `evt_${id}`,
      type: "permission.asked",
      properties: {
        id,
        sessionID: child.id,
        permission: "read",
        patterns: [filePath],
        always: [filePath],
        metadata: { filePath },
      },
    })

  try {
    permission("per_a", "src/a.ts")
    permission("per_b", "src/b.ts")
    await waitForText(mounted.app, "Read src/a.ts")
    mounted.app.mockInput.pressArrow("right")
    mounted.app.mockInput.pressArrow("right")
    mounted.app.mockInput.pressEnter()
    await waitForText(mounted.app, "Reject permission")
    await mounted.app.mockInput.typeText("feedback for A")
    await waitForText(mounted.app, "feedback for A")

    mounted.emit({
      id: "evt_per_a_replied",
      type: "permission.replied",
      properties: { sessionID: child.id, requestID: "per_a", reply: "once" },
    })
    const frame = await waitForText(mounted.app, "Read src/b.ts")
    expect(frame).not.toContain("Reject permission")
    expect(frame).not.toContain("feedback for A")

    mounted.app.mockInput.pressEnter()
    await wait(() => replies.length === 1)
    expect(replies[0]).toMatchObject({ id: "per_b", reply: "once" })
  } finally {
    mounted.app.renderer.destroy()
  }
})

test("mounts a fresh question prompt when the first request changes", async () => {
  await using tmp = await tmpdir()
  const mounted = await mountRoute(tmp.path, () => <Session />)
  const question = (id: string, name: string) =>
    mounted.emit({
      id: `evt_${id}`,
      type: "question.asked",
      properties: {
        id,
        sessionID: child.id,
        questions: ["First", "Second"].map((order) => ({
          question: `${order} question ${name}?`,
          header: `${order} ${name}`,
          options: [{ label: "Yes", description: "Continue" }],
        })),
      },
    })

  try {
    question("que_a", "A")
    question("que_b", "B")
    await waitForText(mounted.app, "First question A?")
    mounted.app.mockInput.pressArrow("right")
    await waitForText(mounted.app, "Second question A?")

    mounted.emit({
      id: "evt_que_a_replied",
      type: "question.replied",
      properties: { sessionID: child.id, requestID: "que_a", answers: [] },
    })
    const frame = await waitForText(mounted.app, "First question B?")
    expect(frame).not.toContain("Second question B?")
  } finally {
    mounted.app.renderer.destroy()
  }
})

test("shows the interrupt hint while subagents run under an idle root", async () => {
  await using tmp = await tmpdir()
  const mounted = await mountRoute(tmp.path, () => <Prompt sessionID={root.id} visible interruptible />)

  try {
    await waitForText(mounted.app, "esc interrupt")
  } finally {
    mounted.app.renderer.destroy()
  }
})

test("aborts busy Task children in another location on the second escape", async () => {
  await using tmp = await tmpdir()
  const moved = { ...root, directory: path.join(directory, "moved") }
  const aborted: string[] = []
  const mounted = await mountRoute(
    tmp.path,
    () => <Prompt sessionID={root.id} visible interruptible />,
    (request) => {
      const url = new URL(request.url)
      if (url.pathname === "/session" && request.method === "GET") return json([moved, child])
      if (url.pathname === `/session/${root.id}`) return json(moved)
      if (url.pathname === "/session/status") return json({ [child.id]: { type: "busy" } })
      const abort = url.pathname.match(/^\/session\/([^/]+)\/abort$/)
      if (!abort) return
      aborted.push(abort[1])
      return json(true)
    },
  )

  try {
    await waitForText(mounted.app, "esc interrupt")
    mounted.app.mockInput.pressEscape()
    mounted.app.mockInput.pressEscape()
    await wait(() => aborted.length === 2)
    expect(aborted.toSorted()).toEqual([child.id, root.id].toSorted())
  } finally {
    mounted.app.renderer.destroy()
  }
})

test("undo reverts a busy session without aborting it", async () => {
  await using tmp = await tmpdir()
  const requests: string[] = []
  const user = {
    id: "msg_user",
    sessionID: root.id,
    role: "user",
    agent: "build",
    model: { providerID: "test", modelID: "model" },
    time: { created: 1 },
  } satisfies UserMessage
  const text = {
    id: "prt_user",
    sessionID: root.id,
    messageID: user.id,
    type: "text",
    text: "change the file",
  } satisfies TextPart
  const assistant = {
    id: "msg_assistant",
    sessionID: root.id,
    role: "assistant",
    agent: "build",
    modelID: "model",
    providerID: "test",
    mode: "build",
    parentID: user.id,
    path: { cwd: directory, root: directory },
    cost: 0,
    tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
    time: { created: 2 },
  } satisfies AssistantMessage
  const mounted = await mountRoute(
    tmp.path,
    () => <Session />,
    (request) => {
      const url = new URL(request.url)
      if (url.pathname === "/session/status") return json({ [root.id]: { type: "busy" } })
      if (url.pathname === `/session/${root.id}/message`)
        return json([
          { info: user, parts: [text] },
          { info: assistant, parts: [] },
        ])
      if (url.pathname === `/session/${root.id}/abort`) {
        requests.push("abort")
        return json(true)
      }
      if (url.pathname !== `/session/${root.id}/revert`) return
      requests.push("revert")
      return json(root)
    },
  )

  try {
    await wait(() => mounted.sync.data.message[root.id]?.length === 2)
    expect(mounted.sync.data.session_status[root.id]).toEqual({ type: "busy" })
    mounted.dispatch("session.undo")
    await wait(() => requests.includes("revert"))
    // Revert stops the running turn on the server. Abort would also cancel every background Task.
    expect(requests).toEqual(["revert"])
  } finally {
    mounted.app.renderer.destroy()
  }
})

test("ignores subagent session loads aborted by quitting", async () => {
  await using tmp = await tmpdir()
  let childLoads = 0
  const message = {
    id: "msg_task",
    sessionID: root.id,
    role: "assistant",
    agent: "build",
    modelID: "model",
    providerID: "test",
    mode: "build",
    parentID: "msg_user",
    path: { cwd: directory, root: directory },
    cost: 0,
    tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
    time: { created: 2 },
  } satisfies AssistantMessage
  const part = {
    id: "prt_task",
    sessionID: root.id,
    messageID: message.id,
    type: "tool",
    callID: "call_task",
    tool: "task",
    state: { status: "running", input: {}, metadata: { sessionId: child.id }, time: { start: 2 } },
  } satisfies ToolPart
  const mounted = await mountRoute(
    tmp.path,
    () => <Session />,
    (request) => {
      const url = new URL(request.url)
      if (url.pathname === `/session/${root.id}/message`) return json([{ info: message, parts: [part] }])
      if (url.pathname !== `/session/${child.id}`) return
      childLoads++
      return new Promise<Response>((_resolve, reject) => {
        request.signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")), {
          once: true,
        })
      })
    },
  )

  try {
    await wait(() => childLoads > 0)
    mounted.app.renderer.destroy()
    // An unhandled rejection fails the running bun test. In the TUI process it exits with code 1.
    await Bun.sleep(50)
  } finally {
    if (!mounted.app.renderer.isDestroyed) mounted.app.renderer.destroy()
  }
})
