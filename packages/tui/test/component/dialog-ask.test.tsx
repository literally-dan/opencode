/** @jsxImportSource @opentui/solid */
import { ScrollBoxRenderable, TextareaRenderable, type Renderable } from "@opentui/core"
import { createDefaultOpenTuiKeymap } from "@opentui/keymap/opentui"
import { testRender, useRenderer } from "@opentui/solid"
import type { Event, GlobalEvent, PermissionRequest } from "@opencode-ai/sdk/v2"
import { describe, expect, test } from "bun:test"
import { mkdir } from "node:fs/promises"
import path from "node:path"
import { onCleanup, onMount } from "solid-js"
import { ArgsProvider } from "../../src/context/args"
import { DataProvider } from "../../src/context/data"
import { EditorContextProvider } from "../../src/context/editor"
import { ExitProvider } from "../../src/context/exit"
import { KVProvider } from "../../src/context/kv"
import { LocationProvider } from "../../src/context/location"
import { PermissionProvider } from "../../src/context/permission"
import { ProjectProvider } from "../../src/context/project"
import { RouteProvider } from "../../src/context/route"
import { SDKProvider } from "../../src/context/sdk"
import { SyncProvider, useSync } from "../../src/context/sync"
import { ThemeProvider } from "../../src/context/theme"
import { TuiConfigProvider } from "../../src/config"
import { Prompt, type PromptRef } from "../../src/component/prompt"
import {
  AskToolActivityView,
  DialogAsk,
  mergeAskThreads,
  mergeAskToolActivity,
  mergeAskTurns,
  reconcileAskToolActivity,
  type DialogAskProps,
} from "../../src/component/dialog-ask"
import { OpencodeKeymapProvider, registerOpencodeKeymap } from "../../src/keymap"
import { LocalProvider } from "../../src/context/local"
import { FrecencyProvider } from "../../src/prompt/frecency"
import { PromptHistoryProvider } from "../../src/prompt/history"
import { PromptStashProvider } from "../../src/prompt/stash"
import { askPermissionMetadata, isAskPermission, normalPermissionCallID } from "../../src/routes/session/permission"
import { DialogProvider, useDialog } from "../../src/ui/dialog"
import { ToastProvider } from "../../src/ui/toast"
import { tmpdir } from "../fixture/fixture"
import { TestTuiContexts } from "../fixture/tui-environment"
import { createEventSource, createFetch, directory, json } from "../fixture/tui-sdk"
import { createTuiResolvedConfig } from "../fixture/tui-runtime"

const props = {
  sessionID: "ses_ask",
  question: "How does this work?",
  agent: "build",
  model: { providerID: "test", modelID: "model" },
} satisfies DialogAskProps

type Handler = (request: Request) => Response | Promise<Response | undefined> | undefined

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

async function mountAsk(root: string, handler: Handler) {
  const state = path.join(root, "state")
  await mkdir(state, { recursive: true })
  await Bun.write(path.join(state, "kv.json"), "{}")
  const events = createEventSource()
  const fallback = createFetch(undefined, events)
  const fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const request = input instanceof Request ? input : new Request(input, init)
    const result = await handler(request)
    return result ?? fallback.fetch(input, init)
  }) as typeof globalThis.fetch
  let sync!: ReturnType<typeof useSync>

  function OpenAsk() {
    const dialog = useDialog()
    sync = useSync()
    onMount(() => dialog.replace(() => <DialogAsk {...props} />))
    return <box />
  }

  function Harness() {
    const renderer = useRenderer()
    const keymap = createDefaultOpenTuiKeymap(renderer)
    const config = createTuiResolvedConfig()
    const off = registerOpencodeKeymap(keymap, renderer, config)
    onCleanup(off)
    return (
      <TestTuiContexts directory={root} paths={{ home: root, state, worktree: root }}>
        <OpencodeKeymapProvider keymap={keymap}>
          <ArgsProvider>
            <KVProvider>
              <ToastProvider>
                <TuiConfigProvider config={config}>
                  <SDKProvider url="http://test" directory={directory} fetch={fetch} events={events.source}>
                    <PermissionProvider>
                      <ProjectProvider>
                        <ExitProvider exit={() => {}}>
                          <SyncProvider>
                            <ThemeProvider mode="dark">
                              <LocationProvider location={{ directory }}>
                                <DialogProvider>
                                  <OpenAsk />
                                </DialogProvider>
                              </LocationProvider>
                            </ThemeProvider>
                          </SyncProvider>
                        </ExitProvider>
                      </ProjectProvider>
                    </PermissionProvider>
                  </SDKProvider>
                </TuiConfigProvider>
              </ToastProvider>
            </KVProvider>
          </ArgsProvider>
        </OpencodeKeymapProvider>
      </TestTuiContexts>
    )
  }

  const app = await testRender(() => <Harness />, { width: 120, height: 40, kittyKeyboard: true })
  await wait(() => sync?.status === "complete")
  return { app, events }
}

async function mountPromptAsk(root: string, handler: Handler) {
  const state = path.join(root, "state")
  await mkdir(state, { recursive: true })
  await Bun.write(path.join(state, "kv.json"), "{}")
  const events = createEventSource()
  const fallback = createFetch(undefined, events)
  const fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const request = input instanceof Request ? input : new Request(input, init)
    const result = await handler(request)
    return result ?? fallback.fetch(input, init)
  }) as typeof globalThis.fetch
  let sync!: ReturnType<typeof useSync>
  let prompt!: PromptRef

  function PromptHarness() {
    sync = useSync()
    return (
      <Prompt
        sessionID="ses_prompt"
        visible
        ref={(value) => {
          if (value) prompt = value
        }}
      />
    )
  }

  function Harness() {
    const renderer = useRenderer()
    const keymap = createDefaultOpenTuiKeymap(renderer)
    const config = createTuiResolvedConfig()
    const off = registerOpencodeKeymap(keymap, renderer, config)
    onCleanup(off)
    return (
      <TestTuiContexts directory={root} paths={{ home: root, state, worktree: root }}>
        <OpencodeKeymapProvider keymap={keymap}>
          <ArgsProvider>
            <KVProvider>
              <ToastProvider>
                <RouteProvider initialRoute={{ type: "session", sessionID: "ses_prompt" }}>
                  <TuiConfigProvider config={config}>
                    <SDKProvider url="http://test" directory={directory} fetch={fetch} events={events.source}>
                      <PermissionProvider>
                        <ProjectProvider>
                          <ExitProvider exit={() => {}}>
                            <SyncProvider>
                              <DataProvider>
                                <ThemeProvider mode="dark">
                                  <LocalProvider>
                                    <PromptStashProvider>
                                      <FrecencyProvider>
                                        <PromptHistoryProvider>
                                          <EditorContextProvider integration={{}}>
                                            <LocationProvider location={{ directory }}>
                                              <DialogProvider>
                                                <PromptHarness />
                                              </DialogProvider>
                                            </LocationProvider>
                                          </EditorContextProvider>
                                        </PromptHistoryProvider>
                                      </FrecencyProvider>
                                    </PromptStashProvider>
                                  </LocalProvider>
                                </ThemeProvider>
                              </DataProvider>
                            </SyncProvider>
                          </ExitProvider>
                        </ProjectProvider>
                      </PermissionProvider>
                    </SDKProvider>
                  </TuiConfigProvider>
                </RouteProvider>
              </ToastProvider>
            </KVProvider>
          </ArgsProvider>
        </OpencodeKeymapProvider>
      </TestTuiContexts>
    )
  }

  const app = await testRender(() => <Harness />, { width: 120, height: 40, kittyKeyboard: true })
  await wait(() => sync?.status === "complete" && prompt !== undefined)
  return { app, prompt }
}

function emit(events: ReturnType<typeof createEventSource>, payload: Event) {
  events.emit({ directory, project: "proj_test", payload } satisfies GlobalEvent)
}

function findConversationScroll(root: Renderable) {
  const scrolls: ScrollBoxRenderable[] = []
  const visit = (renderable: Renderable) => {
    if (renderable instanceof ScrollBoxRenderable) scrolls.push(renderable)
    for (const child of renderable.getChildren()) visit(child)
  }
  visit(root)
  return scrolls.at(-1)
}

function firstTurn(requestID: string) {
  return {
    id: "atn_new_1",
    threadID: "ask_new",
    requestID,
    question: props.question,
    answer: "The first answer",
    text: "The first answer",
    toolActivity: [],
    time: { created: 300 },
  }
}

describe("DialogAsk", () => {
  test("creates, browses, paginates, follows up, and expands persisted tool activity", async () => {
    await using tmp = await tmpdir()
    const asks: Record<string, unknown>[] = []
    const threadQueries: URL[] = []
    const turnQueries: URL[] = []
    let finishInitial!: (response: Response) => void
    const initial = new Promise<Response>((resolve) => {
      finishInitial = resolve
    })

    const mounted = await mountAsk(tmp.path, async (request) => {
      const url = new URL(request.url)
      if (url.pathname === "/session/ses_ask/ask" && request.method === "GET") {
        threadQueries.push(url)
        if (url.searchParams.has("before")) return json({ items: [], more: false })
        return json({
          items: [
            {
              id: "ask_old",
              sessionID: "ses_ask",
              title: "Prior question",
              time: { created: 100, updated: 100 },
            },
          ],
          more: true,
          cursor: "thread_cursor",
        })
      }
      if (url.pathname === "/session/ses_ask/ask" && request.method === "POST") {
        asks.push((await request.clone().json()) as Record<string, unknown>)
        if (asks.length === 1) return initial
        return json({
          id: "atn_new_2",
          threadID: "ask_new",
          requestID: asks[1]?.requestID,
          question: "What next?",
          answer: "The follow-up answer",
          text: "The follow-up answer",
          toolActivity: [
            {
              callID: "call_read",
              tool: "read",
              status: "completed",
              input: '{"filePath":"src/index.ts"}',
              output: "bounded result",
            },
          ],
          time: { created: 400 },
        })
      }
      if (url.pathname === "/session/ses_ask/ask/ask_old" && request.method === "GET") {
        turnQueries.push(url)
        if (url.searchParams.has("before")) {
          return json({
            items: [
              {
                id: "atn_old_0",
                threadID: "ask_old",
                question: "Older question",
                answer: "Older answer",
                toolActivity: [],
                time: { created: 50 },
              },
            ],
            more: false,
          })
        }
        return json({
          items: [
            {
              id: "atn_old_1",
              threadID: "ask_old",
              question: "Prior question",
              answer: "Prior answer",
              toolActivity: [
                {
                  callID: "call_old",
                  tool: "grep",
                  status: "error",
                  input: '{"pattern":"missing"}',
                  error: "persisted failure",
                },
              ],
              time: { created: 100 },
            },
          ],
          more: true,
          cursor: "turn_cursor",
        })
      }
      return undefined
    })

    try {
      expect(await waitForText(mounted.app, "Thinking")).toContain(props.question)
      await wait(() => asks.length === 1)
      finishInitial(json(firstTurn(String(asks[0]?.requestID))))
      let frame = await waitForText(mounted.app, "The first answer")
      expect(frame).toContain("How does this work?")
      expect(asks[0]?.threadID).toBeUndefined()
      expect(typeof asks[0]?.requestID).toBe("string")
      expect(String(asks[0]?.requestID).length).toBeLessThanOrEqual(128)
      await wait(() => {
        void mounted.app.renderOnce()
        return !mounted.app.captureCharFrame().includes("Loading threads")
      })
      await wait(() => mounted.app.renderer.currentFocusedEditor instanceof TextareaRenderable)

      mounted.app.mockInput.pressKey("h", { ctrl: true })
      mounted.app.mockInput.pressKey("o")
      await wait(() => threadQueries.some((url) => url.searchParams.get("before") === "thread_cursor"))

      mounted.app.mockInput.pressArrow("down")
      frame = await waitForText(mounted.app, "Prior answer")
      expect(frame).toContain("Prior question")
      expect(frame).toContain("grep")
      expect(frame).toContain("error")
      mounted.app.mockInput.pressArrow("right")
      mounted.app.mockInput.pressEnter()
      expect(await waitForText(mounted.app, "persisted failure")).toContain('{"pattern":"missing"}')
      mounted.app.mockInput.pressKey("o")
      await wait(() => turnQueries.some((url) => url.searchParams.get("before") === "turn_cursor"))

      mounted.app.mockInput.pressArrow("left")
      mounted.app.mockInput.pressArrow("up")
      mounted.app.mockInput.pressTab()
      mounted.app.mockInput.pressTab()
      await wait(() => mounted.app.renderer.currentFocusedEditor instanceof TextareaRenderable)
      const composer = mounted.app.renderer.currentFocusedEditor
      if (!(composer instanceof TextareaRenderable)) throw new Error("expected Ask composer")
      composer.setText("What next?")
      mounted.app.mockInput.pressEnter()

      frame = await waitForText(mounted.app, "The follow-up answer")
      expect(asks[1]?.threadID).toBe("ask_new")
      expect(asks[1]?.requestID).not.toBe(asks[0]?.requestID)
      expect(frame).toContain("read")
      expect(frame).toContain("completed")
      expect(frame).not.toContain("bounded result")

      mounted.app.mockInput.pressKey("l", { ctrl: true })
      mounted.app.mockInput.pressEnter()
      frame = await waitForText(mounted.app, "bounded result")
      expect(frame).toContain('{"filePath":"src/index.ts"}')
    } finally {
      mounted.app.renderer.destroy()
    }
  })

  test("shows loading failures and retries thread and Ask requests", async () => {
    await using tmp = await tmpdir()
    let threads = 0
    let asks = 0
    const requestIDs: string[] = []
    let failAsk!: (response: Response) => void
    const pendingAsk = new Promise<Response>((resolve) => {
      failAsk = resolve
    })
    const mounted = await mountAsk(tmp.path, async (request) => {
      const url = new URL(request.url)
      if (url.pathname !== "/session/ses_ask/ask") return
      if (request.method === "GET") {
        threads++
        if (threads === 1) throw new Error("thread load failed")
        return json({ items: [], more: false })
      }
      asks++
      const body = (await request.clone().json()) as Record<string, unknown>
      requestIDs.push(String(body.requestID))
      if (asks === 1) return pendingAsk
      return json(firstTurn(String(body.requestID)))
    })

    try {
      let frame = await waitForText(mounted.app, "thread load failed")
      expect(frame).toContain("Thinking")
      mounted.app.mockInput.pressKey("r")
      await wait(() => threads === 2)

      failAsk(json({ message: "ask failed" }, { status: 502 }))
      frame = await waitForText(mounted.app, "ask failed")
      expect(frame).toContain("retry")
      emit(mounted.events, {
        id: "evt_after_error",
        type: "session.ask.tool.activity",
        properties: {
          sessionID: "ses_ask",
          requestID: requestIDs[0],
          threadID: "ask_failed",
          turnID: "atn_failed",
          callID: "call_after_error",
          tool: "after-error",
          status: "running",
          input: "{}",
        },
      })
      await Bun.sleep(30)
      await mounted.app.renderOnce()
      expect(mounted.app.captureCharFrame()).not.toContain("after-error")
      mounted.app.mockInput.pressKey("r")
      expect(await waitForText(mounted.app, "The first answer")).toContain("How does this work?")
      expect(asks).toBe(2)
      expect(requestIDs[1]).not.toBe(requestIDs[0])
    } finally {
      mounted.app.renderer.destroy()
    }
  })

  test("preserves the visible turn anchor when older pages are prepended", async () => {
    await using tmp = await tmpdir()
    const turnQueries: URL[] = []
    const mounted = await mountAsk(tmp.path, async (request) => {
      const url = new URL(request.url)
      if (url.pathname === "/session/ses_ask/ask" && request.method === "POST") {
        const body = (await request.clone().json()) as Record<string, unknown>
        return json(firstTurn(String(body.requestID)))
      }
      if (url.pathname === "/session/ses_ask/ask" && request.method === "GET") {
        return json({
          items: [
            {
              id: "ask_old",
              sessionID: "ses_ask",
              title: "Long history",
              time: { created: 100, updated: 100 },
            },
          ],
          more: false,
        })
      }
      if (url.pathname === "/session/ses_ask/ask/ask_old" && request.method === "GET") {
        turnQueries.push(url)
        if (url.searchParams.has("before")) {
          return json({
            items: Array.from({ length: 3 }, (_, index) => ({
              id: `atn_old_${index}`,
              threadID: "ask_old",
              question: `Older question ${index}`,
              answer: Array.from({ length: 8 }, (_, line) => `Older answer ${index}.${line} ${"x".repeat(100)}`).join(
                "\n\n",
              ),
              toolActivity: [],
              time: { created: index + 1 },
            })),
            more: false,
          })
        }
        return json({
          items: Array.from({ length: 8 }, (_, index) => ({
            id: `atn_latest_${index}`,
            threadID: "ask_old",
            question: `Latest question ${index}`,
            answer: Array.from({ length: 8 }, (_, line) => `Latest answer ${index}.${line} ${"x".repeat(100)}`).join(
              "\n\n",
            ),
            toolActivity: [],
            time: { created: 100 + index },
          })),
          more: true,
          cursor: "turn_cursor",
        })
      }
      return undefined
    })

    try {
      await waitForText(mounted.app, "The first answer")
      await wait(() => mounted.app.renderer.currentFocusedEditor instanceof TextareaRenderable)
      mounted.app.mockInput.pressKey("h", { ctrl: true })
      mounted.app.mockInput.pressArrow("down")
      await waitForText(mounted.app, "Latest answer 7.7")
      const scroll = findConversationScroll(mounted.app.renderer.root)
      if (!scroll) throw new Error("expected Ask conversation scrollbox")
      await mounted.app.renderOnce()
      expect(scroll.scrollHeight).toBeGreaterThan(scroll.viewport.height)
      await wait(() => scroll.scrollTop > 0)
      expect(scroll.scrollTop).toBe(scroll.scrollHeight - scroll.viewport.height)
      await wait(() => {
        void mounted.app.renderOnce()
        return !mounted.app.captureCharFrame().includes("Loading conversation")
      })

      scroll.scrollTo(0)
      mounted.app.mockInput.pressKey("l", { ctrl: true })
      await Bun.sleep(10)
      const paginationTop = scroll.scrollTop
      const paginationHeight = scroll.scrollHeight
      mounted.app.mockInput.pressKey("o")
      await wait(() => turnQueries.some((query) => query.searchParams.get("before") === "turn_cursor")).catch(() => {
        throw new Error(`turn queries: ${turnQueries.map((query) => query.search).join(", ")}`)
      })
      await wait(async () => {
        await mounted.app.renderOnce()
        return scroll.scrollHeight > paginationHeight
      }).catch(() => {
        throw new Error(`scroll height did not grow: ${paginationHeight} -> ${scroll.scrollHeight}`)
      })
      await wait(async () => {
        await mounted.app.renderOnce()
        return scroll.scrollTop === paginationTop + scroll.scrollHeight - paginationHeight
      }).catch(() => {
        throw new Error(
          `scroll anchor moved: top=${scroll.scrollTop}, expected=${paginationTop + scroll.scrollHeight - paginationHeight}`,
        )
      })
      expect(scroll.scrollTop).toBe(paginationTop + scroll.scrollHeight - paginationHeight)
    } finally {
      mounted.app.renderer.destroy()
    }
  })

  test("keeps keyboard selection aligned after a thread refresh reorders the list", async () => {
    await using tmp = await tmpdir()
    let threadLoads = 0
    let activeRequestID = ""
    let finishAsk!: (response: Response) => void
    const pendingAsk = new Promise<Response>((resolve) => {
      finishAsk = resolve
    })
    const mounted = await mountAsk(tmp.path, async (request) => {
      const url = new URL(request.url)
      if (url.pathname === "/session/ses_ask/ask" && request.method === "GET") {
        threadLoads++
        return json({
          items: [
            {
              id: "ask_old",
              sessionID: "ses_ask",
              title: "Reordered thread",
              time: { created: 100, updated: threadLoads === 1 ? 100 : 500 },
            },
          ],
          more: false,
        })
      }
      if (url.pathname === "/session/ses_ask/ask" && request.method === "POST") {
        const body = (await request.clone().json()) as Record<string, unknown>
        activeRequestID = String(body.requestID)
        return pendingAsk
      }
      if (url.pathname === "/session/ses_ask/ask/ask_old" && request.method === "GET") {
        return json({
          items: [
            {
              id: "atn_old",
              threadID: "ask_old",
              question: "Reordered question",
              answer: "Reordered answer",
              toolActivity: [],
              time: { created: 100 },
            },
          ],
          more: false,
        })
      }
      return undefined
    })

    try {
      await wait(() => threadLoads === 1 && activeRequestID.length > 0)
      await Bun.sleep(20)
      finishAsk(json(firstTurn(activeRequestID)))
      await waitForText(mounted.app, "The first answer")
      await wait(() => threadLoads === 2)
      await mounted.app.renderOnce()

      mounted.app.mockInput.pressKey("h", { ctrl: true })
      mounted.app.mockInput.pressArrow("up")
      expect(await waitForText(mounted.app, "Reordered answer")).toContain("Reordered question")
    } finally {
      mounted.app.renderer.destroy()
    }
  })

  test("blocks retry until a cancelled Ask follow-up terminates", async () => {
    await using tmp = await tmpdir()
    const requests: string[] = []
    const asks: Record<string, unknown>[] = []
    let aborts = 0
    let finishAbort!: () => void
    const mounted = await mountAsk(tmp.path, async (request) => {
      const url = new URL(request.url)
      requests.push(`${request.method} ${url.pathname}`)
      if (url.pathname === "/session/ses_ask/ask" && request.method === "GET") {
        return json({ items: [], more: false })
      }
      if (url.pathname === "/session/ses_ask/ask" && request.method === "POST") {
        const body = (await request.clone().json()) as Record<string, unknown>
        asks.push(body)
        if (asks.length === 1) return json(firstTurn(String(body.requestID)))
        if (asks.length === 2) {
          return new Promise<Response>((_resolve, reject) => {
            finishAbort = () => reject(new DOMException("aborted", "AbortError"))
            request.signal.addEventListener("abort", () => aborts++, { once: true })
          })
        }
        return json({
          id: "atn_retry",
          threadID: "ask_new",
          requestID: body.requestID,
          question: body.question,
          answer: "The retry answer",
          text: "The retry answer",
          toolActivity: [],
          time: { created: 400 },
        })
      }
      return undefined
    })

    try {
      await waitForText(mounted.app, "The first answer")
      await wait(() => mounted.app.renderer.currentFocusedEditor instanceof TextareaRenderable)
      const composer = mounted.app.renderer.currentFocusedEditor
      if (!(composer instanceof TextareaRenderable)) throw new Error("expected Ask composer")
      composer.setText("Keep going")
      mounted.app.mockInput.pressEnter()
      await waitForText(mounted.app, "Answering")
      await wait(() => asks.length === 2)
      mounted.app.mockInput.pressKey("c")
      mounted.app.mockInput.pressKey("c")
      expect(await waitForText(mounted.app, "Cancelling Ask")).toContain("Keep going")
      expect(aborts).toBe(1)
      mounted.app.mockInput.pressKey("r")
      await Bun.sleep(30)
      expect(asks).toHaveLength(2)
      expect((await waitForText(mounted.app, "Cancelling Ask")).includes("Ask cancelled")).toBe(false)
      expect(requests.some((request) => request.endsWith("/cancel"))).toBe(false)
      expect(requests).not.toContain("POST /session/ses_ask/abort")

      finishAbort()
      expect(await waitForText(mounted.app, "Ask cancelled")).toContain("Keep going")
      mounted.app.mockInput.pressKey("r")
      expect(await waitForText(mounted.app, "The retry answer")).toContain("Keep going")
      expect(asks).toHaveLength(3)
      expect(asks[2]?.threadID).toBe("ask_new")
      expect(asks[1]?.requestID).not.toBe(asks[0]?.requestID)
      expect(asks[2]?.requestID).not.toBe(asks[1]?.requestID)
      emit(mounted.events, {
        id: "evt_after_cancel",
        type: "session.ask.tool.activity",
        properties: {
          sessionID: "ses_ask",
          requestID: String(asks[1]?.requestID),
          threadID: "ask_new",
          turnID: "atn_cancelled",
          callID: "call_after_cancel",
          tool: "after-cancel",
          status: "running",
          input: "{}",
        },
      })
      await Bun.sleep(30)
      await mounted.app.renderOnce()
      expect(mounted.app.captureCharFrame()).not.toContain("after-cancel")
    } finally {
      mounted.app.renderer.destroy()
    }
  })

  test("aborts an initial Ask once without a cancellation endpoint", async () => {
    await using tmp = await tmpdir()
    const requests: string[] = []
    let activeRequestID = ""
    let aborts = 0
    const mounted = await mountAsk(tmp.path, async (request) => {
      const url = new URL(request.url)
      requests.push(`${request.method} ${url.pathname}`)
      if (url.pathname === "/session/ses_ask/ask" && request.method === "GET") {
        return json({ items: [], more: false })
      }
      if (url.pathname === "/session/ses_ask/ask" && request.method === "POST") {
        const body = (await request.clone().json()) as Record<string, unknown>
        activeRequestID = String(body.requestID)
        return new Promise<Response>((_resolve, reject) => {
          request.signal.addEventListener(
            "abort",
            () => {
              aborts++
              reject(new DOMException("aborted", "AbortError"))
            },
            { once: true },
          )
        })
      }
      return undefined
    })

    try {
      await waitForText(mounted.app, "Answering")
      await wait(() => activeRequestID.length > 0)
      mounted.app.mockInput.pressKey("c")
      mounted.app.mockInput.pressKey("c")
      expect(await waitForText(mounted.app, "Ask cancelled")).toContain(props.question)
      expect(aborts).toBe(1)
      expect(requests).not.toContain("POST /session/ses_ask/abort")
      expect(requests.some((request) => request.endsWith("/cancel"))).toBe(false)
      expect(activeRequestID.length).toBeLessThanOrEqual(128)
    } finally {
      mounted.app.renderer.destroy()
    }
  })

  test("isolates Ask permissions and handles modal allow, always, reject, and Escape", async () => {
    await using tmp = await tmpdir()
    const replies: Record<string, unknown>[] = []
    let activeRequestID = ""
    const mounted = await mountAsk(tmp.path, async (request) => {
      const url = new URL(request.url)
      if (url.pathname === "/session/ses_ask/ask" && request.method === "GET") {
        return json({ items: [], more: false })
      }
      if (url.pathname === "/session/ses_ask/ask" && request.method === "POST") {
        const body = (await request.clone().json()) as Record<string, unknown>
        activeRequestID = String(body.requestID)
        return new Promise<Response>((_resolve, reject) => {
          request.signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")), {
            once: true,
          })
        })
      }
      if (url.pathname.startsWith("/permission/") && url.pathname.endsWith("/reply")) {
        replies.push({
          id: url.pathname.split("/")[2],
          ...((await request.clone().json()) as Record<string, unknown>),
        })
        return json(true)
      }
      return undefined
    })

    try {
      await wait(() => activeRequestID.length > 0)
      emit(mounted.events, {
        id: "evt_activity",
        type: "session.ask.tool.activity",
        properties: {
          sessionID: "ses_ask",
          requestID: activeRequestID,
          threadID: "ask_pending",
          turnID: "atn_pending",
          callID: "call_activity",
          tool: "read",
          status: "running",
          input: "{}",
        },
      })

      const permission = (id: string, requestID: string, turnID = "atn_pending", threadID = "ask_pending") =>
        emit(mounted.events, {
          id: `evt_${id}`,
          type: "permission.asked",
          properties: {
            id,
            sessionID: "ses_ask",
            permission: "read",
            patterns: ["src/index.ts"],
            always: ["src/index.ts"],
            metadata: {
              filepath: "src/index.ts",
              ask: {
                requestID,
                threadID,
                turnID,
                callID: `call_${id}`,
                tool: "read",
              },
            },
          },
        })
      const replied = (id: string, reply: "once" | "always" | "reject") =>
        emit(mounted.events, {
          id: `evt_${id}_reply`,
          type: "permission.replied",
          properties: { sessionID: "ses_ask", requestID: id, reply },
        })

      permission("per_other", crypto.randomUUID())
      permission("per_wrong_turn", activeRequestID, "atn_other")
      permission("per_selected_thread", activeRequestID, "atn_pending", "ask_other")
      await Bun.sleep(30)
      await mounted.app.renderOnce()
      expect(mounted.app.captureCharFrame()).not.toContain("Permission required")

      permission("per_once", activeRequestID)
      let frame = await waitForText(mounted.app, "Permission required")
      expect(frame).toContain("Read src/index.ts")
      expect(frame).toContain("Allow once")

      mounted.app.mockInput.pressEnter()
      await wait(() => replies.length === 1)
      expect(replies[0]?.id).toBe("per_once")
      expect(replies[0]?.reply).toBe("once")
      replied("per_once", "once")
      await wait(() => {
        void mounted.app.renderOnce()
        return !mounted.app.captureCharFrame().includes("Permission required")
      })

      permission("per_always", activeRequestID)
      await waitForText(mounted.app, "Permission required")
      mounted.app.mockInput.pressArrow("right")
      mounted.app.mockInput.pressEnter()
      await waitForText(mounted.app, "Always allow")
      mounted.app.mockInput.pressEnter()
      await wait(() => replies.length === 2)
      expect(replies[1]?.id).toBe("per_always")
      expect(replies[1]?.reply).toBe("always")
      replied("per_always", "always")
      await wait(() => !mounted.app.captureCharFrame().includes("Permission required"))

      permission("per_reject", activeRequestID)
      await waitForText(mounted.app, "Permission required")
      mounted.app.mockInput.pressArrow("right")
      mounted.app.mockInput.pressArrow("right")
      mounted.app.mockInput.pressEnter()
      await wait(() => replies.length === 3)
      expect(replies[2]?.id).toBe("per_reject")
      expect(replies[2]?.reply).toBe("reject")
      replied("per_reject", "reject")
      await wait(() => !mounted.app.captureCharFrame().includes("Permission required"))

      permission("per_escape", activeRequestID)
      await waitForText(mounted.app, "Permission required")
      mounted.app.mockInput.pressEscape()
      await wait(() => replies.length === 4)
      expect(replies[3]?.id).toBe("per_escape")
      expect(replies[3]?.reply).toBe("reject")
      replied("per_escape", "reject")
      await wait(() => !mounted.app.captureCharFrame().includes("Permission required"))
      frame = mounted.app.captureCharFrame()
      expect(frame).toContain("Answering")
    } finally {
      mounted.app.renderer.destroy()
    }
  })

  test("updates live tool activity, rejects unrelated events, and reconciles persisted calls", async () => {
    await using tmp = await tmpdir()
    let activeRequestID = ""
    let finishAsk!: (response: Response) => void
    const pendingAsk = new Promise<Response>((resolve) => {
      finishAsk = resolve
    })
    const mounted = await mountAsk(tmp.path, async (request) => {
      const url = new URL(request.url)
      if (url.pathname === "/session/ses_ask/ask" && request.method === "GET") {
        return json({ items: [], more: false })
      }
      if (url.pathname === "/session/ses_ask/ask" && request.method === "POST") {
        const body = (await request.clone().json()) as Record<string, unknown>
        activeRequestID = String(body.requestID)
        return pendingAsk
      }
      return undefined
    })

    try {
      await waitForText(mounted.app, "Answering")
      await wait(() => activeRequestID.length > 0)
      const otherRequestID = crypto.randomUUID()
      emit(mounted.events, {
        id: "evt_wrong_session",
        type: "session.ask.tool.activity",
        properties: {
          sessionID: "ses_other",
          requestID: activeRequestID,
          threadID: "ask_live",
          turnID: "atn_live",
          callID: "call_wrong_session",
          tool: "wrong-session",
          status: "running",
          input: "{}",
        },
      })
      emit(mounted.events, {
        id: "evt_other_request",
        type: "session.ask.tool.activity",
        properties: {
          sessionID: "ses_ask",
          requestID: otherRequestID,
          threadID: "ask_other",
          turnID: "atn_other",
          callID: "call_other_request",
          tool: "other-request",
          status: "running",
          input: "{}",
        },
      })
      await Bun.sleep(30)
      await mounted.app.renderOnce()
      let frame = mounted.app.captureCharFrame()
      expect(frame).not.toContain("wrong-session")
      expect(frame).not.toContain("other-request")
      emit(mounted.events, {
        id: "evt_read_running",
        type: "session.ask.tool.activity",
        properties: {
          sessionID: "ses_ask",
          requestID: activeRequestID,
          threadID: "ask_live",
          turnID: "atn_live",
          callID: "call_read",
          tool: "read",
          status: "running",
          input: '{"filePath":"src/live.ts"}',
        },
      })
      frame = await waitForText(mounted.app, "read running")
      expect(frame).not.toContain("wrong-session")

      emit(mounted.events, {
        id: "evt_wrong_thread",
        type: "session.ask.tool.activity",
        properties: {
          sessionID: "ses_ask",
          requestID: activeRequestID,
          threadID: "ask_other",
          turnID: "atn_live",
          callID: "call_wrong_thread",
          tool: "wrong-thread",
          status: "running",
          input: "{}",
        },
      })
      emit(mounted.events, {
        id: "evt_wrong_turn",
        type: "session.ask.tool.activity",
        properties: {
          sessionID: "ses_ask",
          requestID: activeRequestID,
          threadID: "ask_live",
          turnID: "atn_other",
          callID: "call_wrong_turn",
          tool: "wrong-turn",
          status: "running",
          input: "{}",
        },
      })
      emit(mounted.events, {
        id: "evt_read_completed",
        type: "session.ask.tool.activity",
        properties: {
          sessionID: "ses_ask",
          requestID: activeRequestID,
          threadID: "ask_live",
          turnID: "atn_live",
          callID: "call_read",
          tool: "read",
          status: "completed",
          input: '{"filePath":"src/live.ts"}',
          output: "live result",
        },
      })
      mounted.app.mockInput.pressKey("l", { ctrl: true })
      mounted.app.mockInput.pressEnter()
      frame = await waitForText(mounted.app, "live result")
      expect(frame).toContain("read completed")
      expect(frame).not.toContain("read running")
      expect(frame).not.toContain("wrong-thread")
      expect(frame).not.toContain("wrong-turn")

      emit(mounted.events, {
        id: "evt_bash_running",
        type: "session.ask.tool.activity",
        properties: {
          sessionID: "ses_ask",
          requestID: activeRequestID,
          threadID: "ask_live",
          turnID: "atn_live",
          callID: "call_bash",
          tool: "bash",
          status: "running",
          input: '{"command":"false"}',
        },
      })
      await waitForText(mounted.app, "bash running")
      emit(mounted.events, {
        id: "evt_bash_error",
        type: "session.ask.tool.activity",
        properties: {
          sessionID: "ses_ask",
          requestID: activeRequestID,
          threadID: "ask_live",
          turnID: "atn_live",
          callID: "call_bash",
          tool: "bash",
          status: "error",
          input: '{"command":"false"}',
          error: "live failure",
        },
      })
      frame = await waitForText(mounted.app, "bash error")
      const bashRow = frame.split("\n").findIndex((line) => line.includes("bash error"))
      const bashColumn = frame.split("\n")[bashRow]?.indexOf("bash error") ?? -1
      expect(bashRow).toBeGreaterThanOrEqual(0)
      expect(bashColumn).toBeGreaterThanOrEqual(0)
      await mounted.app.mockMouse.click(bashColumn, bashRow)
      expect(await waitForText(mounted.app, "live failure")).toContain('{"command":"false"}')
      mounted.app.mockInput.pressEnter()
      await wait(async () => {
        await mounted.app.renderOnce()
        return !mounted.app.captureCharFrame().includes("live failure")
      })
      expect(mounted.app.captureCharFrame()).toContain("live result")
      mounted.app.mockInput.pressKey("[")
      mounted.app.mockInput.pressEnter()

      finishAsk(
        json({
          id: "atn_live",
          threadID: "ask_live",
          requestID: activeRequestID,
          question: props.question,
          answer: "Persisted live answer",
          text: "Persisted live answer",
          toolActivity: [
            {
              callID: "call_read",
              tool: "read",
              status: "completed",
              input: '{"filePath":"src/live.ts"}',
              output: "live result",
            },
            {
              callID: "call_bash",
              tool: "bash",
              status: "error",
              input: '{"command":"false"}',
              error: "live failure",
            },
          ],
          time: { created: 500 },
        }),
      )
      frame = await waitForText(mounted.app, "Persisted live answer")
      expect(frame.split("read completed")).toHaveLength(2)
      expect(frame.split("bash error")).toHaveLength(2)

      emit(mounted.events, {
        id: "evt_after_completion",
        type: "session.ask.tool.activity",
        properties: {
          sessionID: "ses_ask",
          requestID: activeRequestID,
          threadID: "ask_live",
          turnID: "atn_after_completion",
          callID: "call_after_completion",
          tool: "after-completion",
          status: "running",
          input: "{}",
        },
      })
      await Bun.sleep(30)
      await mounted.app.renderOnce()
      expect(mounted.app.captureCharFrame()).not.toContain("after-completion")
    } finally {
      mounted.app.renderer.destroy()
    }
  })

  test("closes an initial Ask by aborting without a cancellation endpoint", async () => {
    await using tmp = await tmpdir()
    const requests: string[] = []
    let aborted = false
    let initialRequestID = ""
    const mounted = await mountAsk(tmp.path, async (request) => {
      const url = new URL(request.url)
      requests.push(`${request.method} ${url.pathname}`)
      if (url.pathname === "/session/ses_ask/ask" && request.method === "GET") {
        return json({ items: [], more: false })
      }
      if (url.pathname === "/session/ses_ask/ask" && request.method === "POST") {
        const body = (await request.clone().json()) as Record<string, unknown>
        initialRequestID = String(body.requestID)
        return new Promise<Response>((_resolve, reject) => {
          request.signal.addEventListener(
            "abort",
            () => {
              aborted = true
              reject(new DOMException("aborted", "AbortError"))
            },
            { once: true },
          )
        })
      }
      return undefined
    })

    try {
      await waitForText(mounted.app, "Answering")
      await wait(() => initialRequestID.length > 0)
      mounted.app.mockInput.pressEscape()
      await wait(() => aborted)
      expect(requests).not.toContain("POST /session/ses_ask/abort")
      expect(requests.some((request) => request.endsWith("/cancel"))).toBe(false)
      expect(initialRequestID.length).toBeLessThanOrEqual(128)
    } finally {
      mounted.app.renderer.destroy()
    }
  })
})

test("Prompt routes /ask directly to the Ask endpoint", async () => {
  await using tmp = await tmpdir()
  const requests: { method: string; path: string; body?: Record<string, unknown> }[] = []
  const model = {
    id: "model",
    providerID: "test",
    api: { id: "model", url: "http://test", npm: "test" },
    name: "Test model",
    capabilities: {
      temperature: true,
      reasoning: false,
      attachment: false,
      toolcall: true,
      input: { text: true, audio: false, image: false, video: false, pdf: false },
      output: { text: true, audio: false, image: false, video: false, pdf: false },
      interleaved: false,
    },
    cost: { input: 0, output: 0, cache: { read: 0, write: 0 } },
    limit: { context: 1000, output: 1000 },
    status: "active",
    options: {},
    headers: {},
    release_date: "2026-01-01",
  }
  const provider = {
    id: "test",
    name: "Test provider",
    source: "custom",
    env: [],
    options: {},
    models: { model },
  }
  const mounted = await mountPromptAsk(tmp.path, async (request) => {
    const url = new URL(request.url)
    const entry = { method: request.method, path: url.pathname } as {
      method: string
      path: string
      body?: Record<string, unknown>
    }
    if (request.method !== "GET") entry.body = (await request.clone().json()) as Record<string, unknown>
    requests.push(entry)
    if (url.pathname === "/agent") {
      return json([
        {
          name: "build",
          mode: "primary",
          hidden: false,
          model: { providerID: "test", modelID: "model" },
          permission: [],
          options: {},
        },
      ])
    }
    if (url.pathname === "/config/providers") {
      return json({ providers: [provider], default: { test: "model" }, connected: ["test"] })
    }
    if (url.pathname === "/provider") return json([provider])
    if (url.pathname === "/session/ses_prompt/ask" && request.method === "GET") {
      return json({ items: [], more: false })
    }
    if (url.pathname === "/session/ses_prompt/ask" && request.method === "POST") {
      return new Promise<Response>((_resolve, reject) => {
        request.signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")), {
          once: true,
        })
      })
    }
    return undefined
  })

  try {
    mounted.prompt.set({ input: "/ask Explain routing", parts: [] })
    mounted.prompt.submit()
    await wait(() => requests.some((request) => request.method === "POST" && request.path.endsWith("/ask")))
    const posts = requests.filter((request) => request.method === "POST")
    expect(posts.map((request) => request.path)).toEqual(["/session/ses_prompt/ask"])
    expect(posts[0]?.body?.question).toBe("Explain routing")
    expect(posts.some((request) => request.path === "/session")).toBe(false)
    expect(posts.some((request) => request.path.includes("/message"))).toBe(false)
    expect(posts.some((request) => request.path.includes("/command"))).toBe(false)
  } finally {
    mounted.app.renderer.destroy()
  }
})

test("renders expanded running and completed tool activity", async () => {
  await using tmp = await tmpdir()
  const state = path.join(tmp.path, "state")
  await mkdir(state, { recursive: true })
  await Bun.write(path.join(state, "kv.json"), "{}")
  const config = createTuiResolvedConfig()
  const app = await testRender(
    () => (
      <TestTuiContexts directory={tmp.path} paths={{ home: tmp.path, state }}>
        <TuiConfigProvider config={config}>
          <KVProvider>
            <ThemeProvider mode="dark">
              <box>
                <AskToolActivityView
                  activity={{ callID: "live", tool: "grep", status: "running", input: '{"pattern":"ask"}' }}
                  selected
                  expanded
                  onSelect={() => {}}
                />
                <AskToolActivityView
                  activity={{
                    callID: "done",
                    tool: "read",
                    status: "completed",
                    input: '{"filePath":"a.ts"}',
                    output: "result text",
                  }}
                  selected={false}
                  expanded
                  onSelect={() => {}}
                />
                <AskToolActivityView
                  activity={{
                    callID: "failed",
                    tool: "bash",
                    status: "error",
                    input: '{"command":"false"}',
                    error: "command failed",
                  }}
                  selected={false}
                  expanded
                  onSelect={() => {}}
                />
              </box>
            </ThemeProvider>
          </KVProvider>
        </TuiConfigProvider>
      </TestTuiContexts>
    ),
    { width: 80, height: 24 },
  )
  try {
    const frame = await waitForText(app, "grep running")
    expect(frame).toContain("grep running")
    expect(frame).toContain('{"pattern":"ask"}')
    expect(frame).toContain("read completed")
    expect(frame).toContain("result text")
    expect(frame).toContain("bash error")
    expect(frame).toContain("command failed")
  } finally {
    app.renderer.destroy()
  }
})

test("identifies Ask permissions and merges paginated records", () => {
  const request = {
    id: "per_ask",
    sessionID: "ses_ask",
    permission: "read",
    patterns: [],
    always: [],
    metadata: {
      ask: { requestID: "req_1", threadID: "ask_1", turnID: "atn_1", callID: "call_1", tool: "read" },
    },
  } satisfies PermissionRequest
  expect(isAskPermission(request)).toBe(true)
  expect(askPermissionMetadata(request)?.requestID).toBe("req_1")
  expect(askPermissionMetadata(request)?.threadID).toBe("ask_1")
  expect(isAskPermission({ ...request, metadata: {} })).toBe(false)
  const askPermission = { ...request, tool: { messageID: "msg_ask", callID: "call_ask" } }
  const normalPermission = {
    ...request,
    id: "per_normal",
    metadata: {},
    tool: { messageID: "msg_normal", callID: "call_normal" },
  }
  expect(normalPermissionCallID([askPermission, normalPermission])).toBe("call_normal")

  const running = {
    callID: "call_1",
    tool: "read",
    status: "running",
    input: "{}",
  } as const
  const completed = { ...running, status: "completed" as const, output: "done" }
  const live = mergeAskToolActivity(mergeAskToolActivity([], running), completed)
  expect(live).toEqual([completed])
  expect(reconcileAskToolActivity(live, [completed])).toEqual([])

  const older = {
    id: "ask_1",
    sessionID: "ses_ask",
    title: "Older",
    time: { created: 1, updated: 1 },
  }
  const newer = {
    id: "ask_2",
    sessionID: "ses_ask",
    title: "Newer",
    time: { created: 2, updated: 2 },
  }
  expect(mergeAskThreads([older], [newer]).map((thread) => thread.id)).toEqual(["ask_2", "ask_1"])
  expect(
    mergeAskTurns(
      [
        {
          id: "atn_2",
          threadID: "ask_1",
          question: "second",
          answer: "second",
          toolActivity: [],
          time: { created: 2 },
        },
      ],
      [
        {
          id: "atn_1",
          threadID: "ask_1",
          question: "first",
          answer: "first",
          toolActivity: [],
          time: { created: 1 },
        },
      ],
    ).map((turn) => turn.id),
  ).toEqual(["atn_1", "atn_2"])
})
