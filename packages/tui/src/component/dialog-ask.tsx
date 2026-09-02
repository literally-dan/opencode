import { ScrollBoxRenderable, TextareaRenderable, TextAttributes } from "@opentui/core"
import { useTerminalDimensions } from "@opentui/solid"
import type {
  EventSessionAskToolActivity,
  SessionAskResponse,
  SessionAskThreadResponse,
  SessionAskThreadsResponse,
} from "@opencode-ai/sdk/v2"
import { createEffect, createMemo, createSignal, For, onCleanup, onMount, Show } from "solid-js"
import { createStore } from "solid-js/store"
import { useBindings } from "../keymap"
import { useTuiConfig } from "../config"
import { useTheme } from "../context/theme"
import { useSDK } from "../context/sdk"
import { useSync } from "../context/sync"
import { useProject } from "../context/project"
import { useEvent } from "../context/event"
import { getScrollAcceleration } from "../util/scroll"
import { errorMessage } from "../util/error"
import { Locale } from "../util/locale"
import { Spinner } from "./spinner"
import { useDialog } from "../ui/dialog"
import { askPermissionMetadata, PermissionPrompt } from "../routes/session/permission"

type AskThread = SessionAskThreadsResponse["items"][number]
type AskTurn = SessionAskThreadResponse["items"][number]
type AskToolActivity = AskTurn["toolActivity"][number] | LiveToolActivity
type Pane = "threads" | "turns" | "composer"
type LoadState = {
  loaded: boolean
  loading: boolean
  more: boolean
  cursor?: string
  error?: string
}
type LiveToolActivity = Pick<
  EventSessionAskToolActivity["properties"],
  "callID" | "tool" | "status" | "input" | "output" | "error"
>
type LiveTurnActivity = {
  threadID: string
  activity: AskToolActivity[]
}
type AskRequestState = {
  status: "idle" | "running" | "cancelling" | "error"
  requestID?: string
  question?: string
  threadID?: string
  identityThreadID?: string
  turnID?: string
  message?: string
}

const PAGE_SIZE = 20

export function mergeAskThreads(current: AskThread[], incoming: AskThread[]) {
  const threads = new Map(current.map((thread) => [thread.id, thread]))
  for (const thread of incoming) threads.set(thread.id, thread)
  return [...threads.values()].toSorted((a, b) => {
    const updated = numericTime(b.time.updated) - numericTime(a.time.updated)
    return updated || b.id.localeCompare(a.id)
  })
}

export function mergeAskTurns(current: AskTurn[], incoming: AskTurn[]) {
  const turns = new Map([...current, ...incoming].map((turn) => [turn.id, turn]))
  return [...turns.values()].toSorted((a, b) => {
    const created = numericTime(a.time.created) - numericTime(b.time.created)
    return created || a.id.localeCompare(b.id)
  })
}

export function mergeAskToolActivity(current: AskToolActivity[], incoming: AskToolActivity) {
  const activity = new Map(current.map((item) => [item.callID, item]))
  activity.set(incoming.callID, incoming)
  return [...activity.values()]
}

export function reconcileAskToolActivity(current: AskToolActivity[], persisted: AskTurn["toolActivity"]) {
  const calls = new Set(persisted.map((activity) => activity.callID))
  return current.filter((activity) => !calls.has(activity.callID))
}

export type DialogAskProps = {
  sessionID: string
  question: string
  agent: string
  model: { providerID: string; modelID: string }
  variant?: string
}

export function DialogAsk(props: DialogAskProps) {
  const dialog = useDialog()
  const dimensions = useTerminalDimensions()
  const tuiConfig = useTuiConfig()
  const { theme, syntax } = useTheme()
  const sdk = useSDK()
  const event = useEvent()
  const sync = useSync()
  const project = useProject()
  const [composerTarget, setComposerTarget] = createSignal<TextareaRenderable>()
  const [store, setStore] = createStore({
    threads: [] as AskThread[],
    selectedID: undefined as string | undefined,
    threadIndex: 0,
    pane: "threads" as Pane,
    turns: {} as Record<string, AskTurn[]>,
    threadPage: { loaded: false, loading: false, more: false } as LoadState,
    turnPage: {} as Record<string, LoadState>,
    expanded: {} as Record<string, boolean>,
    live: {} as Record<string, LiveTurnActivity>,
    activityIndex: 0,
    request: { status: "idle" } as AskRequestState,
  })

  let disposed = false
  let controller: AbortController | undefined
  let threadScroll: ScrollBoxRenderable | undefined
  let turnScroll: ScrollBoxRenderable | undefined

  const location = () => ({
    directory: sync.session.get(props.sessionID)?.directory,
    workspace: project.workspace.current(),
  })
  const selectedThread = createMemo(() => store.threads.find((thread) => thread.id === store.selectedID))
  const selectedTurns = createMemo(() => (store.selectedID ? (store.turns[store.selectedID] ?? []) : []))
  const permissions = createMemo(() => {
    if (store.request.status !== "running") return []
    const requestID = store.request.requestID
    if (!requestID) return []
    return (sync.data.permission[props.sessionID] ?? []).filter((request) => {
      const ask = askPermissionMetadata(request)
      if (!ask || request.sessionID !== props.sessionID || ask.requestID !== requestID) return false
      if (store.request.identityThreadID && ask.threadID !== store.request.identityThreadID) return false
      if (store.request.turnID && ask.turnID !== store.request.turnID) return false
      return true
    })
  })
  const permission = createMemo(() => permissions()[0])
  createEffect(() => {
    const ask = askPermissionMetadata(permission())
    if (!ask) return
    if (!store.request.identityThreadID) setStore("request", "identityThreadID", ask.threadID)
    if (!store.request.turnID) setStore("request", "turnID", ask.turnID)
  })
  const requestVisible = createMemo(
    () =>
      store.request.status !== "idle" &&
      (store.request.threadID === store.selectedID || (!store.request.threadID && !store.selectedID)),
  )
  const requestActivity = createMemo(() =>
    store.request.turnID ? (store.live[store.request.turnID]?.activity ?? []) : [],
  )
  const activities = createMemo(() => {
    const persisted = selectedTurns().flatMap((turn) =>
      turn.toolActivity.map((activity) => ({ key: `${turn.id}:${activity.callID}`, activity })),
    )
    if (!requestVisible() || !store.request.turnID) return persisted
    return [
      ...persisted,
      ...requestActivity().map((activity) => ({ key: `${store.request.turnID}:${activity.callID}`, activity })),
    ]
  })

  const offActivity = event.on("session.ask.tool.activity", (evt) => {
    const activity = evt.properties
    if (activity.sessionID !== props.sessionID || activity.requestID !== store.request.requestID) return
    if (store.request.status !== "running") return
    if (store.request.identityThreadID && activity.threadID !== store.request.identityThreadID) return
    if (store.request.turnID && activity.turnID !== store.request.turnID) return
    const persisted = store.turns[activity.threadID]
      ?.find((turn) => turn.id === activity.turnID)
      ?.toolActivity.some((item) => item.callID === activity.callID)
    if (persisted) return
    const current = store.live[activity.turnID]
    setStore("live", activity.turnID, {
      threadID: activity.threadID,
      activity: mergeAskToolActivity(current?.activity ?? [], activity),
    })
    setStore("request", {
      identityThreadID: activity.threadID,
      turnID: activity.turnID,
    })
  })

  function setPane(pane: Pane) {
    setStore("pane", pane)
    if (pane === "composer") {
      composerTarget()?.focus()
      return
    }
    composerTarget()?.blur()
  }

  function scrollThreadSelection() {
    if (!threadScroll) return
    const child = threadScroll.getChildren()[store.threadIndex]
    if (!child) return
    const y = child.y - threadScroll.y
    if (y < 0) threadScroll.scrollBy(y)
    if (y >= threadScroll.height) threadScroll.scrollBy(y - threadScroll.height + 1)
  }

  async function loadThreads(reset = false) {
    if (store.threadPage.loading) return
    if (!reset && store.threadPage.loaded && !store.threadPage.more) return
    setStore("threadPage", {
      ...store.threadPage,
      loading: true,
      error: undefined,
      ...(reset ? { cursor: undefined } : {}),
    })
    try {
      const result = await sdk.client.session.askThreads({
        sessionID: props.sessionID,
        limit: PAGE_SIZE,
        before: reset ? undefined : store.threadPage.cursor,
        ...location(),
      })
      if (result.error) throw result.error
      if (!result.data) throw new Error("Ask threads returned no data")
      if (disposed) return
      const threads = mergeAskThreads(store.threads, result.data.items)
      const selectedIndex = store.selectedID ? threads.findIndex((thread) => thread.id === store.selectedID) : -1
      setStore("threads", threads)
      setStore(
        "threadIndex",
        selectedIndex >= 0 ? selectedIndex : Math.min(store.threadIndex, Math.max(threads.length - 1, 0)),
      )
      setStore("threadPage", {
        loaded: true,
        loading: false,
        more: result.data.more,
        cursor: result.data.cursor,
      })
      if (!store.selectedID && store.request.status === "idle" && store.threads[0]) {
        await selectThread(store.threads[0].id)
      }
    } catch (error) {
      if (disposed) return
      setStore("threadPage", {
        ...store.threadPage,
        loaded: true,
        loading: false,
        error: errorMessage(error),
      })
    }
  }

  async function loadTurns(threadID: string, reset = false) {
    const page = store.turnPage[threadID] ?? { loaded: false, loading: false, more: false }
    if (page.loading) return
    if (!reset && page.loaded && !page.more) return
    const initial = (store.turns[threadID]?.length ?? 0) === 0
    const scrollHeight = turnScroll?.scrollHeight
    const scrollTop = turnScroll?.scrollTop
    setStore("turnPage", threadID, {
      ...page,
      loading: true,
      error: undefined,
      ...(reset ? { cursor: undefined } : {}),
    })
    try {
      const result = await sdk.client.session.askThread({
        sessionID: props.sessionID,
        threadID,
        limit: PAGE_SIZE,
        before: reset ? undefined : page.cursor,
        ...location(),
      })
      if (result.error) throw result.error
      if (!result.data) throw new Error("Ask thread returned no data")
      if (disposed) return
      for (const turn of result.data.items) {
        const live = store.live[turn.id]
        if (!live) continue
        setStore("live", turn.id, {
          threadID: live.threadID,
          activity: reconcileAskToolActivity(live.activity, turn.toolActivity),
        })
      }
      setStore("turns", threadID, mergeAskTurns(reset ? [] : (store.turns[threadID] ?? []), result.data.items))
      setStore("turnPage", threadID, {
        loaded: true,
        loading: false,
        more: result.data.more,
        cursor: result.data.cursor,
      })
      setTimeout(() => {
        if (disposed || store.selectedID !== threadID) return
        if (initial) {
          turnScroll?.scrollTo(turnScroll.scrollHeight)
          return
        }
        if (!reset && turnScroll && scrollHeight !== undefined && scrollTop !== undefined) {
          turnScroll.scrollTo(scrollTop + turnScroll.scrollHeight - scrollHeight)
        }
      }, 0)
    } catch (error) {
      if (disposed) return
      setStore("turnPage", threadID, {
        ...page,
        loaded: true,
        loading: false,
        error: errorMessage(error),
      })
    }
  }

  async function selectThread(threadID: string) {
    setStore("selectedID", threadID)
    setStore(
      "threadIndex",
      Math.max(
        0,
        store.threads.findIndex((thread) => thread.id === threadID),
      ),
    )
    setStore("activityIndex", 0)
    scrollThreadSelection()
    if (!store.turnPage[threadID]?.loaded) await loadTurns(threadID, true)
  }

  async function moveThread(direction: number) {
    const next = store.threadIndex + direction
    if (next < 0) return
    if (next >= store.threads.length) {
      if (direction > 0 && store.threadPage.more) {
        const size = store.threads.length
        await loadThreads()
        if (store.threads[size]) await selectThread(store.threads[size].id)
      }
      return
    }
    const thread = store.threads[next]
    if (thread) await selectThread(thread.id)
  }

  function moveActivity(direction: number) {
    if (activities().length === 0) return
    setStore("activityIndex", (store.activityIndex + direction + activities().length) % activities().length)
  }

  function toggleActivity(key = activities()[store.activityIndex]?.key) {
    if (!key) return
    setStore("expanded", key, !store.expanded[key])
  }

  function selectActivity(key: string) {
    setPane("turns")
    const index = activities().findIndex((item) => item.key === key)
    if (index >= 0) setStore("activityIndex", index)
    toggleActivity(key)
  }

  async function submit(question: string, threadID?: string) {
    if (store.request.status === "running" || store.request.status === "cancelling") return
    const requestController = new AbortController()
    const requestID = crypto.randomUUID()
    controller = requestController
    setStore("request", {
      status: "running",
      requestID,
      question,
      threadID,
      identityThreadID: threadID,
      turnID: undefined,
      message: undefined,
    })
    setPane(threadID ? "turns" : "threads")
    try {
      const result = await sdk.client.session.ask(
        {
          sessionID: props.sessionID,
          requestID,
          threadID,
          question,
          agent: props.agent,
          model: props.model,
          variant: props.variant,
          ...location(),
        },
        { signal: requestController.signal },
      )
      if (disposed) return
      if (requestController.signal.aborted) return completeCancellation(question, threadID)
      if (result.error) throw result.error
      if (!result.data) throw new Error("Ask returned no data")
      if (result.data.requestID !== requestID) throw new Error("Ask returned a mismatched request ID")
      if (store.request.identityThreadID && result.data.threadID !== store.request.identityThreadID) {
        throw new Error("Ask returned a mismatched thread ID")
      }
      if (store.request.turnID && result.data.id !== store.request.turnID) {
        throw new Error("Ask returned a mismatched turn ID")
      }
      completeRequest(result.data)
    } catch (error) {
      if (disposed) return
      if (requestController.signal.aborted) return completeCancellation(question, threadID)
      const turnID = store.request.turnID
      setStore("request", {
        status: "error",
        requestID: undefined,
        question,
        threadID,
        identityThreadID: undefined,
        turnID: undefined,
        message: errorMessage(error),
      })
      if (turnID && store.live[turnID]) {
        setStore("live", turnID, { threadID: store.live[turnID].threadID, activity: [] })
      }
    }
  }

  function completeCancellation(question: string, threadID?: string) {
    if (store.request.status !== "cancelling") return
    setStore("request", {
      status: "error",
      requestID: undefined,
      question,
      threadID,
      identityThreadID: undefined,
      turnID: undefined,
      message: "Ask cancelled",
    })
  }

  function completeRequest(result: SessionAskResponse) {
    const turn: AskTurn = {
      id: result.id,
      threadID: result.threadID,
      question: result.question,
      answer: result.answer,
      toolActivity: result.toolActivity,
      time: result.time,
    }
    const live = store.live[result.id]
    if (live) {
      setStore("live", result.id, {
        threadID: live.threadID,
        activity: reconcileAskToolActivity(live.activity, result.toolActivity),
      })
    }
    const previous = store.threads.find((thread) => thread.id === result.threadID)
    const thread: AskThread = previous
      ? { ...previous, time: { ...previous.time, updated: result.time.created } }
      : {
          id: result.threadID,
          sessionID: props.sessionID,
          title: Locale.truncate(result.question, 80),
          time: { created: result.time.created, updated: result.time.created },
        }
    setStore("threads", mergeAskThreads(store.threads, [thread]))
    setStore("turns", result.threadID, mergeAskTurns(store.turns[result.threadID] ?? [], [turn]))
    setStore("turnPage", result.threadID, {
      ...(store.turnPage[result.threadID] ?? {}),
      loaded: true,
      loading: false,
      more: store.turnPage[result.threadID]?.more ?? false,
    })
    setStore("selectedID", result.threadID)
    setStore(
      "threadIndex",
      Math.max(
        0,
        store.threads.findIndex((item) => item.id === result.threadID),
      ),
    )
    setStore("request", {
      status: "idle",
      requestID: undefined,
      question: undefined,
      threadID: undefined,
      identityThreadID: undefined,
      turnID: undefined,
      message: undefined,
    })
    setPane("composer")
    setTimeout(() => {
      if (disposed) return
      turnScroll?.scrollTo(turnScroll.scrollHeight)
      composerTarget()?.focus()
    }, 0)
    void loadThreads(true)
  }

  function submitComposer() {
    const target = composerTarget()
    const question = target?.plainText.trim() ?? ""
    if (!question || !store.selectedID) return
    target?.setText("")
    void submit(question, store.selectedID)
  }

  function retry() {
    if (store.request.status !== "error" || !store.request.question) return
    void submit(store.request.question, store.request.threadID)
  }

  function retryCurrent() {
    if (store.request.status === "error") return retry()
    if (store.threadPage.error) return void loadThreads(true)
    if (store.selectedID && store.turnPage[store.selectedID]?.error) {
      void loadTurns(store.selectedID, true)
    }
  }

  function cancel(update = true) {
    if (store.request.status !== "running") return
    const turnID = store.request.turnID
    if (update) {
      setStore("request", "status", "cancelling")
      if (turnID && store.live[turnID]) {
        setStore("live", turnID, { threadID: store.live[turnID].threadID, activity: [] })
      }
    }
    controller?.abort()
  }

  useBindings(() => ({
    mode: "modal",
    enabled: !permission(),
    bindings: [
      { key: "ctrl+h", desc: "Focus Ask threads", group: "Ask", cmd: () => setPane("threads") },
      { key: "ctrl+l", desc: "Focus Ask conversation", group: "Ask", cmd: () => setPane("turns") },
    ],
  }))

  useBindings(() => ({
    mode: "modal",
    enabled: !permission() && store.pane !== "composer",
    bindings: [
      {
        key: "tab",
        desc: "Switch Ask pane",
        group: "Ask",
        cmd: () => {
          const panes: Pane[] = store.selectedID ? ["threads", "turns", "composer"] : ["threads", "turns"]
          setPane(panes[(panes.indexOf(store.pane) + 1) % panes.length])
        },
      },
      { key: "left", desc: "Focus Ask threads", group: "Ask", cmd: () => setPane("threads") },
      { key: "right", desc: "Focus Ask conversation", group: "Ask", cmd: () => setPane("turns") },
      {
        key: "up",
        desc: "Move or scroll up",
        group: "Ask",
        cmd: () => (store.pane === "threads" ? void moveThread(-1) : turnScroll?.scrollBy(-1)),
      },
      {
        key: "down",
        desc: "Move or scroll down",
        group: "Ask",
        cmd: () => (store.pane === "threads" ? void moveThread(1) : turnScroll?.scrollBy(1)),
      },
      {
        key: "pageup",
        desc: "Scroll Ask conversation up",
        group: "Ask",
        cmd: () => turnScroll?.scrollBy(-(turnScroll?.height ?? 1)),
      },
      {
        key: "pagedown",
        desc: "Scroll Ask conversation down",
        group: "Ask",
        cmd: () => turnScroll?.scrollBy(turnScroll?.height ?? 1),
      },
      { key: "[", desc: "Previous Ask tool", group: "Ask", cmd: () => moveActivity(-1) },
      { key: "]", desc: "Next Ask tool", group: "Ask", cmd: () => moveActivity(1) },
      {
        key: "return",
        desc: "Open Ask selection",
        group: "Ask",
        cmd: () => {
          if (store.pane === "turns") toggleActivity()
          if (store.pane === "threads" && store.threads[store.threadIndex]) {
            void selectThread(store.threads[store.threadIndex].id)
          }
        },
      },
      { key: "e", desc: "Expand Ask tool", group: "Ask", cmd: () => toggleActivity() },
      {
        key: "o",
        desc: "Load older Ask items",
        group: "Ask",
        cmd: () => {
          if (store.pane === "threads") return void loadThreads()
          if (store.selectedID) void loadTurns(store.selectedID)
        },
      },
      { key: "c", desc: "Cancel active Ask", group: "Ask", cmd: cancel },
      { key: "r", desc: "Retry failed Ask", group: "Ask", cmd: retryCurrent },
    ],
  }))

  useBindings(() => ({
    mode: "modal",
    target: composerTarget,
    enabled:
      composerTarget() !== undefined &&
      store.request.status !== "running" &&
      store.request.status !== "cancelling" &&
      !permission(),
    priority: 1,
    bindings: [
      { key: "return", desc: "Send Ask follow-up", group: "Ask", cmd: submitComposer },
      { key: "tab", desc: "Focus Ask threads", group: "Ask", cmd: () => setPane("threads") },
    ],
  }))

  onMount(() => {
    dialog.setSize("xlarge")
    void submit(props.question)
    void loadThreads(true)
  })

  onCleanup(() => {
    offActivity()
    disposed = true
    cancel(false)
  })

  return (
    <box
      paddingLeft={1}
      paddingRight={1}
      paddingBottom={1}
      height={Math.max(12, dimensions().height - Math.floor(dimensions().height / 4) - 2)}
    >
      <box flexDirection="row" justifyContent="space-between" flexShrink={0} paddingLeft={1} paddingRight={1}>
        <box flexDirection="row" gap={1}>
          <text attributes={TextAttributes.BOLD} fg={theme.text}>
            Ask
          </text>
          <Show when={store.request.status === "running" || store.request.status === "cancelling"}>
            <Spinner color={theme.textMuted}>
              {store.request.status === "cancelling" ? "Cancelling" : "Thinking"}
            </Spinner>
          </Show>
        </box>
        <text fg={theme.textMuted} onMouseUp={() => dialog.clear()}>
          esc
        </text>
      </box>
      <box flexDirection="row" flexGrow={1} minHeight={7} paddingTop={1}>
        <box
          width={28}
          minWidth={20}
          flexShrink={0}
          paddingRight={1}
          border={["right"]}
          borderColor={store.pane === "threads" ? theme.borderActive : theme.border}
        >
          <text fg={store.pane === "threads" ? theme.text : theme.textMuted}>Threads</text>
          <scrollbox
            ref={(value: ScrollBoxRenderable) => (threadScroll = value)}
            flexGrow={1}
            scrollAcceleration={getScrollAcceleration(tuiConfig)}
          >
            <For each={store.threads}>
              {(thread, index) => {
                const selected = () => thread.id === store.selectedID
                const cursor = () => index() === store.threadIndex && store.pane === "threads"
                return (
                  <box
                    paddingLeft={1}
                    paddingRight={1}
                    backgroundColor={selected() ? theme.backgroundElement : undefined}
                    onMouseOver={() => setStore("threadIndex", index())}
                    onMouseUp={() => {
                      setPane("threads")
                      void selectThread(thread.id)
                    }}
                  >
                    <text fg={cursor() ? theme.accent : selected() ? theme.text : theme.textMuted}>
                      {Locale.truncate(thread.title, 24)}
                    </text>
                    <text fg={theme.textMuted}>{displayTime(thread.time.updated)}</text>
                  </box>
                )
              }}
            </For>
            <Show when={store.threadPage.loading}>
              <box paddingLeft={1} paddingTop={1}>
                <Spinner color={theme.textMuted}>Loading threads</Spinner>
              </box>
            </Show>
            <Show when={store.threadPage.error}>
              <box paddingLeft={1} paddingTop={1} onMouseUp={() => void loadThreads(true)}>
                <text fg={theme.error}>{store.threadPage.error}</text>
                <text fg={theme.textMuted}>r or click to retry</text>
              </box>
            </Show>
            <Show when={!store.threadPage.loading && store.threadPage.more}>
              <box paddingLeft={1} paddingTop={1} onMouseUp={() => void loadThreads()}>
                <text fg={theme.accent}>Load older threads</text>
              </box>
            </Show>
            <Show when={store.threadPage.loaded && store.threads.length === 0 && store.request.status === "idle"}>
              <text fg={theme.textMuted}>No Ask threads</text>
            </Show>
          </scrollbox>
        </box>
        <box paddingLeft={2} flexGrow={1} minWidth={24}>
          <box flexDirection="row" justifyContent="space-between" flexShrink={0} paddingRight={1}>
            <text fg={store.pane === "turns" ? theme.text : theme.textMuted}>
              {selectedThread()?.title ?? "New Ask"}
            </text>
            <Show when={store.request.status === "running"}>
              <text fg={theme.accent}>active</text>
            </Show>
          </box>
          <scrollbox
            ref={(value: ScrollBoxRenderable) => (turnScroll = value)}
            flexGrow={1}
            stickyScroll
            stickyStart="bottom"
            scrollAcceleration={getScrollAcceleration(tuiConfig)}
          >
            <Show when={store.selectedID && store.turnPage[store.selectedID]?.more}>
              <box paddingBottom={1} onMouseUp={() => store.selectedID && void loadTurns(store.selectedID)}>
                <text fg={theme.accent}>Load older turns</text>
              </box>
            </Show>
            <Show when={store.selectedID && store.turnPage[store.selectedID]?.loading}>
              <Spinner color={theme.textMuted}>Loading conversation</Spinner>
            </Show>
            <Show when={store.selectedID && store.turnPage[store.selectedID]?.error}>
              <box paddingBottom={1} onMouseUp={() => store.selectedID && void loadTurns(store.selectedID, true)}>
                <text fg={theme.error}>{store.selectedID ? store.turnPage[store.selectedID]?.error : ""}</text>
                <text fg={theme.textMuted}>r or click to retry</text>
              </box>
            </Show>
            <For each={selectedTurns()}>
              {(turn) => (
                <AskTurnView
                  turn={turn}
                  syntax={syntax()}
                  selectedActivity={activities()[store.activityIndex]?.key}
                  expanded={(key) => store.expanded[key] ?? false}
                  onActivity={selectActivity}
                />
              )}
            </For>
            <Show when={requestVisible()}>
              <box gap={1} paddingBottom={1}>
                <text fg={theme.accent}>You</text>
                <text fg={theme.text}>{store.request.question}</text>
                <For each={requestActivity()}>
                  {(activity) => {
                    const key = () => `${store.request.turnID}:${activity.callID}`
                    return (
                      <AskToolActivityView
                        activity={activity}
                        selected={activities()[store.activityIndex]?.key === key()}
                        expanded={store.expanded[key()] ?? false}
                        onSelect={() => selectActivity(key())}
                      />
                    )
                  }}
                </For>
                <Show when={store.request.status === "running" || store.request.status === "cancelling"}>
                  <Spinner color={theme.textMuted}>
                    {store.request.status === "cancelling" ? "Cancelling Ask" : "Answering"}
                  </Spinner>
                </Show>
                <Show when={store.request.status === "error"}>
                  <box gap={1}>
                    <text fg={theme.error}>{store.request.message}</text>
                    <box flexDirection="row" gap={2}>
                      <text fg={theme.accent} onMouseUp={retry}>
                        retry
                      </text>
                      <text fg={theme.textMuted}>press r</text>
                    </box>
                  </box>
                </Show>
              </box>
            </Show>
          </scrollbox>
          <Show
            when={permission()}
            fallback={
              <box flexShrink={0} paddingTop={1} gap={1}>
                <textarea
                  ref={(value: TextareaRenderable) => setComposerTarget(value)}
                  height={3}
                  placeholder={store.selectedID ? "Follow up in this Ask thread" : "Select a thread to follow up"}
                  placeholderColor={theme.textMuted}
                  textColor={theme.text}
                  focusedTextColor={theme.text}
                  cursorColor={theme.primary}
                  cursorStyle={tuiConfig.cursor}
                  onMouseDown={() => setPane("composer")}
                />
                <text fg={theme.textMuted}>
                  {store.selectedID ? "enter send follow-up" : "Choose a thread on the left"}
                </text>
              </box>
            }
          >
            {(request) => (
              <box flexShrink={0} minHeight={9} maxHeight={15}>
                <PermissionPrompt request={request()} directory={sync.session.get(props.sessionID)?.directory} modal />
              </box>
            )}
          </Show>
        </box>
      </box>
      <box flexShrink={0} paddingLeft={1} paddingTop={1}>
        <text fg={theme.textMuted}>
          ctrl+h/l pane arrows select/scroll [ ] tool enter expand o older c cancel r retry
        </text>
      </box>
    </box>
  )
}

function AskTurnView(props: {
  turn: AskTurn
  syntax: ReturnType<ReturnType<typeof useTheme>["syntax"]>
  selectedActivity?: string
  expanded: (key: string) => boolean
  onActivity: (key: string) => void
}) {
  const { theme } = useTheme()
  return (
    <box gap={1} paddingBottom={2}>
      <text fg={theme.accent}>You</text>
      <text fg={theme.text}>{props.turn.question}</text>
      <For each={props.turn.toolActivity}>
        {(activity) => {
          const key = () => `${props.turn.id}:${activity.callID}`
          return (
            <AskToolActivityView
              activity={activity}
              selected={props.selectedActivity === key()}
              expanded={props.expanded(key())}
              onSelect={() => props.onActivity(key())}
            />
          )
        }}
      </For>
      <text fg={theme.success}>Assistant</text>
      <markdown
        syntaxStyle={props.syntax}
        content={props.turn.answer}
        tableOptions={{ style: "grid" }}
        fg={theme.markdownText}
        bg={theme.backgroundPanel}
      />
    </box>
  )
}

export function AskToolActivityView(props: {
  activity: AskToolActivity
  selected: boolean
  expanded: boolean
  onSelect: () => void
}) {
  const { theme } = useTheme()
  const color = () => {
    if (props.activity.status === "running") return theme.accent
    if (props.activity.status === "error") return theme.error
    return theme.success
  }
  return (
    <box
      paddingLeft={1}
      paddingRight={1}
      backgroundColor={props.selected ? theme.backgroundElement : undefined}
      onMouseUp={props.onSelect}
    >
      <box flexDirection="row" gap={1}>
        <text fg={color()}>
          {props.activity.status === "running" ? "*" : props.activity.status === "error" ? "x" : "+"}
        </text>
        <text fg={theme.text}>{props.activity.tool}</text>
        <text fg={theme.textMuted}>{props.activity.status}</text>
        <text fg={theme.textMuted}>{props.expanded ? "[-]" : "[+]"}</text>
      </box>
      <Show when={props.expanded}>
        <box paddingLeft={2} paddingTop={1} gap={1}>
          <text fg={theme.textMuted}>Input</text>
          <text fg={theme.text}>{props.activity.input || "{}"}</text>
          <Show when={props.activity.output !== undefined}>
            <text fg={theme.textMuted}>Result</text>
            <text fg={theme.text}>{props.activity.output}</text>
          </Show>
          <Show when={props.activity.error !== undefined}>
            <text fg={theme.error}>{props.activity.error}</text>
          </Show>
        </box>
      </Show>
    </box>
  )
}

function numericTime(value: AskThread["time"]["updated"] | AskTurn["time"]["created"]) {
  return typeof value === "number" ? value : 0
}

function displayTime(value: AskThread["time"]["updated"]) {
  return typeof value === "number" ? Locale.todayTimeOrDateTime(value) : ""
}
