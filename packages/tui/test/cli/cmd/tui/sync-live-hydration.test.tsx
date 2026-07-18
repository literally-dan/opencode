/** @jsxImportSource @opentui/solid */
import { expect, test } from "bun:test"
import type { GlobalEvent } from "@opencode-ai/sdk/v2"
import { tmpdir } from "../../../fixture/fixture"
import { json, mount, wait } from "./sync-fixture"

const sessionID = "ses_hydration_race"
const messageID = "msg_hydration_race"
const partID = "prt_hydration_race"
const session = {
  id: sessionID,
  slug: "race",
  projectID: "proj_test",
  title: "race",
  time: { created: 0, updated: 0 },
  version: "1.15.13",
  directory: "/tmp/opencode/packages/opencode",
}
const assistant = {
  id: messageID,
  sessionID,
  role: "assistant" as const,
  agent: "build",
  modelID: "model",
  providerID: "test",
  mode: "build",
  parentID: "msg_user",
  path: { cwd: session.directory, root: session.directory },
  cost: 0,
  tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
  time: { created: 1, completed: 2 },
}

function global(payload: GlobalEvent["payload"]): GlobalEvent {
  return { directory: "/tmp/other", project: "proj_test", payload }
}

test("live messages use creation time with an ID tie-break", async () => {
  await using tmp = await tmpdir()
  await Bun.write(`${tmp.path}/kv.json`, "{}")
  const { app, emit, sync } = await mount(undefined, tmp.path)
  const messages = [
    { ...assistant, id: "msg_a", time: { created: 30, completed: 31 } },
    { ...assistant, id: "msg_z", time: { created: 10, completed: 11 } },
    { ...assistant, id: "msg_m", time: { created: 20, completed: 21 } },
    { ...assistant, id: "msg_b", time: { created: 20, completed: 21 } },
  ]

  try {
    for (const info of messages) {
      emit(global({ id: `evt_${info.id}`, type: "message.updated", properties: { sessionID, info } }))
    }
    await wait(() => sync.data.message[sessionID]?.length === messages.length)

    expect(sync.data.message[sessionID].map((message) => message.id)).toEqual(["msg_z", "msg_b", "msg_m", "msg_a"])
  } finally {
    app.renderer.destroy()
  }
})

test("stale session hydration does not overwrite live message parts", async () => {
  await using tmp = await tmpdir()
  await Bun.write(`${tmp.path}/kv.json`, "{}")

  let resolveMessages!: (response: Response) => void
  const messages = new Promise<Response>((resolve) => {
    resolveMessages = resolve
  })
  let requested = false
  const { app, emit, sync } = await mount((url) => {
    if (url.pathname === `/session/${sessionID}`) return json(session)
    if (url.pathname === `/session/${sessionID}/message`) {
      requested = true
      return messages
    }
    if (url.pathname === `/session/${sessionID}/todo` || url.pathname === `/session/${sessionID}/diff`) return json([])
    return undefined
  }, tmp.path)

  try {
    const hydrate = sync.session.sync(sessionID)
    await wait(() => requested)
    emit(global({ id: "evt_message", type: "message.updated", properties: { sessionID, info: assistant } }))
    emit(
      global({
        id: "evt_part",
        type: "message.part.updated",
        properties: {
          sessionID,
          time: 2,
          part: { id: partID, sessionID, messageID, type: "text", text: "visible live content" },
        },
      }),
    )
    await wait(() => sync.data.part[messageID]?.[0]?.type === "text")

    resolveMessages(
      json([
        {
          info: assistant,
          parts: [{ id: partID, sessionID, messageID, type: "text", text: "" }],
        },
      ]),
    )
    await hydrate

    expect(sync.data.part[messageID][0]).toMatchObject({ text: "visible live content" })
  } finally {
    app.renderer.destroy()
  }
})

test("orphan live deltas do not suppress hydrated parts", async () => {
  await using tmp = await tmpdir()
  await Bun.write(`${tmp.path}/kv.json`, "{}")

  let resolveMessages!: (response: Response) => void
  const messages = new Promise<Response>((resolve) => {
    resolveMessages = resolve
  })
  let requested = false
  const { app, emit, sync } = await mount((url) => {
    if (url.pathname === `/session/${sessionID}`) return json(session)
    if (url.pathname === `/session/${sessionID}/message`) {
      requested = true
      return messages
    }
    if (url.pathname === `/session/${sessionID}/todo` || url.pathname === `/session/${sessionID}/diff`) return json([])
    return undefined
  }, tmp.path)

  try {
    const hydrate = sync.session.sync(sessionID)
    await wait(() => requested)
    emit(
      global({
        id: "evt_delta",
        type: "message.part.delta",
        properties: { sessionID, messageID, partID, field: "text", delta: "ignored until part exists" },
      }),
    )
    resolveMessages(
      json([{ info: assistant, parts: [{ id: partID, sessionID, messageID, type: "text", text: "hydrated" }] }]),
    )
    await hydrate

    expect(sync.data.part[messageID][0]).toMatchObject({ text: "hydrated" })
  } finally {
    app.renderer.destroy()
  }
})

test("hydration does not clear text streamed before it starts", async () => {
  await using tmp = await tmpdir()
  await Bun.write(`${tmp.path}/kv.json`, "{}")

  let resolveMessages!: (response: Response) => void
  const messages = new Promise<Response>((resolve) => {
    resolveMessages = resolve
  })
  let requested = false
  const { app, emit, sync } = await mount((url) => {
    if (url.pathname === `/session/${sessionID}`) return json(session)
    if (url.pathname === `/session/${sessionID}/message`) {
      requested = true
      return messages
    }
    if (url.pathname === `/session/${sessionID}/todo` || url.pathname === `/session/${sessionID}/diff`) return json([])
    return undefined
  }, tmp.path)

  try {
    emit(global({ id: "evt_message", type: "message.updated", properties: { sessionID, info: assistant } }))
    emit(
      global({
        id: "evt_part",
        type: "message.part.updated",
        properties: {
          sessionID,
          time: 1,
          part: { id: partID, sessionID, messageID, type: "text", text: "" },
        },
      }),
    )
    emit(
      global({
        id: "evt_delta",
        type: "message.part.delta",
        properties: { sessionID, messageID, partID, field: "text", delta: "visible streamed content" },
      }),
    )
    await wait(() => sync.data.part[messageID]?.[0]?.type === "text" && sync.data.part[messageID][0].text !== "")
    const hydrate = sync.session.sync(sessionID)
    await wait(() => requested)
    resolveMessages(json([{ info: assistant, parts: [{ id: partID, sessionID, messageID, type: "text", text: "" }] }]))
    await hydrate

    expect(sync.data.part[messageID][0]).toMatchObject({ text: "visible streamed content" })
  } finally {
    app.renderer.destroy()
  }
})

test("live messages merged during hydration retain the 100 message window", async () => {
  await using tmp = await tmpdir()
  await Bun.write(`${tmp.path}/kv.json`, "{}")

  let resolveMessages!: (response: Response) => void
  const messages = new Promise<Response>((resolve) => {
    resolveMessages = resolve
  })
  let requested = false
  const { app, emit, sync } = await mount((url) => {
    if (url.pathname === `/session/${sessionID}`) return json(session)
    if (url.pathname === `/session/${sessionID}/message`) {
      requested = true
      return messages
    }
    if (url.pathname === `/session/${sessionID}/todo` || url.pathname === `/session/${sessionID}/diff`) return json([])
    return undefined
  }, tmp.path)

  try {
    const hydrate = sync.session.sync(sessionID)
    await wait(() => requested)
    const live = { ...assistant, id: "msg_z_live" }
    emit(global({ id: "evt_live", type: "message.updated", properties: { sessionID, info: live } }))
    await wait(() => sync.data.message[sessionID]?.some((message) => message.id === live.id) ?? false)
    resolveMessages(
      json(
        Array.from({ length: 100 }, (_, index) => {
          const id = `msg_${String(index).padStart(3, "0")}`
          return {
            info: { ...assistant, id },
            parts: [{ id: `prt_${id}`, sessionID, messageID: id, type: "text", text: id }],
          }
        }),
      ),
    )
    await hydrate

    expect(sync.data.message[sessionID]).toHaveLength(100)
    expect(sync.data.message[sessionID].at(-1)?.id).toBe(live.id)
    expect(sync.data.message[sessionID].some((message) => message.id === "msg_000")).toBe(false)
    expect(sync.data.part.msg_000).toBeUndefined()
  } finally {
    app.renderer.destroy()
  }
})

test("a message removed during hydration does not regain stale parts", async () => {
  await using tmp = await tmpdir()
  await Bun.write(`${tmp.path}/kv.json`, "{}")

  let resolveMessages!: (response: Response) => void
  const messages = new Promise<Response>((resolve) => {
    resolveMessages = resolve
  })
  let requested = false
  const { app, emit, sync } = await mount((url) => {
    if (url.pathname === `/session/${sessionID}`) return json(session)
    if (url.pathname === `/session/${sessionID}/message`) {
      requested = true
      return messages
    }
    if (url.pathname === `/session/${sessionID}/todo` || url.pathname === `/session/${sessionID}/diff`) return json([])
    return undefined
  }, tmp.path)

  try {
    emit(global({ id: "evt_message", type: "message.updated", properties: { sessionID, info: assistant } }))
    await wait(() => sync.data.message[sessionID]?.length === 1)
    const hydrate = sync.session.sync(sessionID)
    await wait(() => requested)
    emit(global({ id: "evt_removed", type: "message.removed", properties: { sessionID, messageID } }))
    await wait(() => sync.data.message[sessionID]?.length === 0)
    resolveMessages(
      json([{ info: assistant, parts: [{ id: partID, sessionID, messageID, type: "text", text: "stale" }] }]),
    )
    await hydrate

    expect(sync.data.message[sessionID]).toEqual([])
    expect(sync.data.part[messageID]).toBeUndefined()
  } finally {
    app.renderer.destroy()
  }
})

test("hydration keeps only live-touched siblings in an orphan part bucket", async () => {
  await using tmp = await tmpdir()
  await Bun.write(`${tmp.path}/kv.json`, "{}")

  const mixedMessageID = "msg_mixed_orphan"
  const otherMessageID = "msg_other_session"
  let resolveMessages!: (response: Response) => void
  const messages = new Promise<Response>((resolve) => {
    resolveMessages = resolve
  })
  let requested = false
  const { app, emit, sync } = await mount((url) => {
    if (url.pathname === `/session/${sessionID}`) return json(session)
    if (url.pathname === `/session/${sessionID}/message`) {
      requested = true
      return messages
    }
    if (url.pathname === `/session/${sessionID}/todo` || url.pathname === `/session/${sessionID}/diff`) return json([])
    return undefined
  }, tmp.path)

  try {
    sync.set("part", mixedMessageID, [
      { id: "prt_stale_orphan", sessionID, messageID: mixedMessageID, type: "text", text: "stale" },
    ])
    sync.set("part", otherMessageID, [
      {
        id: "prt_other_session",
        sessionID: "ses_other",
        messageID: otherMessageID,
        type: "text",
        text: "other",
      },
    ])
    const hydrate = sync.session.sync(sessionID)
    await wait(() => requested)
    emit(
      global({
        id: "evt_live_orphan",
        type: "message.part.updated",
        properties: {
          sessionID,
          time: 2,
          part: {
            id: "prt_live_orphan",
            sessionID,
            messageID: mixedMessageID,
            type: "text",
            text: "live",
          },
        },
      }),
    )
    await wait(() => sync.data.part[mixedMessageID]?.length === 2)
    resolveMessages(json([{ info: assistant, parts: [] }]))
    await hydrate

    expect(sync.data.part[mixedMessageID]).toEqual([expect.objectContaining({ id: "prt_live_orphan", text: "live" })])
    expect(sync.data.part[otherMessageID]?.[0]).toMatchObject({ text: "other" })
  } finally {
    app.renderer.destroy()
  }
})

test("a deleted session is not resurrected by in-flight hydration", async () => {
  await using tmp = await tmpdir()
  await Bun.write(`${tmp.path}/kv.json`, "{}")

  let resolveMessages!: (response: Response) => void
  const messages = new Promise<Response>((resolve) => {
    resolveMessages = resolve
  })
  let requested = false
  const { app, emit, sync } = await mount((url) => {
    if (url.pathname === `/session/${sessionID}`) return json(session)
    if (url.pathname === `/session/${sessionID}/message`) {
      requested = true
      return messages
    }
    if (url.pathname === `/session/${sessionID}/todo` || url.pathname === `/session/${sessionID}/diff`) return json([])
    return undefined
  }, tmp.path)

  try {
    const hydrate = sync.session.sync(sessionID)
    await wait(() => requested)
    const orphanMessageID = "msg_deleted_orphan"
    sync.set("session", [session])
    sync.set("message", sessionID, [assistant])
    sync.set("part", messageID, [])
    sync.set("part", orphanMessageID, [
      { id: "prt_deleted_orphan", sessionID, messageID: orphanMessageID, type: "text", text: "orphan" },
    ])
    sync.set("todo", sessionID, [{ content: "cleanup", status: "pending", priority: "high" }])
    sync.set("session_diff", sessionID, [
      { file: "file.ts", status: "modified", additions: 1, deletions: 0, patch: "@@" },
    ])
    sync.set("session_status", sessionID, { type: "busy" })
    sync.set("permission", sessionID, [
      { id: "per_cleanup", sessionID, permission: "edit", patterns: ["*"], metadata: {}, always: [] },
    ])
    sync.set("question", sessionID, [{ id: "que_cleanup", sessionID, questions: [] }])
    emit(global({ id: "evt_deleted", type: "session.deleted", properties: { sessionID, info: session } }))
    resolveMessages(
      json([{ info: assistant, parts: [{ id: partID, sessionID, messageID, type: "text", text: "stale" }] }]),
    )
    await hydrate

    expect(sync.session.get(sessionID)).toBeUndefined()
    expect(sync.data.message[sessionID]).toBeUndefined()
    expect(sync.data.part[messageID]).toBeUndefined()
    expect(sync.data.part[orphanMessageID]).toBeUndefined()
    expect(sync.data.todo[sessionID]).toBeUndefined()
    expect(sync.data.session_diff[sessionID]).toBeUndefined()
    expect(sync.data.session_status[sessionID]).toBeUndefined()
    expect(sync.data.permission[sessionID]).toBeUndefined()
    expect(sync.data.question[sessionID]).toBeUndefined()
  } finally {
    app.renderer.destroy()
  }
})

test("a deleted session is not resurrected by an in-flight bootstrap", async () => {
  await using tmp = await tmpdir()
  await Bun.write(`${tmp.path}/kv.json`, "{}")

  let resolveSessions!: (response: Response) => void
  const sessions = new Promise<Response>((resolve) => {
    resolveSessions = resolve
  })
  let resolveStatus!: (response: Response) => void
  const status = new Promise<Response>((resolve) => {
    resolveStatus = resolve
  })
  let lists = 0
  let statuses = 0
  let requested = false
  let statusRequested = false
  const { app, emit, sync } = await mount((url) => {
    if (url.pathname === "/session") {
      lists++
      if (lists === 1) return json([])
      requested = true
      return sessions
    }
    if (url.pathname === "/session/status") {
      statuses++
      if (statuses === 1) return json({})
      statusRequested = true
      return status
    }
    return undefined
  }, tmp.path)

  try {
    sync.set("session", [session])
    sync.set("session_status", sessionID, { type: "busy" })
    sync.set("status", "loading")
    void sync.bootstrap({ fatal: false })
    await wait(() => requested && statusRequested && sync.status === "partial")
    emit(global({ id: "evt_deleted", type: "session.deleted", properties: { sessionID, info: session } }))
    resolveSessions(json([session]))
    resolveStatus(json({ [sessionID]: { type: "busy" } }))
    await wait(() => sync.status === "complete")

    expect(sync.session.get(sessionID)).toBeUndefined()
    expect(sync.data.session_status[sessionID]).toBeUndefined()
  } finally {
    app.renderer.destroy()
  }
})

test("hydration and part removal do not retain empty part buckets", async () => {
  await using tmp = await tmpdir()
  await Bun.write(`${tmp.path}/kv.json`, "{}")

  const { app, emit, sync } = await mount((url) => {
    if (url.pathname === `/session/${sessionID}`) return json(session)
    if (url.pathname === `/session/${sessionID}/message`) return json([{ info: assistant, parts: [] }])
    if (url.pathname === `/session/${sessionID}/todo` || url.pathname === `/session/${sessionID}/diff`) return json([])
    return undefined
  }, tmp.path)

  try {
    await sync.session.sync(sessionID)
    expect(sync.data.part[messageID]).toBeUndefined()

    emit(
      global({
        id: "evt_part",
        type: "message.part.updated",
        properties: {
          sessionID,
          time: 2,
          part: { id: partID, sessionID, messageID, type: "text", text: "temporary" },
        },
      }),
    )
    await wait(() => sync.data.part[messageID]?.length === 1)
    emit(
      global({
        id: "evt_part_removed",
        type: "message.part.removed",
        properties: { sessionID, messageID, partID },
      }),
    )
    await wait(() => sync.data.part[messageID] === undefined)

    expect(sync.data.part[messageID]).toBeUndefined()
  } finally {
    app.renderer.destroy()
  }
})

// A throw inside the sync handler aborts every remaining handler for that event
// and every remaining event in the same 16ms flush batch, because the SDK
// dispatches the whole queue inside one `batch(...)` with no try/catch. So an
// unguarded lookup here presents as a live session silently losing parts.
test("removal events for unknown sessions do not poison the event batch", async () => {
  await using tmp = await tmpdir()
  await Bun.write(`${tmp.path}/kv.json`, "{}")

  const { app, emit, sync } = await mount(undefined, tmp.path)
  try {
    // Neither session nor message has ever been hydrated, so both buckets are
    // absent. The TUI subscribes to every event for the project with no session
    // filter, so another client reverting or a subagent compacting gets here.
    emit(
      global({
        id: "evt_unknown_message_removed",
        type: "message.removed",
        properties: { sessionID: "ses_never_opened", messageID: "msg_never_opened" },
      }),
    )
    emit(
      global({
        id: "evt_unknown_part_removed",
        type: "message.part.removed",
        properties: { sessionID: "ses_never_opened", messageID: "msg_never_opened", partID: "prt_never_opened" },
      }),
    )
    // The batch survived, so a later event in the same queue still applies.
    emit(
      global({
        id: "evt_after_unknown",
        type: "session.updated",
        properties: { sessionID, info: { ...session, title: "applied after unknown removals" } },
      }),
    )

    await wait(() => sync.data.session.some((item) => item.title === "applied after unknown removals"))
    expect(sync.data.message["ses_never_opened"]).toBeUndefined()
    expect(sync.data.part["msg_never_opened"]).toBeUndefined()
  } finally {
    app.renderer.destroy()
  }
})

test("a duplicate session delete does not evict an unrelated guard entry", async () => {
  await using tmp = await tmpdir()
  await Bun.write(`${tmp.path}/kv.json`, "{}")

  const { app, emit, sync } = await mount(undefined, tmp.path)
  try {
    const first = "ses_guard_first"
    // Fill the guard to its cap, then replay one delete. Re-adding an id already
    // present must not evict the oldest entry, or a late event for that session
    // resurrects the caches the delete freed.
    for (let i = 0; i < 256; i++) {
      emit(
        global({
          id: `evt_delete_${i}`,
          type: "session.deleted",
          properties: {
            sessionID: i === 0 ? first : `ses_guard_${i}`,
            info: { ...session, id: i === 0 ? first : `ses_guard_${i}` },
          },
        }),
      )
    }
    emit(
      global({
        id: "evt_delete_dup",
        type: "session.deleted",
        properties: { sessionID: "ses_guard_255", info: { ...session, id: "ses_guard_255" } },
      }),
    )
    emit(
      global({
        id: "evt_late_status",
        type: "session.status",
        properties: { sessionID: first, status: { type: "busy" } },
      }),
    )
    emit(
      global({
        id: "evt_guard_settled",
        type: "session.updated",
        properties: { sessionID, info: { ...session, title: "guard settled" } },
      }),
    )

    await wait(() => sync.data.session.some((item) => item.title === "guard settled"))
    expect(sync.data.session_status[first]).toBeUndefined()
  } finally {
    app.renderer.destroy()
  }
})
