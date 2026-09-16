import { afterEach, describe, expect, test } from "bun:test"
import { EventV2 } from "@opencode-ai/core/event"
import { memoMap } from "@opencode-ai/core/effect/memo-map"
import { Location } from "@opencode-ai/core/location"
import { AbsolutePath } from "@opencode-ai/core/schema"
import { SessionAskEvent } from "@opencode-ai/schema/session-ask-event"
import { SessionID } from "@opencode-ai/schema/session-id"
import { Context, ManagedRuntime, Schema } from "effect"
import { AppNodeBuilderV1 } from "../../src/effect/app-node-builder-v1"
import { HttpApiApp } from "../../src/server/routes/instance/httpapi/server"
import { resetDatabase } from "../fixture/db"
import { disposeAllInstances, tmpdir } from "../fixture/fixture"

const context = Context.empty() as Context.Context<unknown>

function request(route: string, directory: string, init: RequestInit = {}) {
  const headers = new Headers(init.headers)
  headers.set("x-opencode-directory", directory)
  return HttpApiApp.webHandler().handler(
    new Request(`http://localhost${route}`, {
      ...init,
      headers,
    }),
    context,
  )
}

const Event = Schema.Struct({
  id: EventV2.ID,
  type: Schema.String,
  location: Schema.optional(Location.Ref),
  data: Schema.Unknown,
})

async function* eventStream(body: ReadableStream<Uint8Array>) {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  try {
    while (true) {
      const boundary = buffer.match(/(?:\r\n|\r|\n){2}/)
      if (!boundary || boundary.index === undefined) {
        const value = await reader.read()
        if (value.done) return
        buffer += decoder.decode(value.value, { stream: true })
        continue
      }

      const record = buffer.slice(0, boundary.index)
      buffer = buffer.slice(boundary.index + boundary[0].length)
      const data = record
        .split(/\r\n|\r|\n/)
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).replace(/^ /, ""))
      if (data.length) yield Schema.decodeUnknownSync(Event)(JSON.parse(data.join("\n")))
    }
  } finally {
    try {
      await reader.cancel()
    } finally {
      reader.releaseLock()
    }
  }
}

async function readEvent(reader: AsyncIterator<typeof Event.Type>) {
  const value = await reader.next()
  if (value.done) throw new Error("event stream closed")
  return value.value
}

async function readEventType(reader: AsyncIterator<typeof Event.Type>, type: string) {
  for (let index = 0; index < 20; index++) {
    const event = await readEvent(reader)
    if (event.type === type) return event
  }
  throw new Error(`timed out waiting for ${type}`)
}

afterEach(async () => {
  await disposeAllInstances()
  await resetDatabase()
})

describe("v2 location HttpApi", () => {
  test("decodes EventV2 location refs without resolved project metadata", () => {
    expect(
      Schema.decodeUnknownSync(Event)({
        id: "evt_test",
        type: "file.watcher.updated",
        location: { directory: "/tmp/project" },
        data: {},
      }),
    ).toMatchObject({ location: { directory: "/tmp/project" } })
  })

  test("returns command and skill snapshots with resolved locations", async () => {
    await using tmp = await tmpdir({ git: true })

    for (const route of ["/api/command", "/api/skill"]) {
      const response = await request(route, tmp.path)
      expect(response.status).toBe(200)
      const body = (await response.json()) as {
        location: { directory: string; project: { id: string } }
        data: unknown
      }
      expect(body.data).toBeArray()
      expect(body.location.directory).toBe(tmp.path)
      expect(body.location.project.id).toBeTruthy()
    }
  })

  test("streams native EventV2 payloads across locations", async () => {
    await using subscriber = await tmpdir({ git: true })
    await using publisher = await tmpdir({ git: true })
    const response = await request("/api/event", subscriber.path)
    const reader = eventStream(response.body!)
    const connected = await readEvent(reader)
    expect(connected.type).toBe("server.connected")
    expect(connected.location).toBeUndefined()

    const created = await request("/session", publisher.path, { method: "POST" })
    expect(created.status).toBe(200)
    expect(await readEventType(reader, "session.created")).toMatchObject({
      type: "session.created",
      location: { directory: publisher.path },
      data: { sessionID: expect.any(String) },
    })
    await reader.return(undefined)
  })

  test("skips legacy Ask activity and continues streaming canonical events", async () => {
    await using subscriber = await tmpdir({ git: true })
    await using publisher = await tmpdir({ git: true })
    const seeded = await request("/session", publisher.path, { method: "POST" })
    expect(seeded.status).toBe(200)
    const session = (await seeded.json()) as { id: string }
    const response = await request("/api/event", subscriber.path)
    const reader = eventStream(response.body!)
    expect((await readEvent(reader)).type).toBe("server.connected")

    const runtime = ManagedRuntime.make(AppNodeBuilderV1.build(EventV2.node), { memoMap })
    await runtime.runPromise(
      EventV2.Service.use((events) =>
        events.publish(
          SessionAskEvent.ToolActivity,
          {
            sessionID: SessionID.make(session.id),
            requestID: SessionAskEvent.RequestID.make("request-test"),
            threadID: "ask_test",
            turnID: "atn_test",
            callID: "call-test",
            tool: "read",
            status: "running",
            input: "{}",
          },
          { location: { directory: AbsolutePath.make(publisher.path) } },
        ),
      ),
    )
    const following = await request("/session", publisher.path, { method: "POST" })
    expect(following.status).toBe(200)

    const received: Array<typeof Event.Type> = []
    while (!received.some((event) => event.type === "session.created")) {
      received.push(await readEvent(reader))
    }
    expect(received.map((event) => event.type)).not.toContain("session.ask.tool.activity")
    expect(received.find((event) => event.type === "session.created")).toMatchObject({
      location: { directory: publisher.path },
    })
    await reader.return(undefined)
    await runtime.dispose()
  })
})
