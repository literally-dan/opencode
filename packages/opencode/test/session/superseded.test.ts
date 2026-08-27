import { afterEach, describe, expect } from "bun:test"
import { Deferred, Effect, Fiber } from "effect"
import { CrossSpawnSpawner } from "@opencode-ai/core/cross-spawn-spawner"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { AppNodeBuilder } from "@opencode-ai/core/effect/app-node-builder"
import { SessionProjector } from "@opencode-ai/core/session/projector"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { ModelV2 } from "@opencode-ai/core/model"
import { Agent } from "@/agent/agent"
import { Config } from "@/config/config"
import { Session } from "@/session/session"
import { MessageV2 } from "@/session/message-v2"
import { Truncate } from "@/tool/truncate"
import { ToolRegistry } from "@/tool/registry"
import { Superseded } from "@/session/superseded"
import { compactionLockReferences, compactionOf, withCompactionLock } from "@/session/compaction-pruning"
import { MessageID, PartID, SessionID } from "@/session/schema"
import { disposeAllInstances } from "../fixture/fixture"
import { ProviderTest } from "../fake/provider"
import { pollWithTimeout, testEffect } from "../lib/effect"

afterEach(async () => {
  await disposeAllInstances()
})

const ref = {
  providerID: ProviderV2.ID.make("test"),
  modelID: ModelV2.ID.make("test-model"),
}

const it = testEffect(
  AppNodeBuilder.build(
    LayerNode.group([
      Agent.node,
      Config.node,
      CrossSpawnSpawner.node,
      Session.node,
      Truncate.node,
      ToolRegistry.node,
      MessageV2.node,
      SessionProjector.node,
    ]),
  ),
)

// Long enough to clear the MIN_CHARS floor, so the collapse is worth its write.
const CONTENTS = "export const value = 1\n".repeat(80)
const projectionModel = ProviderTest.model({ id: ref.modelID, providerID: ref.providerID })

const seed = Effect.fn(function* () {
  const sessions = yield* Session.Service
  const chat = yield* sessions.create({ title: "superseded" })
  const userID = MessageID.ascending()
  yield* sessions.updateMessage({
    id: userID,
    role: "user",
    sessionID: chat.id,
    agent: "build",
    model: ref,
    time: { created: Date.now() },
  } satisfies SessionV1.User)
  const assistantID = MessageID.ascending()
  yield* sessions.updateMessage({
    id: assistantID,
    role: "assistant",
    sessionID: chat.id,
    parentID: userID,
    mode: "build",
    agent: "build",
    cost: 0,
    path: { cwd: "/repo", root: "/repo" },
    tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
    modelID: ref.modelID,
    providerID: ref.providerID,
    time: { created: Date.now() },
  } satisfies SessionV1.Assistant)

  const tool = Effect.fn(function* (
    name: string,
    input: Record<string, unknown>,
    output: string,
    metadata: Record<string, unknown> = {},
  ) {
    const part = {
      id: PartID.ascending(),
      sessionID: chat.id,
      messageID: assistantID,
      type: "tool" as const,
      tool: name,
      callID: PartID.ascending(),
      state: {
        status: "completed" as const,
        input,
        output,
        title: name,
        metadata,
        time: { start: 0, end: 1 },
      },
    }
    yield* sessions.updatePart(part)
    return part
  })

  // Mirrors what the real read tool records. Coverage is judged on the window a
  // read actually returned and whether it stopped short, never on the offset and
  // limit it asked for, so fixtures have to carry the same thing.
  const read = Effect.fn(function* (
    filePath: string,
    window: { start?: number; end?: number; truncated?: boolean } = {},
    output: string = CONTENTS,
  ) {
    const lineStart = window.start ?? 1
    return yield* tool("read", { filePath, ...(window.start ? { offset: window.start } : {}) }, output, {
      display: {
        type: "file",
        path: filePath,
        lineStart,
        lineEnd: window.end ?? lineStart + 79,
        truncated: window.truncated ?? false,
      },
    })
  })

  return { chat, tool, read, sessions }
})

describe("session.superseded", () => {
  it.instance("collapses a read that a later edit invalidated", () =>
    Effect.gen(function* () {
      const { chat, tool, read, sessions } = yield* seed()
      const stale = yield* read("/repo/src/a.ts")
      const edit = yield* tool("edit", { filePath: "/repo/src/a.ts" }, "edited")

      const collapsed = yield* Superseded.collapse({
        sessionID: SessionID.make(chat.id),
        sessions,
        directory: "/repo",
      })
      expect(collapsed).toBe(1)

      const after = yield* sessions.findPart({ sessionID: chat.id, partID: stale.id })
      const record = after && compactionOf(after)
      expect(record?.summary).toContain("Stale snapshot of /repo/src/a.ts")
      // The note names the part that invalidated it, so the model can go
      // straight to the current state instead of guessing.
      expect(record?.summary).toContain(edit.id)
      // Never destructive: the bytes stay for read_part.
      expect(after?.type === "tool" && after.state.status === "completed" && after.state.output).toBe(CONTENTS)
    }),
  )

  it.instance("holds the compaction lock across its initial read and writes", () =>
    Effect.gen(function* () {
      const { chat, tool, read, sessions } = yield* seed()
      const stale = yield* read("/repo/src/a.ts")
      yield* tool("edit", { filePath: "/repo/src/a.ts" }, "edited")
      const entered = yield* Deferred.make<void>()
      const apply = yield* Deferred.make<void>()
      const release = yield* Deferred.make<void>()
      const holder = yield* withCompactionLock(
        chat.id,
        undefined,
        Effect.gen(function* () {
          yield* Deferred.succeed(entered, undefined)
          yield* Deferred.await(apply)
          const latest = yield* sessions.findPart({ sessionID: chat.id, partID: stale.id })
          if (latest?.type !== "tool" || latest.state.status !== "completed") throw new Error("tool part missing")
          latest.state.compactionGroup = "manual-group"
          latest.state.compactionSummary = "manual summary"
          latest.state.compactionGeneration = 1
          yield* sessions.updatePart(latest)
          yield* Deferred.await(release)
        }),
      ).pipe(Effect.forkScoped)
      yield* Deferred.await(entered)

      const collapse = yield* Superseded.collapse({
        sessionID: SessionID.make(chat.id),
        sessions,
        directory: "/repo",
      }).pipe(Effect.forkScoped)
      yield* pollWithTimeout(
        Effect.sync(() => (compactionLockReferences(chat.id) === 2 ? true : undefined)),
        "superseded collapse did not enter the shared compaction lock",
      )
      expect(collapse.pollUnsafe()).toBeUndefined()

      yield* Deferred.succeed(apply, undefined)
      yield* Deferred.succeed(release, undefined)
      yield* Fiber.join(holder)
      expect(yield* Fiber.join(collapse)).toBe(0)

      const stored = yield* sessions.findPart({ sessionID: chat.id, partID: stale.id })
      expect(stored && compactionOf(stored)).toMatchObject({ group: "manual-group", summary: "manual summary" })
    }),
  )

  it.instance("returns only after the next model projection sees the collapse", () =>
    Effect.gen(function* () {
      const { chat, tool, read, sessions } = yield* seed()
      yield* read("/repo/src/a.ts")
      yield* tool("edit", { filePath: "/repo/src/a.ts" }, "edited")
      const before = JSON.stringify(
        yield* MessageV2.toModelMessagesEffect(yield* sessions.messages({ sessionID: chat.id }), projectionModel),
      )

      expect(before).toContain("export const value = 1")
      expect(yield* Superseded.collapse({ sessionID: chat.id, sessions, directory: "/repo" })).toBe(1)

      const after = JSON.stringify(
        yield* MessageV2.toModelMessagesEffect(yield* sessions.messages({ sessionID: chat.id }), projectionModel),
      )
      expect(after).not.toContain("export const value = 1")
      expect(after).toContain("Stale snapshot of /repo/src/a.ts")
    }),
  )

  it.instance("leaves the newest read of a file authoritative", () =>
    Effect.gen(function* () {
      const { chat, read, sessions } = yield* seed()
      const first = yield* read("/repo/src/a.ts")
      const second = yield* read("/repo/src/a.ts")

      expect(yield* Superseded.collapse({ sessionID: SessionID.make(chat.id), sessions, directory: "/repo" })).toBe(1)

      const older = yield* sessions.findPart({ sessionID: chat.id, partID: first.id })
      const newer = yield* sessions.findPart({ sessionID: chat.id, partID: second.id })
      expect(older && compactionOf(older)).toBeTruthy()
      expect(newer && compactionOf(newer)).toBeFalsy()
    }),
  )

  it.instance("keeps a read of a different range of the same file", () =>
    Effect.gen(function* () {
      const { chat, read, sessions } = yield* seed()
      const head = yield* read("/repo/src/a.ts", { start: 1, end: 100 })
      yield* read("/repo/src/a.ts", { start: 500, end: 600 })

      expect(yield* Superseded.collapse({ sessionID: SessionID.make(chat.id), sessions, directory: "/repo" })).toBe(0)
      const after = yield* sessions.findPart({ sessionID: chat.id, partID: head.id })
      expect(after && compactionOf(after)).toBeFalsy()
    }),
  )

  it.instance("treats relative and absolute paths as the same file", () =>
    Effect.gen(function* () {
      const { chat, tool, read, sessions } = yield* seed()
      const stale = yield* read("src/a.ts")
      yield* tool("write", { filePath: "/repo/src/a.ts" }, "written")

      expect(yield* Superseded.collapse({ sessionID: SessionID.make(chat.id), sessions, directory: "/repo" })).toBe(1)
      const after = yield* sessions.findPart({ sessionID: chat.id, partID: stale.id })
      expect(after && compactionOf(after)).toBeTruthy()
    }),
  )

  it.instance("leaves reads of untouched files and small reads alone", () =>
    Effect.gen(function* () {
      const { chat, tool, read, sessions } = yield* seed()
      const other = yield* read("/repo/src/b.ts")
      const small = yield* read("/repo/src/c.ts", {}, "tiny")
      yield* tool("edit", { filePath: "/repo/src/c.ts" }, "edited")
      yield* tool("edit", { filePath: "/repo/src/a.ts" }, "edited")

      expect(yield* Superseded.collapse({ sessionID: SessionID.make(chat.id), sessions, directory: "/repo" })).toBe(0)
      const untouched = yield* sessions.findPart({ sessionID: chat.id, partID: other.id })
      const tiny = yield* sessions.findPart({ sessionID: chat.id, partID: small.id })
      expect(untouched && compactionOf(untouched)).toBeFalsy()
      expect(tiny && compactionOf(tiny)).toBeFalsy()
    }),
  )

  it.instance("does not let a truncated read retire lines it never returned", () =>
    Effect.gen(function* () {
      const { chat, read, sessions } = yield* seed()
      // Deep window, capped by the line limit.
      const deep = yield* read("/repo/src/a.ts", { start: 4001, end: 6000, truncated: true })
      // A plain re-read of the same file, which stops long before those lines.
      // Judged on its request alone it would look like it covered everything.
      yield* read("/repo/src/a.ts", { start: 1, end: 2000, truncated: true })

      expect(yield* Superseded.collapse({ sessionID: SessionID.make(chat.id), sessions, directory: "/repo" })).toBe(0)
      const after = yield* sessions.findPart({ sessionID: chat.id, partID: deep.id })
      expect(after && compactionOf(after)).toBeFalsy()
    }),
  )

  it.instance("collapses a read that apply_patch invalidated", () =>
    Effect.gen(function* () {
      const { chat, tool, read, sessions } = yield* seed()
      const stale = yield* read("/repo/src/a.ts")
      // apply_patch is handed a patch, never a path, so the files it rewrote are
      // only discoverable from its result.
      const patch = yield* tool("apply_patch", { patchText: "*** Begin Patch" }, "applied", {
        files: [{ filePath: "/repo/src/a.ts", type: "update" }],
      })

      expect(yield* Superseded.collapse({ sessionID: SessionID.make(chat.id), sessions, directory: "/repo" })).toBe(1)
      const after = yield* sessions.findPart({ sessionID: chat.id, partID: stale.id })
      expect(after && compactionOf(after)?.summary).toContain(patch.id)
    }),
  )

  it.instance("is idempotent", () =>
    Effect.gen(function* () {
      const { chat, tool, read, sessions } = yield* seed()
      yield* read("/repo/src/a.ts")
      yield* tool("edit", { filePath: "/repo/src/a.ts" }, "edited")

      expect(yield* Superseded.collapse({ sessionID: SessionID.make(chat.id), sessions, directory: "/repo" })).toBe(1)
      expect(yield* Superseded.collapse({ sessionID: SessionID.make(chat.id), sessions, directory: "/repo" })).toBe(0)
    }),
  )
})
