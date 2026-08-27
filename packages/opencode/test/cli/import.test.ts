import { test, expect } from "bun:test"
import type { Session as SDKSession } from "@opencode-ai/sdk/v2"
import { Database } from "@opencode-ai/core/database/database"
import { ProjectV2 } from "@opencode-ai/core/project"
import { ProjectTable } from "@opencode-ai/core/project/sql"
import { AbsolutePath } from "@opencode-ai/core/schema"
import { SessionTable } from "@opencode-ai/core/session/sql"
import { eq } from "drizzle-orm"
import { Effect } from "effect"
import {
  formatImportFileError,
  parseShareUrl,
  shouldAttachShareAuthHeaders,
  transformShareData,
  upsertImportedSession,
  type ShareData,
} from "../../src/cli/cmd/import"
import { FSUtil } from "@opencode-ai/core/fs-util"
import { PlatformError } from "effect"
import { SessionID } from "../../src/session/schema"
import { testEffect } from "../lib/effect"

const it = testEffect(Database.layerFromPath(":memory:"))

function importedSession(input: { id: string; parentID?: string; taskParentID?: string }): SDKSession {
  return {
    id: input.id,
    slug: input.id,
    projectID: ProjectV2.ID.global,
    directory: "/source",
    ...(input.parentID === undefined ? {} : { parentID: input.parentID }),
    ...(input.taskParentID === undefined ? {} : { taskParentID: input.taskParentID }),
    title: "Imported session",
    version: "test",
    time: { created: 1, updated: 1 },
  }
}

const importLocation = {
  projectID: ProjectV2.ID.global,
  directory: "/target",
  worktree: "/target",
}

const ensureProject = Database.Service.use(({ db }) =>
  db
    .insert(ProjectTable)
    .values({ id: ProjectV2.ID.global, worktree: AbsolutePath.make("/target"), sandboxes: [] })
    .onConflictDoNothing()
    .run()
    .pipe(Effect.orDie),
)

test("formats import file errors", () => {
  expect(
    formatImportFileError(
      "test.json",
      new PlatformError.PlatformError(
        new PlatformError.SystemError({
          _tag: "NotFound",
          module: "FileSystem",
          method: "readFileString",
        }),
      ),
    ),
  ).toBe("File not found: test.json")
  expect(
    formatImportFileError(
      "test.json",
      new PlatformError.PlatformError(
        new PlatformError.SystemError({
          _tag: "PermissionDenied",
          module: "FileSystem",
          method: "readFileString",
        }),
      ),
    ),
  ).toBe("Failed to read file: Permission denied")
  expect(
    formatImportFileError(
      "test.json",
      new FSUtil.FileSystemError({ method: "readJson", cause: new SyntaxError("Unexpected token") }),
    ),
  ).toBe("Invalid JSON in test.json: Unexpected token")
})

// parseShareUrl tests
test("parses valid share URLs", () => {
  expect(parseShareUrl("https://opncd.ai/share/Jsj3hNIW")).toBe("Jsj3hNIW")
  expect(parseShareUrl("https://custom.example.com/share/abc123")).toBe("abc123")
  expect(parseShareUrl("http://localhost:3000/share/test_id-123")).toBe("test_id-123")
})

test("rejects invalid URLs", () => {
  expect(parseShareUrl("https://opncd.ai/s/Jsj3hNIW")).toBeNull() // legacy format
  expect(parseShareUrl("https://opncd.ai/share/")).toBeNull()
  expect(parseShareUrl("https://opncd.ai/share/id/extra")).toBeNull()
  expect(parseShareUrl("not-a-url")).toBeNull()
})

test("only attaches share auth headers for same-origin URLs", () => {
  expect(shouldAttachShareAuthHeaders("https://control.example.com/share/abc", "https://control.example.com")).toBe(
    true,
  )
  expect(shouldAttachShareAuthHeaders("https://other.example.com/share/abc", "https://control.example.com")).toBe(false)
  expect(shouldAttachShareAuthHeaders("https://control.example.com:443/share/abc", "https://control.example.com")).toBe(
    true,
  )
  expect(shouldAttachShareAuthHeaders("not-a-url", "https://control.example.com")).toBe(false)
})

// transformShareData tests
test("transforms share data to storage format", () => {
  const data: ShareData[] = [
    { type: "session", data: { id: "sess-1", title: "Test" } as any },
    { type: "message", data: { id: "msg-1", sessionID: "sess-1" } as any },
    { type: "part", data: { id: "part-1", messageID: "msg-1" } as any },
    { type: "part", data: { id: "part-2", messageID: "msg-1" } as any },
  ]

  const result = transformShareData(data)!

  expect(result.info.id).toBe("sess-1")
  expect(result.messages).toHaveLength(1)
  expect(result.messages[0].parts).toHaveLength(2)
})

test("returns null for invalid share data", () => {
  expect(transformShareData([])).toBeNull()
  expect(transformShareData([{ type: "message", data: {} as any }])).toBeNull()
  expect(transformShareData([{ type: "session", data: { id: "s" } as any }])).toBeNull() // no messages
})

it.live("strips Task ownership from a new imported session", () =>
  Effect.gen(function* () {
    yield* ensureProject
    const { db } = yield* Database.Service
    const id = SessionID.descending()
    const parentID = SessionID.descending()

    yield* upsertImportedSession({
      info: importedSession({ id, parentID, taskParentID: parentID }),
      ...importLocation,
    })

    const row = yield* db
      .select({ parentID: SessionTable.parent_id, taskParentID: SessionTable.task_parent_id })
      .from(SessionTable)
      .where(eq(SessionTable.id, id))
      .get()
      .pipe(Effect.orDie)
    expect(row).toEqual({ parentID, taskParentID: null })
  }),
)

it.live("preserves an existing Task owner during an import conflict", () =>
  Effect.gen(function* () {
    yield* ensureProject
    const { db } = yield* Database.Service
    const id = SessionID.descending()
    const ownerID = SessionID.descending()

    yield* upsertImportedSession({ info: importedSession({ id }), ...importLocation })
    yield* db
      .update(SessionTable)
      .set({ parent_id: ownerID, task_parent_id: ownerID })
      .where(eq(SessionTable.id, id))
      .run()
      .pipe(Effect.orDie)
    yield* upsertImportedSession({
      info: importedSession({ id, parentID: SessionID.descending(), taskParentID: SessionID.descending() }),
      ...importLocation,
    })

    const row = yield* db
      .select({ parentID: SessionTable.parent_id, taskParentID: SessionTable.task_parent_id })
      .from(SessionTable)
      .where(eq(SessionTable.id, id))
      .get()
      .pipe(Effect.orDie)
    expect(row).toEqual({ parentID: ownerID, taskParentID: ownerID })
  }),
)
