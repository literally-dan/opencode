import { describe, expect, test } from "bun:test"
import type { PermissionRequest, Session } from "@opencode-ai/sdk/v2/client"
import { base64Encode } from "@opencode-ai/core/util/encode"
import { autoRespondsPermission, isDirectoryAutoAccepting, sessionAutoAccept } from "./permission-auto-respond"

const session = (input: {
  id: string
  parentID?: string
  taskParentID?: string
  projectID?: string
  workspaceID?: string
  directory?: string
  path?: string
}) =>
  ({
    projectID: "project",
    workspaceID: "workspace",
    directory: "/tmp/project",
    path: "project",
    ...input,
    id: input.id,
  }) as Session

const taskSession = (id: string, parentID: string, input: Partial<Session> = {}) =>
  session({ id, parentID, taskParentID: parentID, ...input })

const permission = (sessionID: string) =>
  ({
    sessionID,
  }) as Pick<PermissionRequest, "sessionID">

describe("autoRespondsPermission", () => {
  test("uses a parent session's directory-scoped auto-accept", () => {
    const directory = "/tmp/project"
    const sessions = [session({ id: "root" }), taskSession("child", "root")]
    const autoAccept = {
      [`${base64Encode(directory)}/root`]: true,
    }

    expect(autoRespondsPermission(autoAccept, sessions, permission("child"), directory)).toBe(true)
  })

  test("does not inherit auto-accept through a parent-only session", () => {
    const directory = "/tmp/project"
    const sessions = [session({ id: "root" }), session({ id: "child", parentID: "root" })]

    expect(
      autoRespondsPermission({ [`${base64Encode(directory)}/root`]: true }, sessions, permission("child"), directory),
    ).toBe(false)
  })

  test("does not inherit auto-accept across a Task location boundary", () => {
    const directory = "/tmp/project"
    const sessions = [session({ id: "root" }), taskSession("child", "root", { directory: "/tmp/foreign-project" })]

    expect(
      autoRespondsPermission({ [`${base64Encode(directory)}/root`]: true }, sessions, permission("child"), directory),
    ).toBe(false)
  })

  test("uses a parent session's legacy auto-accept key", () => {
    const sessions = [session({ id: "root" }), taskSession("child", "root")]

    expect(autoRespondsPermission({ root: true }, sessions, permission("child"), "/tmp/project")).toBe(true)
  })

  test("defaults to requiring approval when no lineage override exists", () => {
    const sessions = [session({ id: "root" }), taskSession("child", "root"), session({ id: "other" })]
    const autoAccept = {
      other: true,
    }

    expect(autoRespondsPermission(autoAccept, sessions, permission("child"), "/tmp/project")).toBe(false)
  })

  test("inherits a parent session's false override", () => {
    const directory = "/tmp/project"
    const sessions = [session({ id: "root" }), taskSession("child", "root")]
    const autoAccept = {
      [`${base64Encode(directory)}/root`]: false,
    }

    expect(autoRespondsPermission(autoAccept, sessions, permission("child"), directory)).toBe(false)
  })

  test("prefers a child override over parent override", () => {
    const directory = "/tmp/project"
    const sessions = [session({ id: "root" }), taskSession("child", "root")]
    const autoAccept = {
      [`${base64Encode(directory)}/root`]: false,
      [`${base64Encode(directory)}/child`]: true,
    }

    expect(autoRespondsPermission(autoAccept, sessions, permission("child"), directory)).toBe(true)
  })

  test("falls back to directory-level auto-accept", () => {
    const directory = "/tmp/project"
    const sessions = [session({ id: "root" })]
    const autoAccept = {
      [`${base64Encode(directory)}/*`]: true,
    }

    expect(autoRespondsPermission(autoAccept, sessions, permission("root"), directory)).toBe(true)
    expect(sessionAutoAccept(autoAccept, sessions, permission("root"), directory)).toBeUndefined()
  })

  test("session-level override takes precedence over directory-level", () => {
    const directory = "/tmp/project"
    const sessions = [session({ id: "root" })]
    const autoAccept = {
      [`${base64Encode(directory)}/*`]: true,
      [`${base64Encode(directory)}/root`]: false,
    }

    expect(autoRespondsPermission(autoAccept, sessions, permission("root"), directory)).toBe(false)
  })

  test("parent false override takes precedence over directory-level auto-accept", () => {
    const directory = "/tmp/project"
    const sessions = [session({ id: "root" }), taskSession("child", "root")]
    const autoAccept = {
      [`${base64Encode(directory)}/*`]: true,
      [`${base64Encode(directory)}/root`]: false,
    }

    expect(autoRespondsPermission(autoAccept, sessions, permission("child"), directory)).toBe(false)
  })

  test("parent true override takes precedence over disabled directory fallback", () => {
    const directory = "/tmp/project"
    const sessions = [session({ id: "root" }), taskSession("child", "root")]
    const autoAccept = {
      [`${base64Encode(directory)}/*`]: false,
      [`${base64Encode(directory)}/root`]: true,
    }

    expect(autoRespondsPermission(autoAccept, sessions, permission("child"), directory)).toBe(true)
  })
})

describe("isDirectoryAutoAccepting", () => {
  test("returns true when directory key is set", () => {
    const directory = "/tmp/project"
    const autoAccept = { [`${base64Encode(directory)}/*`]: true }
    expect(isDirectoryAutoAccepting(autoAccept, directory)).toBe(true)
  })

  test("returns false when directory key is not set", () => {
    expect(isDirectoryAutoAccepting({}, "/tmp/project")).toBe(false)
  })

  test("returns false when directory key is explicitly false", () => {
    const directory = "/tmp/project"
    const autoAccept = { [`${base64Encode(directory)}/*`]: false }
    expect(isDirectoryAutoAccepting(autoAccept, directory)).toBe(false)
  })
})
