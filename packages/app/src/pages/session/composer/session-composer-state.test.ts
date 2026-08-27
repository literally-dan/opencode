import { describe, expect, test } from "bun:test"
import type { PermissionRequest, QuestionRequest, Session } from "@opencode-ai/sdk/v2/client"
import { todoDockAtBoundary, todoState } from "./session-composer-state"
import { sessionPermissionRequest, sessionQuestionRequest, sessionRequestTree } from "./session-request-tree"

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
    directory: "/workspace",
    path: "project",
    ...input,
    id: input.id,
  }) as Session

const taskSession = (id: string, parentID: string, input: Partial<Session> = {}) =>
  session({ id, parentID, taskParentID: parentID, ...input })

const permission = (id: string, sessionID: string) =>
  ({
    id,
    sessionID,
  }) as PermissionRequest

const question = (id: string, sessionID: string) =>
  ({
    id,
    sessionID,
    questions: [],
  }) as QuestionRequest

describe("sessionRequestTree", () => {
  test("lists the session and its Task descendants but not parent-only children", () => {
    const sessions = [
      session({ id: "root" }),
      taskSession("child", "root"),
      taskSession("grand", "child"),
      session({ id: "api-child", parentID: "root" }),
      session({ id: "other" }),
    ]

    expect(sessionRequestTree(sessions, "root")).toEqual(["root", "child", "grand"])
  })
})

describe("sessionPermissionRequest", () => {
  test("prefers the current session permission", () => {
    const sessions = [session({ id: "root" }), taskSession("child", "root")]
    const permissions = {
      root: [permission("perm-root", "root")],
      child: [permission("perm-child", "child")],
    }

    expect(sessionPermissionRequest(sessions, permissions, "root")?.id).toBe("perm-root")
  })

  test("returns a nested child permission", () => {
    const sessions = [
      session({ id: "root" }),
      taskSession("child", "root"),
      taskSession("grand", "child"),
      session({ id: "other" }),
    ]
    const permissions = {
      grand: [permission("perm-grand", "grand")],
      other: [permission("perm-other", "other")],
    }

    expect(sessionPermissionRequest(sessions, permissions, "root")?.id).toBe("perm-grand")
  })

  test("does not route parent-only child requests through the root", () => {
    const sessions = [session({ id: "root" }), session({ id: "child", parentID: "root" })]

    expect(sessionPermissionRequest(sessions, { child: [permission("perm-child", "child")] }, "root")).toBeUndefined()
    expect(sessionQuestionRequest(sessions, { child: [question("q-child", "child")] }, "root")).toBeUndefined()
  })

  test("does not route requests from a session whose Task owner differs from its parent", () => {
    const sessions = [session({ id: "root" }), session({ id: "child", parentID: "root", taskParentID: "other" })]

    expect(sessionPermissionRequest(sessions, { child: [permission("perm-child", "child")] }, "root")).toBeUndefined()
    expect(sessionQuestionRequest(sessions, { child: [question("q-child", "child")] }, "root")).toBeUndefined()
  })

  test("does not route Task requests from another project through the root", () => {
    const sessions = [session({ id: "root" }), taskSession("child", "root", { projectID: "foreign" })]

    expect(sessionPermissionRequest(sessions, { child: [permission("perm-child", "child")] }, "root")).toBeUndefined()
  })

  test("routes Task requests through a root that moved to another location", () => {
    const sessions = [
      session({ id: "root", workspaceID: "moved", directory: "/moved", path: "moved" }),
      taskSession("child", "root"),
      taskSession("grand", "child"),
    ]

    expect(sessionPermissionRequest(sessions, { grand: [permission("perm-grand", "grand")] }, "root")?.id).toBe(
      "perm-grand",
    )
    expect(sessionQuestionRequest(sessions, { child: [question("q-child", "child")] }, "root")?.id).toBe("q-child")
  })

  test("returns undefined without a matching tree permission", () => {
    const sessions = [session({ id: "root" }), taskSession("child", "root")]
    const permissions = {
      other: [permission("perm-other", "other")],
    }

    expect(sessionPermissionRequest(sessions, permissions, "root")).toBeUndefined()
  })

  test("skips filtered permissions in the current tree", () => {
    const sessions = [session({ id: "root" }), taskSession("child", "root")]
    const permissions = {
      root: [permission("perm-root", "root")],
      child: [permission("perm-child", "child")],
    }

    expect(sessionPermissionRequest(sessions, permissions, "root", (item) => item.id !== "perm-root"))?.toMatchObject({
      id: "perm-child",
    })
  })

  test("returns undefined when all tree permissions are filtered out", () => {
    const sessions = [session({ id: "root" }), taskSession("child", "root")]
    const permissions = {
      root: [permission("perm-root", "root")],
      child: [permission("perm-child", "child")],
    }

    expect(sessionPermissionRequest(sessions, permissions, "root", () => false)).toBeUndefined()
  })
})

describe("sessionQuestionRequest", () => {
  test("prefers the current session question", () => {
    const sessions = [session({ id: "root" }), taskSession("child", "root")]
    const questions = {
      root: [question("q-root", "root")],
      child: [question("q-child", "child")],
    }

    expect(sessionQuestionRequest(sessions, questions, "root")?.id).toBe("q-root")
  })

  test("returns a nested child question", () => {
    const sessions = [session({ id: "root" }), taskSession("child", "root"), taskSession("grand", "child")]
    const questions = {
      grand: [question("q-grand", "grand")],
    }

    expect(sessionQuestionRequest(sessions, questions, "root")?.id).toBe("q-grand")
  })
})

describe("todoState", () => {
  test("hides when there are no todos", () => {
    expect(todoState({ count: 0, done: false, live: true })).toBe("hide")
  })

  test("opens while the session is still working", () => {
    expect(todoState({ count: 2, done: false, live: true })).toBe("open")
  })

  test("closes completed todos after a running turn", () => {
    expect(todoState({ count: 2, done: true, live: true })).toBe("close")
  })

  test("clears stale todos when the turn ends", () => {
    expect(todoState({ count: 2, done: false, live: false })).toBe("clear")
  })

  test("clears completed todos when the session is no longer live", () => {
    expect(todoState({ count: 2, done: true, live: false })).toBe("clear")
  })
})

describe("todoDockAtBoundary", () => {
  test("shows active todos when entering a session", () => {
    expect(todoDockAtBoundary("open")).toBe(true)
  })

  test("hides completed todos when entering a session", () => {
    expect(todoDockAtBoundary("close")).toBe(false)
  })
})
