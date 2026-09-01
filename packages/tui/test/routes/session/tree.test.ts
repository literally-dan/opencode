import { describe, expect, test } from "bun:test"
import type { Session } from "@opencode-ai/sdk/v2"
import { busyTaskChildrenElsewhere, sessionTree, sessionTreeBusy } from "../../../src/routes/session/tree"

const session = (id: string, input: Partial<Session> = {}) =>
  ({
    id,
    title: id,
    projectID: "project",
    workspaceID: "workspace",
    directory: "/workspace",
    path: "project",
    ...input,
  }) as Session

const taskSession = (id: string, parentID: string, input: Partial<Session> = {}) =>
  session(id, { parentID, taskParentID: parentID, ...input })

describe("sessionTree", () => {
  test("returns the complete subtree", () => {
    const sessions = [
      session("root"),
      taskSession("child-a", "root"),
      taskSession("child-b", "root"),
      taskSession("grandchild", "child-a"),
      session("other"),
    ]

    expect(sessionTree(sessions, "root").map((item) => item.id)).toEqual(["root", "child-a", "child-b", "grandchild"])
  })

  test("returns an empty list for an unknown root", () => {
    expect(sessionTree([taskSession("child", "root")], "root")).toEqual([])
  })

  test("visits each session once when parent links cycle", () => {
    const sessions = [taskSession("root", "child"), taskSession("child", "root")]

    expect(sessionTree(sessions, "root").map((item) => item.id)).toEqual(["root", "child"])
  })

  test("excludes parent-only sessions", () => {
    const sessions = [session("root"), session("history", { parentID: "root" }), taskSession("task", "root")]

    expect(sessionTree(sessions, "root").map((item) => item.id)).toEqual(["root", "task"])
  })

  test("excludes sessions whose Task owner differs from their parent", () => {
    const sessions = [
      session("root"),
      session("claimed", { parentID: "root", taskParentID: "other" }),
      taskSession("task", "root"),
    ]

    expect(sessionTree(sessions, "root").map((item) => item.id)).toEqual(["root", "task"])
  })

  test("excludes Task edges from another project", () => {
    const sessions = [
      session("root"),
      taskSession("project", "root", { projectID: "foreign" }),
      taskSession("task", "root"),
    ]

    expect(sessionTree(sessions, "root").map((item) => item.id)).toEqual(["root", "task"])
  })

  test("keeps Task children after the root moves to another location", () => {
    const sessions = [
      session("root", { workspaceID: "moved", directory: "/moved", path: "moved" }),
      taskSession("child", "root"),
      taskSession("grandchild", "child"),
    ]

    expect(sessionTree(sessions, "root").map((item) => item.id)).toEqual(["root", "child", "grandchild"])
  })

  test("reports a running Task descendant when the root is idle", () => {
    const sessions = [session("root"), taskSession("child", "root")]

    expect(sessionTreeBusy(sessions, { root: { type: "idle" }, child: { type: "busy" } }, "root")).toBe(true)
  })

  test("ignores running sessions outside the authenticated Task subtree", () => {
    const sessions = [session("root"), session("history", { parentID: "root" })]

    expect(sessionTreeBusy(sessions, { history: { type: "busy" } }, "root")).toBe(false)
  })
})

describe("busyTaskChildrenElsewhere", () => {
  test("returns busy direct Task children outside the root's location", () => {
    const sessions = [
      session("root", { directory: "/moved" }),
      taskSession("old-directory", "root"),
      taskSession("old-workspace", "root", { directory: "/moved", workspaceID: "old" }),
      taskSession("same-location", "root", { directory: "/moved" }),
      taskSession("idle", "root"),
      taskSession("grandchild", "old-directory"),
      session("history", { parentID: "root" }),
    ]
    const busy = { type: "busy" }
    const statuses = {
      "old-directory": busy,
      "old-workspace": busy,
      "same-location": busy,
      idle: { type: "idle" },
      grandchild: busy,
      history: busy,
    }

    expect(busyTaskChildrenElsewhere(sessions, statuses, "root").map((item) => item.id)).toEqual([
      "old-directory",
      "old-workspace",
    ])
  })
})
