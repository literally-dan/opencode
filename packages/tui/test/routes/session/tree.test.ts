import { describe, expect, test } from "bun:test"
import type { Session } from "@opencode-ai/sdk/v2"
import { sessionTree } from "../../../src/routes/session/tree"

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

  test("excludes Task edges with a different location", () => {
    const sessions = [
      session("root"),
      taskSession("project", "root", { projectID: "foreign" }),
      taskSession("workspace", "root", { workspaceID: "foreign" }),
      taskSession("directory", "root", { directory: "/foreign" }),
      taskSession("path", "root", { path: "foreign" }),
      taskSession("legacy-path", "root", { path: undefined }),
    ]

    expect(sessionTree(sessions, "root").map((item) => item.id)).toEqual(["root", "legacy-path"])
  })
})
