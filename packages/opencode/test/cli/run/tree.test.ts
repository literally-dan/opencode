import { describe, expect, test } from "bun:test"
import type { OpencodeClient } from "@opencode-ai/sdk/v2"
import {
  DescendantFetchError,
  fetchDescendants,
  lookupSessionTreeOwnership,
  resolveSessionTreeOwnership,
} from "@/cli/cmd/run/tree"
import { replyPermission } from "@/session/permission-reply"

type SessionInfo = {
  id: string
  title?: string
  parentID?: string
  taskParentID?: string
  projectID: string
  workspaceID?: string
  directory: string
  path?: string
}

function sessionInfo(id: string, input: Partial<Omit<SessionInfo, "id">> = {}): SessionInfo {
  return {
    id,
    projectID: "project",
    directory: "/workspace",
    ...input,
  }
}

function taskInfo(id: string, parentID: string, input: Partial<Omit<SessionInfo, "id" | "parentID">> = {}) {
  return { id, parentID, taskParentID: parentID, ...input }
}

function taskMessages(...sessionIDs: string[]) {
  return [
    {
      info: { role: "assistant" },
      parts: sessionIDs.map((sessionId) => ({
        type: "tool",
        tool: "task",
        state: { metadata: { sessionId } },
      })),
    },
  ]
}

function client(input: {
  children?: (
    sessionID: string,
    signal?: AbortSignal,
  ) => Promise<Array<Pick<SessionInfo, "id"> & Partial<Omit<SessionInfo, "id">>>>
  get?: (sessionID: string) => Promise<Partial<Omit<SessionInfo, "id">>>
  messages?: (sessionID: string) => Promise<ReturnType<typeof taskMessages>>
  reply?: (input: { requestID: string; reply: "once" | "always" | "reject" }) => Promise<{
    data?: boolean
    error?: unknown
  }>
}) {
  return {
    permission: {
      reply: (params: { requestID: string; reply: "once" | "always" | "reject" }) => input.reply?.(params),
    },
    session: {
      children: (params: { sessionID: string }, options?: { signal?: AbortSignal }) =>
        input.children
          ? input
              .children(params.sessionID, options?.signal)
              .then((data) => ({ data: data.map((item) => sessionInfo(item.id, item)) }))
          : undefined,
      get: (params: { sessionID: string }) =>
        (input.get?.(params.sessionID) ?? Promise.resolve({})).then((data) => ({
          data: sessionInfo(params.sessionID, data),
        })),
      messages: (params: { sessionID: string }) =>
        (input.messages?.(params.sessionID) ?? Promise.resolve([])).then((data) => ({ data })),
    },
  } as unknown as OpencodeClient
}

describe("run session tree", () => {
  test("surfaces descendant lookup failures with successfully traversed branches", async () => {
    const sdk = client({
      children: async (sessionID) => {
        if (sessionID === "root") return [taskInfo("child-a", "root"), taskInfo("child-b", "root")]
        if (sessionID === "child-a") throw new Error("child-a unavailable")
        if (sessionID === "child-b") return [taskInfo("grandchild-b", "child-b")]
        return []
      },
    })

    try {
      await fetchDescendants(sdk, "root")
      throw new Error("expected descendant traversal to fail")
    } catch (error) {
      expect(error).toBeInstanceOf(DescendantFetchError)
      if (!(error instanceof DescendantFetchError)) return
      expect(error.failures.map((item) => item.sessionID)).toEqual(["child-a"])
      expect(error.partial.map((item) => item.id)).toEqual(["child-a", "child-b", "grandchild-b"])
    }
  })

  test("does not revisit a cycle", async () => {
    const sdk = client({
      children: async (sessionID) => {
        if (sessionID === "root") return [taskInfo("child", "root")]
        if (sessionID === "child") return [taskInfo("root", "child")]
        return []
      },
    })

    expect(await fetchDescendants(sdk, "root")).toEqual([{ id: "child" }])
  })

  test("returns genuine nested Task descendants", async () => {
    const sdk = client({
      children: async (sessionID) => {
        if (sessionID === "root") return [taskInfo("child", "root", { title: "Child" })]
        if (sessionID === "child") return [taskInfo("grandchild", "child")]
        return []
      },
    })

    expect(await fetchDescendants(sdk, "root")).toEqual([{ id: "child", title: "Child" }, { id: "grandchild" }])
  })

  test("excludes parent-only descendants", async () => {
    const visited: string[] = []
    const sdk = client({
      children: async (sessionID) => {
        visited.push(sessionID)
        if (sessionID === "root") return [{ id: "history", parentID: "root" }]
        return [taskInfo("unexpected", sessionID)]
      },
    })

    expect(await fetchDescendants(sdk, "root")).toEqual([])
    expect(visited).toEqual(["root"])
  })

  test("excludes Task descendants across a location boundary", async () => {
    const sdk = client({
      children: async (sessionID) =>
        sessionID === "root" ? [taskInfo("foreign", "root", { directory: "/foreign" })] : [],
    })

    expect(await fetchDescendants(sdk, "root")).toEqual([])
  })

  test("bounds concurrent descendant lookups", async () => {
    let active = 0
    let maxActive = 0
    const sdk = client({
      children: async (sessionID) => {
        if (sessionID === "root") return Array.from({ length: 6 }, (_, index) => taskInfo(`child-${index}`, "root"))
        active += 1
        maxActive = Math.max(maxActive, active)
        await Bun.sleep(20)
        active -= 1
        return []
      },
    })

    await fetchDescendants(sdk, "root", { concurrency: 2 })
    expect(maxActive).toBe(2)
  })

  test("uses bounded defaults for invalid options", async () => {
    let active = 0
    let maxActive = 0
    const sdk = client({
      children: async (sessionID) => {
        if (sessionID === "root") return Array.from({ length: 6 }, (_, index) => taskInfo(`child-${index}`, "root"))
        active += 1
        maxActive = Math.max(maxActive, active)
        await Bun.sleep(20)
        active -= 1
        return []
      },
    })

    await fetchDescendants(sdk, "root", { concurrency: Number.POSITIVE_INFINITY, limit: Number.NaN })
    expect(maxActive).toBe(4)
  })

  test("returns partial data when the session limit is reached", async () => {
    const sdk = client({
      children: async (sessionID) =>
        sessionID === "root"
          ? [taskInfo("child-a", "root"), taskInfo("child-b", "root"), taskInfo("child-c", "root")]
          : [],
    })

    try {
      await fetchDescendants(sdk, "root", { limit: 2 })
      throw new Error("expected descendant traversal to reach its limit")
    } catch (error) {
      expect(error).toBeInstanceOf(DescendantFetchError)
      if (!(error instanceof DescendantFetchError)) return
      expect(error.partial).toEqual([{ id: "child-a" }])
      expect(String(error.failures[0]?.error)).toContain("session tree exceeds 2 entries")
    }
  })

  test("cancels descendant lookups", async () => {
    let started: (() => void) | undefined
    const ready = new Promise<void>((resolve) => {
      started = resolve
    })
    const sdk = client({
      children: async (sessionID, signal) => {
        if (sessionID === "root") return [taskInfo("child", "root")]
        started?.()
        return new Promise((_, reject) => {
          signal?.addEventListener("abort", () => reject(signal.reason), { once: true })
        })
      },
    })
    const controller = new AbortController()
    const pending = fetchDescendants(sdk, "root", { signal: controller.signal })

    await ready
    controller.abort(new Error("stopped"))
    await expect(pending).rejects.toThrow("stopped")
  })

  test("times out stalled descendant lookups", async () => {
    const sdk = client({
      children: async (sessionID, signal) => {
        if (sessionID === "root") return [taskInfo("child", "root")]
        return new Promise((_, reject) => {
          signal?.addEventListener("abort", () => reject(signal.reason), { once: true })
        })
      },
    })

    await expect(fetchDescendants(sdk, "root", { timeoutMs: 20 })).rejects.toBeInstanceOf(DescendantFetchError)
  })

  test("bounds total traversal time", async () => {
    const sdk = client({
      children: async (sessionID, signal) => {
        if (sessionID === "root") return [taskInfo("child", "root")]
        return new Promise((_, reject) => {
          signal?.addEventListener("abort", () => reject(signal.reason), { once: true })
        })
      },
    })

    const started = Date.now()
    await expect(fetchDescendants(sdk, "root", { timeoutMs: 1_000, totalTimeoutMs: 20 })).rejects.toBeInstanceOf(
      DescendantFetchError,
    )
    expect(Date.now() - started).toBeLessThan(500)
  })

  test("models a failed ancestry lookup as unknown", async () => {
    let lookups = 0
    const sdk = client({
      get: () => {
        lookups += 1
        return Promise.reject(new Error("session lookup unavailable"))
      },
    })

    const ownership = await lookupSessionTreeOwnership(sdk, new Set(["root"]), "leaf")
    expect(ownership.type).toBe("unknown")
    expect(lookups).toBe(1)
    if (ownership.type === "unknown") expect(String(ownership.error)).toContain("session lookup unavailable")
  })

  test("cancels retained ancestry retries", async () => {
    let lookups = 0
    let retryStarted: (() => void) | undefined
    const started = new Promise<void>((resolve) => {
      retryStarted = resolve
    })
    const sdk = client({
      get: () => {
        lookups += 1
        return Promise.reject(new Error("session lookup unavailable"))
      },
    })
    const controller = new AbortController()
    const resolving = resolveSessionTreeOwnership(sdk, new Set(["root"]), "leaf", {
      signal: controller.signal,
      onUnknown: () => retryStarted?.(),
    })

    await started
    controller.abort()
    expect(await resolving).toEqual({ type: "cancelled" })
    const stoppedAt = lookups
    await Bun.sleep(150)
    expect(lookups).toBe(stoppedAt)
  })

  test("cancels a stalled ancestry lookup that ignores abort", async () => {
    let started: (() => void) | undefined
    const ready = new Promise<void>((resolve) => {
      started = resolve
    })
    const sdk = client({
      get: () => {
        started?.()
        return new Promise(() => {})
      },
    })
    const controller = new AbortController()
    const resolving = resolveSessionTreeOwnership(sdk, new Set(["root"]), "leaf", {
      signal: controller.signal,
      attempts: 1,
      timeoutMs: 1_000,
    })

    await ready
    controller.abort()

    expect(await resolving).toEqual({ type: "cancelled" })
  })

  test("times out a stalled ancestry lookup", async () => {
    const sdk = client({ get: () => new Promise(() => {}) })
    const started = Date.now()
    const resolution = await resolveSessionTreeOwnership(sdk, new Set(["root"]), "leaf", {
      signal: new AbortController().signal,
      attempts: 1,
      timeoutMs: 20,
    })

    expect(resolution.type).toBe("exhausted")
    if (resolution.type === "exhausted") expect(String(resolution.error)).toContain("timed out after 20ms")
    expect(Date.now() - started).toBeLessThan(500)
  })

  test("returns foreign only after reaching a known root", async () => {
    const sdk = client({
      get: () => Promise.resolve({ parentID: undefined }),
    })

    expect(await lookupSessionTreeOwnership(sdk, new Set(["root"]), "foreign")).toEqual({ type: "foreign" })
  })

  test("resolves an ancestry path without caching descendant trust", async () => {
    const parents = new Map<string, string | undefined>([
      ["leaf", "middle"],
      ["middle", "root"],
      ["root", undefined],
    ])
    let messageLookups = 0
    const sdk = client({
      get: async (sessionID) => {
        if (!parents.has(sessionID)) throw new Error(`unexpected lookup: ${sessionID}`)
        return { parentID: parents.get(sessionID), taskParentID: parents.get(sessionID) }
      },
      messages: async () => {
        messageLookups += 1
        return []
      },
    })
    const tree = new Set(["root"])

    expect(await resolveSessionTreeOwnership(sdk, tree, "leaf", { signal: new AbortController().signal })).toEqual({
      type: "owned",
      value: undefined,
    })
    expect(tree).toEqual(new Set(["root"]))
    expect(messageLookups).toBe(0)
  })

  test("rejects a claimed parent without a server-owned Task edge", async () => {
    const sdk = client({
      get: async (sessionID) => ({ parentID: sessionID === "leaf" ? "root" : undefined }),
    })

    expect(await lookupSessionTreeOwnership(sdk, new Set(["root"]), "leaf")).toEqual({ type: "foreign" })
  })

  test("rejects Task ancestry across a location boundary", async () => {
    const sdk = client({
      get: async (sessionID) => ({
        parentID: sessionID === "leaf" ? "root" : undefined,
        taskParentID: sessionID === "leaf" ? "root" : undefined,
        directory: sessionID === "leaf" ? "/foreign" : "/workspace",
      }),
      messages: async () => taskMessages("leaf"),
    })

    expect(await lookupSessionTreeOwnership(sdk, new Set(["root"]), "leaf")).toEqual({ type: "foreign" })
  })

  test("revalidates a previously owned descendant after it moves locations", async () => {
    let moved = false
    const sdk = client({
      get: async (sessionID) => ({
        parentID: sessionID === "leaf" ? "root" : undefined,
        taskParentID: sessionID === "leaf" ? "root" : undefined,
        directory: sessionID === "leaf" && moved ? "/foreign" : "/workspace",
      }),
    })
    const tree = new Set(["root"])

    expect(await lookupSessionTreeOwnership(sdk, tree, "leaf")).toEqual({ type: "owned", value: undefined })
    moved = true
    expect(await lookupSessionTreeOwnership(sdk, tree, "leaf")).toEqual({ type: "foreign" })
    expect(tree).toEqual(new Set(["root"]))
  })

  test("accepts a legacy parent without a persisted path", async () => {
    const sdk = client({
      get: async (sessionID) => ({
        parentID: sessionID === "leaf" ? "root" : undefined,
        taskParentID: sessionID === "leaf" ? "root" : undefined,
        path: sessionID === "leaf" ? "/workspace/project" : undefined,
      }),
    })

    expect(await lookupSessionTreeOwnership(sdk, new Set(["root"]), "leaf")).toEqual({
      type: "owned",
      value: undefined,
    })
  })

  test("gives up on an unresolvable ancestry instead of retrying forever", async () => {
    // A parent chain cycle and a deleted parent both fail permanently. Retrying
    // them forever leaves the server's permission deferred unanswered, so the
    // walk has to surface exhaustion to the caller.
    let lookups = 0
    const sdk = client({
      get: (sessionID) => {
        lookups += 1
        const parentID = sessionID === "leaf" ? "middle" : "leaf"
        return Promise.resolve({ parentID, taskParentID: parentID })
      },
      messages: async (sessionID) => (sessionID === "middle" ? taskMessages("leaf") : taskMessages("middle")),
    })

    const resolution = await resolveSessionTreeOwnership(sdk, new Set(["root"]), "leaf", {
      signal: new AbortController().signal,
      attempts: 3,
    })

    expect(resolution.type).toBe("exhausted")
    if (resolution.type === "exhausted") expect(String(resolution.error)).toContain("cycle detected")
    expect(lookups).toBeGreaterThan(0)
  })

  test("gives up when a parent points at a missing session", async () => {
    const sdk = client({
      get: (sessionID) => {
        if (sessionID === "leaf") return Promise.resolve({ parentID: "deleted", taskParentID: "deleted" })
        return Promise.reject(new Error(`session not found: ${sessionID}`))
      },
    })

    const resolution = await resolveSessionTreeOwnership(sdk, new Set(["root"]), "leaf", {
      signal: new AbortController().signal,
      attempts: 2,
    })

    expect(resolution.type).toBe("exhausted")
    if (resolution.type === "exhausted") expect(String(resolution.error)).toContain("session not found")
  })

  test("treats a missing permission request as already answered", async () => {
    // The server drops pending requests when a session aborts and rejects the
    // cascade itself, so a reply that lost that race must not fail the run.
    const sdk = client({
      reply: () =>
        Promise.reject(
          new Error("gone", {
            cause: {
              body: { _tag: "PermissionNotFoundError", requestID: "perm-1", message: "gone" },
              status: 404,
            },
          }),
        ),
    })

    await expect(
      replyPermission(sdk, {
        requestID: "perm-1",
        reply: "once",
        signal: new AbortController().signal,
      }),
    ).resolves.toBeUndefined()
  })

  test("retries the selected permission reply after a transient failure", async () => {
    let replies = 0
    const sdk = client({
      reply: () => {
        replies += 1
        if (replies === 1) return Promise.resolve({ error: new Error("reply failed") })
        return Promise.resolve({ data: true })
      },
    })

    await replyPermission(sdk, {
      requestID: "perm-1",
      reply: "once",
      signal: new AbortController().signal,
    })

    expect(replies).toBe(2)
  })

  test("stops retrying a stalled permission reply when cancelled", async () => {
    let attempts = 0
    const controller = new AbortController()
    const sdk = client({ reply: () => new Promise(() => {}) })
    const reply = replyPermission(sdk, {
      requestID: "perm-1",
      reply: "once",
      signal: controller.signal,
      timeoutMs: 20,
      onRetry: () => {
        attempts += 1
        controller.abort(new Error("stopped"))
      },
    })

    await expect(reply).rejects.toThrow("stopped")
    expect(attempts).toBe(1)
  })
})
