// Shared helpers for walking a session's Task descendant tree from the SDK.
// The server only exposes a one-level `children` endpoint, so consumers
// that need the transitive set (run-mode permission routing, blocker
// bootstrap, etc.) walk the tree client-side. Centralising the BFS here
// keeps the depth invariant consistent across call sites — see
// `run.ts:loop` and `stream.transport.ts:bootstrap`.

import type { OpencodeClient, Session } from "@opencode-ai/sdk/v2"
import { SessionAncestry } from "@/session/ancestry"
import { TaskProvenance } from "@/session/task-provenance"
import { withAbortTimeout } from "@/util/timeout"

export type DescendantInfo = { id: string; title?: string }

export type SessionTreeOwnership = SessionAncestry.Ownership<undefined>
export type SessionTreeResolution = SessionAncestry.Resolution<undefined>

export type DescendantFetchFailure = {
  sessionID: string
  error: unknown
}

export type DescendantFetchOptions = {
  signal?: AbortSignal
  concurrency?: number
  timeoutMs?: number
  totalTimeoutMs?: number
  limit?: number
}

export class DescendantFetchError extends Error {
  readonly partial: DescendantInfo[]
  readonly failures: DescendantFetchFailure[]

  constructor(partial: DescendantInfo[], failures: DescendantFetchFailure[]) {
    super(`failed to fetch children for ${failures.map((item) => item.sessionID).join(", ")}`)
    this.name = "DescendantFetchError"
    this.partial = partial
    this.failures = failures
  }
}

export async function fetchDescendants(
  sdk: OpencodeClient,
  sessionID: string,
  input: DescendantFetchOptions = {},
): Promise<DescendantInfo[]> {
  const out: DescendantInfo[] = []
  const failures: DescendantFetchFailure[] = []
  const seen = new Set<string>([sessionID])
  const concurrency = positiveInteger(input.concurrency, 4)
  const timeoutMs = positiveInteger(input.timeoutMs, 10_000)
  const totalTimeoutMs = positiveInteger(input.totalTimeoutMs, 30_000)
  const limit = positiveInteger(input.limit, 1_000)
  const deadline = Date.now() + totalTimeoutMs
  const signal = input.signal ?? new AbortController().signal
  const sessions = new Map<string, Session>()
  try {
    const root = await withAbortTimeout((signal) => TaskProvenance.getSession(sdk, sessionID, signal), {
      signal,
      timeoutMs: Math.min(timeoutMs, totalTimeoutMs),
      label: `session lookup timed out for ${sessionID}`,
    })
    sessions.set(root.id, root)
  } catch (error) {
    if (input.signal?.aborted) throw input.signal.reason
    throw new DescendantFetchError([], [{ sessionID, error }])
  }
  let limited = false
  let expired = false
  let frontier = [sessionID]
  while (frontier.length) {
    input.signal?.throwIfAborted()
    const current = frontier
    const next: string[] = []
    for (let offset = 0; offset < current.length; offset += concurrency) {
      input.signal?.throwIfAborted()
      const group = current.slice(offset, offset + concurrency)
      const remaining = deadline - Date.now()
      if (remaining <= 0) {
        failures.push({ sessionID: group[0]!, error: new Error(`session tree lookup exceeded ${totalTimeoutMs}ms`) })
        expired = true
        break
      }
      const batches = await Promise.allSettled(
        group.map(async (id) => {
          const timeout = new AbortController()
          const timer = setTimeout(
            () => timeout.abort(new Error(`children lookup timed out for ${id}`)),
            Math.min(timeoutMs, remaining),
          )
          const signal = input.signal ? AbortSignal.any([input.signal, timeout.signal]) : timeout.signal
          try {
            const result = await sdk.session.children({ sessionID: id }, { throwOnError: true, signal })
            if (!result.data) throw new Error(`children lookup returned no data for ${id}`)
            return result.data
          } finally {
            clearTimeout(timer)
          }
        }),
      )
      input.signal?.throwIfAborted()
      for (const [index, batch] of batches.entries()) {
        if (batch.status === "rejected") {
          failures.push({ sessionID: group[index]!, error: batch.reason })
          continue
        }
        const parent = sessions.get(group[index]!)
        if (!parent) continue
        for (const child of batch.value) {
          if (seen.has(child.id)) continue
          if (!isTaskChild(parent, child)) continue
          if (seen.size >= limit) {
            if (!limited) {
              failures.push({ sessionID: child.id, error: new Error(`session tree exceeds ${limit} entries`) })
              limited = true
            }
            continue
          }
          seen.add(child.id)
          sessions.set(child.id, child)
          out.push(child.title === undefined ? { id: child.id } : { id: child.id, title: child.title })
          next.push(child.id)
        }
      }
    }
    if (expired) break
    frontier = next
  }
  if (failures.length) throw new DescendantFetchError(out, failures)
  return out
}

function positiveInteger(value: number | undefined, fallback: number) {
  if (value === undefined || !Number.isFinite(value) || value <= 0) return fallback
  return Math.floor(value)
}

function isTaskChild(parent: Session, child: Session) {
  return (
    child.parentID === parent.id &&
    child.taskParentID === child.parentID &&
    child.projectID === parent.projectID &&
    child.workspaceID === parent.workspaceID &&
    child.directory === parent.directory &&
    (child.path === undefined || parent.path === undefined || child.path === parent.path)
  )
}

// Like `fetchDescendants` but returns just the id set including the root.
// Useful when the caller wants membership tests rather than the full info.
export async function fetchDescendantIDs(
  sdk: OpencodeClient,
  sessionID: string,
  input: DescendantFetchOptions = {},
): Promise<Set<string>> {
  const ids = new Set<string>([sessionID])
  for (const child of await fetchDescendants(sdk, sessionID, input)) {
    ids.add(child.id)
  }
  return ids
}

export function resolveSessionTreeOwnership(
  sdk: OpencodeClient,
  tree: Set<string>,
  sessionID: string,
  input: {
    signal: AbortSignal
    onUnknown?: (input: { error: unknown; attempt: number; retryIn: number }) => void
    attempts?: number
    timeoutMs?: number
  },
): Promise<SessionTreeResolution> {
  return SessionAncestry.resolve({
    signal: input.signal,
    onRetry: input.onUnknown,
    attempts: input.attempts,
    timeoutMs: input.timeoutMs,
    lookup: (signal) => lookupSessionTreeOwnership(sdk, tree, sessionID, signal),
  })
}

export async function lookupSessionTreeOwnership(
  sdk: OpencodeClient,
  tree: Set<string>,
  sessionID: string,
  signal?: AbortSignal,
): Promise<SessionTreeOwnership> {
  try {
    const owned = await hasAncestryInTree(sdk, tree, sessionID, signal)
    return owned ? { type: "owned", value: undefined } : { type: "foreign" }
  } catch (error) {
    return { type: "unknown", error }
  }
}

async function hasAncestryInTree(sdk: OpencodeClient, tree: Set<string>, sessionID: string, signal?: AbortSignal) {
  if (tree.has(sessionID)) return true

  const chain: string[] = []
  const seen = new Set<string>()
  let current = await TaskProvenance.getSession(sdk, sessionID, signal)
  while (true) {
    if (seen.has(current.id)) {
      throw new Error(`parent chain cycle detected for ${sessionID}: ${[...chain, current.id].join(" -> ")}`)
    }
    seen.add(current.id)
    chain.push(current.id)
    if (tree.has(current.id)) return true

    const parent = await TaskProvenance.parent(sdk, current, signal)
    if (!parent) return false
    current = parent
  }
}
