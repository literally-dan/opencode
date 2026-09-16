import type { Session } from "@opencode-ai/sdk/v2"

// Walk the in-memory session list and return the root together with every
// authenticated Task subagent under it. Pure function over a session snapshot so callers
// can wrap it in `createMemo` and have it track only what changed.
//
// Returns `[]` when `rootID` is not in `sessions` so the caller can treat an
// unknown root the same as an empty tree without a special case.
export function sessionTree(sessions: readonly Session[], rootID: string): Session[] {
  const byID = new Map(sessions.map((session) => [session.id, session]))
  const byParent = new Map<string, Session[]>()
  for (const s of sessions) {
    if (!s.parentID) continue
    const parent = byID.get(s.parentID)
    if (!parent || !isTaskChild(parent, s)) continue
    const bucket = byParent.get(s.parentID) ?? []
    bucket.push(s)
    byParent.set(s.parentID, bucket)
  }
  const root = byID.get(rootID)
  if (!root) return []
  const out: Session[] = [root]
  const seen = new Set([rootID])
  // Appending to `queue` while iterating is the BFS frontier; `for...of` over a
  // growing array visits the new entries and avoids shift()'s O(n^2).
  const queue = [rootID]
  for (const id of queue) {
    const kids = byParent.get(id)
    if (!kids) continue
    for (const kid of kids) {
      if (seen.has(kid.id)) continue
      seen.add(kid.id)
      out.push(kid)
      queue.push(kid.id)
    }
  }
  return out
}

export function sessionTreeBusy(
  sessions: readonly Session[],
  statuses: Record<string, { type: string } | undefined>,
  rootID: string,
) {
  return sessionTree(sessions, rootID).some((session) => (statuses[session.id]?.type ?? "idle") !== "idle")
}

// Aborting the root cancels Task jobs only in the root's Instance. Busy direct Task children in
// another location (for example after the root moved) must be aborted in their own Instance.
export function busyTaskChildrenElsewhere(
  sessions: readonly Session[],
  statuses: Record<string, { type: string } | undefined>,
  rootID: string,
) {
  const root = sessions.find((session) => session.id === rootID)
  if (!root) return []
  return sessions.filter(
    (session) =>
      isTaskChild(root, session) &&
      (statuses[session.id]?.type ?? "idle") !== "idle" &&
      (session.directory !== root.directory || session.workspaceID !== root.workspaceID),
  )
}

// Location is not compared: moving a Session does not move its Task children.
function isTaskChild(parent: Session, child: Session) {
  return child.parentID === parent.id && child.taskParentID === child.parentID && child.projectID === parent.projectID
}
