import { base64Encode } from "@opencode-ai/core/util/encode"
import type { Session } from "@opencode-ai/sdk/v2/client"
import { isTaskChild } from "@/utils/task-session"

export function acceptKey(sessionID: string, directory?: string) {
  if (!directory) return sessionID
  return `${base64Encode(directory)}/${sessionID}`
}

export function directoryAcceptKey(directory: string) {
  return `${base64Encode(directory)}/*`
}

function accepted(autoAccept: Record<string, boolean>, sessionID: string, directory?: string) {
  const key = acceptKey(sessionID, directory)
  return autoAccept[key] ?? autoAccept[sessionID]
}

export function isDirectoryAutoAccepting(autoAccept: Record<string, boolean>, directory: string) {
  const key = directoryAcceptKey(directory)
  return autoAccept[key] ?? false
}

function sessionLineage(session: Session[], sessionID: string) {
  const sessions = new Map(session.map((item) => [item.id, item]))
  const seen = new Set([sessionID])
  const ids = [sessionID]

  for (const id of ids) {
    const child = sessions.get(id)
    if (!child?.parentID || seen.has(child.parentID)) continue
    const parent = sessions.get(child.parentID)
    if (!parent || !isTaskChild(parent, child)) continue
    seen.add(parent.id)
    ids.push(parent.id)
  }

  return ids
}

export function autoRespondsPermission(
  autoAccept: Record<string, boolean>,
  session: Session[],
  permission: { sessionID: string },
  directory?: string,
) {
  const value = sessionAutoAccept(autoAccept, session, permission, directory)
  if (value !== undefined) return value
  return directory ? isDirectoryAutoAccepting(autoAccept, directory) : false
}

export function sessionAutoAccept(
  autoAccept: Record<string, boolean>,
  session: Session[],
  permission: { sessionID: string },
  directory?: string,
) {
  return sessionLineage(session, permission.sessionID)
    .map((id) => accepted(autoAccept, id, directory))
    .find((item): item is boolean => item !== undefined)
}
