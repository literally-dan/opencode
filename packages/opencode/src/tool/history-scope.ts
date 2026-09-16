import { SessionID } from "@/session/schema"
import { SessionTaskState } from "@/session/task-state"
import { Effect } from "effect"

const SESSION_ID = /^ses_[0-9a-f]{12}[0-9A-Za-z]{14}$/

export function isCanonicalSessionID(value: string) {
  return SESSION_ID.test(value)
}

// History-recovery tools (read_part, search_session_history) may read the
// current session or any of its ancestors — the documented "read the parent
// session" capability for subagents — but NOT arbitrary unrelated sessions.
// Session ids are time-ordered and guessable, so without this a model (e.g. via
// prompt-injected content) could enumerate and read history from other
// sessions/projects in the same store.
export const isReadableSession = (
  taskState: Pick<SessionTaskState.Interface, "canReadHistory">,
  requested: SessionID,
  current: SessionID,
) => {
  if (!isCanonicalSessionID(requested) || !isCanonicalSessionID(current)) return Effect.succeed(false)
  return taskState.canReadHistory({ sessionID: current, ancestorID: requested })
}
