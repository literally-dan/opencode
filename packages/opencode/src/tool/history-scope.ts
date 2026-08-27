import { Session } from "@/session/session"
import { SessionID } from "@/session/schema"
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
export const isReadableSession = (sessions: Pick<Session.Interface, "get">, requested: SessionID, current: SessionID) =>
  Effect.gen(function* () {
    if (!isCanonicalSessionID(requested) || !isCanonicalSessionID(current)) return false
    if (requested === current) return true
    let cursor: SessionID | undefined = current
    const seen = new Set<SessionID>()
    // Walk the parent chain (bounded against cycles); allow any ancestor.
    for (let i = 0; i < 64 && cursor && !seen.has(cursor); i++) {
      seen.add(cursor)
      const info: { parentID?: SessionID } | undefined = yield* sessions
        .get(cursor)
        .pipe(Effect.catch(() => Effect.succeed(undefined)))
      const parent: SessionID | undefined = info?.parentID
      if (!parent || !isCanonicalSessionID(parent)) break
      if (parent === requested) return true
      cursor = parent
    }
    return false
  })
