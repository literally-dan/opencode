// Permission and question routing for the GitHub agent.
//
// CI has no human to answer a request, and the agent waits until its Session is idle. An
// unanswered request from a background Task keeps the Session busy, so the job would run until
// the workflow timeout. Reject requests from the Session tree, as `opencode run` without --auto
// does. `opencode run` resolves the tree over the SDK; the agent runs in-process, so this reads
// Sessions directly.
import { Effect, Option, Schema } from "effect"
import { UI } from "../ui"
import { EventV2Bridge } from "@/event-v2-bridge"
import { Permission } from "@/permission"
import { Question } from "@/question"
import { Session } from "@/session/session"
import type { SessionID } from "@/session/schema"

export const sessionTreeRequestRejector = Effect.gen(function* () {
  const events = yield* EventV2Bridge.Service
  const sessions = yield* Session.Service
  const permission = yield* Permission.Service
  const question = yield* Question.Service

  return Effect.fn("GithubRequests.rejectSessionTreeRequests")(function* (rootID: SessionID) {
    // Requests from other trees are left alone. A request whose ancestry cannot be resolved is
    // rejected, because nothing else will answer it. Listeners run inside the publishing fiber,
    // so failures are ignored rather than failing the tool call.
    const reject = (sessionID: SessionID, message: string, answer: Effect.Effect<void, unknown>) =>
      ownership(sessions, rootID, sessionID).pipe(
        Effect.catch(() => Effect.succeed("unresolved" as const)),
        Effect.flatMap((owner) =>
          owner === "foreign"
            ? Effect.void
            : Effect.sync(() => UI.println(UI.Style.TEXT_WARNING_BOLD + "!", UI.Style.TEXT_NORMAL + message)).pipe(
                Effect.andThen(answer),
              ),
        ),
        Effect.ignore,
      )

    return yield* events.listen((event) => {
      if (event.type === Permission.Event.Asked.type) {
        const request = Schema.decodeUnknownOption(Permission.Event.Asked.fields.data)(event.data)
        if (Option.isNone(request)) return Effect.void
        return reject(
          request.value.sessionID,
          `permission requested: ${request.value.permission} (${request.value.patterns.join(", ")}); auto-rejecting`,
          permission.reply({ requestID: request.value.id, reply: "reject" }),
        )
      }

      if (event.type === Question.Event.Asked.type) {
        const request = Schema.decodeUnknownOption(Question.Event.Asked.fields.data)(event.data)
        if (Option.isNone(request)) return Effect.void
        return reject(
          request.value.sessionID,
          `question requested: ${request.value.questions.map((item) => item.question).join(" / ")}; auto-rejecting`,
          question.reject(request.value.id),
        )
      }

      return Effect.void
    })
  })
})

// Follows Task parent links the way `opencode run` does. Location is not compared, because moving
// a Session does not move its Task children.
const ownership = Effect.fnUntraced(function* (sessions: Session.Interface, rootID: SessionID, sessionID: SessionID) {
  const seen = new Set<SessionID>()
  let current = yield* sessions.get(sessionID)
  while (current.id !== rootID) {
    if (seen.has(current.id)) return "unresolved" as const
    seen.add(current.id)
    if (!current.taskParentID || current.parentID !== current.taskParentID) return "foreign" as const
    const parent = yield* sessions.get(current.taskParentID)
    if (parent.projectID !== current.projectID) return "foreign" as const
    current = parent
  }
  return "owned" as const
})
