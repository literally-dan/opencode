export * as SessionAskEvent from "./session-ask-event"

import { Schema } from "effect"
import { Event } from "./event"
import { optional } from "./schema"
import { SessionID } from "./session-id"

export const RequestID = Schema.String.check(Schema.isMinLength(1), Schema.isMaxLength(128)).annotate({
  identifier: "SessionAskRequestID",
})
export type RequestID = typeof RequestID.Type

export const ToolActivity = Event.define({
  type: "session.ask.tool.activity",
  schema: {
    sessionID: SessionID,
    requestID: RequestID,
    threadID: Schema.String.check(Schema.isStartsWith("ask_")),
    turnID: Schema.String.check(Schema.isStartsWith("atn_")),
    callID: Schema.String,
    tool: Schema.String,
    status: Schema.Literals(["running", "completed", "error"]),
    input: Schema.String,
    output: optional(Schema.String),
    error: optional(Schema.String),
  },
})

export const Definitions = Event.inventory(ToolActivity)
