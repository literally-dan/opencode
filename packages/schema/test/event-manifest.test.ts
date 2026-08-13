import { describe, expect, test } from "bun:test"
import { Schema } from "effect"
import { FileSystem, Integration, Permission, Project, Reference, Session, Workspace } from "../src"
import { EventManifest } from "../src/event-manifest"
import { IdeEvent } from "../src/ide-event"
import { SessionAskEvent } from "../src/session-ask-event"
import { SessionEvent } from "../src/session-event"
import { SessionID } from "../src/session-id"
import { SessionTodo } from "../src/session-todo"
import { SessionV1 } from "../src/session-v1"
import { WorkspaceEvent } from "../src/workspace-event"

describe("public event manifest", () => {
  test("owns the complete public event surface", () => {
    expect(EventManifest.ServerDefinitions.length).toBe(58)
    expect(EventManifest.Definitions.length).toBe(89)
    expect(SessionV1.Event.Definitions).toEqual([
      SessionV1.Event.Created,
      SessionV1.Event.Updated,
      SessionV1.Event.Deleted,
      SessionV1.Event.MessageUpdated,
      SessionV1.Event.MessageRemoved,
      SessionV1.Event.PartUpdated,
      SessionV1.Event.PartRemoved,
      SessionV1.Event.PartDelta,
      SessionV1.Event.Diff,
      SessionV1.Event.Error,
    ])
    expect(EventManifest.Latest.size).toBe(89)
    expect(EventManifest.Durable.size).toBe(35)
  })

  test("uses canonical definitions for current public events", () => {
    expect(Session.Event).toBe(SessionEvent)
    expect(Session.Event.Definitions).toBe(SessionEvent.Definitions)
    expect(Workspace.Event).toBe(WorkspaceEvent)
    expect(Workspace.Event.Definitions).toBe(WorkspaceEvent.Definitions)
    expect(EventManifest.Latest.get("session.next.step.ended")).toBe(SessionEvent.Step.Ended)
    expect(EventManifest.Latest.get("todo.updated")).toBe(SessionTodo.Event.Updated)
    expect(EventManifest.Latest.get("project.updated")).toBe(Project.Event.Updated)
    expect(Project.Event.Definitions).toEqual([Project.Event.Updated])
    expect(FileSystem.Event.Definitions).toEqual([FileSystem.Event.Edited])
    expect(Integration.Event.Definitions).toEqual([Integration.Event.Updated, Integration.Event.ConnectionUpdated])
    expect(Permission.Event.Definitions).toEqual([Permission.Event.Asked, Permission.Event.Replied])
    expect(Reference.Event.Definitions).toEqual([Reference.Event.Updated])
    expect(EventManifest.Latest.has("ide.installed")).toBe(false)
    expect(IdeEvent.Definitions).toEqual([IdeEvent.Installed])
    expect(EventManifest.Definitions.slice(43, 46)).toEqual([
      SessionV1.Event.PartDelta,
      SessionV1.Event.Diff,
      SessionV1.Event.Error,
    ])
    expect(EventManifest.Durable.has("session.next.step.ended.1")).toBe(false)
    expect(EventManifest.Durable.get("session.next.step.ended.2")).toBe(SessionEvent.Step.Ended)
  })

  test("includes Ask tool activity in the legacy event surface", () => {
    const decode = Schema.decodeUnknownSync(SessionAskEvent.ToolActivity.data)
    const withoutRequestID = {
      sessionID: "ses_test",
      threadID: "ask_test",
      turnID: "atn_test",
      callID: "call_test",
      tool: "read",
      status: "running" as const,
      input: "{}",
    }
    const activity = { ...withoutRequestID, requestID: "client-request" }
    expect(EventManifest.Definitions).toContain(SessionAskEvent.ToolActivity)
    expect(EventManifest.Latest.get("session.ask.tool.activity")).toBe(SessionAskEvent.ToolActivity)
    expect(
      [...EventManifest.Durable.keys()].some(
        (key) => key === SessionAskEvent.ToolActivity.type || key.startsWith(`${SessionAskEvent.ToolActivity.type}.`),
      ),
    ).toBe(false)
    expect(
      new Set<string>(EventManifest.ServerDefinitions.map((definition) => definition.type)).has(
        SessionAskEvent.ToolActivity.type,
      ),
    ).toBe(false)
    expect(decode(activity)).toEqual({
      sessionID: SessionID.make("ses_test"),
      requestID: "client-request",
      threadID: "ask_test",
      turnID: "atn_test",
      callID: "call_test",
      tool: "read",
      status: "running",
      input: "{}",
    })
    expect(() => decode({ ...activity, requestID: "" })).toThrow()
    expect(() => decode({ ...activity, requestID: "x".repeat(129) })).toThrow()
    expect(() => decode(withoutRequestID)).toThrow()
  })
})
