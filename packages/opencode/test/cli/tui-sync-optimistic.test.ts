import { describe, expect, test } from "bun:test"
import type { Message, Part } from "@opencode-ai/sdk/v2"
import { mergeFetchedMessages, optimisticParts } from "@/cli/cmd/tui/context/sync-optimistic"

function user(id: string): Message {
  return {
    id,
    sessionID: "ses_test",
    role: "user",
    time: { created: 1 },
    agent: "build",
    model: { providerID: "test", modelID: "model" },
  }
}

function text(messageID: string, text: string): Part {
  return {
    id: `part_${messageID}`,
    sessionID: "ses_test",
    messageID,
    type: "text",
    text,
  }
}

describe("TUI optimistic prompt sync", () => {
  test("keeps an optimistic message while session sync has not fetched it yet", () => {
    const merged = mergeFetchedMessages({
      currentMessages: [user("msg_2")],
      currentParts: { msg_2: [text("msg_2", "optimistic")] },
      fetched: [{ info: user("msg_1"), parts: [text("msg_1", "persisted")] }],
      optimisticMessages: new Set(["msg_2"]),
    })

    expect(merged.messages.map((message) => message.id)).toEqual(["msg_1", "msg_2"])
    expect(merged.parts.get("msg_1")?.map((part) => (part.type === "text" ? part.text : ""))).toEqual(["persisted"])
    expect(merged.resolved.has("msg_2")).toBe(false)
  })

  test("preserves optimistic parts when sync fetches the message before its parts", () => {
    const merged = mergeFetchedMessages({
      currentMessages: [user("msg_1")],
      currentParts: { msg_1: [text("msg_1", "optimistic")] },
      fetched: [{ info: user("msg_1"), parts: [] }],
      optimisticMessages: new Set(["msg_1"]),
    })

    expect(merged.messages.map((message) => message.id)).toEqual(["msg_1"])
    expect(merged.parts.get("msg_1")?.map((part) => (part.type === "text" ? part.text : ""))).toEqual(["optimistic"])
    expect(merged.resolved.has("msg_1")).toBe(false)
  })

  test("replaces optimistic parts once real fetched parts arrive", () => {
    const merged = mergeFetchedMessages({
      currentMessages: [user("msg_1")],
      currentParts: { msg_1: [text("msg_1", "optimistic")] },
      fetched: [{ info: user("msg_1"), parts: [text("msg_1", "persisted")] }],
      optimisticMessages: new Set(["msg_1"]),
    })

    expect(merged.parts.get("msg_1")?.map((part) => (part.type === "text" ? part.text : ""))).toEqual(["persisted"])
    expect(merged.resolved.has("msg_1")).toBe(true)
  })

  test("strips file URLs from optimistic render parts", () => {
    const parts = optimisticParts({
      sessionID: "ses_test",
      messageID: "msg_1",
      parts: [
        {
          id: "part_file",
          type: "file",
          mime: "image/png",
          filename: "image.png",
          url: "data:image/png;base64,large",
        },
      ],
    })

    expect(parts).toEqual([
      {
        id: "part_file",
        sessionID: "ses_test",
        messageID: "msg_1",
        type: "file",
        mime: "image/png",
        filename: "image.png",
        url: "",
      },
    ])
  })
})
