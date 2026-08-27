import { describe, expect, test } from "bun:test"
import { Schema } from "effect"
import { SessionV1 } from "../src/session-v1"

describe("history compaction schema", () => {
  test("round-trips flat manual compaction metadata on every eligible part kind", () => {
    const group = "compaction-group"
    const text = Schema.decodeUnknownSync(SessionV1.TextPart)({
      id: "prt_text",
      sessionID: "ses_test",
      messageID: "msg_test",
      type: "text",
      text: "settled note",
      compacted: 1,
      compactionGroup: group,
      compactionSummary: "shared summary",
    })
    const reasoning = Schema.decodeUnknownSync(SessionV1.ReasoningPart)({
      id: "prt_reasoning",
      sessionID: "ses_test",
      messageID: "msg_test",
      type: "reasoning",
      text: "provider-independent reasoning",
      time: { start: 1, end: 2 },
      compacted: 1,
      compactionGroup: group,
      compactionSummary: "shared summary",
    })
    const tool = Schema.decodeUnknownSync(SessionV1.ToolStateCompleted)({
      status: "completed",
      input: {},
      output: "large output",
      title: "done",
      metadata: {},
      time: { start: 1, end: 2 },
      compactionGroup: group,
      compactionSummary: "shared summary",
    })

    expect(text.compactionGroup).toBe(group)
    expect(reasoning.compactionGroup).toBe(group)
    expect(tool.compactionGroup).toBe(group)
    expect(Schema.encodeSync(SessionV1.ToolStateCompleted)(tool)).toMatchObject({
      compactionGroup: group,
      compactionSummary: "shared summary",
    })
  })
})
