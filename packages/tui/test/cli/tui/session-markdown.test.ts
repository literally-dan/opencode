import { describe, expect, test } from "bun:test"
import { isMarkdownPartComplete } from "../../../src/routes/session"

describe("isMarkdownPartComplete", () => {
  test("keeps an active part streaming", () => {
    expect(isMarkdownPartComplete({ time: { start: 1 } }, { time: { created: 1 } })).toBe(false)
  })

  test("uses explicit part completion even when the timestamp is zero", () => {
    expect(isMarkdownPartComplete({ time: { start: 0, end: 0 } }, { time: { created: 0 } })).toBe(true)
  })

  test("treats historical text without part timing as complete with its message", () => {
    expect(isMarkdownPartComplete({}, { time: { created: 1, completed: 2 } })).toBe(true)
  })
})
