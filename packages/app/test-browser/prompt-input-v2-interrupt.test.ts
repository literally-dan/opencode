import { describe, expect, test } from "bun:test"
import { createRoot, createSignal, type Accessor } from "solid-js"
import { createStore } from "solid-js/store"
import { createPromptInputV2Controller } from "@opencode-ai/session-ui/v2/prompt-input/interaction"
import type { PromptInputV2PersistedState } from "@opencode-ai/session-ui/v2/prompt-input/types"

function setup(input: { stopping: Accessor<boolean>; working: Accessor<boolean> }) {
  const calls: string[] = []
  const controller = createPromptInputV2Controller({
    store: createStore<PromptInputV2PersistedState>({
      prompt: [{ type: "text", content: "", start: 0, end: 0 }],
      cursor: 0,
      context: { items: [] },
    }),
    commands: () => [],
    context: () => [],
    searchContextFiles: () => [],
    view: {
      submit: {
        stopping: input.stopping,
        working: input.working,
        onSubmit: () => calls.push("submit"),
        onStop: () => calls.push("stop"),
        onInterrupt: () => calls.push("interrupt"),
      },
    },
  })
  return { calls, controller }
}

describe("prompt input v2 keyboard interrupt", () => {
  test("interrupts with Escape and ctrl-g only while working, and keeps the stop button separate", () =>
    createRoot((dispose) => {
      const [working, setWorking] = createSignal(true)
      const { calls, controller } = setup({ stopping: () => true, working })

      expect(controller.onKeyDown(new KeyboardEvent("keydown", { key: "Escape", cancelable: true }))).toBe(true)
      expect(controller.onKeyDown(new KeyboardEvent("keydown", { key: "g", ctrlKey: true, cancelable: true }))).toBe(
        true,
      )
      controller.stop()
      setWorking(false)
      expect(controller.onKeyDown(new KeyboardEvent("keydown", { key: "Escape", cancelable: true }))).toBe(false)

      expect(calls).toEqual(["interrupt", "interrupt", "stop"])
      dispose()
    }))

  test("does not submit while the submit button is a stop button, so Enter on an empty prompt stops nothing", () =>
    createRoot((dispose) => {
      const [stopping, setStopping] = createSignal(true)
      const { calls, controller } = setup({ stopping, working: () => true })

      controller.submit()
      controller.stop()
      setStopping(false)
      controller.submit()

      expect(calls).toEqual(["stop", "submit"])
      dispose()
    }))
})
