import { beforeAll, describe, expect, mock, test } from "bun:test"
import type { Message } from "@opencode-ai/sdk/v2/client"
import { createRoot, createSignal } from "solid-js"
import type { useLocal } from "@/context/local"
import { createPromptState } from "@/context/prompt"

let usePromptInputV2Controller: typeof import("@/components/prompt-input-v2").usePromptInputV2Controller

const aborts: Array<"turn" | undefined> = []
const submits: Event[] = []
const [messages, setMessages] = createSignal<Message[]>([])
const user = { id: "msg_1", sessionID: "ses_1", role: "user", time: { created: 1 } } as Message
const assistant = { id: "msg_2", sessionID: "ses_1", role: "assistant", time: { created: 2 } } as Message

beforeAll(async () => {
  // Keep every real export, so modules imported by later test files still link.
  const [sdk, sync, file, layout, comments, dialog, command, permission, language, platform, submit] =
    await Promise.all([
      import("@/context/sdk"),
      import("@/context/sync"),
      import("@/context/file"),
      import("@/context/layout"),
      import("@/context/comments"),
      import("@opencode-ai/ui/context/dialog"),
      import("@/context/command"),
      import("@/context/permission"),
      import("@/context/language"),
      import("@/context/platform"),
      import("@/components/prompt-input/submit"),
    ])
  mock.module("@/context/sdk", () => ({ ...sdk, useSDK: () => () => ({ directory: "/repo" }) }))
  mock.module("@/context/sync", () => ({
    ...sync,
    useSync: () => () => ({
      data: {
        get message() {
          return { ses_1: messages() }
        },
        reference: [],
        mcp_resource: {},
        command: [],
        session_working: () => true,
      },
      session: { get: () => undefined },
    }),
  }))
  mock.module("@/context/file", () => ({ ...file, useFile: () => ({ pathFromTab: () => undefined }) }))
  mock.module("@/context/layout", () => ({ ...layout, useLayout: () => ({}) }))
  mock.module("@/context/comments", () => ({ ...comments, useComments: () => ({}) }))
  mock.module("@opencode-ai/ui/context/dialog", () => ({ ...dialog, useDialog: () => ({}) }))
  mock.module("@/context/command", () => ({
    ...command,
    useCommand: () => ({ options: [], register: () => undefined, keybindParts: () => [] }),
  }))
  mock.module("@/context/permission", () => ({
    ...permission,
    usePermission: () => ({ isAutoAccepting: () => false }),
  }))
  mock.module("@/context/language", () => ({ ...language, useLanguage: () => ({ t: (key: string) => key }) }))
  mock.module("@/context/platform", () => ({ ...platform, usePlatform: () => ({}) }))
  mock.module("@/components/prompt-input/submit", () => ({
    ...submit,
    createPromptSubmit: () => ({
      abort: async (scope?: "turn") => {
        aborts.push(scope)
      },
      handleSubmit: async (event: Event) => {
        submits.push(event)
      },
    }),
  }))
  usePromptInputV2Controller = (await import("@/components/prompt-input-v2")).usePromptInputV2Controller
})

describe("prompt input v2 controller", () => {
  test("stops only a running turn with Escape, ignores empty Enter, and stops everything with the button", () =>
    createRoot((dispose) => {
      const controller = usePromptInputV2Controller({
        state: createPromptState(),
        history: { entries: () => [], add: () => undefined },
        controls: {
          agents: {
            available: [],
            options: [],
            current: "build",
            loading: false,
            visible: false,
            select: () => undefined,
          },
          model: {
            selection: { variant: { list: () => [] } } as unknown as ReturnType<typeof useLocal>["model"],
            paid: true,
            loading: false,
          },
          session: {
            id: "ses_1",
            tabs: { active: () => undefined, all: () => [], open: () => undefined, setActive: () => undefined },
            reviewPanel: { opened: () => false, open: () => undefined },
          },
        },
      })
      const escape = () => controller.onKeyDown(new KeyboardEvent("keydown", { key: "Escape", cancelable: true }))

      setMessages([user, assistant])
      expect(escape()).toBe(true)
      setMessages([user, { ...assistant, time: { created: 2, completed: 3 } } as Message])
      expect(escape()).toBe(false)
      controller.submit()
      controller.stop()

      expect(aborts).toEqual(["turn", undefined])
      expect(submits).toEqual([])
      dispose()
    }))
})
