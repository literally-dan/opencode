import type { AgentPartInput, FilePartInput, Message, Part, SubtaskPartInput, TextPartInput } from "@opencode-ai/sdk/v2"
import { Binary } from "@opencode-ai/core/util/binary"

export type OptimisticPromptPart = (TextPartInput | FilePartInput | AgentPartInput | SubtaskPartInput) & { id: string }

export function optimisticParts(input: { sessionID: string; messageID: string; parts: OptimisticPromptPart[] }) {
  return input.parts.map((part): Part => {
    const withIDs = {
      ...part,
      sessionID: input.sessionID,
      messageID: input.messageID,
    }
    if (withIDs.type === "file") return { ...withIDs, url: "" }
    return withIDs
  })
}

export function mergeFetchedMessages(input: {
  currentMessages: Message[]
  currentParts: Record<string, Part[] | undefined>
  fetched: { info: Message; parts: Part[] }[]
  optimisticMessages: ReadonlySet<string>
}) {
  const fetchedIDs = new Set(input.fetched.map((message) => message.info.id))
  const messages = input.fetched.map((message) => message.info)
  const parts = new Map<string, Part[]>()
  const resolved = new Set<string>()

  for (const message of input.currentMessages) {
    if (input.optimisticMessages.has(message.id) && !fetchedIDs.has(message.id)) {
      Binary.insert(messages, message, (item) => item.id)
    }
  }

  for (const message of input.fetched) {
    if (message.parts.length > 0) {
      resolved.add(message.info.id)
      parts.set(message.info.id, message.parts)
      continue
    }
    if (input.optimisticMessages.has(message.info.id)) {
      const current = input.currentParts[message.info.id]
      if (current) parts.set(message.info.id, current)
      continue
    }
    parts.set(message.info.id, message.parts)
  }

  return { messages, parts, resolved }
}
