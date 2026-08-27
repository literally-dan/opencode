import { ConfigV1 } from "@opencode-ai/core/v1/config/config"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import type { Provider } from "@/provider/provider"
import { ProviderTransform } from "@/provider/transform"
import { Token } from "@/util/token"

const COMPACTION_BUFFER = 20_000
export const REQUEST_REMINDER_HEADROOM = Token.estimate("x".repeat(3_000))

export function estimateRequest(input: {
  system: string[]
  messages: unknown[]
  tools?: Record<string, { description?: string; inputSchema?: unknown }>
  reminderHeadroom?: number
}) {
  const tools = Object.entries(input.tools ?? {}).map(([name, tool]) => ({
    name,
    description: tool.description,
    inputSchema: tool.inputSchema,
  }))
  return (
    Token.estimate(JSON.stringify({ system: input.system, messages: input.messages, tools })) -
    Math.round(imageExcess(input.messages, undefined, 0) / 4) +
    (input.reminderHeadroom ?? 0)
  )
}

// A provider bills an image by its dimensions, not by its bytes. Counted as text, the base64 of one screenshot reads as
// tens of thousands of tokens, which raises a false context-pressure reminder. Count about 2,000 tokens for an image.
const IMAGE_ESTIMATE_CHARS = 8_000

// Characters of image data above the allowance. A walk is cheaper here than a JSON.stringify replacer, which is
// called for every property of every message.
function imageExcess(value: unknown, holder: unknown, depth: number): number {
  if (typeof value === "string") return isImageData(holder, value) ? value.length - IMAGE_ESTIMATE_CHARS : 0
  if (typeof value !== "object" || value === null || depth > 8) return 0
  if (Array.isArray(value)) return value.reduce((sum: number, item) => sum + imageExcess(item, value, depth + 1), 0)
  return Object.values(value).reduce((sum: number, item) => sum + imageExcess(item, value, depth + 1), 0)
}

function isImageData(holder: unknown, value: string) {
  if (value.length <= IMAGE_ESTIMATE_CHARS) return false
  if (value.startsWith("data:image/")) return true
  if (typeof holder !== "object" || holder === null) return false
  if ("type" in holder && holder.type === "image") return true
  if ("mediaType" in holder && typeof holder.mediaType === "string") return holder.mediaType.startsWith("image/")
  return "mime" in holder && typeof holder.mime === "string" && holder.mime.startsWith("image/")
}

export type Budget =
  | { status: "available"; tokens: number; limit: number; reserved: number }
  | { status: "unbounded"; tokens: 0; limit: 0; reserved: 0 }
  | { status: "impossible"; tokens: 0; limit: number; reserved: number; message: string }

export type Result =
  | { status: "disabled"; overflow: false; count: number; budget: Budget }
  | { status: "unbounded"; overflow: false; count: number; budget: Budget }
  | { status: "impossible"; overflow: false; count: number; budget: Budget & { status: "impossible" } }
  | { status: "available" | "overflow"; overflow: boolean; count: number; budget: Budget & { status: "available" } }

const valid = (value: number | undefined) => (value !== undefined && Number.isFinite(value) && value > 0 ? value : 0)

export function budget(input: { cfg: ConfigV1.Info; model: Provider.Model; outputTokenMax?: number }): Budget {
  const context = valid(input.model.limit.context)
  if (context === 0) return { status: "unbounded", tokens: 0, limit: 0, reserved: 0 }

  const inputLimit = valid(input.model.limit.input)
  const limit = inputLimit || context
  const output = valid(ProviderTransform.maxOutputTokens(input.model, input.outputTokenMax))
  const reserved = Math.max(
    0,
    input.cfg.compaction?.reserved ?? (inputLimit ? Math.min(COMPACTION_BUFFER, output) : output),
  )
  if (reserved >= limit) {
    return {
      status: "impossible",
      tokens: 0,
      limit,
      reserved,
      message: `Compaction cannot run because the reserved token budget (${reserved}) leaves no usable input space for this model (${limit}). Lower compaction.reserved or choose a model with a larger input limit.`,
    }
  }
  return { status: "available", tokens: limit - reserved, limit, reserved }
}

export function usable(input: { cfg: ConfigV1.Info; model: Provider.Model; outputTokenMax?: number }) {
  const result = budget(input)
  return result.status === "available" ? result.tokens : 0
}

export function check(input: {
  cfg: ConfigV1.Info
  tokens: SessionV1.Assistant["tokens"]
  model: Provider.Model
  outputTokenMax?: number
}): Result {
  const component =
    valid(input.tokens.input) +
    valid(input.tokens.output) +
    valid(input.tokens.reasoning) +
    valid(input.tokens.cache.read) +
    valid(input.tokens.cache.write)
  const count = Math.max(valid(input.tokens.total), component)
  const available = budget(input)
  if (input.cfg.compaction?.auto === false) return { status: "disabled", overflow: false, count, budget: available }
  if (available.status === "impossible") {
    return { status: "impossible", overflow: false, count, budget: available }
  }
  if (available.status === "unbounded") return { status: "unbounded", overflow: false, count, budget: available }
  if (count >= available.tokens) return { status: "overflow", overflow: true, count, budget: available }
  return { status: "available", overflow: false, count, budget: available }
}

export function isOverflow(input: Parameters<typeof check>[0]) {
  return check(input).overflow
}
