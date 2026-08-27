import * as Tool from "./tool"
import DESCRIPTION from "./search_session_history.txt"
import { Session } from "@/session/session"
import { SessionID } from "../session/schema"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { isCanonicalSessionID, isReadableSession } from "./history-scope"
import { Effect, Schema } from "effect"

const id = "search_session_history"
const DEFAULT_LIMIT = 20
const MAX_LIMIT = 50
const MAX_PATTERN_CHARS = 256
const MAX_TOOL_CHARS = 128
const MAX_PART_SEARCH_CHARS = 256_000
const MAX_TOTAL_SEARCH_CHARS = 2_000_000
const SNIPPET_RADIUS = 80

export const Parameters = Schema.Struct({
  pattern: Schema.String.annotate({
    description: `Case-insensitive literal text to search for (1-${MAX_PATTERN_CHARS} characters; regex syntax is not interpreted).`,
  }),
  session_id: Schema.optional(Schema.String).annotate({
    description: "Optional session ID; defaults to the current session.",
  }),
  tool: Schema.optional(Schema.String).annotate({
    description: "Optional tool name filter (e.g. `bash`, `read`).",
  }),
  limit: Schema.optional(Schema.Number).annotate({
    description: `Maximum matches to return (default ${DEFAULT_LIMIT}, maximum ${MAX_LIMIT}).`,
  }),
})

type Match = {
  part_id: string
  message_id: string
  part_type: string
  tool?: string
  snippet: string
  created_at?: number
}

type Metadata = {
  sessionID: string
  pattern: string
  matched: number
  truncated: boolean
}

export const SearchSessionHistoryTool = Tool.define<typeof Parameters, Metadata, Session.Service>(
  id,
  Effect.gen(function* () {
    const sessions = yield* Session.Service

    return {
      description: DESCRIPTION,
      parameters: Parameters,
      execute: (params: Schema.Schema.Type<typeof Parameters>, ctx: Tool.Context<Metadata>) =>
        Effect.gen(function* () {
          const invalid = validateInput(params)
          if (invalid) {
            return {
              title: "Invalid history search",
              metadata: {
                sessionID: ctx.sessionID,
                pattern: params.pattern.slice(0, MAX_PATTERN_CHARS),
                matched: 0,
                truncated: false,
              },
              output: invalid,
            }
          }
          const sessionID = params.session_id ? SessionID.make(params.session_id) : ctx.sessionID
          if (params.session_id && !(yield* isReadableSession(sessions, sessionID, ctx.sessionID))) {
            return {
              title: `Session not accessible: ${sessionID}`,
              metadata: { sessionID, pattern: params.pattern, matched: 0, truncated: false },
              output: `Session ${sessionID} is not the current session or one of its ancestors; cross-session search is not allowed.`,
            }
          }
          const limit = params.limit ?? DEFAULT_LIMIT
          const messages = yield* sessions
            .messages({ sessionID })
            .pipe(Effect.catch(() => Effect.succeed([] as SessionV1.WithParts[])))
          const matches = collectMatches(messages, params.pattern, params.tool, limit)
          return formatResult(matches, sessionID, params.pattern, limit)
        }).pipe(Effect.orDie),
    }
  }),
)

function collectMatches(
  messages: SessionV1.WithParts[],
  pattern: string,
  toolFilter: string | undefined,
  limit: number,
): { matches: Match[]; truncated: boolean } {
  const matches: Match[] = []
  const query = pattern.toLowerCase()
  let searched = 0
  let truncated = false
  for (let mi = messages.length - 1; mi >= 0; mi--) {
    const msg = messages[mi]
    for (let pi = msg.parts.length - 1; pi >= 0; pi--) {
      const part = msg.parts[pi]
      const content = extractSearchableContent(part)
      if (!content) continue
      if (toolFilter && (part.type !== "tool" || part.tool !== toolFilter)) continue
      const remaining = MAX_TOTAL_SEARCH_CHARS - searched
      if (remaining <= 0) return { matches, truncated: true }
      const length = Math.min(content.length, MAX_PART_SEARCH_CHARS, remaining)
      const idx = content.slice(0, length).toLowerCase().indexOf(query)
      searched += length
      if (length < content.length) truncated = true
      if (idx < 0) continue
      matches.push({
        part_id: part.id,
        message_id: msg.info.id,
        part_type: part.type,
        tool: part.type === "tool" ? part.tool : undefined,
        snippet: makeSnippet(content, idx, pattern.length),
        created_at: msg.info.time?.created,
      })
      if (matches.length >= limit) {
        return { matches, truncated: true }
      }
    }
  }
  return { matches, truncated }
}

function extractSearchableContent(part: SessionV1.Part): string | undefined {
  if (part.type === "text" || part.type === "reasoning") return part.text
  if (part.type === "tool") {
    const state = part.state
    if (state.status === "completed") return state.output
    if (state.status === "error") return state.error
    return undefined
  }
  return undefined
}

function makeSnippet(content: string, idx: number, matchLen: number): string {
  const start = Math.max(0, idx - SNIPPET_RADIUS)
  const end = Math.min(content.length, idx + matchLen + SNIPPET_RADIUS)
  const prefix = start > 0 ? "…" : ""
  const suffix = end < content.length ? "…" : ""
  return prefix + content.slice(start, end).replace(/\s+/g, " ").trim() + suffix
}

function formatResult(
  result: { matches: Match[]; truncated: boolean },
  sessionID: string,
  pattern: string,
  limit: number,
) {
  const metadata = {
    sessionID,
    pattern,
    matched: result.matches.length,
    truncated: result.truncated,
  }
  if (result.matches.length === 0) {
    return {
      title: "No history matches",
      metadata,
      output: `No literal matches for ${JSON.stringify(pattern)} in session ${sessionID}.${result.truncated ? " Search bounds were reached." : ""}`,
    }
  }
  const lines: string[] = [
    `${result.matches.length}${result.truncated ? "+" : ""} match${
      result.matches.length === 1 ? "" : "es"
    } for literal ${JSON.stringify(pattern)} in session ${sessionID} (newest first${
      result.truncated ? `; truncated by the ${limit}-result or scan bound` : ""
    }):`,
    "",
  ]
  for (const m of result.matches) {
    const head = m.tool ? `${m.part_type}:${m.tool.slice(0, MAX_TOOL_CHARS)}` : m.part_type
    const ts = m.created_at ? ` @ ${new Date(m.created_at).toISOString()}` : ""
    lines.push(`- ${m.part_id.slice(0, 80)} (${head}, msg ${m.message_id.slice(0, 80)}${ts})`)
    lines.push(`  ${m.snippet}`)
  }
  return {
    title: `${result.matches.length}${result.truncated ? "+" : ""} match${result.matches.length === 1 ? "" : "es"}`,
    metadata,
    output: lines.join("\n"),
  }
}

function validateInput(params: Schema.Schema.Type<typeof Parameters>) {
  if (params.pattern.length === 0 || params.pattern.length > MAX_PATTERN_CHARS)
    return `pattern must contain 1-${MAX_PATTERN_CHARS} characters.`
  if (params.session_id && !isCanonicalSessionID(params.session_id)) return "session_id must be a canonical `ses_…` id."
  if (params.tool !== undefined && (params.tool.length === 0 || params.tool.length > MAX_TOOL_CHARS))
    return `tool must contain 1-${MAX_TOOL_CHARS} characters.`
  if (
    params.limit !== undefined &&
    (!Number.isSafeInteger(params.limit) || params.limit < 1 || params.limit > MAX_LIMIT)
  )
    return `limit must be an integer from 1 to ${MAX_LIMIT}.`
  return undefined
}
