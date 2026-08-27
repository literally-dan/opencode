import { SessionID, MessageID } from "./schema"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { ProviderV2 } from "@opencode-ai/core/provider"
import {
  APIError,
  AbortedError,
  Assistant,
  AuthError,
  CompactionPart,
  ContentFilterError,
  ContextOverflowError,
  Info,
  OutputLengthError,
  Part,
  SubtaskPart,
  User,
  WithParts,
} from "@opencode-ai/core/v1/session"

import { NamedError } from "@opencode-ai/core/util/error"
import {
  APICallError,
  convertToModelMessages,
  LoadAPIKeyError,
  RetryError,
  type ModelMessage,
  type UIMessage,
} from "ai"
import { Database } from "@opencode-ai/core/database/database"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { NotFoundError } from "@/storage/storage"
import { and } from "drizzle-orm"
import { desc } from "drizzle-orm"
import { eq } from "drizzle-orm"
import { inArray } from "drizzle-orm"
import { lt } from "drizzle-orm"
import { or } from "drizzle-orm"
import { MessageTable, PartTable, SessionTable } from "@opencode-ai/core/session/sql"
import { ProviderError } from "@/provider/error"
import { iife } from "@/util/iife"
import { errorMessage } from "@/util/error"
import { isMedia } from "@/util/media"
import { compactionOf, projectionChannel, type ProjectionChannel } from "./compaction-pruning"
import type { SystemError } from "bun"
import type { Provider } from "@/provider/provider"
import { Effect, Schema } from "effect"

/** Error shape thrown by Bun's fetch() when gzip/br decompression fails mid-stream */
interface FetchDecompressionError extends Error {
  code: "ZlibError"
  errno: number
  path: string
}

export const SYNTHETIC_ATTACHMENT_PROMPT = "Attached media from tool result:"
export const CONTENT_FILTER_NOTE =
  "[This turn was blocked by the provider's safety classifier and produced no usable output. Do not repeat it — take a different approach, or explain to the user why you cannot proceed.]"
export { isMedia }

function truncateToolOutput(text: string, maxChars?: number) {
  if (!maxChars || text.length <= maxChars) return text
  const omitted = text.length - maxChars
  return `${text.slice(0, maxChars)}\n[Tool output truncated for compaction: omitted ${omitted} chars]`
}

const COMPACTION_SUMMARY_MAX_CHARS = 4_000
// Explicit groups are partitioned by the channel they project into. This keeps
// user instructions, assistant content, and tool results in their original
// authority channels while paying for the full summary only once per channel.
const escapeAttribute = (value: string) =>
  value.replaceAll("&", "&amp;").replaceAll('"', "&quot;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")

const compactedMarker = (partID: string, group?: string) =>
  `<compacted prt="${partID}"${group ? ` group="${escapeAttribute(group)}"` : ""}/>`

function nonblankSummary(current?: string, candidate?: string) {
  if (current?.trim()) return current
  if (candidate?.trim()) return candidate
  return undefined
}

type CompactionRun = {
  channel: ProjectionChannel
  group?: string
  ids: string[]
  summary?: string
  native?: boolean
  // Index of the last tool part in the run, which carries the covering summary
  // inside its tool result. Undefined when the run holds no tool parts.
  lastTool?: number
  end: number
}

// Group a message's parts into maximal contiguous runs sharing one compaction
// group, so a batch compaction renders one covering note instead of repeating
// the same summary once per part. Runs never span messages: each provider
// message has to stay self-describing, otherwise slicing history could leave
// compacted content with no note covering it.
function compactionRuns(parts: SessionV1.Part[]) {
  const byIndex = new Map<number, CompactionRun>()
  let current: { key: string; run: CompactionRun } | undefined
  for (const [index, part] of parts.entries()) {
    const compaction = renderedCompaction(part)
    if (!compaction) {
      current = undefined
      continue
    }
    const channel = projectionChannel(part, "assistant")
    // Native age-based pruning carries no group; consecutive pruned parts are
    // contiguous by construction, so they share one run and one note.
    const key = JSON.stringify([channel, compaction.group ?? (compaction.native ? "native" : part.id)])
    if (current?.key !== key)
      current = {
        key,
        run: { channel, group: compaction.group, ids: [], end: index, native: compaction.native },
      }
    current.run.summary = nonblankSummary(current.run.summary, compaction.summary)
    current.run.end = index
    if (part.type === "tool") current.run.lastTool = index
    current.run.ids.push(part.id)
    byIndex.set(index, current.run)
  }
  return byIndex
}

function compactionGroups(messages: SessionV1.WithParts[]) {
  const groups = new Map<
    string,
    Map<ProjectionChannel, { ids: string[]; summary?: string; carrier: { messageIndex: number; partIndex: number } }>
  >()
  for (const [messageIndex, message] of messages.entries()) {
    if (!rendersStoredParts(message)) continue
    for (const [partIndex, part] of message.parts.entries()) {
      const compaction = renderedCompaction(part)
      if (!compaction?.group) continue
      const channel: ProjectionChannel = projectionChannel(part, message.info.role)
      const group = groups.get(compaction.group) ?? new Map()
      const slice = group.get(channel) ?? {
        ids: [],
        summary: nonblankSummary(undefined, compaction.summary),
        carrier: { messageIndex, partIndex },
      }
      slice.ids.push(part.id)
      slice.summary = nonblankSummary(slice.summary, compaction.summary)
      slice.carrier = { messageIndex, partIndex }
      group.set(channel, slice)
      groups.set(compaction.group, group)
    }
  }
  return groups
}

function renderedCompaction(part: SessionV1.Part) {
  if (part.type === "tool" && part.metadata?.providerExecuted) return undefined
  return compactionOf(part)
}

function coveringRun(run: CompactionRun, groups: ReturnType<typeof compactionGroups>, messageIndex: number) {
  if (!run.group) return run
  const slice = groups.get(run.group)?.get(run.channel)
  if (!slice) return run
  if (slice.carrier.messageIndex !== messageIndex || slice.carrier.partIndex !== run.end) return undefined
  return { ...run, ids: slice.ids, summary: slice.summary }
}

function compactionRunBlock(run: CompactionRun, groups: ReturnType<typeof compactionGroups>, messageIndex: number) {
  const covering = coveringRun(run, groups, messageIndex)
  if (covering) return compactionSummaryBlock(covering)
  const summary = run.group ? groups.get(run.group)?.get(run.channel)?.summary : undefined
  return compactionReferenceBlock(run, summary !== undefined)
}

// Parts that occupy space in the rendered request. Everything else is
// structural bookkeeping, and a span is allowed to swallow it.
const CONTENT_PARTS = new Set(["text", "reasoning", "tool", "file"])
// Structural parts that drive control flow elsewhere. A message carrying one is
// never folded into a span, however thoroughly its content was compacted.
const SPAN_EXCLUDED_PARTS = new Set(["compaction", "subtask"])

// Second-generation compaction: once a run of notes has itself been folded, the
// individual markers stop earning their place. A whole stretch of history
// collapses to one range block, and the messages it covered disappear —
// including their tool results, which is only safe because a tool call and its
// result live in the same part and therefore leave together.
//
// Spans are validated here rather than trusted from the caller, because an
// invalid one is a malformed provider request. Anything that fails validation
// silently falls back to per-part rendering, which is always correct.
function compactionSpans(messages: SessionV1.WithParts[], groups: ReturnType<typeof compactionGroups>) {
  const plan = new Map<number, { block: string } | "dropped">()
  const rendered = messages.map(willRender)
  let start = 0
  while (start < messages.length) {
    const span = spanGroup(messages[start])
    if (!span) {
      start++
      continue
    }
    let end = start
    while (end + 1 < messages.length) {
      const next = spanGroup(messages[end + 1])
      if (!next || next.group !== span.group || next.channel !== span.channel) break
      end++
    }
    // Collapsing closes a gap, so the survivors either side must still
    // alternate. The carrier keeps the first message's role and whatever
    // precedes it already disagreed with that role, so only the far side can
    // collide: give back messages from the end until it does not. Anything
    // handed back is simply rendered per-part, and the next iteration considers
    // it as a span of its own.
    while (end > start && !alternates(messages, rendered, start, end)) end--
    const covered = messages.slice(start, end + 1)
    const parts = covered.flatMap((message) => message.parts.filter((part) => CONTENT_PARTS.has(part.type)))
    const slice = groups.get(span.group)?.get(span.channel)
    const carrier = slice?.carrier
    plan.set(start, {
      block: compactionSpanBlock({
        group: span.group,
        channel: span.channel,
        from: parts.at(0)!.id,
        to: parts.at(-1)!.id,
        parts: parts.length,
        messages: covered.length,
        summary: carrier && carrier.messageIndex >= start && carrier.messageIndex <= end ? slice.summary : undefined,
        carriedLater: slice?.summary !== undefined && carrier !== undefined && carrier.messageIndex > end,
      }),
    })
    for (let index = start + 1; index <= end; index++) plan.set(index, "dropped")
    start = end + 1
  }
  return plan
}

// Whether collapsing [start, end] leaves the transcript alternating. Only the
// message after the span can collide, and it has to be the next one the renderer
// will actually emit rather than the next one in the input.
function alternates(messages: SessionV1.WithParts[], rendered: boolean[], start: number, end: number) {
  // A single-message span drops nothing, so no gap closes.
  if (end === start) return true
  for (let index = end + 1; index < messages.length; index++) {
    if (!rendered[index]) continue
    return messages[index].info.role !== messages[start].info.role
  }
  // The span runs to the end of the transcript; there is nothing to collide with.
  return true
}

// Whether the renderer emits anything at all for this message. Deliberately
// mirrors the skips in the main loop: reasoning about the provider's view of the
// transcript is only sound if it accounts for messages that never arrive.
function hasContentFilterNote(message: SessionV1.WithParts) {
  return (
    message.info.role === "assistant" &&
    message.info.error !== undefined &&
    ContentFilterError.isInstance(message.info.error)
  )
}

function rendersStoredParts(message: SessionV1.WithParts) {
  if (message.parts.length === 0) return false
  if (message.info.role !== "assistant" || !message.info.error) return true
  return (
    AbortedError.isInstance(message.info.error) &&
    message.parts.some((part) => part.type !== "step-start" && part.type !== "reasoning")
  )
}

export function projectErroredTurns(messages: Iterable<WithParts>) {
  // Durable parts stay recoverable, but every model-facing history consumer
  // must see the same errored-turn boundary as toModelMessages.
  return Array.from(messages, (message) =>
    rendersStoredParts(message) || message.parts.length === 0 ? message : { ...message, parts: [] },
  )
}

function willRender(message: SessionV1.WithParts) {
  return hasContentFilterNote(message) || rendersStoredParts(message)
}

// The group a whole message has been folded into, or undefined when the message
// is not a candidate: it must be entirely compacted, at generation 2 or beyond,
// under one group, and carry nothing structural that other code depends on.
function spanGroup(message: SessionV1.WithParts) {
  // An errored turn is dropped by the renderer, so folding it into a span would
  // put it back into the transcript under a covering summary.
  if (!rendersStoredParts(message)) return undefined
  if (message.parts.some((part) => SPAN_EXCLUDED_PARTS.has(part.type))) return undefined
  const content = message.parts.filter((part) => CONTENT_PARTS.has(part.type))
  if (content.length === 0) return undefined
  // Tool calls and results must remain paired in their actual tool channel;
  // replacing the containing assistant message with text would change it.
  if (content.some((part) => part.type === "tool")) return undefined
  const first = renderedCompaction(content[0])
  if (!first?.group || first.generation < 2) return undefined
  const channel = message.info.role
  return content.every((part) => {
    const record = renderedCompaction(part)
    return record !== undefined && record.group === first.group && record.generation >= 2
  })
    ? { group: first.group, channel }
    : undefined
}

function compactionSpanBlock(span: {
  group: string
  channel: ProjectionChannel
  from: string
  to: string
  parts: number
  messages: number
  summary?: string
  carriedLater?: boolean
}) {
  const summary = span.summary?.trim()
  const body = summary
    ? summary.length > COMPACTION_SUMMARY_MAX_CHARS
      ? `${summary.slice(0, COMPACTION_SUMMARY_MAX_CHARS)} [summary truncated]`
      : summary
    : span.carriedLater
      ? "This stretch is covered by its compaction group's later summary."
      : "No summary was recorded. Use list_context and read_part to recover this stretch."
  return [
    `<compacted-span group="${escapeAttribute(span.group)}" channel="${span.channel}" from="${span.from}" to="${span.to}" parts="${span.parts}" messages="${span.messages}">`,
    body,
    "</compacted-span>",
  ].join("\n")
}

// Every covered id is listed, including tools that also retain an inline
// marker, so summaries and markers remain unambiguous across messages.
function compactionSummaryBlock(run: CompactionRun) {
  const group = run.group ? ` group="${escapeAttribute(run.group)}" channel="${run.channel}"` : ""
  const attr = run.ids.length ? ` parts="${run.ids.join(" ")}"` : ""
  const summary = run.summary?.trim()
  const body = summary
    ? summary.length > COMPACTION_SUMMARY_MAX_CHARS
      ? `${summary.slice(0, COMPACTION_SUMMARY_MAX_CHARS)} [summary truncated]`
      : summary
    : run.native
      ? "Older tool output, pruned automatically to reclaim context. Use read_part to recover it verbatim."
      : "No summary was recorded. Use read_part to recover this content verbatim."
  return `<compaction-summary${group}${attr}>\n${body}\n</compaction-summary>`
}

function compactionReferenceBlock(run: CompactionRun, carriedLater: boolean) {
  const body = carriedLater
    ? "The full summary is carried by this group's later compaction-summary for the same channel."
    : "No summary was recorded. Use read_part to recover this content verbatim."
  return `<compaction-ref group="${escapeAttribute(run.group!)}" channel="${run.channel}" parts="${run.ids.join(" ")}">\n${body}\n</compaction-ref>`
}

export const Event = {
  Updated: SessionV1.Event.MessageUpdated,
  Removed: SessionV1.Event.MessageRemoved,
  PartUpdated: SessionV1.Event.PartUpdated,
  PartDelta: SessionV1.Event.PartDelta,
  PartRemoved: SessionV1.Event.PartRemoved,
}

const Cursor = Schema.Struct({
  id: MessageID,
  time: Schema.Finite.check(Schema.isGreaterThanOrEqualTo(0)),
})
type Cursor = typeof Cursor.Type

const decodeCursor = Schema.decodeUnknownSync(Cursor)

export const cursor = {
  encode(input: Cursor) {
    return Buffer.from(JSON.stringify(input)).toString("base64url")
  },
  decode(input: string) {
    return decodeCursor(JSON.parse(Buffer.from(input, "base64url").toString("utf8")))
  },
}

const info = (row: typeof MessageTable.$inferSelect) =>
  ({
    ...row.data,
    id: row.id,
    sessionID: row.session_id,
  }) as Info

const part = (row: typeof PartTable.$inferSelect) =>
  ({
    ...row.data,
    id: row.id,
    sessionID: row.session_id,
    messageID: row.message_id,
  }) as Part

const older = (row: Cursor) =>
  or(lt(MessageTable.time_created, row.time), and(eq(MessageTable.time_created, row.time), lt(MessageTable.id, row.id)))

function hydrate(db: Database.Interface["db"], rows: (typeof MessageTable.$inferSelect)[]) {
  const ids = rows.map((row) => row.id)
  const partByMessage = new Map<string, Part[]>()
  return Effect.gen(function* () {
    if (ids.length > 0) {
      const partRows = yield* db
        .select()
        .from(PartTable)
        .where(inArray(PartTable.message_id, ids))
        .orderBy(PartTable.message_id, PartTable.id)
        .all()
        .pipe(Effect.orDie)
      for (const row of partRows) {
        const next = part(row)
        const list = partByMessage.get(row.message_id)
        if (list) list.push(next)
        else partByMessage.set(row.message_id, [next])
      }
    }

    return rows.map((row) => ({
      info: info(row),
      parts: partByMessage.get(row.id) ?? [],
    }))
  })
}

function providerMeta(metadata: Record<string, any> | undefined) {
  if (!metadata) return undefined
  const { providerExecuted: _, ...rest } = metadata
  return Object.keys(rest).length > 0 ? rest : undefined
}

// Stamps each USER message, and nothing else, with its creation time and a short
// token. The model is the only consumer: `ts` gives it temporal grounding across
// turns, and `rand` gives every message a distinct identity so repeated
// identical prompts do not read as one input.
//
// Both fields are stable per message across every request: `ts` is the immutable
// creation time and the token is an FNV-1a hash of the message id, which already
// carries a random suffix. That stability is load-bearing — a value that changed
// per request would rewrite the tail of every historical user message and
// invalidate provider prompt caching on each turn.
//
// Gated behind `stampUser` so only the live prompt loop stamps; title, summary
// and compaction requests are unaffected.
const userStamp = (id: string, ts: number) => {
  let h = 0x811c9dc5
  for (let i = 0; i < id.length; i++) {
    h ^= id.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  const json = JSON.stringify({ ts, rand: (h >>> 0).toString(16).padStart(8, "0") })
  return `<message-metadata>${json}</message-metadata>`
}

export const toModelMessagesEffect = Effect.fnUntraced(function* (
  input: WithParts[],
  model: Provider.Model,
  options?: { stripMedia?: boolean; toolOutputMaxChars?: number; stampUser?: boolean },
) {
  const projected = projectErroredTurns(input)
  const result: UIMessage[] = []
  const toolNames = new Set<string>()
  // Track media from tool results that need to be injected as user messages
  // for providers that don't support that media type in tool results.
  //
  // OpenAI-compatible APIs only support string content in tool results, so we need
  // to extract media and inject as user messages. Some SDKs only support a subset
  // of media in tool results; e.g. Bedrock supports images but not PDFs there.
  //
  // Only apply this workaround if the model actually supports that media input -
  // otherwise unsupportedParts() will turn it into a user-visible error.
  const supportsMediaInToolResult = (attachment: { mime: string }) => {
    if (model.api.npm === "@ai-sdk/anthropic") return true
    if (model.api.npm === "@ai-sdk/openai") return true
    if (model.api.npm === "@ai-sdk/amazon-bedrock/mantle") return true
    if (model.api.npm === "@ai-sdk/amazon-bedrock") return attachment.mime.startsWith("image/")
    if (model.api.npm === "@ai-sdk/xai") return attachment.mime.startsWith("image/")
    if (model.api.npm === "@ai-sdk/google-vertex/anthropic") return true
    if (model.api.npm === "@ai-sdk/google") {
      const id = model.api.id.toLowerCase()
      return id.includes("gemini-3") && !id.includes("gemini-2")
    }
    return false
  }

  const toModelOutput = (options: { toolCallId: string; input: unknown; output: unknown }) => {
    const output = options.output
    if (typeof output === "string") {
      return { type: "text", value: output }
    }

    if (typeof output === "object") {
      const outputObject = output as {
        text: string
        attachments?: Array<{ mime: string; url: string }>
      }
      const attachments = (outputObject.attachments ?? []).filter((attachment) => {
        return attachment.url.startsWith("data:") && attachment.url.includes(",")
      })

      return {
        type: "content",
        value: [
          ...(outputObject.text ? [{ type: "text", text: outputObject.text }] : []),
          ...attachments.map((attachment) => ({
            type: "media",
            mediaType: attachment.mime,
            data: iife(() => {
              const commaIndex = attachment.url.indexOf(",")
              return commaIndex === -1 ? attachment.url : attachment.url.slice(commaIndex + 1)
            }),
          })),
        ],
      }
    }

    return { type: "json", value: output as never }
  }

  const groups = compactionGroups(projected)
  const spans = compactionSpans(projected, groups)

  for (const [messageIndex, msg] of projected.entries()) {
    if (!willRender(msg)) continue
    const span = spans.get(messageIndex)
    // The rest of the span left with the carrier.
    if (span === "dropped") continue
    if (span) {
      result.push({
        id: msg.info.id,
        role: msg.info.role === "user" ? "user" : "assistant",
        parts: [{ type: "text", text: span.block }],
      })
      continue
    }

    if (msg.info.role === "user") {
      const userMessage: UIMessage = {
        id: msg.info.id,
        role: "user",
        parts: [],
      }
      // Compacted user text and attachments collapse into runs exactly like
      // assistant content. Neither has a structural marker to preserve, so the
      // covering summary is the only thing emitted.
      let run: CompactionRun | undefined
      const closeRun = () => {
        if (!run) return
        userMessage.parts.push({ type: "text", text: compactionRunBlock(run, groups, messageIndex) })
        run = undefined
      }
      for (const [partIndex, part] of msg.parts.entries()) {
        const compaction = compactionOf(part)
        if (compaction && (part.type === "text" || part.type === "file")) {
          const group = compaction.group ?? part.id
          if (run && run.group !== group) closeRun()
          run ??= {
            channel: "user",
            group: compaction.group,
            ids: [],
            summary: nonblankSummary(undefined, compaction.summary),
            end: partIndex,
          }
          run.ids.push(part.id)
          run.summary = nonblankSummary(run.summary, compaction.summary)
          run.end = partIndex
          continue
        }
        closeRun()
        // User message parts should never be empty
        if (part.type === "text" && !part.ignored && part.text !== "")
          userMessage.parts.push({
            type: "text",
            text: part.text,
          })
        // text/plain and directory files are converted into text parts, ignore them
        if (part.type === "file" && part.mime !== "text/plain" && part.mime !== "application/x-directory") {
          if (options?.stripMedia && isMedia(part.mime)) {
            userMessage.parts.push({
              type: "text",
              text: `[Attached ${part.mime}: ${part.filename ?? "file"}]`,
            })
          } else {
            userMessage.parts.push({
              type: "file",
              url: part.url,
              mediaType: part.mime,
              filename: part.filename,
            })
          }
        }

        if (part.type === "compaction") {
          userMessage.parts.push({
            type: "text",
            text: "What did we do so far?",
          })
        }
        if (part.type === "subtask") {
          userMessage.parts.push({
            type: "text",
            text: "The following tool was executed by the user",
          })
        }
      }
      closeRun()
      if (userMessage.parts.length > 0) {
        if (options?.stampUser)
          userMessage.parts.push({ type: "text", text: userStamp(msg.info.id, msg.info.time.created) })
        result.push(userMessage)
      }
    }

    if (msg.info.role === "assistant") {
      const differentModel = `${model.providerID}/${model.id}` !== `${msg.info.providerID}/${msg.info.modelID}`
      const media: Array<{ mime: string; url: string; filename?: string }> = []

      if (
        msg.info.error &&
        !(
          AbortedError.isInstance(msg.info.error) &&
          msg.parts.some((part) => part.type !== "step-start" && part.type !== "reasoning")
        )
      ) {
        // Dropping a classifier-rejected turn entirely leaves no evidence it
        // happened, so the model replays the request that was refused and gets
        // refused again. Replace it with a note instead of its content: the
        // content is what was refused, and any tool calls in it never got
        // results.
        if (ContentFilterError.isInstance(msg.info.error))
          result.push({
            id: msg.info.id,
            role: "assistant",
            parts: [{ type: "text", text: CONTENT_FILTER_NOTE }],
          })
        continue
      }
      const assistantMessage: UIMessage = {
        id: msg.info.id,
        role: "assistant",
        parts: [],
      }
      // Anthropic adaptive thinking can persist assistant turns like:
      // step-start, reasoning(signature), text(""), step-start,
      // reasoning(signature). The empty text part is a structural separator,
      // but it does not carry the signature metadata itself. Dropping it shifts
      // signed thinking positions after step-start splitting/provider regrouping;
      // keeping it as "" is filtered by the AI SDK and rejected by Anthropic.
      // It is unclear whether this shape originates in our stream processing,
      // a proxy, or a lower-level library, but preserving a non-empty separator
      // here is the only safe replay point we have.
      // Use a single space so the separator survives replay without changing
      // the neighboring signed reasoning blocks.
      const hasSignedReasoning = msg.parts.some((part) => {
        if (part.type !== "reasoning") return false
        return part.metadata?.anthropic?.signature != null
      })
      // A compaction summary is emitted as a text block. Anthropic requires
      // thinking to lead the turn, so a summary flushed before a surviving
      // signed reasoning block would reorder text ahead of thinking and be
      // rejected. Hold any such summary until the last live signed block has
      // been emitted; tool markers are structurally pinned and stay in place.
      const lastSignedReasoning = msg.parts.findLastIndex(
        (part) =>
          part.type === "reasoning" && part.compacted === undefined && part.metadata?.anthropic?.signature != null,
      )
      const runs = compactionRuns(msg.parts)
      const firstTool = msg.parts.findIndex((part) => part.type === "tool")
      const channelBlocks =
        firstTool === -1
          ? []
          : msg.parts.flatMap((_, index) => {
              const run = runs.get(index)
              return run?.channel === "assistant" && index === run.end
                ? [compactionRunBlock(run, groups, messageIndex)]
                : []
            })
      const channelBlockIndex = Math.max(firstTool, lastSignedReasoning + 1)
      const flushChannelBlocks = () => {
        for (const block of channelBlocks) assistantMessage.parts.push({ type: "text", text: block })
        channelBlocks.length = 0
      }
      const deferred: string[] = []
      const flushDeferred = () => {
        for (const block of deferred) assistantMessage.parts.push({ type: "text", text: block })
        deferred.length = 0
      }
      for (const [index, part] of msg.parts.entries()) {
        if (index === channelBlockIndex) flushChannelBlocks()
        const run = runs.get(index)
        // Compacted assistant notes, reasoning and attachments have no
        // structural footprint, so they leave nothing behind but their id in
        // the covering summary.
        if (run && part.type !== "tool") {
          if (firstTool === -1 && index === run.end) {
            const block = compactionRunBlock(run, groups, messageIndex)
            // Anthropic wants thinking to lead the turn, so a summary that
            // would land before a surviving signed block waits its turn.
            if (index <= lastSignedReasoning) deferred.push(block)
            else assistantMessage.parts.push({ type: "text", text: block })
          }
          continue
        }
        if (part.type === "text") {
          assistantMessage.parts.push({
            type: "text",
            text: part.text === "" && hasSignedReasoning ? " " : part.text,
            ...(differentModel ? {} : { providerMetadata: part.metadata }),
          })
        }
        if (part.type === "step-start")
          assistantMessage.parts.push({
            type: "step-start",
          })
        if (part.type === "tool") {
          toolNames.add(part.tool)
          if (part.state.status === "completed") {
            // A tool result must survive as a block because the provider pairs
            // it with its tool_use, so a compacted one leaves a minimal marker
            // carrying the part id — that id is what makes read_part recovery
            // possible. The run's covering summary rides inside the last tool
            // result it owns rather than becoming a fresh assistant text block,
            // because emitting text after a tool_use is needless provider risk.
            const outputText = run
              ? index === run.lastTool
                ? [compactedMarker(part.id, run.group), compactionRunBlock(run, groups, messageIndex)].join("\n")
                : compactedMarker(part.id, run.group)
              : truncateToolOutput(part.state.output, options?.toolOutputMaxChars)
            const attachments = run || options?.stripMedia ? [] : (part.state.attachments ?? [])

            // For providers that don't support media in tool results, extract media files
            // (images, PDFs) to be sent as a separate user message
            const mediaAttachments = attachments.filter((a) => isMedia(a.mime))
            const extractedMedia = mediaAttachments.filter((a) => !supportsMediaInToolResult(a))
            if (extractedMedia.length > 0) {
              media.push(...extractedMedia)
            }
            const finalAttachments = attachments.filter((a) => !isMedia(a.mime) || supportsMediaInToolResult(a))

            const output =
              finalAttachments.length > 0
                ? {
                    text: outputText,
                    attachments: finalAttachments,
                  }
                : outputText

            assistantMessage.parts.push({
              type: ("tool-" + part.tool) as `tool-${string}`,
              state: "output-available",
              toolCallId: part.callID,
              // Historical tool calls are demonstrations of valid arguments to
              // the model. Keep their original schema-valid input: synthetic
              // compaction objects can be imitated as fresh, invalid calls.
              input: part.state.input,
              output,
              ...(part.metadata?.providerExecuted ? { providerExecuted: true } : {}),
              ...(differentModel || (run && !run.native) ? {} : { callProviderMetadata: providerMeta(part.metadata) }),
            })
          }
          if (part.state.status === "error") {
            const output = part.state.metadata?.interrupted === true ? part.state.metadata.output : undefined
            if (typeof output === "string") {
              assistantMessage.parts.push({
                type: ("tool-" + part.tool) as `tool-${string}`,
                state: "output-available",
                toolCallId: part.callID,
                input: part.state.input,
                output,
                ...(part.metadata?.providerExecuted ? { providerExecuted: true } : {}),
                ...(differentModel ? {} : { callProviderMetadata: providerMeta(part.metadata) }),
              })
            } else {
              assistantMessage.parts.push({
                type: ("tool-" + part.tool) as `tool-${string}`,
                state: "output-error",
                toolCallId: part.callID,
                input: part.state.input,
                errorText: part.state.error,
                ...(part.metadata?.providerExecuted ? { providerExecuted: true } : {}),
                ...(differentModel ? {} : { callProviderMetadata: providerMeta(part.metadata) }),
              })
            }
          }
          // Handle pending/running tool calls to prevent dangling tool_use blocks
          // Anthropic/Claude APIs require every tool_use to have a corresponding tool_result
          if (part.state.status === "pending" || part.state.status === "running")
            assistantMessage.parts.push({
              type: ("tool-" + part.tool) as `tool-${string}`,
              state: "output-error",
              toolCallId: part.callID,
              input: part.state.input,
              errorText: "[Tool execution was interrupted]",
              ...(part.metadata?.providerExecuted ? { providerExecuted: true } : {}),
              ...(differentModel ? {} : { callProviderMetadata: providerMeta(part.metadata) }),
            })
        }
        if (part.type === "reasoning") {
          if (differentModel) {
            if (part.text.trim().length > 0)
              assistantMessage.parts.push({
                type: "text",
                text: part.text,
              })
            continue
          }
          assistantMessage.parts.push({
            type: "reasoning",
            text: part.text,
            providerMetadata: part.metadata,
          })
        }
        // Past the last surviving signed thinking block it is safe to emit any
        // summary that was held back for ordering.
        if (index === lastSignedReasoning) flushDeferred()
      }
      flushChannelBlocks()
      flushDeferred()
      if (assistantMessage.parts.length > 0) {
        result.push(assistantMessage)
        // Inject pending media as a user message for providers that don't support
        // media (images, PDFs) in tool results
        if (media.length > 0) {
          result.push({
            id: MessageID.ascending(),
            role: "user",
            parts: [
              {
                type: "text" as const,
                text: SYNTHETIC_ATTACHMENT_PROMPT,
              },
              ...media.map((attachment) => ({
                type: "file" as const,
                url: attachment.url,
                mediaType: attachment.mime,
                filename: attachment.filename,
              })),
            ],
          })
        }
      }
    }
  }

  const tools = Object.fromEntries(Array.from(toolNames).map((toolName) => [toolName, { toModelOutput }]))

  return yield* Effect.promise(() =>
    convertToModelMessages(
      result.filter((msg) => msg.parts.some((part) => part.type !== "step-start")),
      {
        //@ts-expect-error (convertToModelMessages expects a ToolSet but only actually needs tools[name]?.toModelOutput)
        tools,
      },
    ),
  )
})

export function toModelMessages(
  input: WithParts[],
  model: Provider.Model,
  options?: { stripMedia?: boolean; toolOutputMaxChars?: number; stampUser?: boolean },
): Promise<ModelMessage[]> {
  return Effect.runPromise(toModelMessagesEffect(input, model, options))
}

export const page = Effect.fn("MessageV2.page")(function* (input: {
  sessionID: SessionID
  limit: number
  before?: string
}) {
  const { db } = yield* Database.Service
  const before = input.before ? cursor.decode(input.before) : undefined
  const where = before
    ? and(eq(MessageTable.session_id, input.sessionID), older(before))
    : eq(MessageTable.session_id, input.sessionID)
  const rows = yield* db
    .select()
    .from(MessageTable)
    .where(where)
    .orderBy(desc(MessageTable.time_created), desc(MessageTable.id))
    .limit(input.limit + 1)
    .all()
    .pipe(Effect.orDie)
  if (rows.length === 0) {
    const row = yield* db
      .select({ id: SessionTable.id })
      .from(SessionTable)
      .where(eq(SessionTable.id, input.sessionID))
      .get()
      .pipe(Effect.orDie)
    if (!row) return yield* new NotFoundError({ message: `Session not found: ${input.sessionID}` })
    return {
      items: [] as WithParts[],
      more: false,
    }
  }

  const more = rows.length > input.limit
  const slice = more ? rows.slice(0, input.limit) : rows
  const items = yield* hydrate(db, slice)
  items.reverse()
  const tail = slice.at(-1)
  return {
    items,
    more,
    cursor: more && tail ? cursor.encode({ id: tail.id, time: tail.time_created }) : undefined,
  }
})

export function stream(sessionID: SessionID) {
  const size = 50
  return Effect.gen(function* () {
    const result = [] as WithParts[]
    let before: string | undefined
    while (true) {
      const next = yield* page({ sessionID, limit: size, before }).pipe(
        Effect.catchIf(NotFoundError.isInstance, () =>
          Effect.succeed({ items: [] as WithParts[], more: false, cursor: undefined }),
        ),
      )
      if (next.items.length === 0) break
      for (let i = next.items.length - 1; i >= 0; i--) {
        const item = next.items[i]
        if (item) result.push(item)
      }
      if (!next.more || !next.cursor) break
      before = next.cursor
    }
    return result
  })
}

export function parts(messageID: MessageID) {
  return Effect.gen(function* () {
    const { db } = yield* Database.Service
    const rows = yield* db
      .select()
      .from(PartTable)
      .where(eq(PartTable.message_id, messageID))
      .orderBy(PartTable.id)
      .all()
      .pipe(Effect.orDie)
    return rows.map(part)
  })
}

export const get = Effect.fn("MessageV2.get")(function* (input: { sessionID: SessionID; messageID: MessageID }) {
  const { db } = yield* Database.Service
  const row = yield* db
    .select()
    .from(MessageTable)
    .where(and(eq(MessageTable.id, input.messageID), eq(MessageTable.session_id, input.sessionID)))
    .get()
    .pipe(Effect.orDie)
  if (!row) return yield* new NotFoundError({ message: `Message not found: ${input.messageID}` })
  return {
    info: info(row),
    parts: yield* parts(input.messageID),
  }
})

export function filterCompacted(msgs: Iterable<WithParts>) {
  const result = [] as WithParts[]
  const completed = new Set<string>()
  let retain: MessageID | undefined
  for (const msg of msgs) {
    result.push(msg)
    if (retain) {
      if (msg.info.id === retain) break
      continue
    }
    if (msg.info.role === "user" && completed.has(msg.info.id)) {
      const part = msg.parts.find((item): item is CompactionPart => item.type === "compaction")
      if (!part) continue
      if (!part.tail_start_id) break
      retain = part.tail_start_id
      if (msg.info.id === retain) break
      continue
    }
    if (msg.info.role === "user" && completed.has(msg.info.id) && msg.parts.some((part) => part.type === "compaction"))
      break
    if (msg.info.role === "assistant" && msg.info.summary && msg.info.finish && !msg.info.error)
      completed.add(msg.info.parentID)
  }
  result.reverse()
  const compactionIndex = result.findLastIndex(
    (msg) =>
      msg.info.role === "user" &&
      msg.parts.some((item): item is CompactionPart => item.type === "compaction" && item.tail_start_id !== undefined),
  )
  const compaction = result[compactionIndex]
  const part = compaction?.parts.find(
    (item): item is CompactionPart => item.type === "compaction" && item.tail_start_id !== undefined,
  )
  const summaryIndex = compaction
    ? result.findIndex(
        (msg, index) =>
          index > compactionIndex &&
          msg.info.role === "assistant" &&
          msg.info.summary &&
          msg.info.parentID === compaction.info.id,
      )
    : -1
  const tailIndex = part?.tail_start_id ? result.findIndex((msg) => msg.info.id === part.tail_start_id) : -1
  if (tailIndex >= 0 && tailIndex < compactionIndex && summaryIndex > compactionIndex) {
    return projectErroredTurns([
      ...result.slice(compactionIndex, summaryIndex + 1),
      ...result.slice(tailIndex, compactionIndex),
      ...result.slice(summaryIndex + 1),
    ])
  }
  return projectErroredTurns(result)
}

export const filterCompactedEffect = Effect.fnUntraced(function* (sessionID: SessionID) {
  return filterCompacted(yield* stream(sessionID))
})

// filterCompacted reorders messages for model consumption
// ([compaction-user, summary, ...retained tail..., continue-user]), so array
// position is not chronological. IDs are only a deterministic tie-breaker
// because imported messages do not necessarily have monotonic IDs.
export function latest(msgs: WithParts[]) {
  let user: User | undefined
  let assistant: Assistant | undefined
  let finished: Assistant | undefined
  for (const msg of msgs) {
    const info = msg.info
    if (info.role === "user" && isAfter(info, user)) user = info
    if (info.role === "assistant" && isAfter(info, assistant)) assistant = info
    if (info.role === "assistant" && info.finish && isAfter(info, finished)) finished = info
  }
  const tasks = msgs.flatMap((m) =>
    finished && !isAfter(m.info, finished)
      ? []
      : m.parts.filter((p): p is CompactionPart | SubtaskPart => p.type === "compaction" || p.type === "subtask"),
  )
  return { user, assistant, finished, tasks }
}

function isAfter(info: Info, other?: Info) {
  if (!other) return true
  if (info.time.created !== other.time.created) return info.time.created > other.time.created
  return info.id > other.id
}

// Transport failures that can recover without changing the request. Every code
// here is still bounded by the global retry limit (~60s of backoff) and then
// surfaces, so a genuinely broken endpoint still reports within a minute.
//
// Bun's native fetch collapses DNS failure, refused connection, TLS mismatch,
// and unroutable network into a single ConnectionRefused with one shared
// message, so it cannot be treated as permanent without also giving up retries
// for wifi flaps, VPN reconnects, and sleep/wake. Bounded retry is the only
// classification that serves both.
const TRANSIENT_SYS_CODES = new Set([
  "ECONNRESET",
  "ETIMEDOUT",
  "EAI_AGAIN",
  "EHOSTUNREACH",
  "ENETUNREACH",
  "EPIPE",
  "UND_ERR_CONNECT_TIMEOUT",
  "UND_ERR_HEADERS_TIMEOUT",
  "UND_ERR_BODY_TIMEOUT",
  "UND_ERR_SOCKET",
  "ConnectionClosed",
  "ConnectionRefused",
  "FailedToOpenSocket",
  "Timeout",
])
// Node and undici report these only for a genuinely unresolvable or refused
// endpoint, so they stay permanent and surface immediately. They are reachable
// through a custom fetch implementation rather than Bun's own.
const PERMANENT_SYS_CODES = new Set(["ENOTFOUND", "ECONNREFUSED", "DNSResolveFailed", "DNSResolutionFailed"])
const TRANSIENT_ERROR_CODES = new Set([
  "TimeoutError",
  "NETWORK_ERROR",
  "ZlibError",
  "ProviderHeaderTimeoutError",
  "ProviderResponseStreamError",
])
const CAUSE_MAX_DEPTH = 8

export function isPermanentTransportCode(code: unknown) {
  return typeof code === "string" && PERMANENT_SYS_CODES.has(code)
}

export function isTransientTransportCode(code: unknown) {
  return typeof code === "string" && (TRANSIENT_SYS_CODES.has(code) || TRANSIENT_ERROR_CODES.has(code))
}

// Exact messages emitted for a prematurely closed transport. Generic text
// such as "network error" or "fetch failed" cannot distinguish a transient
// disconnect from a permanent DNS, TLS, authentication, or endpoint failure.
const TRANSIENT_MESSAGES = new Set(["socket hang up", "sse read timed out", "other side closed", "terminated"])

// Walks the cause chain, which is where a transport fault ends up once the
// provider SDK wraps it. CAUSE_MAX_DEPTH bounds the walk, so a self-referential
// cause cannot spin.
function classifyTransportError(error: unknown) {
  let current = error
  let transient: SystemError | undefined
  for (let depth = 0; depth < CAUSE_MAX_DEPTH; depth++) {
    if (!(current instanceof Error)) break
    const code = "code" in current && typeof current.code === "string" ? current.code : undefined
    if (isPermanentTransportCode(code)) return { type: "permanent" as const, error: current as SystemError }
    if (!transient && code && TRANSIENT_SYS_CODES.has(code)) transient = current as SystemError
    current = current.cause
  }
  if (transient) return { type: "transient" as const, error: transient }
  return undefined
}

export function fromError(
  e: unknown,
  ctx: { providerID: ProviderV2.ID; aborted?: boolean },
): NonNullable<Assistant["error"]> {
  const source = RetryError.isInstance(e) ? e.lastError : e
  const error = classify(source, ctx)
  // A cancelled request must never be retried. Abort can surface as a
  // transport-shaped failure instead of an interrupt when it races the read, so
  // rather than guarding each transport branch, refuse every retryable
  // classification once the turn is known to be aborted.
  if (ctx.aborted && APIError.isInstance(error) && error.data.isRetryable)
    return new AbortedError({ message: error.data.message }, { cause: source }).toObject()
  return error
}

function classify(e: unknown, ctx: { providerID: ProviderV2.ID; aborted?: boolean }): NonNullable<Assistant["error"]> {
  const transport = classifyTransportError(e)
  const transient = transport?.type === "transient" ? transport.error : undefined
  const permanent = transport?.type === "permanent" ? transport.error : undefined
  const sysCode = transient?.code
  switch (true) {
    case e instanceof DOMException && e.name === "AbortError":
      // Controlled timeouts carry their own error types. A bare AbortError is a cancellation.
      return new AbortedError({ message: e.message }, { cause: e }).toObject()
    case e instanceof DOMException && e.name === "TimeoutError":
      return new APIError(
        { message: e.message || "Request timed out", isRetryable: true, metadata: { code: "TimeoutError" } },
        { cause: e },
      ).toObject()
    case OutputLengthError.isInstance(e):
      return e
    case LoadAPIKeyError.isInstance(e):
      return new AuthError(
        {
          providerID: ctx.providerID,
          message: e.message,
        },
        { cause: e },
      ).toObject()
    // An APICallError carrying a transport fault is handled in its own branch
    // below so the response envelope survives; this branch is for a bare
    // transport error with no response.
    case sysCode !== undefined && !APICallError.isInstance(e):
      return new APIError(
        {
          message: sysCode === "ECONNRESET" ? "Connection reset by server" : `Network error (${sysCode})`,
          isRetryable: true,
          metadata: {
            code: sysCode,
            syscall: transient?.syscall ?? "",
            message: transient?.message ?? "",
          },
        },
        { cause: e },
      ).toObject()
    case e instanceof Error && (e as FetchDecompressionError).code === "ZlibError":
      return new APIError(
        {
          message: "Response decompression failed",
          isRetryable: true,
          metadata: {
            code: (e as FetchDecompressionError).code,
            message: e.message,
          },
        },
        { cause: e },
      ).toObject()
    case e instanceof ProviderError.HeaderTimeoutError:
      return new APIError(
        {
          message: e.message,
          isRetryable: true,
          metadata: {
            code: e.name,
            timeoutMs: String(e.ms),
          },
        },
        { cause: e },
      ).toObject()
    case e instanceof ProviderError.ResponseStreamError:
      return new APIError(
        {
          message: e.message,
          isRetryable: true,
          metadata: {
            code: e.name,
          },
        },
        { cause: e },
      ).toObject()
    case APICallError.isInstance(e):
      const parsed = ProviderError.parseAPICallError({
        providerID: ctx.providerID,
        error: e,
      })
      if (parsed.type === "context_overflow") {
        return new ContextOverflowError(
          {
            message: parsed.message,
            responseBody: parsed.responseBody,
          },
          { cause: e },
        ).toObject()
      }

      // Keep the response envelope even when a transport fault is the root
      // cause: retry-after headers and the body are the useful diagnostics, and
      // replacing them with a bare network error loses both.
      const transportError = permanent ?? transient
      return new APIError(
        {
          message: parsed.message,
          statusCode: parsed.statusCode,
          isRetryable: permanent ? false : transient !== undefined || parsed.isRetryable,
          responseHeaders: parsed.responseHeaders,
          responseBody: parsed.responseBody,
          metadata: transportError
            ? {
                ...parsed.metadata,
                code: transportError.code ?? "",
                syscall: transportError.syscall ?? "",
                message: transportError.message ?? "",
              }
            : parsed.metadata,
        },
        { cause: e },
      ).toObject()
    case e instanceof Error && TRANSIENT_MESSAGES.has(e.message.toLowerCase()):
      return new APIError(
        { message: e.message, isRetryable: true, metadata: { code: "NETWORK_ERROR", message: e.message } },
        { cause: e },
      ).toObject()
    case e instanceof Error:
      return new NamedError.Unknown({ message: errorMessage(e) }, { cause: e }).toObject()
    default:
      try {
        const parsed = ProviderError.parseStreamError(e)
        if (parsed) {
          if (parsed.type === "context_overflow") {
            return new ContextOverflowError(
              {
                message: parsed.message,
                responseBody: parsed.responseBody,
              },
              { cause: e },
            ).toObject()
          }
          return new APIError(
            {
              message: parsed.message,
              isRetryable: parsed.isRetryable,
              responseBody: parsed.responseBody,
            },
            {
              cause: e,
            },
          ).toObject()
        }
      } catch {}
      return new NamedError.Unknown({ message: JSON.stringify(e) }, { cause: e }).toObject()
  }
}

export * as MessageV2 from "./message-v2"
export const node = LayerNode.group([Database.node])
