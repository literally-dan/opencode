import { SessionV1 } from "@opencode-ai/core/v1/session"

export type ProjectionChannel = "user" | "assistant" | "tool"

export function projectionChannel(part: SessionV1.Part, messageRole: SessionV1.Info["role"]): ProjectionChannel {
  if (messageRole === "user") return "user"
  if (part.type === "tool") return "tool"
  return "assistant"
}

export function projectionChannelKey(group: string, channel: ProjectionChannel) {
  return JSON.stringify([group, channel])
}
import { Effect, Semaphore } from "effect"

export type Kind = "tool" | "text" | "reasoning" | "file"
export type Eligible = {
  ok: true
  kind: Kind
  // Current rendered cost, not the original size: a re-compaction candidate is
  // only worth its summary, so selection heuristics must not promise the
  // original bytes back a second time.
  chars: number
  generation: number
  tool?: string
}
export type Ineligible = { ok: false; reason: string }

// `live` marks the newest user message (the request being served) and the tail
// turns still in flight. Both are protected regardless of what the model asks
// for: compacting the instruction you are mid-way through executing is never a
// win, and the tail is the working set.
export type MessageContext = {
  role: "user" | "assistant"
  current: boolean
  summary?: boolean
  live?: boolean
}

export type ContextManagementReceiptKind = "list_context" | "compact_results" | "compact_bulk"
export type ContextManagementReceiptPart = SessionV1.ToolPart & { state: SessionV1.ToolStateCompleted }

const PART_ID = /^prt_[0-9a-f]{12}[0-9A-Za-z]{14}$/
export const COMPACTION_SUMMARY_MAX_CHARS = 4_000
export const COMPACTION_MAX_PARTS = 200
export const COMPACTION_PLANNING_SUMMARY = "x".repeat(COMPACTION_SUMMARY_MAX_CHARS)
const COMPACTED_FALLBACK = "No summary was recorded. Use read_part to recover this content verbatim."
export const NATIVE_FALLBACK =
  "Older tool output, pruned automatically to reclaim context. Use read_part to recover it verbatim."
const COMPACTION_REFERENCE =
  "The full summary is carried by this group's later compaction-summary for the same channel."
const ESTIMATED_GROUP = "00000000-0000-0000-0000-000000000000"
const compactionLocks = new Map<string, { semaphore: Semaphore.Semaphore; references: number }>()

export function contextManagementReceiptKind(part: SessionV1.Part): ContextManagementReceiptKind | undefined {
  if (part.type !== "tool" || part.state.status !== "completed" || part.metadata?.providerExecuted) return
  if (part.tool === "list_context" || part.tool === "compact_results" || part.tool === "compact_bulk") return part.tool
}

export function isContextManagementReceipt(part: SessionV1.Part): part is ContextManagementReceiptPart {
  return contextManagementReceiptKind(part) !== undefined
}

export function isFullContextManagementReceipt(part: SessionV1.Part): part is ContextManagementReceiptPart {
  return isContextManagementReceipt(part) && !isManualToolCompaction(part) && part.state.time.compacted === undefined
}

export function isCollapsedContextManagementReceipt(part: SessionV1.Part): part is ContextManagementReceiptPart {
  return isContextManagementReceipt(part) && !isManualToolCompaction(part) && part.state.time.compacted !== undefined
}

export function hasDurableProviderCompletion(message: SessionV1.WithParts) {
  return message.info.role === "assistant" && message.parts.some((part) => part.type === "step-finish")
}

export function contextManagementReceiptSummary(part: SessionV1.Part) {
  if (!isContextManagementReceipt(part)) return
  const kind = contextManagementReceiptKind(part)
  if (!kind) return
  const metadata = part.state.metadata
  const metric = (key: string) => {
    const value: unknown = metadata[key]
    return typeof value === "number" && Number.isSafeInteger(value) && value >= 0 ? String(value) : "unknown"
  }
  const details =
    kind === "list_context"
      ? `total parts=${metric("total")}; compactable parts=${metric("compactable")}; estimated reclaimable characters=${metric("charsCompactable")}`
      : kind === "compact_results"
        ? `compacted parts=${metric("compacted")}; skipped parts=${metric("skipped")}; net characters freed=${metric("charsFreed")}`
        : `compacted parts=${metric("compacted")}; skipped parts=${metric("skipped")}; net characters freed=${metric("charsFreed")}; summarizer calls=${metric("summarizerCalls")}`
  return `Context-management receipt ${part.id} (${kind}): ${details}. Use read_part with part_id="${part.id}" to recover the original full result.`
}

function acquireCompactionLock(sessionID: string) {
  const hit = compactionLocks.get(sessionID)
  if (hit) {
    hit.references++
    return hit
  }
  const next = { semaphore: Semaphore.makeUnsafe(1), references: 1 }
  compactionLocks.set(sessionID, next)
  return next
}

function releaseCompactionLock(sessionID: string, entry: ReturnType<typeof acquireCompactionLock>) {
  entry.references--
  if (entry.references === 0 && compactionLocks.get(sessionID) === entry) compactionLocks.delete(sessionID)
}

export function compactionLockCount() {
  return compactionLocks.size
}

export function compactionLockReferences(sessionID: string) {
  return compactionLocks.get(sessionID)?.references ?? 0
}

export function withCompactionLock<A, E, R>(
  sessionID: string,
  signal: AbortSignal | undefined,
  effect: Effect.Effect<A, E, R>,
) {
  return Effect.acquireUseRelease(
    Effect.sync(() => acquireCompactionLock(sessionID)),
    (entry) => {
      const locked = entry.semaphore.withPermits(1)(effect)
      if (!signal) return locked
      const abort = Effect.callback<void>((resume) => {
        if (signal.aborted) return resume(Effect.void)
        const handler = () => resume(Effect.void)
        signal.addEventListener("abort", handler, { once: true })
        return Effect.sync(() => signal.removeEventListener("abort", handler))
      })
      return Effect.raceFirst(locked, abort.pipe(Effect.flatMap(() => Effect.interrupt)))
    },
    (entry) => Effect.sync(() => releaseCompactionLock(sessionID, entry)),
  )
}

export function isCanonicalPartID(value: string) {
  return PART_ID.test(value)
}

export type Located = {
  part: SessionV1.Part
  messageID: string
  context: MessageContext
}

// One index of everything the model can see this turn, shared by every caller
// that needs to reason about compaction candidates, so they cannot disagree
// about what is eligible or which message is live.
//
// The live region is the newest user message only — the instruction currently
// being served. Deliberately not "everything after it": the big tool outputs
// produced in service of that instruction are the most valuable thing to
// compact once the model has taken what it needs from them.
export function indexParts(messages: SessionV1.WithParts[], currentMessageID: string) {
  const liveMessageID = messages.findLast((msg) => msg.info.role === "user")?.info.id
  const located = new Map<string, Located>()
  const ambiguous = new Set<string>()
  for (const msg of messages) {
    for (const part of msg.parts) {
      if (located.has(part.id)) {
        ambiguous.add(part.id)
        continue
      }
      located.set(part.id, {
        part,
        messageID: msg.info.id,
        context: {
          role: msg.info.role,
          current: msg.info.id === currentMessageID,
          summary: msg.info.role === "assistant" ? msg.info.summary : undefined,
          live: msg.info.id === liveMessageID,
        },
      })
    }
  }
  // A duplicated id is unresolvable, so fail closed rather than compact the
  // wrong part.
  for (const partID of ambiguous) located.delete(partID)
  return { located, ambiguous }
}

export function compactionInventory(
  messages: SessionV1.WithParts[],
  currentMessageID: string,
  durableMessages: SessionV1.WithParts[] = messages,
) {
  const visible = new Set<string>(messages.flatMap((message) => message.parts.map((part) => part.id)))
  const { located, ambiguous } = indexParts(durableMessages, currentMessageID)
  const roles = new Map(durableMessages.map((message) => [message.info.id, message.info.role]))
  const eligible = new Map(
    [...located.entries()].flatMap(([partID, found]) => {
      if (!visible.has(partID)) return []
      const allowed = eligibility(found.part, found.context)
      return allowed.ok ? [[partID, { ...found, allowed }] as const] : []
    }),
  )
  const selected = new Set(eligible.keys())
  const unsafe = unsafeCompactionGroups(
    durableMessages.flatMap((message) => message.parts),
    selected,
    ambiguous,
    roles,
  )
  const blocked = new Map<string, string>()
  const safe = [...eligible.entries()].flatMap(([partID, found]) => {
    const group = compactionOf(found.part)?.group
    const reason = group
      ? unsafe.get(projectionChannelKey(group, projectionChannel(found.part, found.context.role)))
      : undefined
    if (!reason)
      return [
        [partID, { ...found, savings: compactionSavings([found.part], COMPACTION_PLANNING_SUMMARY, roles) }] as const,
      ]
    blocked.set(partID, reason)
    return []
  })
  const ranked = safe.toSorted((a, b) => b[1].savings - a[1].savings || a[0].localeCompare(b[0]))
  const planned = ranked.slice(0, COMPACTION_MAX_PARTS)
  for (const [partID] of ranked.slice(COMPACTION_MAX_PARTS)) {
    blocked.set(partID, `is outside this inventory's ${COMPACTION_MAX_PARTS}-part actionable limit`)
  }
  const actionable = new Map(planned)
  return { located, ambiguous, roles, actionable, blocked }
}

// The compaction record, read uniformly across the three shapes that carry it
// (tool state, text/reasoning, file). Returns undefined for a live part.
export function compactionOf(part: SessionV1.Part) {
  if (part.type === "tool") {
    if (part.state.status !== "completed") return undefined
    if (part.state.compactionGroup === undefined && part.state.time.compacted === undefined) return undefined
    const manual = isManualToolCompaction(part)
    const administrative = isCollapsedContextManagementReceipt(part)
    return {
      group: part.state.compactionGroup,
      summary: manual ? part.state.compactionSummary : contextManagementReceiptSummary(part),
      generation: Math.max(part.state.compactionGeneration ?? 1, 1),
      // `false` remains the manual-compaction signal used by overflow handling.
      // Administrative receipts keep a distinct undefined state while their
      // structured metadata supplies the projected summary.
      native: administrative ? undefined : !manual,
    }
  }
  if (part.type === "text" || part.type === "reasoning" || part.type === "file") {
    if (part.compacted === undefined) return undefined
    return {
      group: part.compactionGroup,
      summary: part.compactionSummary,
      generation: Math.max(part.compactionGeneration ?? 1, 1),
      native: false,
    }
  }
  return undefined
}

export function setCompaction(
  part: SessionV1.Part,
  record: { at: number; group: string; summary?: string; generation: number },
) {
  if (part.type === "tool" && part.state.status === "completed") {
    // Manual compaction owns the part from here; clearing the native marker
    // keeps `prune`'s backwards scan from treating it as its own boundary.
    delete part.state.time.compacted
    part.state.compactionGroup = record.group
    part.state.compactionGeneration = record.generation
    if (record.summary) part.state.compactionSummary = record.summary
    return
  }
  if (part.type === "text" || part.type === "reasoning" || part.type === "file") {
    part.compacted = record.at
    part.compactionGroup = record.group
    part.compactionGeneration = record.generation
    if (record.summary) part.compactionSummary = record.summary
  }
}

export function isManualToolCompaction(part: SessionV1.ToolPart) {
  return (
    part.state.status === "completed" &&
    (part.state.compactionGroup !== undefined || part.state.compactionSummary !== undefined)
  )
}

export function isCompacted(part: SessionV1.Part) {
  return compactionOf(part) !== undefined
}

// What a summarizer should read. A part that already carries a summary is
// re-compacted from that summary, not from the original bytes — folding notes
// is the whole point of meta-compaction, and re-reading megabytes to produce a
// coarser note would defeat it. Originals stay in storage either way.
export function compactionInput(part: SessionV1.Part) {
  const existing = compactionOf(part)
  if (existing?.summary) return existing.summary
  if (part.type === "tool" && part.state.status === "completed")
    return `Input: ${JSON.stringify(part.state.input)}\nOutput:\n${part.state.output}`
  if (part.type === "text" || part.type === "reasoning") return part.text
  if (part.type === "file") return `[${part.mime}] ${part.filename ?? part.url.slice(0, 200)}`
  return ""
}

// Rendered cost of a part right now. Compacted tool calls still retain a marker
// and their original input, while their group retains one wrapper and its recovery ids.
export function effectiveChars(part: SessionV1.Part) {
  return renderedChars([part])
}

export function renderedChars(
  parts: readonly SessionV1.Part[],
  roleByMessage?: ReadonlyMap<string, SessionV1.Info["role"]>,
) {
  const groups = new Map<string, SessionV1.Part[]>()
  let total = 0
  for (const part of parts) {
    const existing = renderedCompaction(part)
    if (!existing) {
      total += freshRenderedChars(part)
      continue
    }
    if (!existing.group) {
      total += compactedRenderedChars([part], existing.summary, existing.native === true)
      continue
    }
    const role = roleByMessage?.get(part.messageID) ?? inferredRole(part)
    const key = projectionChannelKey(existing.group, projectionChannel(part, role))
    groups.set(key, [...(groups.get(key) ?? []), part])
  }
  for (const members of groups.values()) {
    const existing = renderedCompaction(members[0]!)!
    const role = projectionChannel(members[0]!, roleByMessage?.get(members[0]!.messageID) ?? inferredRole(members[0]!))
    const summary = members.map((part) => renderedCompaction(part)?.summary).find((value) => value !== undefined)
    const normal = compactedRenderedChars(members, summary, false, existing.group, role)
    total +=
      existing.generation >= 2 && members.every((part) => part.type !== "tool")
        ? Math.min(normal, compactedSpanChars(members, summary, existing.group!, role))
        : normal
  }
  return total
}

export function compactionRenderedChars(
  parts: readonly SessionV1.Part[],
  summary: string,
  roleByMessage?: ReadonlyMap<string, SessionV1.Info["role"]>,
) {
  const roles = new Map<string, SessionV1.Part[]>()
  for (const part of parts) {
    const role = roleByMessage?.get(part.messageID) ?? inferredRole(part)
    const channel = projectionChannel(part, role)
    roles.set(channel, [...(roles.get(channel) ?? []), part])
  }
  return [...roles.entries()].reduce(
    (total, [role, members]) => total + compactedRenderedChars(members, summary, false, ESTIMATED_GROUP, role, true),
    0,
  )
}

export function compactionSavings(
  parts: readonly SessionV1.Part[],
  summary: string,
  roleByMessage?: ReadonlyMap<string, SessionV1.Info["role"]>,
) {
  return renderedChars(parts, roleByMessage) - compactionRenderedChars(parts, summary, roleByMessage)
}

function renderedCompaction(part: SessionV1.Part) {
  if (part.type === "tool" && part.metadata?.providerExecuted) return undefined
  return compactionOf(part)
}

function freshRenderedChars(part: SessionV1.Part) {
  if (part.type === "tool")
    return part.state.status === "completed"
      ? removableTextChars(part.state.output) + serializedChars(part.state.input)
      : 0
  if (part.type === "text" || part.type === "reasoning") return removableTextChars(part.text)
  // File projection is provider- and URL-dependent. Counting none of it is the
  // only safe lower bound without evaluating MessageV2 for a concrete model.
  if (part.type === "file") return 0
  return 0
}

function compactedRenderedChars(
  parts: readonly SessionV1.Part[],
  summary: string | undefined,
  native: boolean,
  group?: string,
  channel = "assistant",
  conservativeReferences = false,
) {
  const ids = parts.map((part) => part.id)
  const fixed = parts.reduce((sum, part) => sum + compactedMemberChars(part, group), 0)
  const groupAttr = group ? ` group="${escapeAttribute(group)}" channel="${channel}"` : ""
  const attr = ids.length ? ` parts="${ids.join(" ")}"` : ""
  const trimmed = summary?.trim()
  const body = trimmed
    ? trimmed.length > COMPACTION_SUMMARY_MAX_CHARS
      ? `${trimmed.slice(0, COMPACTION_SUMMARY_MAX_CHARS)} [summary truncated]`
      : trimmed
    : native
      ? NATIVE_FALLBACK
      : COMPACTED_FALLBACK
  const references = conservativeReferences
    ? parts
        .slice(0, -1)
        .reduce(
          (total, part) =>
            total +
            insertedTextChars(
              `<compaction-ref group="${escapeAttribute(group!)}" channel="${channel}" parts="${part.id}">\n${COMPACTION_REFERENCE}\n</compaction-ref>`,
            ),
          0,
        )
    : 0
  return (
    1 +
    fixed +
    references +
    insertedTextChars(`<compaction-summary${groupAttr}${attr}>\n${body}\n</compaction-summary>`)
  )
}

function compactedMemberChars(part: SessionV1.Part, group?: string) {
  if (part.type !== "tool") return 0
  const marker = `<compacted prt="${part.id}"${group ? ` group="${escapeAttribute(group)}"` : ""}/>`.length
  const encodedMarker = marker + (group ? 6 : 4)
  if (part.state.status !== "completed") return encodedMarker
  return encodedMarker + serializedChars(part.state.input)
}

function compactedSpanChars(
  parts: readonly SessionV1.Part[],
  summary: string | undefined,
  group: string,
  channel: string,
) {
  const trimmed = summary?.trim()
  const body = trimmed
    ? trimmed.length > COMPACTION_SUMMARY_MAX_CHARS
      ? `${trimmed.slice(0, COMPACTION_SUMMARY_MAX_CHARS)} [summary truncated]`
      : trimmed
    : "No summary was recorded. Use list_context and read_part to recover this stretch."
  return insertedTextChars(
    `<compacted-span group="${escapeAttribute(group)}" channel="${channel}" from="${parts[0]!.id}" to="${parts.at(-1)!.id}" parts="${parts.length}" messages="1">\n${body}\n</compacted-span>`,
  )
}

function inferredRole(part: SessionV1.Part): SessionV1.Info["role"] {
  return part.type === "tool" || part.type === "reasoning" ? "assistant" : "user"
}

function escapeAttribute(value: string) {
  return value.replaceAll("&", "&amp;").replaceAll('"', "&quot;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
}

function serializedChars(value: unknown) {
  if (value === undefined) return 0
  return JSON.stringify(value).length
}

function removableTextChars(value: string) {
  return Math.max(insertedTextChars(value) - 2, 0)
}

function insertedTextChars(value: string) {
  return JSON.stringify(value).length
}

// Accumulator for the rendered cost of several parts at once. A run renders one
// covering note however many parts it spans, so charging every member of a group
// for the whole summary inflates the total by the size of the group. Later
// members still contribute their recovery id, tool marker, and original input.
export function charsCharger() {
  const groups = new Set<string>()
  return (part: SessionV1.Part, chars: number, role: "user" | "assistant") => {
    const group = renderedCompaction(part)?.group
    if (group === undefined) return chars
    const key = projectionChannelKey(group, projectionChannel(part, role))
    if (!groups.has(key)) {
      groups.add(key)
      return chars
    }
    const fixed = compactedMemberChars(part, group)
    return fixed + part.id.length + 1
  }
}

export function unsafeCompactionGroups(
  parts: Iterable<SessionV1.Part>,
  selected: ReadonlySet<string>,
  ambiguous: ReadonlySet<string>,
  roleByMessage: ReadonlyMap<string, SessionV1.Info["role"]>,
) {
  const groups = new Map<string, SessionV1.Part[]>()
  for (const part of parts) {
    const group = compactionOf(part)?.group
    if (!group) continue
    const key = projectionChannelKey(
      group,
      projectionChannel(part, roleByMessage.get(part.messageID) ?? inferredRole(part)),
    )
    groups.set(key, [...(groups.get(key) ?? []), part])
  }
  return new Map(
    [...groups.entries()].flatMap(([group, members]) => {
      if (!members.some((part) => selected.has(part.id))) return []
      if (members.some((part) => ambiguous.has(part.id)))
        return [[group, "prior compaction group contains ambiguous duplicate physical members"]]
      if (members.some((part) => !selected.has(part.id)))
        return [[group, "belongs to a prior compaction group; select every physical member to refold it"]]
      if (members.length > 1)
        return [
          [
            group,
            "belongs to a multi-member prior compaction group, which cannot be refolded crash-atomically without a grouped persistence event",
          ],
        ]
      return []
    }),
  )
}

// Single source of truth for "can this part be model-compacted right now",
// shared by compact_results, compact_bulk, their `select` resolvers, the
// list_context inventory, and the context-pressure manifest.
//
// Anything the model produced or consumed is fair game — including user text,
// media attachments, signed reasoning, and content that has already been
// compacted once. What stays protected is content that is structurally load
// bearing (snapshots and patches drive revert; compaction and subtask parts
// drive the run loop), still mutating (the streaming message, pending tools),
// or still in active use (the live request and the tail turns).
export function eligibility(part: SessionV1.Part, context: MessageContext): Eligible | Ineligible {
  if (!isCanonicalPartID(part.id)) return { ok: false, reason: "has a malformed part id" }
  if (isContextManagementReceipt(part))
    return { ok: false, reason: "is a context-management receipt that collapses automatically after one request" }
  if (context.current) return { ok: false, reason: "belongs to the current assistant message" }
  if (context.live) return { ok: false, reason: "belongs to the live request or a tail turn still in use" }
  const existing = compactionOf(part)
  const generation = existing ? existing.generation : 0

  if (part.type === "tool") {
    if (part.state.status !== "completed")
      return { ok: false, reason: `tool call is ${part.state.status}, not completed` }
    if (part.metadata?.providerExecuted)
      return { ok: false, reason: "is provider-executed and requires its original replay metadata" }
    // A natively pruned part has no summary to fold and no bytes left in the
    // rendered prefix, so re-compacting it would reclaim nothing.
    if (existing?.native) return { ok: false, reason: "already pruned natively, nothing left to reclaim" }
    if (existing && !existing.summary)
      return { ok: false, reason: "already compacted without a summary, nothing left to reclaim" }
    return { ok: true, kind: "tool", chars: effectiveChars(part), generation, tool: part.tool }
  }

  if (part.type === "text") {
    // Synthetic and ignored text is structural scaffolding (compaction
    // follow-ups, injected reminders); it is not content and is not addressable.
    if (part.synthetic || part.ignored) return { ok: false, reason: "is a structural or ignored text part" }
    if (context.role === "assistant" && context.summary)
      return { ok: false, reason: "belongs to the native session summary" }
    if (existing && !existing.summary)
      return { ok: false, reason: "already compacted without a summary, nothing left to reclaim" }
    if (!part.text.trim()) return { ok: false, reason: "is empty" }
    return { ok: true, kind: "text", chars: effectiveChars(part), generation }
  }

  if (part.type === "reasoning") {
    if (context.role !== "assistant") return { ok: false, reason: "is a non-assistant reasoning part" }
    if (context.summary) return { ok: false, reason: "belongs to the native session summary" }
    if (existing && !existing.summary)
      return { ok: false, reason: "already compacted without a summary, nothing left to reclaim" }
    if (!part.text.trim()) return { ok: false, reason: "is empty" }
    // Signed reasoning is compactable: the renderer drops the thinking block
    // and emits a text breadcrumb, reusing the same provider-safe path as a
    // model switch, so no block is ever sent whose signature stopped matching
    // its content. The signature itself is untouched in storage, so
    // un-compacting restores a valid block.
    return { ok: true, kind: "reasoning", chars: effectiveChars(part), generation }
  }

  if (part.type === "file") {
    // text/plain and directory files are converted to text upstream and never
    // reach the model as file parts, so excluding them here costs nothing.
    if (part.mime === "text/plain" || part.mime === "application/x-directory")
      return { ok: false, reason: "is an inlined text or directory file part" }
    if (existing && !existing.summary)
      return { ok: false, reason: "already compacted without a summary, nothing left to reclaim" }
    return { ok: true, kind: "file", chars: effectiveChars(part), generation }
  }

  return { ok: false, reason: `is a ${part.type} part, which is structural and cannot be compacted` }
}
