import * as Tool from "./tool"
import DESCRIPTION from "./compact_results.txt"
import { Session } from "@/session/session"
import {
  COMPACTION_SUMMARY_MAX_CHARS,
  compactionRenderedChars,
  compactionOf,
  compactionSavings,
  eligibility,
  indexParts,
  isCanonicalPartID,
  projectionChannel,
  projectionChannelKey,
  renderedChars,
  setCompaction,
  unsafeCompactionGroups,
  withCompactionLock,
} from "@/session/compaction-pruning"
import type { Located } from "@/session/compaction-pruning"
import { Cause, Effect, Exit, Schema } from "effect"

const id = "compact_results"
const MAX_SELECTED_PARTS = 200
const MAX_TOOL_FILTERS = 100
const MAX_TOOL_NAME_CHARS = 128

const Selector = Schema.Struct({
  types: Schema.optional(Schema.mutable(Schema.Array(Schema.Literals(["tool", "text", "reasoning", "file"])))).annotate(
    {
      description: "Restrict to these part kinds. Omit to match all four.",
    },
  ),
  tools: Schema.optional(Schema.mutable(Schema.Array(Schema.String))).annotate({
    description: "Restrict tool parts to these tool names (e.g. `read`, `bash`, `grep`).",
  }),
  min_chars: Schema.optional(Schema.Number).annotate({
    description: "Only match parts at least this many characters long.",
  }),
  before: Schema.optional(Schema.String).annotate({
    description: "Only match parts older than this canonical `prt_…` id (part ids sort chronologically).",
  }),
  after: Schema.optional(Schema.String).annotate({
    description: "Only match parts newer than this `prt_…` id.",
  }),
  keep_last: Schema.optional(Schema.Number).annotate({
    description: "Exclude a non-negative integer number of the most recent matches.",
  }),
})

export const Parameters = Schema.Struct({
  part_ids: Schema.optional(Schema.mutable(Schema.Array(Schema.String))).annotate({
    description: `One or more canonical \`prt_…\` ids to compact (maximum ${MAX_SELECTED_PARTS} unique ids after duplicate removal). Duplicates are ignored. An empty list selects nothing. Unioned with \`select\`.`,
  }),
  select: Schema.optional(Selector).annotate({
    description:
      "Compact eligible parts matching every supplied filter. An empty selector is rejected; empty `types` or `tools` arrays match nothing. Ineligible/current parts and ambiguous duplicate ids are never selected.",
  }),
  summary: Schema.String.annotate({
    description: `Required. One note (maximum ${COMPACTION_SUMMARY_MAX_CHARS} characters) covering everything this call compacts, rendered once for the whole run rather than once per part. Preserve what was done and what was concluded — paths, identifiers, decisions, numbers, errors — because the parts themselves are leaving your context.`,
  }),
})

type Outcome =
  | { partID: string; status: "compacted"; label: string }
  | { partID: string; status: "skipped"; reason: string }

export const CompactResultsTool = Tool.define(
  id,
  Effect.gen(function* () {
    const sessions = yield* Session.Service

    return {
      description: DESCRIPTION,
      parameters: Parameters,
      execute: (params: Schema.Schema.Type<typeof Parameters>, ctx: Tool.Context) =>
        Effect.gen(function* () {
          const invalid = validateInput(params)
          if (invalid) return invalidResult(invalid)

          return yield* withCompactionLock(
            ctx.sessionID,
            ctx.abort,
            Effect.gen(function* () {
              const messages = yield* sessions.messages({ sessionID: ctx.sessionID })
              const { located, ambiguous } = indexParts(messages, ctx.messageID)
              const roles = new Map(messages.map((message) => [message.info.id, message.info.role]))
              const selected = params.select ? resolveSelection(params.select, located) : []
              const ids = [...new Set([...(params.part_ids ?? []), ...selected])]
              if (ids.length > MAX_SELECTED_PARTS)
                return invalidResult(
                  `Selection resolved to ${ids.length} parts after duplicate removal; refine it to at most ${MAX_SELECTED_PARTS} parts per call.`,
                )
              if (ids.length === 0)
                return {
                  title: "Nothing to compact",
                  metadata: { compacted: 0, skipped: 0, charsFreed: 0 },
                  output:
                    "No parts matched. Pass `part_ids` and/or a `select` filter (use `list_context` to see ids and sizes).",
                }

              const outcomes: Outcome[] = []
              const eligible = new Set(
                ids.filter((partID) => {
                  const found = located.get(partID)
                  return found ? eligibility(found.part, found.context).ok : false
                }),
              )
              const unsafe = unsafeCompactionGroups(
                messages.flatMap((message) => message.parts),
                eligible,
                ambiguous,
                roles,
              )
              const candidates: {
                partID: string
                found: Located
                eligible: Extract<ReturnType<typeof eligibility>, { ok: true }>
              }[] = []
              for (const partID of ids) {
                const found = located.get(partID)
                if (!found) {
                  outcomes.push({
                    partID,
                    status: "skipped",
                    reason: ambiguous.has(partID)
                      ? "ambiguous duplicate id in current context"
                      : "not found in current context",
                  })
                  continue
                }
                const allowed = eligibility(found.part, found.context)
                if (!allowed.ok) {
                  outcomes.push({ partID, status: "skipped", reason: allowed.reason })
                  continue
                }
                const prior = compactionOf(found.part)?.group
                const reason = prior
                  ? unsafe.get(projectionChannelKey(prior, projectionChannel(found.part, found.context.role)))
                  : undefined
                if (reason) {
                  outcomes.push({ partID, status: "skipped", reason })
                  continue
                }
                candidates.push({ partID, found, eligible: allowed })
              }

              const summary = params.summary.trim()
              if (candidates.length > 0) {
                const originals = candidates.map((candidate) => candidate.found.part)
                const net = compactionSavings(originals, summary, roles)
                const ordered = candidates.toSorted(
                  (a, b) =>
                    compactionSavings([b.found.part], summary, roles) -
                    compactionSavings([a.found.part], summary, roles),
                )
                const unsafePrefix = ordered.findIndex(
                  (_, index) =>
                    compactionSavings(
                      ordered.slice(0, index + 1).map((candidate) => candidate.found.part),
                      summary,
                      roles,
                    ) <= 0,
                )
                const reason =
                  net <= 0
                    ? `would not reduce rendered context (estimated net ${net.toLocaleString()} chars); keep it or write a shorter summary`
                    : unsafePrefix >= 0
                      ? `would create a non-saving partial state after write ${unsafePrefix + 1}, but grouped part writes are not crash-atomic`
                      : undefined
                if (reason) {
                  for (const candidate of candidates)
                    outcomes.push({ partID: candidate.partID, status: "skipped", reason })
                } else {
                  const generation = Math.max(...candidates.map((candidate) => candidate.eligible.generation)) + 1
                  const compactedAt = Date.now()
                  const compactionGroup = crypto.randomUUID()
                  let stopped: string | undefined
                  const persisted: typeof ordered = []
                  for (const [index, candidate] of ordered.entries()) {
                    const part = structuredClone(candidate.found.part)
                    setCompaction(part, {
                      at: compactedAt,
                      group: compactionGroup,
                      summary,
                      generation,
                    })
                    const result = yield* Effect.exit(sessions.updatePart(part))
                    if (Exit.isFailure(result)) {
                      if (Cause.hasInterruptsOnly(result.cause)) return yield* Effect.interrupt
                      stopped = Cause.pretty(result.cause)
                      outcomes.push({ partID: candidate.partID, status: "skipped", reason: "persistence failed" })
                      for (const remaining of ordered.slice(index + 1))
                        outcomes.push({
                          partID: remaining.partID,
                          status: "skipped",
                          reason: "not attempted after persistence failure",
                        })
                      break
                    }
                    persisted.push(candidate)
                    outcomes.push({ partID: candidate.partID, status: "compacted", label: candidate.eligible.kind })
                  }

                  const compacted = outcomes.filter((outcome) => outcome.status === "compacted")
                  const freed = persisted.length
                    ? renderedChars(
                        persisted.map((candidate) => candidate.found.part),
                        roles,
                      ) -
                      compactionRenderedChars(
                        persisted.map((candidate) => candidate.found.part),
                        summary,
                        roles,
                      )
                    : 0
                  const lines = [
                    stopped
                      ? `Persisted ${compacted.length} of ${candidates.length} compactions before a persistence failure (${freed.toLocaleString()} net chars freed). Unpersisted parts remain unchanged.`
                      : `Compacted ${compacted.length} of ${ids.length} parts (${freed.toLocaleString()} net chars freed). Effective starting next assistant step.`,
                    ...outcomes.map((outcome) =>
                      outcome.status === "compacted"
                        ? `- ${outcome.partID} (${outcome.label}) → compacted`
                        : `- ${outcome.partID} → skipped: ${outcome.reason}`,
                    ),
                  ]
                  return {
                    title: stopped
                      ? compacted.length
                        ? `Partially compacted ${compacted.length} parts`
                        : "Compaction persistence failed"
                      : compacted.length === 1
                        ? "Compacted 1 part"
                        : `Compacted ${compacted.length} parts`,
                    metadata: {
                      compacted: compacted.length,
                      skipped: outcomes.length - compacted.length,
                      charsFreed: freed,
                      ...(stopped ? { partial: compacted.length > 0, persistenceError: stopped } : {}),
                    },
                    output: lines.join("\n"),
                  }
                }
              }

              return {
                title: "Compacted 0 parts",
                metadata: { compacted: 0, skipped: outcomes.length, charsFreed: 0 },
                output: [
                  `Compacted 0 of ${ids.length} parts.`,
                  ...outcomes.map((outcome) =>
                    outcome.status === "compacted"
                      ? `- ${outcome.partID} (${outcome.label}) → compacted`
                      : `- ${outcome.partID} → skipped: ${outcome.reason}`,
                  ),
                ].join("\n"),
              }
            }),
          )
        }).pipe(Effect.orDie),
    }
  }),
)

// Resolve a `select` filter into a chronological list of part ids. Part ids are
// ascending, so a plain string sort/compare is chronological order.
function resolveSelection(sel: Schema.Schema.Type<typeof Selector>, located: Map<string, Located>) {
  const matched = [...located.values()]
    .flatMap(({ part, context }) => {
      const e = eligibility(part, context)
      if (!e.ok) return []
      if (sel.types && !sel.types.includes(e.kind)) return []
      if (sel.tools && (e.kind !== "tool" || !e.tool || !sel.tools.includes(e.tool))) return []
      if (sel.min_chars !== undefined && e.chars < sel.min_chars) return []
      if (sel.before !== undefined && !(part.id < sel.before)) return []
      if (sel.after !== undefined && !(part.id > sel.after)) return []
      return [part.id]
    })
    .sort()
  if (sel.keep_last !== undefined && sel.keep_last > 0)
    return matched.slice(0, Math.max(0, matched.length - sel.keep_last))
  return matched
}

function validateInput(params: Schema.Schema.Type<typeof Parameters>) {
  if (!params.summary.trim())
    return "A summary is required: every compacted part must be covered by a note explaining what it contained."
  if (params.summary.trim().length > COMPACTION_SUMMARY_MAX_CHARS)
    return `Summary exceeds the ${COMPACTION_SUMMARY_MAX_CHARS.toLocaleString()} character limit.`
  const malformed = params.part_ids?.find((partID) => !isCanonicalPartID(partID))
  if (malformed) return `Malformed part id: ${JSON.stringify(malformed.slice(0, 80))}.`
  if (!params.select) return undefined
  if (Object.values(params.select).every((value) => value === undefined))
    return "An empty select object is not allowed; supply at least one filter."
  if (params.select.before && !isCanonicalPartID(params.select.before))
    return "select.before must be a canonical part id."
  if (params.select.after && !isCanonicalPartID(params.select.after)) return "select.after must be a canonical part id."
  if (params.select.before && params.select.after && params.select.after >= params.select.before)
    return "select.after must be older than select.before."
  if ((params.select.types?.length ?? 0) > 4) return "select.types accepts at most four entries."
  if (!nonNegativeInteger(params.select.min_chars)) return "select.min_chars must be a non-negative integer."
  if (!nonNegativeInteger(params.select.keep_last)) return "select.keep_last must be a non-negative integer."
  if ((params.select.tools?.length ?? 0) > MAX_TOOL_FILTERS)
    return `select.tools accepts at most ${MAX_TOOL_FILTERS} names.`
  if (params.select.tools?.some((tool) => tool.length === 0 || tool.length > MAX_TOOL_NAME_CHARS))
    return `Each select.tools name must contain 1-${MAX_TOOL_NAME_CHARS} characters.`
  return undefined
}

function nonNegativeInteger(value: number | undefined) {
  return value === undefined || (Number.isSafeInteger(value) && value >= 0)
}

function invalidResult(reason: string) {
  return {
    title: "Invalid compaction request",
    metadata: { compacted: 0, skipped: 0, charsFreed: 0 },
    output: reason,
  }
}
