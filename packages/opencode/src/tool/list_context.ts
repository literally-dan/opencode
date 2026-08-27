import * as Tool from "./tool"
import DESCRIPTION from "./list_context.txt"
import {
  compactionOf,
  effectiveChars,
  eligibility,
  indexParts,
  isCompacted,
  renderedChars,
} from "@/session/compaction-pruning"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { Effect, Schema } from "effect"

const id = "list_context"
const DEFAULT_LIMIT = 80
const MAX_LIMIT = 200

export const Parameters = Schema.Struct({
  compactable_only: Schema.optional(Schema.Boolean).annotate({
    description: "Only list parts that can be compacted right now (default false: list everything in context).",
  }),
  min_chars: Schema.optional(Schema.Number).annotate({
    description: "Only list parts at least this many characters long (default 0).",
  }),
  limit: Schema.optional(Schema.Number).annotate({
    description: `Maximum number of parts to list, largest first (default ${DEFAULT_LIMIT}, maximum ${MAX_LIMIT}).`,
  }),
})

export const ListContextTool = Tool.define(
  id,
  Effect.gen(function* () {
    return {
      description: DESCRIPTION,
      parameters: Parameters,
      execute: (params: Schema.Schema.Type<typeof Parameters>, ctx: Tool.Context) =>
        Effect.gen(function* () {
          const minChars = params.min_chars ?? 0
          const limit = params.limit ?? DEFAULT_LIMIT
          if (
            !Number.isSafeInteger(minChars) ||
            minChars < 0 ||
            !Number.isSafeInteger(limit) ||
            limit < 1 ||
            limit > MAX_LIMIT
          )
            return {
              title: "Invalid context inventory request",
              metadata: { total: 0, compactable: 0, charsCompactable: 0 },
              output: `min_chars must be a non-negative integer and limit must be an integer from 1 to ${MAX_LIMIT}.`,
            }

          // Turn numbers by chronological message-id order, not array position
          // (filterCompacted may reorder messages for the model).
          const turnByMessage = new Map(
            [...new Set(ctx.messages.map((m) => m.info.id))].sort().map((mid, i) => [mid, i + 1]),
          )

          const { located } = indexParts(ctx.messages, ctx.messageID)
          const rows = ctx.messages.flatMap((msg) =>
            msg.parts.flatMap((part) => {
              const chars = effectiveChars(part)
              if (chars < minChars) return []
              const found = located.get(part.id)
              const e = found
                ? eligibility(part, found.context)
                : ({ ok: false, reason: "ambiguous duplicate id in current context" } as const)
              if (params.compactable_only && !e.ok) return []
              const generation = compactionOf(part)?.generation
              const status = e.ok
                ? generation
                  ? `compactable (already folded ${generation}x)`
                  : "compactable"
                : isCompacted(part)
                  ? "already compacted"
                  : `keep (${e.reason})`
              return [
                {
                  id: part.id.slice(0, 80),
                  turn: turnByMessage.get(msg.info.id) ?? 0,
                  role: msg.info.role,
                  label: part.type === "tool" ? `tool:${part.tool.slice(0, 80).replace(/\s+/g, " ")}` : part.type,
                  chars,
                  part,
                  compactable: e.ok,
                  status,
                  preview: preview(part),
                },
              ]
            }),
          )

          const compactable = rows.filter((r) => r.compactable)
          const charsCompactable = renderedChars(
            compactable.map((row) => row.part),
            new Map(ctx.messages.map((message) => [message.info.id, message.info.role])),
          )
          const shown = rows.toSorted((a, b) => b.chars - a.chars).slice(0, limit)

          const header = `Context inventory for ${ctx.sessionID} — ${rows.length} part(s) listed, ${compactable.length} compactable (~${charsCompactable.toLocaleString()} chars). Pass ids to compact_results, or use its \`select\` filters. Largest first:`
          const lines = shown.map(
            (r) =>
              `- ${r.id} · turn ${r.turn} · ${r.role} ${r.label} · ${r.chars.toLocaleString()} chars · ${r.status} · ${JSON.stringify(r.preview)}`,
          )
          if (rows.length > shown.length)
            lines.push(`… ${rows.length - shown.length} more not shown (raise \`limit\`).`)

          return {
            title: `${rows.length} parts, ${compactable.length} compactable`,
            metadata: { total: rows.length, compactable: compactable.length, charsCompactable },
            output: [header, "", ...lines].join("\n"),
          }
        }).pipe(Effect.orDie),
    }
  }),
)

function preview(part: SessionV1.Part) {
  const compacted = compactionOf(part)
  const text = compacted
    ? (compacted.summary ?? "Compacted content; use read_part to recover it verbatim.")
    : part.type === "tool"
      ? part.state.status === "completed"
        ? part.state.output
        : part.state.status
      : part.type === "text" || part.type === "reasoning"
        ? part.text
        : part.type === "file"
          ? (part.filename ?? part.url)
          : part.type
  // Slice before normalizing so we never scan a multi-MB output/data-URL.
  return text.slice(0, 200).replace(/\s+/g, " ").trim().slice(0, 80)
}
