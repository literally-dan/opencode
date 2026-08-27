import * as Tool from "./tool"
import DESCRIPTION from "./read_part.txt"
import { Session } from "@/session/session"
import { SessionID, PartID } from "../session/schema"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { compactionOf, isCanonicalPartID } from "@/session/compaction-pruning"
import { isCanonicalSessionID, isReadableSession } from "./history-scope"
import { Effect, Schema } from "effect"

const id = "read_part"

export const Parameters = Schema.Struct({
  part_id: Schema.String.annotate({
    description: "The `prt_…` ID of the part to read.",
  }),
  session_id: Schema.optional(Schema.String).annotate({
    description: "Optional session ID; defaults to the current session.",
  }),
})

type Metadata = {
  sessionID: string
  partID: string
  found: boolean
  partType?: string
  tool?: string
  inline?: boolean
}

export const ReadPartTool = Tool.define<typeof Parameters, Metadata, Session.Service>(
  id,
  Effect.gen(function* () {
    const sessions = yield* Session.Service

    return {
      description: DESCRIPTION,
      parameters: Parameters,
      execute: (params: Schema.Schema.Type<typeof Parameters>, ctx: Tool.Context<Metadata>) =>
        Effect.gen(function* () {
          if (params.session_id && !isCanonicalSessionID(params.session_id)) {
            return {
              title: "Invalid session ID",
              metadata: { sessionID: ctx.sessionID, partID: params.part_id, found: false },
              output: "session_id must be a canonical `ses_…` id.",
            }
          }
          const sessionID = params.session_id ? SessionID.make(params.session_id) : ctx.sessionID
          if (params.session_id && !(yield* isReadableSession(sessions, sessionID, ctx.sessionID))) {
            return {
              title: `Session not accessible: ${sessionID}`,
              metadata: { sessionID, partID: params.part_id, found: false },
              output: `Session ${sessionID} is not the current session or one of its ancestors; cross-session reads are not allowed.`,
            }
          }
          if (!isCanonicalPartID(params.part_id)) {
            return {
              title: "Invalid part ID",
              metadata: { sessionID, partID: params.part_id, found: false },
              output: "part_id must be a canonical `prt_…` id.",
            }
          }
          const part = yield* sessions
            .findPart({ sessionID, partID: PartID.make(params.part_id) })
            .pipe(Effect.catch(() => Effect.succeed(undefined)))
          if (!part) {
            return {
              title: `Part not found: ${params.part_id}`,
              metadata: { sessionID, partID: params.part_id, found: false },
              output: `No part with id ${params.part_id} in session ${sessionID}.`,
            }
          }
          return formatPart(part, sessionID)
        }).pipe(Effect.orDie),
    }
  }),
)

// An unknown part type serialises whole, and a file part's url is routinely a
// multi-MB base64 data URL. Media is handed back as a real attachment instead
// (see below); anything left as text gets clamped so recovery cannot blow the
// context this tool exists to protect.
const MAX_OPAQUE_CHARS = 4_000

function clamp(value: string) {
  if (value.length <= MAX_OPAQUE_CHARS) return value
  return `${value.slice(0, MAX_OPAQUE_CHARS)}… [${value.length - MAX_OPAQUE_CHARS} more characters omitted]`
}

// A compacted part still holds its original bytes, so describe what was folded
// and how thin the summary has become. Generation climbs on every re-compaction.
function compactionNote(part: SessionV1.Part) {
  const record = compactionOf(part)
  if (!record) return []
  if (record.native)
    return ["Note: pruned automatically for age; the output below is the original, recovered verbatim."]
  return [
    `Note: compacted${record.generation > 1 ? ` and re-folded ${record.generation - 1} time${record.generation === 2 ? "" : "s"}` : ""}; the content below is the original, recovered verbatim.`,
    ...(record.summary ? [`Summary shown in its place: ${record.summary}`] : []),
  ]
}

function formatPart(part: SessionV1.Part, sessionID: string) {
  const base = { sessionID, partID: part.id, found: true as const, partType: part.type }
  if (part.type === "tool") {
    const state = part.state
    const lines: string[] = [`Tool: ${part.tool}`, `Call ID: ${part.callID}`, `Status: ${state.status}`]
    if (state.status === "completed") {
      lines.push(...compactionNote(part))
      lines.push("", "Input:", JSON.stringify(state.input, null, 2), "", "Output:", state.output)
      const referenced = state.attachments?.filter((item) => !item.url.startsWith("data:")) ?? []
      if (referenced.length)
        lines.push(
          "",
          "Attachments:",
          ...referenced.map((item) => `- ${item.filename ?? "unnamed"} (${item.mime}): ${item.url}`),
        )
      const inline = state.attachments?.filter((item) => item.url.startsWith("data:")) ?? []
      return {
        title: `Tool part ${part.tool} (${part.callID})`,
        metadata: { ...base, tool: part.tool },
        output: lines.join("\n"),
        // Only inline payloads survive the tool-result attachment renderer.
        // Referenced URLs are included in text above so they remain recoverable.
        ...(inline.length ? { attachments: inline.map(attachment) } : {}),
      }
    }
    if (state.status === "error") {
      lines.push("", "Input:", JSON.stringify(state.input, null, 2), "", "Error:", state.error)
    } else if (state.status === "running") {
      lines.push("", "Input:", JSON.stringify(state.input, null, 2))
    }
    return {
      title: `Tool part ${part.tool} (${part.callID})`,
      metadata: { ...base, tool: part.tool },
      output: lines.join("\n"),
    }
  }
  if (part.type === "text" || part.type === "reasoning") {
    return {
      title: `${part.type} part`,
      metadata: base,
      output: [...compactionNote(part), ...(compactionOf(part) ? [""] : []), part.text ?? ""].join("\n"),
    }
  }
  if (part.type === "file") {
    // Returning the file as an attachment means the model sees the image again
    // rather than a description of it, so a compacted attachment is genuinely
    // recoverable. Only inline payloads survive the round trip though — the
    // renderer drops any attachment that is not a data URL — so a referenced
    // file has to fall back to its location rather than promise a body that
    // will not arrive.
    const inline = part.url.startsWith("data:")
    return {
      title: `File part`,
      metadata: { ...base, inline },
      output: [
        ...compactionNote(part),
        `Filename: ${part.filename ?? ""}`,
        `Mime: ${part.mime}`,
        inline ? "Returned as an attachment below." : `URL: ${clamp(part.url)}`,
      ].join("\n"),
      ...(inline ? { attachments: [attachment(part)] } : {}),
    }
  }
  return {
    title: `${part.type} part`,
    metadata: base,
    output: clamp(JSON.stringify(part, null, 2)),
  }
}

function attachment(part: SessionV1.FilePart) {
  return {
    type: "file" as const,
    mime: part.mime,
    url: part.url,
    ...(part.filename ? { filename: part.filename } : {}),
    ...(part.source ? { source: part.source } : {}),
  }
}
