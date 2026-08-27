import path from "path"
import { Effect } from "effect"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { SessionID } from "./schema"
import { Session } from "./session"
import { compactionOf, setCompaction, withCompactionLock } from "./compaction-pruning"

// Tools whose output is a snapshot of file contents. Only these are collapsed:
// an edit or write result is a diff or a confirmation, which is a record of
// what happened rather than a claim about current state, and is small anyway.
const SNAPSHOT_TOOLS = new Set(["read"])
// Tools that invalidate every earlier snapshot of the files they touch.
const MUTATING_TOOLS = new Set(["write", "edit", "apply_patch"])

// Below this a collapse costs a durable write and a marker to save almost
// nothing, so it is not worth the churn.
const MIN_CHARS = 1_000

type Entry =
  | {
      kind: "snapshot"
      part: SessionV1.ToolPart
      file: string
      start: number
      end: number
      // Whether the read returned everything from `start` to the end of the
      // file. A truncated read covers only the window it actually produced.
      complete: boolean
    }
  | { kind: "mutation"; part: SessionV1.ToolPart; files: Set<string> }

// Collapse file reads that a later step has provably superseded.
//
// This is the one compaction the runtime performs on its own, because it needs
// no judgement: if a file was rewritten after it was read, the earlier read no
// longer describes anything real. Leaving it in context is not just wasteful,
// it is wrong — the model will happily reason about, and edit against, content
// that no longer matches disk. The note points at the newer touch, and the
// original bytes stay recoverable through read_part like any other compaction.
// Takes the already-bound session service and instance directory rather than
// requiring them, so the caller's run loop stays free of extra requirements.
export const collapse = Effect.fn("session.superseded.collapse")(function* (input: {
  sessionID: SessionID
  sessions: Session.Interface
  directory: string
}) {
  return yield* withCompactionLock(
    input.sessionID,
    undefined,
    Effect.gen(function* () {
      const messages = yield* input.sessions.messages({ sessionID: input.sessionID })

      const entries: Entry[] = []
      for (const msg of messages) {
        // Relative paths resolve against the directory the turn actually ran in, so
        // a session whose cwd moved cannot conflate two different files that happen
        // to share a relative path.
        const directory = (msg.info.role === "assistant" ? msg.info.path?.cwd : undefined) ?? input.directory
        for (const part of msg.parts) {
          if (part.type !== "tool" || part.state.status !== "completed") continue
          if (MUTATING_TOOLS.has(part.tool)) {
            const files = mutatedFiles(part.state, directory)
            if (files.size) entries.push({ kind: "mutation", part, files })
            continue
          }
          if (!SNAPSHOT_TOOLS.has(part.tool)) continue
          const window = readWindow(part.state, directory)
          if (window) entries.push({ kind: "snapshot", part, ...window })
        }
      }

      const collapsed: SessionV1.ToolPart[] = []
      for (const [position, entry] of entries.entries()) {
        if (entry.kind !== "snapshot") continue
        if (compactionOf(entry.part)) continue
        if (entry.part.state.status !== "completed" || entry.part.state.output.length < MIN_CHARS) continue
        const supersededBy = entries.find((later, laterPosition) => {
          if (laterPosition <= position) return false
          if (later.kind === "mutation") return later.files.has(entry.file)
          // A later read only retires an earlier one when it demonstrably returned a
          // superset of those lines. An untruncated read reaches the end of the
          // file, so starting at or before the earlier window is enough.
          if (later.file !== entry.file || later.start > entry.start) return false
          return later.complete || later.end >= entry.end
        })
        if (!supersededBy) continue
        collapsed.push(entry.part)
        setCompaction(entry.part, {
          at: Date.now(),
          // One group per collapse: each note names a different newer touch, and a
          // shared group would render only the first of them.
          group: `superseded:${entry.part.id}`,
          generation: 1,
          summary: [
            `Stale snapshot of ${entry.file}.`,
            supersededBy.kind === "mutation"
              ? `The file was modified afterwards by \`${supersededBy.part.tool}\` (${supersededBy.part.id}), so this copy no longer matches disk.`
              : `A later read (${supersededBy.part.id}) returned these lines again, so this copy is redundant.`,
            `Use read_part to recover it verbatim, or read the file again for current content.`,
          ].join(" "),
        })
        yield* input.sessions.updatePart(entry.part)
      }

      return collapsed.length
    }),
  )
})

// Paths reach the tools either absolute or relative to the run directory, and
// the same file can arrive both ways, so normalise before comparing.
function resolveFile(input: unknown, directory: string) {
  if (typeof input !== "object" || input === null) return undefined
  const filePath = (input as { filePath?: unknown }).filePath
  if (typeof filePath !== "string" || !filePath) return undefined
  return path.resolve(directory, filePath)
}

function mutatedFiles(state: Extract<SessionV1.ToolPart["state"], { status: "completed" }>, directory: string) {
  const files = new Set<string>()
  const direct = resolveFile(state.input, directory)
  if (direct) files.add(direct)
  // `apply_patch` is handed a patch rather than a path and can rewrite several
  // files at once, so the only record of what it touched is its result.
  const listed = (state.metadata as { files?: unknown } | undefined)?.files
  if (Array.isArray(listed))
    for (const entry of listed) {
      if (typeof entry !== "object" || entry === null) continue
      for (const key of ["filePath", "movePath"]) {
        const value = (entry as Record<string, unknown>)[key]
        if (typeof value === "string" && value) files.add(path.resolve(directory, value))
      }
    }
  return files
}

// The window a read actually returned, not the one it asked for. `read` caps
// output at 2,000 lines and 50KB, so a request for a whole file routinely comes
// back with a fraction of it; trusting the request would let a truncated read
// retire a complete one. Anything that does not report a line window — a
// directory listing, an image — is left alone.
function readWindow(state: Extract<SessionV1.ToolPart["state"], { status: "completed" }>, directory: string) {
  const file = resolveFile(state.input, directory)
  if (!file) return undefined
  const display = state.metadata?.display
  if (typeof display !== "object" || display === null) return undefined
  const window = display as { lineStart?: unknown; lineEnd?: unknown; truncated?: unknown }
  if (typeof window.lineStart !== "number" || typeof window.lineEnd !== "number") return undefined
  return { file, start: window.lineStart, end: window.lineEnd, complete: window.truncated !== true }
}

export * as Superseded from "./superseded"
