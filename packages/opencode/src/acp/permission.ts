import type {
  AgentSideConnection,
  PermissionOption,
  RequestPermissionResponse,
  ToolCallContent,
  ToolCallLocation,
  ToolCallUpdate,
} from "@agentclientprotocol/sdk"
import type { Event, OpencodeClient } from "@opencode-ai/sdk/v2"
import { applyPatch } from "diff"
import { exists, readText } from "@/util/filesystem"
import type { ACPSession } from "./session"
import { pendingToolCall, toLocations, type ToolInput } from "./tool"
import { SessionAncestry } from "@/session/ancestry"
import { replyPermission } from "@/session/permission-reply"
import { TaskProvenance } from "@/session/task-provenance"
import { Effect } from "effect"

type PermissionEvent = Extract<Event, { type: "permission.asked" }>
type Reply = "once" | "always" | "reject"
type Connection = Partial<Pick<AgentSideConnection, "requestPermission" | "writeTextFile">>

const permissionOptions: PermissionOption[] = [
  { optionId: "once", kind: "allow_once", name: "Allow once" },
  { optionId: "always", kind: "allow_always", name: "Always allow" },
  { optionId: "reject", kind: "reject_once", name: "Reject" },
]

type Ownership = SessionAncestry.Ownership<ACPSession.Info>

export class Handler {
  private readonly routing = new Map<string, Promise<void>>()
  private readonly queues = new Map<string, Promise<void>>()
  private readonly pending = new Map<string, { sessionID: string; controller: AbortController }>()
  private stopped = false
  constructor(
    private readonly input: {
      sdk: OpencodeClient
      connection: Connection
      session: ACPSession.Interface
    },
  ) {}

  forgetSession(sessionID: string) {
    for (const pending of this.pending.values()) {
      if (pending.sessionID === sessionID) pending.controller.abort()
    }
  }

  stop() {
    this.stopped = true
    for (const pending of this.pending.values()) pending.controller.abort()
    this.pending.clear()
    this.routing.clear()
    this.queues.clear()
  }

  handle(event: PermissionEvent, directory?: string) {
    const permission = event.properties
    if (this.stopped || this.pending.has(permission.id)) return
    const controller = new AbortController()
    this.pending.set(permission.id, { sessionID: permission.sessionID, controller })
    const previous = this.routing.get(permission.sessionID) ?? Promise.resolve()
    const next = previous
      .then(() => (controller.signal.aborted ? undefined : this.route(event, controller.signal, directory)))
      .catch((error) =>
        Effect.runSync(
          Effect.logError("failed to route permission request", {
            requestID: permission.id,
            sessionID: permission.sessionID,
            error,
          }),
        ),
      )
      .finally(() => {
        if (this.routing.get(permission.sessionID) === next) this.routing.delete(permission.sessionID)
        if (this.pending.get(permission.id)?.controller === controller) this.pending.delete(permission.id)
      })
    this.routing.set(permission.sessionID, next)
  }

  private async route(event: PermissionEvent, signal: AbortSignal, directory?: string) {
    const permission = event.properties
    const ownership = await this.resolveOwnership(permission, signal)
    if (ownership.type === "foreign" || ownership.type === "cancelled") return
    if (ownership.type === "exhausted") {
      // A cycle or a deleted parent never resolves. Without a reply the nested
      // tool call waits on an untimed deferred forever, so answer it.
      Effect.runSync(
        Effect.logError("permission ancestry unresolvable; rejecting", {
          requestID: permission.id,
          sessionID: permission.sessionID,
          error: ownership.error,
        }),
      )
      await this.reply(permission.id, "reject", signal, directory).catch(() => {})
      return
    }
    const session = ownership.value

    // The ACP client displays nested prompts against the managed ancestor.
    // Queue by that same ID so sibling leaves cannot open competing prompts
    // for one displayed session.
    const previous = this.queues.get(session.id) ?? Promise.resolve()
    const next = previous
      .then(() => (signal.aborted ? undefined : this.process(event, session, signal)))
      .catch((error) => (signal.aborted ? undefined : this.rejectOwnedPermission(event, error, session.cwd, signal)))
      .finally(() => {
        if (this.queues.get(session.id) === next) {
          this.queues.delete(session.id)
        }
      })
    this.queues.set(session.id, next)
    await next
  }

  private resolveOwnership(permission: PermissionEvent["properties"], signal: AbortSignal) {
    return SessionAncestry.resolve({
      signal,
      lookup: (current) => this.lookupOwnership(permission.sessionID, current),
      onRetry: (retry) => {
        if (retry.attempt !== 1 && retry.attempt % 5 !== 0) return
        Effect.runSync(
          Effect.logWarning("permission ownership unresolved; retrying ancestry lookup", {
            requestID: permission.id,
            sessionID: permission.sessionID,
            attempt: retry.attempt,
            retryIn: retry.retryIn,
            error: retry.error,
          }),
        )
      },
    })
  }

  private async lookupOwnership(sessionID: string, signal: AbortSignal): Promise<Ownership> {
    try {
      const session = await this.resolveManagedAncestor(sessionID, signal)
      return session ? { type: "owned", value: session } : { type: "foreign" }
    } catch (error) {
      return { type: "unknown", error }
    }
  }

  // Climb the parent chain from `sessionID` and return the first ACP
  // session the client created (root or load). Used to route permission
  // prompts for nested subagent sessions — the ACP client never sees the
  // leaf id, so we report against the managed ancestor instead. Returns
  // undefined if the chain reaches the root without hitting a managed
  // session, in which case the prompt is foreign (e.g. another ACP
  // agent's session).
  private async resolveManagedAncestor(sessionID: string, signal: AbortSignal) {
    const direct = await Effect.runPromise(this.input.session.tryGet(sessionID))
    if (direct) return direct
    let current = await TaskProvenance.getSession(this.input.sdk, sessionID, signal)
    const seen = new Set<string>()
    while (!seen.has(current.id)) {
      seen.add(current.id)
      const parent = await TaskProvenance.parent(this.input.sdk, current, signal)
      if (!parent) return undefined
      const managed = await Effect.runPromise(this.input.session.tryGet(parent.id))
      if (managed) return managed
      current = parent
    }
    throw new Error(`parent chain cycle detected for ${sessionID}: ${Array.from(seen).join(" -> ")}`)
  }

  private async process(event: PermissionEvent, session: ACPSession.Info, signal: AbortSignal) {
    const permission = event.properties
    // Permissions raised inside a nested subagent carry the deepest
    // session's ID, which the ACP client never explicitly created. Walk
    // up to the managed ancestor so the prompt is routed somewhere the
    // client knows about; otherwise the deferred on the server never
    // resolves and the nested tool call (including MCP calls) hangs.
    if (!this.input.connection.requestPermission) {
      await this.reply(permission.id, "reject", signal, session.cwd)
      return
    }

    const result = await this.input.connection.requestPermission({
      // Route to the managed ancestor's id (resolved above), not the
      // leaf subagent id the client never created.
      sessionId: session.id,
      toolCall: await permissionToolCall({
        toolCallId: permission.tool?.callID ?? permission.id,
        toolName: permission.permission,
        input: permission.metadata,
      }),
      options: permissionOptions,
    })

    const reply = selectedReply(result)
    if (reply !== "once" && reply !== "always") {
      await this.reply(permission.id, "reject", signal, session.cwd)
      return
    }

    if (permission.permission === "edit") {
      await this.writeProposedEdit(session.id, permission.metadata).catch(() => {})
    }

    await this.reply(permission.id, reply, signal, session.cwd)
  }

  private async rejectOwnedPermission(event: PermissionEvent, error: unknown, directory: string, signal: AbortSignal) {
    const permission = event.properties
    Effect.runSync(
      Effect.logError("failed to process owned permission request", {
        requestID: permission.id,
        sessionID: permission.sessionID,
        error,
      }),
    )
    await this.reply(permission.id, "reject", signal, directory).catch((replyError) => {
      Effect.runSync(
        Effect.logError("failed to reject owned permission request", {
          requestID: permission.id,
          sessionID: permission.sessionID,
          error: replyError,
        }),
      )
    })
  }

  private reply(requestID: string, reply: Reply, signal: AbortSignal, directory?: string) {
    // A permission dialog already answered by the user still gets one bounded
    // delivery attempt during shutdown; queued dialogs remain cancelled.
    const finishing = signal.aborted
    return replyPermission(this.input.sdk, {
      requestID,
      reply,
      directory,
      signal: finishing ? AbortSignal.timeout(10_000) : signal,
      maxAttempts: finishing ? 1 : undefined,
      onRetry: (retry) => {
        if (retry.attempt !== 1 && retry.attempt % 5 !== 0) return
        Effect.runSync(
          Effect.logWarning("permission reply failed; retrying", {
            requestID,
            reply,
            attempt: retry.attempt,
            retryIn: retry.retryIn,
            error: retry.error,
          }),
        )
      },
    })
  }

  private async writeProposedEdit(sessionId: string, metadata: ToolInput) {
    const filepath = stringValue(metadata.filepath)
    const diff = stringValue(metadata.diff)
    if (!filepath || !diff || !this.input.connection.writeTextFile) return

    const content = (await exists(filepath)) ? await readText(filepath) : ""
    const next = applyPatch(content, diff)
    if (next === false) {
      return
    }

    void this.input.connection.writeTextFile({
      sessionId,
      path: filepath,
      content: next,
    })
  }
}

async function permissionToolCall(input: {
  readonly toolCallId: string
  readonly toolName: string
  readonly input: ToolInput
}): Promise<ToolCallUpdate> {
  const toolCall = pendingToolCall({
    toolCallId: input.toolCallId,
    toolName: input.toolName,
    state: {
      input: input.input,
      title: permissionTitle(input.toolName, input.input),
    },
  })
  const content = await permissionContent(input.toolName, input.input)
  return {
    ...toolCall,
    locations: permissionLocations(input.toolName, input.input),
    ...(content.length ? { content } : {}),
  }
}

function permissionTitle(toolName: string, input: ToolInput) {
  const tool = toolName.toLocaleLowerCase()
  switch (tool) {
    case "external_directory":
      return stringValue(input.description) ?? stringValue(input.command) ?? stringValue(input.parentDir)

    case "webfetch":
      return stringValue(input.url)

    case "websearch":
      return stringValue(input.query)

    case "grep":
    case "glob":
      return stringValue(input.pattern)

    case "read":
    case "edit":
    case "write":
      return editTitle(input)

    default:
      return undefined
  }
}

function editTitle(input: ToolInput) {
  const files = fileMetadata(input)
  if (files.length === 1) return files[0]?.relativePath ?? files[0]?.filePath
  if (files.length > 1) return `${files.length} files`
  return stringValue(input.filePath) ?? stringValue(input.filepath) ?? stringValue(input.path)
}

function permissionLocations(toolName: string, input: ToolInput): ToolCallLocation[] {
  const files = fileMetadata(input)
  if (files.length) {
    return Array.from(
      new Set(files.flatMap((file) => [file.filePath, file.movePath].filter((path): path is string => !!path))),
      (path) => ({ path }),
    )
  }
  return toLocations(toolName, input)
}

async function permissionContent(toolName: string, input: ToolInput): Promise<ToolCallContent[]> {
  if (toolName.toLocaleLowerCase() !== "edit") return []

  const files = fileMetadata(input)
  if (files.length) return diffContentForFiles(files)

  const filepath = stringValue(input.filepath) ?? stringValue(input.filePath)
  const diff = stringValue(input.diff)
  if (!filepath || !diff) return []
  const content = await diffContentForPatch(filepath, diff)
  return content ? [content] : []
}

async function diffContentForFiles(files: PermissionFileMetadata[]) {
  const content = await Promise.all(
    files.map(async (file) => {
      if (!file.patch) return []
      const content = await diffContentForPatch(file.filePath, file.patch, file.movePath)
      return content ? [content] : []
    }),
  )
  return content.flat()
}

async function diffContentForPatch(filepath: string, diff: string, displayPath = filepath) {
  const content = (await exists(filepath)) ? await readText(filepath) : ""
  const next = applyPatch(content, diff)
  if (next === false) return undefined
  return {
    type: "diff" as const,
    path: displayPath,
    oldText: content,
    newText: next,
  }
}

function selectedReply(result: RequestPermissionResponse): Reply {
  if (result.outcome.outcome !== "selected") return "reject"
  if (result.outcome.optionId === "once" || result.outcome.optionId === "always") return result.outcome.optionId
  return "reject"
}

function stringValue(value: unknown) {
  return typeof value === "string" ? value : undefined
}

type PermissionFileMetadata = {
  readonly filePath: string
  readonly relativePath?: string
  readonly movePath?: string
  readonly patch?: string
}

function fileMetadata(input: ToolInput): PermissionFileMetadata[] {
  if (!Array.isArray(input.files)) return []
  return input.files.flatMap((file): PermissionFileMetadata[] => {
    if (!file || typeof file !== "object") return []
    const info = file as Record<string, unknown>
    const filePath = stringValue(info.filePath)
    if (!filePath) return []
    return [
      {
        filePath,
        relativePath: stringValue(info.relativePath),
        movePath: stringValue(info.movePath),
        patch: stringValue(info.patch),
      },
    ]
  })
}

export * as ACPPermission from "./permission"
