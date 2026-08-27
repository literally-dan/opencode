import type { OpencodeClient } from "@opencode-ai/sdk/v2"
import { withAbortTimeout } from "@/util/timeout"
import { SessionAncestry } from "./ancestry"

type Reply = "once" | "always" | "reject"

const REQUEST_TIMEOUT_MS = 10_000

export async function replyPermission(
  sdk: OpencodeClient,
  input: {
    requestID: string
    reply: Reply
    directory?: string
    signal: AbortSignal
    timeoutMs?: number
    maxAttempts?: number
    onRetry?: (input: { error: unknown; attempt: number; retryIn: number }) => void
  },
) {
  for (let attempt = 1; ; attempt++) {
    input.signal.throwIfAborted()
    const timeoutMs = input.timeoutMs ?? REQUEST_TIMEOUT_MS
    const result = await withAbortTimeout(
      async (signal) => {
        const result: unknown = await sdk.permission.reply(
          {
            requestID: input.requestID,
            reply: input.reply,
            ...(input.directory ? { directory: input.directory } : {}),
          },
          { throwOnError: true, signal },
        )
        if (isRecord(result) && "error" in result && result.error !== undefined) throw result.error
      },
      {
        signal: input.signal,
        timeoutMs,
        label: `permission reply timed out after ${timeoutMs}ms`,
      },
    ).then(
      () => ({ done: true }) as const,
      (error) => ({ done: false, error }) as const,
    )
    if (result.done || isPermissionNotFound(result.error)) return
    if (input.signal.aborted) input.signal.throwIfAborted()
    if (!isRetryable(result.error)) throw result.error
    if (attempt >= (input.maxAttempts ?? Number.POSITIVE_INFINITY)) throw result.error
    const retryIn = SessionAncestry.retryDelay(attempt)
    input.onRetry?.({ error: result.error, attempt, retryIn })
    await wait(retryIn, input.signal)
  }
}

function isPermissionNotFound(error: unknown) {
  const body = errorBody(error)
  return isRecord(body) && body._tag === "PermissionNotFoundError"
}

function isRetryable(error: unknown) {
  const status = errorStatus(error)
  if (status !== undefined) return status === 408 || status === 429 || status >= 500
  const body = errorBody(error)
  return !isRecord(body) || typeof body._tag !== "string"
}

function errorBody(error: unknown): unknown {
  if (!isRecord(error)) return error
  if (typeof error._tag === "string") return error
  if (!isRecord(error.cause) || !("body" in error.cause)) return error
  return error.cause.body
}

function errorStatus(error: unknown) {
  if (!isRecord(error)) return undefined
  if (typeof error.status === "number") return error.status
  if (!isRecord(error.cause)) return undefined
  return typeof error.cause.status === "number" ? error.cause.status : undefined
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null
}

function wait(delay: number, signal: AbortSignal) {
  signal.throwIfAborted()
  return new Promise<void>((resolve, reject) => {
    const finish = (error?: unknown) => {
      clearTimeout(timer)
      signal.removeEventListener("abort", onAbort)
      if (error !== undefined) {
        reject(error)
        return
      }
      resolve()
    }
    const onAbort = () => finish(signal.reason)
    const timer = setTimeout(() => finish(), delay)
    signal.addEventListener("abort", onAbort, { once: true })
    if (signal.aborted) onAbort()
  })
}
