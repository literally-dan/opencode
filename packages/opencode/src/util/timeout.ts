export function withTimeout<T>(promise: Promise<T>, ms: number, label?: string): Promise<T> {
  let timeout: NodeJS.Timeout
  return Promise.race([
    promise.finally(() => {
      clearTimeout(timeout)
    }),
    new Promise<never>((_, reject) => {
      timeout = setTimeout(() => {
        reject(new Error(label ?? `Operation timed out after ${ms}ms`))
      }, ms)
    }),
  ])
}

export function withAbortTimeout<T>(
  operation: (signal: AbortSignal) => Promise<T>,
  input: { signal: AbortSignal; timeoutMs: number; label?: string },
) {
  if (input.signal.aborted) return Promise.reject(input.signal.reason)
  return new Promise<T>((resolve, reject) => {
    let settled = false
    const timeout = new AbortController()
    const finish = (result: { value: T } | { error: unknown }) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      input.signal.removeEventListener("abort", onAbort)
      if ("error" in result) {
        reject(result.error)
        return
      }
      resolve(result.value)
    }
    const onAbort = () => finish({ error: input.signal.reason })
    const timer = setTimeout(() => {
      const error = new Error(input.label ?? `Operation timed out after ${input.timeoutMs}ms`)
      timeout.abort(error)
      finish({ error })
    }, input.timeoutMs)
    input.signal.addEventListener("abort", onAbort, { once: true })
    Promise.resolve()
      .then(() => operation(AbortSignal.any([input.signal, timeout.signal])))
      .then(
        (value) => finish({ value }),
        (error) => finish({ error }),
      )
    if (input.signal.aborted) onAbort()
  })
}
