export type Definition = {
  [method: string]: (input: any, signal: AbortSignal) => any
}

export type Target = {
  postMessage: (data: string) => void | null
  onmessage: ((evt: MessageEvent<string>) => unknown) | null
}

export function listen(rpc: Definition, target?: Target) {
  const active = new Map<number, AbortController>()
  const handler = async (evt: MessageEvent<string>) => {
    const parsed = JSON.parse(evt.data)
    if (parsed.type === "rpc.cancel") {
      const controller = active.get(parsed.id)
      if (!controller) return
      active.delete(parsed.id)
      controller.abort()
      return
    }
    if (parsed.type === "rpc.request") {
      const controller = new AbortController()
      active.set(parsed.id, controller)
      try {
        const result = await rpc[parsed.method](parsed.input, controller.signal)
        send(JSON.stringify({ type: "rpc.result", result, id: parsed.id }))
      } catch (error) {
        send(
          JSON.stringify({
            type: "rpc.error",
            error:
              error instanceof Error
                ? { name: error.name, message: error.message, stack: error.stack }
                : { name: "Error", message: String(error) },
            id: parsed.id,
          }),
        )
      } finally {
        active.delete(parsed.id)
      }
    }
  }
  if (target) target.onmessage = handler
  else onmessage = handler

  function send(data: string) {
    if (target) return target.postMessage(data)
    postMessage(data)
  }
}

export function emit(event: string, data: unknown) {
  postMessage(JSON.stringify({ type: "rpc.event", event, data }))
}

export function client<T extends Definition>(target: Target) {
  const pending = new Map<
    number,
    {
      resolve: (result: any) => void
      reject: (error?: unknown) => void
      signal?: AbortSignal
      onAbort?: () => void
    }
  >()
  const listeners = new Map<string, Set<(data: any) => void>>()
  let id = 0
  const take = (requestID: number) => {
    const request = pending.get(requestID)
    if (!request) return undefined
    pending.delete(requestID)
    if (request.signal && request.onAbort) request.signal.removeEventListener("abort", request.onAbort)
    return request
  }
  target.onmessage = async (evt) => {
    const parsed = JSON.parse(evt.data)
    if (parsed.type === "rpc.result") {
      take(parsed.id)?.resolve(parsed.result)
    }
    if (parsed.type === "rpc.error") {
      const request = take(parsed.id)
      if (request) {
        const error = new Error(parsed.error.message)
        error.name = parsed.error.name
        error.stack = parsed.error.stack
        request.reject(error)
      }
    }
    if (parsed.type === "rpc.event") {
      const handlers = listeners.get(parsed.event)
      if (handlers) {
        for (const handler of handlers) {
          handler(parsed.data)
        }
      }
    }
  }
  return {
    call<Method extends keyof T>(
      method: Method,
      input: Parameters<T[Method]>[0],
      options?: { signal?: AbortSignal },
    ): Promise<Awaited<ReturnType<T[Method]>>> {
      const requestId = id++
      return new Promise((resolve, reject) => {
        if (options?.signal?.aborted) {
          reject(options.signal.reason)
          return
        }
        const onAbort = () => {
          const request = take(requestId)
          if (!request) return
          try {
            target.postMessage(JSON.stringify({ type: "rpc.cancel", id: requestId }))
          } catch {}
          request.reject(options?.signal?.reason)
        }
        pending.set(requestId, { resolve, reject, signal: options?.signal, onAbort })
        options?.signal?.addEventListener("abort", onAbort, { once: true })
        try {
          target.postMessage(JSON.stringify({ type: "rpc.request", method, input, id: requestId }))
        } catch (error) {
          take(requestId)?.reject(error)
        }
      })
    },
    on<Data>(event: string, handler: (data: Data) => void) {
      let handlers = listeners.get(event)
      if (!handlers) {
        handlers = new Set()
        listeners.set(event, handlers)
      }
      handlers.add(handler)
      return () => {
        handlers!.delete(handler)
      }
    },
  }
}

export * as Rpc from "./rpc"
