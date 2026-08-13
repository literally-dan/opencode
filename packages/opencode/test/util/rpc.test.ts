import { describe, expect, test } from "bun:test"
import { getEventListeners } from "node:events"
import { Rpc } from "@/util/rpc"

function harness<T extends Rpc.Definition>(definition: T) {
  const clientTarget: Rpc.Target = {
    onmessage: null,
    postMessage: (data) => queueMicrotask(() => serverTarget.onmessage?.(new MessageEvent("message", { data }))),
  }
  const serverTarget: Rpc.Target = {
    onmessage: null,
    postMessage: (data) => queueMicrotask(() => clientTarget.onmessage?.(new MessageEvent("message", { data }))),
  }
  Rpc.listen(definition, serverTarget)
  return {
    client: Rpc.client<T>(clientTarget),
    send: clientTarget.postMessage,
  }
}

function rejection<T>(promise: Promise<T>) {
  return promise.then(
    () => {
      throw new Error("expected rejection")
    },
    (error: unknown) => error,
  )
}

describe("util.rpc cancellation", () => {
  test("propagates abort to the active handler", async () => {
    const started = Promise.withResolvers<void>()
    const aborted = Promise.withResolvers<AbortSignal>()
    const rpc = harness({
      wait: (_input: undefined, signal: AbortSignal) => {
        started.resolve()
        return new Promise<void>((resolve) => {
          signal.addEventListener(
            "abort",
            () => {
              aborted.resolve(signal)
              resolve()
            },
            { once: true },
          )
        })
      },
    })
    const controller = new AbortController()
    const reason = new Error("cancelled")
    const pending = rpc.client.call("wait", undefined, { signal: controller.signal })

    await started.promise
    controller.abort(reason)

    expect(await rejection(pending)).toBe(reason)
    expect((await aborted.promise).aborted).toBe(true)
  })

  test("isolates concurrent request cancellation", async () => {
    const started = Promise.withResolvers<void>()
    const firstAborted = Promise.withResolvers<void>()
    const secondDone = Promise.withResolvers<string>()
    const signals = new Map<string, AbortSignal>()
    const rpc = harness({
      wait: (input: string, signal: AbortSignal) => {
        signals.set(input, signal)
        if (signals.size === 2) started.resolve()
        if (input === "second") return secondDone.promise
        return new Promise<string>((resolve) => {
          signal.addEventListener(
            "abort",
            () => {
              firstAborted.resolve()
              resolve("first aborted")
            },
            { once: true },
          )
        })
      },
    })
    const firstController = new AbortController()
    const first = rpc.client.call("wait", "first", { signal: firstController.signal })
    const second = rpc.client.call("wait", "second")

    await started.promise
    firstController.abort()
    await firstAborted.promise
    expect(signals.get("second")?.aborted).toBe(false)
    secondDone.resolve("second completed")

    expect(await rejection(first)).toBeInstanceOf(DOMException)
    expect(await second).toBe("second completed")
  })

  test("does not dispatch a pre-aborted request", async () => {
    let calls = 0
    const rpc = harness({
      run: () => {
        calls += 1
      },
    })
    const controller = new AbortController()
    const reason = new Error("already cancelled")
    controller.abort(reason)

    expect(await rejection(rpc.client.call("run", undefined, { signal: controller.signal }))).toBe(reason)
    await Promise.resolve()
    expect(calls).toBe(0)
  })

  test("ignores late and duplicate cancellation", async () => {
    let signal: AbortSignal | undefined
    const rpc = harness({
      run: (_input: undefined, inputSignal: AbortSignal) => {
        signal = inputSignal
        return "completed"
      },
    })

    expect(await rpc.client.call("run", undefined)).toBe("completed")
    rpc.send(JSON.stringify({ type: "rpc.cancel", id: 0 }))
    rpc.send(JSON.stringify({ type: "rpc.cancel", id: 0 }))
    await Promise.resolve()

    expect(signal?.aborted).toBe(false)
  })

  test("removes abort listeners on every client terminal path", async () => {
    const handlerAborted = Promise.withResolvers<void>()
    const rpc = harness({
      succeed: () => "ok",
      fail: () => {
        throw new Error("failed")
      },
      wait: (_input: undefined, signal: AbortSignal) =>
        new Promise<void>((resolve) =>
          signal.addEventListener(
            "abort",
            () => {
              handlerAborted.resolve()
              resolve()
            },
            { once: true },
          ),
        ),
    })
    const succeeded = new AbortController()
    const failed = new AbortController()
    const aborted = new AbortController()

    expect(await rpc.client.call("succeed", undefined, { signal: succeeded.signal })).toBe("ok")
    expect(await rejection(rpc.client.call("fail", undefined, { signal: failed.signal }))).toEqual(new Error("failed"))
    const pending = rpc.client.call("wait", undefined, { signal: aborted.signal })
    aborted.abort()
    expect(await rejection(pending)).toBeInstanceOf(DOMException)
    await handlerAborted.promise

    expect(getEventListeners(succeeded.signal, "abort")).toHaveLength(0)
    expect(getEventListeners(failed.signal, "abort")).toHaveLength(0)
    expect(getEventListeners(aborted.signal, "abort")).toHaveLength(0)
  })
})
