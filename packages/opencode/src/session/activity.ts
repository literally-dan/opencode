import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { InstanceRef } from "@/effect/instance-ref"
import { InstanceState } from "@/effect/instance-state"
import { EventV2Bridge } from "@/event-v2-bridge"
import { Context, Effect, Layer } from "effect"
import { SessionID } from "./schema"

export interface Interface {
  readonly list: () => Effect.Effect<Map<SessionID, number>>
  readonly touch: (sessionID: SessionID, time?: number) => Effect.Effect<void>
}

export class Service extends Context.Service<Service, Interface>()("@opencode/SessionActivity") {}

const layer = Layer.effect(
  Service,
  Effect.gen(function* () {
    const events = yield* EventV2Bridge.Service
    const state = yield* InstanceState.make(
      Effect.fn("SessionActivity.state")(() => Effect.succeed(new Map<SessionID, number>())),
    )

    const list = Effect.fn("SessionActivity.list")(function* () {
      return new Map(yield* InstanceState.get(state))
    })

    const touch = Effect.fn("SessionActivity.touch")(function* (sessionID: SessionID, time = Date.now()) {
      const data = yield* InstanceState.get(state)
      data.set(sessionID, Math.max(data.get(sessionID) ?? 0, time))
    })

    const unsubscribe = yield* events.listen((event) =>
      Effect.gen(function* () {
        const sessionID = sessionIDOf(event.data)
        if (!sessionID) return
        // Listeners run in the publisher fiber, and a listener defect fails a non-durable publish. V2 routes and
        // runner drains publish without an InstanceRef, so skip them. Task state also reads durable timestamps.
        if (!(yield* InstanceRef)) return
        yield* touch(sessionID)
      }),
    )
    yield* Effect.addFinalizer(() => unsubscribe)

    return Service.of({ list, touch })
  }),
)

function sessionIDOf(data: unknown) {
  if (typeof data !== "object" || data === null) return undefined
  const sessionID = Reflect.get(data, "sessionID")
  return typeof sessionID === "string" ? SessionID.make(sessionID) : undefined
}

export const node = LayerNode.make({ service: Service, layer, deps: [EventV2Bridge.node] })

export * as SessionActivity from "./activity"
