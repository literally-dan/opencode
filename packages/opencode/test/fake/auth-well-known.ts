import { AuthWellKnown } from "@opencode-ai/core/auth-well-known"
import { Effect, Layer } from "effect"

export const AuthWellKnownTest = {
  empty: Layer.mock(AuthWellKnown.Service, {
    configs: () => Effect.succeed([]),
  }),
}
