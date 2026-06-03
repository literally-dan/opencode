import { describe, expect } from "bun:test"
import path from "path"
import { Effect, Layer } from "effect"
import { HttpClient, HttpClientResponse } from "effect/unstable/http"
import { FSUtil } from "@opencode-ai/core/fs-util"
import { Global } from "@opencode-ai/core/global"
import { Substitution } from "@opencode-ai/core/substitution"
import { AuthWellKnown } from "@opencode-ai/core/auth-well-known"
import { tmpdir } from "./fixture/tmpdir"
import { testEffect } from "./lib/effect"

const it = testEffect(Layer.empty)

const unexpectedHttpClient = HttpClient.make((request) => Effect.die(`unexpected http request: ${request.url}`))

const withAuthWellKnown = <A, E, R>(
  dir: string,
  effect: Effect.Effect<A, E, R | AuthWellKnown.Service>,
  client = unexpectedHttpClient,
) =>
  effect.pipe(
    Effect.provide(AuthWellKnown.layer),
    Effect.provide(FSUtil.defaultLayer),
    Effect.provide(Global.layerWith({ data: dir })),
    Effect.provide(Layer.succeed(HttpClient.HttpClient, client)),
    Effect.provide(Substitution.defaultLayer),
  )

const wellKnownConfigClient = HttpClient.make((request) => {
  if (request.url === "https://example.com/.well-known/opencode") {
    return Effect.succeed(
      HttpClientResponse.fromWeb(
        request,
        Response.json({
          config: { instructions: ["local"] },
          remote_config: {
            url: "https://remote.example.com/config",
            headers: {
              authorization: "Bearer {env:TEST_TOKEN}",
            },
          },
        }),
      ),
    )
  }
  if (request.url === "https://remote.example.com/config") {
    expect(request.headers.authorization).toBe("Bearer secret")
    return Effect.succeed(HttpClientResponse.fromWeb(request, Response.json({ model: "remote/model" })))
  }
  return Effect.succeed(HttpClientResponse.fromWeb(request, new Response(null, { status: 404 })))
})

describe("AuthWellKnown", () => {
  it.live("stores well-known credentials", () =>
    Effect.gen(function* () {
      const tmp = yield* Effect.acquireRelease(
        Effect.promise(() => tmpdir()),
        (tmp) => Effect.promise(() => tmp[Symbol.asyncDispose]()),
      )

      yield* withAuthWellKnown(
        tmp.path,
        Effect.gen(function* () {
          const auth = yield* AuthWellKnown.Service
          yield* auth.set("https://example.com/", new AuthWellKnown.Entry({ key: "TEST_TOKEN", token: "secret" }))
        }),
      )

      expect(yield* Effect.promise(() => Bun.file(path.join(tmp.path, "well-known.json")).json())).toEqual({
        "https://example.com": {
          key: "TEST_TOKEN",
          token: "secret",
        },
      })
    }),
  )

  it.live("migrates legacy well-known auth records", () =>
    Effect.gen(function* () {
      const tmp = yield* Effect.acquireRelease(
        Effect.promise(() => tmpdir()),
        (tmp) => Effect.promise(() => tmp[Symbol.asyncDispose]()),
      )
      yield* Effect.promise(() =>
        Bun.write(
          path.join(tmp.path, "auth.json"),
          JSON.stringify({
            "https://example.com": {
              type: "wellknown",
              key: "TEST_TOKEN",
              token: "secret",
            },
          }),
        ),
      )

      const entry = yield* withAuthWellKnown(
        tmp.path,
        Effect.gen(function* () {
          const auth = yield* AuthWellKnown.Service
          return yield* auth.get("https://example.com/")
        }),
      )

      expect(entry).toEqual({
        key: "TEST_TOKEN",
        token: "secret",
      })
      expect(yield* Effect.promise(() => Bun.file(path.join(tmp.path, "well-known.json")).json())).toEqual({
        "https://example.com": {
          key: "TEST_TOKEN",
          token: "secret",
        },
      })
    }),
  )

  it.live("loads config documents", () =>
    Effect.gen(function* () {
      const tmp = yield* Effect.acquireRelease(
        Effect.promise(() => tmpdir()),
        (tmp) => Effect.promise(() => tmp[Symbol.asyncDispose]()),
      )
      yield* Effect.promise(() =>
        Bun.write(
          path.join(tmp.path, "well-known.json"),
          JSON.stringify({
            "https://example.com": {
              key: "TEST_TOKEN",
              token: "secret",
            },
          }),
        ),
      )

      const result = yield* withAuthWellKnown(
        tmp.path,
        Effect.gen(function* () {
          const auth = yield* AuthWellKnown.Service
          return yield* auth.configs()
        }),
        wellKnownConfigClient,
      )

      expect(result).toEqual([
        {
          url: "https://example.com",
          source: "https://example.com/.well-known/opencode",
          dir: "https://example.com/.well-known",
          content: { instructions: ["local"] },
        },
        {
          url: "https://remote.example.com/config",
          source: "https://remote.example.com/config",
          dir: "https://remote.example.com",
          content: { model: "remote/model" },
        },
      ])
    }),
  )
})
