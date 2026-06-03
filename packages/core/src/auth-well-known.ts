export * as AuthWellKnown from "./auth-well-known"

import path from "path"
import { Context, Effect, Layer, Option, Schema, SynchronizedRef } from "effect"
import { FetchHttpClient, HttpClient, HttpClientRequest } from "effect/unstable/http"
import { FSUtil } from "./fs-util"
import { Global } from "./global"
import { Substitution } from "./substitution"

export class Entry extends Schema.Class<Entry>("AuthWellKnown.Entry")({
  key: Schema.String,
  token: Schema.String,
}) {}

export class FileWriteError extends Schema.TaggedErrorClass<FileWriteError>()("AuthWellKnown.FileWriteError", {
  operation: Schema.Union([Schema.Literal("migrate"), Schema.Literal("write")]),
  cause: Schema.Defect,
}) {}

export class RemoteConfigError extends Schema.TaggedErrorClass<RemoteConfigError>()("AuthWellKnown.RemoteConfigError", {
  url: Schema.String,
  status: Schema.Number.pipe(Schema.optional),
  cause: Schema.Defect.pipe(Schema.optional),
}) {}

export type Error = FileWriteError | RemoteConfigError

const RemoteConfig = Schema.Struct({
  url: Schema.String,
  headers: Schema.Record(Schema.String, Schema.String).pipe(Schema.optional),
})

export class Metadata extends Schema.Class<Metadata>("AuthWellKnown.Metadata")({
  auth: Schema.Struct({
    command: Schema.Array(Schema.String),
    env: Schema.String,
  }).pipe(Schema.optional),
  config: Schema.Record(Schema.String, Schema.Unknown).pipe(Schema.optional),
  remote_config: RemoteConfig.pipe(Schema.optional),
}) {}

export type ConfigDocument = {
  url: string
  source: string
  dir: string
  content: unknown
}

export interface Interface {
  readonly all: () => Effect.Effect<Record<string, Entry>, Error>
  readonly get: (url: string) => Effect.Effect<Entry | undefined, Error>
  readonly set: (url: string, entry: Entry) => Effect.Effect<void, Error>
  readonly remove: (url: string) => Effect.Effect<void, Error>
  readonly metadata: (url: string) => Effect.Effect<Metadata, Error>
  readonly configs: () => Effect.Effect<ConfigDocument[], Error>
}

export class Service extends Context.Service<Service, Interface>()("@opencode/v2/AuthWellKnown") {}
const decodeMetadata = Schema.decodeUnknownEffect(Metadata)
const decodeRemoteConfig = Schema.decodeUnknownEffect(RemoteConfig)

function loadLegacyAuth(input: {
  fsys: FSUtil.Interface
  dataDir: string
  write: (data: Record<string, Entry>) => Effect.Effect<void, Error>
}) {
  return Effect.gen(function* () {
    const decodeLegacy = Schema.decodeUnknownOption(Schema.Record(Schema.String, Schema.Unknown))
    const decodeLegacyCredential = Schema.decodeUnknownOption(
      Schema.Struct({
        type: Schema.Literal("wellknown"),
        key: Schema.String,
        token: Schema.String,
      }),
    )
    const legacy = Object.fromEntries(
      Object.entries(
        Option.getOrElse(
          decodeLegacy(
            yield* input.fsys.readJson(path.join(input.dataDir, "auth.json")).pipe(Effect.orElseSucceed(() => null)),
          ),
          () => ({}),
        ),
      ).flatMap(([url, value]) => {
        const decoded = Option.getOrUndefined(decodeLegacyCredential(value))
        return decoded ? [[url.replace(/\/+$/, ""), new Entry({ key: decoded.key, token: decoded.token })]] : []
      }),
    )
    if (Object.keys(legacy).length > 0) yield* input.write(legacy).pipe(Effect.ignore)
    return legacy
  })
}

export const layer = Layer.effect(
  Service,
  Effect.gen(function* () {
    const fsys = yield* FSUtil.Service
    const global = yield* Global.Service
    const http = yield* HttpClient.HttpClient
    const substitution = yield* Substitution.Service
    const file = path.join(global.data, "well-known.json")
    const decodeEntries = Schema.decodeUnknownOption(Schema.Record(Schema.String, Entry))
    const normalizeUrl = (url: string) => url.replace(/\/+$/, "")

    const write = (operation: "migrate" | "write", data: Record<string, Entry>) =>
      fsys.writeJson(file, data, 0o600).pipe(Effect.mapError((cause) => new FileWriteError({ operation, cause })))

    const load: () => Effect.Effect<Record<string, Entry>> = Effect.fnUntraced(function* () {
      const current = yield* fsys.readJson(file).pipe(Effect.orElseSucceed(() => null))
      if (current && typeof current === "object")
        return Option.getOrElse(decodeEntries(current), () => ({}) as Record<string, Entry>)
      return yield* loadLegacyAuth({ fsys, dataDir: global.data, write: (data) => write("migrate", data) })
    })

    const state = SynchronizedRef.makeUnsafe<Record<string, Entry>>(yield* load())

    const metadata = Effect.fn("AuthWellKnown.metadata")(function* (url: string) {
      const normalized = normalizeUrl(url)
      const source = `${normalized}/.well-known/opencode`
      const response = yield* HttpClientRequest.get(source).pipe(
        HttpClientRequest.acceptJson,
        http.execute,
        Effect.mapError((cause) => new RemoteConfigError({ url: source, cause })),
      )
      if (response.status < 200 || response.status >= 300) {
        return yield* new RemoteConfigError({ url: source, status: response.status })
      }
      const metadata = yield* response.json.pipe(
        Effect.flatMap(decodeMetadata),
        Effect.mapError((cause) => new RemoteConfigError({ url: source, cause })),
      )
      return { url: normalized, source, dir: path.dirname(source), metadata }
    })

    const remote = Effect.fn("AuthWellKnown.remote")(function* (input: { url: string; headers?: Record<string, string> }) {
      const response = yield* HttpClientRequest.get(input.url).pipe(
        HttpClientRequest.acceptJson,
        input.headers ? HttpClientRequest.setHeaders(input.headers) : (request) => request,
        http.execute,
        Effect.mapError((cause) => new RemoteConfigError({ url: input.url, cause })),
      )
      if (response.status < 200 || response.status >= 300) {
        return yield* new RemoteConfigError({ url: input.url, status: response.status })
      }
      return yield* response.json.pipe(Effect.mapError((cause) => new RemoteConfigError({ url: input.url, cause })))
    })

    return Service.of({
      all: Effect.fn("AuthWellKnown.all")(function* () {
        return yield* SynchronizedRef.get(state)
      }),

      get: Effect.fn("AuthWellKnown.get")(function* (url) {
        return (yield* SynchronizedRef.get(state))[normalizeUrl(url)]
      }),

      set: Effect.fn("AuthWellKnown.set")(function* (url, entry) {
        yield* SynchronizedRef.updateEffect(
          state,
          Effect.fnUntraced(function* (data) {
            const next = { ...data, [normalizeUrl(url)]: entry }
            yield* write("write", next)
            return next
          }),
        )
      }),

      remove: Effect.fn("AuthWellKnown.remove")(function* (url) {
        yield* SynchronizedRef.updateEffect(
          state,
          Effect.fnUntraced(function* (data) {
            const next = { ...data }
            delete next[url]
            delete next[normalizeUrl(url)]
            yield* write("write", next)
            return next
          }),
        )
      }),

      metadata: Effect.fn("AuthWellKnown.metadata.public")(function* (url) {
        return (yield* metadata(url)).metadata
      }),

      configs: Effect.fn("AuthWellKnown.configs")(function* () {
        const documents = yield* Effect.all(
          Object.entries(yield* SynchronizedRef.get(state)).map(([url, entry]) =>
            Effect.gen(function* () {
              const configs: ConfigDocument[] = []
              const response = yield* metadata(url)
              const env = { [entry.key]: entry.token }
              if (response.metadata.config) {
                configs.push({
                  url: response.url,
                  source: response.source,
                  dir: response.dir,
                  content: response.metadata.config,
                })
              }
              if (response.metadata.remote_config) {
                const remoteConfig = yield* substitution
                  .substitute({
                    text: JSON.stringify(response.metadata.remote_config),
                    type: "virtual",
                    dir: response.url,
                    source: response.source,
                    env,
                  })
                  .pipe(
                    Effect.flatMap((text) =>
                      Effect.try({
                        try: () => JSON.parse(text) as unknown,
                        catch: (cause) => new RemoteConfigError({ url: response.source, cause }),
                      }),
                    ),
                    Effect.flatMap(decodeRemoteConfig),
                    Effect.mapError((cause) => new RemoteConfigError({ url: response.source, cause })),
                  )
                configs.push({
                  url: remoteConfig.url,
                  source: remoteConfig.url,
                  dir: path.dirname(remoteConfig.url),
                  content: yield* remote({ url: remoteConfig.url, headers: remoteConfig.headers }),
                })
              }
              return configs
            }),
          ),
          { concurrency: "unbounded" },
        )
        return documents.flat()
      }),
    })
  }),
)

export const defaultLayer = layer.pipe(
  Layer.provide(FSUtil.defaultLayer),
  Layer.provide(Global.defaultLayer),
  Layer.provide(FetchHttpClient.layer),
  Layer.provide(Substitution.defaultLayer),
)
