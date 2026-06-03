export * as Substitution from "./substitution"

import os from "os"
import path from "path"
import { Context, Effect, Layer, Schema } from "effect"
import { FSUtil } from "./fs-util"

type Source =
  | {
      type: "path"
      path: string
    }
  | {
      type: "virtual"
      source: string
      dir: string
    }

export type Input = Source & {
  text: string
  missing?: "error" | "empty"
  env?: Record<string, string | undefined>
}

export class FileReferenceError extends Schema.TaggedErrorClass<FileReferenceError>()("Substitution.FileReferenceError", {
  source: Schema.String,
  token: Schema.String,
  resolved: Schema.String,
  cause: Schema.Defect,
}) {}

export type Error = FileReferenceError

export interface Interface {
  readonly substitute: (input: Input) => Effect.Effect<string, Error>
}

export class Service extends Context.Service<Service, Interface>()("@opencode/v2/Substitution") {}

export const layer = Layer.effect(
  Service,
  Effect.gen(function* () {
    const fs = yield* FSUtil.Service

    return Service.of({
      substitute: Effect.fn("Substitution.substitute")(function* (input) {
        const missing = input.missing ?? "error"
        const text = input.text.replace(/\{env:([^}]+)\}/g, (_, varName) => {
          return input.env?.[varName] ?? process.env[varName] ?? ""
        })

        const fileMatches = Array.from(text.matchAll(/\{file:[^}]+\}/g))
        if (!fileMatches.length) return text

        const configDir = input.type === "path" ? path.dirname(input.path) : input.dir
        const configSource = input.type === "path" ? input.path : input.source
        let out = ""
        let cursor = 0

        for (const match of fileMatches) {
          const token = match[0]
          const index = match.index!
          out += text.slice(cursor, index)

          const lineStart = text.lastIndexOf("\n", index - 1) + 1
          const prefix = text.slice(lineStart, index).trimStart()
          if (prefix.startsWith("//")) {
            out += token
            cursor = index + token.length
            continue
          }

          const reference = token.replace(/^\{file:/, "").replace(/\}$/, "")
          const filepath = reference.startsWith("~/") ? path.join(os.homedir(), reference.slice(2)) : reference
          const resolved = path.isAbsolute(filepath) ? filepath : path.resolve(configDir, filepath)
          const content = yield* fs.readFileString(resolved).pipe(
            Effect.catch((cause) => {
              if (missing === "empty") return Effect.succeed("")
              return Effect.fail(new FileReferenceError({ source: configSource, token, resolved, cause }))
            }),
          )

          out += JSON.stringify(content.trim()).slice(1, -1)
          cursor = index + token.length
        }

        out += text.slice(cursor)
        return out
      }),
    })
  }),
)

export const defaultLayer = layer.pipe(Layer.provide(FSUtil.defaultLayer))
