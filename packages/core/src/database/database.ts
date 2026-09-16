export * as Database from "./database"

import { EffectDrizzleSqlite } from "@opencode-ai/effect-drizzle-sqlite"
import { layer as sqliteLayer } from "#sqlite"
import { Cause, Context, Effect, Layer, Predicate } from "effect"
import { isSqlError } from "effect/unstable/sql/SqlError"
import { Global } from "../global"
import { Flag } from "../flag/flag"
import { isAbsolute, join } from "path"
import { DatabaseMigration } from "./migration"
import { InstallationChannel } from "../installation/version"
import { makeGlobalNode } from "../effect/app-node"

const makeDatabase = EffectDrizzleSqlite.makeWithDefaults()
type DatabaseShape = Effect.Success<typeof makeDatabase>

export interface Interface {
  db: DatabaseShape
}

export class Service extends Context.Service<Service, Interface>()("@opencode/v2/storage/Database") {}

const layer = Layer.effect(
  Service,
  Effect.gen(function* () {
    const db = yield* makeDatabase

    yield* db.run("PRAGMA journal_mode = WAL")
    yield* db.run("PRAGMA synchronous = NORMAL")
    yield* db.run("PRAGMA busy_timeout = 5000")
    yield* db.run("PRAGMA cache_size = -64000")
    yield* db.run("PRAGMA foreign_keys = ON")
    yield* db.run("PRAGMA wal_checkpoint(PASSIVE)")
    yield* DatabaseMigration.apply(db)

    return { db }
  }).pipe(Effect.orDie),
)

export function layerFromPath(filename: string) {
  return layer.pipe(Layer.provide(sqliteLayer({ filename })))
}

export function path() {
  if (Flag.OPENCODE_DB) {
    if (Flag.OPENCODE_DB === ":memory:" || isAbsolute(Flag.OPENCODE_DB)) return Flag.OPENCODE_DB
    return join(Global.Path.data, Flag.OPENCODE_DB)
  }
  if (
    ["latest", "beta", "prod"].includes(InstallationChannel) ||
    process.env.OPENCODE_DISABLE_CHANNEL_DB === "1" ||
    process.env.OPENCODE_DISABLE_CHANNEL_DB === "true"
  )
    return join(Global.Path.data, "opencode.db")
  return join(Global.Path.data, `opencode-${InstallationChannel.replace(/[^a-zA-Z0-9._-]/g, "-")}.db`)
}

export function retryLockTimeout<A, E, R>(effect: Effect.Effect<A, E, R>, retries = 1): Effect.Effect<A, E, R> {
  const run = (remaining: number): Effect.Effect<A, E, R> =>
    effect.pipe(
      Effect.catchCause((cause) => {
        const reason = cause.reasons.length === 1 ? cause.reasons[0] : undefined
        if (remaining === 0 || !reason || !Cause.isDieReason(reason) || !isLockTimeout(reason.defect))
          return Effect.failCause(cause)
        return Effect.sleep(100).pipe(Effect.andThen(run(remaining - 1)))
      }),
    )
  return run(retries)
}

function isLockTimeout(defect: unknown): boolean {
  if (isSqlError(defect)) return defect.reason._tag === "LockTimeoutError"
  // Drizzle statements wrap the SqlError in an EffectDrizzleQueryError whose cause is a Cause.
  if (!Predicate.isTagged(defect, "EffectDrizzleQueryError") || !Predicate.hasProperty(defect, "cause")) return false
  return Cause.isCause(defect.cause) && isLockTimeout(Cause.squash(defect.cause))
}

export const node = makeGlobalNode({ service: Service, layer: layerFromPath(path()), deps: [] })
