export * as Sqlite from "./sqlite"

import { Context } from "effect"
import type { drizzle } from "drizzle-orm/bun-sqlite"

export type DrizzleClient = ReturnType<typeof drizzle>
export class Native extends Context.Service<Native, unknown>()("@opencode-ai/core/database/SqliteNative") {}
export class Drizzle extends Context.Service<Drizzle, DrizzleClient>()("@opencode-ai/core/database/SqliteDrizzle") {}

/** node:sqlite reports the SQLite result code as `errcode` with code ERR_SQLITE_ERROR. Classification reads `errno`. */
export function withErrno(cause: unknown) {
  if (!(cause instanceof Error) || !("errcode" in cause) || typeof cause.errcode !== "number" || "errno" in cause)
    return cause
  return Object.assign(cause, { errno: cause.errcode })
}
