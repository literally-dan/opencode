import { expect, test } from "bun:test"
import SqliteDatabase from "bun:sqlite"
import path from "path"
import { Database } from "../src/database/database"
import { Sqlite } from "../src/database/sqlite"
import { Cause, Effect, Exit } from "effect"
import { classifySqliteError, LockTimeoutError, SqlError } from "effect/unstable/sql/SqlError"
import { tmpdir } from "./fixture/tmpdir"

test("classifies node:sqlite busy errors as lock timeouts", () => {
  // node:sqlite is not available under Bun, so this uses the error shape node:sqlite throws for SQLITE_BUSY.
  const busy = Object.assign(new Error("database is locked"), {
    code: "ERR_SQLITE_ERROR",
    errcode: 5,
    errstr: "database is locked",
  })
  expect(classifySqliteError(busy)._tag).toBe("UnknownError")
  expect(classifySqliteError(Sqlite.withErrno(busy))._tag).toBe("LockTimeoutError")
  const unique = Object.assign(new Error("UNIQUE constraint failed: t.x"), { code: "ERR_SQLITE_ERROR", errcode: 2067 })
  expect(classifySqliteError(Sqlite.withErrno(unique))._tag).toBe("UniqueViolation")
})

test("retries a Drizzle statement blocked by a real database lock", async () => {
  await using dir = await tmpdir()
  const file = path.join(dir.path, "lock.db")
  const holder = new SqliteDatabase(file)
  let attempts = 0
  const result = await Effect.runPromise(
    Effect.gen(function* () {
      const { db } = yield* Database.Service
      yield* db.run("PRAGMA busy_timeout = 0")
      yield* db.run("CREATE TABLE lock_test (value INTEGER)")
      holder.run("BEGIN IMMEDIATE")
      return yield* Database.retryLockTimeout(
        Effect.suspend(() => {
          attempts++
          if (attempts === 2) holder.run("ROLLBACK")
          return db.run("INSERT INTO lock_test (value) VALUES (1)").pipe(Effect.orDie, Effect.as("saved"))
        }),
      )
    }).pipe(Effect.provide(Database.layerFromPath(file)), Effect.ensuring(Effect.sync(() => holder.close()))),
  )

  expect(result).toBe("saved")
  expect(attempts).toBe(2)
})

test("retries a defect caused by a database lock timeout", async () => {
  let attempts = 0
  const lock = new SqlError({
    reason: new LockTimeoutError({ cause: new Error("database is locked"), operation: "execute" }),
  })
  const result = await Effect.runPromise(
    Database.retryLockTimeout(
      Effect.suspend(() => {
        attempts++
        return attempts === 1 ? Effect.die(lock) : Effect.succeed("saved")
      }),
    ),
  )

  expect(result).toBe("saved")
  expect(attempts).toBe(2)
})

test("does not retry other defects", async () => {
  let attempts = 0
  const error = new Error("invalid data")
  const exit = await Effect.runPromise(
    Database.retryLockTimeout(
      Effect.suspend(() => {
        attempts++
        return Effect.die(error)
      }),
    ).pipe(Effect.exit),
  )

  expect(Exit.isFailure(exit)).toBe(true)
  expect(attempts).toBe(1)
})

test("returns the original lock defect after exhausting retries", async () => {
  let attempts = 0
  const lock = new SqlError({
    reason: new LockTimeoutError({ cause: new Error("database is locked"), operation: "execute" }),
  })
  const exit = await Effect.runPromise(
    Database.retryLockTimeout(
      Effect.suspend(() => {
        attempts++
        return Effect.die(lock)
      }),
    ).pipe(Effect.exit),
  )

  expect(Exit.isFailure(exit)).toBe(true)
  if (Exit.isSuccess(exit)) return
  expect(Cause.squash(exit.cause)).toBe(lock)
  expect(attempts).toBe(2)
})

test("does not retry typed lock failures", async () => {
  let attempts = 0
  const lock = new SqlError({
    reason: new LockTimeoutError({ cause: new Error("database is locked"), operation: "execute" }),
  })
  const exit = await Effect.runPromise(
    Database.retryLockTimeout(
      Effect.suspend(() => {
        attempts++
        return Effect.fail(lock)
      }),
    ).pipe(Effect.exit),
  )

  expect(Exit.isFailure(exit)).toBe(true)
  expect(attempts).toBe(1)
})
