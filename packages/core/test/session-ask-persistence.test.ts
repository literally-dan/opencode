import { describe, expect } from "bun:test"
import { asc, eq, sql } from "drizzle-orm"
import { Effect, Exit } from "effect"
import { Database } from "@opencode-ai/core/database/database"
import { AppNodeBuilder } from "@opencode-ai/core/effect/app-node-builder"
import { ProjectV2 } from "@opencode-ai/core/project"
import { ProjectTable } from "@opencode-ai/core/project/sql"
import { AbsolutePath } from "@opencode-ai/core/schema"
import { SessionSchema } from "@opencode-ai/core/session/schema"
import {
  SESSION_ASK_TITLE_MAX_CHARS,
  SESSION_ASK_TOOL_ACTIVITY_MAX_BYTES,
  SessionAskThreadTable,
  SessionAskTurnTable,
  SessionTable,
} from "@opencode-ai/core/session/sql"
import { testEffect } from "./lib/effect"

const it = testEffect(AppNodeBuilder.build(Database.node))
const sessionID = SessionSchema.ID.make("ses_ask_persistence")

const setup = Effect.gen(function* () {
  const { db } = yield* Database.Service
  yield* db
    .insert(ProjectTable)
    .values({ id: ProjectV2.ID.global, worktree: AbsolutePath.make("/project"), sandboxes: [] })
    .run()
  yield* db
    .insert(SessionTable)
    .values({
      id: sessionID,
      project_id: ProjectV2.ID.global,
      slug: "ask-persistence",
      directory: "/project",
      title: "Ask persistence",
      version: "test",
      time_created: 10,
      time_updated: 20,
    })
    .run()
})

describe("Session Ask persistence", () => {
  it.effect("stores ordered threads and turns without updating the Session and cascades deletes", () =>
    Effect.gen(function* () {
      yield* setup
      const { db } = yield* Database.Service
      yield* db
        .insert(SessionAskThreadTable)
        .values([
          { id: "ask_b", session_id: sessionID, title: "Second", created_at: 2, updated_at: 5 },
          { id: "ask_a", session_id: sessionID, title: "First", created_at: 1, updated_at: 5 },
        ])
        .run()
      yield* db
        .insert(SessionAskTurnTable)
        .values([
          {
            id: "atn_b",
            thread_id: "ask_a",
            question: "Second question",
            answer: "Second answer",
            tool_activity: [],
            created_at: 4,
          },
          {
            id: "atn_a",
            thread_id: "ask_a",
            question: "First question",
            answer: "First answer",
            tool_activity: [
              {
                callID: "call_1",
                tool: "read",
                status: "completed",
                input: '{"filePath":"README.md"}',
                output: "contents",
              },
            ],
            created_at: 3,
          },
          {
            id: "atn_c",
            thread_id: "ask_b",
            question: "Other thread",
            answer: "Other answer",
            tool_activity: [],
            created_at: 6,
          },
        ])
        .run()

      expect(
        yield* db
          .select({ id: SessionAskThreadTable.id })
          .from(SessionAskThreadTable)
          .orderBy(asc(SessionAskThreadTable.updated_at), asc(SessionAskThreadTable.id))
          .all(),
      ).toEqual([{ id: "ask_a" }, { id: "ask_b" }])
      expect(
        yield* db
          .select({ id: SessionAskTurnTable.id, toolActivity: SessionAskTurnTable.tool_activity })
          .from(SessionAskTurnTable)
          .where(eq(SessionAskTurnTable.thread_id, "ask_a"))
          .orderBy(asc(SessionAskTurnTable.id))
          .all(),
      ).toEqual([
        {
          id: "atn_a",
          toolActivity: [
            {
              callID: "call_1",
              tool: "read",
              status: "completed",
              input: '{"filePath":"README.md"}',
              output: "contents",
            },
          ],
        },
        { id: "atn_b", toolActivity: [] },
      ])
      expect(yield* db.select({ updated: SessionTable.time_updated }).from(SessionTable).get()).toEqual({ updated: 20 })
      expect(
        (yield* db.all<{ name: string }>(sql`PRAGMA index_info(session_ask_thread_session_updated_id_idx)`)).map(
          (column) => column.name,
        ),
      ).toEqual(["session_id", "updated_at", "id"])
      expect(
        (yield* db.all<{ name: string }>(sql`PRAGMA index_info(session_ask_turn_thread_id_id_idx)`)).map(
          (column) => column.name,
        ),
      ).toEqual(["thread_id", "id"])

      yield* db.delete(SessionAskThreadTable).where(eq(SessionAskThreadTable.id, "ask_a")).run()
      expect(yield* db.select({ id: SessionAskTurnTable.id }).from(SessionAskTurnTable).all()).toEqual([
        { id: "atn_c" },
      ])

      yield* db.delete(SessionTable).where(eq(SessionTable.id, sessionID)).run()
      expect(yield* db.select().from(SessionAskThreadTable).all()).toEqual([])
      expect(yield* db.select().from(SessionAskTurnTable).all()).toEqual([])
    }),
  )

  it.effect("rejects unbounded titles and invalid or unbounded tool activity", () =>
    Effect.gen(function* () {
      yield* setup
      const { db } = yield* Database.Service
      yield* db
        .insert(SessionAskThreadTable)
        .values({
          id: "ask_valid",
          session_id: sessionID,
          title: "x".repeat(SESSION_ASK_TITLE_MAX_CHARS),
          created_at: 1,
          updated_at: 1,
        })
        .run()

      expect(
        Exit.isFailure(
          yield* db
            .insert(SessionAskThreadTable)
            .values({
              id: "ask_title_too_long",
              session_id: sessionID,
              title: "x".repeat(SESSION_ASK_TITLE_MAX_CHARS + 1),
              created_at: 1,
              updated_at: 1,
            })
            .run()
            .pipe(Effect.exit),
        ),
      ).toBe(true)
      expect(
        Exit.isFailure(
          yield* db
            .run(
              sql`
              INSERT INTO session_ask_turn (id, thread_id, question, answer, tool_activity, created_at)
              VALUES ('atn_invalid_json', 'ask_valid', 'Question', 'Answer', 'not-json', 1)
            `,
            )
            .pipe(Effect.exit),
        ),
      ).toBe(true)
      expect(
        Exit.isFailure(
          yield* db
            .run(
              sql`
              INSERT INTO session_ask_turn (id, thread_id, question, answer, tool_activity, created_at)
              VALUES ('atn_not_array', 'ask_valid', 'Question', 'Answer', '{}', 1)
            `,
            )
            .pipe(Effect.exit),
        ),
      ).toBe(true)
      expect(
        Exit.isFailure(
          yield* db
            .insert(SessionAskTurnTable)
            .values({
              id: "atn_too_large",
              thread_id: "ask_valid",
              question: "Question",
              answer: "Answer",
              tool_activity: [
                {
                  callID: "call_large",
                  tool: "read",
                  status: "completed",
                  input: "{}",
                  output: "x".repeat(SESSION_ASK_TOOL_ACTIVITY_MAX_BYTES),
                },
              ],
              created_at: 1,
            })
            .run()
            .pipe(Effect.exit),
        ),
      ).toBe(true)
    }),
  )
})
