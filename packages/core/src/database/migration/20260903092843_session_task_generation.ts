import { Effect } from "effect"
import type { DatabaseMigration } from "../migration"

export default {
  id: "20260903092843_session_task_generation",
  up(tx) {
    return Effect.gen(function* () {
      yield* tx.run(`ALTER TABLE \`session_task\` ADD \`generation\` integer DEFAULT 1 NOT NULL;`)
    })
  },
} satisfies DatabaseMigration.Migration
