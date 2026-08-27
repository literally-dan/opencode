import { Effect } from "effect"
import type { DatabaseMigration } from "../migration"

export default {
  id: "20260827164832_session_task_parent",
  up(tx) {
    return Effect.gen(function* () {
      yield* tx.run(`ALTER TABLE \`session\` ADD \`task_parent_id\` text;`)
      yield* tx.run(`CREATE INDEX \`session_task_parent_idx\` ON \`session\` (\`task_parent_id\`);`)
    })
  },
} satisfies DatabaseMigration.Migration
