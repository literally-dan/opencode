import { Effect } from "effect"
import type { DatabaseMigration } from "../migration"

export default {
  id: "20260903084930_session_task",
  up(tx) {
    return Effect.gen(function* () {
      yield* tx.run(`
        CREATE TABLE \`session_task\` (
          \`session_id\` text PRIMARY KEY,
          \`ancestor_access\` text DEFAULT 'history' NOT NULL,
          \`status\` text DEFAULT 'running' NOT NULL,
          \`error\` text,
          \`time_created\` integer NOT NULL,
          \`time_updated\` integer NOT NULL,
          \`time_completed\` integer,
          \`delivery_session_id\` text,
          \`time_delivered\` integer,
          CONSTRAINT \`fk_session_task_session_id_session_id_fk\` FOREIGN KEY (\`session_id\`) REFERENCES \`session\`(\`id\`) ON DELETE CASCADE
        );
      `)
    })
  },
} satisfies DatabaseMigration.Migration
