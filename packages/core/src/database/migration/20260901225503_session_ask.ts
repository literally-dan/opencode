import { Effect } from "effect"
import type { DatabaseMigration } from "../migration"

export default {
  id: "20260901225503_session_ask",
  up(tx) {
    return Effect.gen(function* () {
      yield* tx.run(`
        CREATE TABLE \`session_ask_thread\` (
          \`id\` text PRIMARY KEY,
          \`session_id\` text NOT NULL,
          \`title\` text NOT NULL,
          \`created_at\` integer NOT NULL,
          \`updated_at\` integer NOT NULL,
          CONSTRAINT \`fk_session_ask_thread_session_id_session_id_fk\` FOREIGN KEY (\`session_id\`) REFERENCES \`session\`(\`id\`) ON DELETE CASCADE,
          CONSTRAINT "session_ask_thread_title_length_check" CHECK(length("title") <= 256)
        );
      `)
      yield* tx.run(`
        CREATE TABLE \`session_ask_turn\` (
          \`id\` text PRIMARY KEY,
          \`thread_id\` text NOT NULL,
          \`question\` text NOT NULL,
          \`answer\` text NOT NULL,
          \`tool_activity\` text NOT NULL,
          \`created_at\` integer NOT NULL,
          CONSTRAINT \`fk_session_ask_turn_thread_id_session_ask_thread_id_fk\` FOREIGN KEY (\`thread_id\`) REFERENCES \`session_ask_thread\`(\`id\`) ON DELETE CASCADE,
          CONSTRAINT "session_ask_turn_tool_activity_json_check" CHECK(json_valid("tool_activity") AND json_type("tool_activity") = 'array'),
          CONSTRAINT "session_ask_turn_tool_activity_size_check" CHECK(length(cast("tool_activity" as blob)) <= 1048576)
        );
      `)
      yield* tx.run(
        `CREATE INDEX \`session_ask_thread_session_updated_id_idx\` ON \`session_ask_thread\` (\`session_id\`,\`updated_at\`,\`id\`);`,
      )
      yield* tx.run(
        `CREATE INDEX \`session_ask_turn_thread_id_id_idx\` ON \`session_ask_turn\` (\`thread_id\`,\`id\`);`,
      )
    })
  },
} satisfies DatabaseMigration.Migration
