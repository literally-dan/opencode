import { Effect } from "effect"
import type { DatabaseMigration } from "../migration"

export default {
  id: "20261001220119_session_root_index",
  up(tx) {
    return Effect.gen(function* () {
      yield* tx.run(
        `CREATE INDEX \`session_root_idx\` ON \`session\` (\`project_id\`,\`time_updated\`) WHERE parent_id is null;`,
      )
    })
  },
} satisfies DatabaseMigration.Migration
