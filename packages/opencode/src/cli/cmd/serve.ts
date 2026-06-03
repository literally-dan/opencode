import { Effect } from "effect"
import { ServerDiscovery } from "@/cli/server-discovery"
import { effectCmd } from "../effect-cmd"
import { withNetworkOptions, resolveNetworkOptions } from "../network"
import { Flag } from "@opencode-ai/core/flag/flag"

export const ServeCommand = effectCmd({
  command: "serve",
  builder: (yargs) =>
    withNetworkOptions(yargs).option("discoverable", {
      type: "boolean",
      describe: "write this server to the local discovery file for default TUI startup",
      default: false,
    }),
  describe: "starts a headless opencode server",
  // Server loads instances per-request via x-opencode-directory header — no
  // need for an ambient project InstanceContext at startup.
  instance: false,
  handler: (args) =>
    Effect.fn("Cli.serve")(function* (args) {
      const { Server } = yield* Effect.promise(() => import("../../server/server"))
      if (!Flag.OPENCODE_SERVER_PASSWORD) {
        console.log("Warning: OPENCODE_SERVER_PASSWORD is not set; server is unsecured.")
      }
      const opts = yield* resolveNetworkOptions(args)
      const server = yield* Effect.promise(() => Server.listen(opts))
      const discovery = args.discoverable ? yield* ServerDiscovery.Service : undefined
      if (discovery) {
        yield* discovery.write(server.url)
        process.on("exit", ServerDiscovery.removeSync)
      }
      console.log(`opencode server listening on http://${server.hostname}:${server.port}`)

      yield* Effect.never.pipe(
        Effect.ensuring(
          discovery
            ? discovery.remove().pipe(Effect.ensuring(Effect.sync(() => process.off("exit", ServerDiscovery.removeSync))))
            : Effect.void,
        ),
      )
    })(args).pipe(Effect.provide(ServerDiscovery.defaultLayer)),
})
