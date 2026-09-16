// Subprocess integration tests for `opencode run` (non-interactive mode).
// These exercise the real CLI binary against a TestLLMServer running in the
// same process. See `test/lib/cli-process.ts` for the harness — each test uses
// `opencode.run(message, opts?)` to spawn `bun src/index.ts run ...` with
// `OPENCODE_CONFIG_CONTENT` providing the test provider config inline.
import { describe, expect } from "bun:test"
import { createOpencodeClient } from "@opencode-ai/sdk/v2"
import { Effect } from "effect"
import { reply, type Reply } from "../../lib/llm-server"
import { cliIt, type CliFixture } from "../../lib/cli-process"
import { awaitWithTimeout, pollWithTimeout } from "../../lib/effect"
import { testProviderConfig } from "../../lib/test-provider"

// Queues a root turn that starts one background task and then answers "launched".
// The child reply decides how the task ends.
function queueBackgroundTask(llm: CliFixture["llm"], child: Reply) {
  return Effect.gen(function* () {
    yield* llm.pushMatch(
      (hit) => JSON.stringify(hit.body).includes("delegate in the background"),
      reply().tool("task", {
        description: "inspect bug",
        prompt: "Inspect the background path.",
        subagent_type: "general",
      }),
    )
    // Only the child prompt has a subagent-context marker with a parent session ID.
    yield* llm.pushMatch((hit) => JSON.stringify(hit.body).includes("subagent-context parent-session-id="), child)
    yield* llm.textMatch((hit) => JSON.stringify(hit.body).includes("Background task started"), "launched")
  })
}

// Completes a short time after the model received the root turn that follows the task start,
// so the run has printed that turn and waits for the background result.
function rootTurnAnswered(llm: CliFixture["llm"]) {
  return pollWithTimeout(
    llm.hits.pipe(
      Effect.map((hits) =>
        hits.some((hit) => JSON.stringify(hit.body).includes("Background task started")) ? true : undefined,
      ),
    ),
    "root turn never reported the background task start",
    "30 seconds",
  ).pipe(Effect.andThen(Effect.sleep("2 seconds")))
}

// Forwards requests to the server. Event stream number `n` (from 1) closes when `cut(n)` resolves. Event
// stream requests after the first wait for `reopen`, so a test can change the session while the run is
// disconnected.
function eventStreamProxy(
  target: string,
  input: { cut: (stream: number) => Promise<void> | undefined; reopen?: Promise<void> },
) {
  const seen = { streams: 0, statusChecks: 0 }
  return Effect.acquireRelease(
    Effect.sync(() => {
      const server = Bun.serve({
        port: 0,
        idleTimeout: 0,
        async fetch(request) {
          const url = new URL(request.url)
          const stream = url.pathname === "/event" ? ++seen.streams : undefined
          if (url.pathname === "/session/status") seen.statusChecks += 1
          if (stream !== undefined && stream > 1) await input.reopen
          const headers = new Headers(request.headers)
          headers.delete("host")
          const upstream = await fetch(new URL(url.pathname + url.search, target), {
            method: request.method,
            headers,
            body: request.method === "GET" || request.method === "HEAD" ? undefined : await request.arrayBuffer(),
            signal: request.signal,
          })
          const cut = stream === undefined ? undefined : input.cut(stream)
          if (!cut || !upstream.body) return upstream
          const reader = upstream.body.getReader()
          const body = new ReadableStream<Uint8Array>({
            start(controller) {
              void cut.then(() => {
                reader.cancel().catch(() => {})
                controller.close()
              })
              void (async () => {
                while (true) {
                  const next = await reader.read().catch(() => ({ done: true as const, value: undefined }))
                  if (next.done) return
                  try {
                    controller.enqueue(next.value)
                  } catch {
                    return
                  }
                }
              })()
            },
          })
          return new Response(body, { status: upstream.status, headers: upstream.headers })
        },
      })
      return { url: server.url.origin, seen, stop: () => server.stop(true) }
    }),
    (proxy) => Effect.promise(() => proxy.stop()),
  )
}

describe("opencode run (non-interactive subprocess)", () => {
  // Happy path: prompt completes, output reaches stdout, process exits 0.
  // If this fails, all the others likely will too — debug here first.
  cliIt.concurrent(
    "exits 0 and writes the response to stdout on a successful prompt",
    ({ llm, opencode }) =>
      Effect.gen(function* () {
        yield* llm.text("hello from the test llm")
        const result = yield* opencode.run("say hi")
        opencode.expectExit(result, 0)
        expect(result.stdout).toBe("hello from the test llm\n")
      }),
    60_000,
  )

  cliIt.concurrent(
    "prints each completed text part in order around a tool continuation",
    ({ llm, opencode }) =>
      Effect.gen(function* () {
        yield* llm.push(
          reply().text("  before tool  ").tool("bash", {
            command: "printf tool-output",
            description: "Print deterministic output",
          }),
        )
        yield* llm.text("  after tool  ")

        const result = yield* opencode.run("use a tool", {
          extraArgs: ["--dangerously-skip-permissions"],
        })

        opencode.expectExit(result, 0)
        expect(result.stdout).toBe("before tool\nafter tool\n")
      }),
    60_000,
  )

  cliIt.concurrent(
    "prints reasoning before text only with --thinking",
    ({ llm, opencode }) =>
      Effect.gen(function* () {
        yield* llm.reason("  considering  ", { text: "  answer  " })
        const thinking = yield* opencode.run("think", { extraArgs: ["--thinking"] })
        opencode.expectExit(thinking, 0)
        expect(thinking.stdout).toBe("Thinking: considering\nanswer\n")

        yield* llm.reason("hidden", { text: "visible" })
        const plain = yield* opencode.run("think again")
        opencode.expectExit(plain, 0)
        expect(plain.stdout).toBe("visible\n")
      }),
    60_000,
  )

  // Regression for #27371: an unknown model used to hang the process forever
  // waiting on a session.status === idle event that never arrived. The fix
  // makes the SDK call surface an error promptly so the process exits nonzero.
  // We assert nonzero exit AND wall-clock under the harness timeout — a hang
  // would expire the timeout and produce a different (signal-killed) failure.
  cliIt.concurrent(
    "exits nonzero promptly when the model is unknown (regression for #27371)",
    ({ opencode }) =>
      Effect.gen(function* () {
        const result = yield* opencode.run("say hi", {
          model: "test/nonexistent-model",
          timeoutMs: 15_000,
        })
        expect(result.exitCode).not.toBe(0)
        expect(result.durationMs).toBeLessThan(15_000)
      }),
    30_000,
  )

  // The test provider's SSE error item is interpreted by the SDK as an unknown
  // finish, not a fatal provider/session error. Unknown finishes should continue
  // the prompt loop so a subsequent response can complete the run.
  cliIt.concurrent(
    "unknown stream finish preserves partial output and continues",
    ({ llm, opencode }) =>
      Effect.gen(function* () {
        yield* llm.push(
          reply().text("partial response").tool("bash", {
            command: "printf tool",
            description: "Print deterministic output",
          }),
        )
        yield* llm.fail("upstream provider exploded mid-stream")
        yield* llm.text("recovered")
        const result = yield* opencode.run("trigger midstream error", { timeoutMs: 30_000 })
        expect(result.exitCode).toBe(0)
        expect(result.stdout).toBe("partial response\nrecovered\n")
        expect(result.stderr).not.toContain("upstream provider exploded mid-stream")
      }),
    60_000,
  )

  // --format json puts one JSON object per line on stdout for each emitted
  // event. Consumers (CI scripts, tooling) parse this stream. Asserts the
  // shape so a future event-emit change has to update this expectation.
  cliIt.concurrent(
    "--format json emits parseable line-delimited JSON to stdout",
    ({ llm, opencode }) =>
      Effect.gen(function* () {
        yield* llm.text("structured output")
        const result = yield* opencode.run("say hi", { format: "json" })
        opencode.expectExit(result, 0)

        const events = opencode.parseJsonEvents(result.stdout)
        expect(events.length).toBeGreaterThan(0)
        for (const evt of events) {
          expect(typeof evt.type).toBe("string")
          expect(typeof evt.sessionID).toBe("string")
        }
        expect(events.map((event) => event.type)).toEqual(["step_start", "text", "step_finish"])
        expect(events.map(({ timestamp: _, sessionID: __, ...event }) => event)).toEqual([
          { type: "step_start", part: expect.objectContaining({ type: "step-start" }) },
          {
            type: "text",
            part: expect.objectContaining({ type: "text", text: "structured output" }),
          },
          { type: "step_finish", part: expect.objectContaining({ type: "step-finish" }) },
        ])
        expect(result.stdout.endsWith("\n")).toBe(true)
        expect(
          result.stdout
            .split("\n")
            .slice(0, -1)
            .every((line) => line.length > 0),
        ).toBe(true)
      }),
    60_000,
  )

  cliIt.concurrent(
    "--format json emits a pure error record for a rejected prompt request",
    ({ opencode }) =>
      Effect.gen(function* () {
        const result = yield* opencode.run("use an unknown model", {
          model: "test/nonexistent-model",
          format: "json",
        })

        expect(result.exitCode).not.toBe(0)
        const events = opencode.parseJsonEvents(result.stdout)
        expect(events.map((event) => event.type)).toEqual(["error"])
        expect(events[0]).toEqual({
          type: "error",
          timestamp: expect.any(Number),
          sessionID: expect.any(String),
          error: expect.any(Object),
        })
        expect(result.stdout.split("\n").filter(Boolean)).toHaveLength(1)
      }),
    30_000,
  )

  cliIt.concurrent(
    "--format json preserves reasoning, tool, and continuation ordering",
    ({ llm, opencode }) =>
      Effect.gen(function* () {
        yield* llm.push(
          reply().reason("reasoning").text("before").tool("bash", {
            command: "printf tool",
            description: "Print deterministic output",
          }),
        )
        yield* llm.text("after")

        const result = yield* opencode.run("exercise json records", {
          format: "json",
          extraArgs: ["--thinking", "--dangerously-skip-permissions"],
        })

        expect(result.exitCode).toBe(0)
        const events = opencode.parseJsonEvents(result.stdout)
        expect(events.map((event) => event.type)).toEqual([
          "step_start",
          "reasoning",
          "text",
          "tool_use",
          "step_finish",
          "step_start",
          "text",
          "step_finish",
        ])
        expect(events.find((event) => event.type === "reasoning")?.part).toEqual(
          expect.objectContaining({ type: "reasoning", text: "reasoning" }),
        )
        expect(events.find((event) => event.type === "tool_use")?.part).toEqual(
          expect.objectContaining({
            type: "tool",
            tool: "bash",
            state: expect.objectContaining({ status: "completed" }),
          }),
        )
        expect(
          result.stdout
            .split("\n")
            .slice(0, -1)
            .every((line) => line.startsWith("{")),
        ).toBe(true)
      }),
    60_000,
  )

  cliIt.concurrent(
    "--format json records an unknown stream finish and continuation",
    ({ llm, opencode }) =>
      Effect.gen(function* () {
        yield* llm.push(
          reply().text("partial json").tool("bash", {
            command: "printf tool",
            description: "Print deterministic output",
          }),
        )
        yield* llm.fail("provider failed")
        yield* llm.text("recovered")
        const result = yield* opencode.run("fail after output", { format: "json" })

        const events = opencode.parseJsonEvents(result.stdout)
        expect(result.exitCode).toBe(0)
        expect(events.map((event) => event.type)).toEqual([
          "step_start",
          "text",
          "tool_use",
          "step_finish",
          "step_start",
          "step_finish",
          "step_start",
          "text",
          "step_finish",
        ])
        expect(events[1]?.part).toEqual(expect.objectContaining({ type: "text", text: "partial json" }))
        expect(events[5]?.part).toEqual(expect.objectContaining({ type: "step-finish", reason: "unknown" }))
        expect(events[7]?.part).toEqual(expect.objectContaining({ type: "text", text: "recovered" }))
        expect(events.at(-1)?.part).toEqual(expect.objectContaining({ type: "step-finish", reason: "stop" }))
      }),
    60_000,
  )

  cliIt.concurrent(
    "rejects requested permissions by default and allows them with the dangerous flag",
    ({ home, llm, opencode }) =>
      Effect.gen(function* () {
        yield* llm.tool("bash", { command: "rm -f denied-file", description: "Remove a test file" })
        yield* llm.text("continued after rejection")
        const denied = yield* opencode.run("request permission", { permission: { bash: "ask" } })
        opencode.expectExit(denied, 0)
        expect(denied.stderr).toContain("permission requested: bash")
        expect(denied.stdout).toBe("")

        yield* llm.reset
        yield* llm.tool("bash", { command: "rm -f allowed-file", description: "Remove a test file" })
        yield* llm.text("continued after approval")
        const allowed = yield* opencode.run("request permission", {
          permission: { bash: "ask" },
          extraArgs: ["--dangerously-skip-permissions"],
        })
        opencode.expectExit(allowed, 0)
        expect(allowed.stderr).not.toContain("permission requested: bash")
        expect(allowed.stdout).toContain("continued after approval")

        yield* llm.reset
        yield* llm.tool("bash", { command: "touch explicitly-denied", description: "Create a denied marker" })
        yield* llm.text("continued after explicit denial")
        const explicitlyDenied = yield* opencode.run("request denied permission", {
          permission: { bash: "deny" },
          extraArgs: ["--dangerously-skip-permissions"],
        })
        opencode.expectExit(explicitlyDenied, 0)
        expect(explicitlyDenied.stdout).toContain("continued after explicit denial")
        expect(yield* Effect.promise(() => Bun.file(`${home}/explicitly-denied`).exists())).toBe(false)
      }),
    60_000,
  )

  cliIt.concurrent(
    "answers permission requests from nested Task sessions",
    ({ home, llm, opencode }) =>
      Effect.gen(function* () {
        yield* llm.push(
          reply().tool("task", {
            description: "delegate once",
            prompt: "Delegate the marker file to another agent.",
            subagent_type: "nested",
          }),
          reply().tool("task", {
            description: "delegate twice",
            prompt: "Create the nested marker file.",
            subagent_type: "nested",
          }),
          reply().tool("bash", {
            command: `touch ${JSON.stringify(`${home}/nested-approved`)}`,
            description: "Create the nested marker",
          }),
          reply().text("grandchild completed"),
          reply().text("child completed"),
          reply().text("root completed"),
        )

        const result = yield* opencode.run("delegate this task twice", {
          extraArgs: ["--auto"],
          timeoutMs: 60_000,
          env: {
            OPENCODE_CONFIG_CONTENT: JSON.stringify({
              ...testProviderConfig(llm.url),
              // Required: the default of 1 stops the child from delegating, so
              // the bash permission would come from the child and the test would
              // never exercise a grandchild at all.
              subagent_depth: 2,
              permission: { task: "allow", bash: "ask" },
              agent: {
                nested: {
                  description: "Delegates nested test work.",
                  mode: "subagent",
                  permission: { task: "allow", bash: "ask" },
                },
              },
            }),
          },
        })

        opencode.expectExit(result, 0)
        expect(yield* Effect.promise(() => Bun.file(`${home}/nested-approved`).exists())).toBe(true)
        // Only a grandchild is ever prompted with the second delegation, so its
        // presence proves the permission came from depth 2 rather than the child.
        const inputs = JSON.stringify(yield* llm.inputs)
        expect(inputs).toContain("Create the nested marker file.")
        expect(inputs).not.toContain("depth limit")
      }),
    60_000,
  )

  cliIt.concurrent(
    "waits for background task results before exiting",
    ({ llm, opencode }) =>
      Effect.gen(function* () {
        // Release the child only after the root turn that launched it has been answered,
        // so the run must stay attached until the background result is delivered.
        yield* queueBackgroundTask(
          llm,
          reply()
            .wait(Effect.runPromise(rootTurnAnswered(llm).pipe(Effect.ignore)))
            .text("child result")
            .stop(),
        )
        yield* llm.textMatch(
          (hit) => JSON.stringify(hit.body).includes("Background task completed"),
          "final answer after background task",
        )

        const result = yield* opencode.run("delegate in the background", {
          timeoutMs: 45_000,
          env: {
            OPENCODE_CONFIG_CONTENT: JSON.stringify({
              ...testProviderConfig(llm.url),
              permission: { task: "allow" },
            }),
          },
        })

        opencode.expectExit(result, 0)
        expect(result.stdout).toBe("launched\nfinal answer after background task\n")
      }),
    60_000,
  )

  cliIt.live(
    "attach mode answers permissions raised after the run starts",
    ({ home, llm, opencode }) =>
      Effect.gen(function* () {
        const config = {
          ...testProviderConfig(llm.url),
          permission: { bash: "ask" },
        }
        yield* llm.push(
          reply().tool("bash", {
            command: `touch ${JSON.stringify(`${home}/attached-approved`)}`,
            description: "Create the attached marker",
          }),
          reply().text("attached completed"),
        )
        const server = yield* opencode.serve({
          env: { OPENCODE_CONFIG_CONTENT: JSON.stringify(config) },
        })
        const sdk = createOpencodeClient({ baseUrl: server.url, directory: home })
        const root = yield* Effect.promise(() => sdk.session.create({ title: "root" }))
        if (!root.data) return yield* Effect.fail(new Error("failed to create root session"))

        const result = yield* opencode.run("create the marker", {
          extraArgs: ["--attach", server.url, "--session", root.data.id, "--auto", "--"],
          timeoutMs: 60_000,
          env: { OPENCODE_CONFIG_CONTENT: JSON.stringify(config) },
        })

        opencode.expectExit(result, 0)
        expect(yield* Effect.promise(() => Bun.file(`${home}/attached-approved`).exists())).toBe(true)
      }),
    90_000,
  )

  cliIt.live(
    "attach mode answers a permission raised before its event stream connects",
    ({ home, llm, opencode }) =>
      Effect.gen(function* () {
        const marker = `${home}/late-stream-approved`
        const config = {
          ...testProviderConfig(llm.url),
          permission: { bash: "ask" },
        }
        yield* llm.push(
          reply().tool("bash", {
            command: `touch ${JSON.stringify(marker)}`,
            description: "Create the late stream marker",
          }),
          reply().text("late stream completed"),
        )
        const server = yield* opencode.serve({
          env: { OPENCODE_CONFIG_CONTENT: JSON.stringify(config) },
        })
        const sdk = createOpencodeClient({ baseUrl: server.url, directory: home })
        const root = yield* Effect.promise(() => sdk.session.create({ title: "root" }))
        if (!root.data) return yield* Effect.fail(new Error("failed to create root session"))

        // The proxy holds the run's event stream until the permission is pending. The run lists
        // permissions before it subscribes, so only recovery on connection can answer this request.
        const stream = Promise.withResolvers<void>()
        const proxy = yield* Effect.acquireRelease(
          Effect.sync(() =>
            Bun.serve({
              port: 0,
              idleTimeout: 0,
              async fetch(request) {
                const url = new URL(request.url)
                if (url.pathname === "/event") await stream.promise
                const headers = new Headers(request.headers)
                headers.delete("host")
                return fetch(new URL(url.pathname + url.search, server.url), {
                  method: request.method,
                  headers,
                  body: request.method === "GET" || request.method === "HEAD" ? undefined : await request.arrayBuffer(),
                  signal: request.signal,
                })
              },
            }),
          ),
          (proxy) => Effect.promise(() => proxy.stop(true)),
        )

        const run = yield* opencode.startRun("create the marker", {
          extraArgs: ["--attach", proxy.url.origin, "--session", root.data.id, "--auto", "--"],
          env: { OPENCODE_CONFIG_CONTENT: JSON.stringify(config) },
        })
        yield* pollWithTimeout(
          Effect.promise(() => sdk.permission.list()).pipe(
            Effect.map((response) => response.data?.find((permission) => permission.sessionID === root.data.id)),
          ),
          "the attached turn never raised its permission",
          "30 seconds",
        )
        expect(yield* Effect.promise(() => Bun.file(marker).exists())).toBe(false)

        stream.resolve()
        const result = yield* awaitWithTimeout(
          run.result,
          "attached run did not answer the permission after its event stream connected",
          "45 seconds",
        )

        opencode.expectExit(result, 0)
        expect(result.stdout).toContain("late stream completed")
        expect(yield* Effect.promise(() => Bun.file(marker).exists())).toBe(true)
      }),
    90_000,
  )

  cliIt.live(
    "attach mode leaves permissions pending before the run for a human",
    ({ home, llm, opencode }) =>
      Effect.gen(function* () {
        const marker = `${home}/preexisting-approved`
        const config = {
          ...testProviderConfig(llm.url),
          permission: { task: "allow", bash: "ask" },
        }
        const body = (hit: { body: unknown }) => JSON.stringify(hit.body)
        yield* llm.pushMatch(
          (hit) => body(hit).includes("start the preexisting task"),
          reply().tool("task", {
            description: "create marker",
            prompt: "Create the preexisting marker.",
            subagent_type: "general",
          }),
        )
        yield* llm.pushMatch(
          (hit) => body(hit).includes("subagent-context parent-session-id="),
          reply().tool("bash", {
            command: `touch ${JSON.stringify(marker)}`,
            description: "Create the preexisting marker",
          }),
        )
        yield* llm.textMatch((hit) => body(hit).includes("continue from the root"), "attached completed")
        yield* llm.textMatch((hit) => body(hit).includes("Background task started"), "launched")
        yield* llm.textMatch((hit) => body(hit).includes("Background task completed"), "approved by a human")
        const server = yield* opencode.serve({
          env: { OPENCODE_CONFIG_CONTENT: JSON.stringify(config) },
        })
        const sdk = createOpencodeClient({ baseUrl: server.url, directory: home })
        const root = yield* Effect.promise(() => sdk.session.create({ title: "root" }))
        if (!root.data) return yield* Effect.fail(new Error("failed to create root session"))
        const admission = yield* Effect.promise(() =>
          sdk.session.promptAsync({
            sessionID: root.data.id,
            agent: "build",
            model: { providerID: "test", modelID: "test-model" },
            parts: [{ type: "text", text: "start the preexisting task" }],
          }),
        )
        if (admission.error) return yield* Effect.fail(new Error("failed to admit root prompt"))

        // The background child asks while the root turn is free to accept the attached prompt.
        const pending = yield* pollWithTimeout(
          Effect.promise(() => sdk.permission.list()).pipe(
            Effect.map((response) => response.data?.find((permission) => permission.sessionID !== root.data.id)),
          ),
          "child permission never became pending",
          "15 seconds",
        )

        const run = yield* opencode.startRun("continue from the root", {
          extraArgs: ["--attach", server.url, "--session", root.data.id, "--auto", "--"],
          env: { OPENCODE_CONFIG_CONTENT: JSON.stringify(config) },
        })
        yield* pollWithTimeout(
          llm.hits.pipe(
            Effect.map((hits) => (hits.some((hit) => body(hit).includes("continue from the root")) ? true : undefined)),
          ),
          "attached turn never reached the model",
          "30 seconds",
        )
        yield* Effect.sleep("2 seconds")
        const remaining = yield* Effect.promise(() => sdk.permission.list())
        expect(remaining.data?.map((permission) => permission.id)).toContain(pending.id)
        expect(yield* Effect.promise(() => Bun.file(marker).exists())).toBe(false)

        // The attached run waits for the background result, which needs the human reply.
        yield* Effect.promise(() => sdk.permission.reply({ requestID: pending.id, reply: "once" }))
        const result = yield* awaitWithTimeout(run.result, "attached run did not exit after delivery", "45 seconds")

        opencode.expectExit(result, 0)
        expect(result.stdout).toContain("attached completed")
        expect(result.stdout).toContain("approved by a human")
        expect(result.stderr).toContain("waiting for a human to answer permission bash")
        expect(result.stderr.match(/waiting for a human to answer/g)).toHaveLength(1)
        expect(yield* Effect.promise(() => Bun.file(marker).exists())).toBe(true)
      }),
    90_000,
  )

  cliIt.live(
    "attach mode waits for background task results before exiting",
    ({ llm, opencode }) =>
      Effect.gen(function* () {
        const config = { ...testProviderConfig(llm.url), permission: { task: "allow" } }
        yield* queueBackgroundTask(
          llm,
          reply()
            .wait(Effect.runPromise(rootTurnAnswered(llm).pipe(Effect.ignore)))
            .text("child result")
            .stop(),
        )
        yield* llm.textMatch(
          (hit) => JSON.stringify(hit.body).includes("Background task completed"),
          "final answer after background task",
        )
        const server = yield* opencode.serve({
          env: { OPENCODE_CONFIG_CONTENT: JSON.stringify(config) },
        })

        const result = yield* opencode.run("delegate in the background", {
          extraArgs: ["--attach", server.url, "--"],
          timeoutMs: 60_000,
          env: { OPENCODE_CONFIG_CONTENT: JSON.stringify(config) },
        })

        opencode.expectExit(result, 0)
        expect(result.stdout).toBe("launched\nfinal answer after background task\n")
      }),
    90_000,
  )

  cliIt.live(
    "attach mode reconnects a dropped event stream and prints the background result",
    ({ llm, opencode }) =>
      Effect.gen(function* () {
        const config = { ...testProviderConfig(llm.url), permission: { task: "allow" } }
        const release = Promise.withResolvers<void>()
        yield* queueBackgroundTask(llm, reply().wait(release.promise).text("child result").stop())
        yield* llm.textMatch(
          (hit) => JSON.stringify(hit.body).includes("Background task completed"),
          "final answer after background task",
        )
        const server = yield* opencode.serve({
          env: { OPENCODE_CONFIG_CONTENT: JSON.stringify(config) },
        })
        // End the waiting child request before the server stops, even when an assertion fails.
        yield* Effect.addFinalizer(() => Effect.sync(() => release.resolve()))
        const cut = Promise.withResolvers<void>()
        const proxy = yield* eventStreamProxy(server.url, { cut: (stream) => (stream === 1 ? cut.promise : undefined) })
        const run = yield* opencode.startRun("delegate in the background", {
          extraArgs: ["--attach", proxy.url, "--"],
          env: { OPENCODE_CONFIG_CONTENT: JSON.stringify(config) },
        })
        yield* rootTurnAnswered(llm)

        cut.resolve()
        yield* pollWithTimeout(
          Effect.sync(() => (proxy.seen.statusChecks > 0 ? true : undefined)),
          "attached run never checked the session status after reconnecting",
          "20 seconds",
        )
        release.resolve()
        const result = yield* awaitWithTimeout(run.result, "attached run did not exit after reconnecting", "45 seconds")

        opencode.expectExit(result, 0)
        expect(proxy.seen.streams).toBe(2)
        expect(result.stdout).toBe("launched\nfinal answer after background task\n")
        expect(result.stderr).toContain("event stream disconnected; reconnecting (1/5)")
      }),
    90_000,
  )

  cliIt.live(
    "attach mode keeps reconnecting while each new event stream works",
    ({ llm, opencode }) =>
      Effect.gen(function* () {
        const config = { ...testProviderConfig(llm.url), permission: { task: "allow" } }
        const release = Promise.withResolvers<void>()
        yield* queueBackgroundTask(llm, reply().wait(release.promise).text("child result").stop())
        yield* llm.textMatch(
          (hit) => JSON.stringify(hit.body).includes("Background task completed"),
          "final answer after background task",
        )
        const server = yield* opencode.serve({
          env: { OPENCODE_CONFIG_CONTENT: JSON.stringify(config) },
        })
        // End the waiting child request before the server stops, even when an assertion fails.
        yield* Effect.addFinalizer(() => Effect.sync(() => release.resolve()))
        // A proxy that cuts long connections drops the stream more often than the reconnect budget allows.
        const cuts = Array.from({ length: 6 }, () => Promise.withResolvers<void>())
        const proxy = yield* eventStreamProxy(server.url, { cut: (stream) => cuts[stream - 1]?.promise })
        const run = yield* opencode.startRun("delegate in the background", {
          extraArgs: ["--attach", proxy.url, "--"],
          env: { OPENCODE_CONFIG_CONTENT: JSON.stringify(config) },
        })
        yield* rootTurnAnswered(llm)

        for (const [index, cut] of cuts.entries()) {
          cut.resolve()
          yield* pollWithTimeout(
            Effect.sync(() => (proxy.seen.statusChecks > index ? true : undefined)),
            `attached run never checked the session status after reconnect ${index + 1}`,
            "20 seconds",
          )
        }
        release.resolve()
        const result = yield* awaitWithTimeout(run.result, "attached run did not exit after reconnecting", "45 seconds")

        opencode.expectExit(result, 0)
        expect(result.stdout).toBe("launched\nfinal answer after background task\n")
        expect(result.stderr.match(/event stream disconnected; reconnecting \(1\/5\)/g)).toHaveLength(6)
        expect(result.stderr).not.toContain("reconnecting (2/5)")
      }),
    90_000,
  )

  cliIt.live(
    "attach mode fails after reconnecting to a session that became idle while disconnected",
    ({ home, llm, opencode }) =>
      Effect.gen(function* () {
        const config = { ...testProviderConfig(llm.url), permission: { task: "allow" } }
        const release = Promise.withResolvers<void>()
        yield* queueBackgroundTask(llm, reply().wait(release.promise).text("child result").stop())
        yield* llm.textMatch(
          (hit) => JSON.stringify(hit.body).includes("Background task completed"),
          "final answer after background task",
        )
        const server = yield* opencode.serve({
          env: { OPENCODE_CONFIG_CONTENT: JSON.stringify(config) },
        })
        const sdk = createOpencodeClient({ baseUrl: server.url, directory: home })
        const cut = Promise.withResolvers<void>()
        const reopen = Promise.withResolvers<void>()
        // End the waiting requests before the server stops, even when an assertion fails.
        yield* Effect.addFinalizer(() =>
          Effect.sync(() => {
            release.resolve()
            reopen.resolve()
          }),
        )
        const proxy = yield* eventStreamProxy(server.url, {
          cut: (stream) => (stream === 1 ? cut.promise : undefined),
          reopen: reopen.promise,
        })
        const run = yield* opencode.startRun("delegate in the background", {
          extraArgs: ["--attach", proxy.url, "--"],
          env: { OPENCODE_CONFIG_CONTENT: JSON.stringify(config) },
        })
        yield* rootTurnAnswered(llm)

        // The background result is delivered while the run has no event stream, so the run misses the idle event.
        cut.resolve()
        yield* pollWithTimeout(
          Effect.sync(() => (proxy.seen.streams > 1 ? true : undefined)),
          "attached run never tried to reconnect",
          "20 seconds",
        )
        release.resolve()
        yield* pollWithTimeout(
          Effect.promise(() => sdk.session.status()).pipe(
            Effect.map((response) => (Object.keys(response.data ?? {}).length === 0 ? true : undefined)),
          ),
          "server session never became idle",
          "30 seconds",
        )
        reopen.resolve()
        const result = yield* awaitWithTimeout(run.result, "attached run did not exit after reconnecting", "30 seconds")

        // The final answer and any session error were published while the run had no stream.
        opencode.expectExit(result, 1)
        expect(result.stdout).toStartWith("launched\n")
        expect(result.stderr).toContain("event stream disconnected; reconnecting (1/5)")
        expect(result.stderr).toContain(
          "session became idle while the event stream was disconnected; output from that time is missing",
        )
      }),
    90_000,
  )

  cliIt.live(
    "attach mode fails without reconnecting when the server disposes the instance",
    ({ home, llm, opencode }) =>
      Effect.gen(function* () {
        const config = { ...testProviderConfig(llm.url), permission: { task: "allow" } }
        const release = Promise.withResolvers<void>()
        yield* queueBackgroundTask(llm, reply().wait(release.promise).text("child result").stop())
        yield* llm.textMatch(
          (hit) => JSON.stringify(hit.body).includes("Background task completed"),
          "final answer after background task",
        )
        const server = yield* opencode.serve({
          env: { OPENCODE_CONFIG_CONTENT: JSON.stringify(config) },
        })
        // End the waiting child request before the server stops, even when an assertion fails.
        yield* Effect.addFinalizer(() => Effect.sync(() => release.resolve()))
        const run = yield* opencode.startRun("delegate in the background", {
          extraArgs: ["--attach", server.url, "--"],
          env: { OPENCODE_CONFIG_CONTENT: JSON.stringify(config) },
        })
        yield* rootTurnAnswered(llm)

        // Clients dispose the instance, for example after a provider login. That stops the session's work.
        const sdk = createOpencodeClient({ baseUrl: server.url, directory: home })
        yield* Effect.promise(() => sdk.instance.dispose())
        const result = yield* awaitWithTimeout(run.result, "attached run did not exit after disposal", "45 seconds")

        opencode.expectExit(result, 1)
        expect(result.stdout).toBe("launched\n")
        expect(result.stderr).toContain("the server disposed the instance before the session became idle")
        expect(result.stderr).not.toContain("reconnecting")
      }),
    90_000,
  )

  cliIt.live(
    "attach mode exits when the server stops while it waits for background results",
    ({ llm, opencode }) =>
      Effect.gen(function* () {
        const config = { ...testProviderConfig(llm.url), permission: { task: "allow" } }
        yield* queueBackgroundTask(llm, reply().hang())
        const server = yield* opencode.serve({
          env: { OPENCODE_CONFIG_CONTENT: JSON.stringify(config) },
        })
        const run = yield* opencode.startRun("delegate in the background", {
          extraArgs: ["--attach", server.url, "--"],
          env: { OPENCODE_CONFIG_CONTENT: JSON.stringify(config) },
        })
        yield* rootTurnAnswered(llm)

        server.kill()
        // The run uses its reconnect budget, about 12 seconds of backoff, before it gives up.
        const result = yield* awaitWithTimeout(
          run.result,
          "attached run did not exit after the server stopped",
          "40 seconds",
        )

        expect(result.exitCode).not.toBe(0)
        expect(result.stdout).toBe("launched\n")
        expect(result.stderr).toContain("event stream disconnected; reconnecting (5/5)")
        expect(result.stderr).toContain("lost the server event stream before the session became idle")
      }),
    90_000,
  )

  cliIt.live(
    "SIGINT interrupts an attached run that waits for background results",
    ({ llm, opencode }) =>
      Effect.gen(function* () {
        const config = { ...testProviderConfig(llm.url), permission: { task: "allow" } }
        yield* queueBackgroundTask(llm, reply().hang())
        const server = yield* opencode.serve({
          env: { OPENCODE_CONFIG_CONTENT: JSON.stringify(config) },
        })
        const run = yield* opencode.startRun("delegate in the background", {
          extraArgs: ["--attach", server.url, "--"],
          env: { OPENCODE_CONFIG_CONTENT: JSON.stringify(config) },
        })
        yield* rootTurnAnswered(llm)

        run.interrupt()
        const result = yield* awaitWithTimeout(run.result, "attached run did not exit after SIGINT", "20 seconds")

        expect(result.exitCode).not.toBe(0)
        // Stop the server before the model server closes, so the hanging child request ends.
        server.kill()
        yield* Effect.promise(() => server.exited)
      }),
    60_000,
  )

  cliIt.live(
    "attach mode sends client-local file contents without a shared path",
    ({ home, llm, opencode }) =>
      Effect.gen(function* () {
        const source = `${home}/client-only.txt`
        const sentinel = "client-only attachment sentinel"
        yield* Effect.promise(() => Bun.write(source, sentinel))
        yield* llm.text("attachment received")
        const server = yield* opencode.serve()

        const result = yield* opencode.run("read the attachment", {
          extraArgs: ["--attach", server.url, `--file=${source}`, "--"],
        })

        opencode.expectExit(result, 0)
        const input = JSON.stringify(yield* llm.inputs)
        expect(input).toContain(sentinel)
        expect(input).not.toContain(`file://${source}`)
      }),
    60_000,
  )

  cliIt.concurrent(
    "attach mode rejects local directories before prompt admission",
    ({ home, opencode }) =>
      Effect.gen(function* () {
        const result = yield* opencode.run("read the directory", {
          extraArgs: ["--attach", "http://127.0.0.1:1", `--file=${home}`, "--"],
        })

        expect(result.exitCode).not.toBe(0)
        expect(result.stderr).toContain("Cannot attach local directory without a shared filesystem")
      }),
    30_000,
  )

  cliIt.live(
    "SIGINT interrupts an active non-interactive run without leaking the process",
    ({ llm, opencode }) =>
      Effect.gen(function* () {
        yield* llm.hang
        const run = yield* opencode.startRun("wait forever")
        yield* llm.wait(1)
        run.interrupt()
        const result = yield* run.result

        expect(result.exitCode).not.toBe(0)
        expect(result.durationMs).toBeLessThan(30_000)
      }),
    30_000,
  )
})
