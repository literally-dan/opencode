import { describe, expect } from "bun:test"
import { PermissionV1 } from "@opencode-ai/core/v1/permission"
import { SessionProjector } from "@opencode-ai/core/session/projector"
import { CrossSpawnSpawner } from "@opencode-ai/core/cross-spawn-spawner"
import { AppNodeBuilder } from "@opencode-ai/core/effect/app-node-builder"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { Cause, Effect, Exit, Fiber, Layer } from "effect"
import { sessionTreeRequestRejector } from "@/cli/cmd/github.requests"
import { RuntimeFlags } from "@/effect/runtime-flags"
import { EventV2Bridge } from "@/event-v2-bridge"
import { Permission } from "@/permission"
import { InstanceBootstrap } from "@/project/bootstrap"
import { InstanceStore } from "@/project/instance-store"
import { Question } from "@/question"
import { Session } from "@/session/session"
import type { SessionID } from "@/session/schema"
import { testEffect } from "../lib/effect"

const it = testEffect(
  AppNodeBuilder.build(
    LayerNode.group([
      Session.node,
      Permission.node,
      Question.node,
      EventV2Bridge.node,
      SessionProjector.node,
      CrossSpawnSpawner.node,
      InstanceStore.node,
    ]),
    [
      [RuntimeFlags.node, RuntimeFlags.layer({ experimentalWorkspaces: false })],
      [
        InstanceBootstrap.node,
        Layer.succeed(InstanceBootstrap.Service, InstanceBootstrap.Service.of({ run: Effect.void })),
      ],
    ],
  ),
)

const askPermission = Effect.fn("GithubRequestsTest.askPermission")(function* (sessionID: SessionID) {
  const permission = yield* Permission.Service
  return yield* permission.ask({
    sessionID,
    permission: "bash",
    patterns: ["rm -rf build"],
    metadata: {},
    always: [],
    ruleset: [{ permission: "bash", pattern: "*", action: "ask" }],
  })
})

const askQuestion = Effect.fn("GithubRequestsTest.askQuestion")(function* (sessionID: SessionID) {
  const question = yield* Question.Service
  return yield* question.ask({
    sessionID,
    questions: [{ question: "Which branch?", header: "Branch", options: [{ label: "main", description: "Default" }] }],
  })
})

const failure = <A, E, R>(self: Effect.Effect<A, E, R>) =>
  self.pipe(
    Effect.timeout("2 seconds"),
    Effect.exit,
    Effect.map((exit) => (Exit.isFailure(exit) ? Cause.squash(exit.cause) : undefined)),
  )

describe("GitHub agent request rejection", () => {
  it.instance("rejects permission and question requests from the Session and its Task descendants", () =>
    Effect.gen(function* () {
      const sessions = yield* Session.Service
      const root = yield* sessions.create({})
      const child = yield* sessions.createTask({ parentID: root.id })
      const grandchild = yield* sessions.createTask({ parentID: child.id })
      const reject = yield* sessionTreeRequestRejector
      const off = yield* reject(root.id)
      yield* Effect.addFinalizer(() => off)

      expect(yield* failure(askPermission(root.id))).toBeInstanceOf(PermissionV1.RejectedError)
      expect(yield* failure(askPermission(grandchild.id))).toBeInstanceOf(PermissionV1.RejectedError)
      expect(yield* failure(askQuestion(child.id))).toBeInstanceOf(Question.RejectedError)
    }),
  )

  it.instance("leaves requests from Sessions outside the tree pending", () =>
    Effect.gen(function* () {
      const sessions = yield* Session.Service
      const permission = yield* Permission.Service
      const question = yield* Question.Service
      const root = yield* sessions.create({})
      const other = yield* sessions.create({})
      // A plain child Session is not a Task child, so it is not part of the tree.
      const plainChild = yield* sessions.create({ parentID: root.id })
      const reject = yield* sessionTreeRequestRejector
      const off = yield* reject(root.id)
      yield* Effect.addFinalizer(() => off)

      const permissionFiber = yield* askPermission(other.id).pipe(Effect.forkScoped)
      const questionFiber = yield* askQuestion(plainChild.id).pipe(Effect.forkScoped)
      const pending = yield* Effect.gen(function* () {
        while (true) {
          const permissions = yield* permission.list()
          const questions = yield* question.list()
          if (permissions.length === 1 && questions.length === 1) return { permissions, questions }
          yield* Effect.sleep("10 millis")
        }
      }).pipe(Effect.timeout("2 seconds"))
      expect(pending.permissions.map((item) => item.sessionID)).toEqual([other.id])
      expect(pending.questions.map((item) => item.sessionID)).toEqual([plainChild.id])

      yield* permission.reply({ requestID: pending.permissions[0].id, reply: "once" })
      yield* question.reply({ requestID: pending.questions[0].id, answers: [["main"]] })
      expect(Exit.isSuccess(yield* Fiber.await(permissionFiber))).toBe(true)
      expect(Exit.isSuccess(yield* Fiber.await(questionFiber))).toBe(true)
    }),
  )
})
