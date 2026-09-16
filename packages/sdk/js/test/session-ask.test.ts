import { expect, test } from "bun:test"
import type {
  BadRequestError,
  ConflictError,
  InvalidRequestError,
  SessionAskCancelErrors,
  SessionAskErrors,
} from "../src/v2/gen/types.gen"

type Equal<Left, Right> = [Left] extends [Right] ? ([Right] extends [Left] ? true : false) : false

test("uses the runtime Ask HTTP error shapes", () => {
  const askBadRequest: Equal<SessionAskErrors[400], BadRequestError | InvalidRequestError> = true
  const askConflict: Equal<SessionAskErrors[409], ConflictError> = true
  const cancelBadRequest: Equal<SessionAskCancelErrors[400], BadRequestError> = true

  expect(askBadRequest).toBe(true)
  expect(askConflict).toBe(true)
  expect(cancelBadRequest).toBe(true)
})
