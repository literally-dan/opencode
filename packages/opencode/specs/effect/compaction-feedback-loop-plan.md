# Compaction Feedback Loop Plan

Status: Approved.

## Objective

Stop the context-pressure reminder and context-management tools from creating a self-sustaining compaction loop. Preserve provider prompt-cache reuse and cost accounting while keeping original tool results recoverable.

## Source-Confirmed Problem

The native whole-history compaction path is bounded. The repeated behavior comes from a separate feedback loop:

1. Context pressure emits a trailing reminder.
2. The model calls `list_context`, `compact_results`, or `compact_bulk`.
3. The full tool receipt becomes durable model history.
4. On a later step, that receipt becomes a new manual-compaction candidate and changes the reminder fingerprint.
5. Another reminder and receipt repeat the cycle even after useful space was reclaimed.

Current `dev` already filters crash-unsafe multi-member prior compaction groups out of reminder inventory. One selector-order defect remains: `compact_results` applies `keep_last` and the 200-part limit before it rejects those groups.

## Cache Constraints

These constraints are acceptance requirements.

1. Do not change tool membership, tool descriptions, tool schemas, or system prompt text.
2. Do not move the pressure reminder into the system prompt or durable history.
3. Preserve tool call IDs and original tool inputs so provider tool-call/result pairing remains valid.
4. Preserve each original tool output in durable storage and through `read_part`.
5. Change each stale context-management receipt at most once. Never maintain a repeatedly rewritten aggregate receipt.
6. Keep every provider-visible prefix before the changed receipt byte-stable.
7. Keep session cache keys, cache TTLs, cache read/write normalization, usage pricing, and overflow accounting unchanged.
8. On explicit-cache providers, do not place an automatic cache breakpoint at or after the transient suffix that starts with a full context-management receipt or ephemeral pressure reminder.
9. Do not change provider-managed automatic caching modes. Anthropic automatic `cacheControl` and Cloudflare AI Gateway automatic placement remain provider-controlled.

The planned receipt lifecycle changes the receipt projection once after the model has consumed its full output. OpenCode-controlled explicit cache markers stop before that transient receipt, so the one-request full form does not create a cache entry that becomes unusable as soon as the receipt collapses. All earlier tool, system, and history prefixes remain reusable. The bounded projection then stays identical on later requests.

## Scope

### 1. Classify Context-Management Receipts

Add shared predicates and deterministic projection helpers in `packages/opencode/src/session/compaction-pruning.ts` for completed local receipts from:

- `list_context`
- `compact_results`
- `compact_bulk`

Provider-executed tools are not context-management receipts. Manually compacted parts retain their existing behavior.

Build bounded summaries only from existing structured tool metadata:

- `list_context`: total parts, compactable parts, and estimated reclaimable characters.
- `compact_results`: compacted parts, skipped parts, and net characters freed.
- `compact_bulk`: compacted parts, skipped parts, net characters freed, and summarizer calls.

Do not copy arbitrary error text into the summary. Include the receipt part ID so `read_part` remains the explicit recovery path.

### 2. Give Receipts a Full-Then-Bounded Lifecycle

Add an effectful operation to the Session compaction service in `packages/opencode/src/session/compaction.ts`, with pure classification and summary helpers in `compaction-pruning.ts`.

At the prompt loop boundary in `packages/opencode/src/session/prompt.ts`:

1. Load projected history.
2. Leave a completed context-management receipt unchanged until a strictly later assistant message contains a durable `step-finish` part. A newer user message or empty, aborted, cancelled, or pre-stream assistant shell does not prove that the provider consumed the receipt.
3. After that durable consumption proof exists, clone the receipt part and set the existing `state.time.compacted` marker under the per-Session compaction lock.
4. Preserve `state.output`, `state.input`, `state.metadata`, `callID`, and all other durable fields.
5. Do not set `compactionGroup`, `compactionSummary`, or `compactionGeneration`; those fields remain reserved for model-authored manual compaction.
6. Reload or return the projected messages after writes so the current request uses the bounded form.

Projection uses the deterministic metadata summary instead of the generic native-pruning fallback for these receipts. Repeated collapse calls perform no writes and render identical text.

Update native age-based pruning so its backward boundary scan continues past receipt markers. A receipt lifecycle marker must not look like the boundary of an older native-pruning run.

### 3. Exclude Receipts From Manual Candidates

Make `eligibility()` reject all three context-management receipt types before fresh or refolded candidate handling. This central rule must apply to:

- pressure reminder manifests and fingerprints
- `list_context`
- `compact_results` explicit and selector paths
- `compact_bulk`

`list_context` may still display a receipt as `keep (...)`; it must never report one as actionable.

### 4. Correct Selector Ordering

In `packages/opencode/src/tool/compact_results.ts`, resolve selector matches against durable history in this order:

1. Match the selector against otherwise eligible parts.
2. Remove ambiguous and crash-unsafe prior-group matches.
3. Apply `keep_last` to the remaining chronological actionable matches.
4. Union explicit `part_ids` with selector results and remove duplicates.
5. Enforce the 200-part request limit.

Explicit IDs keep their current per-ID skip diagnostics. Unsupported prior groups selected only through `select` do not consume `keep_last` or the 200-part limit.

Do not add atomic multi-member group refolding.

### 5. Acknowledge Successful Compaction

When the latest assistant message for the current user request contains a completed local `compact_results` or `compact_bulk` receipt with `charsFreed > 0`:

1. Calculate current pressure and the post-compaction candidate fingerprint as normal.
2. Store that fingerprint in the existing process-local reminder state.
3. Suppress the immediate duplicate reminder.

No timer, durable counter, schema field, or system text is added. A later candidate-set change or soft-to-hard pressure-band change produces a new fingerprint and can emit a new reminder.

### 6. Keep Explicit Cache Markers Before the Transient Suffix

The full receipt and reminder remain provider-visible. Add an internal cache-prefix limit that restricts automatic explicit cache-marker selection to stable ModelMessages before the transient suffix.

Calculate the limit after message projection:

1. Find the newest source assistant message that contains a full, not-yet-collapsed context-management receipt.
2. Use its preserved tool call ID to find the first projected ModelMessage generated from that source message.
3. End the eligible cache prefix before that ModelMessage. This safely excludes the assistant tool call, grouped tool result, pressure reminder, final-step instruction, and any other later volatile messages from automatic cache-marker selection.
4. If there is no full receipt but a pressure reminder or final-step instruction is appended, end the eligible cache prefix before the first appended ephemeral message.
5. If no volatile suffix exists, do not set a limit.

Parallel tool calls in the same source assistant message share the limit. The entire projected assistant/tool group remains provider-visible and correctly ordered; only automatic cache-marker eligibility changes.

AI SDK path:

- Pass the internal limit through `packages/opencode/src/session/llm.ts` to `ProviderTransform.message`.
- Choose the final two non-system messages only from the eligible stable prefix.
- Keep the same explicit marker targets as the equivalent request with the volatile suffix omitted.

Native `@opencode-ai/llm` path:

- Pass the limit through the internal stream input and native request conversion.
- Add optional canonical message metadata such as `cacheable: false` in `packages/llm/src/schema/messages.ts` for messages after the eligible stable prefix.
- Treat the first `cacheable: false` message as the end of automatic cache-point selection in `packages/llm/src/cache-policy.ts`. Do not skip it and then mark a later message because a later marker would still cache the transient receipt.
- Keep explicit/manual cache hints authoritative.
- Ensure Anthropic, Bedrock, OpenAI, and Gemini body lowerers do not serialize this internal field.

OpenAI-compatible session cache keys and implicit caching stay unchanged. Gemini behavior stays unchanged. Anthropic automatic `cacheControl` and Cloudflare AI Gateway automatic caching remain unchanged because safely overriding provider-managed placement requires separate policy work.

## Subsystem Ownership

After approval, implementation may proceed in parallel at non-overlapping boundaries:

1. Receipt lifecycle: `compaction-pruning.ts`, `compaction.ts`, message projection, native pruning, and focused tests.
2. Selector ordering: `compact_results.ts` and history-tool tests.
3. Explicit cache markers: `packages/llm` cache policy/schema, `session/llm.ts`, provider transform, and focused tests.
4. Prompt integration: prompt-loop collapse, successful-progress acknowledgement, cache exclusion control, and the end-to-end feedback-loop test.

Every implementation and review agent must receive this plan path.

## Acceptance Tests

### Receipt Lifecycle

- A context-management receipt is visible in full on the first provider request that consumes it.
- On the next request, it projects as the bounded deterministic summary.
- A newer empty, aborted, cancelled, or pre-stream assistant shell does not collapse it.
- A strictly later durable `step-finish` collapses it, including after restart recovery.
- Later requests project byte-identical summary text.
- A repeated collapse call performs no second durable write.
- `read_part` returns the original full result after projection is bounded.
- Provider-executed tools and manually compacted parts are unchanged.
- Context-management receipts never become actionable candidates.
- Native pruning continues past receipt lifecycle markers.
- Receipt lifecycle markers do not count as manual user-content compaction in overflow decisions.

### Selector Ordering

- More than 200 raw selector matches succeed when unsafe prior groups leave at most 200 actionable matches.
- `keep_last` counts only actionable matches after unsafe-group filtering.
- Explicit unsafe IDs retain clear skip diagnostics.
- Reminder inventory remains free of unsafe prior groups.

### Feedback Loop

Add a deterministic prompt-loop test with sustained pressure and queued model responses for `list_context`, `compact_results`, an unrelated small tool, and final text.

Assert:

- The initial pressure reminder appears once.
- The model sees each context-management receipt in full once.
- Older receipts become stable bounded projections.
- Successful compaction acknowledges the post-compaction fingerprint.
- No immediate second reminder appears solely because receipts changed history.
- A real candidate-set or pressure-band change can emit a new reminder.
- Original outputs remain recoverable.

### Cache Behavior

Use deterministic request and protocol tests; do not require credentials.

- Tools and system messages are identical with and without a transient suffix.
- The provider-visible history before a collapsed receipt is identical.
- A receipt changes projection once and remains identical afterward.
- Anthropic and Bedrock automatic explicit marker selection stops at the first `cacheable: false` message.
- Explicit marker targets match the equivalent request with the volatile suffix omitted.
- A full receipt remains in the provider body but has no automatic `cache_control` or `cachePoint` marker at or after it.
- Cover a receipt alone, a receipt followed by a reminder, a receipt followed by a final-step instruction, and grouped parallel tool results.
- A reminder without a receipt also remains outside the automatic explicit cached prefix.
- Manual cache hints still win.
- OpenAI and Gemini requests gain no explicit marker or serialized internal cache field.
- Existing cache read/write usage and pricing tests remain unchanged and pass.

## Verification

Run tests from package directories, never from the repository root.

1. Focused `packages/opencode` history, message projection, compaction, provider transform, native LLM, and prompt suites.
2. Focused `packages/llm` cache-policy and Anthropic/Bedrock protocol suites.
3. `bun typecheck` in each changed package.
4. The relevant broader package test suites if focused checks pass.
5. A final review limited to this plan, the changed files, and their direct call paths.

## Excluded Work

- Changing the 65% soft threshold, 85% hard threshold, 45% target, reserved tokens, or reminder manifest limits.
- Automatic selection or summarization of user/task content.
- Timed cooldowns, durable reminder counters, or the currently unused `ignored` escalation counter.
- Removing tool-call/result pairs or rewriting tool inputs.
- Deleting receipt output or weakening `read_part` recovery.
- Atomic multi-member prior-group refolding, grouped durable events, App reducers for such events, or normalized group storage.
- Native compaction redesign or retained-tail policy changes.
- Changing provider-managed automatic cache modes.
- Protocol, Server `HttpApi`, generated client, or legacy SDK changes.

## Residual Risk

Provider-managed automatic caching can still choose the ephemeral reminder as a cache boundary. This applies to Anthropic automatic `cacheControl` and Cloudflare AI Gateway automatic caching. The plan leaves those modes unchanged because switching cache policy on high-context turns can invalidate established prefixes and increase cost. Any change there requires recorded provider validation and a separate rollout plan.
