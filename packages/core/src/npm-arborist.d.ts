// Arborist ships no types for this internal module. Its `process` export is the emitter that Arborist listens on for
// process signals while it writes packages.
declare module "@npmcli/arborist/lib/signal-handling.js" {
  import type { EventEmitter } from "events"

  const signalHandling: { process: EventEmitter }
  export default signalHandling
}
