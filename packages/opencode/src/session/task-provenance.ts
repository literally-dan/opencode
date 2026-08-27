import type { OpencodeClient } from "@opencode-ai/sdk/v2"

export namespace TaskProvenance {
  export async function getSession(sdk: OpencodeClient, sessionID: string, signal?: AbortSignal) {
    const result = await sdk.session.get({ sessionID }, { throwOnError: true, ...(signal ? { signal } : {}) })
    if (!result.data) throw new Error(`session lookup returned no data for ${sessionID}`)
    return result.data
  }

  export async function parent(
    sdk: OpencodeClient,
    child: Awaited<ReturnType<typeof getSession>>,
    signal?: AbortSignal,
  ) {
    if (!child.taskParentID || child.parentID !== child.taskParentID) return undefined
    const parent = await getSession(sdk, child.taskParentID, signal)
    if (
      child.projectID !== parent.projectID ||
      child.workspaceID !== parent.workspaceID ||
      child.directory !== parent.directory ||
      (child.path !== undefined && parent.path !== undefined && child.path !== parent.path)
    )
      return undefined
    return parent
  }
}
