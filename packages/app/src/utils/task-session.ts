import type { Session } from "@opencode-ai/sdk/v2/client"

export function isTaskChild(parent: Session, child: Session) {
  return (
    child.parentID === parent.id &&
    child.taskParentID === child.parentID &&
    child.projectID === parent.projectID &&
    child.workspaceID === parent.workspaceID &&
    child.directory === parent.directory &&
    (child.path === undefined || parent.path === undefined || child.path === parent.path)
  )
}
