import type { Session } from "@opencode-ai/sdk/v2/client"

// Location is not compared: moving a Session does not move its Task children.
export function isTaskChild(parent: Session, child: Session) {
  return child.parentID === parent.id && child.taskParentID === child.parentID && child.projectID === parent.projectID
}
