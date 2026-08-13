import { TextAttributes } from "@opentui/core"
import { useTerminalDimensions } from "@opentui/solid"
import { useBindings } from "../keymap"
import { useTuiConfig } from "../config"
import { useTheme } from "../context/theme"
import { getScrollAcceleration } from "../util/scroll"
import { Spinner } from "./spinner"
import { useDialog } from "../ui/dialog"

export function DialogAsk(props: { question: string; answer?: string }) {
  const dialog = useDialog()
  const dimensions = useTerminalDimensions()
  const tuiConfig = useTuiConfig()
  const { theme, syntax } = useTheme()

  useBindings(() => ({
    bindings: [
      {
        key: "return",
        desc: "Close answer",
        group: "Dialog",
        cmd: () => dialog.clear(),
      },
    ],
  }))

  return (
    <box
      paddingLeft={2}
      paddingRight={2}
      paddingBottom={1}
      gap={1}
      height={Math.max(6, Math.min(32, dimensions().height - 4))}
    >
      <box flexDirection="row" justifyContent="space-between" flexShrink={0}>
        <text attributes={TextAttributes.BOLD} fg={theme.text}>
          Ask
        </text>
        <text fg={theme.textMuted} onMouseUp={() => dialog.clear()}>
          esc
        </text>
      </box>
      <text fg={theme.textMuted}>{props.question}</text>
      <box flexGrow={1} minHeight={3}>
        {props.answer === undefined ? (
          <Spinner>Thinking</Spinner>
        ) : (
          <scrollbox flexGrow={1} scrollAcceleration={getScrollAcceleration(tuiConfig)}>
            <markdown
              syntaxStyle={syntax()}
              content={props.answer}
              tableOptions={{ style: "grid" }}
              fg={theme.markdownText}
              bg={theme.backgroundPanel}
            />
          </scrollbox>
        )}
      </box>
      <text fg={theme.textMuted}>enter close</text>
    </box>
  )
}
