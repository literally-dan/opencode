import { execFile } from "node:child_process"
import { BrowserWindow, Notification, app, clipboard, dialog, ipcMain, shell } from "electron"
import type { IpcMainEvent, IpcMainInvokeEvent } from "electron"
import type { DesktopMenuAction } from "@opencode-ai/app/desktop-menu"

import type {
  FatalRendererError,
  ServerReadyData,
  TitlebarTheme,
  WindowConfig,
  WslServerConfig,
  WslServersEvent,
  WslServersState,
} from "../preload/types"
import { runDesktopMenuAction } from "./desktop-menu-actions"
import { getStore } from "./store"
import { getPinchZoomEnabled, setPinchZoomEnabled, setTitlebar, updateTitlebar } from "./windows"

const pickerFilters = (ext?: string[]) => {
  if (!ext || ext.length === 0) return undefined
  return [{ name: "Files", extensions: ext }]
}

type Deps = {
  killSidecar: () => Promise<void> | void
  awaitInitialization: () => Promise<ServerReadyData>
  relaunch: () => void
  getWslServersState: () => Promise<WslServersState> | WslServersState
  onWslServersEvent: (listener: (event: WslServersEvent) => void) => () => void
  wslServersProbeRuntime: () => Promise<void> | void
  wslServersRefreshDistros: () => Promise<void> | void
  wslServersInstallWsl: () => Promise<void> | void
  wslServersInstallDistro: (name: string) => Promise<void> | void
  wslServersProbeDistro: (name: string) => Promise<void> | void
  wslServersProbeOpencode: (name: string) => Promise<void> | void
  wslServersInstallOpencode: (name: string) => Promise<void> | void
  wslServersOpenTerminal: (name: string) => Promise<void> | void
  wslServersAddServer: (distro: string) => Promise<WslServerConfig> | WslServerConfig
  wslServersRemoveServer: (id: string) => Promise<void> | void
  wslServersStartServer: (id: string) => Promise<void> | void
  getWindowConfig: () => Promise<WindowConfig> | WindowConfig
  consumeInitialDeepLinks: () => Promise<string[]> | string[]
  getDefaultServerUrl: () => Promise<string | null> | string | null
  setDefaultServerUrl: (url: string | null) => Promise<void> | void
  getDisplayBackend: () => Promise<string | null>
  setDisplayBackend: (backend: string | null) => Promise<void> | void
  parseMarkdown: (markdown: string) => Promise<string> | string
  checkAppExists: (appName: string) => Promise<boolean> | boolean
  wslPath: (path: string, mode: "windows" | "linux" | null, distro?: string | null) => Promise<string>
  resolveAppPath: (appName: string) => Promise<string | null>
  runUpdater: (alertOnFail: boolean) => Promise<void> | void
  checkUpdate: () => Promise<{ updateAvailable: boolean; version?: string }>
  installUpdate: () => Promise<void> | void
  setBackgroundColor: (color: string) => void
  exportDebugLogs: () => Promise<string>
  recordFatalRendererError: (error: FatalRendererError) => Promise<void> | void
}

export function registerIpcHandlers(deps: Deps) {
  const requireString = (name: string, value: unknown) => {
    if (typeof value === "string" && value.length > 0) return value
    throw new Error(`Invalid ${name}`)
  }

  const wslSubscriptions = new Map<number, () => void>()
  const unsubscribeWsl = (id: number) => {
    const off = wslSubscriptions.get(id)
    if (!off) return
    off()
    wslSubscriptions.delete(id)
  }

  app.once("will-quit", () => {
    for (const off of wslSubscriptions.values()) off()
    wslSubscriptions.clear()
  })

  ipcMain.handle("kill-sidecar", () => deps.killSidecar())
  ipcMain.handle("await-initialization", () => deps.awaitInitialization())
  ipcMain.handle("wsl-servers-subscribe", (event) => {
    const id = event.sender.id
    if (wslSubscriptions.has(id)) return
    wslSubscriptions.set(
      id,
      deps.onWslServersEvent((payload) => {
        if (event.sender.isDestroyed()) {
          unsubscribeWsl(id)
          return
        }
        event.sender.send("wsl-servers-event", payload)
      }),
    )
    event.sender.once("destroyed", () => unsubscribeWsl(id))
  })
  ipcMain.handle("wsl-servers-unsubscribe", (event) => unsubscribeWsl(event.sender.id))
  ipcMain.handle("wsl-servers-get-state", () => deps.getWslServersState())
  ipcMain.handle("wsl-servers-probe-runtime", () => deps.wslServersProbeRuntime())
  ipcMain.handle("wsl-servers-refresh-distros", () => deps.wslServersRefreshDistros())
  ipcMain.handle("wsl-servers-install-wsl", () => deps.wslServersInstallWsl())
  ipcMain.handle("wsl-servers-install-distro", (_event: IpcMainInvokeEvent, name: string) =>
    deps.wslServersInstallDistro(requireString("distro", name)),
  )
  ipcMain.handle("wsl-servers-probe-distro", (_event: IpcMainInvokeEvent, name: string) =>
    deps.wslServersProbeDistro(requireString("distro", name)),
  )
  ipcMain.handle("wsl-servers-probe-opencode", (_event: IpcMainInvokeEvent, name: string) =>
    deps.wslServersProbeOpencode(requireString("distro", name)),
  )
  ipcMain.handle("wsl-servers-install-opencode", (_event: IpcMainInvokeEvent, name: string) =>
    deps.wslServersInstallOpencode(requireString("distro", name)),
  )
  ipcMain.handle("wsl-servers-open-terminal", (_event: IpcMainInvokeEvent, name: string) =>
    deps.wslServersOpenTerminal(requireString("distro", name)),
  )
  ipcMain.handle("wsl-servers-add", (_event: IpcMainInvokeEvent, distro: string) =>
    deps.wslServersAddServer(requireString("distro", distro)),
  )
  ipcMain.handle("wsl-servers-remove", (_event: IpcMainInvokeEvent, id: string) =>
    deps.wslServersRemoveServer(requireString("server id", id)),
  )
  ipcMain.handle("wsl-servers-start", (_event: IpcMainInvokeEvent, id: string) =>
    deps.wslServersStartServer(requireString("server id", id)),
  )
  ipcMain.handle("get-window-config", () => deps.getWindowConfig())
  ipcMain.handle("consume-initial-deep-links", () => deps.consumeInitialDeepLinks())
  ipcMain.handle("get-default-server-url", () => deps.getDefaultServerUrl())
  ipcMain.handle("set-default-server-url", (_event: IpcMainInvokeEvent, url: string | null) =>
    deps.setDefaultServerUrl(url),
  )
  ipcMain.handle("get-display-backend", () => deps.getDisplayBackend())
  ipcMain.handle("set-display-backend", (_event: IpcMainInvokeEvent, backend: string | null) =>
    deps.setDisplayBackend(backend),
  )
  ipcMain.handle("parse-markdown", (_event: IpcMainInvokeEvent, markdown: string) => deps.parseMarkdown(markdown))
  ipcMain.handle("check-app-exists", (_event: IpcMainInvokeEvent, appName: string) => deps.checkAppExists(appName))
  ipcMain.handle(
    "wsl-path",
    (_event: IpcMainInvokeEvent, path: string, mode: "windows" | "linux" | null, distro?: string | null) =>
      deps.wslPath(path, mode, distro),
  )
  ipcMain.handle("resolve-app-path", (_event: IpcMainInvokeEvent, appName: string) => deps.resolveAppPath(appName))
  ipcMain.handle("run-updater", (_event: IpcMainInvokeEvent, alertOnFail: boolean) => deps.runUpdater(alertOnFail))
  ipcMain.handle("check-update", () => deps.checkUpdate())
  ipcMain.handle("install-update", () => deps.installUpdate())
  ipcMain.handle("set-background-color", (_event: IpcMainInvokeEvent, color: string) => deps.setBackgroundColor(color))
  ipcMain.handle("export-debug-logs", () => deps.exportDebugLogs())
  ipcMain.handle("record-fatal-renderer-error", (_event: IpcMainInvokeEvent, error: FatalRendererError) =>
    deps.recordFatalRendererError(error),
  )
  ipcMain.handle("store-get", (_event: IpcMainInvokeEvent, name: string, key: string) => {
    try {
      const store = getStore(name)
      const value = store.get(key)
      if (value === undefined || value === null) return null
      return typeof value === "string" ? value : JSON.stringify(value)
    } catch {
      return null
    }
  })
  ipcMain.handle("store-set", (_event: IpcMainInvokeEvent, name: string, key: string, value: string) => {
    getStore(name).set(key, value)
  })
  ipcMain.handle("store-delete", (_event: IpcMainInvokeEvent, name: string, key: string) => {
    getStore(name).delete(key)
  })
  ipcMain.handle("store-clear", (_event: IpcMainInvokeEvent, name: string) => {
    getStore(name).clear()
  })
  ipcMain.handle("store-keys", (_event: IpcMainInvokeEvent, name: string) => {
    const store = getStore(name)
    return Object.keys(store.store)
  })
  ipcMain.handle("store-length", (_event: IpcMainInvokeEvent, name: string) => {
    const store = getStore(name)
    return Object.keys(store.store).length
  })

  ipcMain.handle(
    "open-directory-picker",
    async (_event: IpcMainInvokeEvent, opts?: { multiple?: boolean; title?: string; defaultPath?: string }) => {
      const result = await dialog.showOpenDialog({
        properties: ["openDirectory", ...(opts?.multiple ? ["multiSelections" as const] : []), "createDirectory"],
        title: opts?.title ?? "Choose a folder",
        defaultPath: opts?.defaultPath,
      })
      if (result.canceled) return null
      return opts?.multiple ? result.filePaths : result.filePaths[0]
    },
  )

  ipcMain.handle(
    "open-file-picker",
    async (
      _event: IpcMainInvokeEvent,
      opts?: { multiple?: boolean; title?: string; defaultPath?: string; accept?: string[]; extensions?: string[] },
    ) => {
      const result = await dialog.showOpenDialog({
        properties: ["openFile", ...(opts?.multiple ? ["multiSelections" as const] : [])],
        title: opts?.title ?? "Choose a file",
        defaultPath: opts?.defaultPath,
        filters: pickerFilters(opts?.extensions),
      })
      if (result.canceled) return null
      return opts?.multiple ? result.filePaths : result.filePaths[0]
    },
  )

  ipcMain.handle(
    "save-file-picker",
    async (_event: IpcMainInvokeEvent, opts?: { title?: string; defaultPath?: string }) => {
      const result = await dialog.showSaveDialog({
        title: opts?.title ?? "Save file",
        defaultPath: opts?.defaultPath,
      })
      if (result.canceled) return null
      return result.filePath ?? null
    },
  )

  ipcMain.on("open-link", (_event: IpcMainEvent, url: string) => {
    void shell.openExternal(url)
  })

  ipcMain.handle("open-path", async (_event: IpcMainInvokeEvent, path: string, app?: string) => {
    if (!app) return shell.openPath(path)
    await new Promise<void>((resolve, reject) => {
      const [cmd, args] =
        process.platform === "darwin" ? (["open", ["-a", app, path]] as const) : ([app, [path]] as const)
      execFile(cmd, args, (err) => (err ? reject(err) : resolve()))
    })
  })

  ipcMain.handle("read-clipboard-image", () => {
    const image = clipboard.readImage()
    if (image.isEmpty()) return null
    const buffer = image.toPNG().buffer
    const size = image.getSize()
    return { buffer, width: size.width, height: size.height }
  })

  ipcMain.on("show-notification", (_event: IpcMainEvent, title: string, body?: string) => {
    new Notification({ title, body }).show()
  })

  ipcMain.handle("get-window-count", () => BrowserWindow.getAllWindows().length)

  ipcMain.handle("get-window-focused", (event: IpcMainInvokeEvent) => {
    const win = BrowserWindow.fromWebContents(event.sender)
    return win?.isFocused() ?? false
  })

  ipcMain.handle("set-window-focus", (event: IpcMainInvokeEvent) => {
    const win = BrowserWindow.fromWebContents(event.sender)
    win?.focus()
  })

  ipcMain.handle("show-window", (event: IpcMainInvokeEvent) => {
    const win = BrowserWindow.fromWebContents(event.sender)
    win?.show()
  })

  ipcMain.on("relaunch", () => {
    deps.relaunch()
  })

  ipcMain.handle("get-zoom-factor", (event: IpcMainInvokeEvent) => event.sender.getZoomFactor())
  ipcMain.handle("set-zoom-factor", (event: IpcMainInvokeEvent, factor: number) => {
    event.sender.setZoomFactor(factor)
    const win = BrowserWindow.fromWebContents(event.sender)
    if (!win) return
    updateTitlebar(win)
  })
  ipcMain.handle("get-pinch-zoom-enabled", () => getPinchZoomEnabled())
  ipcMain.handle("set-pinch-zoom-enabled", (_event: IpcMainInvokeEvent, enabled: boolean) => {
    setPinchZoomEnabled(enabled)
  })
  ipcMain.handle("set-titlebar", (event: IpcMainInvokeEvent, theme: TitlebarTheme) => {
    const win = BrowserWindow.fromWebContents(event.sender)
    if (!win) return
    setTitlebar(win, theme)
  })
  ipcMain.handle("run-desktop-menu-action", (event: IpcMainInvokeEvent, action: DesktopMenuAction) => {
    runDesktopMenuAction(BrowserWindow.fromWebContents(event.sender), action)
  })
}

export function sendMenuCommand(win: BrowserWindow, id: string) {
  win.webContents.send("menu-command", id)
}

export function sendDeepLinks(win: BrowserWindow, urls: string[]) {
  win.webContents.send("deep-link", urls)
}
