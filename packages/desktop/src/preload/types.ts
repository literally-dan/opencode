import type { DesktopMenuAction } from "@opencode-ai/app/desktop-menu"

export type ServerReadyData = {
  url: string
  username: string | null
  password: string | null
}

export type SqliteMigrationProgress = { type: "InProgress"; value: number } | { type: "Done" }

export type WslRuntimeCheck = {
  available: boolean
  version: string | null
  error: string | null
}
export type WslInstalledDistro = {
  name: string
  version: number | null
  isDefault: boolean
}
export type WslOnlineDistro = {
  name: string
  label: string
}
export type WslDistroProbe = {
  name: string
  canExecute: boolean
  hasBash: boolean
  hasCurl: boolean
  error: string | null
}
export type WslOpencodeCheck = {
  distro: string
  resolvedPath: string | null
  version: string | null
  expectedVersion: string | null
  matchesDesktop: boolean | null
  error: string | null
}
export type WslServerConfig = {
  id: string
  distro: string
}

export type WslServerRuntime =
  | { kind: "starting" }
  | { kind: "ready"; url: string; username: string | null; password: string | null }
  | { kind: "failed"; message: string }
  | { kind: "stopped" }

export type WslServerItem = {
  config: WslServerConfig
  runtime: WslServerRuntime
}

export type WslJob =
  | { kind: "runtime"; startedAt: number }
  | { kind: "distros"; startedAt: number }
  | { kind: "install-wsl"; startedAt: number }
  | { kind: "install-distro"; distro: string; startedAt: number }
  | { kind: "probe-distro"; distro: string; startedAt: number }
  | { kind: "probe-opencode"; distro: string; startedAt: number }
  | { kind: "install-opencode"; distro: string; startedAt: number }

export type WslServersState = {
  runtime: WslRuntimeCheck | null
  installed: WslInstalledDistro[]
  online: WslOnlineDistro[]
  distroProbes: Record<string, WslDistroProbe>
  opencodeChecks: Record<string, WslOpencodeCheck>
  pendingRestart: boolean
  servers: WslServerItem[]
  job: WslJob | null
}
export type WslServersEvent = { type: "state"; state: WslServersState }

export type WslServersAPI = {
  getState: () => Promise<WslServersState>
  subscribe: (cb: (event: WslServersEvent) => void) => () => void
  probeRuntime: () => Promise<void>
  refreshDistros: () => Promise<void>
  installWsl: () => Promise<void>
  installDistro: (name: string) => Promise<void>
  probeDistro: (name: string) => Promise<void>
  probeOpencode: (name: string) => Promise<void>
  installOpencode: (name: string) => Promise<void>
  openTerminal: (name: string) => Promise<void>
  addServer: (distro: string) => Promise<WslServerConfig>
  removeServer: (id: string) => Promise<void>
  startServer: (id: string) => Promise<void>
}

export type LinuxDisplayBackend = "wayland" | "auto"
export type TitlebarTheme = {
  mode: "light" | "dark"
}
export type WindowConfig = {
  updaterEnabled: boolean
}

export type FatalRendererError = {
  error: string
  url: string
  version?: string
  platform: string
  os?: string
}

export type ElectronAPI = {
  killSidecar: () => Promise<void>
  installCli: () => Promise<string>
  awaitInitialization: () => Promise<ServerReadyData>
  wslServers: WslServersAPI
  getWindowConfig: () => Promise<WindowConfig>
  consumeInitialDeepLinks: () => Promise<string[]>
  getDefaultServerUrl: () => Promise<string | null>
  setDefaultServerUrl: (url: string | null) => Promise<void>
  getDisplayBackend: () => Promise<LinuxDisplayBackend | null>
  setDisplayBackend: (backend: LinuxDisplayBackend | null) => Promise<void>
  parseMarkdownCommand: (markdown: string) => Promise<string>
  checkAppExists: (appName: string) => Promise<boolean>
  wslPath: (path: string, mode: "windows" | "linux" | null, distro?: string | null) => Promise<string>
  resolveAppPath: (appName: string) => Promise<string | null>
  storeGet: (name: string, key: string) => Promise<string | null>
  storeSet: (name: string, key: string, value: string) => Promise<void>
  storeDelete: (name: string, key: string) => Promise<void>
  storeClear: (name: string) => Promise<void>
  storeKeys: (name: string) => Promise<string[]>
  storeLength: (name: string) => Promise<number>

  getWindowCount: () => Promise<number>
  onMenuCommand: (cb: (id: string) => void) => () => void
  onDeepLink: (cb: (urls: string[]) => void) => () => void

  openDirectoryPicker: (opts?: {
    multiple?: boolean
    title?: string
    defaultPath?: string
  }) => Promise<string | string[] | null>
  openFilePicker: (opts?: {
    multiple?: boolean
    title?: string
    defaultPath?: string
    accept?: string[]
    extensions?: string[]
  }) => Promise<string | string[] | null>
  saveFilePicker: (opts?: { title?: string; defaultPath?: string }) => Promise<string | null>
  openLink: (url: string) => void
  openPath: (path: string, app?: string) => Promise<void>
  readClipboardImage: () => Promise<{ buffer: ArrayBuffer; width: number; height: number } | null>
  showNotification: (title: string, body?: string) => void
  getWindowFocused: () => Promise<boolean>
  setWindowFocus: () => Promise<void>
  showWindow: () => Promise<void>
  relaunch: () => void
  getZoomFactor: () => Promise<number>
  setZoomFactor: (factor: number) => Promise<void>
  getPinchZoomEnabled: () => Promise<boolean>
  setPinchZoomEnabled: (enabled: boolean) => Promise<void>
  onPinchZoomEnabledChanged: (cb: (enabled: boolean) => void) => () => void
  onZoomFactorChanged: (cb: (factor: number) => void) => () => void
  setTitlebar: (theme: TitlebarTheme) => Promise<void>
  runDesktopMenuAction: (action: DesktopMenuAction) => Promise<void>
  runUpdater: (alertOnFail: boolean) => Promise<void>
  checkUpdate: () => Promise<{ updateAvailable: boolean; version?: string }>
  installUpdate: () => Promise<void>
  setBackgroundColor: (color: string) => Promise<void>
  exportDebugLogs: () => Promise<string>
  recordFatalRendererError: (error: FatalRendererError) => Promise<void>
}
