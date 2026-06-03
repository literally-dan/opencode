import { createSimpleContext } from "@opencode-ai/ui/context"
import type { AsyncStorage, SyncStorage } from "@solid-primitives/storage"
import type { Accessor } from "solid-js"
import type { DesktopMenuAction } from "../desktop-menu"
import { ServerConnection } from "./server"

type PickerPaths = string | string[] | null
type OpenDirectoryPickerOptions = { title?: string; multiple?: boolean }
type OpenFilePickerOptions = { title?: string; multiple?: boolean; accept?: string[]; extensions?: string[] }
type SaveFilePickerOptions = { title?: string; defaultPath?: string }
type UpdateInfo = { updateAvailable: boolean; version?: string }
type PlatformName = "web" | "desktop"
type DesktopOS = "macos" | "windows" | "linux"

export type FatalRendererErrorLog = {
  error: string
  url: string
  version?: string
  platform: PlatformName
  os?: DesktopOS
}

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

export type WslServersPlatform = {
  getState(): Promise<WslServersState>
  subscribe(cb: (event: WslServersEvent) => void): () => void
  probeRuntime(): Promise<void>
  refreshDistros(): Promise<void>
  installWsl(): Promise<void>
  installDistro(name: string): Promise<void>
  probeDistro(name: string): Promise<void>
  probeOpencode(name: string): Promise<void>
  installOpencode(name: string): Promise<void>
  openTerminal(name: string): Promise<void>
  addServer(distro: string): Promise<WslServerConfig>
  removeServer(id: string): Promise<void>
  startServer(id: string): Promise<void>
}

export type Platform = {
  /** Platform discriminator */
  platform: PlatformName

  /** Desktop OS (Tauri only) */
  os?: DesktopOS

  /** App version */
  version?: string

  /** Open a URL in the default browser */
  openLink(url: string): void

  /** Open a local path in a local app (desktop only) */
  openPath?(path: string, app?: string): Promise<void>

  /** Restart the app  */
  restart(): Promise<void>

  /** Navigate back in history */
  back(): void

  /** Navigate forward in history */
  forward(): void

  /** Send a system notification (optional deep link) */
  notify(title: string, description?: string, href?: string): Promise<void>

  /** Open directory picker dialog (native on Tauri, server-backed on web) */
  openDirectoryPickerDialog?(opts?: OpenDirectoryPickerOptions): Promise<PickerPaths>

  /** Open native file picker dialog (Tauri only) */
  openFilePickerDialog?(opts?: OpenFilePickerOptions): Promise<PickerPaths>

  /** Save file picker dialog (Tauri only) */
  saveFilePickerDialog?(opts?: SaveFilePickerOptions): Promise<string | null>

  /** Storage mechanism, defaults to localStorage */
  storage?: (name?: string) => SyncStorage | AsyncStorage

  /** Check for a downloadable desktop update */
  checkUpdate?(): Promise<UpdateInfo>

  /** Install the downloaded update using the platform restart flow */
  updateAndRestart?(): Promise<void>

  /** Fetch override */
  fetch?: typeof fetch

  /** Get the configured default server URL (platform-specific) */
  getDefaultServer?(): Promise<ServerConnection.Key | null>

  /** Set the default server URL to use on app startup (platform-specific) */
  setDefaultServer?(url: ServerConnection.Key | null): Promise<void> | void

  /** Manage WSL sidecar servers (Electron on Windows only) */
  wslServers?: WslServersPlatform

  /** Get the preferred display backend (desktop only) */
  getDisplayBackend?(): Promise<DisplayBackend | null> | DisplayBackend | null

  /** Set the preferred display backend (desktop only) */
  setDisplayBackend?(backend: DisplayBackend): Promise<void>

  /** Parse markdown to HTML using native parser (desktop only, returns unprocessed code blocks) */
  parseMarkdown?(markdown: string): Promise<string>

  /** Webview zoom level (desktop only) */
  webviewZoom?: Accessor<number>

  /** Get whether native pinch/Ctrl-scroll zoom gestures are enabled (desktop only) */
  getPinchZoomEnabled?(): Promise<boolean> | boolean

  /** Allow native pinch/Ctrl-scroll zoom gestures (desktop only) */
  setPinchZoomEnabled?(enabled: boolean): Promise<void> | void

  /** Run a desktop-only menu action from the app chrome */
  runDesktopMenuAction?(action: DesktopMenuAction): Promise<void> | void

  /** Check if an editor app exists (desktop only) */
  checkAppExists?(appName: string): Promise<boolean>

  /** Read image from clipboard (desktop only) */
  readClipboardImage?(): Promise<File | null>

  /** Export collected diagnostic logs (desktop only) */
  exportDebugLogs?(): Promise<string>

  /** Record a fatal renderer error in platform logs (desktop only) */
  recordFatalRendererError?(error: FatalRendererErrorLog): Promise<void>
}

export type DisplayBackend = "auto" | "wayland"

export const { use: usePlatform, provider: PlatformProvider } = createSimpleContext({
  name: "Platform",
  init: (props: { value: Platform }) => {
    return props.value
  },
})
