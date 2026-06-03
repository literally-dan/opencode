import { execFile } from "node:child_process"
import { access, readFile, readdir } from "node:fs/promises"
import { dirname, extname, join } from "node:path"
import util from "node:util"
import { resolveWslHome, runWslInDistro } from "./wsl"

const execFilePromise = util.promisify(execFile)

const exists = (path: string) =>
  access(path)
    .then(() => true)
    .catch(() => false)

export function checkAppExists(appName: string) {
  if (process.platform === "win32") return true
  if (process.platform === "linux") return true
  return checkMacosApp(appName)
}

export function resolveAppPath(appName: string) {
  if (process.platform !== "win32") return appName
  return resolveWindowsAppPath(appName)
}

// Parses `\\wsl$\<distro>\...` and `\\wsl.localhost\<distro>\...` UNC paths that
// point *into* a WSL distro's rootfs. `wslpath -u` cannot handle these reliably:
// backslashes get shell-collapsed when passed through `wsl.exe`, turning
// `\\wsl.localhost\Debian\home\luke` into `/mnt/c/wsl.localhostDebianhomeluke`,
// which is a valid-looking path that wedges opencode on DrvFs stat calls.
function parseWslUncPath(value: string): { distro: string; subpath: string } | null {
  // Normalise separators; both `\\` and `//` prefixes mean UNC.
  const normalised = value.replace(/\\/g, "/").replace(/^\/+/, "//")
  const match = /^\/\/(wsl\$|wsl\.localhost)\/([^/]+)(?:\/(.*))?$/i.exec(normalised)
  if (!match) return null
  const distro = match[2]
  const subpath = match[3] ?? ""
  return { distro, subpath }
}

export async function wslPath(path: string, mode: "windows" | "linux" | null, distro?: string | null): Promise<string> {
  if (process.platform !== "win32") return path

  // `\\wsl$\<distro>\...` / `\\wsl.localhost\<distro>\...` -> `/<subpath>` in
  // the target distro. Do the conversion in-process rather than shelling out
  // to `wslpath -u`, which mangles backslashes via wsl.exe's command-line
  // joiner. If the requested distro differs from the UNC distro, we still
  // translate literally — callers are responsible for only picking paths
  // inside the active distro.
  if (mode === "linux") {
    const unc = parseWslUncPath(path)
    if (unc) return `/${unc.subpath}`
  }

  const flag = mode === "windows" ? "-w" : "-u"
  try {
    const resolved = path.startsWith("~") ? `${await resolveWslHome(distro)}${path.slice(1)}` : path
    const input = mode === "linux" ? resolved.replace(/\\/g, "/") : resolved
    const output = await runWslInDistro(["wslpath", flag, input], distro)
    if (output.code !== 0) {
      throw new Error(output.stderr || output.stdout || `wslpath exited with code ${output.code}`)
    }
    return output.stdout.trim()
  } catch (error) {
    throw new Error(`Failed to run wslpath: ${String(error)}`, { cause: error })
  }
}

async function checkMacosApp(appName: string) {
  const locations = [`/Applications/${appName}.app`, `/System/Applications/${appName}.app`]

  const home = process.env.HOME
  if (home) locations.push(`${home}/Applications/${appName}.app`)

  for (const location of locations) {
    if (await exists(location)) return true
  }

  return execFilePromise("which", [appName])
    .then(() => true)
    .catch(() => false)
}

async function resolveWindowsAppPath(appName: string): Promise<string | null> {
  let output: string
  try {
    output = await execFilePromise("where", [appName]).then((r) => r.stdout.toString())
  } catch {
    return null
  }

  const paths = output
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0)

  const hasExt = (path: string, ext: string) => extname(path).toLowerCase() === `.${ext}`

  const exe = paths.find((path) => hasExt(path, "exe"))
  if (exe) return exe

  const resolveCmd = async (path: string) => {
    const content = await readFile(path, "utf8")
    for (const token of content.split('"').map((value: string) => value.trim())) {
      const lower = token.toLowerCase()
      if (!lower.includes(".exe")) continue

      const index = lower.indexOf("%~dp0")
      if (index >= 0) {
        const base = dirname(path)
        const suffix = token.slice(index + 5)
        const resolved = suffix
          .replace(/\//g, "\\")
          .split("\\")
          .filter((part: string) => part && part !== ".")
          .reduce((current: string, part: string) => {
            if (part === "..") return dirname(current)
            return join(current, part)
          }, base)

        if (await exists(resolved)) return resolved
      }

      if (await exists(token)) return token
    }

    return null
  }

  for (const path of paths) {
    if (hasExt(path, "cmd") || hasExt(path, "bat")) {
      const resolved = await resolveCmd(path)
      if (resolved) return resolved
    }

    if (!extname(path)) {
      const cmd = `${path}.cmd`
      if (await exists(cmd)) {
        const resolved = await resolveCmd(cmd)
        if (resolved) return resolved
      }

      const bat = `${path}.bat`
      if (await exists(bat)) {
        const resolved = await resolveCmd(bat)
        if (resolved) return resolved
      }
    }
  }

  const key = appName
    .split("")
    .filter((value: string) => /[a-z0-9]/i.test(value))
    .map((value: string) => value.toLowerCase())
    .join("")

  if (key) {
    for (const path of paths) {
      const dirs = [dirname(path), dirname(dirname(path)), dirname(dirname(dirname(path)))]
      for (const dir of dirs) {
        try {
          for (const entry of await readdir(dir)) {
            const candidate = join(dir, entry)
            if (!hasExt(candidate, "exe")) continue
            const stem = entry.replace(/\.exe$/i, "")
            const name = stem
              .split("")
              .filter((value: string) => /[a-z0-9]/i.test(value))
              .map((value: string) => value.toLowerCase())
              .join("")
            if (name.includes(key) || key.includes(name)) return candidate
          }
        } catch {
          continue
        }
      }
    }
  }

  return paths[0] ?? null
}
