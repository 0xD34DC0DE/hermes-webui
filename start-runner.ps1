<#
.SYNOPSIS
    Start the Hermes WebUI Runner server (out-of-process AIAgent host).

.DESCRIPTION
    Phase 2 of the graceful-restart plan. Launches
    api.runner_server in a foreground PowerShell process bound to a
    distinct port (default 8788) so the WebUI server (default 8787) can
    reach it via HERMES_WEBUI_RUNNER_BASE_URL.

    Mirrors start.ps1's discovery: .env loading, Python resolution,
    HERMES_HOME / HERMES_WEBUI_AGENT_DIR validation. The runner uses the
    same Python interpreter and the same agent install directory as the
    WebUI -- they share the AIAgent dependency.

    Writes runner.pid under HERMES_WEBUI_STATE_DIR so restart.ps1 can
    manage the runner's lifecycle alongside the WebUI's. Reads
    HERMES_WEBUI_RUNNER_PORT (default 8788) for the bind port and
    HERMES_WEBUI_RUNNER_HOST (default 127.0.0.1) for the bind host.

.PARAMETER Port
    TCP port the runner binds to. Overrides HERMES_WEBUI_RUNNER_PORT.

.PARAMETER BindHost
    Bind address. Overrides HERMES_WEBUI_RUNNER_HOST. Default 127.0.0.1.

.EXAMPLE
    .\start-runner.ps1
    # Foreground; bind 127.0.0.1:8788.

.EXAMPLE
    $env:HERMES_WEBUI_RUNNER_PORT = '9000'
    .\start-runner.ps1

.LINK
    Phase 2 plan: .hermes/plans/2026-06-18_181200-phase2-runner-server-design.md
#>

[CmdletBinding()]
param(
    [int]$Port = 0,
    [string]$BindHost = ''
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSCommandPath

# === Load .env (mirror start.ps1's filtering) =======================
$envFile = Join-Path $RepoRoot '.env'
if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile -Encoding UTF8) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith('#') -or -not $trimmed.Contains('=')) { continue }
        $kv = $trimmed -split '=', 2
        $key = ($kv[0].Trim() -replace '^export\s+', '')
        if ($key -in @('UID', 'GID', 'EUID', 'EGID', 'PPID')) { continue }
        if ($key -notmatch '^[A-Za-z_][A-Za-z0-9_]*$') { continue }
        if ($null -ne [Environment]::GetEnvironmentVariable($key)) { continue }
        $val = $kv[1]
        if ($val -match '^"(.*)"$') { $val = $Matches[1] }
        elseif ($val -match "^'(.*)'$") { $val = $Matches[1] }
        [Environment]::SetEnvironmentVariable($key, $val)
    }
}

# === Find Python (mirror start.ps1's order) =============================
$Python = $env:HERMES_WEBUI_PYTHON
if (-not $Python) {
    foreach ($candidate in @('python3', 'python', 'py')) {
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($cmd) { $Python = $cmd.Source; break }
    }
}
if (-not $Python) {
    Write-Error 'Python 3 is required to run api.runner_server.'
    exit 1
}

# === Find Hermes Agent dir (same as start.ps1) ===========================
$AgentDir = $env:HERMES_WEBUI_AGENT_DIR
if (-not $AgentDir) {
    $candidates = @()
    $candidates += (Join-Path $env:USERPROFILE '.hermes\hermes-agent')
    foreach ($root in @($env:LOCALAPPDATA, ${env:ProgramW6432}, ${env:ProgramFiles}, ${env:ProgramFiles(x86)})) {
        if ($root) { $candidates += (Join-Path $root 'hermes\hermes-agent') }
    }
    $candidates += (Join-Path (Split-Path -Parent $RepoRoot) 'hermes-agent')
    $candidates = $candidates | Select-Object -Unique
    foreach ($c in $candidates) {
        if (Test-Path (Join-Path $c 'hermes_cli') -PathType Container) { $AgentDir = $c; break }
    }
}
if (-not $AgentDir) {
    Write-Error 'hermes-agent not found. Set HERMES_WEBUI_AGENT_DIR explicitly.'
    exit 1
}

# === Prefer the agent's venv Python if available ========================
$agentVenvPython = Join-Path $AgentDir 'venv\Scripts\python.exe'
if (Test-Path $agentVenvPython) {
    $Python = $agentVenvPython
}

# === Resolve bind + state defaults =======================================
$BindHostFinal = if ($BindHost) { $BindHost } elseif ($env:HERMES_WEBUI_RUNNER_HOST) { $env:HERMES_WEBUI_RUNNER_HOST } else { '127.0.0.1' }
$PortFinal = if ($Port) {
    $Port
} elseif ($env:HERMES_WEBUI_RUNNER_PORT) {
    $parsedPort = 0
    if (-not [int]::TryParse($env:HERMES_WEBUI_RUNNER_PORT, [ref]$parsedPort)) {
        Write-Error "HERMES_WEBUI_RUNNER_PORT='$($env:HERMES_WEBUI_RUNNER_PORT)' is not a valid integer port."
        exit 1
    }
    if ($parsedPort -lt 1 -or $parsedPort -gt 65535) {
        Write-Error "HERMES_WEBUI_RUNNER_PORT=$parsedPort is out of TCP-port range."
        exit 1
    }
    $parsedPort
} else {
    8788
}
$env:HERMES_WEBUI_RUNNER_HOST = $BindHostFinal
$env:HERMES_WEBUI_RUNNER_PORT = "$PortFinal"
if (-not $env:HERMES_HOME) {
    if ($env:LOCALAPPDATA) {
        $env:HERMES_HOME = Join-Path $env:LOCALAPPDATA 'hermes'
    } else {
        $env:HERMES_HOME = Join-Path $env:USERPROFILE '.hermes'
    }
}
if (-not $env:HERMES_WEBUI_STATE_DIR) {
    $env:HERMES_WEBUI_STATE_DIR = Join-Path $env:HERMES_HOME 'webui'
}

# === Ensure dirs exist ==================================================
New-Item -ItemType Directory -Force -Path $env:HERMES_HOME | Out-Null
New-Item -ItemType Directory -Force -Path $env:HERMES_WEBUI_STATE_DIR | Out-Null

# === Probe for a port collision =========================================
# The runner's port must not collide with the WebUI's. If the operator
# set them to the same port (likely a typo), fail fast with a clear
# message rather than dying on bind inside Python.
$webuiPort = if ($env:HERMES_WEBUI_PORT) { [int]$env:HERMES_WEBUI_PORT } else { 8787 }
if ($PortFinal -eq $webuiPort) {
    Write-Error "HERMES_WEBUI_RUNNER_PORT=$PortFinal collides with HERMES_WEBUI_PORT=$webuiPort. Pick distinct ports (default runner=8788, webui=8787)."
    exit 1
}

# === Write runner.pid before launch =====================================
# Mirrors server.py's pattern: write the pid file in a separate small
# Python invocation so a future restart.ps1 can find us. The pid file
# is the same JSON shape as server.pid for restart tooling symmetry.
$runnerPidPayload = @{
    pid = $PID
    port = $PortFinal
    host = $BindHostFinal
    started_at = (Get-Date).ToUniversalTime().ToString('o')
    role = 'runner'
} | ConvertTo-Json -Compress
$pidFile = Join-Path $env:HERMES_WEBUI_STATE_DIR 'runner.pid'
$tmpPid = "$pidFile.tmp"
[System.IO.File]::WriteAllText($tmpPid, $runnerPidPayload, [System.Text.UTF8Encoding]::new($false))
Move-Item -Force $tmpPid $pidFile

# === Launch (foreground, matches start.ps1) =============================
Write-Host "[start-runner.ps1] Hermes WebUI Runner (Phase 2)" -ForegroundColor Cyan
Write-Host "[start-runner.ps1] Python:     $Python"
Write-Host "[start-runner.ps1] Agent dir:  $AgentDir"
Write-Host "[start-runner.ps1] State dir:  $env:HERMES_WEBUI_STATE_DIR"
Write-Host "[start-runner.ps1] Binding:    ${BindHostFinal}:${PortFinal}"
Write-Host ""

try {
    # `python -m api.runner_server` launches the in-process runner.
    # The runner uses the same AIAgent install as the WebUI; nothing
    # extra to import. We deliberately do NOT add --reload or
    # auto-restart flags: the runner is meant to be a long-lived
    # process, and a hot-reload during a chat turn would lose state.
    & $Python -m api.runner_server
} finally {
    # Clean up the pid file on exit (matches server.py's clear_pid_file).
    if (Test-Path $pidFile) {
        try {
            Remove-Item -Force $pidFile
        } catch {
            Write-Host "[start-runner.ps1] Could not remove runner.pid: $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
}