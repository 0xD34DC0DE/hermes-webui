<#
.SYNOPSIS
    Gracefully restart the Hermes WebUI server.

.DESCRIPTION
    Phase 1 of the "graceful server restart" plan. Reads
    HERMES_WEBUI_STATE_DIR/server.pid, sends a Ctrl-C / SIGINT to the
    old python process, waits for the TCP port to free, then launches
    start.ps1 in a brand-new, fully-detached PowerShell process so the
    new server is owned by a different PID and a different process tree
    than the caller.

    Mirrors start.ps1's .env handling and Python/agent discovery so a
    restart picks up exactly the same configuration as the original
    launch -- no env vars are re-read from the parent PowerShell session.
    That is intentional: the restart must succeed even if the operator
    did not pre-set HERMES_HOME / HERMES_WEBUI_STATE_DIR in their shell.

.PARAMETER GraceSeconds
    Time to wait between sending the SIGINT and probing for port release.
    Default 4 (a hair more than the default /api/server/restart grace of
    2s, so the server has finished draining in-flight SSE responses before
    we look for the port to free).

.PARAMETER PortTimeout
    Maximum seconds to wait for the TCP port to free before giving up.
    Default 15 -- sized to cover the Windows SO_EXCLUSIVEADDRUSE cleanup
    window that the existing QuietHTTPServer.server_bind retry loop
    (server.py:164) protects against on the bind side.

.EXAMPLE
    .\restart.ps1
    # Graceful restart with default timeouts.

.EXAMPLE
    .\restart.ps1 -GraceSeconds 8 -PortTimeout 30
    # Longer drain + longer wait, useful for big in-flight SSE streams.

.LINK
    Phase 1 plan: .hermes/plans/2026-06-18_181133-graceful-server-restart.md
#>

[CmdletBinding()]
param(
    [int]$GraceSeconds = 4,
    [int]$PortTimeout = 15,
    [switch]$WithRunner = $false
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSCommandPath

# === Load .env (mirror start.ps1's discovery verbatim) ======================
# We deliberately re-read .env here rather than inheriting from the calling
# process. The operator may have run `pwsh -File restart.ps1` from anywhere,
# and we cannot rely on them having exported HERMES_HOME / agent dir vars.
$envFile = Join-Path $RepoRoot '.env'
if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile -Encoding UTF8) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith('#') -or -not $trimmed.Contains('=')) { continue }
        $kv = $trimmed -split '=', 2
        $key = ($kv[0].Trim() -replace '^export\s+', '')
        # Filter out shell-readonly vars (UID, GID, EUID, EGID, PPID) per start.sh
        if ($key -in @('UID', 'GID', 'EUID', 'EGID', 'PPID')) { continue }
        if ($key -notmatch '^[A-Za-z_][A-Za-z0-9_]*$') { continue }
        # Same explicit $null check start.ps1 uses: an env var explicitly set
        # to '' is still considered "set" and should NOT be overridden by .env.
        if ($null -ne [Environment]::GetEnvironmentVariable($key)) { continue }
        $val = $kv[1]
        if ($val -match '^"(.*)"$') { $val = $Matches[1] }
        elseif ($val -match "^'(.*)'$") { $val = $Matches[1] }
        [Environment]::SetEnvironmentVariable($key, $val)
    }
}

# === Resolve state dir + read server.pid ===================================
if (-not $env:HERMES_WEBUI_STATE_DIR) {
    if ($env:HERMES_HOME) {
        $env:HERMES_WEBUI_STATE_DIR = Join-Path $env:HERMES_HOME 'webui'
    } elseif ($env:LOCALAPPDATA) {
        $env:HERMES_HOME = Join-Path $env:LOCALAPPDATA 'hermes'
        $env:HERMES_WEBUI_STATE_DIR = Join-Path $env:HERMES_HOME 'webui'
    } else {
        $env:HERMES_HOME = Join-Path $env:USERPROFILE '.hermes'
        $env:HERMES_WEBUI_STATE_DIR = Join-Path $env:HERMES_HOME 'webui'
    }
}

$pidFile = Join-Path $env:HERMES_WEBUI_STATE_DIR 'server.pid'
if (-not (Test-Path $pidFile)) {
    Write-Error "server.pid not found at $pidFile -- is the server running?"
    exit 2
}

# server.pid is JSON (not raw int) -- see api/server_lifecycle.write_pid_file.
# ConvertFrom-Json is used rather than [int]::Parse so we can validate the
# payload's shape and surface a clear error if a future version adds fields
# that the script doesn't know about.
$pidPayload = Get-Content $pidFile -Raw | ConvertFrom-Json -ErrorAction Stop
if (-not $pidPayload.pid -or -not $pidPayload.port) {
    Write-Error "server.pid at $pidFile is missing required 'pid' or 'port' field."
    exit 2
}
$oldPid = [int]$pidPayload.pid
$port = [int]$pidPayload.port
$host = if ($pidPayload.host) { [string]$pidPayload.host } else { '127.0.0.1' }

Write-Host "[restart.ps1] Stopping old server pid=$oldPid ${host}:${port}" -ForegroundColor Cyan

# === Stop the old server ====================================================
# SIGINT is what _handle_shutdown uses. SIGINT triggers Python's
# KeyboardInterrupt, the serve_forever() finally block runs, and the
# server.pid file is removed before exit (see server.py clear_pid_file).
# Using Stop-Process -Force would be faster but skips the graceful drain,
# so we prefer the interrupt signal first and fall back to -Force only if
# the process is still alive after the grace window.
try {
    Stop-Process -Id $oldPid -SignalKind Interrupt -ErrorAction Stop
} catch {
    Write-Host "[restart.ps1] Old process $oldPid not running: $($_.Exception.Message)" -ForegroundColor Yellow
}

if ($GraceSeconds -gt 0) {
    Write-Host "[restart.ps1] Waiting $GraceSeconds s for graceful drain..." -ForegroundColor DarkGray
    Start-Sleep -Seconds $GraceSeconds
}

# === Wait for the port to actually free ====================================
# We probe with [System.Net.Sockets.TcpClient]::ConnectAsync -- much faster
# than Test-NetConnection (which has a ~250ms per-call overhead) and gives
# us a deterministic 200ms-per-probe bound. On Windows the SO_EXCLUSIVEADDRUSE
# cleanup can take a couple of seconds; 15s default gives comfortable headroom.
$portFree = $false
$deadline = (Get-Date).AddSeconds($PortTimeout)
$probeCount = 0
while ((Get-Date) -lt $deadline) {
    $probeCount += 1
    try {
        $tcp = [System.Net.Sockets.TcpClient]::new()
        $iar = $tcp.BeginConnect($host, $port, $null, $null)
        $ok = $iar.AsyncWaitHandle.WaitOne(200)
        if (-not $ok) {
            # Connection is taking too long -- assume something is still bound.
            $tcp.Close()
        } else {
            $tcp.EndConnect($iar)
            $tcp.Close()
        }
    } catch {
        # Connect failed -- port is free.
        $portFree = $true
        break
    }
    Start-Sleep -Milliseconds 100
}

if (-not $portFree) {
    Write-Error "Port ${host}:${port} still in use after $PortTimeout s -- refusing to relaunch. Try -PortTimeout 60 or stop the old process manually."
    exit 3
}
Write-Host "[restart.ps1] Port ${host}:${port} is free after $probeCount probe(s)" -ForegroundColor Green

# === Launch detached replacement ============================================
$startScript = Join-Path $RepoRoot 'start.ps1'
if (-not (Test-Path $startScript)) {
    Write-Error "start.ps1 not found at $startScript -- is restart.ps1 in the hermes-webui repo root?"
    exit 4
}

# Start-Process with -WindowStyle Hidden detaches the new PowerShell from
# this script's console. We intentionally do NOT use CreateNewProcessGroup
# here because the python child sets up its own SIGINT handling and is
# happy to inherit the console session; what we need is for the new PS to
# outlive *this* PS, which Start-Process guarantees.
#
# -PassThru returns a Process object so a CI step could capture the new
# PowerShell PID if it wanted to. For interactive use we just log it.
$pwshArgs = @(
    '-NoLogo'
    '-NoProfile'
    '-ExecutionPolicy', 'Bypass'
    '-File', $startScript
)
$proc = Start-Process -FilePath 'powershell.exe' `
                      -ArgumentList $pwshArgs `
                      -WindowStyle Hidden `
                      -PassThru

Write-Host "[restart.ps1] Launched detached start.ps1 (pwsh pid=$($proc.Id))" -ForegroundColor Cyan
Write-Host "[restart.ps1] New server PID will appear in $pidFile once it binds."
Write-Host "[restart.ps1] Watch with: Get-Content '$pidFile' -Wait"

# === Optional runner restart (Phase 2) ===================================
# When -WithRunner is passed, also SIGINT the old runner, wait for its
# port to free, and launch a detached start-runner.ps1 in a fresh
# PowerShell. The new runner inherits the same HERMES_HOME /
# HERMES_WEBUI_STATE_DIR / HERMES_WEBUI_RUNNER_PORT as the WebUI so the
# WebUI's runtime-adapter can find it on the same URL after restart.
#
# This is intentionally the LAST step. Order matters: the runner must
# be alive before the WebUI comes back up, otherwise the WebUI's
# runtime-adapter (set to runner-local) would reject requests until it
# does. The WebUI's existing client-side retry on a runner outage
# covers the brief gap during a real outage, but a planned restart
# should never leave the WebUI pointing at a dead runner.
if ($WithRunner) {
    $runnerPidFile = Join-Path $env:HERMES_WEBUI_STATE_DIR 'runner.pid'
    $runnerScript = Join-Path $RepoRoot 'start-runner.ps1'

    if (Test-Path $runnerPidFile) {
        try {
            $runnerPayload = Get-Content $runnerPidFile -Raw | ConvertFrom-Json -ErrorAction Stop
            $runnerPid = [int]$runnerPayload.pid
            $runnerPort = [int]$runnerPayload.port
            $runnerHost = if ($runnerPayload.host) { [string]$runnerPayload.host } else { '127.0.0.1' }
            Write-Host "[restart.ps1] Stopping old runner pid=$runnerPid ${runnerHost}:${runnerPort}" -ForegroundColor Cyan
            try {
                Stop-Process -Id $runnerPid -SignalKind Interrupt -ErrorAction Stop
            } catch {
                Write-Host "[restart.ps1] Old runner $runnerPid not running: $($_.Exception.Message)" -ForegroundColor Yellow
            }
            # Wait for runner port to free.
            $deadline = (Get-Date).AddSeconds($PortTimeout)
            $portFree = $false
            while ((Get-Date) -lt $deadline) {
                try {
                    $tcp = [System.Net.Sockets.TcpClient]::new()
                    $iar = $tcp.BeginConnect($runnerHost, $runnerPort, $null, $null)
                    $ok = $iar.AsyncWaitHandle.WaitOne(200)
                    if (-not $ok) { $tcp.Close() }
                    else { $tcp.EndConnect($iar); $tcp.Close() }
                } catch { $portFree = $true; break }
                Start-Sleep -Milliseconds 100
            }
            if (-not $portFree) {
                Write-Host "[restart.ps1] WARN: runner port still bound after $PortTimeout s -- continuing anyway." -ForegroundColor Yellow
            }
        } catch {
            Write-Host "[restart.ps1] Could not parse runner.pid: $($_.Exception.Message)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "[restart.ps1] No runner.pid at $runnerPidFile -- assuming no runner was running." -ForegroundColor Yellow
    }

    if (Test-Path $runnerScript) {
        $runnerPwshArgs = @('-NoLogo', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $runnerScript)
        $rproc = Start-Process -FilePath 'powershell.exe' `
                                -ArgumentList $runnerPwshArgs `
                                -WindowStyle Hidden `
                                -PassThru
        Write-Host "[restart.ps1] Launched detached start-runner.ps1 (pwsh pid=$($rproc.Id))" -ForegroundColor Cyan
    } else {
        Write-Host "[restart.ps1] WARN: $runnerScript not found -- skipping runner relaunch." -ForegroundColor Yellow
    }
}

exit 0