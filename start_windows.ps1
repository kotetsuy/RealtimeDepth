param([string]$Config = (Join-Path $PSScriptRoot 'config.windows.yaml'), [int]$TimeoutSeconds = 180)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv-rocm72-windows\Scripts\python.exe'
$stateDir = Join-Path $PSScriptRoot '.windows-state'
$stateFile = Join-Path $stateDir 'server.json'
if (!(Test-Path $python)) { throw 'Run setup_windows.ps1 first.' }
$Config = (Resolve-Path -LiteralPath $Config).Path
if (Test-Path $stateFile) { throw 'Server state already exists. Run stop_windows.ps1 before starting.' }
New-Item -ItemType Directory -Force $stateDir | Out-Null
$oldConfig = $env:CONFIG_PATH
$oldEncoding = $env:PYTHONIOENCODING
$process = $null
try {
    $env:CONFIG_PATH = $Config
    $env:PYTHONIOENCODING = 'utf-8'
    $port = & $python (Join-Path $PSScriptRoot 'check_windows.py') --port
    if ($LASTEXITCODE -ne 0) { throw 'Cannot read server port.' }
    # Refuse an occupied port so another server cannot satisfy readiness.
    $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, [int]$port)
    try { $listener.Start() } finally { $listener.Stop() }
    $appPath = Join-Path $PSScriptRoot 'app.py'
    $process = Start-Process -FilePath $python -ArgumentList @('-u', ('"' + $appPath + '"')) -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $stateDir 'stdout.log') -RedirectStandardError (Join-Path $stateDir 'stderr.log')
    @{ pid = $process.Id; started = $process.StartTime.ToUniversalTime().ToString('o'); python = $python; config = $Config } | ConvertTo-Json | Set-Content -Encoding UTF8 $stateFile
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $process.Refresh()
        if ($process.HasExited) { throw "Server exited. See $stateDir\stderr.log" }
        try {
            $status = Invoke-RestMethod "http://127.0.0.1:$port/stats" -TimeoutSec 2
            if ($null -ne $status.fps) {
                Write-Host "Ready: http://127.0.0.1:$port/ (PID $($process.Id))"
                Write-Host ($status | ConvertTo-Json -Compress)
                return
            }
        } catch { }
        Start-Sleep -Milliseconds 500
    }
    throw "Readiness timed out. See $stateDir logs."
} catch {
    if ($process -and !$process.HasExited) { $process.Kill(); $process.WaitForExit() }
    if ($process -and (Test-Path $stateFile)) { Remove-Item -LiteralPath $stateFile }
    throw
} finally {
    $env:CONFIG_PATH = $oldConfig
    $env:PYTHONIOENCODING = $oldEncoding
}
