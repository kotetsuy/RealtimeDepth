$ErrorActionPreference = 'Stop'
$stateFile = Join-Path $PSScriptRoot '.windows-state\server.json'
if (!(Test-Path $stateFile)) { Write-Host 'No managed server.'; return }
$state = Get-Content -Raw -Encoding UTF8 $stateFile | ConvertFrom-Json
$process = Get-Process -Id $state.pid -ErrorAction SilentlyContinue
if ($process) {
    if ($process.StartTime.ToUniversalTime().ToString('o') -ne $state.started -or $process.Path -ne $state.python) {
        throw 'PID identity mismatch; refusing to stop an unrelated process.'
    }
    Stop-Process -Id $process.Id
    $process.WaitForExit()
}
Remove-Item -LiteralPath $stateFile
Write-Host 'Server stopped.'
