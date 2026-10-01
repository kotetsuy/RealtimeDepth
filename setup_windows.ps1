param([string]$Python = '', [switch]$DownloadModel)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Executable $Arguments" }
}
$venvPython = Join-Path $PSScriptRoot '.venv-rocm72-windows\Scripts\python.exe'
if (!(Test-Path $venvPython)) {
    if ($Python) {
        Invoke-Checked $Python @('-m', 'venv', '.venv-rocm72-windows')
    } elseif (Test-Path '.python\python.exe') {
        Invoke-Checked '.\.python\python.exe' @('-m', 'venv', '.venv-rocm72-windows')
    } else {
        Invoke-Checked 'py' @('-3.12', '-m', 'venv', '.venv-rocm72-windows')
    }
}
Invoke-Checked $venvPython @('-m', 'pip', 'install', '--upgrade', 'pip')
Invoke-Checked $venvPython @('-m', 'pip', 'install', '-r', 'requirements-rocm72-windows.txt')
Invoke-Checked $venvPython @('-m', 'pip', 'install', '-r', 'requirements.txt')
Invoke-Checked $venvPython @('-m', 'pip', 'check')
Invoke-Checked $venvPython @('check_windows.py')
if ($DownloadModel) {
    if (!(Test-Path 'Depth-Anything-V2')) {
        Invoke-Checked 'git' @('clone', 'https://github.com/DepthAnything/Depth-Anything-V2.git', 'Depth-Anything-V2')
    }
    $checkpoint = 'Depth-Anything-V2\checkpoints\depth_anything_v2_vits.pth'
    if (!(Test-Path $checkpoint)) {
        New-Item -ItemType Directory -Force 'Depth-Anything-V2\checkpoints' | Out-Null
        Invoke-WebRequest 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth' -OutFile "$checkpoint.partial"
        Move-Item -LiteralPath "$checkpoint.partial" -Destination $checkpoint
    }
}
if (!(Test-Path 'Depth-Anything-V2\depth_anything_v2') -or !(Test-Path 'Depth-Anything-V2\checkpoints\depth_anything_v2_vits.pth')) {
    throw 'Model source/Small checkpoint missing. Run setup_windows.ps1 -DownloadModel or follow WINDOWS.md.'
}
Invoke-Checked 'git' @('-C', 'Depth-Anything-V2', 'rev-parse', 'HEAD')
Write-Host 'Setup complete. Run .\start_windows.ps1'
