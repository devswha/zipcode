# Start llama-server for zipcode's remote-mode inference.
#
# Assumes setup.ps1 has already been run (llama.cpp + model present).
# Keep this PowerShell window open while the server is running; Ctrl+C
# to stop.
#
# Usage:
#   .\start-server.ps1
#   .\start-server.ps1 -Port 9000 -ContextSize 65536
#   .\start-server.ps1 -ModelPath C:\zipcode-server\models\some-other.gguf

[CmdletBinding()]
param(
    [string] $WorkDir      = "C:\zipcode-server",
    [int]    $Port         = 8080,
    [string] $Alias        = "zipcode-remote",
    [string] $ModelPath    = "",
    [int]    $ContextSize  = 131072,
    [int]    $GpuLayers    = 999,
    [switch] $NoFlashAttn
)

$ErrorActionPreference = 'Stop'

$LlamaBin = Join-Path $WorkDir "llama\llama-server.exe"
if (-not (Test-Path $LlamaBin)) {
    Write-Error "llama-server.exe not found at $LlamaBin. Run .\setup.ps1 first."
}

if (-not $ModelPath) {
    # Pick the first .gguf under models/ — if there are several, the user
    # should pass -ModelPath explicitly.
    $candidates = @(Get-ChildItem -Path (Join-Path $WorkDir "models") -Filter "*.gguf" -ErrorAction SilentlyContinue)
    if ($candidates.Count -eq 0) {
        Write-Error "No .gguf files found in $WorkDir\models. Run .\setup.ps1 or pass -ModelPath."
    } elseif ($candidates.Count -gt 1) {
        Write-Host "Multiple GGUF files found, using the first alphabetically:" -ForegroundColor Yellow
        $candidates | ForEach-Object { Write-Host "  $($_.Name)" }
        Write-Host "Pass -ModelPath explicitly to pick a different one."
        $ModelPath = $candidates[0].FullName
    } else {
        $ModelPath = $candidates[0].FullName
    }
}

Write-Host "=== Starting llama-server ===" -ForegroundColor Cyan
Write-Host "Model  : $ModelPath"
Write-Host "Port   : $Port (listening on 0.0.0.0)"
Write-Host "Alias  : $Alias"
Write-Host "Ctx    : $ContextSize tokens"
Write-Host "GPU ly : $GpuLayers"
Write-Host "FlashA : $(-not $NoFlashAttn.IsPresent)"
Write-Host ""

$lanIp = (Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.InterfaceAlias -match "Ethernet|Wi-Fi" -and $_.IPAddress -notmatch "^169" } |
    Select-Object -First 1).IPAddress
if ($lanIp) {
    Write-Host "Linux client config:" -ForegroundColor Green
    Write-Host "  export ZIPCODE_LLAMA_SERVER_URL=http://${lanIp}:$Port"
    Write-Host "  export ZIPCODE_LLAMA_SERVER_ALIAS=$Alias"
    Write-Host ""
}

# Build argv
$argList = @(
    "-m", $ModelPath,
    "--host", "0.0.0.0",
    "--port", $Port,
    "--alias", $Alias,
    "--jinja",
    "-c", $ContextSize,
    "-ngl", $GpuLayers
)
if (-not $NoFlashAttn.IsPresent) {
    $argList += @("-fa", "on")
}

Write-Host "Launching... (Ctrl+C to stop)" -ForegroundColor Yellow
Write-Host ""
& $LlamaBin @argList
