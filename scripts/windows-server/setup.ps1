# Windows zipcode remote-server setup.
#
# Prepares a Windows PC (tested: RTX 4080 Super, Win 11) to serve
# Gemma 4 26B A4B Claude-Opus-Distill as an HTTP inference endpoint for
# zipcode's remote mode (ZIPCODE_LLAMA_SERVER_URL). One-shot install:
# downloads llama.cpp CUDA Windows build, downloads the model, opens a
# firewall rule. Idempotent — safe to re-run.
#
# Usage (PowerShell, admin required for firewall):
#   .\setup.ps1
#   .\setup.ps1 -Port 8080 -ModelUrl <direct-gguf-url>
#   .\setup.ps1 -LlamaCppVersion b7400     # pin a specific llama.cpp release
#
# After success, run start-server.ps1 to bring the server up.

[CmdletBinding()]
param(
    [string] $WorkDir = "C:\zipcode-server",
    [int]    $Port = 8080,
    # Default model: TeichAI Gemma 4 26B A4B Claude Opus Distill, Q4_K_M
    # (~16.8 GB, fits RTX 4080 Super 16 GB VRAM comfortably). The
    # repository uses dot-separated lowercase quant suffixes — do NOT
    # change `.q4_k_m.gguf` to `-Q4_K_M.gguf`, HuggingFace is
    # case-sensitive. Override with -ModelUrl for a different quant or
    # variant (e.g. Q5_K_M for slightly better quality if VRAM allows).
    [string] $ModelUrl = "https://huggingface.co/TeichAI/gemma-4-26B-A4B-it-Claude-Opus-Distill-GGUF/resolve/main/gemma-4-26B-A4B-it-Claude-Opus-Distill.q4_k_m.gguf",
    [string] $ModelFileName = "gemma-4-26B-A4B-it-Claude-Opus-Distill.q4_k_m.gguf",
    # Pin to a specific llama.cpp release tag (e.g. "b7400"). "latest"
    # queries GitHub's release API.
    [string] $LlamaCppVersion = "latest",
    # CUDA toolkit version for the llama.cpp Windows binaries. Recent
    # llama.cpp releases (since ~b7800) ship separate 12.4 and 13.1
    # variants. Match this to your NVIDIA driver's supported CUDA
    # version (check `nvidia-smi` top-right "CUDA Version").
    [string] $CudaVersion = "13.1",
    [switch] $SkipModelDownload,
    [switch] $SkipFirewall
)

$ErrorActionPreference = 'Stop'
Write-Host "=== zipcode remote-server setup ===" -ForegroundColor Cyan
Write-Host "WorkDir   : $WorkDir"
Write-Host "Port      : $Port"
Write-Host "Model URL : $ModelUrl"
Write-Host ""

# ── Step 1: CUDA sanity check ────────────────────────────────────────
Write-Host "[1/5] Checking NVIDIA driver..." -ForegroundColor Yellow
$nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $nvsmi) {
    Write-Error "nvidia-smi not found. Install the latest NVIDIA driver before running this script."
}
& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
Write-Host ""

# ── Step 2: Work directory ───────────────────────────────────────────
Write-Host "[2/5] Preparing work directory $WorkDir ..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $WorkDir "llama")  | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $WorkDir "models") | Out-Null

# ── Step 3: llama.cpp CUDA Windows build ─────────────────────────────
$LlamaBin = Join-Path $WorkDir "llama\llama-server.exe"
if (Test-Path $LlamaBin) {
    Write-Host "[3/5] llama-server.exe already present, skipping download." -ForegroundColor Green
} else {
    Write-Host "[3/5] Downloading llama.cpp CUDA Windows build..." -ForegroundColor Yellow

    # Resolve release tag
    if ($LlamaCppVersion -eq "latest") {
        $release = Invoke-RestMethod "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
        $tag = $release.tag_name
        Write-Host "  Latest release: $tag"
    } else {
        $tag = $LlamaCppVersion
    }

    # llama.cpp release assets (since ~b7800, CUDA is versioned):
    #   llama-<tag>-bin-win-cuda-<cuda>-x64.zip    (main binaries)
    #   cudart-llama-bin-win-cuda-<cuda>-x64.zip   (cudart runtime)
    $mainZip   = Join-Path $WorkDir "llama-cuda.zip"
    $cudartZip = Join-Path $WorkDir "llama-cudart.zip"
    $baseUrl   = "https://github.com/ggml-org/llama.cpp/releases/download/$tag"

    Write-Host "  CUDA variant: $CudaVersion"
    Write-Host "  Fetching main binaries..."
    Invoke-WebRequest -Uri "$baseUrl/llama-$tag-bin-win-cuda-$CudaVersion-x64.zip" -OutFile $mainZip
    Write-Host "  Fetching cudart runtime..."
    Invoke-WebRequest -Uri "$baseUrl/cudart-llama-bin-win-cuda-$CudaVersion-x64.zip" -OutFile $cudartZip

    Write-Host "  Extracting..."
    Expand-Archive -Force $mainZip   -DestinationPath (Join-Path $WorkDir "llama")
    Expand-Archive -Force $cudartZip -DestinationPath (Join-Path $WorkDir "llama")
    Remove-Item $mainZip, $cudartZip

    if (-not (Test-Path $LlamaBin)) {
        Write-Error "llama-server.exe not found after extraction. Check release asset naming."
    }
    Write-Host "  OK" -ForegroundColor Green
}

# Quick sanity run. llama-server writes init/version info on stderr;
# under $ErrorActionPreference='Stop' PowerShell promotes native stderr
# to a terminating error, so downgrade EAP around the probe and discard
# the noisy output.
Write-Host "  Version probe: " -NoNewline
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    & $LlamaBin --version 2>$null | Out-Null
} finally {
    $ErrorActionPreference = $prevEAP
}
Write-Host "OK (exit=$LASTEXITCODE)" -ForegroundColor Green

# ── Step 4: Model download ───────────────────────────────────────────
$ModelPath = Join-Path $WorkDir "models\$ModelFileName"
if ($SkipModelDownload) {
    Write-Host "[4/5] -SkipModelDownload set — leaving model alone." -ForegroundColor DarkYellow
} elseif (Test-Path $ModelPath) {
    $size = (Get-Item $ModelPath).Length / 1GB
    Write-Host ("[4/5] Model already present ({0:N2} GB), skipping download." -f $size) -ForegroundColor Green
} else {
    Write-Host "[4/5] Downloading model (large, can take 10-30 min depending on connection)..." -ForegroundColor Yellow
    Write-Host "  URL : $ModelUrl"
    Write-Host "  Path: $ModelPath"
    # curl.exe handles HuggingFace's 302 redirects well and shows progress
    # on stderr. Under $ErrorActionPreference='Stop' PowerShell promotes
    # native stderr into a terminating error on the first progress line,
    # so downgrade EAP around the call.
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & curl.exe -L --fail -o $ModelPath $ModelUrl
    } finally {
        $ErrorActionPreference = $prevEAP
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Model download failed (curl exit $LASTEXITCODE). Check the URL or network."
    }
    $size = (Get-Item $ModelPath).Length / 1GB
    Write-Host ("  OK ({0:N2} GB)" -f $size) -ForegroundColor Green
}
Write-Host ""

# ── Step 5: Firewall rule ────────────────────────────────────────────
if ($SkipFirewall) {
    Write-Host "[5/5] -SkipFirewall set — skipping firewall rule." -ForegroundColor DarkYellow
} else {
    Write-Host "[5/5] Ensuring Windows Firewall allows inbound TCP $Port on Private profile..." -ForegroundColor Yellow
    $ruleName = "zipcode llama-server ($Port)"
    $existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "  Rule already exists, leaving it." -ForegroundColor Green
    } else {
        try {
            New-NetFirewallRule -DisplayName $ruleName `
                -Direction Inbound `
                -Protocol TCP -LocalPort $Port `
                -Action Allow -Profile Private | Out-Null
            Write-Host "  Created rule '$ruleName'" -ForegroundColor Green
        } catch {
            Write-Warning "Could not create firewall rule (likely not admin). Error: $_"
            Write-Warning "Re-run this script from an elevated PowerShell, or add the rule manually:"
            Write-Warning "  New-NetFirewallRule -DisplayName '$ruleName' -Direction Inbound -Protocol TCP -LocalPort $Port -Action Allow -Profile Private"
        }
    }
}
Write-Host ""

# ── Summary ──────────────────────────────────────────────────────────
Write-Host "=== Setup complete ===" -ForegroundColor Cyan
Write-Host "Start the server with:" -ForegroundColor White
Write-Host "  .\start-server.ps1" -ForegroundColor Green
Write-Host ""
Write-Host "Then from your Linux client:" -ForegroundColor White
$lanIp = (Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.InterfaceAlias -match "Ethernet|Wi-Fi" -and $_.IPAddress -notmatch "^169" } |
    Select-Object -First 1).IPAddress
if ($lanIp) {
    Write-Host "  export ZIPCODE_LLAMA_SERVER_URL=http://${lanIp}:$Port" -ForegroundColor Green
    Write-Host "  export ZIPCODE_LLAMA_SERVER_ALIAS=zipcode-remote" -ForegroundColor Green
} else {
    Write-Host "  (could not auto-detect LAN IP, run 'ipconfig' to find it)" -ForegroundColor DarkYellow
}
