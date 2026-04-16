# Windows zipcode remote-server setup

Turn a Windows PC with an NVIDIA GPU into a LAN-accessible llama-server
that the main Linux dev box can point zipcode at via
`ZIPCODE_LLAMA_SERVER_URL`. Intended for running bigger Gemma 4 models
(26B A4B / 31B) that don't fit on the laptop's 8 GB VRAM.

## Prerequisites

- Windows 10/11
- NVIDIA GPU with recent driver (RTX 4080 Super tested; anything with
  CUDA 12 support works). Verify with `nvidia-smi` in PowerShell.
- ~20 GB free disk space (llama.cpp + 15 GB model)
- Same LAN as the Linux client
- PowerShell 7+ recommended (built-in 5.1 also works)

## One-shot setup

Open PowerShell **as Administrator** (needed for the firewall rule),
then:

```powershell
cd <path-where-you-cloned-the-scripts>\windows-server

# Allow unsigned local scripts for this session only
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Full install — downloads llama.cpp (~300 MB) + model (~15 GB) + firewall rule
.\setup.ps1
```

The defaults target:
- Work dir: `C:\zipcode-server\`
- Port: 8080
- Model: [TeichAI Gemma 4 26B A4B Claude-Opus-Distill, Q4_K_M](https://huggingface.co/TeichAI/gemma-4-26B-A4B-it-Claude-Opus-Distill-GGUF)

Common overrides:
```powershell
.\setup.ps1 -WorkDir "D:\llm" -Port 9090
.\setup.ps1 -ModelUrl "<different-gguf-url>" -ModelFileName "my-model.gguf"
.\setup.ps1 -LlamaCppVersion b7400                  # pin a specific release
.\setup.ps1 -SkipModelDownload                      # if you already have a .gguf
.\setup.ps1 -SkipFirewall                           # if running non-admin
```

At the end, `setup.ps1` prints the exact `export` lines you need on the
Linux client.

## Start the server

```powershell
.\start-server.ps1
```

This stays in the foreground — keep the PowerShell window open. Ctrl+C
to stop.

Common overrides:
```powershell
.\start-server.ps1 -Port 9000 -ContextSize 65536    # tune port / ctx
.\start-server.ps1 -GpuLayers 40                    # partial CPU offload
.\start-server.ps1 -NoFlashAttn                     # if flash-attn misbehaves
.\start-server.ps1 -ModelPath "C:\alt\model.gguf"   # pick a different GGUF
```

## Verify from the Linux client

```bash
# Replace 192.168.x.x with the Windows LAN IP that setup.ps1 printed
curl -s http://192.168.x.x:8080/health
# Expected: {"status":"ok"}
```

Then:
```bash
export ZIPCODE_LLAMA_SERVER_URL=http://192.168.x.x:8080
export ZIPCODE_LLAMA_SERVER_ALIAS=zipcode-remote
zipcode prompt "Hello" --model ~/.zipcode/models/any.gguf --ui plain
```

The `--model` arg is still required by the CLI parser but ignored in
remote mode.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `nvidia-smi` not found | Install NVIDIA driver first (https://www.nvidia.com/drivers) |
| `connection refused` from Linux | Firewall rule missing (re-run `setup.ps1` as admin) OR network profile is `Public` not `Private` |
| `{"status":"loading"}` persists > 2 min | Model still loading; 26B can take ~90 s |
| `CUDA error: out of memory` | Lower `-GpuLayers` (e.g. `40`) or `-ContextSize` (e.g. `32768`) |
| Multiple `.gguf` files found warning | Pass `-ModelPath` explicitly to `start-server.ps1` |
| PowerShell refuses to run the scripts | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once per session |

## For Windows Claude Code — handoff prompt

If you want a Windows-side Claude Code to pick this up and handle it,
hand it this prompt (adjust the repo path if needed):

> I want to set up this Windows PC as a remote llama-server for my Linux
> zipcode development box. The scripts at `.\setup.ps1` and
> `.\start-server.ps1` in the current folder handle it. Please:
>
> 1. Read `README.md` in this folder.
> 2. Run `.\setup.ps1` in an elevated PowerShell. Stop if
>    `nvidia-smi` is missing and tell me what driver version I have.
> 3. After setup finishes, print the `export ZIPCODE_LLAMA_SERVER_URL=…`
>    line it shows — I'll copy that to my Linux machine.
> 4. Then run `.\start-server.ps1` and leave it running. Before that,
>    confirm with me which default flags it will use.
>
> Do not modify the scripts themselves without asking me. If any step
> fails, stop and report the exact error.

## Files

- `setup.ps1` — one-shot install: llama.cpp CUDA Windows build, model
  download, Windows Firewall rule for the chosen port.
- `start-server.ps1` — launches llama-server with the right flags for
  zipcode's OpenAI-compatible `/v1/chat/completions` + SSE path.
- `README.md` — this file.

## Default flags reference

`start-server.ps1` runs llama-server with:

```
--host 0.0.0.0 --port 8080 --alias zipcode-remote
--jinja                     # Use the GGUF's embedded Gemma 4 chat template
-c 131072                   # 128K context (Gemma 4 native max)
-ngl 999                    # Offload all layers to GPU
-fa on                      # Flash attention
```

These match the defaults zipcode's Linux binary uses when it spawns its
own local llama-server, so behavior is consistent whether zipcode is
talking to a locally-spawned or a remote server.
