# run.ps1 — launch BTC Consensus Signals locally (Windows / PowerShell)
#   .\run.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Load .env if present (KEY=VALUE lines, # comments ignored).
if (Test-Path .env) {
    Get-Content .env | Where-Object { $_ -match '^\s*[^#=]+=' } | ForEach-Object {
        $parts = $_ -split '=', 2
        $key = $parts[0].Trim()
        $val = $parts[1].Trim()
        if ($key -and $val) { [Environment]::SetEnvironmentVariable($key, $val) }
    }
}

# Local storage default (HF deployment uses /data via the Dockerfile).
if (-not $env:DATA_DIR) { $env:DATA_DIR = Join-Path $PSScriptRoot 'data' }

Write-Host "DATA_DIR = $($env:DATA_DIR)"
Write-Host "Starting on http://localhost:7860  (Ctrl+C to stop)"
python -m uvicorn app:app --host 0.0.0.0 --port 7860
