# ==============================================================================
# Whole Body Compute - 60-Second Sovereign Agent Runtime Installer (PowerShell)
# Hosted at:
#   irm https://wholebodycompute.com/begin.ps1 | iex
#   irm https://wholebodycompute.org/begin.ps1 | iex
# ==============================================================================
$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "🧬 Whole Body Compute - Sovereign Agent OS Awakening (Windows)" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════════════"
Write-Host "  • Platform: Windows ($([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture))"
Write-Host "  ! Note: Apple Silicon Metal organs degrade gracefully on Windows; Blood-Brain Barrier AST Slicer, Secret Guard, Memory & FinOps run natively." -ForegroundColor Yellow

function Add-UserBinToPath {
    foreach ($dir in @("$HOME\.local\bin", "$HOME\.cargo\bin", "$env:APPDATA\Python\Scripts")) {
        if ((Test-Path $dir) -and ($env:Path -notlike "*$dir*")) {
            $env:Path = "$dir;$env:Path"
        }
    }
}

Add-UserBinToPath

$bodyCmd = Get-Command body -ErrorAction SilentlyContinue
if ($bodyCmd) {
    Write-Host "  ✓ Found existing 'body' CLI: $($bodyCmd.Source)" -ForegroundColor Green
} else {
    Write-Host "  • Installing whole-body-compute CLI..."
    $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uvCmd) {
        Write-Host "  • Bootstrapping uv package manager (no admin required)..."
        try {
            $env:UV_PRINT_QUIET = "1"
            Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        } catch {
            Write-Host "  ! uv bootstrap notice: $_" -ForegroundColor Yellow
        }
        Add-UserBinToPath
        $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
    }

    if ($uvCmd) {
        try {
            & $uvCmd.Source tool install whole-body-compute 2>&1 | Out-Null
        } catch {}
        Add-UserBinToPath
        $bodyCmd = Get-Command body -ErrorAction SilentlyContinue
        if (-not $bodyCmd) {
            try {
                & $uvCmd.Source tool install "git+https://github.com/wholebodycompute/whole-body-compute.git" 2>&1 | Out-Null
            } catch {}
            Add-UserBinToPath
            $bodyCmd = Get-Command body -ErrorAction SilentlyContinue
        }
    }

    if (-not $bodyCmd) {
        $pyCmd = Get-Command python -ErrorAction SilentlyContinue
        if (-not $pyCmd) {
            $pyCmd = Get-Command python3 -ErrorAction SilentlyContinue
        }
        if ($pyCmd) {
            try {
                & $pyCmd.Source -m pip install --user whole-body-compute 2>&1 | Out-Null
            } catch {}
            Add-UserBinToPath
            $bodyCmd = Get-Command body -ErrorAction SilentlyContinue
        }
    }
}

Add-UserBinToPath
$bodyCmd = Get-Command body -ErrorAction SilentlyContinue
if ($bodyCmd) {
    & $bodyCmd.Source begin @args
} elseif (Test-Path "$HOME\.local\bin\body.exe") {
    & "$HOME\.local\bin\body.exe" begin @args
} else {
    Write-Host "  ! Automatic CLI install did not complete on this Windows setup." -ForegroundColor Yellow
    Write-Host "  Install explicitly in PowerShell with:"
    Write-Host "    powershell -ExecutionPolicy ByPass -c `"irm https://astral.sh/uv/install.ps1 | iex`""
    Write-Host "    uv tool install whole-body-compute; body begin"
    Write-Host "  Or inside Ubuntu (WSL2):"
    Write-Host "    curl -fsSL https://wholebodycompute.com/begin | sh"
}
