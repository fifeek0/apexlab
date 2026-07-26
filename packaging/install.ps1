# One-command install for power users (Windows PowerShell):
#   irm https://raw.githubusercontent.com/<org>/<repo>/main/packaging/install.ps1 | iex
#
# Installs uv (standalone Python manager), clones the repo and registers the
# iracing-* commands on PATH via `uv tool`. No system Python required.

$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..."
    irm https://astral.sh/uv/install.ps1 | iex
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

$repo = "$env:USERPROFILE\iracing-suite"
if (-not (Test-Path $repo)) {
    git clone https://github.com/REPO_PLACEHOLDER/iracing-suite.git $repo
} else {
    git -C $repo pull --ff-only
}

Write-Host "Installing the suite (this downloads Python + Qt on first run)..."
uv tool install --force --from "$repo\apps\analysis" iracing-analysis --with "$repo\packages\iracing-core"
uv tool install --force --from "$repo\apps\overlay" iracing-overlay --with "$repo\packages\iracing-core"

Write-Host ""
Write-Host "Done. Commands available: iracing-analysis, iracing-engineer, garage61-harvest, iracing-overlay"
Write-Host "Try: iracing-analysis --demo"
