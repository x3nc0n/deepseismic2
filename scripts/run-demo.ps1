<#
.SYNOPSIS
    One-command demo launcher for DeepSeismic2 PoC.

.DESCRIPTION
    Starts Azurite (local blob storage), the FastAPI backend, and your choice
    of UI (Streamlit, Gradio, or terminal chat). Ctrl+C stops everything.

.EXAMPLE
    .\scripts\run-demo.ps1                  # Streamlit (default)
    .\scripts\run-demo.ps1 -UI gradio       # Gradio
    .\scripts\run-demo.ps1 -UI chat         # Terminal chat
    .\scripts\run-demo.ps1 -SkipAzurite     # Skip storage (mock mode only)
#>

param(
    [ValidateSet("streamlit", "gradio", "chat")]
    [string]$UI = "streamlit",
    [switch]$SkipAzurite
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

# Ensure we're in the project root
Push-Location $root

try {
    # Check prerequisites
    Write-Host "`n=== DeepSeismic2 Demo ===" -ForegroundColor Cyan
    Write-Host ""

    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        Write-Error "Python not found. Install Python 3.11+."
    }

    # Check if venv exists
    if (-not (Test-Path ".venv\Scripts\activate.ps1")) {
        Write-Host "Creating virtual environment..." -ForegroundColor Yellow
        python -m venv .venv
        & .venv\Scripts\activate.ps1
        pip install -e ".[dev,ui]" --quiet
    }

    # Set environment for mock mode
    $env:DEEPSEISMIC_MOCK_MODE = "true"
    $env:MOCK_LLM = "true"
    $env:PYTHONIOENCODING = "utf-8"

    # Generate sample data if not present
    $sampleSegy = Join-Path $root "data\volve\seismic\sample_volume.segy"
    if (-not (Test-Path $sampleSegy)) {
        Write-Host "Generating synthetic sample data..." -ForegroundColor Yellow
        python scripts/download_volve.py --sample
        Write-Host ""
    }

    # Start Azurite (unless skipped)
    $azuriteJob = $null
    if (-not $SkipAzurite) {
        if (Get-Command docker -ErrorAction SilentlyContinue) {
            Write-Host "Starting Azurite (local blob storage)..." -ForegroundColor Yellow
            docker compose -f docker/docker-compose.yml up -d azurite 2>$null
            Start-Sleep -Seconds 2

            # Check if running
            $azuriteRunning = docker compose -f docker/docker-compose.yml ps azurite --format json 2>$null | ConvertFrom-Json
            if ($azuriteRunning) {
                Write-Host "  Azurite running on port 10000" -ForegroundColor Green
            } else {
                Write-Host "  Azurite failed to start - continuing in pure mock mode" -ForegroundColor Yellow
            }
        } else {
            Write-Host "  Docker not found - running in pure mock mode" -ForegroundColor Yellow
        }
    }

    # Start FastAPI backend
    Write-Host "Starting FastAPI backend on http://localhost:8000 ..." -ForegroundColor Yellow
    $apiJob = Start-Job -ScriptBlock {
        Set-Location $using:root
        $env:DEEPSEISMIC_MOCK_MODE = "true"
        $env:MOCK_LLM = "true"
        & "$using:root\.venv\Scripts\python.exe" -m uvicorn deepseismic.api.main:app --host 0.0.0.0 --port 8000 2>&1
    }
    Start-Sleep -Seconds 3

    # Verify API is up
    try {
        $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 5
        Write-Host "  API running: $($health.status)" -ForegroundColor Green
    } catch {
        Write-Host "  API may still be starting..." -ForegroundColor Yellow
    }

    # Start UI
    Write-Host ""
    switch ($UI) {
        "streamlit" {
            Write-Host "Starting Streamlit UI..." -ForegroundColor Cyan
            Write-Host "  Open: http://localhost:8501" -ForegroundColor Green
            Write-Host ""
            Write-Host "Press Ctrl+C to stop all services." -ForegroundColor DarkGray
            & .venv\Scripts\python.exe -m streamlit run src/deepseismic/ui/streamlit_app.py --server.port 8501
        }
        "gradio" {
            Write-Host "Starting Gradio UI..." -ForegroundColor Cyan
            Write-Host "  Open: http://localhost:7860" -ForegroundColor Green
            Write-Host ""
            Write-Host "Press Ctrl+C to stop all services." -ForegroundColor DarkGray
            & .venv\Scripts\python.exe src/deepseismic/ui/gradio_app.py
        }
        "chat" {
            Write-Host "Starting terminal chat..." -ForegroundColor Cyan
            Write-Host ""
            & .venv\Scripts\python.exe -m deepseismic.ui.chat
        }
    }
} finally {
    # Cleanup
    Write-Host "`nShutting down..." -ForegroundColor Yellow

    if ($apiJob) {
        Stop-Job $apiJob -ErrorAction SilentlyContinue
        Remove-Job $apiJob -ErrorAction SilentlyContinue
    }

    if (-not $SkipAzurite) {
        if (Get-Command docker -ErrorAction SilentlyContinue) {
            docker compose -f docker/docker-compose.yml stop azurite 2>$null
        }
    }

    Pop-Location
    Write-Host "Done." -ForegroundColor Green
}
