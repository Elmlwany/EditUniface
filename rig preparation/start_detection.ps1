Write-Host "Starting Rig Preparation Detection System..." -ForegroundColor Green
Write-Host ""

# Get script directory and go to parent
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Split-Path -Parent $scriptDir)

Write-Host "Activating Python environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

Write-Host ""
Write-Host "Starting detection system with camera index 1..." -ForegroundColor Yellow
Write-Host ""

python "rig preparation\rig_prep_detection_advanced.py"

Write-Host ""
Write-Host "Detection system closed." -ForegroundColor Green
Read-Host "Press Enter to exit"
