$ErrorActionPreference = "Stop"
$ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Logs = Join-Path $ProjectRoot "outputs\local-training"
New-Item -ItemType Directory -Path $Logs -Force | Out-Null
$Process = Start-Process -FilePath $Python -ArgumentList '-u scripts/run_local.py' -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $Logs "launcher.log") -RedirectStandardError (Join-Path $Logs "launcher-error.log") -PassThru
Write-Host "Local training queue PID: $($Process.Id)"
Write-Host "Status: $(Join-Path $Logs 'status.json')"
