param(
    [string]$Python = "D:\anaconda3\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$CacheDir = Join-Path $ProjectRoot ".uv-cache"
$Requirements = Join-Path $ProjectRoot "requirements.txt"
$VerifyScript = Join-Path $ProjectRoot "scripts\verify_environment.py"
$Uv = (Get-Command uv -ErrorAction Stop).Source

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python interpreter not found: $Python"
}

& $Uv --cache-dir $CacheDir venv --python $Python (Join-Path $ProjectRoot ".venv")
& $Uv --cache-dir $CacheDir pip install --python $VenvPython -r $Requirements
& $Uv --cache-dir $CacheDir pip check --python $VenvPython
& $VenvPython $VerifyScript
& $VenvPython -m pytest -q
& $VenvPython -m src.sweep --suite smoke --device cpu

Write-Host "Environment ready: $VenvPython"

