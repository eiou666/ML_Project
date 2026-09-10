param(
    [string]$Python = "",
    [switch]$SkipValidation
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
Set-Location -LiteralPath $ProjectRoot
$env:TEMP = Join-Path $ProjectRoot "tmp\install"
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
if (-not $Python) {
    $Candidate = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
    if (Test-Path -LiteralPath $Candidate) { $Python = $Candidate }
    else { $Python = (Get-Command python -ErrorAction Stop).Source }
}
& $Python -c "import sys; assert (3,11) <= sys.version_info[:2] <= (3,12), 'CUDA 12.1 setup requires Python 3.11 or 3.12'; print(sys.version)"
if ($LASTEXITCODE -ne 0) { throw "Unsupported or missing base Python: $Python" }
$VenvRoot = [IO.Path]::GetFullPath((Join-Path $ProjectRoot ".venv"))
$VenvPython = Join-Path $VenvRoot "Scripts\python.exe"
if (Test-Path -LiteralPath $VenvRoot) {
    $VenvWorks = $false
    if (Test-Path -LiteralPath $VenvPython) {
        & $VenvPython -c "import sys; assert (3,11) <= sys.version_info[:2] <= (3,12)"
        $VenvWorks = $LASTEXITCODE -eq 0
    }
    if (-not $VenvWorks) {
        $Backup = [IO.Path]::GetFullPath((Join-Path $ProjectRoot ("tmp\venv-legacy-" + (Get-Date -Format "yyyyMMdd-HHmmss"))))
        $Prefix = $ProjectRoot.TrimEnd('\') + '\'
        if (-not $VenvRoot.StartsWith($Prefix) -or -not $Backup.StartsWith($Prefix)) { throw "Environment paths must stay inside the project" }
        New-Item -ItemType Directory -Path (Split-Path $Backup) -Force | Out-Null
        Write-Host "Preserving old environment: $VenvRoot -> $Backup"
        Move-Item -LiteralPath $VenvRoot -Destination $Backup
    }
}
if (-not (Test-Path -LiteralPath $VenvPython)) {
    & $Python -m venv $VenvRoot
    if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed" }
}
& $VenvPython -m ensurepip --upgrade --default-pip
if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed" }
$env:PIP_CACHE_DIR = Join-Path $ProjectRoot "tmp\pip-cache"
$env:PYTHONUTF8 = "1"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$LocalWheel = Join-Path $ProjectRoot "tmp\wheels\torch-2.5.1+cu121-cp312-cp312-win_amd64.whl"
if (Test-Path -LiteralPath $LocalWheel) {
    & $VenvPython -m pip install --disable-pip-version-check $LocalWheel "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu121
} else {
    & $VenvPython -m pip install --disable-pip-version-check "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu121
}
if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed" }
& $VenvPython -m pip install --disable-pip-version-check -r requirements.txt -c requirements-cuda.txt
if ($LASTEXITCODE -ne 0) { throw "Project dependency installation failed" }
& $VenvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "Dependency check failed" }
& $VenvPython -c "from importlib.metadata import distributions; from pathlib import Path; rows=sorted({d.metadata['Name'].lower()+'=='+d.version for d in distributions()}); Path('requirements-local-lock.txt').write_text('\n'.join(rows)+'\n', encoding='utf-8')"
if ($LASTEXITCODE -ne 0) { throw "Local dependency lock export failed" }
if (-not $SkipValidation) {
    & $VenvPython scripts/verify_environment.py --require-cuda
    if ($LASTEXITCODE -ne 0) { throw "CUDA environment verification failed" }
    & $VenvPython -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw "Project tests failed" }
}
Write-Host "CUDA environment ready: $VenvPython"
