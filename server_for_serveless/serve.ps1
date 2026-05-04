# Equivalente Windows a server_for_serveless/serve.sh
param(
    [switch] $UiOnly
)
$ErrorActionPreference = "Stop"
$ScriptDir = $PSScriptRoot
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Set-Location $RepoRoot

$venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Criando ambiente virtual..."
    python -m venv .venv
    $venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
}

& $venvPython -c "import fastapi, uvicorn, bcrypt, psycopg2, dotenv, httpx" 2>$null | Out-Null
if (-not $?) {
    Write-Host "Instalando dependências do servidor..."
    & (Join-Path $RepoRoot ".venv\Scripts\pip.exe") install -r "server_for_serveless\requirements.txt"
}

$pyArgs = @((Join-Path $ScriptDir "serve_lora.py"))
if ($UiOnly) {
    $pyArgs += "--ui-only"
}
if ($args.Count -gt 0) {
    $pyArgs += $args
}
& $venvPython @pyArgs
