# Arranque leve no Windows: login + interface, sem carregar o modelo (equivalente a serve_ui.sh).
$ErrorActionPreference = "Stop"
& "$PSScriptRoot\serve.ps1" -UiOnly @args
