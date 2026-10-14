# Atalhos de desenvolvimento no Windows.
# Uso:  .\tasks.ps1 <tarefa>
#   lint     ruff (lint e formatacao) e mypy
#   format   aplica a formatacao do ruff
#   test     testes com cobertura
#   check    lint + test (o mesmo que o CI)
#   lab      abre o LibrasVision Lab
#   app      abre o LibrasVision (usuario final)
param([Parameter(Position = 0)][string]$Tarefa = "check")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = ".venv\Scripts\python.exe"
$env:PYTHONPATH = "src"

function Run([string[]]$Cmd) {
    & $py @Cmd
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

switch ($Tarefa) {
    "lint" {
        Run @("-m", "ruff", "check", "src", "scripts", "tests", "empacotamento")
        Run @("-m", "ruff", "format", "--check", "src", "scripts", "tests", "empacotamento")
        Run @("-m", "mypy")
    }
    "format" { Run @("-m", "ruff", "format", "src", "scripts", "tests", "empacotamento") }
    "test" { Run @("-m", "pytest", "--cov", "--cov-report=term") }
    "check" { & $PSCommandPath lint; if ($LASTEXITCODE) { exit $LASTEXITCODE }; & $PSCommandPath test }
    "lab" { Run @("-m", "libras.app") }
    "app" { Run @("-m", "libras.app_real") }
    default { Write-Host "Tarefa desconhecida: $Tarefa. Opcoes: lint, format, test, check, lab, app"; exit 2 }
}
