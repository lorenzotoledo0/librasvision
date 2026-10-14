# Instala o LibrasVision no Windows: cria .venv com Python 3.11 e instala as dependencias.
# Uso (na raiz do projeto):  .\scripts\setup_windows.ps1 [-Dev] [-SemCamera]
#   -Dev        instala tambem as ferramentas de desenvolvimento (ruff, mypy, pytest-cov, pre-commit)
#   -SemCamera  pula o teste da webcam na verificacao final
param(
    [switch]$Dev,
    [switch]$SemCamera
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

& py -3.11 --version *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python 3.11 nao encontrado."
    Write-Host "Baixe o 'Windows installer (64-bit)' do Python 3.11 em https://www.python.org/downloads/windows/"
    Write-Host "Na instalacao, mantenha marcadas as opcoes 'py launcher' e 'tcl/tk and IDLE'."
    exit 1
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Criando ambiente .venv com Python 3.11..."
    & py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Falha ao criar o ambiente virtual." }
}

$py = ".venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Falha ao atualizar o pip." }
& $py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Falha na instalacao das dependencias." }
if ($Dev) {
    & $py -m pip install -e ".[dev]"
    if ($LASTEXITCODE -ne 0) { throw "Falha na instalacao das ferramentas de desenvolvimento." }
    & ".venv\Scripts\pre-commit.exe" install
}
& $py -m pip check
if ($LASTEXITCODE -ne 0) { throw "pip check encontrou conflitos de dependencias." }

Write-Host ""
Write-Host "Verificando o ambiente..."
if ($SemCamera) { & $py scripts\verificar_ambiente.py --sem-camera } else { & $py scripts\verificar_ambiente.py }
Write-Host ""
Write-Host "Bancada da equipe: .\LibrasVisionLab.bat   App do usuario final: .\LibrasVision.bat"
