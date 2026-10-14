@echo off
REM Abre o LibrasVision Lab: a BANCADA da equipe (coleta, dataset, treino, avaliacao, diagnostico).
cd /d "%~dp0"
if not exist .venv\Scripts\pythonw.exe (
  echo Ambiente nao instalado. Rode instalar.bat primeiro.
  pause
  exit /b 1
)
set PYTHONPATH=%~dp0src
start "" .venv\Scripts\pythonw.exe -m libras.app %*
